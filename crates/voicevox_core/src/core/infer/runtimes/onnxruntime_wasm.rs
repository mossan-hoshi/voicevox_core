//! ブラウザ向けの[`InferenceRuntime`]実装。
//!
//! ONNX Runtimeをネイティブライブラリとしてリンクする代わりに、Emscriptenの
//! JSライブラリ(`wasm_library.js`)越しに[onnxruntime-web]へ推論を委譲する。
//! `ort`クレートには一切依存しない。
//!
//! JS側の推論は非同期だが、`-sASYNCIFY=1`によりRustからは同期関数として見える。
//! そのため[`InferenceRuntime::run_blocking`]をそのまま実装でき、
//! ブロッキング版APIおよびC APIは無改造で通る。
//!
//! モジュールの構造とアイテム名は`onnxruntime`モジュールと同一に保つこと。
//! `runtimes.rs`が`#[path]`でこのファイルを`onnxruntime`として読み込むため、
//! これより上の層はどちらの実装かを意識しない。
//!
//! [onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/

use std::{
    ffi::{CStr, CString, c_char, c_int},
    sync::Arc,
};

use anyhow::{anyhow, bail, ensure};
use duplicate::duplicate_item;
use ndarray::{Array, ArrayD, Dimension, IxDyn};
use serde::Deserialize;

use super::super::{
    super::{
        devices::{DeviceSpec, GpuSpec, SupportedDevices},
        voice_model::ModelBytes,
    },
    InferenceRuntime, InferenceSessionOptions, InputScalarKind, OutputScalarKind, OutputTensor,
    ParamInfo, PushInputTensor,
};

/// 必要なONNX Runtime 1.xの最小マイナーバージョン。
///
/// `ort`を用いないため`ort::sys::ORT_API_VERSION`は参照できず、直接書く。
const LIB_MIN_REQUIRED_MINOR_VERSION: u32 = 17;
const LIB_MAX_SUPPORTED_MINOR_VERSION: u32 = 29;

static SINGLETON: once_cell::sync::OnceCell<Inner> = once_cell::sync::OnceCell::new();

#[derive(Debug)]
struct Inner {
    _private: (),
}

impl Inner {
    fn get() -> Option<&'static Self> {
        SINGLETON.get()
    }

    fn get_or_try_init() -> crate::Result<&'static Self> {
        SINGLETON.get_or_try_init(|| {
            // JS側のonnxruntime-webの読み込みはここで完了させる。ASYNCIFYにより
            // 非同期処理の完了を待ってから戻ってくる。
            let status = unsafe { ort_web_init() };
            ensure_ok(status, "failed to initialize onnxruntime-web").map_err(|source| {
                crate::error::ErrorRepr::InitInferenceRuntime {
                    runtime_display_name:
                        <self::blocking::Onnxruntime as InferenceRuntime>::DISPLAY_NAME,
                    source,
                }
            })?;
            Ok(Inner { _private: () })
        })
    }
}

// `wasm_library.js`が実装する。
//
// いずれもASYNCIFY下で非同期処理の完了を待ってから戻る。エラーは負の値で表され、
// 詳細は`ort_web_take_last_error`で取り出す。
unsafe extern "C" {
    fn ort_web_init() -> c_int;

    /// ONNXのバイト列からセッションを作る。
    ///
    /// 成功すると非負のセッションIDを返し、`out_meta_json`に入出力のメタ情報を表す
    /// JSON文字列(malloc済み)を書き込む。
    fn ort_web_session_new(
        model_ptr: *const u8,
        model_len: usize,
        out_meta_json: *mut *mut c_char,
    ) -> c_int;

    fn ort_web_session_release(session_id: c_int);

    /// 推論を実行する。
    ///
    /// `inputs_json`は入力テンソルの名前・データ型・形状・データのポインタを表すJSON。
    /// 成功すると`out_outputs_json`に出力を表すJSON文字列(malloc済み)を書き込む。
    /// JSON中のデータのポインタもmalloc済みで、呼び出し側が解放する。
    fn ort_web_session_run(
        session_id: c_int,
        inputs_json: *const c_char,
        out_outputs_json: *mut *mut c_char,
    ) -> c_int;

    /// 直近のエラーメッセージ(malloc済み)を取り出す。無ければヌル。
    fn ort_web_take_last_error() -> *mut c_char;
}

/// JS側がmallocした文字列を受け取り、解放まで面倒を見る。
fn take_js_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: JS側は`stringToNewUTF8`でNUL終端の文字列をmallocしている。
    let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    // SAFETY: JS側のmallocとこちらのfreeは同一のアロケータ。
    unsafe { libc_free(ptr.cast()) };
    Some(s)
}

unsafe extern "C" {
    #[link_name = "free"]
    fn libc_free(ptr: *mut std::ffi::c_void);
}

fn ensure_ok(status: c_int, context: &str) -> anyhow::Result<()> {
    if status >= 0 {
        return Ok(());
    }
    let detail = take_js_string(unsafe { ort_web_take_last_error() })
        .unwrap_or_else(|| "unknown error".to_owned());
    bail!("{context}: {detail}");
}

#[derive(Deserialize)]
struct ParamMeta {
    name: String,
    /// onnxruntime-webの`Tensor.type`の文字列。`"float32"`や`"int64"`など。
    #[serde(rename = "type")]
    ty: String,
    /// 形状が判らない場合はヌル。
    ndim: Option<usize>,
}

#[derive(Deserialize)]
struct SessionMeta {
    inputs: Vec<ParamMeta>,
    outputs: Vec<ParamMeta>,
}

#[derive(Deserialize)]
struct OutputMeta {
    #[serde(rename = "type")]
    ty: String,
    dims: Vec<usize>,
    /// wasmメモリ上のデータのアドレス。malloc済みなのでこちらで解放する。
    ptr: u32,
}

/// セッションのハンドル。落とすとJS側も解放する。
pub(crate) struct WasmSession {
    id: c_int,
}

impl Drop for WasmSession {
    fn drop(&mut self) {
        unsafe { ort_web_session_release(self.id) };
    }
}

pub(crate) struct OnnxruntimeRunContext {
    sess: Arc<WasmSession>,
    inputs: Vec<InputTensor>,
}

struct InputTensor {
    name: &'static str,
    ty: &'static str,
    dims: Vec<usize>,
    /// 連続領域に直したデータ。JSON中ではこのバッファのアドレスを渡す。
    data: Vec<u8>,
}

impl From<Arc<WasmSession>> for OnnxruntimeRunContext {
    fn from(sess: Arc<WasmSession>) -> Self {
        Self {
            sess,
            inputs: vec![],
        }
    }
}

impl PushInputTensor for OnnxruntimeRunContext {
    #[duplicate_item(
        method           T       TY_STR;
        [ push_int64 ]   [ i64 ] [ "int64" ];
        [ push_float32 ] [ f32 ] [ "float32" ];
    )]
    fn method(
        &mut self,
        name: &'static str,
        tensor: Array<T, impl Dimension + 'static>,
    ) -> anyhow::Result<()> {
        let dims = tensor.shape().to_owned();
        let standard = tensor.as_standard_layout();
        let slice = standard
            .as_slice()
            .expect("`as_standard_layout`した後なので連続しているはず");
        self.inputs.push(InputTensor {
            name,
            ty: TY_STR,
            dims,
            data: bytemuck::cast_slice::<T, u8>(slice).to_owned(),
        });
        Ok(())
    }
}

fn dtype_to_input_kind(ty: &str, name: &str) -> anyhow::Result<InputScalarKind> {
    match ty {
        "float32" => Ok(InputScalarKind::Float32),
        "int64" => Ok(InputScalarKind::Int64),
        _ => Err(anyhow!("unsupported input datatype `{ty}` for `{name}`")),
    }
}

fn dtype_to_output_kind(ty: &str, name: &str) -> anyhow::Result<OutputScalarKind> {
    match ty {
        "float32" => Ok(OutputScalarKind::Float32),
        "int64" => Ok(OutputScalarKind::Int64),
        _ => Err(anyhow!("unsupported output datatype `{ty}` for `{name}`")),
    }
}

impl InferenceRuntime for self::blocking::Onnxruntime {
    type Session = WasmSession;
    type RunContext = OnnxruntimeRunContext;

    const DISPLAY_NAME: &'static str = "onnxruntime-web";

    fn supported_devices(&self) -> crate::Result<SupportedDevices> {
        // ブラウザではWASM実行プロバイダ(CPU)のみを使う。
        Ok(SupportedDevices {
            cpu: true,
            cuda: false,
            dml: false,
        })
    }

    fn test_gpu(&self, gpu: GpuSpec) -> anyhow::Result<()> {
        bail!("{gpu} is not supported on `{}`", Self::DISPLAY_NAME);
    }

    fn new_session(
        &self,
        model: &ModelBytes,
        options: InferenceSessionOptions,
    ) -> anyhow::Result<(
        Self::Session,
        Vec<ParamInfo<InputScalarKind>>,
        Vec<ParamInfo<OutputScalarKind>>,
    )> {
        match options.device {
            DeviceSpec::Cpu => {}
            DeviceSpec::Gpu(gpu) => {
                bail!("{gpu} is not supported on `{}`", Self::DISPLAY_NAME);
            }
        }
        // `cpu_num_threads`はcross-origin isolationが無い前提のため無視する
        // (onnxruntime-webが自動的に1スレッドになる)。

        let onnx = match model {
            ModelBytes::Onnx(onnx) => onnx,
            ModelBytes::VvBin(_) => bail!(
                "`{}` does not support the \"vv-bin\" format; \
                 the voice model must contain ONNX models",
                Self::DISPLAY_NAME,
            ),
        };

        let mut meta_json: *mut c_char = std::ptr::null_mut();
        let id = unsafe { ort_web_session_new(onnx.as_ptr(), onnx.len(), &mut meta_json) };
        ensure_ok(id, "failed to create an inference session")?;
        let sess = WasmSession { id };

        let meta_json =
            take_js_string(meta_json).ok_or_else(|| anyhow!("missing session metadata"))?;
        let meta = serde_json::from_str::<SessionMeta>(&meta_json)?;

        let input_param_infos = meta
            .inputs
            .iter()
            .map(|info| {
                Ok(ParamInfo {
                    dt: dtype_to_input_kind(&info.ty, &info.name)?,
                    name: info.name.clone().into(),
                    ndim: info.ndim,
                })
            })
            .collect::<anyhow::Result<_>>()?;

        let output_param_infos = meta
            .outputs
            .iter()
            .map(|info| {
                Ok(ParamInfo {
                    dt: dtype_to_output_kind(&info.ty, &info.name)?,
                    name: info.name.clone().into(),
                    ndim: info.ndim,
                })
            })
            .collect::<anyhow::Result<_>>()?;

        Ok((sess, input_param_infos, output_param_infos))
    }

    fn run_blocking(
        OnnxruntimeRunContext { sess, inputs }: Self::RunContext,
    ) -> anyhow::Result<Vec<OutputTensor>> {
        let inputs_json = serde_json::to_string(
            &inputs
                .iter()
                .map(|i| {
                    serde_json::json!({
                        "name": i.name,
                        "type": i.ty,
                        "dims": i.dims,
                        "ptr": i.data.as_ptr() as u32,
                        "byteLength": i.data.len(),
                    })
                })
                .collect::<Vec<_>>(),
        )?;
        let inputs_json = CString::new(inputs_json)?;

        let mut outputs_json: *mut c_char = std::ptr::null_mut();
        let status =
            unsafe { ort_web_session_run(sess.id, inputs_json.as_ptr(), &mut outputs_json) };
        // `inputs`はここまで生かしておく必要がある (JS側がポインタ越しに読むため)。
        drop(inputs);
        ensure_ok(status, "inference failed")?;

        let outputs_json =
            take_js_string(outputs_json).ok_or_else(|| anyhow!("missing inference outputs"))?;
        serde_json::from_str::<Vec<OutputMeta>>(&outputs_json)?
            .into_iter()
            .map(extract_output)
            .collect()
    }

    async fn run_async(
        ctx: Self::RunContext,
        _cancellable: bool,
    ) -> anyhow::Result<Vec<OutputTensor>> {
        // シングルスレッドのwasmでは`run_blocking`と等価。
        Self::run_blocking(ctx)
    }
}

/// JS側がmallocしたバッファから`OutputTensor`を作り、バッファを解放する。
fn extract_output(meta: OutputMeta) -> anyhow::Result<OutputTensor> {
    let len = meta.dims.iter().product::<usize>();
    let ptr = meta.ptr as *mut u8;
    ensure!(!ptr.is_null(), "null output tensor");

    let shape = IxDyn(&meta.dims);
    let tensor = match &*meta.ty {
        "float32" => {
            // SAFETY: JS側は`dims`の積×4バイトをmallocして書き込んでいる。
            let slice = unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), len) };
            OutputTensor::Float32(ArrayD::from_shape_vec(shape, slice.to_owned())?)
        }
        "int64" => {
            // SAFETY: 同上 (×8バイト)。
            let slice = unsafe { std::slice::from_raw_parts(ptr.cast::<i64>(), len) };
            OutputTensor::Int64(ArrayD::from_shape_vec(shape, slice.to_owned())?)
        }
        ty => {
            unsafe { libc_free(ptr.cast()) };
            bail!("unexpected output tensor element data type `{ty}`");
        }
    };
    unsafe { libc_free(ptr.cast()) };
    Ok(tensor)
}

pub(crate) mod blocking {
    use ref_cast::{RefCastCustom, ref_cast_custom};

    use crate::SupportedDevices;

    use super::{super::super::InferenceRuntime, Inner};

    /// ONNX Runtime。
    ///
    /// シングルトンであり、インスタンスは高々一つ。インスタンスは[非同期版の`Onnxruntime`]と共有される。
    ///
    /// このビルドでは[onnxruntime-web]に推論を委譲する。
    ///
    /// [非同期版の`Onnxruntime`]: crate::nonblocking::Onnxruntime
    /// [onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/
    #[cfg_attr(doc, doc(alias = "VoicevoxOnnxruntime"))]
    #[derive(Debug, RefCastCustom)]
    #[repr(transparent)]
    pub struct Onnxruntime(Inner);

    impl Onnxruntime {
        /// 必要なONNX Runtime 1.xの最小マイナーバージョン。
        #[cfg_attr(
            doc,
            doc(alias = "voicevox_get_onnxruntime_lib_min_required_minor_version")
        )]
        pub const LIB_MIN_REQUIRED_MINOR_VERSION: u32 = 17;

        /// サポートされるONNX Runtime 1.xの最大マイナーバージョン。
        #[cfg_attr(
            doc,
            doc(alias = "voicevox_get_onnxruntime_lib_max_supported_minor_version")
        )]
        pub const LIB_MAX_SUPPORTED_MINOR_VERSION: u32 = 29;

        #[ref_cast_custom]
        const fn new(inner: &Inner) -> &Self;

        /// インスタンスが既に作られているならそれを得る。
        ///
        /// 作られていなければ`None`を返す。
        #[cfg_attr(doc, doc(alias = "voicevox_onnxruntime_get"))]
        pub fn get() -> Option<&'static Self> {
            Inner::get().map(Self::new)
        }

        /// onnxruntime-webを読み込んで初期化する。
        ///
        /// 一度成功したら以後は同じ参照を返す。
        #[cfg_attr(doc, doc(alias = "voicevox_onnxruntime_init_once"))]
        pub fn init_once() -> crate::Result<&'static Self> {
            Inner::get_or_try_init().map(Onnxruntime::new)
        }

        /// ONNX Runtimeとして利用可能なデバイスの情報を取得する。
        #[cfg_attr(doc, doc(alias = "voicevox_onnxruntime_create_supported_devices_json"))]
        pub fn supported_devices(&self) -> crate::Result<SupportedDevices> {
            <Self as InferenceRuntime>::supported_devices(self)
        }
    }

    const _: () = assert!(
        Onnxruntime::LIB_MIN_REQUIRED_MINOR_VERSION == super::LIB_MIN_REQUIRED_MINOR_VERSION,
    );
    const _: () = assert!(
        Onnxruntime::LIB_MAX_SUPPORTED_MINOR_VERSION == super::LIB_MAX_SUPPORTED_MINOR_VERSION,
    );
}

pub(crate) mod nonblocking {
    use ref_cast::{RefCastCustom, ref_cast_custom};

    use crate::SupportedDevices;

    /// ONNX Runtime。
    ///
    /// シングルトンであり、インスタンスは高々一つ。インスタンスは[ブロッキング版の`Onnxruntime`]と共有される。
    ///
    /// [ブロッキング版の`Onnxruntime`]: crate::blocking::Onnxruntime
    #[derive(Debug, RefCastCustom)]
    #[repr(transparent)]
    pub struct Onnxruntime(pub(crate) super::blocking::Onnxruntime);

    impl Onnxruntime {
        /// 必要なONNX Runtime 1.xの最小マイナーバージョン。
        pub const LIB_MIN_REQUIRED_MINOR_VERSION: u32 =
            super::blocking::Onnxruntime::LIB_MIN_REQUIRED_MINOR_VERSION;

        /// サポートされるONNX Runtime 1.xの最大マイナーバージョン。
        pub const LIB_MAX_SUPPORTED_MINOR_VERSION: u32 =
            super::blocking::Onnxruntime::LIB_MAX_SUPPORTED_MINOR_VERSION;

        #[ref_cast_custom]
        pub(crate) const fn from_blocking(blocking: &super::blocking::Onnxruntime) -> &Self;

        /// インスタンスが既に作られているならそれを得る。
        ///
        /// 作られていなければ`None`を返す。
        pub fn get() -> Option<&'static Self> {
            super::blocking::Onnxruntime::get().map(Self::from_blocking)
        }

        /// ONNX Runtimeとして利用可能なデバイスの情報を取得する。
        pub fn supported_devices(&self) -> crate::Result<SupportedDevices> {
            self.0.supported_devices()
        }
    }
}

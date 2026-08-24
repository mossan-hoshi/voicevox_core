//! wasm(emscripten)向けの薄いシムAPI。
//!
//! 本来のC APIはオプションを構造体の値渡しで受け取るが、wasm32のC ABIでは
//! 構造体の値渡しは間接渡し(呼び出し側がスタックに置いてポインタを渡す)になるため、
//! Emscriptenの`ccall`からは正しく呼べない。ここではオプションをスカラー引数に
//! ばらした版を用意して、JS側がABIを推測しなくて済むようにする。
//!
//! 引数と戻り値の意味は元の関数と同じ。

use std::{ffi::c_char, ptr::NonNull};

use crate::{
    OpenJtalkRc, VoicevoxAccelerationMode, VoicevoxInitializeOptions,
    VoicevoxLoadVoiceModelOptions, VoicevoxOnExistingVoiceModelId, VoicevoxOnnxruntime,
    VoicevoxStyleId, VoicevoxSynthesisOptions, VoicevoxSynthesizer, VoicevoxVoiceModelFile,
    result_code::VoicevoxResultCode,
};

/// ::voicevox_synthesizer_new の、オプションをスカラーで受け取る版。
///
/// `acceleration_mode`は ::VoicevoxAccelerationMode の値。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn voicevox_wasm_synthesizer_new(
    onnxruntime: &'static VoicevoxOnnxruntime,
    open_jtalk: *const OpenJtalkRc,
    acceleration_mode: i32,
    cpu_num_threads: u16,
    out_synthesizer: NonNull<NonNull<VoicevoxSynthesizer>>,
) -> VoicevoxResultCode {
    let acceleration_mode = match acceleration_mode {
        0 => VoicevoxAccelerationMode::VOICEVOX_ACCELERATION_MODE_AUTO,
        1 => VoicevoxAccelerationMode::VOICEVOX_ACCELERATION_MODE_CPU,
        2 => VoicevoxAccelerationMode::VOICEVOX_ACCELERATION_MODE_GPU,
        unknown => {
            // ブラウザではCPU以外を選べないため、既定値に倒しても実害は無い。
            tracing::warn!("unknown acceleration mode: {unknown}; falling back to `AUTO`");
            VoicevoxAccelerationMode::VOICEVOX_ACCELERATION_MODE_AUTO
        }
    };
    let options = VoicevoxInitializeOptions {
        acceleration_mode,
        cpu_num_threads,
    };
    // SAFETY: 呼び出し側の責任は元の関数と同じ。
    unsafe {
        crate::voicevox_synthesizer_new(onnxruntime, open_jtalk, options, out_synthesizer)
    }
}

/// ::voicevox_synthesizer_load_voice_model の、オプションをスカラーで受け取る版。
///
/// `on_existing`は ::VoicevoxOnExistingVoiceModelId の値。
#[unsafe(no_mangle)]
pub extern "C" fn voicevox_wasm_synthesizer_load_voice_model(
    synthesizer: *const VoicevoxSynthesizer,
    model: *const VoicevoxVoiceModelFile,
    on_existing: i32,
) -> VoicevoxResultCode {
    let on_existing = match on_existing {
        0 => VoicevoxOnExistingVoiceModelId::VOICEVOX_ON_EXISTING_VOICE_MODEL_ID_ERROR,
        1 => VoicevoxOnExistingVoiceModelId::VOICEVOX_ON_EXISTING_VOICE_MODEL_ID_RELOAD,
        2 => VoicevoxOnExistingVoiceModelId::VOICEVOX_ON_EXISTING_VOICE_MODEL_ID_SKIP,
        unknown => {
            tracing::warn!("unknown `on_existing`: {unknown}; falling back to `ERROR`");
            VoicevoxOnExistingVoiceModelId::VOICEVOX_ON_EXISTING_VOICE_MODEL_ID_ERROR
        }
    };
    crate::voicevox_synthesizer_load_voice_model(
        synthesizer,
        model,
        VoicevoxLoadVoiceModelOptions { on_existing },
    )
}

/// ::voicevox_synthesizer_synthesis の、オプションをスカラーで受け取る版。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn voicevox_wasm_synthesizer_synthesis(
    synthesizer: *const VoicevoxSynthesizer,
    audio_query_json: *const c_char,
    style_id: VoicevoxStyleId,
    enable_interrogative_upspeak: i32,
    output_wav_length: NonNull<usize>,
    output_wav: NonNull<NonNull<u8>>,
) -> VoicevoxResultCode {
    let options = VoicevoxSynthesisOptions {
        enable_interrogative_upspeak: enable_interrogative_upspeak != 0,
    };
    // SAFETY: 呼び出し側の責任は元の関数と同じ。
    unsafe {
        crate::voicevox_synthesizer_synthesis(
            synthesizer,
            audio_query_json,
            style_id,
            options,
            output_wav_length,
            output_wav,
        )
    }
}

/// ::voicevox_synthesizer_tts の、オプションをスカラーで受け取る版。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn voicevox_wasm_synthesizer_tts(
    synthesizer: *const VoicevoxSynthesizer,
    text: *const c_char,
    style_id: VoicevoxStyleId,
    enable_interrogative_upspeak: i32,
    output_wav_length: NonNull<usize>,
    output_wav: NonNull<NonNull<u8>>,
) -> VoicevoxResultCode {
    let options = crate::VoicevoxTtsOptions {
        enable_interrogative_upspeak: enable_interrogative_upspeak != 0,
    };
    // SAFETY: 呼び出し側の責任は元の関数と同じ。
    unsafe {
        crate::voicevox_synthesizer_tts(
            synthesizer,
            text,
            style_id,
            options,
            output_wav_length,
            output_wav,
        )
    }
}

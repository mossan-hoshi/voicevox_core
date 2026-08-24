//! emscripten向けにビルドするときのリンカ引数を組み立てる。
//!
//! それ以外のターゲットでは何もしない。

use std::{env, fs, path::Path};

/// wasmモジュールの初期メモリ。
///
/// iOS Safariでの失敗を避けるため小さく確保して、`ALLOW_MEMORY_GROWTH`で伸ばす。
const INITIAL_MEMORY: &str = "64MB";
const MAXIMUM_MEMORY: &str = "2GB";

/// このビルドで有効になっていないfeature。
///
/// これらにgateされた関数は存在しないため、エクスポート対象から外す。
const DISABLED_FEATURES: &[&str] = &["load-onnxruntime", "link-onnxruntime"];

/// スキャンしないファイル。
///
/// `compatible_engine`は`load-onnxruntime`かつ非emscripten専用。
const SKIPPED_FILES: &[&str] = &["compatible_engine.rs"];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("emscripten") {
        return;
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("`CARGO_MANIFEST_DIR` should be set");
    let crates_dir = Path::new(&manifest_dir)
        .parent()
        .expect("the crate should live under `crates/`");

    let c_api_src = crates_dir.join("voicevox_core_c_api").join("src");
    let mut exported = collect_exported_functions(&c_api_src);
    // `ccall`で確保・解放するために要る。
    exported.extend(["_malloc".to_owned(), "_free".to_owned()]);
    exported.sort();
    exported.dedup();

    let js_library = crates_dir.join("voicevox_core").join("wasm_library.js");
    let js_library = js_library
        .canonicalize()
        .unwrap_or_else(|e| panic!("{}: {e}", js_library.display()));
    println!("cargo::rerun-if-changed={}", js_library.display());

    let runtime_methods = [
        "ccall",
        "cwrap",
        "FS",
        "HEAPU8",
        "HEAPU32",
        "HEAPF32",
        "UTF8ToString",
        "stringToUTF8",
        "lengthBytesUTF8",
        "stringToNewUTF8",
        "getValue",
        "setValue",
    ];

    let link_args = [
        "--no-entry".to_owned(),
        "-sMODULARIZE=1".to_owned(),
        "-sEXPORT_ES6=1".to_owned(),
        "-sEXPORT_NAME=VoicevoxCore".to_owned(),
        // Web WorkerからもWebページからも読めるようにする。
        "-sENVIRONMENT=web,worker".to_owned(),
        // JS側の非同期処理(onnxruntime-webの推論)をRustからは同期に見せる。
        "-sASYNCIFY=1".to_owned(),
        "-sASYNCIFY_STACK_SIZE=65536".to_owned(),
        format!("-sINITIAL_MEMORY={INITIAL_MEMORY}"),
        format!("-sMAXIMUM_MEMORY={MAXIMUM_MEMORY}"),
        "-sALLOW_MEMORY_GROWTH=1".to_owned(),
        // 辞書とVVMをMEMFSに置くため。
        "-sFORCE_FILESYSTEM=1".to_owned(),
        format!("-sEXPORTED_FUNCTIONS={}", exported.join(",")),
        format!("-sEXPORTED_RUNTIME_METHODS={}", runtime_methods.join(",")),
        format!("--js-library={}", js_library.display()),
    ];

    for arg in link_args {
        println!("cargo::rustc-link-arg-bins={arg}");
    }
}

/// `#[unsafe(no_mangle)] pub … extern "C" fn …`をソースから拾い、
/// Emscriptenのシンボル名(先頭に`_`が付く)にする。
///
/// これらはライブラリ側(rlib)にあり、バイナリからは参照されないため、
/// `-sEXPORTED_FUNCTIONS`で名指ししないとリンカに落とされる。
///
/// 無効なfeatureにgateされた関数は実在しないので飛ばす。名指しすると
/// wasm-ldが"symbol exported via --export not found"で失敗する。
fn collect_exported_functions(src_dir: &Path) -> Vec<String> {
    let mut names = vec![];

    for entry in fs::read_dir(src_dir).unwrap_or_else(|e| panic!("{}: {e}", src_dir.display())) {
        let path = entry.expect("failed to read a directory entry").path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if SKIPPED_FILES.contains(&file_name) {
            continue;
        }
        println!("cargo::rerun-if-changed={}", path.display());

        let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut no_mangle = false;
        let mut disabled = false;
        for line in src.lines() {
            let line = line.trim();
            if line.starts_with("#[unsafe(no_mangle)]") {
                no_mangle = true;
                continue;
            }
            if line.starts_with("#[cfg(") {
                disabled |= DISABLED_FEATURES.iter().any(|f| {
                    line.contains(&format!("feature = \"{f}\"")) && !line.contains("any(")
                });
                continue;
            }
            if !no_mangle {
                continue;
            }
            if let Some(name) = parse_extern_c_fn_name(line) {
                if !disabled {
                    names.push(format!("_{name}"));
                }
                no_mangle = false;
                disabled = false;
            } else if !line.is_empty() && !line.starts_with("#[") && !line.starts_with("//") {
                // 属性でも空行でもコメントでもない行が来たら、対応する関数を見失っている。
                no_mangle = false;
                disabled = false;
            }
        }
    }

    assert!(
        !names.is_empty(),
        "found no `#[unsafe(no_mangle)]` functions in {}",
        src_dir.display(),
    );
    names
}

/// `pub … extern "C" fn <name>(`から`<name>`を取り出す。
fn parse_extern_c_fn_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("pub ")?;
    let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);
    let rest = rest.strip_prefix("extern \"C\" fn ")?;
    let name = rest.split(['(', '<']).next()?.trim();
    (!name.is_empty()).then_some(name)
}

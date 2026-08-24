// `web-onnxruntime`では`ort`ごとONNX Runtimeをリンクせず、JSブリッジ越しに
// onnxruntime-webへ推論を委譲する実装に差し替える。モジュールの構造とアイテム名は
// `onnxruntime`と同一に保たれているため、これより上の層は無改造で通る。
#[cfg(not(feature = "web-onnxruntime"))]
pub(crate) mod onnxruntime;

#[cfg(feature = "web-onnxruntime")]
#[path = "runtimes/onnxruntime_wasm.rs"]
pub(crate) mod onnxruntime;

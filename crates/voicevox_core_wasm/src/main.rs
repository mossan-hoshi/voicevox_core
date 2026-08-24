//! emscripten向けのモジュールを生成するためだけのバイナリターゲット。
//!
//! C APIの各関数は`voicevox_core_c_api`のライブラリ側にあり、ここからは参照しない。
//! `build.rs`が`-sEXPORTED_FUNCTIONS`で名指しすることでリンカに落とされずに残る。
//! `extern crate`はrlibをリンク対象に含めるために要る。
//!
//! `--no-entry`を付けているため`main`は呼ばれない。

extern crate c_api;

fn main() {}

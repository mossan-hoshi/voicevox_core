# wasm(ブラウザ)向けビルド

**このフォーク独自のドキュメント。upstream には存在しないし、PRも出さない。**
このブランチ `web/wasm-0.17.0`（tag `0.17.0` 起点）が成果物の唯一の置き場所。

`wasm32-unknown-emscripten` 向けに voicevox_core をビルドし、ブラウザで
完全クライアントサイドの音声合成を動かすための手順と、実装上の判断の記録。

関連ドキュメント:
- `docs/custom-model-training.md` — 自前で音声モデルを学習してVVMを作る方法

## 方式

**ONNX Runtime をリンクせず、Emscripten の js-library 越しに
[onnxruntime-web] へ推論を委譲する。** `ort` クレートは wasm ビルドから外れる。

```
voicevox_core (Rust)
  ├─ Open JTalk (C/C++)      → 同じ wasm に同居。辞書は Emscripten の仮想FS
  └─ InferenceRuntime trait
       └─ onnxruntime_wasm.rs → wasm_library.js → onnxruntime-web (JS)
                                 ↑ ASYNCIFY で同期に見せる
```

`-sASYNCIFY=1` により JS 側の非同期推論を Rust からは同期関数として扱えるので、
`run_blocking` をそのまま実装でき、**ブロッキングAPIとC APIは無改造で通る**。

この構成のおかげで、VOICEVOX製 ONNX Runtime を wasm 向けに静的ビルドする
(前例が無く、そもそも復号処理が非公開なので不可能) 必要が無くなっている。

[onnxruntime-web]: https://onnxruntime.ai/docs/tutorials/web/

## ビルド

```bash
source ~/emsdk/emsdk_env.sh

# cdylib(emscriptenではSIDE_MODULE)のリンクに -fPIC が要る
export CFLAGS="-ffunction-sections -fdata-sections -fno-exceptions -fPIC"
export CXXFLAGS="$CFLAGS"

cargo +nightly build -p voicevox_core_wasm \
  --target wasm32-unknown-emscripten \
  --profile web-release \
  -Z build-std=std,panic_abort
```

成果物: `target/wasm32-unknown-emscripten/web-release/voicevox_core_web.{js,wasm}`

前提環境（実測で確認した組み合わせ）:

| ツール | バージョン |
|---|---|
| emsdk | 6.0.0 |
| Rust | nightly（`build-std` に必須） |
| cmake | 3.28 |

WSL2 (Ubuntu 24.04) で確認。Windows ネイティブは非推奨。

## 実装上の判断と、ハマりどころ

全部実測で確かめた。同じ罠を踏み直さないための記録。

### 1. `cdylib` は使えない。bin ターゲットが要る

現在の Rust は emscripten の `cdylib` に `-sSIDE_MODULE=2` を渡すため、
JSグルー(`.js`)が生成されず `MODULARIZE`/`EXPORT_ES6` も無視される。

→ `crates/voicevox_core_wasm` パッケージを新設し、`voicevox_core_c_api` の
rlib をリンクする bin ターゲットとしてビルドする。
`voicevox_core_c_api` の `[lib] crate-type` に `rlib` を追加した。

### 2. `-Z build-std=std,panic_abort` + `panic = "abort"` が必須

Rust が emscripten に渡す `-fwasm-exceptions` は `-sASYNCIFY=1` と非互換
（emcc が warning を出す）。プリビルドの std は wasm-exceptions 付きで
コンパイルされているため、std ごと組み直すしかない。**つまり nightly 必須。**

`[profile.web-release]` に `panic = "abort"` を入れてある。

### 3. リンカ引数は `cargo::rustc-link-arg-bins=` で渡す

`RUSTFLAGS` や `.cargo/config.toml` のグローバル指定だと、同時にビルドされる
cdylib にも適用されて壊れる。`crates/voicevox_core_wasm/build.rs` から
bin 限定で渡している。

### 4. rlib 内の `no_mangle` シンボルは `-sEXPORTED_FUNCTIONS` で保持される

bin 側から参照する必要は無い（実測確認済み）。
`build.rs` が `voicevox_core_c_api/src/*.rs` を走査して
`#[unsafe(no_mangle)] pub … extern "C" fn` を拾い、`_` を付けて列挙する。

**cfg で無効な関数を列挙するとリンクエラーになる**ので、
`load-onnxruntime` / `link-onnxruntime` にgateされた関数と
`compatible_engine.rs` は除外している。

### 5. C/C++ は `-fPIC` でビルドする

bin 側には不要だが、同時にビルドされる `voicevox_core_c_api` の cdylib
(SIDE_MODULE) のリンクに要る。無いと libopenjtalk.a で
`relocation R_WASM_MEMORY_ADDR_SLEB cannot be used against symbol` が出る。

### 6. bin と lib の出力ファイル名が衝突する

`voicevox_core_c_api` の `[lib] name` が `voicevox_core` なので、bin を
同名にすると `deps/voicevox_core.wasm` が衝突する(rust-lang/cargo#6313)。
bin 名は `voicevox_core_web` にしてある。配信時にリネームする場合は
`.js` 内の wasm ファイル名参照も置換すること。

### 7. `chrono::Local` は emscripten で動かない

iana-time-zone 越しに wasm-bindgen を呼び、
`cannot call wasm-bindgen imported functions on non-wasm targets` でパニックする。
`voicevox_core_c_api/src/lib.rs` のログ時刻を emscripten では `Utc` にし、
chrono の feature も `clock` → `now` に落として iana-time-zone ごと外した。

### 8. 構造体の値渡しは `ccall` から呼べない

wasm32 の C ABI では構造体の値渡しが間接渡しになるため、Emscripten の
`ccall` から `voicevox_synthesizer_new` などをそのまま呼ぶと壊れる。

→ `voicevox_core_c_api/src/wasm.rs` にオプションをスカラーにばらした
シムAPIを用意した。JS からはこちらを呼ぶこと。

| シム | 元の関数 |
|---|---|
| `voicevox_wasm_synthesizer_new` | `voicevox_synthesizer_new` |
| `voicevox_wasm_synthesizer_load_voice_model` | `voicevox_synthesizer_load_voice_model` |
| `voicevox_wasm_synthesizer_synthesis` | `voicevox_synthesizer_synthesis` |
| `voicevox_wasm_synthesizer_tts` | `voicevox_synthesizer_tts` |

## JS 側の使い方

推論に到達する呼び出しは ASYNCIFY 越しなので `{async: true}` + `await` が要る。

```js
import * as ort from 'onnxruntime-web';
const factory = (await import('./voicevox_core_web.js')).default;
const M = await factory({ ortModule: ort });   // or Module.ortScriptUrl

// 推論に到達しない → 同期
M.ccall('voicevox_open_jtalk_rc_new', 'number', ['number','number'], [dictPtr, out]);

// 推論に到達する → async
await M.ccall('voicevox_onnxruntime_init_once', 'number', ['number'], [out], {async: true});
await M.ccall('voicevox_wasm_synthesizer_load_voice_model', 'number',
              ['number','number','number'], [synth, model, 0], {async: true});
await M.ccall('voicevox_synthesizer_create_audio_query', 'number',
              ['number','number','number','number'], [synth, text, styleId, out], {async: true});
await M.ccall('voicevox_wasm_synthesizer_synthesis', 'number',
              ['number','number','number','number','number','number'],
              [synth, query, styleId, 1, lenOut, wavOut], {async: true});
```

**ASYNCIFY 中の再入は abort する。** 呼び出しは直列化すること。

onnxruntime-web の渡し方は2通り:
- `Module.ortModule` に import 済みのモジュールを渡す
- `Module.ortScriptUrl` に URL を渡して動的 import させる（`Module.ortWasmPaths` も）

辞書は Emscripten の仮想FSに置く。`open_jtalk_dic_utf_8-1.11` に実在するのは
`sys.dic` / `unk.dic` / `matrix.bin` / `char.bin` / `left-id.def` / `right-id.def` /
`pos-id.def` / `rewrite.def` の8ファイル（`feature.def` は無い）。

## 動作確認の状況

Node (WSL2, 4コア) で `model/sample.vvm`(ONNX形式, style_id 302) を使い、
テキスト → AudioQuery → WAV まで完走を確認済み。音声波形も出ている
(peak 3817 / RMS 471 / 非ゼロ 97%)。

ブラウザ(Chrome)でも wasm コンパイル 111ms、onnxruntime-web 読み込み、
仮想FSへの辞書展開まで確認済み。

### サイズ

| | 生 | gzip後 |
|---|---|---|
| `voicevox_core_web.wasm` | 3.8 MB | **1.2 MB** |
| `voicevox_core_web.js` | 86 KB | — |

### 速度（Node, sample.vvm）

| テキスト | 音声長 | 1スレッド | 4スレッド |
|---|---|---|---|
| 「こんにちは。」 | 1.06秒 | 3181ms (RTF 3.01) | 1281ms (**RTF 1.21**) |
| 27文字 | 5.18秒 | — | 5264ms (**RTF 1.02**) |

初期化コスト(1回だけ): wasmインスタンス化 40-180ms / 辞書展開(102MB) 180-600ms /
`open_jtalk_rc_new` 130-250ms / VVMのセッション作成 4-5秒。

**マルチスレッドで約2.5倍速くなる。** ただし SharedArrayBuffer には
cross-origin isolation (COOP/COEP) が要る。無い場合 onnxruntime-web は
自動的に1スレッドになる(`wasm_library.js` が明示的にも設定している)。

## 音声モデルの制約（重要）

**公開されている VVM (voicevox_vvm の全リリース) は `vv_bin` 形式で、
onnxruntime-web では読めない。**

- `vv_bin` は**暗号化された ONNX**。復号は VOICEVOX 製 ONNX Runtime の中だけ
  (VOICEVOX/ort#8, voicevox_project#24)
- 復号処理は `onnxruntime-builder` が参照する非公開フォークにあり、
  wasm ビルドターゲットも存在しない
- voicevox_vvm と VOICEVOX ONNX Runtime の利用規約は
  「逆コンパイル・リバースエンジニアリング及びこれらの方法の公開」を明示的に禁止。
  **復号ルートは取らない**

`onnxruntime_wasm.rs` の `new_session` は `ModelBytes::VvBin` を
明示的なエラーメッセージで弾く。

→ **ONNX形式のVVMを自前で用意する。** `docs/custom-model-training.md` を参照。

## 変更したファイル

| ファイル | 変更 |
|---|---|
| `Cargo.toml` | `[profile.web-release]`、open_jtalk をフォークに向ける |
| `crates/voicevox_core/Cargo.toml` | `web-onnxruntime` feature、`ort` を optional 化 |
| `crates/voicevox_core/build.rs` | `compile_error!` を3択に |
| `crates/voicevox_core/src/lib.rs` | feature の相互排他チェック |
| `crates/voicevox_core/src/core/devices.rs` | `SupportedDevices::THIS` に web 分岐 |
| `crates/voicevox_core/src/core/infer/runtimes.rs` | cfg でモジュールを差し替え |
| `crates/voicevox_core/src/core/infer/runtimes/onnxruntime_wasm.rs` | **新規**。InferenceRuntime 第2実装 |
| `crates/voicevox_core/wasm_library.js` | **新規**。onnxruntime-web への JS ブリッジ |
| `crates/voicevox_core/src/__internal.rs` | doctest_fixtures を web では外す |
| `crates/voicevox_core_c_api/Cargo.toml` | `rlib` 追加、`web-onnxruntime`、chrono/process_path のターゲット分け |
| `crates/voicevox_core_c_api/src/lib.rs` | `wasm` モジュール、init_once の gate、chrono、compatible_engine |
| `crates/voicevox_core_c_api/src/c_impls.rs` | `init_once` の gate |
| `crates/voicevox_core_c_api/src/wasm.rs` | **新規**。シムAPI |
| `crates/voicevox_core_wasm/` | **新規パッケージ**。bin ターゲットと build.rs |

依存フォーク: [mossan-hoshi/open_jtalk-rs](https://github.com/mossan-hoshi/open_jtalk-rs) の
`web/wasm` ブランチ（upstream rev `7c87b422` + wasm32向け bindings 1コミット）。
Open JTalk の C/C++ は emcc で無改造で通り、足りないのは事前生成 bindings だけだった。

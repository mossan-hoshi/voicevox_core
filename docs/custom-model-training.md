# VOICEVOX互換 ONNXモデル 入出力仕様

voicevox_core 0.17.0 のソースから読み取った、自前でモデルを学習してVVMを作るための仕様。
数値はすべてコードから確認したもので、推測は「推定」と明記する。

対象は `talk` ドメイン（最小構成）。必要なONNXは **3つだけ**。

---

## 全体像

```
テキスト
  ↓ Open JTalk (voicevox_core が担当。学習不要)
アクセント句 + 音素列
  ↓ ① predict_duration          音素ごとの長さ
  ↓ ② predict_intonation        モーラごとの音高(f0)
  ↓ (フレーム展開。voicevox_core が担当)
  ↓ ③ decode                    波形
WAV (24000Hz)
```

`AudioQuery` は ① と ② の出力を編集可能な形で保持したもの。だから
話速・音高・アクセント・「間」の調整が効く。この構造を持てるのが
Piper に対する優位点。

---

## 音素セット（語彙サイズ 45）

`crates/voicevox_core/src/engine/acoustic_feature_extractor.rs` の
`PhonemeCode` enum (L399-446) の判別子がそのままID。

| ID | 音素 | ID | 音素 | ID | 音素 |
|---|---|---|---|---|---|
| 0 | `pau` | 15 | `f` | 30 | `o` |
| 1 | `A` | 16 | `g` | 31 | `p` |
| 2 | `E` | 17 | `gw` | 32 | `py` |
| 3 | `I` | 18 | `gy` | 33 | `r` |
| 4 | `N` | 19 | `h` | 34 | `ry` |
| 5 | `O` | 20 | `hy` | 35 | `s` |
| 6 | `U` | 21 | `i` | 36 | `sh` |
| 7 | `a` | 22 | `j` | 37 | `t` |
| 8 | `b` | 23 | `k` | 38 | `ts` |
| 9 | `by` | 24 | `kw` | 39 | `ty` |
| 10 | `ch` | 25 | `ky` | 40 | `u` |
| 11 | `cl` | 26 | `m` | 41 | `v` |
| 12 | `d` | 27 | `my` | 42 | `w` |
| 13 | `dy` | 28 | `n` | 43 | `y` |
| 14 | `e` | 29 | `ny` | 44 | `z` |

- 並びは「`pau` を先頭に、あとはアルファベット昇順（大文字が先）」
- 大文字 `A E I O U` は**無声化母音**
- `cl` は促音（っ）、`N` は撥音（ん）
- Open JTalk の `sil` は ID化の際に `pau`(0) に潰される
- 無声扱いの音素: `A I U E O cl pau`（`N` は有声扱い）

---

## ① predict_duration

音素列から音素ごとの長さ（秒）を出す。

| | 名前 | 型 | 形 | 内容 |
|---|---|---|---|---|
| 入力 | `phoneme_list` | int64 | `[N]` | 音素ID列 |
| 入力 | `speaker_id` | int64 | `[1]` | 話者ID |
| 出力 | `phoneme_length` | float32 | `[N]` | 各音素の長さ（秒） |

音素列の組み立て（`interpret_query.rs` L16-51）:

```
[pau] + 各モーラ([子音?] + [母音]) + [pau]
```

---

## ② predict_intonation

アクセント情報からモーラごとの音高を出す。**長さはモーラ単位**（音素単位ではない）。

| | 名前 | 型 | 形 | 内容 |
|---|---|---|---|---|
| 入力 | `length` | int64 | `[]` スカラー | モーラ数 |
| 入力 | `vowel_phoneme_list` | int64 | `[N_mora]` | モーラ末尾の音素ID |
| 入力 | `consonant_phoneme_list` | int64 | `[N_mora]` | 子音の音素ID。無ければ `-1` |
| 入力 | `start_accent_list` | int64 | `[N_mora]` | 0/1 |
| 入力 | `end_accent_list` | int64 | `[N_mora]` | 0/1 |
| 入力 | `start_accent_phrase_list` | int64 | `[N_mora]` | 0/1 |
| 入力 | `end_accent_phrase_list` | int64 | `[N_mora]` | 0/1 |
| 入力 | `speaker_id` | int64 | `[1]` | 話者ID |
| 出力 | `f0_list` | float32 | `[N_mora]` | 音高 |

4つのフラグの意味（`synthesizer.rs` L635-745）:

| 配列 | 1 が立つ位置 |
|---|---|
| `start_accent_list` | アクセントが上がるモーラ |
| `end_accent_list` | アクセント核（下がる直前）のモーラ |
| `start_accent_phrase_list` | アクセント句の先頭モーラ |
| `end_accent_phrase_list` | アクセント句の末尾モーラ |

`vowel_phoneme_list` に入りうるのは13種（`pau A E I N O U a cl e i o u`）、
`consonant_phoneme_list` は `-1` + 子音32種。

**f0 の単位**: コード上に明示は無いが、テスト値 `5.905218` / `5.565851` が
`exp()` するとそれぞれ約367Hz / 261Hz になるため **自然対数スケールのHz と推定**。
無声母音のモーラは voicevox_core 側で `0.0` に上書きされる。

---

## ③ decode

f0と音素のフレーム列から波形を出す。要するに音響モデル＋ボコーダ。

| | 名前 | 型 | 形 | 内容 |
|---|---|---|---|---|
| 入力 | `f0` | float32 | `[T, 1]` | フレームごとの音高 |
| 入力 | `phoneme` | float32 | `[T, 45]` | フレームごとの音素 one-hot |
| 入力 | `speaker_id` | int64 | `[1]` | 話者ID |
| 出力 | `wave` | float32 | `[T * 256]` | 波形 |

### フレームの決まり方

```
サンプリングレート = 24000 Hz
hop長             = 256 samples
フレームレート     = 24000 / 256 = 93.75 frames/sec
1フレーム         = 約 10.667 ms
```

各音素のフレーム数（`interpret_query.rs` L164-174）:

```rust
frames = round_ties_even(round_ties_even(phoneme_length * 93.75) / speed_scale)
```

四捨五入ではなく**偶数丸めを2段**。VOICEVOX ENGINE と揃えるため。

`T` = 全音素のフレーム数の合計。**出力波形長 = T × 256**。

### phoneme

完全な one-hot。`phoneme[t][音素ID] = 1.0`、他は0。次元45は音素数と一致。

### f0

モーラの f0 をそのモーラの**子音＋母音の合計フレーム数分**複製する
（子音フレームも同じ値を持つ）。無声フレームは `0.0`。

### パディング（重要）

`decode` に渡す前に、voicevox_core が前後に **38フレーム**のパディングを付ける
（`core/adjust/pre.rs`）。`0.4秒 × 24000 / 256 = 37.5 → 38`。

- `f0`: 前後に `0.0` を38フレーム
- `phoneme`: 前後に「`pau` の列だけ 1.0」の one-hot を38フレーム

推論後に `wave[38*256 .. len-38*256]` を切り出す。
**学習時も同じパディングを前提にする必要がある。**

---

## ストリーミング合成（任意）

`decode` を2段に分けると、チャンク単位の逐次再生ができる。
Android側で使っている構成。

| モデル | 入力 | 出力 |
|---|---|---|
| `generate_full_intermediate` | `f0 [T,1]`, `phoneme [T,45]`, `speaker_id [1]` | `spec [T, ?]` |
| `render_audio_segment` | `spec [T, ?]` | `wave [T*256]` |

要するに音響モデル（→メルスペクトログラム等）とボコーダの分離。
追加で `MARGIN = 14` フレームのマージンを扱う。

---

## VVMの作り方

`.vvm` は単なる zip。

```
mymodel.vvm
├── manifest.json
├── metas.json
├── predict_duration.onnx
├── predict_intonation.onnx
└── decode.onnx
```

### manifest.json

```json
{
  "vvm_format_version": 2,
  "id": "<UUID>",
  "metas_filename": "metas.json",
  "talk": {
    "predict_duration":   { "type": "onnx", "filename": "predict_duration.onnx" },
    "predict_intonation": { "type": "onnx", "filename": "predict_intonation.onnx" },
    "decode":             { "type": "onnx", "filename": "decode.onnx" },
    "style_id_to_inner_voice_id": { "302": 0 }
  }
}
```

`style_id_to_inner_voice_id` は「公開スタイルID → モデル内の speaker_id」の対応。
1話者なら `{"<好きなID>": 0}` でよい。

### metas.json

```json
[
  {
    "name": "キャラ名",
    "styles": [ { "name": "ノーマル", "id": 302 } ],
    "speaker_uuid": "<UUID>",
    "version": "0.0.1"
  }
]
```

`styles[].type` は省略時 `"talk"`。ストリーミングを使うなら `"streaming_talk"`。

---

## 学習データとして必要なもの

合成コーパス（qwen3-tts等）から作る場合、音声と書き起こしに加えて以下が要る:

| モデル | 教師データ | 作り方 |
|---|---|---|
| `predict_duration` | 音素ごとの実測長 | Open JTalkで音素列を出し、音声と強制アラインメント（Julius / MFA等） |
| `predict_intonation` | モーラごとのf0 | WORLD (pyworld) 等でf0抽出 → モーラ区間で代表値を取る → log化 |
| `decode` | (f0, phoneme one-hot) → 波形 | 上記2つのフレーム展開結果と元音声のペア |

つまり**強制アラインメントとf0抽出のパイプラインが必要**。
Piperは書き起こしと音声だけで済むので、この点の手間は増える。

---

## 参照した実装

- `crates/voicevox_core/src/core/infer/domains/talk.rs` — 入出力シグネチャの正
- `crates/voicevox_core/src/core/infer/domains/streaming_talk.rs` — ストリーミング版
- `crates/voicevox_core/src/engine/acoustic_feature_extractor.rs` — 音素とID
- `crates/voicevox_core/src/engine/talk/interpret_query.rs` — フレーム展開
- `crates/voicevox_core/src/synthesizer.rs` — アクセントフラグの組み立て、パディング
- `crates/voicevox_core/src/engine/mora_mappings.rs` — モーラ144種と(子音,母音)の対応
- `crates/voicevox_core/src/core/metas.rs` — metas.json のスキーマ
- `docs/guide/dev/vvm.md` — VVM形式の公式説明
- `model/sample.vvm/` — ONNX形式の実例（動作確認済み）

---

# 学習ルート（全部公開されている）

2026-08-24 に GitHub API で実在とファイル構成を確認した。

## リポジトリ

| リポジトリ | 役割 | 最終push | 備考 |
|---|---|---|---|
| [Hiroshiba/vv_core_inference](https://github.com/Hiroshiba/vv_core_inference) | 推論 + **ONNXエクスポート** | 2025-12-03 | MIT。`convert.py` |
| [Hiroshiba/yukarin_s](https://github.com/Hiroshiba/yukarin_s) | ① predict_duration の学習 | 2025-05-27 | `train.py` |
| [Hiroshiba/yukarin_sa](https://github.com/Hiroshiba/yukarin_sa) | ② predict_intonation の学習 | 2025-05-27 | `train.py` |
| [Hiroshiba/yukarin_sosoa](https://github.com/Hiroshiba/yukarin_sosoa) | ③ 音響モデルの学習 | 2025-12-08 | `train.py` |
| [Hiroshiba/hifi-gan](https://github.com/Hiroshiba/hifi-gan) | ③ ボコーダの学習 | 2025-05-27 | `train.py` |

`vv_core_inference` の README に明記されている:

> VOICEVOX のコア内で用いられているディープラーニングモデルの推論コード。**VOICEVOX コア用の onnx モデルを制作できる。**

> ### 公開している意図
> VOICEVOX コアでの音声合成をより高速・軽量にするための手法の議論や提案を受けられるようにするためです。

ただし各学習リポジトリのREADMEは中身がほぼ空（`# hiho-pytorch-base` の1行だけ等）で、
**学習手順は文書化されていない**。コードを読んで進める必要がある。

## ONNXエクスポート

```bash
uv run convert.py \
  --yukarin_s_model_dir     "model/yukarin_s" \
  --yukarin_sa_model_dir    "model/yukarin_sa" \
  --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
  --hifigan_model_dir       "model/hifigan"
```

- `yukarin_sosoa` フォルダに **hifi_gan と合わせた `decode.onnx`** が出る
- `speaker_ids` に何を指定しても、出るONNXは全 speaker_id 対応

つまり学習した4モデル → `convert.py` → voicevox_core が読む3つのONNX、という流れ。

## 必要な学習データ

### ① yukarin_s (duration)

`yukarin_s/dataset.py` より、入力は **Julius形式のラベルファイル** 1種類だけ:

```python
OjtPhoneme.load_julius_list(self.phoneme_list_path)
phoneme_length = numpy.array([p.end - p.start for p in phoneme_list_data])
```

音素の開始・終了時刻が入った `.lab`。**強制アラインメントの出力そのもの**。

### ② yukarin_sa (intonation)

音素ラベル + モーラごとのf0。

### ③ yukarin_sosoa (音響モデル)

`yukarin_sosoa/config.py` の `DatasetConfig` より、以下のパスリストが要る:

| 項目 | 内容 |
|---|---|
| `f0_pathlist_path` | フレーム単位のf0 |
| `phoneme_pathlist_path` | フレーム単位の音素 |
| `spec_pathlist_path` | スペクトログラム |
| `silence_pathlist_path` | 無音フラグ |
| `phoneme_list_pathlist_path` | 音素ラベル（Julius形式） |
| `volume_pathlist_path` | 音量（任意） |
| `speaker_dict_path` | 話者辞書（多話者時） |

### まとめ: 前処理パイプライン

```
音声 + 書き起こし
  ├→ Open JTalk で音素列
  ├→ 強制アラインメント (Julius / MFA) → .lab（音素の開始終了時刻）
  ├→ f0抽出 (pyworld 等) → フレーム単位 + モーラ単位
  ├→ スペクトログラム抽出
  └→ 無音区間検出
```

**Piperは「音声＋書き起こし」だけで済むので、ここが追加コスト。**
強制アラインメントとf0抽出のパイプラインを組む必要がある。

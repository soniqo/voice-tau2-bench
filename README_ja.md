# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[speech-swift](https://github.com/soniqo/speech-swift) でローカル推論する独立した **τ-Voice 音声エージェントベンチマーク**です。Apple Silicon 上で顧客対応の会話、ツール利用、タスク完了を評価します。

固定された英語 `voice-v1` は計 **278 タスク**です。航空 **50**、小売 **114**、通信 **114**。タスク、方針、ツール、データベース、報酬基準は上流の固定内容を保持します。

## ローカルモデル構成

| 役割 | モデル | 精度 |
| --- | --- | --- |
| エージェント、通話者、傾聴判断、意味評価 | Gemma 4 E4B IT | INT4 |
| 両者の音声合成 | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| 音声認識 | Whisper Large-v3 (full, CoreML) | FP16 |
| 発話検出 | Silero VAD (CoreML) | Native |

モデルのダウンロードは準備時のみです。推論にクラウドや API キーは使いません。必要な speech-swift ビルドとモデルの準備は[実行環境の設定](docs/runtime-setup.md)を参照してください。

## 実測結果

開発用サンプル：固定順で各分野の最初のタスクを選択し、計 **3 タスク**、各 **1 回**、シード **42**、会話期限 **600 秒**、クリーンな `control` 電話音声を使用。機材は **Apple M5 Pro、統合メモリ 48 GB**。成功率は上記の構成全体を評価します。

`soniqo-local-v1` · `2026-10-23`

| エージェントモデル | 航空 | 小売 | 通信 | 全体 pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

この小さな決定的サンプルはローカルの通話者・評価モデルを使う独自設定です。278 タスク全体のスコアや公式ランキングを示しません。タイムアウトはゼロ点、推論例外は未採点です。

## クイックスタート

Python **3.12/3.13**、Apple Silicon、**macOS 15+** が必要です。タスクの前に[ローカルモデルサービス](docs/runtime-setup.md#serve)を起動してください。

```bash
git clone https://github.com/soniqo/voice-tau2-bench.git
cd voice-tau2-bench
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

```bash
uv run --frozen tau-voice batch \
  --domain all --num-tasks 1 --trials 1 --seed 42 \
  --max-seconds 600 --complexity control \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-baseline
```

各タスクはツールと会話履歴、報酬内訳、両者の WAV、時間、設定、ソースとモデルの指紋を保存します。同じバッチは `--resume` で再開します。`--num-tasks` を省略すると全 278 タスクを評価します。

## 検証と評価

```bash
make test
make lint
make verify
```

[評価規則](docs/evaluation.md)、[構成](docs/local-voice-extraction.md)、[モデル実行記録](docs/results/gemma4-e4b-local-600.json)を参照してください。Mock とコンポーネントの検証は開発用で、タスクスコアには含みません。

## 出典

[Sierra Research の τ²-Bench](https://github.com/sierra-research/tau2-bench)、コミット `05736c27219b769d0be507574dd3135010916644` に由来します。元の [MIT ライセンス](LICENSE)を保持します。利用時は [τ-Voice 論文](https://arxiv.org/abs/2603.13686)を引用してください。

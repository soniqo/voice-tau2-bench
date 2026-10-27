# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) は、Apple Silicon 上の**ローカル音声エージェント性能**を測るため、この独立した τ-Voice ベンチマークを管理しています。エージェント、模擬発信者、音声モデル、意味評価モデルを [speech-swift](https://github.com/soniqo/speech-swift) で実行します。

## ドキュメント

- [セットアップとモデルファイル](docs/runtime-setup.md)
- [評価と採点](docs/evaluation.md)
- [実装](docs/local-voice-extraction.md)
- [モデル実行記録](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B の実行記録と動作パラメータ](docs/validation/gemma4-12b-voice-e2e.md)

## 測定するもの

音声エージェントが会話とツール利用で顧客対応タスクを完了できるかを測ります。タスク報酬は、求められる最終データベース状態と伝達内容を確認します。

固定された英語 `voice-v1` は計 **278 タスク**です。航空 **50**、小売 **114**、通信 **114**。タスク、方針、ツール、データベース、報酬基準は上流の固定内容を保持します。

## テストしたモデルと結果

| 役割 | モデル | 精度 |
| --- | --- | --- |
| エージェント | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| 通話者、傾聴判断、意味評価 | Gemma 4 E4B IT | INT4 |
| 両者の音声合成 | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| 音声認識 | Whisper Large-v3 (full, CoreML) | FP16 |
| 発話検出 | Silero VAD (CoreML) | Native |

開発用サンプル：固定順で各分野の最初のタスクを選択し、計 **3 タスク**、各 **1 回**、シード **42**、会話期限 **600 秒**、クリーンな `control` 電話音声を使用。機材は **Apple M5 Pro、統合メモリ 48 GB**。成功率は上記の構成全体を評価します。

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| エージェントモデル | 航空 | 小売 | 通信 | 全体 pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

この小さな決定的サンプルはローカルの通話者・評価モデルを使う独自設定です。278 タスク全体のスコアや公式ランキングを示しません。タイムアウトはゼロ点、推論例外は未採点です。

12B のサンプルは未完了です。通信タスクでネイティブ推論エラーが発生し、未採点です。制御された 12B 音声チェックでは、正しいデータベース書き込みが **1.0**、誤った書き込みの再評価が **0.0** でした。[動作パラメータ](docs/validation/gemma4-12b-voice-e2e.md)。発信者の入力は固定で、ベンチマークの得点には含めません。分野別の実行には生成モデルの発信者を使い、各ランタイムのリビジョンはリンク先の記録に保存しています。

以前のローカル v2 開発診断：**Gemma 4 12B IT (INT4)、生の結果：航空 0/1、小売 0/1、通信 0/1**。状態変更を必要とする 3 タスクを各 1 回、シード **70** で実行し、すべての通話が **300 秒のシミュレーション上限**（**実時間の監視上限は 900 秒**）に達しました。この短縮実行の結果は、選択した 3 タスクのみを対象とします。[結果と検証済みパラメータ](docs/validation/development-smoke.md)。

これらの診断結果は[ツール履歴と発話タイミングの修正](docs/validation/tool-history-and-pacing.md)より前のものです。新しい機能検証では呼び出し側・判定側の **84/84** 項目に合格し、固定音声による正しい書き込みは **1.0**、誤った書き込みの再生は **0.0** でした。

`soniqo-local-gemma12b-v2` · `2026-10-27` · `--max-seconds 300 --wall-timeout 900`

## ローカルで実行

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

チェック: `make test` · `make lint` · `make verify`.

出典: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice 論文](https://arxiv.org/abs/2603.13686) · [MIT ライセンス](LICENSE).

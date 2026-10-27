# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) は、音声エージェントが会話とツール利用でタスクを完了できるかを測定する本ベンチマークを保守しています。固定された英語データには **278 タスク**が含まれます：航空 **50**、小売 **114**、通信 **114**。

[セットアップ](docs/runtime-setup.md) · [採点](docs/evaluation.md) · [モデル比較と制限](docs/model-comparison.md)

## モデルとシステム

| エージェントまたはシステム | 現在の対応状況 |
| --- | --- |
| Gemma 4 12B IT, INT4 | 検証済みのローカル v2 プロファイル |
| Gemma 4 E4B IT, INT4 | ローカルランタイムで対応。同条件の v2 比較は未実施 |
| OpenAI GPT-Live + 明示したバックエンドエージェント | アダプターが必要 |
| OpenAI Realtime | アダプターが必要 |
| Gemini Live | アダプターが必要 |
| ElevenLabs Agents | アダプターが必要 |

ローカル構成は [speech-swift](https://github.com/soniqo/speech-swift) を使用します：完全版 Whisper Large-v3 FP16、VoxCPM2 主重み FP16 / AudioVAE FP32、Silero VAD、呼び出し側・聞き取り補助・判定役の Gemma E4B。クラウドシステムはまだ統合・評価していません。

## 比較指標

| 指標 | 測定内容 |
| --- | --- |
| タスク成功率、pass^1 | 必要な最終データベース状態と伝達内容 |
| カバレッジと失敗 | 採点済み/予定試行数、通常終了、採点対象外の推論エラー |
| 遅延、p50 / p95 | エージェントの LLM・ASR 実行時間と TTS の最初の音声までの時間 |
| 音声応答の遅延 | 呼び出し側の発話終了からエージェントの最初の音声まで。標準集計は未実装 |
| ツールエラー率 | 失敗・拒否されたエージェント呼び出し / 試行した呼び出し |
| 完了時間 | シミュレーション通話時間と実測経過時間を別々に記録 |
| 最大メモリ | 測定範囲を明記したネイティブプロセスの物理メモリ使用量 |
| API コスト | 試行・成功試行あたりの USD。失敗試行も含む |

同じタスク、シード、通話制限、音響条件、呼び出し側・判定設定で比較します。ローカル LLM の比較では音声構成を固定します。音声ネイティブおよびマネージドエージェントでは構成全体を明示します。各コンポーネントの時間と音声応答全体の遅延は別の指標です。

## 記録済みのローカル診断

| エージェント | 採点済み / 予定 | Pass^1 | ツールエラー | LLM p50 / p95 | TTS 最初の音声 p50 / p95 | API コスト / 試行 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

航空タスク 12、1 試行、シード 70、v2。上限は **600 シミュレーション秒 / 1,200 実時間秒**です。通話は **312.9 シミュレーション秒 / 554.4 実測秒**で通常終了しました。ハードウェア：Apple M5 Pro、48 GB、共有ワークステーション。開発用の 1 試行であり、同条件の複数モデルランキングは未実施です。 [データと測定定義](docs/results/gemma4-12b-local-v2-airline.json).

## ローカル実行

Python **3.12/3.13**、Apple Silicon、**macOS 15+**。[セットアップ](docs/runtime-setup.md)と [v2 実行](docs/local-v2.md)に従い、ローカルサービスの起動、呼び出し側・判定役の検証、タスク実行を行います。

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

`--num-tasks`、`--trials`、`--max-seconds`、`--wall-timeout` で実行を制限します。クラウドアダプターと USD 支出上限は**未実装**です。現在の推論はローカルで行われ、クラウド API キーは不要です。

確認: `make test` · `make lint` · `make verify`.

出典: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).

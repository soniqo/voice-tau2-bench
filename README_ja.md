# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) は、音声エージェントが会話とツール利用でタスクを完了できるかを測定する本ベンチマークを保守しています。固定された英語データには **278 タスク**が含まれます：航空 **50**、小売 **114**、通信 **114**。

[セットアップ](docs/runtime-setup.md) · [採点](docs/evaluation.md) · [モデル比較と制限](docs/model-comparison.md)

## モデルとシステム

| エージェントまたはシステム | 現在の対応状況 |
| --- | --- |
| Gemma 4 12B IT, INT4 | 検証済みのローカル v2 プロファイル |
| Gemma 4 E4B IT, INT4 | ローカルランタイムで対応。同条件の v2 比較は未実施 |
| Amazon Nova 2.5 Sonic | ネイティブ音声を検証済み。Retail pass^1 **1/1**（v6 の開発試行 1 件、上限 420 秒。[記録](docs/results/nova25-v6-retail-development.json)） |
| OpenAI GPT-Live + 明示したバックエンドエージェント | アダプターが必要 |
| OpenAI Realtime | アダプターが必要 |
| Gemini Live | アダプターが必要 |
| ElevenLabs Agents | アダプターが必要 |

ローカル構成は [speech-swift](https://github.com/soniqo/speech-swift) を使用します：完全版 Whisper Large-v3 FP16、VoxCPM2 主重み FP16 / AudioVAE FP32、Silero VAD、呼び出し側・聞き取り補助・判定役の Gemma E4B。Nova 2.5 Sonic は任意の[ネイティブ音声ランナー](docs/nova.md)で利用でき、呼び出し側と判定役はローカルで動作します。

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

同じタスク、シード、通話制限、音響条件、呼び出し側・判定設定で比較します。ローカル LLM の比較では音声構成を固定します。音声ネイティブおよびマネージドエージェントでは構成全体を明示します。各コンポーネントの時間と音声応答全体の遅延は別の指標です。

## 記録済みの開発試験結果

各行は開発用の 1 試行です。タスクと時計の設定が異なり、同条件のモデルランキングは未実施です。

| エージェント | 条件 / タスク | 採点済み / 予定 | Pass^1 | ツールエラー | 実時間 |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · 航空 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · 小売 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

航空タスク 12、1 試行、シード 70、v2。上限は **600 シミュレーション秒 / 1,200 実時間秒**です。通話は **312.9 シミュレーション秒 / 554.4 実測秒**で通常終了しました。ハードウェア：Apple M5 Pro、48 GB、共有ワークステーション。開発用の 1 試行であり、同条件の複数モデルランキングは未実施です。 [データと測定定義](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova：1 試行、シード 70、control 音響条件、**実時間 / 壁時計の上限 420 秒**。指定の 2 商品を交換し、**5/5** の参照操作と最終データベース状態が一致しました。通話は **音声 259.0 秒 / 実時間 296.3 秒**で通常終了しました。[実行記録と制限](docs/results/nova25-v6-retail-development.json)。

## ローカル実行

Python **3.12/3.13**、Apple Silicon、**macOS 15+**。[セットアップ](docs/runtime-setup.md)と [v2 実行](docs/local-v2.md)に従い、ローカルサービスの起動、呼び出し側・判定役の検証、タスク実行を行います。

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

このコマンドは固定されたネイティブランタイムをビルドし、モデルをダウンロードし、発信者と評価器を校正して、制御された音声スモークテストを実行します。ドメインベンチマークのスコアは生成しません。新しい出力ディレクトリを使用してください。[セットアップとタスク実行](docs/runtime-setup.md#one-command-local-run)。

`--num-tasks` · `--trials` · `--max-seconds` · `--wall-timeout`. 既定の実行はローカルです。任意の [Nova 実行](docs/nova.md)には明示的な AWS プロファイルが必要です。永続的な予算記録と使用量のしきい値を提供しますが、プロバイダーが強制する料金上限ではありません。

確認: `make test` · `make lint` · `make verify`.

出典: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).

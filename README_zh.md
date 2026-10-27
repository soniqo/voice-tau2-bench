# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) 维护此独立的 τ-Voice 基准，用于衡量 Apple Silicon 上的**本地语音智能体性能**。智能体、模拟来电者、语音模型和语义评分模型均通过 [speech-swift](https://github.com/soniqo/speech-swift) 运行。

## 文档

- [环境设置与模型文件](docs/runtime-setup.md)
- [评估与评分](docs/evaluation.md)
- [实现方式](docs/local-voice-extraction.md)
- [模型运行记录](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B 运行记录与可用参数](docs/validation/gemma4-12b-voice-e2e.md)

## 衡量什么

我们衡量语音智能体能否通过对话和工具使用完成客服任务。任务奖励检查所需的最终数据库状态和沟通内容。

固定的英语 `voice-v1` 包含 **278 个任务**：航空 **50**、零售 **114**、电信 **114**。任务、策略、工具、数据库和奖励规则保留上游指纹。

## 已测试模型与结果

| 职责 | 模型 | 精度 |
| --- | --- | --- |
| 智能体 | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| 来电者、倾听决策、语义评分 | Gemma 4 E4B IT | INT4 |
| 双方语音生成 | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| 语音识别 | Whisper Large-v3 (full, CoreML) | FP16 |
| 语音检测 | Silero VAD (CoreML) | Native |

开发样本：每个领域按固定顺序选择首个任务，合计 **3 个任务**，每个 **1 次**，种子 **42**，对话时限 **600 秒**，干净的 `control` 电话音频。硬件：**Apple M5 Pro，48 GB 统一内存**。任务成功率衡量上述完整模型栈。

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| 智能体模型 | 航空 | 零售 | 电信 | 总计 pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

该小型确定性样本使用本地来电者和评分模型，属于自定义配置，不能代表完整 278 任务得分或上游排行榜结果。超时得零分；推理异常不计分。

12B 样本尚未完成：电信任务出现本地推理错误，未获得分数。受控的 12B 语音检查：正确数据库写入得分 **1.0**，错误写入重放得分 **0.0**。[可用参数](docs/validation/gemma4-12b-voice-e2e.md)。此检查使用固定的来电语句，不计入基准分数。领域测试使用生成式来电者；链接报告记录各次运行的运行时版本。

较早的本地 v2 开发诊断：**Gemma 4 12B IT (INT4)，原始结果：航空 0/1、零售 0/1、电信 0/1**。三个需要改变状态的任务，各运行一次，种子为 **70**；所有通话均达到 **300 秒模拟时长上限**（**900 秒实际时间超时**）。这些缩短运行的结果仅涵盖所选的三个任务。[结果与已验证参数](docs/validation/development-smoke.md)。

这些诊断结果早于[工具历史和调用者节奏修复](docs/validation/tool-history-and-pacing.md)。新的功能验证通过了 **84/84** 项调用者/评判检查；受控语音写入得分 **1.0**，错误写入回放得分 **0.0**。

`soniqo-local-gemma12b-v2` · `2026-10-27` · `--max-seconds 300 --wall-timeout 900`

## 本地运行

需要 Python **3.12/3.13**、Apple Silicon 和 **macOS 15+**。运行任务前启动[本地模型服务](docs/runtime-setup.md#serve)。

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

每个任务保存工具/对话轨迹、奖励分项、双方 WAV、耗时、设置及源码和模型指纹。使用 `--resume` 继续同一批次；省略 `--num-tasks` 可评估全部 278 个任务。

检查: `make test` · `make lint` · `make verify`.

来源: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice 论文](https://arxiv.org/abs/2603.13686) · [MIT 许可](LICENSE).

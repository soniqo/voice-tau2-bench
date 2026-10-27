# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) 维护此基准，衡量语音智能体能否通过口语对话和工具调用完成任务。冻结的英语任务集共有 **278 项任务**：航空 **50**、零售 **114**、电信 **114**。

[配置](docs/runtime-setup.md) · [评分](docs/evaluation.md) · [模型比较与限制](docs/model-comparison.md)

## 模型与系统

| 智能体或系统 | 当前支持 |
| --- | --- |
| Gemma 4 12B IT, INT4 | 已验证的本地 v2 配置 |
| Gemma 4 E4B IT, INT4 | 本地运行时支持；同条件 v2 比较尚待完成 |
| OpenAI GPT-Live + 明确指定的后端智能体 | 需要适配器 |
| OpenAI Realtime | 需要适配器 |
| Gemini Live | 需要适配器 |
| ElevenLabs Agents | 需要适配器 |

本地技术栈使用 [speech-swift](https://github.com/soniqo/speech-swift)：完整版 Whisper Large-v3 FP16、VoxCPM2 主权重 FP16 / AudioVAE FP32、Silero VAD，以及作为调用者、听取辅助和评判的 Gemma E4B。云端系统尚未接入，也未在此基准中测试。

## 比较指标

| 指标 | 衡量内容 |
| --- | --- |
| 任务成功率，pass^1 | 要求的最终数据库状态与沟通结果 |
| 覆盖与失败 | 已评分/计划试验数、正常结束、未评分的推理错误 |
| 延迟，p50 / p95 | 智能体 LLM 和 ASR 执行时间；TTS 首段音频时间 |
| 语音响应延迟 | 调用者说话结束至智能体首段音频；标准化汇总尚待实现 |
| 工具错误率 | 失败或被拒绝的智能体调用 / 尝试调用总数 |
| 完成时间 | 分别记录模拟通话时长与实测墙钟时间 |
| 峰值内存 | 原生进程物理内存占用，并注明测量范围 |
| API 成本 | 每次试验及每次成功试验的美元成本；包括失败尝试 |

比较时应使用相同的任务、种子、通话限制、声学条件和调用者/评判配置。比较本地 LLM 时保持语音技术栈一致。音频原生和托管智能体必须注明完整技术栈。组件耗时与端到端语音响应延迟分别统计。

## 已记录的本地诊断

| 智能体 | 已评分 / 计划 | Pass^1 | 工具错误 | LLM p50 / p95 | TTS 首段音频 p50 / p95 | API 成本 / 试验 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

航空任务 12，单次试验，种子 70，v2；最长 **600 秒模拟时间 / 1,200 秒墙钟时间**。通话在 **312.9 秒模拟时间 / 554.4 秒实测墙钟时间**后正常结束。硬件：Apple M5 Pro，48 GB，共享工作站。这是一次开发试验；同条件的多模型排名尚待完成。 [数据与测量定义](docs/results/gemma4-12b-local-v2-airline.json).

## 本地运行

Python **3.12/3.13**、Apple Silicon、**macOS 15+**。按[配置](docs/runtime-setup.md)和 [v2 运行](docs/local-v2.md)文档启动本地服务、验证调用者/评判并执行任务。

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

使用 `--num-tasks`、`--trials`、`--max-seconds` 和 `--wall-timeout` 限制运行。云端适配器和美元支出上限**尚未实现**。当前推理完全在本地运行，无需云端 API 密钥。

检查: `make test` · `make lint` · `make verify`.

来源: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).

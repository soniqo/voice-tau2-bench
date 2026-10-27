# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintains this benchmark to measure whether voice agents complete tasks through spoken conversation and tool use. The frozen English frame has **278 tasks**: airline **50**, retail **114**, telecom **114**.

[Setup](docs/runtime-setup.md) · [Scoring](docs/evaluation.md) · [Model comparison and limits](docs/model-comparison.md)

## Models and systems

| Agent or system | Current support |
| --- | --- |
| Gemma 4 12B IT, INT4 | Tested local v2 profile |
| Gemma 4 E4B IT, INT4 | Local runtime supported; matched v2 comparison pending |
| OpenAI GPT-Live + a named backend agent | Adapter required |
| OpenAI Realtime | Adapter required |
| Gemini Live | Adapter required |
| ElevenLabs Agents | Adapter required |

The local stack uses [speech-swift](https://github.com/soniqo/speech-swift): full Whisper Large-v3 FP16, VoxCPM2 FP16 main weights / FP32 AudioVAE, Silero VAD, and Gemma E4B for caller/listener/judge. Cloud systems have not been integrated or benchmarked here.

## Comparison metrics

| Metric | What it measures |
| --- | --- |
| Task success, pass^1 | Required final database state and communication |
| Coverage and failures | Scored/planned trials, normal stops, unscored inference errors |
| Latency, p50 / p95 | Agent LLM and ASR execution; TTS time to first audio |
| Voice response delay | Caller end of speech to first agent audio; standardized summary pending |
| Tool error rate | Failed or rejected agent calls / attempted calls |
| Completion time | Simulated call time and measured wall time, separately |
| Peak memory | Native process physical footprint, with measurement scope |
| API cost | USD per trial and per successful trial; include failed attempts |

Compare the same tasks, seeds, call limits, acoustics and caller/judge configuration. Keep the speech stack fixed when comparing local LLMs. Audio-native and managed agents must name their complete stack. Component timings are separate from end-to-end voice response delay.

## Recorded local diagnostic

| Agent | Scored / planned | Pass^1 | Tool errors | LLM p50 / p95 | TTS first audio p50 / p95 | API cost / trial |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Airline task 12, one trial, seed 70, v2; **600 simulated seconds / 1,200 wall seconds** maximum. The call ended normally after **312.9 simulated seconds / 554.4 measured wall seconds**. Hardware: Apple M5 Pro, 48 GB, shared workstation. This is one development trial; a matched multi-model ranking remains pending. [Data and measurement definitions](docs/results/gemma4-12b-local-v2-airline.json).

## Run locally

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Follow [setup](docs/runtime-setup.md) and [v2 execution](docs/local-v2.md) to start the local service, qualify the caller/judge, and run tasks.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Limit execution with `--num-tasks`, `--trials`, `--max-seconds` and `--wall-timeout`. Cloud adapters and a USD spending cap are **not implemented**. Current inference is local and needs no cloud API key.

Checks: `make test` · `make lint` · `make verify`.

Source: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice paper](https://arxiv.org/abs/2603.13686) · [MIT license](LICENSE).

# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintains this benchmark to measure whether voice agents complete tasks through spoken conversation and tool use. The frozen English frame has **278 tasks**: airline **50**, retail **114**, telecom **114**.

[Setup](docs/runtime-setup.md) · [Scoring](docs/evaluation.md) · [Model comparison and limits](docs/model-comparison.md)

## Models and systems

| Agent or system | Current support |
| --- | --- |
| Gemma 4 12B IT, INT4 | Tested local v2 profile |
| Gemma 4 E4B IT, INT4 | Local runtime supported; matched v2 comparison pending |
| Amazon Nova 2.5 Sonic | Native audio validated; Retail pass^1 **1/1** (one v6 development trial, 420 s cap; [record](docs/results/nova25-v6-retail-development.json)) |
| OpenAI GPT-Live + a named backend agent | [Experimental adapter tested offline; live validation pending](docs/providers.md) |
| OpenAI Realtime | [Experimental adapter tested offline; live validation pending](docs/providers.md) |
| Gemini Live | [Experimental adapter tested offline; live validation pending](docs/providers.md) |
| ElevenLabs Agents | [Experimental adapter tested offline; live validation pending](docs/providers.md) |

The local stack uses [speech-swift](https://github.com/soniqo/speech-swift): full Whisper Large-v3 FP16, VoxCPM2 FP16 main weights / FP32 AudioVAE, Silero VAD, and Gemma E4B for caller/listener/judge. Nova 2.5 Sonic has an optional [native audio runner](docs/nova.md); its caller and judge remain local.

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

Compare the same tasks, seeds, call limits, acoustics and caller/judge configuration. Keep the speech stack fixed when comparing local LLMs. Audio-native and managed agents must name their complete stack. Component timings are separate from end-to-end voice response delay.

## Recorded development results

One development trial per row. Tasks and clocks differ; a matched model ranking remains pending.

| Agent | Condition / task | Scored / planned | Pass^1 | Tool errors | Wall time |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · Airline 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · Retail 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

Airline task 12, one trial, seed 70, v2; **600 simulated seconds / 1,200 wall seconds** maximum. The call ended normally after **312.9 simulated seconds / 554.4 measured wall seconds**. Hardware: Apple M5 Pro, 48 GB, shared workstation. This is one development trial; a matched multi-model ranking remains pending. [Data and measurement definitions](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova: one trial, seed 70, control acoustics, **420 s real-time / wall cap**. Both requested items were exchanged; all **5/5** reference actions and the final database state matched. The call ended normally after **259.0 audio seconds / 296.3 wall seconds**. [Run record and limits](docs/results/nova25-v6-retail-development.json).

## Run locally

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Follow [setup](docs/runtime-setup.md) and [v2 execution](docs/local-v2.md) to start the local service, qualify the caller/judge, and run tasks.

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

This command builds the pinned native runtime, downloads model assets, calibrates the caller/judge, and runs a controlled voice smoke check. It produces no domain benchmark score. Use a fresh output directory. [Setup and task runs](docs/runtime-setup.md#one-command-local-run).

Limit execution with `--num-tasks`, `--trials`, `--max-seconds` and `--wall-timeout`. The default run stays local. Optional runners: [Nova](docs/nova.md) · [OpenAI, Gemini and ElevenLabs](docs/providers.md).

Checks: `make test` · `make lint` · `make verify`.

Source: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice paper](https://arxiv.org/abs/2603.13686) · [MIT license](LICENSE).

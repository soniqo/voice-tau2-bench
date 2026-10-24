# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintains this standalone τ-Voice benchmark to measure **local voice-agent performance** on Apple Silicon. We run the agent, simulated caller, speech models, and semantic judge through [speech-swift](https://github.com/soniqo/speech-swift).

## Documentation

- [Setup and model files](docs/runtime-setup.md)
- [Evaluation and scoring](docs/evaluation.md)
- [Implementation](docs/local-voice-extraction.md)
- [Recorded model run](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B run and working parameters](docs/validation/gemma4-12b-voice-e2e.md)

## What we measure

We measure whether a voice agent completes customer-service tasks through conversation and tool use. Task rewards check the required final database state and communication.

The frozen English `voice-v1` frame contains **278 tasks**: airline **50**, retail **114**, telecom **114**. Tasks, policies, tools, databases, and reward criteria retain their upstream pins.

## Models tested and results

| Role | Model | Precision |
| --- | --- | --- |
| Agent | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Caller, listening helpers, semantic judge | Gemma 4 E4B IT | INT4 |
| Agent and caller speech | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Speech recognition | Whisper Large-v3 (full, CoreML) | FP16 |
| Speech detection | Silero VAD (CoreML) | Native |

Development slice: the first frozen task in each domain, **3 tasks total**, **1 trial**, seed **42**, **600-second conversation deadline**, clean `control` telephony. Hardware: **Apple M5 Pro, 48 GB unified memory**. Task success measures the complete stack above.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Agent model | Airline | Retail | Telecom | Overall pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

This small deterministic slice is a custom local profile with a local caller and judge. It does not establish a full 278-task score or an upstream leaderboard result. Timeouts score zero; inference exceptions remain unscored.

The 12B slice is incomplete: Telecom returned a native inference error and remains unscored. Controlled 12B voice check: **1.0** for the correct database write and **0.0** for a wrong-write replay. [Working parameters](docs/validation/gemma4-12b-voice-e2e.md). The caller input is fixed; this check contributes no benchmark score. Domain runs use the generative caller, and the linked reports record each runtime revision.

V2 reruns remain diagnostic: one airline refusal case scored **1.0** for an unchanged database; retail timed out, and telecom exposed caller-control faults. The **18-call held-out pilot is incomplete**. These runs do not establish an overall model pass rate. [Rerun findings and parameters](docs/validation/caller-rerun.md).

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

## Run locally

Use Python **3.12/3.13** and Apple Silicon with **macOS 15+**. Start the [local model service](docs/runtime-setup.md#serve) before running tasks.

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

Each task saves its tool/transcript trajectory, reward components, caller/agent WAV files, timings, settings, and source/model fingerprints. Resume the same batch with `--resume`. Omit `--num-tasks` to evaluate all 278 tasks.

Checks: `make test` · `make lint` · `make verify`.

Source: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice paper](https://arxiv.org/abs/2603.13686) · [MIT license](LICENSE).

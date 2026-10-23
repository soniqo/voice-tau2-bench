# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Standalone **τ-Voice agent benchmark** with local inference through [speech-swift](https://github.com/soniqo/speech-swift). Evaluate spoken customer-service conversations, tool use, and task completion on Apple Silicon.

The frozen English `voice-v1` frame contains **278 tasks**: airline **50**, retail **114**, telecom **114**. Tasks, policies, tools, databases, and reward criteria retain their upstream pins.

## Local model stack

| Role | Model | Precision |
| --- | --- | --- |
| Agent, caller, listening helpers, semantic judge | Gemma 4 E4B IT | INT4 |
| Agent and caller speech | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Speech recognition | Whisper Large-v3 (full, CoreML) | FP16 |
| Speech detection | Silero VAD (CoreML) | Native |

Model downloads happen during setup; inference uses no cloud providers or API keys. See [runtime setup](docs/runtime-setup.md) for the matching speech-swift build and model staging.

## Measured results

Development slice: the first frozen task in each domain, **3 tasks total**, **1 trial**, seed **42**, **600-second conversation deadline**, clean `control` telephony. Hardware: **Apple M5 Pro, 48 GB unified memory**. Task success measures the complete stack above.

`soniqo-local-v1` · `2026-10-23`

| Agent model | Airline | Retail | Telecom | Overall pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

This small deterministic slice is a custom local profile with a local caller and judge. It does not establish a full 278-task score or an upstream leaderboard result. Timeouts score zero; inference exceptions remain unscored.

## Quick start

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

## Checks and evaluation

```bash
make test
make lint
make verify
```

See [evaluation rules](docs/evaluation.md), [runtime architecture](docs/local-voice-extraction.md), and the [recorded model run](docs/results/gemma4-e4b-local-600.json). Mock fixtures and component checks are development checks and do not contribute to the task score.

## Attribution

Derived from [Sierra Research’s τ²-Bench](https://github.com/sierra-research/tau2-bench) at `05736c27219b769d0be507574dd3135010916644`. Original code/data retain the [MIT license](LICENSE). Cite the [τ-Voice paper](https://arxiv.org/abs/2603.13686) when using the benchmark.

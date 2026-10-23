# Local model runtime

`voice-tau2-bench` uses Python 3.12/3.13 for tasks, tools, simulation, and grading. Model inference runs through a matching source build of [speech-swift](https://github.com/soniqo/speech-swift) on Apple Silicon with macOS 15 or later. The tested machine is an Apple M5 Pro with 48 GB unified memory.

The runtime needs the native inference service with Gemma tool/result history, VoxCPM2 streaming and FP16 main weights, full Whisper Large-v3, stateful Silero VAD, and cancellation. These extensions are available on `feat/local-voice-runtime` pending release; the matching build exposes `--inference-manifest`. For reproducing the published baseline, use the source commits recorded in its [JSON report](results/gemma4-e4b-local-600.json).

## Build and stage

Build the matching speech-swift checkout first:

```bash
cd ../speech-swift
make build
cd ../voice-tau2-bench
```

The release build includes compiled MLX Metal shaders. In the benchmark checkout, stage the pinned model files:

```bash
uv sync --frozen --extra dev
python3 scripts/prepare_speech_swift_models.py \
  --runtime-source ../speech-swift \
  --output data/simulations/local-runtime/models.json
```

Setup downloads model assets once and records their immutable revisions and content hashes. The VoxCPM2 source checkpoint is unquantized BF16; the runtime explicitly converts its main parameters to FP16 and keeps AudioVAE in FP32. Full Large-v3 uses the pinned 32-layer CoreML bundle. Inference then runs offline without API keys or cloud SDKs.

Rebuild and regenerate the manifest together after native source changes. Keep the generated manifest and model caches local; their machine-specific directories are recorded in each run's provenance.

## Serve

Keep this process running in its own terminal:

```bash
python3 scripts/run_speech_swift_guarded.py \
  --server ../speech-swift/.build/release/speech-server \
  --manifest data/simulations/local-runtime/models.json \
  --metrics data/simulations/local-runtime/memory.jsonl \
  --port 8787
```

Models load once. Heavy inference requests run sequentially; VAD keeps separate lightweight stream state. The launcher samples native-process physical footprint and stops its child above 24 GiB or below 2 GiB free disk. Run one benchmark process at a time.

## Run and inspect

### Gemma 4 12B agent

The `soniqo-local-gemma12b-v1` profile selects Gemma 4 12B Unified INT4 for the
agent and retains E4B INT4 for the caller, listening helpers, and semantic judge.
VoxCPM2 FP16, full Whisper Large-v3 FP16, and Silero VAD are unchanged.

Stage the additional model with the speech-swift source that includes
`Gemma4UnifiedChat`:

```bash
python3 scripts/prepare_speech_swift_models.py \
  --agent-model gemma4-12b-it \
  --runtime-source ../speech-swift \
  --output models-12b.json
```

Start the guarded service with `--manifest models-12b.json`, then run a complete
spoken development task with an observable database write:

```bash
uv run --frozen tau-voice run --domain mock --task-id create_task_1 \
  --profile configs/voice/soniqo-local-gemma12b-v1.json \
  --base-url http://127.0.0.1:8787 --seed 42 --max-seconds 600 \
  --complexity control --output data/simulations/gemma12b-smoke
```

The mock task is a functional check and contributes no score to the frozen
278-task benchmark. For domain evaluation, pass the same `--profile` to
`tau-voice batch --domain all --num-tasks 1` with the other run settings.

The opt-in pipeline test uses a fixed request spoken by VoxCPM2: create
"Important Meeting" for user_1. The agent receives only the rendered telephony
audio through the normal orchestrator, ASR, model, tool, and response path.
The test requires a successful database write, then changes the recorded tool
argument in a separate negative control and checks that grading returns zero.
Its artifacts identify the controlled caller and record the exact spoken text.
This is a functional check; the normal CLI and scored domain runs retain the
generative E4B caller.

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v1.json \
TAU_VOICE_FULL_E2E=1 \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k full_voice_task
```

### Default E4B run

From another terminal in the benchmark checkout:

```bash
uv run --frozen tau-voice batch \
  --domain all --num-tasks 1 --trials 1 --seed 42 \
  --max-seconds 600 --complexity control \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-baseline
```

`--num-tasks` limits each domain in frozen order. Omit it for all 278 tasks. Repeat trials with `--trials`; resume the same command with `--resume`. Resume verifies source, model, configuration, and saved-artifact fingerprints. Exceptions remain unscored and stop the batch.

The `control` profile uses clean telephony. `--complexity regular` samples the preserved acoustic effects and requires their audio assets. Every task saves separate caller/agent WAV files, transcript/tool trajectory, reward components, inference timings, settings, and provenance. `summary.json` reports completed trials and pass^k.

For development, use `tau-voice run --domain mock --task-id create_task_1`; mock tasks do not contribute to the benchmark score. Opt into actual-model component checks with:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
  uv run --frozen pytest -q -m local_models
```

Run component checks separately from scored tasks so they do not compete for the shared inference queue.

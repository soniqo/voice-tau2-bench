# Local model runtime

`voice-tau2-bench` uses Python 3.12/3.13 for tasks, tools, simulation, and grading. Model inference runs through a matching source build of [speech-swift](https://github.com/soniqo/speech-swift) on Apple Silicon with macOS 15 or later. The tested machine is an Apple M5 Pro with 48 GB unified memory.

The runtime needs the native inference service with Gemma tool/result history, VoxCPM2 streaming and FP16 main weights, full Whisper Large-v3, stateful Silero VAD, and cancellation. These extensions are currently prepared on `feat/local-voice-runtime`; a release without `--inference-manifest` cannot serve this profile.

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

# Local model runtime

`voice-tau2-bench` uses Python 3.12/3.13 for tasks, tools, simulation, and grading. Model inference runs through a matching source build of [speech-swift](https://github.com/soniqo/speech-swift) on Apple Silicon with macOS 15 or later. The tested machine is an Apple M5 Pro with 48 GB unified memory.

The runtime needs the native inference service with Gemma tool/result history, VoxCPM2 streaming and FP16 main weights, full Whisper Large-v3, stateful Silero VAD, and cancellation. The E4B service is on speech-swift main; the matching build exposes `--inference-manifest`. The 12B adapter is also on main; its introduction is recorded in [speech-swift PR #520](https://github.com/soniqo/speech-swift/pull/520). Use the exact source revisions recorded in the [E4B run](results/gemma4-e4b-local-600.json), [12B domain attempt](results/gemma4-12b-local-600.json), or [successful 12B voice check](validation/gemma4-12b-voice-e2e.md) when reproducing them.

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
  --agent-model gemma4-12b-it \
  --runtime-source ../speech-swift \
  --output data/simulations/local-runtime/models.json
```

Setup downloads model assets once and records their immutable revisions and content hashes. The VoxCPM2 source checkpoint is unquantized BF16; the runtime explicitly converts its main parameters to FP16 and keeps AudioVAE in FP32. Full Large-v3 uses the pinned 32-layer CoreML bundle. Inference then runs offline without API keys or cloud SDKs.

Rebuild and regenerate the manifest together after native source changes. Keep the generated manifest and model caches local; their machine-specific directories are recorded in each run's provenance.

For histories containing a user turn after tool results, use speech-swift
`35299ba6c168f5cd6e8c1862433c3d4a7e55087b` or a descendant containing the
[Gemma turn-boundary fix](https://github.com/soniqo/speech-swift/pull/539).
The native formatter must close the previous model turn before opening that
user turn. Qualify again after rebuilding; a previous runtime's qualification
does not cover the changed native source.

To diagnose a native chat failure, add
`--failure-capture-directory /absolute/path/to/private-failures` when staging.
A matching speech-swift build then writes failed request bodies and, for
completed 12B generations, the exact prompt, token IDs, raw completion, and stop
reason before parsing. The directory is created privately; an existing directory
must already deny group and other access. These opt-in records contain
conversation/tool content and remain local.

The benchmark also retains the exact sent JSON body, HTTP status/body, and error
type for failed chat calls in `failed-run.json` under the inference events.
Match the native and benchmark records by request ID. Successful request bodies
and failed audio payloads are not retained by this failure-capture path. Errors
still propagate and remain unscored; capture never retries or repairs them.

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

The current tested configuration is the 12B v2 agent with a local E4B caller,
listener and judge. Qualify this source/runtime combination, then run one task
into a fresh directory:

```bash
uv run --frozen tau-voice qualify \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-v2-qualification.json

uv run --frozen tau-voice run --domain airline --task-id 12 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 70 --complexity control \
  --max-seconds 600 --wall-timeout 1200 \
  --output data/simulations/local-v2-airline
```

See [model comparison and limits](model-comparison.md) for available systems,
metric definitions, and the [recorded local diagnostic](results/gemma4-12b-local-v2-airline.json).
The v1 examples below reproduce historical profiles and should not be mixed
with v2 results.

For the experimental simulated-clock profile, caller/judge qualification, and
the predeclared 18-call pilot, follow [local v2 execution](local-v2.md). It uses
the same model assets as the 12B profile below. The v1 commands retain their
original wall-clock behavior and historical result provenance.

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

The recorded 12B domain attempt completed airline and retail with zero rewards.
Telecom failed with a native inference error and remains unscored. Its
[report](results/gemma4-12b-local-600.json) is an incomplete development slice.
The [controlled voice check and working parameters](validation/gemma4-12b-voice-e2e.md)
are reported separately.

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
TAU_VOICE_E2E_OUTPUT=data/simulations/gemma12b-voice-check \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k full_voice_task
```

Choose a fresh output directory for each functional-check attempt.

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

See the [exact failure capture and recovery checks](validation/gemma4-12b-exact-capture.md) for the corrected tool-terminator case, a complete scored Telecom retry, remaining task failures, and the repeated positive/negative spoken pipeline check.

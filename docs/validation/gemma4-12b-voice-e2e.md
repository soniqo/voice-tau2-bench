# Successful Gemma 4 12B voice run

Gemma 4 12B completed a controlled spoken task through VoxCPM2, Whisper Large-v3, native tool calling, the environment, and database grading. The agent called `get_users`, then `create_task(title="Important Meeting", user_id="user_1")`. The live run earned **1.0**. A separate recorded trajectory with a deliberately wrong title earned **0.0** when regraded.

This is a functional check with a fixed spoken caller request. It does not exercise the caller LLM and contributes no score to the frozen 278-task frame. Normal CLI runs and domain scores use the generative E4B caller.

The tested stack used Apple M5 Pro with 48 GB unified memory, the `soniqo-local-gemma12b-v1` profile, Gemma 4 12B Unified INT4, VoxCPM2 FP16 main weights with FP32 AudioVAE, full Whisper Large-v3 CoreML FP16, and Silero VAD. Agent generation used temperature 0, max_tokens 1024, repetition penalty 1, and the checkpoint's thinking-disabled prefix. Simulation used seed 42, clean `control` telephony, 50 ms ticks, 8 kHz mu-law audio, and a 600-second deadline.

The controlled request was: “Please create a task called Important Meeting for user underscore one.” Whisper recognized “Please create a task called Important Meeting for User_1.” The run recorded 3 recognized instances of the request. The full test took 59.62 seconds including caller synthesis and validation; the simulation itself took 49.85 seconds.

Use speech-swift commit `ce28f6fa63f8dafcf9f1bb3d7d70665ec9d31f79` and benchmark commit `a223955bda101da39fb45b3e4628bcccc628e410`. The pinned 12B model is `mlx-community/gemma-4-12B-it-4bit` at revision `73bcf09092aa277861d5a191b989b666f7f32e8f`. All model file hashes, source hashes, settings, and run artifact hashes are in [the validation report](gemma4-12b-voice-e2e.json).

Build speech-swift with its compiled Metal shaders, then stage the pinned assets from the benchmark checkout:

```bash
python3 scripts/prepare_speech_swift_models.py \
  --agent-model gemma4-12b-it \
  --runtime-source ../speech-swift \
  --output models-12b.json

python3 scripts/run_speech_swift_guarded.py \
  --server ../speech-swift/.build/release/speech-server \
  --manifest models-12b.json --metrics memory-12b.jsonl --port 8787
```

In a second terminal, run the functional check. Use a fresh output directory for each attempt:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v1.json \
TAU_VOICE_FULL_E2E=1 \
TAU_VOICE_E2E_OUTPUT=data/simulations/gemma12b-voice-check \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k full_voice_task
```

Run the normal generative-caller domain slice separately:

```bash
uv run --frozen tau-voice batch \
  --profile configs/voice/soniqo-local-gemma12b-v1.json \
  --domain all --num-tasks 1 --trials 1 --seed 42 \
  --max-seconds 600 --complexity control \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/gemma12b-domain-slice
```

The earlier mock trial with the generative caller, before the prompt and tool-name fixes, created “Meeting Task” and received 0.0. Its artifacts are retained as a failed development attempt. The controlled success above uses the fixed implementation and a different caller mode.

Validation also passed 12 focused native unit tests, a real 12B tool/history/cancellation test, and eight native endpoint checks. An independent Whisper readback of the final 20 seconds of the saved agent WAV recognized the confirmation that Important Meeting was created for User_1.

The [normal domain attempt](../results/gemma4-12b-local-600.json) used its own recorded runtime revision: airline 0/1, retail 0/1, and Telecom unscored after native HTTP 500. A reconstructed diagnostic request on the final binary logged a malformed tool-argument literal. This remains a limitation; the controlled success does not establish a domain benchmark score.

See the [domain failure analysis](gemma4-12b-domain-analysis.md) for the timeout
counterfactual, identifier recovery failure, protocol uncertainty, and latency
measurements.

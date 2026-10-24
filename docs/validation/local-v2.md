# Local v2 validation

The real 12B spoken pipeline passed: the requested database write scored **1.0**,
and a separate replay with the successful write's title changed scored **0.0**.
The local customer policy and semantic judge passed **19/19** qualification checks.
All **467 offline tests**, lint, 64 frozen-input checks, and both upstream replays
passed. The [machine-readable record](local-v2.json) records source and artifact
hashes, model roles, settings, and the scope of each check.

## Spoken agent regression

The functional test rendered a fixed request through VoxCPM2, sent only its
telephony audio to the agent, used full Whisper Large-v3, and ran Gemma 4 12B with
real tools and streamed speech. It took 171.64 seconds including reference speech
generation and assertions; the simulation recorded 116.2 audio seconds across
2,324 ticks. This is a pipeline check, not a latency benchmark or a domain score.

ASR first returned `User_1`, and the first creation attempt failed. The agent then used `get_users`,
received the canonical `user_1`, and created `Important Meeting` successfully.
The canonical identifier came from a tool available to the agent. No reference
action or gold transcript entered its input.

The negative control changes the successfully executed `create_task`, rather than
an earlier failed attempt. The database grader rejects that wrong write. The
final model test passed with the following parameters:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v2.json \
TAU_VOICE_FULL_E2E=1 \
TAU_VOICE_E2E_OUTPUT=data/simulations/local-v2-spoken-check \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k full_voice_task
```

Use a fresh output directory and the staged native runtime described in
[runtime setup](../runtime-setup.md). This test explicitly replaces the generative
customer with fixed spoken input and marks that substitution in its artifacts.
Customer qualification is tested separately; the controlled test cannot produce
a scored domain benchmark result.

Three native chat requests were cancelled by the agent during the call. Request-ID
matching confirms those cancellation intents. There were no unexplained inference
failures or non-cancellation native chat failures.

## Customer qualification and domain checks

The final preflight exercises the same local caller guidelines, role-labelled
history, terminal assessment and available-tool handling used by v2 runs. It
allows one declared read-only notification check before checking the customer's
answer; an invented agent tool, a repeated read loop, or an incorrect final answer
fails. The completion/refusal label is diagnostic: qualification checks whether
the customer continues or stops at the correct point.

Early development probes exposed an incomplete agent tool history, an invalid
customer tool, repeated greetings, and a five-second wait that repeatedly
interrupted longer local processing. These probes are not model scores. V2 now
uses an explicit, shared 15-second caller wait and keeps the agent's inference
latency visible. Its clock and caller condition differ from historical v1 runs.

The airline refusal probe was stopped after 184.7 simulated seconds and remains
unscored. Its captured trace explains a remaining latency limitation: five agent
chat calls took 14.98–23.76 seconds each. ASR + chat + first TTS audio required at
least 19.44–28.73 seconds, excluding VAD and any extra queue delay. The customer
repeated after the configured 15-second silence, superseding replies before they
became audible. Extending the caller wait from five seconds to fifteen did not
resolve this case. These are development measurements, not a general speed claim.

The harness now exposes that local response latency; it does not hide it behind
customer computation or force a successful stop. Native prompt processing and
response latency need improvement before this stack can support the full pilot
reliably under the recorded interaction condition.

The predeclared 18-call held-out pilot has not run. Published historical model scores remain
unchanged; neither this functional check nor customer calibration establishes a
new domain pass rate. See [local v2 execution](../local-v2.md) for pilot commands
and the precise profile differences.

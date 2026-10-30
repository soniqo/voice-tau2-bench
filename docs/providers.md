# Optional voice-provider adapters

`tau-voice-provider` connects OpenAI Realtime, GPT-Live with a named Responses
backend, Gemini Live, or a configured ElevenLabs Agent to the existing voice
runner. These experimental adapters have **offline integration tests**. Gemini
and ElevenLabs also have recorded live audio/tool controls; matched domain
scores are pending. The recorded Nova score
uses the separate [validated Nova runner](nova.md).

The default `tau-voice` command stays local. Install the optional transport only
when using these adapters:

```bash
uv sync --frozen --extra dev --extra cloud-voice
```

## Shared setup and a first task

Start the loopback service using [runtime setup](runtime-setup.md). The provider
supplies the agent's conversational audio and reasoning. Caller, listener,
judge and caller TTS remain local; the fixed opening greeting uses VoxCPM2.
The runner checks these four native roles rather than requiring local agent
ASR/LLM/TTS. All adapters use real-time v6 with the separately recorded
`prepared-opener-v1` setup condition. The runner synthesizes the exact seeded
greeting once before starting the conversation watchdog or connecting to the
provider. Playback of that greeting still counts toward the conversation.
Its audio/text hashes, seed and preparation duration are saved in `setup.json`.

Qualify the caller/judge against the **current working source** and native
runtime. Source changes invalidate older qualifications. This operation uses
local inference only:

```bash
uv run --frozen tau-voice-provider --qualify-only \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/provider-qualification.json
```

After setting the provider's credential in your shell, this explicit opt-in
starts a paid Realtime conversation. Never put a credential value in a command
argument, repository file or result artifact:

```bash
uv run --frozen --extra cloud-voice tau-voice-provider \
  --allow-cloud --provider openai-realtime \
  --model gpt-realtime-2.1-mini --voice marin \
  --domain mock --task-id create_task_1 --seed 70 \
  --qualification data/simulations/provider-qualification.json \
  --max-seconds 90 --budget-usd 1 --stop-usd 0.25 \
  --budget-ledger data/simulations/provider-budget.json \
  --output data/simulations/realtime-control
```

The expected mock task creates an ordinary record through `create_task`. Its
frozen grader checks the actual write. This is an integration check, not a
domain pass rate. For development coverage, replace the mock selection with
`--domain all --task-plan configs/pilots/state-change-development-3.json` and
choose fresh output. A verified ElevenLabs preset covers one domain at a time.
`--num-tasks`, `--trials`, and `--resume` work through the existing sequential
batch runner. A single `--task-id` is a fresh, one-domain, one-trial run.

## OpenAI Realtime

Use `--provider openai-realtime --model gpt-realtime-2.1-mini --voice marin`
or explicitly select `gpt-realtime-2.1`. The default credential variable is
`OPENAI_API_KEY`; `--api-key-env` changes its **name**, not its value.

The GA WebSocket session uses PCM16 at 24 kHz, server VAD and ordinary function
tools, with a 1,024-token output limit. The adapter waits for `response.done`
before releasing complete speech and tool calls. Input speech clears queued
playback; conversation truncation uses the cursor actually delivered to the
caller, including completed responses that have not finished playing. Cancelled
responses still contribute their reported usage. See the official
[Realtime conversation guide](https://developers.openai.com/api/docs/guides/realtime-conversations).

## GPT-Live with Responses

Use `--provider openai-live --model gpt-live-1 --backend-model gpt-6-luna
--voice marin`, with `OPENAI_API_KEY`. The backend is part of the measured
system. This adapter uses Responses delegation; a local Gemma backend is not
implemented by this adapter.

The primary Live connection carries input audio and control messages. An
authenticated sideband supplies timestamped reflected PCM24k audio, captions,
nested Responses events and usage. Function calls come from
`response.output_item.done`, and results return with their original call IDs.
Voice duration and backend tokens are recorded separately. Closure waits for
`session.closed` and the delegated response lifecycle. See
[Live delegation](https://developers.openai.com/api/docs/guides/live-delegation?delegation-mode=responses)
and [server controls](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live).

**Live endpointing is experimental.** The API has no output-turn-complete
event. This adapter declares an RMS threshold of 180, 800 ms of silence and
500 ms of caption grace, using timestamped reflected audio. This identifies
speech boundaries; a pause does not prove a business answer is complete.
Missing reflected speech, absent transcripts or late captions are errors.
Primary audio is not counted again as sideband playback. Live results must
identify this condition and need a live timing audit before a matched ranking.

## Gemini Live

Use `--provider gemini-live --model gemini-3.8-live --voice Kore`, with
`GEMINI_API_KEY` (or an explicit `--api-key-env` name). Input PCM is 16 kHz;
output is 24 kHz. The session declares audio output, independent input/output
transcription, the domain function schemas and a 1,024-token output limit.

Audio and transcript can arrive independently. The adapter releases a spoken
turn at `turnComplete`, preserves tool IDs and handles tool cancellation.
`generationComplete` alone does not end the spoken turn. A `goAway` event ends
the attempt as an error; there is no automatic reconnection or hidden retry.
See the [Live API reference](https://ai.google.dev/api/live).

For Google Cloud/Vertex access, also set `--vertex-project YOUR_PROJECT
--vertex-location us-central1`. The credential variable then defaults to
`GOOGLE_OAUTH_ACCESS_TOKEN`, carrying a short-lived bearer token rather than
an AI Studio key. Obtain it locally after Google Cloud sign-in:

```bash
gcloud auth login --update-adc
export GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth application-default print-access-token)"
```

The adapter uses the regional Vertex WebSocket and fully qualified model
resource. The project is hashed in result settings. A non-global region with
verified pricing is required. Vertex Gemini 3.8 receives the already delivered
greeting through `contextUpdate`; it is added to public system context without
injecting a synthetic caller message. See Google's
[Live session guidance](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api/start-manage-session).

## ElevenLabs Agents

This adapter requires an existing agent with the exact domain policy and
**client** tools. It does not create or update remote agents. Export the expected
configuration offline, choosing the actual reasoning model and voice ID:

```bash
uv run --frozen tau-voice-provider \
  --export-eleven-config data/simulations/eleven-mock-config.json \
  --domain mock --model gpt-4.1 --voice YOUR_VOICE_ID \
  --max-seconds 90 --session-seconds 105
```

Configure the agent in ElevenLabs using that JSON. Disable the provider's first
greeting, knowledge base, built-in/server tools, language overrides and
multi-agent workflows. Enable all exported client events, including
`agent_response_complete`. The exported audio formats are PCM16k input and
PCM24k output. The adapter validates schemas, descriptions, model, voice,
policy, event selection and maximum duration before requesting a signed socket.

With `ELEVENLABS_API_KEY` set, this read-only inspection returns the resolved
configuration fingerprint without starting inference:

```bash
uv run --frozen tau-voice-provider \
  --inspect-eleven-agent --agent-id YOUR_AGENT_ID
```

Use the returned fingerprint for `--preset-sha256`. It covers the conversation
configuration, resolved tool definitions, platform settings and workflow. Any
change requires inspection again. For the shared task command, select
`--provider elevenlabs --model gpt-4.1 --voice YOUR_VOICE_ID`, and add:

```text
--agent-id YOUR_AGENT_ID --preset-sha256 YOUR_VERIFIED_SHA256
--usd-per-credit YOUR_PLAN_CONVERSION --max-usd-per-minute YOUR_MAXIMUM_RATE
```

Those numeric rates must reflect your plan and configured system. The declared
duration at the maximum rate must fit inside `--stop-usd`. Response audio and
text are buffered until `agent_response_complete`; tools execute through the
benchmark orchestrator, with matching IDs. See ElevenLabs'
[WebSocket guide](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets)
and [client events](https://elevenlabs.io/docs/eleven-agents/customization/events/client-events).

The verified preset requires response-bearing client tools. Tool-call events
may omit `expects_response`; an explicit `false` remains unsupported. Audio
whose event ID is at or before the last interruption is discarded, matching
the provider SDK's playback behavior.
The API's neutral schema defaults and inert start node are accepted. Nonempty
argument overrides, active workflows, knowledge bases, MCP and server tools
remain unsupported. Resolved tools must agree with inline definitions.

## Limits, results and accounting

`--max-seconds` bounds the conversation, with one connection attempt per trial.
`--session-seconds` sets the separate provider window. It defaults to the call
limit plus 15 seconds and must leave at least that much room for closure. The
provider window is capped at 420 seconds, so this CLI accepts call limits up to
405 seconds. ElevenLabs also requires a window of at least 60 seconds. For
example, a short control can use `--max-seconds 45 --session-seconds 60`;
the 90-second example above uses 105 seconds. Export and run with the same
limits so the ElevenLabs preset matches. Shorter or mismatched windows fail
before provider inference.
Limits and qualification are checked before inference. Transport, provider,
invalid-caller and shutdown failures propagate and remain unscored. An ordinary
task failure or timeout receives the original frozen reward. The agent sees
actual caller audio and public policy/tools only; private goals, gold speech
and reference actions are never sent to it.

Adapters buffer complete provider audio/transcripts before telephony playback,
which adds latency. The declared Live noise gate is a separate endpointing
condition. Output text is aligned proportionally to **delivered** audio, not
word timestamps. Reports must distinguish this adapter delay from model latency.
Tools already issued remain atomic when speech is interrupted.

The persistent ledger reserves `--stop-usd` plus 0.10 before a connection.
Known usage can trigger a stop; delayed usage may overshoot it. This is not a
provider-enforced spending ceiling. Active ledger ownership prevents concurrent
use. `--resume` retains reservations from dead clients, and unknown final usage
keeps the reservation. Keep the same ledger between resumed attempts.

Realtime records response token splits. Live records cumulative session seconds
and each backend response once. Gemini retains every response usage observation
(including re-billed context), accepts Vertex's candidate-token counter names,
and preserves partial tool-response meters without inventing zeros. It has no
authoritative session-final meter;
its final USD total stays unknown and its reservation remains. ElevenLabs
records final conversation credits and raw charging metadata. Credits can
exclude voice surcharges and are not a final USD invoice; USD stays unknown and
its reservation remains. Quality can be graded when the conversation and tools
settle even if final billing is unknown. Unknown billing is never reported as
zero. Price revisions and raw usage remain in machine-readable run artifacts.

`simulation.json`, `agent-events.json`, audio, provenance and batch records use
the existing artifact format. Credentials and signed connection URLs are not
written to those files. Exact provider/model/backend/preset and endpointing
settings form part of resume provenance.

`wall_seconds` continues to include greeting preparation, the call and grading.
`setup_seconds` reports pre-call greeting preparation; `call_wall_seconds`
reports the orchestrator run, including connection and shutdown work. Only
greeting synthesis moves outside the conversation watchdog. A preparation
failure is recorded as an unscored setup error, with no provider connection;
its unused reservation is released. These timings do not change the frozen
reward or retroactively change earlier trials.

## Validation

Offline wire fixtures run all four adapters through the production task runner,
real mock-domain tools and frozen evaluator. The correct write scores 1;
replaying a deliberately wrong write scores 0. They cover protocol integration,
not model quality or real account access. Offline tests also cover audio
conversion, interruptions, tool IDs, final usage, malformed meters, preflight
checks and shutdown failures. Live Gemini and ElevenLabs pilots are described
in the [validation record](results/gemini-eleven-live-controls.json). The
OpenAI adapters still have offline validation only.

The wire controls also simulate a 10-second local greeting delay and check that
the conversation still finishes under its own limit. To use real local VoxCPM2
for the opening with fixture provider events, start the loopback service and run:

```bash
TAU_SETUP_NATIVE_E2E=1 SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
  uv run --frozen --extra cloud-voice pytest -q -m local_models \
  tests/test_voice_local/test_provider_pipeline.py::test_real_local_opener_prepares_before_offline_provider_control
```

This checks native greeting preparation, real tools and the frozen grader;
it makes no cloud inference calls and does not measure a provider model.
The 2026-10-30 local check prepared a four-second greeting in 10.08 seconds,
then completed the conversation in 4.62 seconds under its separate ten-second
limit. The correct write scored 1 and the deliberately wrong write scored 0.

`tests/test_voice_local/test_provider_e2e.py` is an opt-in real-provider test.
It requires `TAU_PROVIDER_E2E=1`, `SPEECH_SWIFT_INFERENCE_URL`, a current
`TAU_PROVIDER_QUALIFICATION` file and explicit `TAU_PROVIDER`,
`TAU_PROVIDER_MODEL`, `TAU_PROVIDER_VOICE`, `TAU_PROVIDER_KEY_ENV` variables.
Live additionally needs `TAU_PROVIDER_BACKEND_MODEL`. ElevenLabs needs the
`TAU_PROVIDER_AGENT_ID`, `TAU_PROVIDER_PRESET_SHA256`,
`TAU_PROVIDER_USD_PER_CREDIT` and `TAU_PROVIDER_MAX_USD_PER_MINUTE` variables.
Run one provider test at a time with `pytest -m local_models
tests/test_voice_local/test_provider_e2e.py`; it starts a paid 90-second trial.

For an isolated mock-task protocol control, set `TAU_PROVIDER_RECORDED_CONTROL=1`
and `TAU_PROVIDER_REQUEST_WAV` to an actual spoken request. This explicit test
mode uses the production audio/tool runner and frozen state grader with a
recorded caller; it does not claim to qualify a caller model or produce a
domain score. Its provenance marks `functional_check_only` and
`standard_leaderboard: false`. `TAU_PROVIDER_OPENER_WAV` can reuse the same
previously synthesized VoxCPM2 greeting, recording its hash and cache mode.

`TAU_PROVIDER_MAX_SECONDS` controls the trial clock;
`TAU_PROVIDER_SESSION_SECONDS` can provide a separately recorded session window
for orderly closure. It uses the same call-plus-15-second default and bounds as
the CLI; ElevenLabs' window must be at least 60 seconds. Set
`TAU_PROVIDER_BUDGET_LEDGER`, `TAU_PROVIDER_BUDGET_USD` and
`TAU_PROVIDER_STOP_USD` explicitly when running more than one paid control.

The first live pilots on 2026-10-30 found and fixed ElevenLabs schema/default
serialization incompatibilities and Vertex usage-counter aliases. Gemini then
completed the recorded call normally but scored 0 because it created
`important meeting` instead of the requested `Important Meeting`. ElevenLabs
made the exact correct write in both initial calls, which timed out under the
short clock. A separate frozen state check confirms those writes and rejects a
deliberately wrong title; it does not replace their original timeout rewards.
The live VoxCPM2 opening took 8–10 seconds before its first audio frame. These
historical calls used inline greeting synthesis. Current calls record
`prepared-opener-v1` and prepare that audio before starting the watchdog;
their setup condition must be distinguished in comparisons. The earlier
timeout rewards remain unchanged. These controls are diagnostics and do not
establish a matched domain ranking.

The next two live controls on 2026-10-30 used `prepared-opener-v1`, fresh local
VoxCPM2 synthesis, the same recorded request, a 45-second conversation limit
and a 60-second provider window. Both ended with a normal caller stop and
created the exact requested `Important Meeting` title. Their frozen trial
rewards are 1; deliberately wrong writes score 0. The
[validation record](results/gemini-eleven-live-controls.json) retains all six
attempts, including the earlier failures.

| Provider | Mock-task reward | Tool errors / attempts | Greeting preparation | Total wall time |
| --- | --- | --- | --- | --- |
| ElevenLabs, GPT-4.1 / Sarah | 1 | 0/1 | 7.88 s | 55.16 s |
| Gemini Live, Gemini 3.8 / Kore | 1 | 0/2 | 6.97 s | 39.54 s |

Total wall time includes preparation and shutdown. ElevenLabs' orchestrator
duration was 47.26 seconds including shutdown, with a normal caller stop.
Gemini's orchestrator duration was 32.55 seconds. These are one mock-task
control per provider with a recorded caller. Caller-model qualification and
airline, retail and telecom scores remain separate work.

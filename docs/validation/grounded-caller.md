# Customer device controller validation

The telecom caller now derives phone-status answers from actual customer tool
observations. This addresses the [longer-call failure](caller-rerun.md), where
the caller answered a VPN question without checking its device. It is an
explicit experimental caller condition, enabled by `device_grounding: true` in
`configs/voice/soniqo-local-gemma12b-v2.json`.

The evaluated agent remains Gemma 4 12B IT INT4. The local conversation model and
semantic judge remain Gemma 4 E4B IT INT4; speech remains VoxCPM2 with FP16 main
parameters and FP32 AudioVAE, full Whisper Large-v3 CoreML FP16, and Silero VAD.
The preserved native binary, checkpoint pins, frozen tasks, policies, tools and
reward criteria are unchanged. Caller changes require new source-matched
qualification and a separate result cohort.

## Verified checks

- **594 offline tests** pass, including real fixture-device reads and mutations,
  opposite initial/desired states, no-op requests, stale observations, compound
  instructions, app permission arguments, APN reset timing and persona concerns.
- All **64 frozen inputs / 278 tasks** verify. Both upstream replay rewards match,
  and lint passes.
- Fresh native caller/judge qualification passed **49/49** checks. It includes the
  longer VPN regression, three VPN states, spoken-result checks, read-before-write
  and no-op cases, stale observations after a write, unfinished speech turns and
  affected-line questions.
- Three actual voice checks passed for VPN disabled, enabled-but-disconnected,
  and connected. Each executes the real customer read and sends the derived
  answer through VoxCPM2, 8 kHz telephone conversion and full Whisper Large-v3.
  No chat model supplies the device fact in these checks. Each clip also passes
  the actual Silero VAD threshold used by the agent.
- Two further actual speech checks preserve different caller-known phone numbers
  exactly through VoxCPM2, telephone conversion and Whisper. The controller uses
  the caller's supplied known information, without calling a model or device tool.

The first voice attempt transcribed the raw uppercase `OFF` as `OF`. The spoken
template now says “My VPN is disabled.” The audit retains the original tool
result. The connected/disconnected distinction is preserved; ASR output is never
replaced or corrected with reference text. These are functional checks, not
domain model scores.

The complete controlled spoken agent pipeline also passes: the real database
write scores **1.0** and the wrong-write replay scores **0.0**. It ends with
`user_stop` after 45.65 simulated seconds / 50.58 wall seconds, with two
intentional cancellations and no unexplained inference failures. This fixed
caller functional check contributes no domain score.

A fresh generative spoken telecom diagnostic is required before the held-out
pilot. The pilot retains its nine declared tasks, two trials and seeds 70/71.
Current results and limitations are recorded in [the validation summary](grounded-caller.json).

## Interruption scheduling regression

The first generative diagnostic timed out at 352.8 simulated seconds /
1,327.15 wall seconds with a raw timeout reward of 0. The runner had a scheduling
defect: physically completed replies that had been canceled by a barge-in still
blocked new work until their old simulated deadlines. Recorded gaps included
81.34, 89.94 and 113.19 seconds of completed stream work. No native inference
failure occurred. This timeout is not a clean measure of agent capability.

Completed canceled replies now release the worker after preserved ASR results
reach their own deadlines. Running work still must finish physically, and
uninterrupted replies retain measured latency. Two regression tests failed
before this fix; they now pass, alongside controls for ordinary timing and
hearing lifecycle. Actual execution remains in the accounting records. Response
length and shutdown queue state are also recorded for diagnosing future stalls.

## Caller speech boundaries and contact questions

The next diagnostic timed out at 315.65 simulated seconds / 1,240.53 wall
seconds. The agent completed two account reads, but no customer device actions
ran. No native inference failure occurred. The caller repeatedly interrupted
question prefixes and later repeated its complaint instead of answering which
phone number was affected. This is another caller defect, not a clean agent score.

Speech-frame merging and role conversion now preserve the last frame's utterance
boundary. The grounded caller waits on incomplete phrases, including “which
phone…”, and can still answer a complete question while speech continues. The
configured silence check-in remains available. Singular/plural phone-number and
affected-line questions use only the caller's known information; missing numbers
are explicitly unknown. Five captured regression tests failed before this fix.

## Execution and evidence

The controller interprets a bounded set of English device requests. It never
reads the device database, expected actions, reward state or agent-private tool
results. It returns calls from the declared customer interface for the ordinary
orchestrator to execute. An observation stores the actual result and source
tool-call ID. Customer claims and service-agent statements cannot supply facts.

For “turn airplane mode off,” an unknown state first requires a network-status
read. An observed ON state permits the toggle; an observed OFF state produces
an unchanged-state report. Every customer write invalidates earlier observations.
Results of rejected or failed calls are not accepted as status evidence.

Requests involving multiple steps retain their order. Vague follow-ups repeat the
previous actual speed/MMS check when known, or request clarification. Checking VPN after an
airplane-mode change performs a new read. Parameters for app permissions, APN
names and network modes come from the heard instruction. Unsupported or
ambiguous instructions get clarification. An agent saying it will perform its
own lookup does not trigger an unrequested customer action.

Explicit photo-safety concerns pause rebooting until the agent reassures the
customer. The language model continues to handle conversation, caller goals and
business choices. The controller does not decide that the evaluated agent has
succeeded, repair its identifiers or change reward rules. See
[local v2 execution](../local-v2.md) for timing and completion policy.

## Reproduction

Start the preserved local model service, then use a fresh output directory:

```bash
uv run --frozen tau-voice qualify \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/device-qualification.json

SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v2.json \
TAU_VOICE_DEVICE_E2E_OUTPUT=data/simulations/device-voice-checks \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py -k grounded_vpn

uv run --frozen tau-voice run --domain telecom \
  --task-id '[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]' \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/device-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 42 \
  --max-seconds 600 --wall-timeout 1200 --complexity control \
  --output data/simulations/device-telecom-diagnostic
```

Run model operations one at a time. The diagnostic keeps the 15-second customer
wait, 600-simulated-second cap and 1,200-wall-second watchdog. The held-out pilot
keeps 2,400 simulated seconds and 7,200 wall seconds per call. Preserve failed
attempts; customer/native failures remain unscored.

# Caller acceptance and spoken speed results

The local v2 caller now checks an explicitly stated speed-test acceptance condition
against actual customer tool evidence. It also treats “try to connect” as a service
test and speaks the measured quality outside parentheses.

This is a follow-up to the [grounded caller validation](grounded-caller.md).
The model stack, frozen 64 inputs / 278 tasks, domain policies, tools and rewards
are preserved. Changed caller sources require new qualification and separate
result cohorts.

## What the spoken run exposed

The previous development call ended with `user_stop` after 504 simulated seconds
and scored 0. The agent had turned Airplane Mode off. A subsequent connection
attempt produced an enabled-setting answer, and the caller accepted closing
remarks without a successful speed test. The caller's terminal assessment also
confused excellent radio signal with excellent internet speed. The frozen grader
correctly rejected the final state. This caller failure excludes that run from
accepted model-performance results.

A separate positive voice check exposed an acoustic problem: the raw result
`275.00 Mbps (Excellent)` reached Whisper as a rate without its quality label.
The current spoken form says “The speed test reports Excellent internet speed,
at 275.00 megabits per second.” The original result remains the tool evidence.
Other measured qualities retain their actual labels; failed tests retain their
failure text.

## Behavior and scope

The guard recognizes the caller's explicit instruction that the issue is resolved
only when a speed test returns Excellent. An unsupported completion proposal stays
CONTINUE. Closing remarks trigger a read-only speed test while that condition is
unverified. A successful result is reported before ordinary terminal assessment
resumes.

Evidence must come from a successful, declared `run_speed_test` call. Radio
signal, an enabled setting, an error result or an unsupported claim cannot satisfy
that condition. Device writes invalidate prior observations. Verification leaves
repair choices with the evaluated agent; it only observes the requested service.
Callers without this explicit acceptance wording retain their existing behavior.
The adapter remains a finite English caller condition.

## Verified checks

- **651 offline tests**, lint, **64 frozen inputs / 278 tasks**, and both upstream
  replay rewards pass.
- Fresh source-matched live caller/judge qualification passed **71/71** with no
  recorded errors, including missing versus verified speed evidence and the
  connection-attempt wording.
- **10 native checks** passed: three VPN states, two phone numbers, failed and
  Excellent speed results through VoxCPM2 → telephony → Silero → full Whisper;
  two changed-result history checks; and a controlled spoken database task.
- The controlled spoken task scores **1.0** with matching database state. Its
  wrong-write replay scores **0.0**. Functional controls contribute no domain
  model score.

The [machine-readable validation record](caller-goals.json) contains source,
qualification and artifact hashes, exact speed-test voice results and the prior
rejected diagnostic.

## Run conditions and pending scores

The stack remains Gemma 4 12B IT INT4 for the agent, E4B IT INT4 for caller/helpers,
VoxCPM2 FP16 main parameters with FP32 AudioVAE, full Whisper Large-v3 CoreML FP16,
and Silero VAD. Runs use the published 2,400-second simulated / 7,200-second wall
budget on an Apple M5 Pro with 48 GiB unified memory and a 24 GiB native guard.

These are shared-workstation measurements. Resident lightweight capture stayed
open, and a local watchdog was configured to yield to active heavy work.
Recorded timings are not an uncontended latency claim.

A new-source spoken audit precedes the unchanged nine-task, two-trial pilot
(seeds 70/71). No new accepted aggregate model score is available yet.

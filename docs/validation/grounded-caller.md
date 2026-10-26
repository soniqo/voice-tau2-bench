# Grounded caller validation

This record covers the initial grounded caller revision. See the
[acceptance and spoken-speed follow-up](caller-goals.md) for current verification
and the later caller-completion defect.

The experimental local v2 caller derives recognized phone-status answers from
actual customer tool observations. It also preserves heard speech boundaries,
distinguishes agent offers from customer actions, and keeps reported results
from replacing answers to new questions. These changes address caller/runtime
faults observed in longer spoken calls.

The stack remains Gemma 4 12B IT INT4 for the agent, Gemma 4 E4B IT INT4 for
conversation/control/judging, VoxCPM2 with FP16 main parameters and FP32 AudioVAE,
full Whisper Large-v3 CoreML FP16, and Silero VAD. The native binary/model pins,
64 frozen inputs, 278 tasks, policies, tools and reward rules are preserved.
This is an explicit experimental caller condition, separate from historical v1
results and the standard leaderboard. Changed sources, suites or models require
fresh qualification and a separate result cohort.

## Verified checks

- **635 offline tests**, lint, all **64 frozen inputs / 278 tasks**, and both
  upstream replay rewards pass.
- Source-matched live caller/judge qualification passed **68/68** with no errors.
  Cases cover real phone reads/writes, opposite states, no-op toggles, stale
  observations, missing parameters, persona concerns, complete/incomplete speech,
  agent offers, and service-working/data-speed versus enabled-setting questions.
- **Eight actual-model checks** passed: three VPN states and two exact phone
  numbers through VoxCPM2, 8 kHz telephone conversion, Silero VAD and full Whisper;
  two counterfactual tool-result checks, one per Gemma model; and the controlled
  spoken database pipeline. The short result-history checks verify basic result
  transport, not long-context planning quality.
- The controlled spoken pipeline performs a real write and scores **1.0** with
  matching database state. A wrong-write replay scores **0.0**. It ends with
  `user_stop` at **39.9 simulated / 44.80 wall seconds**, with no cancellations
  or unexplained inference failures. This functional check contributes no domain
  model score.

These are functional checks. An external speech process was observed during the
subsequent diagnostic, so exclusive model execution was not verified. The recorded
timings are not an uncontended latency measurement.

The first raw `OFF` voice attempt was transcribed as `OF`. The grounded spoken
form now says “My VPN is disabled,” retaining the original tool result as evidence.
Enabled-but-disconnected stays distinct from a connected VPN. ASR output is never
replaced with reference text.

## Evidence and behavior

The controller reads heard requests and actual customer tool messages. It cannot
read the device database, reference actions, grading state or private agent tools.
It returns declared calls for the ordinary orchestrator to execute. Unknown toggle
states require a read; already-correct states are reported without toggling. Writes
invalidate old observations. Failed/rejected results cannot establish facts.

Requests preserve recognized step order and require heard parameters. Explicit
photo concerns pause rebooting until reassurance. Functional internet/MMS questions
run the corresponding service test. Requests to report restored or working
connections also test the service, including after a requested setting change.
Known numbers use caller-known information; location and original symptoms stay
conversational. Recognized requests with ambiguous or missing parameters receive
clarification; other conversation remains model-driven under the finite English
adapter.

Current 2G/3G/4G/5G questions read the actual connection rather than the configured
network preference. A broad phone hint cannot override that specific read.

Device dispatch, known-number answers, follow-ups and tool availability use the
same action owner. A customer reporting request such as “let me know” stays
distinct from an offer to perform the agent’s own work. An agent's own offer
restricts the caller to SPEAK; the local model still chooses its answer from its
scenario/persona. Only an unspoken tool
result has priority over a new question. The latest heard agent turn is presented
directly before the labelled history.

The v2 turn guard applies across airline, retail and telecom, including gaps in
unfinished speech. Completed questions can still permit barge-in, and the
configured silence check-in remains available. Silent agent tools do not reset
speech timing. Cancelled replies release completed work after preserved hearing
reaches its own deadline; running/uninterrupted work retain measured latency.

Qualification fails on reply-generation/evaluation exceptions and rejects
error-bearing reports. The saved passing reports were also audited; none contained
recorded errors. A failing 59/60 qualification correctly exposed downstream tool
selection overriding an agent-offer deferral. The structural SPEAK restriction
and current 68-case report address that failure. A separate 65/66 failure caught
a VPN question whose explanatory mention of connection speed overrode its topic;
the scoped connectivity rule and an offline regression cover that case.

## Development runs and pilot

Development calls exposed caller/runtime faults in cancelled-work scheduling,
speech timing, device intent and action ownership. Their raw rewards and separate
source identities remain in the [validation record](grounded-caller.json) and
[earlier caller audit](caller-rerun.md); they are excluded from accepted agent
performance results.

The generative telecom audit uses the published v2 budget: 2,400 simulated
seconds and a 7,200-second wall limit. Earlier 600-second debugging runs remain
separate. The pilot starts after this audit and retains its nine tasks, two trials
and seeds 70/71. Functional checks and passing
qualification establish a preflight floor, not domain accuracy or perfect caller
reliability. No new aggregate model score is claimed yet.

The latest audit was interrupted after external speech inference was detected.
The saved 688.2-second partial trace has four real customer reads and five
grounded reports, including the formerly blocked current-network question. The
heard requests and their evidence were audited; no completed task outcome is
claimed. It remains unscored, and the pilot has not started. A fresh
exclusive-runtime audit is required before collecting the declared pilot.

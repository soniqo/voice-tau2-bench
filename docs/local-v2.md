# Local v2 execution

`soniqo-local-gemma12b-v2` is an experimental local profile for evaluating voice
agents. It keeps the frozen 278 English airline, retail, and telecom tasks,
their policies and tools, and the existing database and communication graders.
Historical v1 runs remain development results under their original settings.
Changing the caller policy and clock creates a separate experimental condition;
v2 results must not be combined with v1 scores or presented as official τ-Voice
leaderboard results.

See [validation results and working parameters](validation/local-v2.md) for the
spoken pipeline checks, qualification results, and known limits. The
[native cache follow-up](validation/native-prefill.md) verifies the newer runtime.
The [device-controller follow-up](validation/grounded-caller.md) records current
caller validation and held-out pilot progress. The
[acceptance and spoken-speed follow-up](validation/caller-goals.md) records the
goal-verification checks. The [setting-direction follow-up](validation/caller-switches.md)
records the preceding caller revision.
The [status-bar follow-up](validation/caller-views.md) records the latest read-target
correction and validation.
The [NL-judge follow-up](validation/nl-judge.md) records preceding whole-source
qualification and the completed development-call audit. See
[pilot grading coverage](validation/pilot-grading.md) for reporting groups.
The [terminal-JSON follow-up](validation/terminal-json.md) records the shortened
run's caller-format failure, bounded reassessment fix and native controls.
The [completed development run](validation/development-smoke.md) records all three
short calls, their raw outcomes and the remaining agent/runtime investigation.
The [tool-history and caller-pacing follow-up](validation/tool-history-and-pacing.md)
records the integration defects found in those traces and the validation of their fixes.

## Faster development runs

Use the separate three-task plan for one state-changing task per domain, one
trial each, and a shorter call budget. After starting the pinned native service
and qualifying the current source, run:

```bash
uv run --frozen tau-voice batch --domain all \
  --task-plan configs/pilots/local-v2-development.json \
  --trials 1 --seed 70 --complexity control \
  --max-seconds 300 --wall-timeout 900 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-v2-development
```

The CLI flags set the budgets; the plan's recommendation fields are descriptive.
Each call has five simulated minutes and a fifteen-minute wall watchdog. Inspect
tool calls, heard speech, caller decisions and errors after each trial. A timeout
is a zero under the frozen grader, but the shortened limit may prevent a valid
longer solution. Inference and caller-control errors remain unscored and stop
the batch. Keep these results separate from the 18-call pilot and full-frame
scores. Preserve an interrupted pilot and resume it only with its original
source, models and settings.

## Models

| Role | Model |
| --- | --- |
| Evaluated agent | Gemma 4 12B IT, INT4 |
| Customer, turn-taking helpers, communication judge | Gemma 4 E4B IT, INT4 |
| Customer and agent speech | VoxCPM2, FP16 main parameters, FP32 AudioVAE |
| Agent recognition | Full Whisper Large-v3, CoreML FP16 |
| Speech detection | Silero VAD |

Use the pinned native service and model staging described in
[runtime setup](runtime-setup.md). All inference stays on the loopback service.
One benchmark process owns the service; heavy inference runs sequentially.

## Call time and wall time

The default call limit is **2,400 simulated seconds**, matching the voice limit
in the pinned upstream configuration. Audio advances in 50 ms ticks. Customer
LLM and speech-generation computation happens between ticks, without consuming
the agent's call budget. A separate **7,200 wall-second watchdog** bounds a slow
or stalled run. `max_steps` identifies the call cap; `timeout` identifies the wall
watchdog. Both receive zero reward under the unchanged evaluator. The watchdog is
checked between ticks; an in-flight native request can delay that check.

Agent latency remains measured. ASR, agent LLM, and streaming TTS execution
durations schedule when transcripts, tools, and audio become available in call
time. An agent response physically completed while the customer was computing
stays buffered until its measured latency has elapsed in the simulation. Queue
waits behind customer inference are recorded separately. The agent cannot start
another response early merely because its background worker has already finished.
Interruptions discard superseded speech and tools while preserving recognition
of speech already heard.

After a completed answer, the local v2 customer waits for a new heard agent turn.
At 15 simulated seconds of silence, it asks whether the agent is still there
instead of repeating an identifier or instruction. Completion and refusal are
assessed before waiting. A validated continuing assessment is reused only while
the meaningful caller-visible history, scenario, model and tool schemas remain
identical; empty audio frames cannot trigger another assessment. The silence
clock still determines when to check in. V1 retains its five-second setting. This common
local condition gives the ASR/LLM/TTS cascade time to begin responding; all agent
execution time still counts, and every model uses the same patience setting.
The selected value is recorded in the profile and per-trial settings. It changes
the interaction condition and must be reported with any result.

This is a paced local cascade. It does not reproduce an audio-native provider's
network timing or claim equivalence to a live telephone deployment. Speech text
alignment remains estimated until synthesis completes.

## Qualify the customer and judge

Before running v2, execute the separate hand-labelled calibration suite:

```bash
uv run --frozen tau-voice qualify \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-v2-qualification.json
```

The 84 checks cover completed versus promised work, required follow-ups after a
refusal, incomplete answers, unknown identifiers, goal disclosure, role separation
and positive/negative semantic facts. Fifty-one use the full 30-tool telecom
customer interface, including actual reads and writes in fresh fixture devices.
They cover read-before-toggle and no-op behavior, stale facts after a write,
compound instructions, required parameters, persona reassurance, VPN states,
service-working and data-speed questions versus enabled settings, and agent offers that must not
become customer actions. Streaming cases include incomplete speech and completed
questions heard during an ongoing turn; four cases exercise callers without a
phone controller, including waiting and checking in after an answered identity question.
Two further cases distinguish completed work followed by a closing help question
from an actual pending action followed by the same question.
Three spending cases check an opening price limit, the permitted alternative
after an over-budget quote, and completion with a known fee inside the limit.
Two judge cases exercise the actual NL-assertion prompt in addition to the six
communication cases. NL results must cover every requested assertion with a
Boolean value. A judge-model mismatch fails qualification before inference.

These fixtures are outside the frozen task frame. Passing checks is a preflight
requirement, not evidence of domain accuracy or complete customer reliability.
Reports from different sources, suites, models or execution conditions do not
qualify a current run. Reply-generation and evaluation exceptions fail
qualification even after a valid terminal assessment. Error-bearing reports
cannot enable a benchmark run.

A model-driven customer controller reads only the customer's scenario and the
heard dialogue. It decides whether to continue, finish completed work, or finish
an addressed refusal. It never sees reference actions, grading state, or agent
private tool results. A terminal assessment requires a supporting quote actually
heard from the service agent. A conservative English-language guard keeps the
call open if that quote merely offers or promises a future action, or consists
only of an English confirmation question. A completed-action statement followed
by a question is still eligible for a terminal decision. These guards
exclude a final courtesy question such as “Is there anything else I can help you
with?” from the future-offer check while checking all preceding quoted evidence.
A courtesy question alone still cannot prove completion. The
[closing-question follow-up](validation/closing-question.md) records the captured
false continuation and its regression coverage. These guards
cannot generate a stop. A terminal proposal with malformed JSON syntax, invalid
control-field values (such as a misspelled decision enum), or without a supporting agent quote gets
at most one model reassessment using the same caller-visible
scenario and conversation. The rejected proposal is recorded. The reassessment
must pass the same evidence checks; a second unsupported proposal leaves the
trial unscored. Assessment tool calls and inference errors propagate immediately.
JSON syntax and typed control-field errors share the single bounded schema
reassessment above; this does not add another retry after a rejected quote.
Every rejected response is retained, and no decision or
evidence is filled in by the runtime.

The caller states scenario spending limits in its opening request, asks for an
unknown total before consenting to a paid change, and follows the scenario's
alternative when a quote exceeds its limit. A completion claim with an unknown
price does not establish that an explicit spending limit was respected. These
are caller instructions; the agent learns them only through ordinary spoken
dialogue. See the [spending-constraint checks](validation/spending-constraints.md).
Customer device tool results are observations, and a setting change or strong
signal alone does not prove that the requested service works end to end.
The v2 profile explicitly enables `execution.device_grounding`. For telecom,
`CustomerDeviceController` handles recognized requests before the model-driven
TOOL/SPEAK path. It reads only heard agent speech and actual customer tool
results. It returns declared calls for normal execution, reads unknown settings
before toggling, and reports an already-correct setting without changing it.
Every customer write invalidates cached device observations; successful results
restore only facts they actually report. Rejected/failed calls and service-agent
private results cannot become customer facts.

Phone-status speech comes from those observations using deterministic rendering.
VPN enabled-but-disconnected remains distinct from a disabled or connected VPN.
The controller records source tool IDs, preserves requested action order, asks
for missing/ambiguous parameters and pauses rebooting for explicit photo-safety
concerns until the agent reassures the caller. It can only keep a pending device
exchange open, never manufacture a successful terminal state or reward.

In v2, speech-frame merging and caller role conversion preserve the last frame's
utterance boundary across airline, retail and telecom. The caller waits on an unfinished phrase, while
a complete question can still prompt a response during agent speech. After the
configured idle interval it can ask whether the agent is still there. The guard
also covers gaps inside unfinished speech; a complete question can still permit
barge-in. The selected behavior is recorded in `settings.json`. Phone-number
questions use only the number explicitly supplied in the caller's known information;
an unknown number is reported as unknown, without a device check.
Agent tool calls remain silent and do not reset the caller's speech timer.
Describing the original reported symptoms stays with the conversation model;
an explicit device check still requires the actual customer tool result.
Mentioning data in a location, line-selection or symptom question does not request
a Boolean setting read. Internet-working questions and speed tests use the speed
test; MMS capability questions use the MMS test. A distinct request to run a speed
test performs a new test. Negated test requests require clarification and execute
no tool. Explicit setting changes retain their read-before-write behavior.
Switch directions bind to the requested setting or coordinated setting list:
“turn off Airplane Mode on your device” requests OFF; the location does not supply
an ON direction. A reason such as “to restore mobile data” does not request a
second setting change. Unsupported or contradictory directions require clarification.
Status-bar questions read the actual status-bar view. Example labels such as
“Data Disabled” do not replace that view with a cached mobile-data setting.
Explicit coordinated setting and status-bar reads preserve both observations.
Tool dispatch, known-number replies and vague follow-ups share the same check of
who was asked to act. Agent offers such as “Would you like me to check that?”
stay conversational and cannot become a customer recheck. Follow-ups inspect the
latest customer request, excluding an agent's own promised work in its preamble.
Agent-only offers and promises restrict the caller to SPEAK mode before the
model chooses a response. The local model still decides how to answer using its
scenario and persona. Already-spoken tool results remain in history, while only
an unspoken result takes priority over a new question. The next heard agent turn
is also presented directly before the labelled conversation data.

Other conversation, caller goals, account information, and business decisions still use
the local caller model. This finite English device adapter is an explicit
experimental caller condition; clarification for an unrecognized instruction is
part of that condition. It does not read hidden device state or reference actions,
repair the evaluated agent's arguments, or substitute gold speech. Historical v1
profiles retain their original execution settings and do not enable it.
Caller control reads heard turns: adjacent speech fragments from the same speaker
are merged, while speaker changes and tool-call/result boundaries are preserved.
For open calls outside the device controller, a separate TOOL/SPEAK decision reads the
heard request, caller scenario/persona and actual tool history. TOOL requires a
native customer function call before speech; saying the action happened does not
execute it. The tool request repeats the actual latest heard agent statement
before the labelled history so it is treated as the pending instruction rather
than an old transcript item. A prose response in TOOL mode gets one model
reassessment with the same heard context. A second prose response leaves the
trial unscored. Account questions without a matching customer tool route to
SPEAK, where the caller can admit missing information. Unknown
tools, invalid arguments and inference errors fail immediately. The ordinary
orchestrator executes validated tool calls; the controller never changes device
state itself. The leading routing label is used; any following annotation is
recorded and does not select a tool or change its arguments. SPEAK turns receive
no tools and answer the latest agent question using actual customer tool results.
Persona-specific clarification requirements still apply. Terminal decisions
allow 384 tokens, routing allows 16, and normal
caller generation uses its configured limit. No reference actions or private
agent state enter these choices.

Calibration measures continuing versus stopping; the
completion/refusal label remains diagnostic. Open calls generate a customer response with
customer tools, the scenario, and persona instructions. V2 uses explicit local
caller guidelines in production and qualification, avoiding the inherited vague
opener instruction. Caller system reminders are combined
into one initial system turn for Gemma; the dialogue is supplied as labelled
customer/service-agent/tool records to avoid speaker-role confusion. Malformed decisions, an unsupported early
stop, or rejected customer tool calls invalidate the trial and leave it unscored.

`run` and `batch` require a passing report that matches the working source hash,
calibration fixture hash, caller/helper/judge roles, and native model pins. Source
or model changes require qualification again; a documentation-only commit does
not. Failed qualification exits with a nonzero status and preserves its report.

When the caller explicitly requires an Excellent speed-test result, v2 binds
completion to the successful customer measurement. Signal strength and enabled
settings cannot satisfy that condition. Closing remarks trigger a read-only
verification while it is unverified; device writes invalidate old observations.
The evaluated agent still chooses the repair actions. Successful speed results
speak their actual quality and rate in plain words, with the original tool result
retained as evidence. See the [goal-verification record](validation/caller-goals.md).

## Run a functional check

```bash
uv run --frozen tau-voice run --domain mock --task-id create_task_1 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 42 \
  --max-seconds 600 --wall-timeout 1200 --complexity control \
  --output data/simulations/local-v2-functional
```

Mock tasks test the spoken request → ASR → agent → tool → database → spoken
response path. They contribute no domain score. Each output directory must be
fresh. Omitting both duration overrides selects the profile defaults above.

The v2 agent receives general instructions to use declared tools, ask the caller
to act on their own device, and recover identifiers from visible successful
lookups. A mixed batch containing an undeclared tool is entirely rejected before
execution, with matching tool-error messages. Duplicate call IDs and a third
identical failed call are rejected. A corrected argument can still be tried.
Rejected calls remain in the trajectory with `execution_status: rejected`, so
database replay does not execute them. Error exhaustion remains a failed run;
it never becomes an artificial successful stop. No identifier, goal, or tool
argument is filled in from reference answers.

## Held-out pilot

The predeclared plan contains nine tasks sampled uniformly within the three
domains, excluding the three historical first-task debug cases. Selection uses
seed 20261024. Two trial seeds produce 18 calls per evaluated agent:

```bash
uv run --frozen tau-voice batch --domain all \
  --task-plan configs/pilots/local-v2-heldout.json --trials 2 --seed 70 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 --complexity control \
  --output data/simulations/local-v2-pilot
```

The plan is a small pilot, not a leaderboard sample. It is balanced by domain,
not stratified by task difficulty. Preserve its IDs and settings before examining
scores. `--task-plan` and `--num-tasks` are mutually exclusive. The manifest saves
the full selection, jobs, seeds, budgets, qualification fingerprint and provenance.
`--resume` requires an exact match and verifies saved artifact hashes.

## Inspect results

Every trial saves settings, source/model pins, audio, inference events, the full
trajectory and reward components. `simulation.info` adds agent/customer/judge
queue and execution times, tool attempts and rejections, customer control
decisions, and native failure attribution by request ID. Requested cancellations
are separated from unexplained inference failures; captures from other trials in
the same directory do not inflate the current trial's failure count.

Native builds that expose `cachedPrefixTokens` also save `cached_prefix_tokens`
in each chat inference event and assistant message. Zero means a cold prefill;
an absent field means the runtime did not report this metric. The effective
native prefix-cache budget is recorded in provenance. Reusing an unchanged
system/tool prefix can reduce agent latency; its measured execution time still
counts. Report the native revision, cache budget, and cold/warm request counts
with timing results. Never prefill private task goals or reference answers.
See [native cache validation](validation/native-prefill.md) for the measured
speedup and completed HTTP, qualification, and spoken pipeline checks.

Batch summaries show planned/completed trials, scored coverage, termination
counts, and pass^k for tasks with all requested trials. A failed inference or
invalid customer stops the batch, remains unscored, and saves the failed job and
partial coverage in `summary.json`. Unscored trials include pending jobs; inspect
`failed_trial` to identify the attempted failure. Do not report an incomplete slice
as a full benchmark result.

Start with clean telephony (`control`). Add `regular` acoustics only with the
required upstream audio assets staged and after confirming the baseline pipeline.

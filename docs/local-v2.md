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
The held-out pilot has not run.

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

The local v2 customer waits 15 simulated seconds after their own speech before
repeating a request into silence. V1 retains its five-second setting. This common
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

The 19 checks cover completed versus promised work, required follow-ups after a
refusal, incomplete answers, customer device tools, unknown identifiers, and
positive/negative semantic facts. They also check customer/agent role separation,
goal disclosure, and customer claims that lack agent confirmation. These fixtures are outside the frozen task
frame. Passing checks is a preflight requirement, not evidence of domain accuracy
or complete customer reliability.

A model-driven customer controller reads only the customer's scenario and the
heard dialogue. It decides whether to continue, finish completed work, or finish
an addressed refusal. It never sees reference actions, grading state, or agent
private tool results. A terminal assessment requires a supporting quote actually
heard from the service agent. A conservative English-language guard keeps the
call open if that quote merely offers or promises a future action; the guard
cannot generate a stop. Calibration measures continuing versus stopping; the
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

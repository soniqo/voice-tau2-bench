# Local confidence pilot

The predeclared [30-task plan](../../configs/pilots/local-v3-confidence-30.json)
samples ten task IDs uniformly without replacement from each frozen English
domain, using selection seed `20261028`. Selection uses no outcome or difficulty
information and excludes no tasks. Some tasks may have appeared in earlier
development runs; this is a development sample, not a held-out comparison.

The condition is Gemma 4 12B IT INT4 as agent, E4B INT4 as caller/listener/judge,
Whisper Large-v3 CoreML FP16, VoxCPM2 FP16 with FP32 AudioVAE, and Silero VAD.
Each task has one trial with seed 70, control acoustics, a 600-second simulated
call budget and a 1,200-second wall watchdog. Inference stays local. The local
caller, audio condition, and time budget must be reported beside any score;
these results are separate from the upstream standard voice leaderboard.

## Packaged launcher validation — 2026-10-28

The documented launcher completed after building a clean, pinned native checkout
and installing the frozen Python dependencies in a new environment. Fresh setup
exposed an omitted Swift dependency lockfile; the package now bundles the exact
tested pins and rejects modified dependency files. Model loading also required
freeing disk space: rebuilding files from earlier validation runs reclaimed
16 GiB, leaving about 21 GiB free before the successful retry.

All 91 caller/judge calibration cases passed. The spoken controlled write scored
1; the deliberately wrong write scored 0. The fresh Python environment passed
901 offline tests with one optional source-fixture check skipped. Running with
that fixture configured passed all 902 tests. Lint, all 64 frozen input pins and
both upstream grading replays passed.

The native source SHA-256 was
`1986c2739d7246f857c45bb8467622d705a65bc22e44862d31c93db56a6af6ab`,
dependency lock SHA-256
`c4a0fb2db41abbc1f4fb90919adf27c0e9110809544580eb5ce065c1b4897900`,
and built server SHA-256
`8a28a0940d6cd6d8b401e5fcc361ba36c9240d69e3aef8b377b7f4de3f1c1f7d`.
This validates setup and the controlled pipeline. The 30-task domain result
requires its separate saved trial evidence.

## Caller validation after the first domain call

The initial v3 attempt completed airline task 39 normally at its 600-second
simulated limit, after 796.1 wall seconds, with no unresolved inference failures.
Its raw reward was 0 for `max_steps`. The caller repeatedly restated the
cancellation request instead of answering the agent's legitimate question about
the reason. This consumes the call budget and confounds an agent-quality score;
the raw trace is retained and excluded from model comparison. Another Swift
build then interrupted the batch. Its completed checkpoint survived resume.

A separate v4 condition adds `question_focus` to the caller generation context.
In the captured caller-only replay, v3 reproduced the non-answer and v4 supplied
the scenario-supported reason. Seven independent actual-model regressions also
passed: cancellation reasons, a colour mismatch, two counterfactual dates, an
unknown identifier and an unknown motive. No reference actions or private caller
facts were sent to the agent. The task plan and grading remain unchanged.

V4 uses `configs/voice/soniqo-local-gemma12b-v4.json` and a separate 98-case caller
and judge suite. Its domain attempts must use fresh output directories and must
not be combined with the earlier v3 condition.

The packaged v4 launcher passed all 98 calibration cases, the spoken positive
control (reward 1), and the wrong-write control (reward 0). The final offline
suite passed 907 tests, with 49 actual-model checks deselected. The seven focused
actual-model caller regressions passed separately. Native source and dependency
pins remain the same as above. The validated executable benchmark-source SHA-256
is `2e2387ff04bc819e8c73b81a6eed23d6ef66ae289c3ffd6a95703afeb2601c86`.
The fixed 30-task v4 domain run is separate from these setup checks.

The first v4 call, airline task 39, reached the 600-second simulated limit after
825.5 wall seconds and received raw reward 0, with no unresolved inference
failures. The agent fetched the customer profile, then repeatedly asked for
flight selection or dates despite the stated scope of all upcoming flights.
It never inspected a reservation's details or cancelled a flight. This is one
local-system diagnostic, not an overall model accuracy estimate. The remaining
cohort and its per-domain metrics require their saved completed-call evidence.

## Run the domain subset

First run the [packaged smoke check](../runtime-setup.md#one-command-local-run).
Its correct spoken write must score 1 and its deliberately wrong write must
score 0. Restart the same guarded runtime with the resulting `models.json` and
retain its `qualification.json`. The source, binary, models, profile and native
configuration must match that qualification.

With that runtime listening on port 8787, run the fixed subset:

```bash
uv run --frozen tau-voice batch --domain all \
  --task-plan configs/pilots/local-v3-confidence-30.json \
  --profile configs/voice/soniqo-local-gemma12b-v4.json \
  --qualification data/simulations/first-local-run/qualification.json \
  --base-url http://127.0.0.1:8787 --trials 1 --seed 70 \
  --complexity control --max-seconds 600 --wall-timeout 1200 \
  --output data/simulations/local-v4-confidence-30
```

The batch saves each completed call before advancing. An inference or setup
error remains unscored and stops execution for diagnosis. After reviewing the
error, use `--resume` with identical settings and source to retain completed
results. Interrupted artifacts are archived; completed quality failures are
never retried to obtain a passing result.

Report successful tasks, scored coverage, infrastructure errors, unattempted
tasks, call duration and response latency separately. Ten tasks per domain give
an early diagnostic; publish full-domain results for a broader comparison.

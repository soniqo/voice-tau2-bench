# Amazon Nova 2.5 Sonic

The optional `tau-voice-nova` runner connects Nova's native speech input/output
and function calls to the existing full-duplex orchestrator, domain tools and
frozen evaluator. Airline, retail and telecom tasks retain their original data,
policies and rewards. The default `tau-voice` command remains local.

## Setup and run

Start the loopback runtime following [runtime setup](runtime-setup.md). It runs
the caller/listener/judge with Gemma E4B and caller speech with VoxCPM2. These
four roles are checked explicitly; Nova does not use the local agent LLM, ASR or
conversation TTS. The common fixed opening greeting uses VoxCPM2. Only its
already delivered text is added as assistant history; no synthetic customer
text, private goals or reference actions are sent to Nova.

```bash
uv sync --frozen --extra dev --extra nova

# Local caller/judge calibration; no AWS inference.
uv run --frozen --extra nova tau-voice-nova --qualify-only \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/nova-caller-qualification.json

# Validated Retail task; use your configured local AWS profile.
uv run --frozen --extra nova tau-voice-nova \
  --domain retail --task-id 0 --seed 70 --complexity control \
  --aws-profile benchmark \
  --qualification data/simulations/nova-caller-qualification.json \
  --base-url http://127.0.0.1:8787 \
  --max-seconds 420 --budget-usd 1 --stop-usd 0.25 \
  --budget-ledger data/simulations/nova-budget.json \
  --output data/simulations/nova-retail-0
```

For a three-domain development batch, replace `--domain retail --task-id 0`
with `--domain all --task-plan configs/pilots/state-change-development-3.json`
and use a fresh output directory. This expands beyond the validated Retail call.
For a batch, `--num-tasks` or `--task-plan` selects frozen task IDs. `--trials`
sets repetitions. `--resume` verifies source, model, settings, ledger-path and
artifact fingerprints before skipping completed trials. Failed inference/caller
attempts remain unscored, with their artifacts preserved. Successful task
execution can still score zero if its actions do not satisfy the task.

## Clock and audio

Nova uses the explicit real-time v6 interaction condition. It cannot run on the
accelerated local simulation clock. Calls exchange 50 ms frames of 8 kHz mono
mu-law audio; the adapter statefully converts them to 16 kHz PCM16 input and
converts Nova's 24 kHz PCM16 output back to telephony audio.

The adapter buffers a speech segment until its final provider transcript is
available, then emits audio frames with proportional delivered text. Speculative
text never enters caller history or grading. This adds adapter latency and is
not word-level alignment. The AUDIO content-end marker determines the final
spoken frame; FINAL TEXT markers can identify different segment boundaries.
Interruption clears unplayed speech and unissued
tools. Already emitted tools retain atomic execution and matching result IDs.
All tools execute through the ordinary orchestrator, never in an SDK callback.

For a matched Gemma comparison, use `configs/voice/soniqo-local-gemma12b-v6.json`
with the local runner and the same qualification, tasks, seed, acoustics and
limits. Old v2–v5 simulation-clock results are separate diagnostics. The seed
controls the caller/acoustics; it is not a Nova sampling seed. Nova's sampling
parameters are recorded in each run.

## Execution and accounting

The runner currently supports one bounded connection per trial, with calls of
**1–420 seconds**. It rejects longer calls before inference. AWS documents an
[eight-minute connection limit](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html).
Automatic conversation continuation is not implemented; a connection failure is
an unscored error. The worker closes both stream directions and drains final
usage before accepting a result. Background SDK failures and unconfirmed final
usage prevent grading rather than becoming successful or zero-cost trials.

`--budget-usd` is a persistent campaign budget. Before connecting, the ledger
reserves `--stop-usd` plus $0.10 for delayed usage. Unknown-cost attempts keep
their reservation; an active reservation prevents another process from using
the same ledger. Final usage settles the reservation. Do not delete or replace
the ledger to resume a run. With `--resume`, a reservation whose owning process
has exited becomes unknown-cost and still counts against the budget; a live
owner continues to block a second session. The observed-usage threshold closes the stream,
but delayed provider metering can overshoot; this is not a provider-enforced
dollar ceiling. Local caller inference has zero API cost; hardware/energy are
not included.

The region/model are pinned to `us-east-1` / `amazon.nova-2-5-sonic`. Rates are
from the [AWS regional price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json)
published 2026-10-28: input/output speech $3/$12 per million tokens, text
$0.33/$2.75. Cumulative meters are counted once per session. Credentials are
resolved only at connection time and are not written to run artifacts.

Runs save ordinary simulation/audio/tool artifacts plus the provider event
record, usage, cost and external-agent configuration. Native process memory
covers the local supporting runtime, not Nova's cloud model memory. Full-call
wall time, audio duration, component timing and task success are distinct.

## Validated pipeline control

On 2026-10-29, the complete production runner passed a two-turn recorded-caller
control with a live local VoxCPM2 greeting, Nova speech, ordinary tool execution
and the frozen evaluator. The correct `create_task` write scored **1**; replaying
a deliberately wrong title scored **0**. After the turn-boundary fix, the fresh
control confirmed final provider usage of **$0.00332019**, with
**28.80 audio seconds / 39.75 wall seconds**. This proves
the integration and evaluator control, rather than an airline/retail/telecom
pass rate. [Machine-readable control](results/nova25-full-pipeline-control.json).

The initial source passed **958 offline tests** and all **103** v6 caller/judge
calibration cases. Its subsequent three-task, four-minute pilot exposed an
adapter defect: using the FINAL TEXT stop reason left some completed spoken
requests open. All three calls timed out. Their frozen raw rewards and usage
are preserved as [adapter diagnostics](results/nova25-v6-adapter-diagnostic.json),
not a model ranking. Airline also exposed ambiguous spoken ID punctuation.

The fix uses AUDIO stop reasons and adds a regression test for the captured
multi-part pattern. Offline replay of four captured Telecom replies changed
from **0/4** to **4/4** correctly placed turn endings; **959 offline tests** pass.
The fixed source also passed a fresh **103/103** caller/judge calibration and
the two-turn live control above.
Domain traces still require caller-adherence review; calibration does not
establish a model ranking.

## Domain development check

The fixed source completed **Retail task 0, one trial, seed 70**, with control
acoustics and a **420-second real-time / wall cap**. Nova authenticated the
caller, inspected the order and both products, obtained the replacement and
payment choices, then exchanged both required items. It correctly reported the
**$16.63 refund** to the chosen original card. The caller ended normally.

The frozen database check passed, and all **5/5** reference actions matched:
**pass^1 1/1**, **0/5 tool errors**, **259.0 audio seconds / 296.30 wall seconds**.
Final metered cost was **$0.03151542**. The four delivered Nova turns had correctly
placed audio end markers. [Measured record](results/nova25-v6-retail-development.json).

This is one development task and seed. Airline and telecom have no post-fix
domain score yet, and the historical Gemma simulation-clock trials use different
conditions. A model ranking requires matched multi-task coverage.

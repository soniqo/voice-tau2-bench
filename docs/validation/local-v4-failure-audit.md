# Local v4 pilot: what failed

Audited on 2026-10-29. This is a development run of the complete local voice
stack, with a custom caller, control acoustics, one seed and a 600-second
simulation budget. It is not an official leaderboard submission or an isolated
measurement of Gemma's reasoning.

The [machine-readable pilot record](../results/gemma4-12b-local-v4-pilot.json)
includes model roles, budgets, coverage and source fingerprints.

| Domain | Passed / scored | Unscored caller errors |
|---|---:|---:|
| Airline | 2/10 | 0 |
| Retail | 0/10 | 0 |
| Telecom | 0/7 | 3 |
| Total | 2/27 | 3 |

All 30 planned tasks were attempted. Of the 27 scored trials, 21 reached the
simulation limit, one reached the tool-error limit, one reached the wall
watchdog, and four ended with the caller stopping. Premature termination scores
zero; this does not establish that every intermediate action was wrong. The
three caller exceptions remain unscored.

Both passes were policy refusals (airline 46 and 47), with the expected database
left intact. They do not demonstrate successful booking, exchange or repair.
Separate controlled spoken write tests establish that a correct database change
passes and a deliberately wrong change fails.

## Trace findings

- **Exact identifiers break the retail flow.** All ten retail calls stall in
  customer or order lookup. Six show name/identity failures involving speech
  recognition and inadequate repair. In the other four, order lookup fails;
  in three of these the profile tool explicitly returned `#W...` identifiers
  and the agent submitted `W...`, dropping the prefix. Repeated lookup requests
  consume the conversation budget. Do not silently correct model arguments.
- **The agent does not consistently use information it already has.** Airline
  39 requests cancellation of all upcoming flights, but the agent asks for
  destinations/dates instead of inspecting reservations. Airline 44 repeatedly
  claims it cannot determine flight duration. Airline 2 does not resolve a
  deliberately incorrect passenger count from reservation records.
- **Valid functions can execute the wrong transaction.** In airline 37, the
  agent quotes a $27 upgrade, then executes a simulated $1,194 charge against a
  different payment method than requested. These are fictional task transactions,
  not cloud charges. JSON validity does not establish correct authorization.
- **Caller behavior contaminates several failures.** Telecom's router labels
  offers such as “I can check…” as customer device instructions. It also demands
  a phone tool for an unknown plan-usage answer. These account for the three
  unscored exceptions. Some scored calls show caller defects too: irrelevant
  repeated answers, a placeholder date of birth, promised payment without its
  tool, or acceptance of an incomplete repair. The score cannot isolate the agent.
- **Increasing the limit would not repair these problems.** It would slow
  iteration and change conditions without fixing identifiers, invented
  verification requirements, unsupported price quotes or caller adherence.
  None of the 497 recorded agent generations hit the 1,024-token output cap;
  output truncation does not explain this pilot's failures.

## Changes and next comparison

The explicit `soniqo-local-gemma12b-v5` candidate adds `routing_focus` while
retaining v4's question focus. It distinguishes offers from customer instructions,
keeps account-usage questions conversational, answers line selection from
caller-known phone information, and treats passive network-mode questions as
reads. It does not expose private caller facts to the agent or change frozen
tasks, tools, policies or rewards. V4 retains its previous routing behavior.

The v5 caller/judge suite extends the previous 98 fixtures with five independent
cases. Current sources and models require a new passing actual-model
qualification before a scored run. Fixtures do not establish perfect caller
fidelity; payment execution, unknown answers and repair completion still need
domain-trace validation.

Validation of this candidate: **938 offline tests passed; 103/103 actual-model
caller/judge fixtures passed**, including all 98 previous fixtures. An initial
calibration caught a closing-help-question regression; the corrected rule retains
the required speed check. Final calibration saved independent fixture results
with source/model/fixture fingerprints, so resource interruptions could resume
without retrying failed checks. The 64 frozen inputs, 278-task frame and upstream
positive/negative grading replays remained unchanged.

For iteration, use captured caller regressions and positive/negative database
controls first. Then predeclare a small subset with genuine state-changing
tasks in each domain. Hold caller, acoustics, seeds and budgets fixed across
agents. Report task success, coverage and caller errors separately. Native
speech-to-speech comparisons also require matched pacing and interruption handling.

The [three-task development plan](../../configs/pilots/state-change-development-3.json)
fixes one state-changing case per domain before the next condition is run. It is
intended for debugging, includes previously exposed tasks, and is not held out.

Agent improvements belong in a separate experiment: preserve tool-returned
identifiers verbatim; repair spoken names through spelling; inspect reservations
before asking for their contents; calculate prices from tools; confirm the exact
payment method; reconcile each write. Do not retrospectively replace v4 scores.

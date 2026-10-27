# Caller assessment JSON reassessment

The first shortened three-domain run stopped after its airline call. Airline
task 12 reached the 300-second call cap, scored 0.0, and took 457.6 wall seconds.
It made account and reservation lookups but no state-changing call. Retail task
40 then failed after 64.75 simulated seconds because the local caller returned
an assessment with unquoted JSON keys. Retail remains unscored; telecom was not
attempted. These are partial development results, not an aggregate model score.

The terminal controller already allowed one reassessment for invalid typed
fields or unsupported evidence, but JSON syntax errors bypassed it. Syntax
errors now share that same two-request total limit. The model receives the same
caller-visible scenario and heard conversation, its rejected response, the
validation error and the required schema. The runtime does not rename keys,
fill fields, invent evidence or choose a fallback decision.

The new response must pass ordinary schema and evidence validation. Agent offers,
pending customer-device exchanges and unverified service goals retain their
existing guards. Assessment tool calls and native inference errors still fail
immediately. A second invalid response leaves the trial unscored. Rejected
outputs are retained, and successful reassessments account for both requests'
tokens, cost and generation time. Initial prompts, model arguments, frozen data,
tools, policies and grading rules are unchanged.

Offline verification passes 767 tests, lint, all 64 frozen inputs / 278 tasks,
and both upstream replay controls. Two actual-model checks inject a malformed
first response and require the native model to reassess pending versus completed
work correctly. Both pass. The controlled spoken database-write check also
passes: the real write scores 1.0 and a wrong-write replay scores 0.0. Its caller
is scripted, so it is a functional pipeline check rather than a domain score.
Fresh whole-source qualification passes all 77 caller/judge checks without
errors. See the [validation record](terminal-json.json) for source and artifact
hashes.

The reusable [development plan](../../configs/pilots/local-v2-development.json)
selects the first predeclared state-changing task per domain: airline 12, retail
40, and the Easy telecom data-saver/roaming case. Use the exact shortened-run
command in [local v2 execution](../local-v2.md#faster-development-runs). No task
selection was changed in response to a model outcome. A new source requires
fresh qualification and a separate output directory; old errors and partial
results remain preserved. Shortened-run scores must stay separate from the
original 18-call pilot.

The [completed three-task rerun](development-smoke.md) recovered the same malformed
retail response and finished all three calls. It records raw outcomes and their
trace analysis separately from the partial pre-fix run.

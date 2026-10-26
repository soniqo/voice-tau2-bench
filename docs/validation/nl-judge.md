# Natural-language judge validation

The NL-assertion evaluator previously accepted an omitted or empty `results`
array as an empty set of checks. The resulting `all([])` produced 1.0 even when
the task requested assertions. Both evaluator paths now require one typed
Boolean result for every requested outcome, with matching outcome text and
multiplicity. Valid orderings and rewards are preserved. Absent task criteria
remain neutral and make no inference calls.

The native check also exposed a transport-format issue: Gemma returned valid JSON
inside a complete Markdown fence, which the old parser rejected. The parser now
accepts one complete JSON fence without changing its values. Prose wrappers,
multiple objects, missing/extra/duplicate outcomes, malformed fields and tool
calls remain errors. Such trials are unscored; they are not converted to a pass
or a quality failure. Judge prompts, model arguments, reward bases and valid
reward composition are preserved.

Qualification adds correct-amount and wrong-amount cases through the actual NL
judge prompt. It also checks that the NL judge matches the declared model.
Native controls grade two requested outcomes, producing 1.0 for a correct refund
and 0.0 for the wrong amount while retaining the other successful outcome.

Verification passed **762 offline tests**, lint, all **64 frozen inputs / 278
tasks**, and both upstream replay rewards. Fresh whole-source qualification
passed **77/77** without errors. **Fourteen native checks** passed: the two NL
controls plus the existing twelve speech/tool controls. The controlled spoken
database write scores **1.0**; its wrong-write replay scores **0.0**.

## Development call audit

The preceding telecom diagnostic completed its full 2,400-second simulated budget
with reward 0.0. It made identity/details, speed and usage checks but no repair.
The agent repeated two variants of the same 2G/3G question 49 times. Actual ASR
retained Airplane Mode ON and no service, although other technical field
boundaries degraded through speech and recognition.

All 44 device reports match successful customer tool evidence. There were no
unaccounted inference failures or unexpected native cancellations. One
conservative clarification and technical speech remain caller-condition limits.
A separate state replay also fails both environment assertions, independently
of the call cap. This is an excluded development diagnostic, not an aggregate
model score or an official leaderboard result.

The only executable changes since that call are the NL judge validator and
qualification. Caller, agent, audio, clock, model and frozen inputs are identical;
the caller audit therefore applies to the same interaction implementation.
The pilot uses fresh whole-source qualification and its own provenance.

See [the validation record](nl-judge.json) for checks and artifact hashes, and
[pilot grading coverage](pilot-grading.md) for the declared state-changing subset.
The unchanged 18-call pilot follows these checks. No new accepted aggregate model
score is available yet.

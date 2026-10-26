# Pilot grading coverage

The nine tasks and seeds 70/71 remain those in the
[predeclared task plan](../../configs/pilots/local-v2-heldout.json). This coverage
audit was performed before any pilot results. No task, policy, reference action,
reward basis or task selection changes.

Synthetic empty trajectories were replayed against the frozen grader with a
normal `user_stop` termination. Inference was disabled. Three tasks score 1.0
under this control: their target databases are unchanged and the selected
communication or NL components contain no checks. This is a grading limitation;
these controls are not successful model runs. A timeout still scores zero.

| Task | Selected reward basis | Empty-trajectory result |
| --- | --- | --- |
| Airline 38 | DB × COMMUNICATE | 1.0; no communication facts; four NL annotations are outside the basis |
| Airline 47 | DB × COMMUNICATE | 1.0; no communication facts; one NL annotation is outside the basis |
| Retail 10 | DB × NL_ASSERTION | 1.0; no NL assertions |

The other six tasks require a changed state under the no-action replay:
Airline 12; Retail 40 and 84; and all three selected telecom tasks. Their state
component scores zero without actions. The telecom tasks use environment
assertions; Retail 40 also has two required NL assertions. Initial state and
criteria are identical across both planned seeds. Reference replay produced no
action errors.

Reporting keeps the primary frozen-grader pass^1 over all **18 trials**. A
separate state-changing result covers **12 trials**: one airline task, two retail
tasks and three telecom tasks. That subset has a different domain mix. The three
remaining tasks retain **six trials** and require trace review before a passing
grader result can be described as completed work. Neither reporting group is
selected from model outcomes.

The [machine-readable record](pilot-grading.json) contains the exact task IDs,
reward bases, no-action state checks, synthetic full rewards and reporting groups.
See [evaluation](../evaluation.md) for component composition and pass^k.

The NL judge's response contract is separate from empty task criteria. With
requested assertions, missing or mismatched results cannot pass through
`all([])`. They raise an evaluation error and remain unscored. Valid Boolean
results retain the existing reward calculation and judge prompt.

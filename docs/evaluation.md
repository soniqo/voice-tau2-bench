# How τ-Voice evaluates an agent

A task gives the simulated caller a private goal and gives the agent a domain policy plus tools. The agent must understand the spoken request, follow policy, and use real tools to change the simulated environment. The caller may also have tools, especially in telecom. Its private scenario and reference actions are never supplied to the agent.

## Task reward

The evaluator replays the reference actions on a fresh environment to derive the expected final database. It separately replays the actual agent and caller actions, then compares the resulting databases. Reference actions describe one solution; a different valid sequence that produces the required final state can pass. Exact tool matching matters only when the task explicitly lists ACTION in its reward basis; airline, retail, and telecom use final-state and communication checks instead.

Some tasks also require the agent to communicate specific facts. The semantic communication evaluator checks each expected fact against the delivered agent transcript using the preserved judge prompt. This local profile runs that judge with Gemma, so judge errors remain a possible source of scoring differences. Additional environment assertions run when present.

Only reward components selected by the task's reward basis contribute to its final reward. They combine multiplicatively: failing one required component makes the task reward zero. A timeout, tick limit, or premature termination gets zero before successful-completion grading. An infrastructure or inference exception is unscored and stops the batch; it is not converted into a zero-quality trial.

## Repeated trials

Pass^1 is the fraction of successful trials, averaged over tasks. Pass^k measures consistent success across k attempts. With n completed trials and c successes for a task, its estimate is C(c,k) / C(n,k). For example, two successes in three attempts yield pass^1 = 2/3, pass^2 = 1/3, and pass^3 = 0. This is different from the probability of at least one success.

Batch summaries exclude incomplete tasks until all requested trials finish. Results from the development-only mock domain are not part of the 278-task frame.

## Speech diagnostics

Saved WAV files, recognition output, interruptions, inference timings, and estimated delivered-text alignment help explain task failures. They are separate diagnostics; audio quality or speed is not itself the task reward. Successful database changes do not override a conversation timeout.

These results use custom local caller, voice, and judge models. They must be reported separately from upstream leaderboard results, with the recorded source and model provenance.

# Closing questions after completed work

The first airline rerun after the tool-history and pacing fixes completed its
account and reservation reads and both reservation writes. It still reached the
300-second simulated cap, scoring 0.0 after 453.46 trial wall seconds. The caller
model proposed `COMPLETE` three times, quoting the agent's statement that the
business upgrade and two checked bags were complete. The runtime overrode all
three because that quote ended with “Is there anything else I can help you with?”

The future-offer guard matched “I can” inside this closing question as if the
agent had promised an unfinished action. The fix excludes only a final generic
help question from that check. The preceding exact heard evidence is still
checked for future actions, and a standalone question still cannot prove
completion. A valid, supported model terminal proposal remains required; the
guard never supplies a decision or creates a successful stop.

Regression tests cover the captured completion, related help questions, a
completed first action with a second action still promised, and a courtesy
question alone. Two separate qualification cases bring the suite to 81 checks.
Rebuild qualification for this caller source before running a domain trial.

## Separate agent-policy failure

The same run passed `total_baggages=2, nonfree_baggages=2` to the baggage tool,
which charged $100. The two bags should be free under the task's existing
membership/cabin policy. Independent strict environment replay returns 0.0 with
`db_match=false`, even without the timeout gate. Correcting caller termination
cannot turn that incorrect write into a pass.

All four agent tool calls executed successfully. There were no unhandled
inference failures or unexpected native cancellations. One requested cancellation
at shutdown is retained in the diagnostics. The original trajectory, tool
results and zero reward remain preserved; the environment replay is a separate
diagnostic, and no frozen task, policy, tool or reward rule is changed.

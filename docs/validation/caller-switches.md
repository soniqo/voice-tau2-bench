# Caller setting directions

A spoken development call exposed a caller parser failure. The agent asked
“Please turn off Airplane Mode on your device.” The caller treated the location's
“on” as another switch direction and repeatedly asked for clarification. That
interrupted call is unscored and excluded from model-performance results.

The device adapter now binds the direction to the action and its named setting
or coordinated setting list. Prefix and postfix directions work with device
locations. Explanatory text such as “to restore mobile data” does not extend the
list of settings to change. Ordered requests with “then turn it off” retain their
existing behavior. Contradictory or unsupported directions still require
clarification.

The caller reads actual customer tools before toggling and leaves a setting
alone when it already matches the requested state. It does not read hidden
device state or reference actions. Repair decisions remain with the evaluated
agent. This remains a finite English adapter under the experimental local v2
caller condition.

The same-input reproducer is ambiguous on the preceding revision and requests
only Airplane Mode OFF on this revision. Regression coverage includes opposite
initial states, both switch directions, device locations, coordinated settings,
an unrelated setting mentioned in an explanation, and contradictory directions.
The separate qualification suite includes three additional real-device fixtures
for Airplane Mode, Wi-Fi, and roaming.

That qualification first caught a terminal assessment accepting the instruction
as completed work. The terminal guard now uses the same history-only device
response as caller generation: a pending action, observation, or report keeps
the call open. Checking this response executes no tool. The failed report is
retained; the guard cannot create a successful stop.

Verification passed **685 offline tests**, lint, all frozen-input checks and both
replay rewards. Fresh source-matched live qualification passed **74/74** with no
errors. **Ten native controls** passed: three VPN states, two phone numbers, two
speed outcomes through speech and telephony, two changed-result model checks,
and a controlled spoken database task. The correct write scores **1.0**; its
wrong-write replay scores **0.0**.

Validation results and artifact hashes are recorded in
[the machine-readable record](caller-switches.json). The
[previous acceptance checks](caller-goals.md) describe the unchanged speed-goal
guard and spoken measurement format.

The 64 frozen inputs / 278 tasks, policies, tools, rewards, model pins, and pilot
selection are preserved. Functional checks do not contribute a domain score.
A fresh spoken audit precedes the unchanged 18-call pilot. No new accepted
aggregate model score is available yet. Measurements use a shared workstation;
lightweight resident capture is allowed, while the watchdog yields to heavy work.

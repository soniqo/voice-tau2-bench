# Caller status-bar observations

A spoken development call exposed a read-target error. The agent asked about
status-bar icons, giving “No Signal” and “Data Disabled” as examples. The caller
matched the word “Data” and reused a cached “Mobile Data Enabled: Yes” fact. That
answer omitted the requested view, including its Airplane Mode icon. The call was
interrupted, preserved, and excluded from model scores.

An explicit status-bar question now reads the actual `check_status_bar` result.
Setting names used as icon examples do not select a Boolean setting read.
An explicit coordinated request such as “check mobile data and status bar”
still reads both, in the requested order. These operations observe the device;
they do not change settings or choose a repair for the evaluated agent.

A counterfactual voice check also exposed garbled labels when the emoji-rich
view was spoken directly. The caller now speaks the visible labels in plain
words, including the observed signal bars, strength, network type, data label,
and battery percentage. Original tool text remains the evidence. Radio strength
still cannot satisfy an internet-speed acceptance condition.

Regression cases reproduce the original request with both Airplane Mode states
and both mobile-data states, including an existing cached data observation.
They also cover alternate example wording and both coordinated read orders.
The qualification suite adds the original failure against all telecom customer
tools and a real fixture device.
Native voice checks cover opposite Airplane Mode states and require the visible
data, signal, network, and battery labels to survive speech and telephony.

Verification passed **704 offline tests**, lint, all frozen-input checks and both
replay rewards. Fresh source-matched qualification passed **75/75** without errors.
**Twelve native controls** passed, including both status-bar recordings, the
existing VPN/phone/speed checks, changed-result history in both models, and a
controlled spoken database write. That write scores **1.0**; its wrong-write
replay scores **0.0**.

The adapter still uses only heard requests and actual customer tool history.
This is a finite English caller condition under experimental local v2.
The 64 frozen inputs / 278 tasks, policies, tools, rewards, model pins and pilot
selection are preserved. Functional controls are excluded from domain scores.

See [the validation record](caller-views.json) for source and artifact hashes.
The [setting-direction checks](caller-switches.md) describe the preceding
revision. A fresh spoken audit precedes the unchanged 18-call pilot; no new
accepted aggregate model score is available yet. Timings use a shared workstation
and make no exclusive-latency claim.

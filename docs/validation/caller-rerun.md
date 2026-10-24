# Local voice rerun: caller validity before model scores

The October 24–25 reruns uncovered both an agent identifier error and faults in
the simulated customer's control. The held-out 18-call pilot is incomplete.
These development diagnostics do not establish an overall model pass rate.
Historical v1 scores remain separate from these v2 runs and caller revisions.

## Completed diagnostics

| Condition | Diagnostic | Recorded result | Interpretation |
| --- | --- | --- | --- |
| Earlier native-cache validation | Airline task `0`, seed `42` | Reward 1.0, normal customer stop | A refusal case with an unchanged database; no communication assertions. See [the original validation](native-prefill.md). |
| Benchmark `2e674765` | Retail task `0`, seed `42` | Reward 0.0, wall timeout | The agent authenticated the customer and received `#W2378156` from `get_user_details`, then called `get_order_details` without `#`. No writes executed. |
| Benchmark `2e674765` | Telecom airplane-mode/roaming diagnostic, seed `42` | Reward 0.0, wall timeout; caller validity failed | The customer repeatedly claimed airplane mode was off without executing the mutation. Device state stayed unchanged. This result cannot isolate agent capability. |
| Subsequent caller-repair candidates | Same telecom diagnostic | Unscored caller-policy errors | Invalid control fields or customer tool responses stopped the calls. These are not additional zero scores. |
| Benchmark `5dcfba27` | Same telecom diagnostic after 32-check qualification | Stopped unscored after the caller validity audit failed | The caller answered a VPN question without a VPN observation. A read-only speed test and network-status check executed, but no phone-setting mutation ran. |

The retail and telecom timeout rewards were assigned for premature termination.
Their database checks were null; they are not independent completed database
grading results. Retail lasted 546.7 simulated seconds and 1,209.01 wall seconds;
telecom lasted 546.9 simulated seconds and 1,201.66 wall seconds. Artifact checks
matched all eight saved run files in each completed diagnostic.

## What the rerun found

- Heard agent speech was stored in small streaming fragments. Treating each
  fragment as a separate request gave the caller incomplete context. Caller
  control now coalesces adjacent heard speech while preserving speaker and tool
  boundaries.
- A spoken customer claim does not execute a phone action. Tool turns require
  declared native function calls with validated arguments, followed by ordinary
  environment execution. The wrapper never supplies tool effects itself.
- Account data usage belongs to the agent's account tools. Unknown account
  information should get an honest answer; an available requested device check
  should execute its customer tool.
- Repeating a toggle can reverse a setting that is already correct. Qualification
  includes this no-op case as well as real airplane-mode and roaming mutations,
  pending requests, unsolicited actions, and persona-specific reassurance.
- Control syntax and quote failures remain visible. A single bounded model
  reassessment may repair typed control fields or unsupported evidence using
  the same heard conversation. Further failure leaves the call unscored.

Caller qualification is a preflight gate, not a domain score. New source and
fixture hashes require a fresh report. A passing fixture suite alone does not
establish that a longer generative conversation is reliable.

The `5dcfba27` candidate passed **32/32 live caller/judge checks**, including the
unknown account question, unobserved VPN read, already-correct setting, and
actual fixture-device mutations. **517 offline tests**, lint, all 64 frozen
inputs (278 tasks), and both upstream replay checks passed. The subsequent
spoken diagnostic failed its validity audit: after account discussion and a
speed test, the agent asked whether the customer was using a VPN, and the
customer said no without calling `check_vpn_status`. The scenario explicitly
requires all device answers to be grounded in tool results. The customer also
failed to relay a network-status observation at first.

The run was stopped deliberately at **221.35 simulated seconds**, leaving it
unscored. Its trace records `KeyboardInterrupt`, not a native inference crash.
There were no unexplained failed requests or native chat failures. One native
cancellation matched the recorded shutdown cancellation intent. The owned model
service was stopped after saving the audit.

Qualification now includes a **33rd check** derived from this longer context;
the old 32-check report no longer enables runs, and the expanded suite has no
passing live certificate. An offline regression test confirms that the captured
ungrounded answer fails qualification; **518 offline tests** pass with that
regression included. The held-out pilot remains gated, and
the caller changes remain draft. The [validation summary](caller-rerun.json)
records the source, fixture fingerprints and failed live audit. It is not a
reusable qualification certificate.

## Fixed execution parameters

| Role or setting | Value |
| --- | --- |
| Service agent | Gemma 4 12B IT, INT4 |
| Customer, listener, semantic judge | Gemma 4 E4B IT, INT4 |
| Both voices | VoxCPM2, FP16 main parameters with FP32 AudioVAE |
| Recognition | Full Whisper Large-v3, CoreML FP16 |
| VAD | Silero VAD |
| Profile | `configs/voice/soniqo-local-gemma12b-v2.json` |
| Telephone condition | `control` |
| Diagnostic budget | 600 simulated seconds / 1,200 wall seconds |
| Customer idle wait | 15 simulated seconds |
| Pilot budget | 2,400 simulated seconds / 7,200 wall seconds per call |
| Pilot selection | Three declared tasks per domain, two trials, seeds 70/71 |
| Physical memory / free disk guards | 24 GiB / 2 GiB |

The native binary SHA-256 is
`cc538f99363f066888436e953a64042421b40806ffcdd6e5dfa5f7e4e577dbbb`,
at native revision `03e525fc7a42d71dafa5777dfe59264689c2b7cf` plus the recorded
source fingerprint. Agent and customer checkpoint revisions are
`73bcf09092aa277861d5a191b989b666f7f32e8f` and
`561d32b215dea1fd2747e5367b094581813588b3`. The native runtime and models were
not changed during caller repair. All inference used the loopback service.

For a fresh source-matched qualification and diagnostic:

```bash
uv run --frozen tau-voice qualify \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/rerun-qualification.json

uv run --frozen tau-voice run --domain telecom \
  --task-id '[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]' \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/rerun-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 42 \
  --max-seconds 600 --wall-timeout 1200 --complexity control \
  --output data/simulations/rerun-telecom
```

Use fresh output directories and preserve failed attempts. Audit actual customer
tool execution before using a repaired caller for the held-out pilot. Frozen
tasks, policies, tools, databases and reward criteria remain unchanged.

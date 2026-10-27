# Three-task local v2 development run

This historical run predates the [tool-history and caller-pacing fixes](tool-history-and-pacing.md).
Its zero rewards remain raw development diagnostics, with the original artifacts preserved.

All three selected calls completed on 2026-10-27 with the qualified local stack.
Each reached its 300-second simulated call cap. The total measured trial wall
time was **23 minutes 11 seconds**. These raw diagnostic outcomes cover the
selected tasks and shortened budget; they do not establish full-frame accuracy.

| Domain | Task | Raw reward | Termination | Wall time |
| --- | --- | --- | --- | --- |
| Airline | 12 | 0.0 | `max_steps` | 484.3 s |
| Retail | 40 | 0.0 | `max_steps` | 462.3 s |
| Telecom | Easy data-saver / roaming case | 0.0 | `max_steps` | 444.3 s |

The run had no unhandled inference failures, unexpected native cancellations or
artifact-integrity errors. All five reported customer-device observations match
successful customer tool results. A separate offline environment replay also
fails the required state checks for every case; both telecom assertions remain
false. That replay is diagnostic and does not replace the frozen reward.

The preceding shortened run stopped on malformed caller-assessment JSON before
reaching telecom. The [bounded reassessment fix](terminal-json.md) recovered the
exact same malformed retail response during this run. The rejected response and
validated `CONTINUE` reassessment are retained. Its two requests count in the
recorded inference accounting.

## What the traces show

- **Airline:** account and reservation lookups succeeded. The agent later claimed
  it could not locate the account and repeatedly asked for its ID. Subsequent
  recognition also changed the spoken identifier. The final archived native
  agent request contains both successful tool results with matching call IDs.
  This verifies delivery to the native endpoint. Subsequent investigation found
  an unclosed native model turn before a following user message and a caller
  that repeated an already answered identity question. Correcting the native
  boundary alone still reproduces the false account-failure claim on the
  captured garbled follow-up. No state-changing call ran in this original trial.
- **Retail:** account lookup succeeded, but the agent omitted the required `#`
  from an order ID in two order lookups. Both returned `Order not found`.
  The frozen tool and its argument rules remain unchanged.
- **Telecom:** the customer executed the requested data-saver change. The agent
  also requested two customer-only tools itself; the tool guard rejected those
  calls before execution. No customer handset-roaming change ran. The call ended
  with the required final state unmet.

These traces point to identifier handling, recovery from tool results and
separation of agent/customer actions as the next development targets. The short
cap can also cut off a valid longer solution. Keep these results separate from
the original 18-call pilot, which remains paused under its original source.

## Tested parameters

- Agent: **Gemma 4 12B IT INT4**; caller, listener and judge: **Gemma 4 E4B IT INT4**.
- Speech: **VoxCPM2 FP16 main weights / FP32 AudioVAE**; recognition: full
  **Whisper Large-v3 CoreML FP16**; VAD: **Silero**.
- Profile: `soniqo-local-gemma12b-v2`; state-changing development plan; **one
  trial per task**, seed **70**, clean `control` telephony, 15-second caller patience.
- Limits: **300 simulated seconds / 900 wall seconds per call**.
- Runtime: pinned local native service, **24 GiB physical-footprint limit**,
  Apple M5 Pro with 48 GB unified memory. This was a shared workstation;
  the timings are not an uncontended performance comparison.
- Validation: **767 offline tests**, lint, **64 frozen inputs / 278 tasks**, both
  replay controls, **77/77** caller/judge qualification checks and three fresh
  native functional checks. The controlled spoken write scores 1.0 and its
  wrong-write replay scores 0.0; that scripted-caller control contributes no
  domain benchmark score.

After following [runtime setup](../runtime-setup.md), qualify the current source
and run:

```bash
uv run --frozen tau-voice qualify \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-v2-qualification.json

uv run --frozen tau-voice batch --domain all \
  --task-plan configs/pilots/local-v2-development.json \
  --trials 1 --seed 70 --complexity control \
  --max-seconds 300 --wall-timeout 900 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-v2-development
```

The [machine-readable record](../results/gemma4-12b-local-v2-dev-300.json) contains
source and model pins, exact flags, rewards, timings, artifact hashes and the
offline state checks. Source identity is
`8548dfaee24b21eeb75e8364567b4973945f6a89ef8550b57ba95c4f06266924`.

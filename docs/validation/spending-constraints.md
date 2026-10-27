# Caller spending constraints

The airline development trace exposed a caller-quality failure independent of
the agent's tool handling. The caller omitted its $650 upgrade limit when
introducing the request and consented before the agent stated a price. The agent
then charged $1,200 for the cabin change and $100 for bags that should be free.
Those diagnostics cannot establish agent performance against a faithful caller.

Four fast checks against the actual E4B caller reproduce the problem without
running a complete voice call. Before the guidance change, two fail: the opening
omits the spending cap, and an unknown-price consent request receives “Yes,
please proceed.” The other checks cover the permitted alternative after an
over-budget quote and a completed action with a fee inside its limit.

The caller now states explicit spending limits up front, asks for an unknown
total before agreeing to a paid change, and follows conditional alternatives
when their conditions arise. The terminal evaluator also requires evidence
that a paid change respects an explicit spending limit. These are model
instructions, with no deterministic insertion of scenario facts into agent
messages. The agent receives only the caller's rendered speech through the
normal full Whisper Large-v3 recognition path.

All four targeted native checks pass after the change. Three qualification
fixtures bring the caller/judge suite to 84 checks; the unknown-price question
is additionally covered by the native regression. Conditional alternatives need
to be stated when their condition applies; the opening need not recite them all.

The frozen task, membership policy, model assets and reward rules retain their
original contents. The incorrect writes and historical zero rewards remain
preserved. New caller instructions require fresh qualification and separate
run output. These checks cover the reported failure; they do not establish
complete caller reliability or a full-frame model pass rate.

## Validation and parameters

The current caller revision passes 788 offline tests, lint, all 64 frozen inputs
covering 278 tasks, and both replay controls. Fifteen fresh native checks pass,
including all four spending cases and the full spoken database write. That
controlled write scores 1.0; its wrong-write replay scores 0.0. Fresh caller/judge
qualification passes 84/84 cases with no failing or error-bearing cases. Source,
model and artifact hashes are in the [validation record](spending-constraints.json).

This retains Gemma 4 12B IT INT4 for the agent; E4B IT INT4 for caller, listener
and judge; VoxCPM2 FP16 main weights with FP32 AudioVAE; full Whisper Large-v3
CoreML FP16; and Silero VAD. The native service has a 24 GiB physical-footprint
limit on an Apple M5 Pro with 48 GB unified memory. Timing is from a shared
workstation and does not establish an uncontended performance comparison.

Qualify the current source, then use a fresh directory for the focused trial:

```bash
uv run --frozen tau-voice run --domain airline --task-id 12 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 70 --complexity control \
  --max-seconds 600 --wall-timeout 1200 \
  --output data/simulations/local-v2-airline-spending
```

The earlier 300-second cap cut off another call just after its reservation
writes, before the final spoken confirmation. The 600-second cap is still a
shortened development budget; it remains separate from the full pilot's
2,400-second call limit. Retail and telecom have not been rerun under this caller
revision. Preserve each cohort's source, settings and results separately.

## Focused airline result

The corrected-source trial ended normally with `user_stop`, after 312.9 simulated
seconds and 554.42 measured trial wall seconds. Its frozen reward is **0.0**,
with `db_match=false`. Independent strict environment replay also returns 0.0.
There were no inference failures or native cancellations.

The agent heard the $650 spending cap in the recognized opening request. The
caller later asked for the total cost and gave consent after the agent quoted
$100. The agent's calculation cancelled out the fare terms and left only the
two $50 bag fees. It then executed an upgrade charging $1,200 and added two
non-free bags charging $100. The Gold membership policy makes those bags free.
All amounts are fictional benchmark data; tools mutate only the test database.

One failed user lookup followed an incorrectly recognized identifier; a repeated
spoken identifier recovered the lookup. The caller also gave an earlier general
go-ahead before receiving a price, despite the revised guidance. It subsequently
requested the cost before the writes. This remains a caller-quality limitation;
the normal stop and incorrect final state do not establish a full-frame agent
accuracy result. The recorded write and its zero reward remain unchanged.

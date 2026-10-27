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

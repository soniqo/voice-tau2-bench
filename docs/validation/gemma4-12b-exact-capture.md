# Exact 12B failure capture and recovery checks

The instrumented Telecom attempt captured the exact failed HTTP request and raw
model completion. Native and benchmark request bodies match byte for byte. Its
[diagnostic record](gemma4-12b-exact-capture.json) keeps source/model revisions,
reference comparisons, and hashes of the retained local artifacts. The earlier
domain attempts and their scores remain unchanged.

## Telecom: a duplicate closing marker rejected a valid call

After customer lookup and reads of `L1001` and `L1002`, the model generated:

```text
<|tool_call>call:get_details_by_id{id:<|"|>L1003<|"|>}<tool_call|><tool_call|>
```

The function and quoted argument are complete. A second adjacent closing marker
caused `malformed("orphan tool call terminator")`. The attempt ended after 3,351
ticks (167.55 seconds of conversation audio) and remains unscored. The generation
stopped normally at token ID 50 after 23 generated tokens, below the 1,024-token
budget. Its 8,112-token native prompt was within the 262,144-token context limit.

The same pinned INT4 weights generated the identical completion in three local
reference comparisons using `mlx-lm` 0.32.2:

| Prompt/tokenizer variant | Prompt tokens | Generated tokens, including stop | Raw output matches native |
| --- | --- | --- | --- |
| Exact native token IDs | 8,112 | 24 | Yes |
| Native prompt, reference tokenizer | 8,108 | 24 | Yes |
| Full checkpoint chat template | 7,862 | 24 | Yes |

The pinned reference parser accepts this output as one `get_details_by_id` call
with `{"id":"L1003"}`. Neither tokenizer differences nor template formatting
explains this particular failure. This is a separate condition from the earlier
reproduced unquoted-string error.

The correction consumes repeated adjacent closing markers only after validating
a complete 12B call. It preserves the function and argument values and creates
no additional call. Standalone or separated orphan markers, incomplete calls,
and invalid argument syntax remain errors. E4B and the public codec retain their
existing strict behavior.

On speech-swift revision `26c901c`, the captured request returned one valid call
with the same function, argument, and 23 generated tokens. Only the request ID
changed for replay. This is an integration diagnostic, not a task reward: reading
another line does not establish successful troubleshooting.

### Complete corrected-runtime retry: scored zero from timeout

The same frozen task, profile, seed 42, control telephony, and 600-second deadline
completed without an inference exception. It recorded 11,211 ticks (560.55 seconds
of conversation audio) and took 609.97 seconds including cleanup. Its reward was
**0.0**, with termination `timeout`.

The agent completed five customer/line/plan reads. The caller ran its device-side
`run_speed_test` tool. The agent subsequently tried to call that caller-only tool
twice itself; both attempts received `Tool 'run_speed_test' not found`. This is a
tool-selection failure exposed after the protocol correction. No new native chat
failure record was produced.

A separate environment replay changed only termination to `user_stop` to inspect
the final state. It still returned zero: the mobile-data and internet-speed
assertions were unmet. This diagnostic does not replace the official result.
Unlike the earlier airline refusal, the corrected Telecom attempt left the task
unresolved even without the timeout gate.

## Caller termination: increasing caller size did not fix this check

A separate check reconstructed the caller's context from the saved airline
ticks through the last delivered agent reply. It retained the task scenario and
normal caller framing. The original in-flight caller request was not captured,
so this is a reconstruction, unlike the exact Telecom request above.

E4B asked whether a cancellation refund was possible. The 12B caller repeated
that it wanted cancellation only with a refund. Neither emitted a stop signal.
This single next-response check does not establish a caller accuracy rate or a
new airline score. It does show that a larger caller alone did not resolve the
observed termination behavior in this context.

## Retail: both models asked the caller to verify the identifier again

A controlled reconstruction used observed ASR text, successful authentication,
and two failed order lookups. It executed real local retail tools and exposed
their normal results. No canonical order ID or reference action was inserted
into the agent's prompt.

Both E4B and 12B asked the caller to check the order number again. Neither chose
another lookup in its next response. This tests the recovery decision in that
context, rather than a complete exchange or a domain reward. The original
retail zero remains unchanged.

## Separate guidance probe

An additional controlled probe added general caller guidance to acknowledge a
policy refusal after required follow-ups, and general agent guidance to retrieve
canonical identifiers through available tools after a failed lookup. It supplied
no canonical order ID or reference action to the agent. Neither E4B nor 12B
emitted a caller stop signal or selected another retail lookup in its next reply.
The exact guidance and local artifact hash are recorded in the JSON report.
These single-context probes do not establish general model capability. No
guidance was added to the default profiles or used in the corrected voice trial.

## Capture and validation

The benchmark now retains the exact sent JSON body and error response for failed
chat requests. An optional `--failure-capture-directory` during model staging
enables private native failure records, including the rendered 12B prompt,
input/generated token IDs, raw output before parsing, and finish reason. Capture
does not retry a request or convert an exception into a scored result.

Validation passed 24 focused native unit tests, both real-weight trace/history
tests, and live request-body matching. A deliberately truncated tool call still
returned HTTP 500 while retaining its raw trace locally. Hosted native CI ran
1,787 tests, with 46 skipped and zero failures; CodeQL passed. The benchmark's
410 offline tests, lint/format checks, frozen inputs, and upstream replays passed.

The complete controlled spoken database-write test also passed on the corrected
runtime in 58.08 seconds: the correct write earned 1.0 and a separately regraded
wrong-write control earned 0.0. It used a fixed spoken caller request and does
not contribute a domain score. Historical results remain unchanged; the new
Telecom retry is a separate one-task result on its own runtime revision.

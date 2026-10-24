# Why the 12B domain attempt did not pass

The recorded attempt changed the agent to Gemma 4 12B INT4. It retained the E4B
caller, listening helpers, and judges, along with VoxCPM2 FP16, full Whisper
Large-v3 FP16, and Silero VAD. It measures the entire pipeline. The two scored
zeros and one unscored error do not establish an agent-only accuracy result.

The [original report](../results/gemma4-12b-local-600.json) remains unchanged.
The [diagnostic record](gemma4-12b-domain-analysis.json) captures the timeout
counterfactual and source revisions separately.

| Domain | Recorded outcome | What the trace establishes |
| --- | --- | --- |
| Airline | 0; timeout | The agent looked up the reservation and user, then refused the ineligible cancellation. The caller continued asking for exceptions and refunds until the deadline. |
| Retail | 0; timeout | An incorrect ZIP transcription was corrected. The agent then repeated a failing lookup of `W2378156`; the database identifier is `#W2378156`. No exchange was performed. |
| Telecom | Unscored; native HTTP 500 | The agent made 15 account/line reads after three recognized caller utterances. A reconstructed diagnostic request reproduced a malformed tool-argument literal. The original failed completion was not saved verbatim. |

## Airline: the zero hides a correct refusal

The evaluator returns zero for a timeout before checking the database or
communication criteria. A separate diagnostic replay kept every recorded tool
call unchanged and changed only the termination reason to `user_stop` to inspect
the final database state. The environment check returned `db_match: true` and
reward 1.0. This task has no required communication facts.

That is a counterfactual diagnostic, not a new benchmark result. The official
reward remains zero. It shows that caller termination, rather than a wrong
cancellation, prevented this attempt from passing. The agent currently depends
on the simulated caller ending a normal conversation; its explicit stop path
handles transfer completion.

## Retail: identifier recovery failed after recognition recovered

The first ZIP recognition was `1CN122`; a later attempt recovered `19122` and
the user lookup succeeded. Both subsequent order lookups used `W2378156` and
returned `Order not found`. The caller's generated speech also omitted the `#`,
so the missing prefix cannot be attributed solely to ASR.

The policy permits looking up the authenticated customer's profile and orders.
Reading that profile to recover the canonical order identifier was an available
next step. The agent instead repeated the same unsuccessful lookup. This is an
observed tool-use/recovery failure, not evidence that the exchange grader was
broken. No canonical identifier or reference action should be injected into an
agent prompt to make a rerun pass.

## Telecom: planning and protocol failures need separate checks

The agent repeatedly read the customer's three lines instead of advancing the
diagnosis or asking for device-side checks. The reconstructed native prompt was
10,214 tokens, below the checkpoint's 262,144-token context limit. The configured
generation budget was 1,024 tokens, with thinking disabled.

The diagnostic native log reported `Gemma4ToolError.malformed("invalid literal")`.
The HTTP failure remains unscored. Without the original raw completion, the
error alone cannot identify the offending literal, establish whether generation
was truncated, or distinguish a generation problem from an adapter mismatch.
Record the raw completion and stop reason in a separate controlled diagnostic
before choosing a fix.

An offline comparison found differences between our formatter and the pinned
checkpoint template: extra Pydantic title metadata, and differing representations
of tool-response strings/objects and null values. The checkpoint template
supports both its legacy object form and standard string tool responses, so a
formatting difference alone is not proof of a bug.

## Latency limits the useful conversation

The same native process serializes chat, synthesis, and recognition. In the
airline attempt there were 12 agent chat requests and 21 E4B chat requests for
the caller/helpers. Retail had the same counts. Median caller-synthesis request
wall times were 13.41 seconds for airline and 16.92 seconds for retail; median
agent-stream wall times were 18.89 and 17.74 seconds respectively.

These request measurements include queue waits and overlap. They must not be
added as exclusive computation time. They nevertheless show why a 600-second
conversation permits relatively few useful exchanges. Increasing agent size
does not remove caller or speech latency.

## What the successful check establishes

The [controlled spoken check](gemma4-12b-voice-e2e.md) produced the required
database write, earned 1.0, and rejected a separately regraded wrong write with
0.0. Independent ASR readback confirmed the saved spoken response. It validates
the audio, recognition, native tool, synthesis, and grading path for that request.
Its fixed caller does not validate the generative caller or complex task planning.

A tokenizer comparison on the saved Telecom prompt also found 10,214 Swift
tokens versus 10,210 reference tokens. The differences were confined to policy
heading BOM characters and an emoji/variation selector; numeric identifiers in
that prompt matched. These are pre-existing encoder differences. Their effect
on these task outcomes has not been demonstrated.

## Next diagnostic steps

1. Check caller adherence and termination on a correct refusal using the recorded
   agent responses. Validate a stronger local caller as a separate explicit profile.
2. Compare the same saved ASR/tool history through the native adapter and the
   checkpoint reference, retaining raw output, token counts, and stop reason.
3. Test canonical identifier recovery after an unsuccessful lookup, using only
   information available from the customer's tools.
4. Rerun the same frozen tasks and seed after any correction, with new provenance.
   Preserve the existing scores, frozen inputs, and reward rules.

INT4 quantization, disabled thinking, and model capability are possible factors;
this attempt does not isolate their individual effects. Changing model size alone
cannot diagnose them.

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
The original failed completion was not saved verbatim, so its exact contents
remain unknown. A separate same-weight comparison reproduced a malformed call
from the reconstructed history and isolated a prompt correction. All comparisons
below used the pinned INT4 weights, temperature 0, a 1,024-token output budget,
256-token prefill batches, and `mlx-lm` 0.32.2. They ran locally.

| Reconstructed-history variant | Generated tokens, including stop | Result |
| --- | --- | --- |
| Original Swift prompt and token IDs | 21 | Invalid unquoted `id:L1002` |
| Original Swift prompt, reference tokenizer | 21 | Same invalid literal |
| Checkpoint standard chat template | 23 | Valid quoted string `L1002` |
| Canonical tool schema only | 21 | Same invalid literal |
| Remove schema titles only | 23 | Valid quoted string |
| Preserve tool-result content as text only | 23 | Valid quoted string |
| Canonical schema and text tool results | 23 | Valid quoted string |

The invalid completion was
`<|tool_call>call:get_details_by_id{id:L1002}<tool_call|>`.
The valid completion enclosed `L1002` in the checkpoint's `<|"|>` string
delimiters. Both stopped normally after a short generation. Output-budget
exhaustion and tokenizer differences do not explain this reproduced failure.
The comparison also shows prompt sensitivity; it does not establish that every
malformed call has the same cause.

The checkpoint template supports legacy object responses and standard string
tool-message content. The native 12B adapter previously decoded JSON text into
object fields before rendering it. The correction preserves that content as
text, matching the standard template. The public codec's existing structured
representation remains in use for E4B. This changes neither accepted tool-call
syntax nor the task's tools, prompts, reference actions, or reward rules.

With speech-swift revision `86f500577faa1a8c4c76b623063d65894dc70524`, the actual
Swift 12B adapter parsed the reconstructed history successfully and returned
`get_details_by_id` with `{"id":"L1002"}`. Fourteen focused unit tests and the
existing real-model tool/result-history/cancellation test also passed. This
establishes a fix for the reproduced serialization failure. The next call still
repeats a line lookup; valid syntax alone does not establish successful diagnosis
or task completion. The original Telecom attempt remains unscored.

### Complete voice-task rerun: still unscored

The [separate rerun record](gemma4-12b-telecom-tool-text.json) uses that corrected
runtime with the same frozen Telecom task, profile, seed 42, `control` telephony,
and 600-second deadline. The E4B caller remained generative. It made three valid
calls: customer lookup, then reads of `L1001` and `L1002`. After 3,357 ticks
(167.85 seconds of conversation audio), a subsequent 12B chat request failed
with `malformed("orphan tool call terminator")`. No caller device tool ran.
The trial remains unscored, with its original failure artifacts retained.

This is a different invalid-output condition. The text-preservation correction
does not make the complete task reliable. The raw failed completion was not
captured, so the parser error must not be presented as a verbatim generation.
Further diagnosis needs exact failed request/completion capture and a comparison
against the checkpoint template before changing parsing rules or model settings.

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
2. Capture the exact failed Telecom request and raw completion, then check the
   entire 12B chat-template representation against the checkpoint reference.
3. Test canonical identifier recovery after an unsuccessful lookup, using only
   information available from the customer's tools.
4. Rerun the same frozen tasks and seed after any correction, with new provenance.
   Preserve the existing scores, frozen inputs, and reward rules.

INT4 quantization, disabled thinking, and model capability are possible factors;
this attempt does not isolate their individual effects. Changing model size alone
cannot diagnose them.

# Tool history and caller pacing

The three-task development run exposed two integration defects. These fixes
change the native runtime and experimental caller condition, so subsequent
runs need fresh qualification and separate output. Historical zero rewards
remain preserved development diagnostics.

## Native history

Gemma tool results continue the current model turn. The native formatter failed
to close that turn before opening a following user turn, including after chained
tool calls. The [runtime correction](https://github.com/soniqo/speech-swift/pull/539)
adds that boundary while retaining assistant continuations in the same turn.
The focused native release checks pass, including two different tool-result
markers read by the actual staged 12B model after a new user message.

The exact previously agent-visible airline request was replayed against both
binaries. After two successful lookups, both selected the same flight-status
call. After the garbled repeated user identifier, both falsely claimed that the
account could not be found. The boundary defect is real, but this replay does
not establish it as the sole cause of that model response. These request replays
execute no domain tools and produce no benchmark score.

## Caller pacing

The caller could repeat its answer before hearing a new agent response. In the
airline trace, repeated speech introduced different recognized identifiers while
the agent was working on successful lookups. Non-phone callers now wait after
a completed response and use “Are you still there?” at the existing 15-second
silence threshold. This leaves agent execution time visible.

The first waiting decision still checks for completed or refused work. Only a
validated `CONTINUE` assessment can be reused for identical meaningful
caller-visible input. A new heard turn or customer tool result requires fresh
processing. The original assessment retains its cost, usage and generation
time; a cached wait records no new inference. Terminal acknowledgments and
pending customer tool reports remain covered by regression tests.

The qualification suite adds separate non-phone waiting and check-in cases,
bringing it to 79 checks. The frozen 64 inputs, 278 tasks, domain tools,
policies, model pins and reward rules retain their original contents.

## Validation

- 781 offline tests passed, with lint, frozen-input verification and both replay
  controls passing.
- 27 focused native release tests passed, including the actual 12B tool-result
  marker test.
- Nine HTTP/native functional checks passed with the pinned 12B/E4B models. These
  include caller waiting and completion, bounded malformed-JSON reassessment,
  changed tool-result reads, and the full spoken database-write pipeline.
- The controlled spoken write scored 1.0; changing its successful write in the
  separate replay scored 0.0. This scripted-caller check produces no domain score.
- Fresh qualification passed all 79 caller/judge cases with the updated caller
  source and native runtime, without failing or error-bearing cases.

The [machine-readable validation record](tool-history-and-pacing.json) retains
source identities, model revisions, control outcomes and artifact hashes.

The stack remains Gemma 4 12B INT4 for the agent, E4B INT4 for caller/helpers/judge,
VoxCPM2 FP16 main weights with FP32 AudioVAE, full Whisper Large-v3 CoreML FP16,
and Silero VAD. Qualification and domain trials must match the new runtime and
caller source fingerprints.

## Reproduce a short check

Rebuild the corrected runtime, regenerate its manifest, start the guarded local
service and run fresh qualification as described in [runtime setup](../runtime-setup.md).
Use a new output directory for the focused airline trial:

```bash
uv run --frozen tau-voice run --domain airline --task-id 12 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/local-v2-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 70 --complexity control \
  --max-seconds 300 --wall-timeout 900 \
  --output data/simulations/local-v2-airline-pacing
```

Inspect successful tool results, heard speech, caller decisions and final state.
One shortened trial establishes a development outcome only. Inference failures
and invalid caller control remain unscored; they must not become model failures
or successful stops.

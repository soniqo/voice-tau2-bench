# Native prompt-cache validation

The native Gemma 4 12B runtime can reuse an unchanged system/tool prefix.
Benchmark traces now preserve its optional `cachedPrefixTokens` value as
`cached_prefix_tokens`, and native provenance records the effective cache budget.
Absent metadata remains distinct from a cold request reporting zero.
The [validation record](native-prefill.json) contains source/model pins, artifact
hashes, settings, and the scope of each completed check.

## Verified runtime behavior

On an Apple M5 Pro with 48 GiB unified memory, the native real-weight release
test compared an 8,499-token synthetic prompt and a 15-token greedy tool call:

| Native request | Reused tokens | Wall seconds |
| --- | ---: | ---: |
| Original prefill, first reference | 0 | 10.180 |
| Optimized cold | 0 | 9.352 |
| Optimized warm | 8,448 | 0.778 |
| Original prefill, second reference | 0 | 10.292 |

Prompt tokens, generated tokens, raw output, and tool arguments matched exactly.
A second user request longer than the sliding window and a repeat of the first
request also matched fresh-cache reference generations. Separate actual-model
checks passed for tool-result history, cancellation, and malformed-output capture.
Ten native configuration/prefix unit tests passed, and the release HTTP server
built successfully.

The retained prefix used 557,514,752 bytes (about 532 MiB). The 12B default budget
is 1,024 MiB; E4B retains its 512 MiB default. The shared environment override is
`SPEECH_SWIFT_GEMMA4_PREFIX_CACHE_MB`; `0` disables retention. Only system/tool
state is retained, and each request extends an independent working copy. The
first request still pays its prefill cost. These synthetic timing measurements
do not establish a domain pass rate or guarantee full voice response latency.

## Benchmark validation and remaining work

All 475 offline tests, lint, 64 frozen inputs, and both replay rewards pass.
Transport tests cover absent, zero, positive, and invalid cache counts; invalid
metadata remains an unscored error. An opt-in actual-model HTTP test checks cold
and warm replies and trace metadata.

The complete voice service now reaches HTTP readiness under the unchanged
24 GiB physical-footprint and 2 GiB free-disk guards. Two earlier startup attempts
hit the disk guard; package-cache and archived-compiler-cache cleanup resolved
that infrastructure limit. Pinned model files, the preserved runtime binary,
repository work in progress, and the complete benchmark backup verified unchanged.

Against the new binary, the actual-model HTTP cache test passed (1.54 seconds),
and caller/judge qualification passed **19/19** checks. The controlled full spoken
pipeline also passed (62.50 seconds including reference synthesis and assertions):
the requested database write scored **1.0**, and a replay with the successful
write's title changed scored **0.0**. The call contained 910 ticks, or 45.5 simulated
audio seconds, and ended with `user_stop`. Two native cancellations matched
recorded agent cancellation intents; there were no unexplained inference failures.

This functional check substitutes fixed spoken customer input and contributes
no domain score. It exercises actual VoxCPM2 speech, telephony conversion, full
Whisper Large-v3, Gemma 4 12B, tools, database grading, and streamed agent speech.
The 18-call held-out pilot remains unrun; historical domain scores remain unchanged.

## Airline diagnostic

Gemma 4 12B completed airline task `0`, seed `42`, with **reward 1.0** and
`user_stop`. The generative customer was used under the unchanged local v2
15-second caller wait, clean `control` telephony, a 600-simulated-second cap,
and a 1,200-wall-second watchdog. The call lasted **498.75 simulated seconds**;
the runner took **1,095.61 wall seconds** (18.3 minutes).

This is a refusal case: the customer does not want cancellation without a
refund, and the expected database stays unchanged. This frozen task has **no
communication assertions**. Its reward therefore verifies the expected database
outcome and normal termination; it does not independently verify the spoken
policy explanation. The controlled mock test above supplies the separate
positive/negative check for a real database write.

The agent first looked up `EH-GLP-3`, as transcribed by ASR, and received an error.
It then used `EHGLP3` successfully. These were its only tool calls; neither
modified the database. It recovered the identifier from the conversation without
reference actions or task answers being inserted into its input.

Of 21 successful agent chat requests, **20 reused 4,608 prefix tokens**. The cold
chat took 8.67 seconds; warm chats took 3.90–6.28 seconds, including conversation
prefill and generation. ASR took 0.82–1.77 seconds per utterance. There were no
unexplained failed requests, native chat failures, or unexpected cancellations.
The call demonstrates a passing domain diagnostic, with substantial conversation
and customer-computation overhead still visible in the timing records.

Across the post-cleanup service session, peak physical footprint was **16.89 GiB**
on the 48 GiB machine, and minimum free disk space was **7.08 GiB**. Neither
resource guard triggered. The test service was stopped after validation.

One known diagnostic task is not an overall model score. Retail and telecom have
not been rerun under this condition, and the predeclared held-out pilot remains
unrun. Do not combine this result with the historical v1 slice.

## Reproduction

The native tests use the pinned 12B INT4 checkpoint at revision
`73bcf09092aa277861d5a191b989b666f7f32e8f`, release builds, and compiled MLX Metal
shaders. See the runtime's
[prefill report](https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/gemma4-prefill.md)
for the native test command and measurement limits.

After starting the complete staged service, run the new HTTP check:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v2.json \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k native_prefix_reuse
```

Run the full spoken check with a fresh output directory:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v2.json \
TAU_VOICE_FULL_E2E=1 \
TAU_VOICE_E2E_OUTPUT=data/simulations/prefix-cache-spoken-check \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k full_voice_task
```

The 19-check qualification suite and domain commands are documented in
[local v2 execution](../local-v2.md). Use a fresh qualification report matching
the current benchmark source and native provenance. Keep the same 15-second
caller wait and count measured agent execution time normally.

The completed airline diagnostic used:

```bash
uv run --frozen tau-voice run --domain airline --task-id 0 \
  --profile configs/voice/soniqo-local-gemma12b-v2.json \
  --qualification data/simulations/prefix-cache-qualification.json \
  --base-url http://127.0.0.1:8787 --seed 42 \
  --max-seconds 600 --wall-timeout 1200 --complexity control \
  --output data/simulations/prefix-cache-airline-0
```

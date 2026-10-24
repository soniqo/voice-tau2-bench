# Native prompt-cache validation

The native Gemma 4 12B runtime can reuse an unchanged system/tool prefix.
Benchmark traces now preserve its optional `cachedPrefixTokens` value as
`cached_prefix_tokens`, and native provenance records the effective cache budget.
Absent metadata remains distinct from a cold request reporting zero.

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

The complete voice service has not yet reached HTTP readiness with this binary.
Both startup attempts stopped at the local resource guard when free disk space
fell below 2 GiB, before any qualification or domain request. The second attempt
followed cleanup of this task's rebuildable files. These are infrastructure
failures, not model failures or scored tasks. The existing
[spoken pipeline result](local-v2.md) used the earlier native binary.

Caller/judge qualification, the actual-model HTTP cache test, and the airline
rerun must be repeated after enough local disk space is available. The 18-call
held-out pilot remains unrun; historical domain scores remain unchanged.

## Reproduction

The native tests use the pinned 12B INT4 checkpoint at revision
`73bcf09092aa277861d5a191b989b666f7f32e8f`, release builds, and compiled MLX Metal
shaders. See the runtime's
[prefill report](https://github.com/soniqo/speech-swift/blob/perf/gemma4-prefill/docs/benchmarks/gemma4-prefill.md)
for the native test command and measurement limits.

After starting the complete staged service, run the new HTTP check:

```bash
SPEECH_SWIFT_INFERENCE_URL=http://127.0.0.1:8787 \
TAU_VOICE_PROFILE=configs/voice/soniqo-local-gemma12b-v2.json \
uv run --frozen pytest -q tests/test_voice_local/test_real_models.py \
  -k native_prefix_reuse
```

Then run the 19-check qualification suite and domain commands documented in
[local v2 execution](../local-v2.md). Use a fresh qualification report matching
the current benchmark source and native provenance. Keep the same 15-second
caller wait and count measured agent execution time normally.

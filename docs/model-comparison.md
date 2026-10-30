# Comparing voice agents

Compare complete spoken task execution: can the agent hear the caller, use the
domain tools correctly, and reach the required result? The frozen evaluator
checks final state and selected communication requirements. See [scoring](evaluation.md).

## Available agents and adapters

Local inference uses the loopback speech-swift service. The optional
[Nova 2.5 Sonic runner](nova.md) uses native cloud audio and tools, with the
caller/listener/judge and caller speech still running locally.
Gemma 4 12B IT INT4 has a tested v2 profile. Gemma 4 E4B IT INT4 is supported by the
runtime and has a historical v1 profile; an equivalent v2 agent profile and a
matched run are still needed before comparing it with the current 12B result.

The following experimental [optional adapters](providers.md) are implemented
and tested offline through the production runner, real tools and frozen grader.
Real-provider validation and domain scores are pending:

| System | What a comparison must identify |
| --- | --- |
| [GPT-Live](https://developers.openai.com/api/docs/guides/live) | Voice model **and** delegated reasoning/tool backend |
| [OpenAI Realtime](https://developers.openai.com/api/docs/guides/realtime) | Speech-to-speech model, tools, turn detection and audio settings |
| [Gemini Live](https://ai.google.dev/gemini-api/docs/live-api) | Live model, tool handling and audio settings |
| [ElevenLabs Agents](https://elevenlabs.io/docs/eleven-agents/libraries/web-sockets) | Agent configuration, underlying LLM, voice and client tools |

GPT-Live separates spoken conversation from delegated task execution. This
adapter uses a named Responses backend and timestamped reflected audio with a
declared experimental noise gate. A local backend is not implemented here.
Realtime and Gemini Live can handle audio and tool calls within their voice
sessions. ElevenLabs Agents is a configured agent system rather than a TTS model
alone. A model supporting speech does not by itself make it a benchmark adapter.

## Metrics and definitions

Always publish trial counts alongside rates, including incomplete coverage and
inference failures. A timeout receives the frozen zero reward. An inference or
invalid-caller exception remains unscored and must stay visible in coverage.

| Metric | Definition | Current evidence |
| --- | --- | --- |
| Task success, pass^1 | Average trial success under the frozen task reward | Trial reward and batch summary |
| Coverage | Scored trials / planned trials; unscored errors separately | Batch summary or explicit diagnostic count |
| Normal completion | Calls ending through ordinary caller/agent stop / attempted calls | `termination_reason` |
| LLM latency p50 / p95 | Successful **agent** chat request `execution_seconds` | `simulation.info.inference_accounting` |
| ASR latency p50 / p95 | Successful **agent** ASR request `execution_seconds` | Same accounting records |
| TTS first audio p50 / p95 | Speech request start to first returned audio chunk | Successful agent `stream_speech` events in `inference.json` |
| Voice response delay p50 / p95 | Caller end of speech to first agent audio | Standardized aggregate pending; inspect saved ticks |
| Tool error rate | Failed or rejected agent calls / all agent attempts | `simulation.info.tool_metrics.assistant` |
| Completion time | Simulated audio duration and measured trial wall duration | Ticks and `simulation.duration` |
| Peak memory | Maximum sampled physical footprint of the native process | Guard's `memory.jsonl`; state sampling interval and measured phase |

For percentile values, filter by actor and operation, retain successful requests,
sort the samples, and linearly interpolate at index `(n - 1) * p`. Publish the
number of samples. Keep queue time separate from execution time; do not silently
drop cancelled or failed requests from failure reporting.
Report cache state and budget too: the current LLM sample contains one cold
request and sixteen requests with a reused system/tool prefix, with a 1 GiB
native prefix-cache budget. Functional checks and qualification preceded the call.

TTS first-audio latency starts at the synthesis request. End-to-end voice response
delay starts when the caller stops speaking and includes recognition, reasoning,
tools, queuing and synthesis. The saved tick trajectory supports investigation,
but a standardized aggregate for that response delay is not currently exported.
Do not label component latency as whole-agent voice response latency.

A successful tool execution can still produce the wrong business result. Tool
error rate therefore complements task success. Audio quality, WER and MOS are
not automatically included in this task benchmark's reward.

Memory measurements cover the local service, including its loaded supporting
models. A full-validation-session peak also includes qualification and functional
checks. Cloud client memory is not the provider's model memory and must be
labelled separately.

## Matched comparison protocol

1. Declare task IDs, trial seeds, acoustics and limits before execution. Use the
   same state-changing development subset for each model before expanding.
2. Hold caller, listener, judge and interaction settings fixed. For local LLM
   comparisons, keep ASR, TTS, VAD and hardware fixed too.
3. Qualify the caller/judge against the current source and runtime. Audit caller
   adherence on domain traces; qualification does not establish perfect fidelity.
4. Run sequentially, preserve errors and report coverage. Do not retry a failed
   trial silently or replace the agent's heard audio with gold text.
5. Report pass^1 first, then latency, tool errors, completion time and memory.
   Publish source/model revisions and the complete stack.

The [local 12B record](results/gemma4-12b-local-v2-airline.json) is a single
12B development call. It ended normally and failed its final-state check. It is
not a matched comparison with the older E4B or 12B v1 runs. Their historical
records remain in `docs/results/` under their original settings.

Nova 2.5 Sonic's [post-fix Retail record](results/nova25-v6-retail-development.json)
passed **1/1** development trials with **0/5** tool errors and a normal caller
stop. It used real-time v6, seed 70, control acoustics
and a 420-second cap. Its 296.30-second wall duration and 259.0-second audio
duration measure the complete call, including local caller work. This one-task
result establishes a working state-changing voice call; matched multi-model
coverage remains pending.

## Execution limits

Current batch controls are `--num-tasks` or an explicit `--task-plan`, `--trials`,
`--max-seconds` and `--wall-timeout`. The v2 clock distinguishes simulated call
seconds from a wall watchdog. The watchdog is checked between ticks; an in-flight
request can delay it. These controls bound execution without guaranteeing a
provider-dollar ceiling. Local inference stays local. The explicit optional
[Nova runner](nova.md) now supports complete domain calls with real-time pacing,
tool execution, persistent spending reservations and usage accounting. Its
[functional controls](validation/nova25-pilot.md) are separate from domain scores.

The [new provider runner](providers.md) requires `--allow-cloud`, bounded
sessions and persistent reservations. It closes the voice connections and
accounts for delegated work. Gemini session-final billing and ElevenLabs USD
totals remain unknown; their reservations are retained. Offline controls are
separate from real model scores and do not establish provider latency or quality.

The provider runner records `prepared-opener-v1`: local greeting synthesis
finishes before its conversation watchdog and provider connection start.
`wall_seconds` still includes that work; `setup_seconds` and
`call_wall_seconds` separate preparation from the conversation. Match this
setup condition when comparing runs. Earlier inline-greeting results and
their rewards remain unchanged.

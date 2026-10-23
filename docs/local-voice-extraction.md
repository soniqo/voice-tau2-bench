# Standalone local τ-Voice

The `soniqo-local-v1` profile keeps the English `voice-v1` customer-service frame at upstream `05736c27219b769d0be507574dd3135010916644`: 50 airline, 114 retail, and 114 telecom tasks. `tau-voice verify` checks 64 frozen inputs, including domain source, tools, tasks, policies, databases, and the English pack. Neither the extraction nor model tuning rewrites those pins.

## Runtime

Python owns simulation, tools, the caller, effects, artifacts, and grading. One persistent native speech-swift process loads the models once. Its shared scheduler serializes chat, synthesis, and transcription; CoreML VAD keeps independent state for each stream.

| Role | Native model | Precision |
| --- | --- | --- |
| Agent, caller, listening helpers, semantic judge | Gemma 4 E4B IT | INT4 |
| Agent and caller speech | VoxCPM2 | FP16 main parameters, FP32 AudioVAE; mixed arithmetic |
| Agent recognition | Full Whisper Large-v3 CoreML | FP16 bundle |
| Speech detection | Silero VAD CoreML | Model's native precision |

The native server requires full tool declarations, native tool/result history, structured output, streaming speech, and cancellation. Profile checks reject substitutions such as Whisper Turbo or VoxCPM2 INT8/BF16 main parameters. The native provenance endpoint exposes actual parameter dtype counts and memory counters in addition to staged asset pins.

`prepare_speech_swift_models.py` downloads immutable revisions, verifies LFS SHA-256 or Git blob hashes, and records every asset. It fingerprints the native working source, including uncommitted files. Rebuild and regenerate the manifest together after native source changes. The server starts offline and loads VoxCPM2's tokenizer from the same staged folder without changing it.

The guarded launcher samples macOS physical footprint, including GPU allocations, and stops its own child above 24 GiB or below 2 GiB free disk. The service separately caps the MLX allocator at 20 GiB and unused cache at 128 MiB. CoreML memory contributes to process footprint. Run one model service and one benchmark at a time.

## Conversation and scoring

Each 50 ms simulation tick exchanges 8 kHz μ-law audio. The cascaded agent reads actual incoming audio, never the caller's gold text or speech labels. Stateful resampling feeds Silero; endpointed audio goes to full Large-v3. Gemma receives its recognized text plus domain policy and real tool results. VoxCPM2 streams the reply back through the stateful telephony converter. Barge-in cancels reply generation and discards stale audio and tool events. Recognition already in progress finishes and its transcript is retained across interruptions; recognition errors still propagate.

The caller retains the upstream scenario prompt, persona machinery, turn taking, and tool interface. Chat/TTS/ASR are explicitly bound for the entire simulation and evaluation. Missing backends and inference failures raise errors. A generation made stale by newly heard speech is discarded before it can be spoken.

The evaluator derives the target database by replaying reference actions on a fresh environment and compares it with actual tool effects. Communication facts use the preserved semantic prompt with the local Gemma judge. Timeout and premature termination receive zero. See [evaluation](evaluation.md).

## Runs, resume, and artifacts

`tau-voice run` selects one frozen task. `tau-voice batch` runs the full selected frame sequentially, optionally limiting each domain or repeating trials. Trial seeds are base seed plus trial index; chat uses temperature zero. The default deadline is 600 seconds per task.

Each task saves `runtime.json`, `profile.json`, `provenance.json`, `settings.json`, `simulation.json`, `inference.json`, `agent.wav`, and `user.wav`. Provenance fingerprints the Python working source, dependencies, profile, frozen inputs, native source, and model assets. Errors during simulation or grading save `failed-run.json` and propagate without becoming a scored failure.

Batch checkpoints publish atomically. `--resume` accepts only an identical task plan, settings, source, and model configuration, and verifies hashes of completed artifacts. It preserves an interrupted attempt under `interrupted/` before retrying that unscored trial. Process IDs and current memory counters may change across an otherwise identical server restart. Summary pass^k uses only tasks with all requested trials completed.

## Verification and limits

Offline tests cover domain tools, audio processing, streaming, frozen inputs, provider isolation, and batch recovery. Two synthetic replays match traces captured against the unmodified upstream commit, including a successful task write and the upstream notification case whose database reward is zero. They measure regression parity, not model quality.

Actual-model checks cover native Gemma calls/results, VoxCPM2-to-Large-v3 recognition, VAD, cancellation, semantic judging, repeated context memory, released VAD sessions, and streamed/buffered speech agreement. Development runs include a complete real-model mock task with reward 1 and a later completed task with reward 0 after a spoken identifier was transcribed with different capitalization. These are functional checks, not a 278-task quality estimate. The [validation record](validation/local-voice-smoke.json) documents the final smoke run separately. The final run completed one task per domain without runtime errors, but all three reached the 180-second deadline and scored zero. A fresh process verified and reused all completed trials through --resume. Component validation passed 394 offline tests and seven actual-native-model checks. Peak sampled native-process physical footprint for the service session was 10.62 GiB on the 48 GB test machine. No full-frame quality estimate is established.

The local caller, voices, listening helpers, and judge change the experiment. The default clean `control` acoustic profile also differs from noisy settings. Agent spoken-text alignment is estimated while speech is still generating. Heavy model requests share one queue, and long speech generation can be slower than real time. Reports are explicitly marked as a custom local profile and do not establish upstream latency or leaderboard comparability.

Text execution, cloud inference integrations, retrieval/intake domains, authoring tools, and the upstream web leaderboard are removed. Pure historical configuration shapes remain where shared trajectory models need them; they provide no cloud execution path.

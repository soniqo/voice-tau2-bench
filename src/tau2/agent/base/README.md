# Voice participant building blocks

The voice benchmark exchanges caller and agent chunks through `FullDuplexParticipant.get_next_chunk()` every 50 ms. Tool calls are atomic, with results delivered on the following tick.

| Module | Responsibility |
| --- | --- |
| `participant.py` | Participant lifecycle, seed handling, and full-duplex protocol |
| `streaming.py` | Tick history, pending input, turn-taking state, and chunk buffers |
| `voice.py` | Runtime synthesis/transcription binding and voice state |
| `llm_config.py` | Model identity and generation arguments |

`SpeechSwiftAgent` extends `FullDuplexAgent`, and its `CascadeState` extends `StreamingState`. It recognizes actual caller audio, sends tool-aware conversation history to the local model, and streams synthesized replies. Caller simulation uses `VoiceStreamingUserSimulator` and the preserved turn-taking and effects machinery.

Keep assistant tool calls and results linked by ID. A message contains text or tool calls. Preserve only delivered outgoing speech in history; retain incoming recognition across reply interruptions. Cancel stale replies and surface recognition errors.

See the [local runtime architecture](../../../../docs/local-voice-extraction.md), [runtime setup](../../../../docs/runtime-setup.md), and [evaluation rules](../../../../docs/evaluation.md). Tests for the local transport and agent are under `tests/test_voice_local/`; shared conversation tests are under `tests/test_streaming/`.

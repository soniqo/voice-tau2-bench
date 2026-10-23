# Voice runtime instructions

The local entry point is cli.py, task execution is local_run.py, and sequential trials live in batch.py. speech_swift.py accepts loopback endpoints only. Chat, synthesis, transcription, and semantic evaluation use explicit run-scoped backends; errors propagate without cloud fallback or replacement gold text.

Native TTS returns 48 kHz mono PCM16. Normalize caller synthesis to 16 kHz before the preserved effects pipeline. The conversation exchanges 50 ms frames of 8 kHz mono μ-law. Preserve stateful resampling and telephony conversion, resetting state on interruption.

Keep effect order: speech effects, background noise, burst noise, telephony conversion, frame drops. Regular acoustics require the requested audio assets. Provider-specific vocal tags do not imply equivalent VoxCPM2 behavior.

Record model assets and working source, not Git HEAD alone. Capability checks must reject precision/model substitutions. Use bounded, sequential native inference, independent VAD streams, and release request/stream state after use.

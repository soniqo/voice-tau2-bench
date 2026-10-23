# Agent instructions

SpeechSwiftAgent extends FullDuplexAgent and CascadeState extends StreamingState. Constructor arguments include tools, domain_policy, and an explicit backend. Keep tool calls atomic and deliver their results through the orchestrator.

The agent reads actual caller audio only. Never consult gold text, audio scripts, private scenarios, or contains_speech labels. Preserve cancellation epochs so interrupted inference cannot emit stale audio or tools. Store only delivered speech in conversation history and preserve native tool call/result IDs.

Use Pydantic state models. Messages contain text or tool_calls, not both. Test audio frame validation, endpointing, interruption, failure propagation, tool flow, and history accounting under tests/test_voice_local.

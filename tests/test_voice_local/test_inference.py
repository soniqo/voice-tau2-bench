import pytest
from pydantic import BaseModel, ValidationError

from tau2.agent.base.streaming import evaluate_listener_reactions
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import SpeechEffectsConfig, UserSpeechInsert
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.voice import (
    ElevenLabsTTSConfig,
    RuntimeTranscriptionConfig,
    RuntimeTTSConfig,
    SynthesisConfig,
    TranscriptionConfig,
    TranscriptionResult,
    VoiceSettings,
)
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.inference import (
    InferenceBackends,
    generate,
    get_inference_backends,
    use_inference_backends,
)
from tau2.user.user_simulator_streaming import (
    UserStreamingState,
    VoiceStreamingUserSimulator,
    user_interruption_policy,
)
from tau2.voice.synthesis.audio_effects.speech_generator import OutOfTurnSpeechGenerator
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.transcription.transcribe import transcribe_audio


class Backend:
    def __init__(self):
        self.calls = []
        self.reply = "I need help with a reservation."

    def generate(self, model, messages, **kwargs):
        assert model == "local-gemma", (
            "unexpected model must not fall back to a provider"
        )
        self.calls.append(("chat", kwargs.get("call_name"), messages))
        return AssistantMessage(role="assistant", content=self.reply, cost=0.0)

    def synthesize(self, text, config):
        assert get_inference_backends() is not None
        self.calls.append(("tts", config.voice_id, text))
        return AudioData(
            data=b"\0\0" * 4800,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=48000),
        )

    def transcribe(self, audio, config):
        self.calls.append(("asr", config.model, audio))
        return TranscriptionResult(transcript="The local transcript.")


@pytest.fixture
def backend():
    return Backend()


def bind(backend):
    return use_inference_backends(InferenceBackends(backend, backend, backend))


def test_caller_uses_injected_gemma_and_native_voice_without_provider_voice_id(backend):
    settings = VoiceSettings(
        synthesis_config=SynthesisConfig(
            provider="runtime",
            provider_config=RuntimeTTSConfig(
                model_id="voxcpm2", voice_id="local-caller"
            ),
            speech_effects_config=SpeechEffectsConfig(
                enable_vocal_tics=False,
                enable_non_directed_phrases=False,
                enable_dynamic_muffling=False,
            ),
        ),
        transcription_config=None,
    )
    caller = VoiceStreamingUserSimulator(
        tools=[],
        instructions="Ask about your reservation.",
        llm="local-gemma",
        llm_args={},
        voice_settings=settings,
        decision_model="local-gemma",
        decision_model_args={},
    )
    with bind(backend):
        state = caller.get_init_state()
        message, _ = caller._generate_full_duplex_voice_message(None, state)
    assert message.is_audio
    assert message.audio_format.sample_rate == 16000
    assert message.audio_script_gold == backend.reply
    assert [(kind, name) for kind, name, _ in backend.calls] == [
        ("chat", "user_streaming_response"),
        ("tts", "local-caller"),
    ]
    assert get_inference_backends() is None


def test_runtime_asr_receives_original_telephony_audio(backend):
    audio = AudioData(
        data=b"\x7f" * 400,
        format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    with bind(backend):
        result = transcribe_audio(
            audio, RuntimeTranscriptionConfig(model="whisper-large-v3")
        )
    assert result.transcript == "The local transcript."
    assert backend.calls[0][2] is audio


def test_out_of_turn_tts_threads_inherit_the_runtime(backend):
    generator = OutOfTurnSpeechGenerator(
        voice_id="local-caller",
        provider="runtime",
        provider_config=RuntimeTTSConfig(model_id="voxcpm2", voice_id="local-caller"),
    )
    items = [
        UserSpeechInsert(text="One moment.", type="non_directed_phrase"),
        UserSpeechInsert(text="Hang on.", type="non_directed_phrase"),
    ]
    with bind(backend):
        generator.generate_all(items)
    assert set(generator.generated_audio) == {item.text for item in items}
    assert len(backend.calls) == 2


def test_bound_run_rejects_legacy_speech_calls(backend):
    audio = AudioData(
        data=b"\0\0",
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
    )
    with bind(backend):
        with pytest.raises(ValueError, match="legacy synthesis"):
            synthesize_voice("hello", "elevenlabs", ElevenLabsTTSConfig())
        with pytest.raises(ValueError, match="legacy transcription"):
            transcribe_audio(audio, TranscriptionConfig())
    assert backend.calls == []


def test_runtime_calls_require_explicit_binding():
    with pytest.raises(RuntimeError, match="injected inference backends"):
        synthesize_voice(
            "hello", "runtime", RuntimeTTSConfig(model_id="voxcpm2", voice_id="caller")
        )


def test_runtime_config_roundtrip_and_provider_mismatch():
    config = SynthesisConfig(
        provider="runtime",
        provider_config=RuntimeTTSConfig(model_id="voxcpm2", voice_id="caller"),
    )
    restored = SynthesisConfig.model_validate_json(config.model_dump_json())
    assert isinstance(restored.provider_config, RuntimeTTSConfig)
    with pytest.raises(ValidationError):
        SynthesisConfig(provider="runtime", provider_config=ElevenLabsTTSConfig())


def test_local_judge_uses_the_same_semantic_evaluator(backend):
    backend.reply = '{"reasoning":"The refund meaning was conveyed.","met":true}'
    with bind(backend):
        checks = CommunicateEvaluator.evaluate_communicate_info(
            [
                AssistantMessage(
                    role="assistant", content="You will get twenty dollars back."
                )
            ],
            ["refund is $20"],
            scoring=CommunicateScoring(
                judge=SemanticCommunicateSettings(model="local-gemma", model_args={})
            ),
        )
    assert checks[0].met is True
    assert backend.calls[0][1] == "communicate_info_eval"


def test_local_structured_reply_validation_still_applies(backend):
    class Reply(BaseModel):
        met: bool

    backend.reply = '{"met":true}'
    with bind(backend):
        reply = generate(
            "local-gemma",
            [UserMessage(role="user", content="Check")],
            response_model=Reply,
        )
    assert Reply.model_validate_json(reply.content).met is True


def test_listener_uses_explicit_local_model_and_surfaces_configuration_errors(backend):
    state = UserStreamingState(system_messages=[], messages=[])
    state.input_turn_taking_buffer = [
        AssistantMessage(
            role="assistant",
            content="Would you like to change the flight?",
            contains_speech=True,
        )
    ]
    backend.reply = "YES"
    with bind(backend):
        decision = user_interruption_policy(
            state, decision_model="local-gemma", decision_model_args={}
        )
        assert decision.decision is True
        with pytest.raises(AssertionError, match="unexpected model"):
            user_interruption_policy(
                state, decision_model="unconfigured", decision_model_args={}
            )
    assert backend.calls[0][1] == "interruption_decision"


def test_parallel_listener_failures_do_not_silently_disable_local_interruptions(
    backend,
):
    state = UserStreamingState(system_messages=[], messages=[])

    def failed_listener(_state):
        raise RuntimeError("Local listener configuration failed")

    with (
        bind(backend),
        pytest.raises(RuntimeError, match="Local listener configuration"),
    ):
        evaluate_listener_reactions(state, failed_listener, None)

"""Opt-in checks against a server loaded with the actual pinned local models."""

import hashlib
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pytest
from jiwer import wer

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.voice import RuntimeTranscriptionConfig, RuntimeTTSConfig
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.inference import InferenceBackends, use_inference_backends
from tau2.voice.runtime_profile import load_profile
from tau2.voice.speech_swift import SpeechSwiftBackend, SpeechSwiftError, pcm16

pytestmark = pytest.mark.local_models


@pytest.fixture(scope="module")
def runtime():
    url = os.environ.get("SPEECH_SWIFT_INFERENCE_URL")
    if not url:
        pytest.skip(
            "Set SPEECH_SWIFT_INFERENCE_URL to the native server with staged models"
        )
    backend = SpeechSwiftBackend(url)
    profile_path = os.environ.get("TAU_VOICE_PROFILE")
    profile = load_profile(Path(profile_path) if profile_path else None)
    profile.check_runtime(backend.capabilities())
    native = backend.provenance()
    tokenizer = Path(native["configuration"]["voxDirectory"]) / "tokenizer_config.json"
    pinned = next(
        entry["sha256"]
        for entry in native["configuration"]["assets"]["vox"]["files"]
        if entry["path"] == "tokenizer_config.json"
    )
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == pinned
    yield backend
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == pinned
    backend.close()


@pytest.mark.parametrize("model", ["gemma4-e4b-it", "gemma4-12b-it"])
def test_real_gemma_native_tool_call_and_result_history(runtime, model):
    if model not in runtime.capabilities().models:
        pytest.skip(f"{model} is not loaded in this model service")

    def record_note(text: str) -> str:
        """Record the exact text of a note.

        Args:
            text: The note's exact text.
        """
        return "recorded"

    messages = [
        SystemMessage(
            role="system",
            content="Use record_note when asked to save a note. After a successful result, say that the note was recorded.",
        ),
        UserMessage(
            role="user", content="Save a note with the exact text: launch review"
        ),
    ]
    reply = runtime.generate(
        model,
        messages,
        tools=[Tool(record_note)],
        temperature=0,
        max_tokens=512,
    )
    assert reply.tool_calls and reply.tool_calls[0].name == "record_note"
    assert reply.tool_calls[0].arguments == {"text": "launch review"}
    result = runtime.generate(
        model,
        [
            *messages,
            reply,
            ToolMessage(
                role="tool",
                id=reply.tool_calls[0].id,
                requestor="assistant",
                content='{"status":"recorded"}',
            ),
        ],
        tools=[Tool(record_note)],
        temperature=0,
        max_tokens=512,
    )
    assert not result.tool_calls
    assert "record" in result.content.lower()


def test_real_fp16_speech_full_large_v3_and_stateful_vad(runtime):
    text = "Hello. Your appointment is confirmed for tomorrow morning."
    audio = runtime.synthesize(
        text,
        RuntimeTTSConfig(
            model_id="voxcpm2", voice_id="A clear adult female voice speaking English."
        ),
    )
    assert audio.format.sample_rate == 48000
    assert 1 < audio.duration < 15
    result = runtime.transcribe(
        audio, RuntimeTranscriptionConfig(model="whisper-large-v3")
    )

    def normalize(value):
        return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()

    assert wer(normalize(text), normalize(result.transcript)) <= 0.25, result.transcript
    audio16 = pcm16(audio, 16000)
    framed = audio16.data[: len(audio16.data) // 1024 * 1024]
    probabilities = runtime.vad(framed, uuid.uuid4().hex, reset=True)
    assert len(probabilities) == len(framed) // 1024
    assert max(probabilities) > 0.5


def test_real_streaming_tts_cancels_after_first_audio(runtime):
    request_id = uuid.uuid4().hex
    chunks = runtime.stream_speech(
        "Please listen to this longer message. " * 50,
        request_id=request_id,
        voice="A clear adult female voice speaking English.",
    )
    first = next(chunks)
    assert first.duration > 0
    runtime.cancel(request_id)
    with pytest.raises(SpeechSwiftError):
        list(chunks)


def test_real_local_semantic_judge_uses_the_preserved_prompt(runtime):
    scoring = CommunicateScoring(
        scorer="semantic",
        judge=SemanticCommunicateSettings(
            model="gemma4-e4b-it", model_args={"temperature": 0, "max_tokens": 1024}
        ),
    )
    with use_inference_backends(
        InferenceBackends(chat=runtime, synthesis=runtime, transcription=runtime)
    ):
        checks = CommunicateEvaluator.evaluate_communicate_info(
            [
                AssistantMessage(
                    role="assistant",
                    content="Your refund of twenty-five dollars will arrive tomorrow.",
                )
            ],
            ["The refund amount is $25", "The refund amount is $500"],
            scoring=scoring,
        )
    assert [check.met for check in checks] == [True, False]


def test_real_long_context_releases_request_buffers(runtime):
    for _ in range(3):
        reply = runtime.generate(
            "gemma4-e4b-it",
            [
                SystemMessage(
                    role="system",
                    content="Read the context, then answer with the single word READY.",
                ),
                UserMessage(
                    role="user",
                    content="Context: "
                    + "A customer called about an order and requested a refund. " * 300,
                ),
            ],
            temperature=0,
            max_tokens=16,
        )
        assert reply.content.strip()
        memory = runtime.client.get("/v1/inference/provenance").json()
        assert memory["mlx_cache_bytes"] < 256 * 1024**2
        assert memory["mlx_active_bytes"] < 16 * 1024**3


def test_real_vad_streams_are_released_between_tasks(runtime):
    for _ in range(66):
        assert len(runtime.vad(b"\x00" * 1024, uuid.uuid4().hex, end_stream=True)) == 1


def test_real_incremental_speech_matches_buffered_speech(runtime):
    text = "Your appointment is confirmed for tomorrow morning. Please bring your identification to the front desk."
    voice = "A clear adult female voice speaking English."
    buffered = runtime.synthesize(
        text, RuntimeTTSConfig(model_id="voxcpm2", voice_id=voice, seed=42)
    )
    chunks = list(runtime.stream_speech(text, request_id=uuid.uuid4().hex, voice=voice))
    streamed = b"".join(chunk.data for chunk in chunks)
    assert len(streamed) == len(buffered.data)
    delta = np.frombuffer(streamed, dtype="<i2").astype(np.float32) - np.frombuffer(
        buffered.data, dtype="<i2"
    )
    assert np.sqrt(np.mean(delta**2)) < 32


def test_real_full_voice_task_writes_database_and_grader_rejects_wrong_write(
    runtime, tmp_path
):
    """Opt-in full conversation plus a negative control for database grading."""
    if os.environ.get("TAU_VOICE_FULL_E2E") != "1":
        pytest.skip("Set TAU_VOICE_FULL_E2E=1 to run a complete spoken task")
    import json

    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode
    from tau2.voice.local_run import run_local_task, task_context

    profile_path = os.environ.get("TAU_VOICE_PROFILE")
    output = Path(os.environ.get("TAU_VOICE_E2E_OUTPUT", str(tmp_path / "voice-task")))
    result = run_local_task(
        domain="mock",
        task_id="create_task_1",
        base_url=str(runtime.client.base_url).rstrip("/"),
        output=output,
        seed=42,
        max_seconds=600,
        complexity="control",
        profile_path=Path(profile_path) if profile_path else None,
    )
    assert result["reward"] == 1, result
    simulation = SimulationRun.model_validate_json(
        (output / "simulation.json").read_text()
    )
    assert simulation.reward_info.db_check.db_match is True
    assert simulation.info["local_metrics"]["asr"]
    assert all(
        (output / f"{role}.wav").stat().st_size > 44 for role in ("agent", "user")
    )
    calls = [call for tick in simulation.ticks for call in tick.agent_tool_calls]
    assert any(
        call.name == "create_task" and call.arguments["title"] == "Important Meeting"
        for call in calls
    )
    events = json.loads((output / "inference.json").read_text())
    expected_model = (
        load_profile(Path(profile_path) if profile_path else None)
        .roles["agent"]
        .model_id
    )
    assert any(
        row["operation"] == "chat" and row["model"] == expected_model for row in events
    )
    assert {"chat", "speech", "stream_speech", "transcribe", "vad"} <= {
        row["operation"] for row in events
    }
    wrong = simulation.model_copy(deep=True)
    changed = next(
        call
        for tick in wrong.ticks
        for call in tick.agent_tool_calls
        if call.name == "create_task"
    )
    changed.arguments["title"] = "Deliberately wrong title"
    task, _, _ = task_context("mock", "create_task_1", 42)
    grading = evaluate_simulation(
        simulation=wrong,
        task=task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=False,
        domain="mock",
        mode=CommunicationMode.FULL_DUPLEX,
        strict_replay=False,
        communicate_scoring=CommunicateScoring(
            scorer="semantic", judge=SemanticCommunicateSettings(model="gemma4-e4b-it")
        ),
    )
    assert grading.reward == 0
    assert grading.db_check.db_match is False
    (output / "negative-control.json").write_text(
        grading.model_dump_json(indent=2) + "\n"
    )

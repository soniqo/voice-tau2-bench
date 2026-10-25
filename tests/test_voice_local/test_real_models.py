"""Opt-in checks against a server loaded with the actual pinned local models."""

import audioop
import hashlib
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from jiwer import wer

from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
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
from tau2.user.user_simulator_base import FullDuplexUser
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


def test_real_native_prefix_reuse_is_reported_without_changing_the_reply(runtime):
    model = runtime.capabilities().models.get("gemma4-12b-it")
    if model is None or "system_prefix_cache" not in model.features:
        pytest.skip("The native 12B runtime does not enable prefix-cache reporting")
    budget = runtime.provenance()["configuration"][
        "gemma4_12b_prefix_cache_budget_bytes"
    ]
    assert type(budget) is int and budget > 0
    messages = [
        SystemMessage(
            role="system",
            content=f"Session {uuid.uuid4().hex}. "
            + "Reference notes describe routine scheduling policies. " * 100
            + "Reply with the single word READY.",
        ),
        UserMessage(role="user", content="Please confirm you are ready."),
    ]
    cold = runtime.generate("gemma4-12b-it", messages, temperature=0, max_tokens=16)
    warm = runtime.generate("gemma4-12b-it", messages, temperature=0, max_tokens=16)
    assert cold.raw_data["cached_prefix_tokens"] == 0
    assert warm.raw_data["cached_prefix_tokens"] >= 256
    assert cold.content == warm.content
    assert "READY" in warm.content
    assert (
        runtime.events[-1]["cached_prefix_tokens"]
        == warm.raw_data["cached_prefix_tokens"]
    )


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


@pytest.mark.parametrize(
    "enabled,connected", [(False, False), (True, False), (True, True)]
)
def test_real_grounded_vpn_tool_result_survives_speech_and_telephony(
    runtime, tmp_path, enabled, connected
):
    """Actual device result -> VoxCPM2 -> telephone audio -> full Whisper.

    A functional voice check; no domain score and no caller model may supply facts.
    """
    import json
    import wave

    from tau2.domains.telecom.environment import get_environment
    from tau2.voice.caller_policy import QualifiedCallerBackend
    from tau2.voice.provenance import benchmark_sources

    environment = get_environment()
    current = environment.user_tools.db.device
    environment.user_tools.db.device = type(current).model_validate(
        {
            **current.model_dump(),
            "vpn_enabled_setting": enabled,
            "vpn_connected": connected,
            "vpn_details": None,
        }
    )
    tools = environment.get_user_tools()
    caller = QualifiedCallerBackend(
        runtime, "Follow the agent's device-check instructions.", device_grounding=True
    )
    history = [UserMessage(role="user", content="Are you currently using a VPN?")]
    start = len(runtime.events)
    request = caller.generate(
        "gemma4-e4b-it", history, tools=tools, call_name="user_streaming_response"
    )
    assert (
        len(request.tool_calls) == 1
        and request.tool_calls[0].name == "check_vpn_status"
    )
    call = request.tool_calls[0]
    observed = next(tool for tool in tools if tool.name == call.name)(**call.arguments)
    reply = caller.generate(
        "gemma4-e4b-it",
        [
            *history,
            request,
            ToolMessage(role="tool", requestor="user", id=call.id, content=observed),
        ],
        tools=tools,
        call_name="user_streaming_response",
    )
    expected = (
        "My VPN is connected."
        if connected
        else "My VPN is enabled in settings, but it is not connected."
        if enabled
        else "My VPN is disabled."
    )
    assert reply.content == expected
    assert reply.raw_data["device_grounding"]["observations"][0]["text"] == observed
    assert not any(event["operation"] == "chat" for event in runtime.events[start:])
    synthesized = runtime.synthesize(
        reply.content,
        RuntimeTTSConfig(
            model_id="voxcpm2",
            voice_id="A natural adult male voice speaking American English.",
            seed=42,
        ),
    )
    telephone = AudioData(
        data=audioop.ulaw2lin(audioop.lin2ulaw(pcm16(synthesized, 8000).data, 2), 2),
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=8000),
    )
    detection_audio = pcm16(telephone, 16000).data
    detection_audio = detection_audio[: len(detection_audio) // 1024 * 1024]
    probabilities = runtime.vad(
        detection_audio, uuid.uuid4().hex, reset=True, end_stream=True
    )
    recognized = runtime.transcribe(
        telephone, RuntimeTranscriptionConfig(model="whisper-large-v3")
    ).transcript
    root = Path(os.environ.get("TAU_VOICE_DEVICE_E2E_OUTPUT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    name = f"vpn-enabled-{enabled}-connected-{connected}"
    with wave.open(str(root / (name + ".wav")), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(telephone.data)
    (root / (name + ".json")).write_text(
        json.dumps(
            {
                "functional_check_only": True,
                "tool": call.name,
                "tool_result": observed,
                "spoken": reply.content,
                "transcript": recognized,
                "vad_probabilities": probabilities,
                "device_grounding": reply.raw_data["device_grounding"],
                "inference": runtime.events[start:],
                "benchmark": benchmark_sources(),
                "native": runtime.provenance(),
            },
            indent=2,
        )
        + "\n"
    )
    text = re.sub(r"[^a-z ]", " ", recognized.lower())
    assert sum(value >= 0.5 for value in probabilities) >= 4, probabilities
    if not enabled:
        assert re.search(r"\bdisabled\b", text), recognized
    elif not connected:
        assert "enabled" in text and "not connected" in text, recognized
    else:
        assert "connected" in text and "not connected" not in text, recognized


@pytest.mark.parametrize("number", ["555-123-2002", "555-123-9876"])
def test_real_known_phone_answer_survives_speech_and_telephony(
    runtime, tmp_path, number
):
    """Caller-owned identifiers must survive the actual acoustic path."""
    import json
    import wave

    from tau2.voice.caller_policy import QualifiedCallerBackend
    from tau2.voice.provenance import benchmark_sources

    caller = QualifiedCallerBackend(
        runtime,
        f"Known info:\nYour phone number is {number}.\nTask instructions:\nRestore your internet.",
        device_grounding=True,
    )
    start = len(runtime.events)
    reply = caller.generate(
        "gemma4-e4b-it",
        [UserMessage(role="user", content="Which of your phone numbers is affected?")],
        tools=[],
        call_name="user_streaming_response",
    )
    assert reply.content == f"My phone number is {number}." and not reply.is_tool_call()
    assert not any(event["operation"] == "chat" for event in runtime.events[start:])
    synthesized = runtime.synthesize(
        reply.content,
        RuntimeTTSConfig(
            model_id="voxcpm2",
            voice_id="A natural adult male voice speaking American English.",
            seed=42,
        ),
    )
    telephone = AudioData(
        data=audioop.ulaw2lin(audioop.lin2ulaw(pcm16(synthesized, 8000).data, 2), 2),
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=8000),
    )
    detection = pcm16(telephone, 16000).data
    probabilities = runtime.vad(
        detection[: len(detection) // 1024 * 1024],
        uuid.uuid4().hex,
        reset=True,
        end_stream=True,
    )
    recognized = runtime.transcribe(
        telephone, RuntimeTranscriptionConfig(model="whisper-large-v3")
    ).transcript
    root = Path(os.environ.get("TAU_VOICE_DEVICE_E2E_OUTPUT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    name = "phone-" + number
    with wave.open(str(root / (name + ".wav")), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(telephone.data)
    (root / (name + ".json")).write_text(
        json.dumps(
            {
                "functional_check_only": True,
                "number": number,
                "spoken": reply.content,
                "transcript": recognized,
                "vad_probabilities": probabilities,
                "device_grounding": reply.raw_data["device_grounding"],
                "inference": runtime.events[start:],
                "benchmark": benchmark_sources(),
                "native": runtime.provenance(),
            },
            indent=2,
        )
        + "\n"
    )
    assert sum(value >= 0.5 for value in probabilities) >= 4, probabilities
    assert re.sub(r"\D", "", recognized) == re.sub(r"\D", "", number), recognized


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
    runtime, tmp_path, monkeypatch
):
    """Opt-in full conversation plus a negative control for database grading."""
    if os.environ.get("TAU_VOICE_FULL_E2E") != "1":
        pytest.skip("Set TAU_VOICE_FULL_E2E=1 to run a complete spoken task")
    import json

    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode
    from tau2.voice import local_run
    from tau2.voice.local_run import run_local_task, task_context

    # A fixed spoken request makes this a pipeline check, independent of a caller LLM's omissions.
    caller_text = (
        "Please create a task called Important Meeting for user underscore one."
    )
    caller_events_start = len(runtime.events)
    caller_audio = runtime.synthesize(
        caller_text,
        RuntimeTTSConfig(
            model_id="voxcpm2",
            voice_id="A clear adult male voice speaking American English.",
            seed=42,
        ),
    )
    speech = audioop.lin2ulaw(pcm16(caller_audio, 8000).data, 2)

    class SpokenUser(FullDuplexUser):
        stale_generation_count = 0

        def get_init_state(self, message_history=None):
            return {"phase": "opener", "offset": 0, "heard": ""}

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            time.sleep(0.05)
            if participant_chunk is not None:
                state["heard"] += participant_chunk.content or ""
                if (
                    participant_chunk.contains_speech
                    and participant_chunk.is_final_chunk
                ):
                    if state["phase"] == "opener":
                        state.update(phase="send", heard="")
                    elif state["phase"] == "wait":
                        if any(
                            word in state["heard"].lower()
                            for word in ("created", "added", "saved", "done")
                        ):
                            return UserMessage(
                                role="user", content="###STOP###", contains_speech=False
                            ), state
                        state.update(phase="send", offset=0, heard="")
            active = state["phase"] == "send"
            offset = state["offset"]
            frame = speech[offset : offset + 400] if active else b""
            if active:
                state["offset"] += len(frame)
                if state["offset"] >= len(speech):
                    state["phase"] = "wait"
            return UserMessage(
                role="user",
                contains_speech=bool(frame),
                cost=0,
                is_audio=True,
                audio_content=audio_bytes_to_string(frame.ljust(400, b"\xff")),
                audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
                is_final_chunk=active and state["phase"] == "wait",
            ), state

    monkeypatch.setattr(
        local_run, "VoiceStreamingUserSimulator", lambda **_: SpokenUser()
    )

    profile_path = os.environ.get("TAU_VOICE_PROFILE")
    selected = load_profile(Path(profile_path) if profile_path else None)
    if selected.execution.require_qualification:
        # This isolates the real spoken agent pipeline with a scripted caller.
        # It is a functional check and cannot produce a domain benchmark score.
        monkeypatch.setattr(
            local_run,
            "require_qualification",
            lambda *args: {"controlled_caller": True, "standard_leaderboard": False},
        )
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
    simulation = SimulationRun.model_validate_json(
        (output / "simulation.json").read_text()
    )
    simulation.info.update(
        caller_mode="controlled_spoken_input", caller_text=caller_text
    )
    (output / "simulation.json").write_text(simulation.model_dump_json(indent=2) + "\n")
    (output / "controlled-caller.json").write_text(
        json.dumps(
            {
                "text": caller_text,
                "mode": "controlled_spoken_input",
                "caller_llm_used": False,
                "test_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "inference": runtime.events[caller_events_start:],
            },
            indent=2,
        )
        + "\n"
    )
    assert result["reward"] == 1, result
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
    assert {"chat", "stream_speech", "transcribe", "vad"} <= {
        row["operation"] for row in events
    }
    wrong = simulation.model_copy(deep=True)
    changed = next(
        call
        for tick in wrong.ticks
        for call in tick.agent_tool_calls
        if call.name == "create_task"
        and any(
            result.id == call.id
            and not result.error
            and result.execution_status == "executed"
            for result in tick.agent_tool_results
        )
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

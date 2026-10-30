"""Offline wire fixtures through the real agent, tools, orchestrator and grader.

Synthetic PCM and provider events validate integration, never model quality.
"""

import asyncio
import base64
import copy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from tau2.agent import provider_agent
from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator import base as orchestrator_base
from tau2.orchestrator.modes import CommunicationMode
from tau2.user.user_simulator_base import FullDuplexUser
from tau2.voice import local_run
from tau2.voice.provider_run import ProviderBudget, ProviderFactory, session_class
from tau2.voice.providers.base import ProviderSettings, agent_instructions
from tau2.voice.providers.elevenlabs import eleven_config, preset_fingerprint
from tau2.voice.runtime_profile import RuntimeCapabilities, load_profile


class Wire:
    def __init__(self, peer, role):
        self.peer, self.role = peer, role
        self.queue = asyncio.Queue()
        self.closed = False

    def put(self, value):
        self.queue.put_nowait(value)

    async def recv(self):
        value = await self.queue.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def send(self, value):
        self.peer.client_events.append(value)
        await self.peer.handle(self, value)

    async def close(self):
        if not self.closed:
            self.closed = True
            self.put(EOFError())


class Peer:
    def __init__(self, provider, model):
        self.provider, self.model = provider, model
        self.main, self.side = None, None
        self.client_events = []
        self.sent_call = False
        self.sent_reply = False

    async def connect(self, url, headers):
        wire = Wire(self, "side" if url.endswith("/attach") else "main")
        if wire.role == "side":
            self.side = wire
        else:
            self.main = wire
        return wire

    def usage(self):
        return {
            "input_tokens": 3,
            "output_tokens": 3,
            "input_token_details": {"text_tokens": 1, "audio_tokens": 2},
            "output_token_details": {"text_tokens": 1, "audio_tokens": 2},
        }

    def called(self):
        call = {
            "type": "function_call",
            "call_id": "wire-call",
            "name": "create_task",
            "arguments": json.dumps(
                {"user_id": "user_1", "title": "Important Meeting"}
            ),
        }
        if self.provider == "openai-realtime":
            self.main.put(
                {
                    "type": "response.done",
                    "response": {
                        "id": "function",
                        "status": "completed",
                        "usage": self.usage(),
                        "output": [call],
                    },
                }
            )
        elif self.provider == "openai-live":
            self.side.put(
                {
                    "type": "response.event",
                    "delegation_id": "delegation",
                    "event": {"type": "response.output_item.done", "item": call},
                }
            )
            self.side.put(
                {
                    "type": "response.event",
                    "event": {
                        "type": "response.completed",
                        "response": {
                            "id": "backend-function",
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                        },
                    },
                }
            )
        elif self.provider == "gemini-live":
            self.main.put(
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "wire-call",
                                "name": "create_task",
                                "args": {
                                    "user_id": "user_1",
                                    "title": "Important Meeting",
                                },
                            }
                        ]
                    }
                }
            )
        else:
            self.main.put(
                {
                    "type": "client_tool_call",
                    "client_tool_call": {
                        "tool_call_id": "wire-call",
                        "tool_name": "create_task",
                        "parameters": {
                            "user_id": "user_1",
                            "title": "Important Meeting",
                        },
                    },
                }
            )

    def reply(self):
        if self.sent_reply:
            return
        self.sent_reply = True
        text = "Your Important Meeting task has been created."
        pcm = b"\x00\x10" * 2400
        encoded = base64.b64encode(pcm).decode()
        if self.provider == "openai-realtime":
            self.main.put(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "spoken",
                    "item_id": "reply",
                    "content_index": 0,
                    "delta": encoded,
                }
            )
            self.main.put(
                {
                    "type": "response.done",
                    "response": {
                        "id": "spoken",
                        "status": "completed",
                        "usage": self.usage(),
                        "output": [
                            {
                                "type": "message",
                                "id": "reply",
                                "content": [{"type": "audio", "transcript": text}],
                            }
                        ],
                    },
                }
            )
        elif self.provider == "openai-live":
            self.side.put(
                {
                    "type": "response.event",
                    "event": {
                        "type": "response.completed",
                        "response": {
                            "id": "backend-reply",
                            "usage": {"input_tokens": 5, "output_tokens": 3},
                        },
                    },
                }
            )
            self.side.put(
                {
                    "type": "session.output_audio.delta",
                    "start_ms": 100,
                    "end_ms": 200,
                    "delta": encoded,
                }
            )
            self.side.put(
                {
                    "type": "session.output_transcript.delta",
                    "start_ms": 100,
                    "end_ms": 200,
                    "delta": text,
                }
            )
            self.side.put(
                {
                    "type": "session.output_audio.delta",
                    "start_ms": 200,
                    "end_ms": 1600,
                    "delta": base64.b64encode(bytes(1400 * 48)).decode(),
                }
            )
            self.side.put({"type": "session.usage.updated", "usage": {"seconds": 2}})
        elif self.provider == "gemini-live":
            self.main.put(
                {
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "responseTokenCount": 3,
                        "promptTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 1},
                            {"modality": "AUDIO", "tokenCount": 2},
                        ],
                        "responseTokensDetails": [
                            {"modality": "TEXT", "tokenCount": 1},
                            {"modality": "AUDIO", "tokenCount": 2},
                        ],
                    }
                }
            )
            self.main.put(
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": encoded,
                                    }
                                }
                            ]
                        },
                        "outputTranscription": {"text": text},
                        "turnComplete": True,
                    }
                }
            )
        else:
            self.main.put(
                {
                    "type": "audio",
                    "audio_event": {"audio_base_64": encoded, "event_id": 1},
                }
            )
            self.main.put(
                {
                    "type": "agent_response",
                    "agent_response_event": {"agent_response": text, "event_id": 1},
                }
            )
            self.main.put(
                {
                    "type": "agent_response_complete",
                    "agent_response_complete_event": {"event_id": 1},
                }
            )

    async def handle(self, wire, value):
        kind = value.get("type")
        if kind == "session.update":
            wire.put(
                {
                    "type": "session.updated",
                    "session": {"id": "session", "model": self.model},
                }
            )
        elif kind == "session.start":
            wire.put(
                {
                    "type": "session.started",
                    "session": {"id": "session", "model": self.model},
                }
            )
        elif "setup" in value:
            wire.put({"setupComplete": {}})
        elif kind == "conversation_initiation_client_data":
            wire.put(
                {
                    "type": "conversation_initiation_metadata",
                    "conversation_initiation_metadata_event": {
                        "conversation_id": "conv_fixture",
                        "user_input_audio_format": "pcm_16000",
                        "agent_output_audio_format": "pcm_24000",
                    },
                }
            )
        audio = (
            value.get("audio")
            if kind in {"input_audio_buffer.append", "session.input_audio.append"}
            else value.get("realtimeInput", {}).get("audio", {}).get("data")
            or value.get("user_audio_chunk")
        )
        if audio and not self.sent_call and any(base64.b64decode(audio)):
            self.sent_call = True
            self.called()
        if (
            (
                kind == "conversation.item.create"
                and value.get("item", {}).get("type") == "function_call_output"
            )
            or kind == "client_tool_result"
            or "toolResponse" in value
            or kind == "response.item.create"
        ):
            self.reply()
        elif kind == "input_audio_buffer.clear":
            wire.put({"type": "input_audio_buffer.cleared"})
        elif kind == "session.close":
            self.side.put(
                {
                    "type": "session.closed",
                    "reason": "close_requested",
                    "usage": {"seconds": 2.5},
                }
            )


class RecordedToneCaller(FullDuplexUser):
    tools = []
    stale_generation_count = 0

    def get_init_state(self, message_history=None):
        return {"phase": "opening", "frames": 0}

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        if (
            participant_chunk
            and participant_chunk.contains_speech
            and participant_chunk.is_final_chunk
        ):
            if state["phase"] == "opening":
                state["phase"] = "send"
            elif state["phase"] == "wait":
                return UserMessage(
                    role="user", content="###STOP###", contains_speech=False
                ), state
        speech = state["phase"] == "send"
        if speech:
            state["frames"] += 1
            if state["frames"] == 2:
                state["phase"] = "wait"
        return UserMessage(
            role="user",
            is_audio=True,
            contains_speech=speech,
            content="PRIVATE-CALLER-GOLD",
            audio_script_gold="PRIVATE-CALLER-GOLD",
            audio_content=audio_bytes_to_string(
                b"\x90" * 400 if speech else b"\xff" * 400
            ),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            is_final_chunk=not speech,
        ), state


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai-realtime", "gpt-realtime-2.1"),
        ("openai-live", "gpt-live-1"),
        ("gemini-live", "gemini-3.8-live"),
        ("elevenlabs", "gpt-4.1"),
    ],
)
@pytest.mark.parametrize("setup_delay", [0, 10])
def test_offline_provider_wire_through_production_runner_and_frozen_grader(
    tmp_path,
    monkeypatch,
    provider,
    model,
    setup_delay,
    native_opener_url=None,
    setup_failure=False,
    greeting_source=None,
):
    monkeypatch.setenv("VOICE_TEST_KEY", "private-test-key")
    profile_path = (
        local_run.repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json"
    )
    profile = load_profile(profile_path)
    models = {}
    for role in ("caller", "listener", "judge", "caller_tts"):
        requirement = profile.roles[role]
        if requirement.model_id in models:
            models[requirement.model_id].features |= requirement.features
        else:
            models[requirement.model_id] = requirement.model_copy(deep=True)
    backend = SimpleNamespace(
        capabilities=lambda: RuntimeCapabilities(
            runtime="offline-fixture", revision="fixture", models=models
        ),
        provenance=lambda: {
            "configuration": {"assets": {"fixture": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
        events=[],
        close=lambda: None,
        synthesize=lambda *_: AudioData(
            data=bytes(4800),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        ),
    )
    wall = SimpleNamespace(now=0)
    if setup_delay:
        timer = SimpleNamespace(
            perf_counter=lambda: wall.now, monotonic=time.monotonic, sleep=time.sleep
        )
        monkeypatch.setattr(orchestrator_base, "time", timer)
        monkeypatch.setattr(local_run, "time", timer)
        monkeypatch.setattr(provider_agent, "time", timer)
        original_synthesize = backend.synthesize

        def slow_synthesize(*args):
            wall.now += setup_delay
            return original_synthesize(*args)

        backend.synthesize = slow_synthesize
    if setup_failure:

        def failed_synthesize(*args):
            raise RuntimeError("Local greeting synthesis failed")

        backend.synthesize = failed_synthesize
    if native_opener_url:
        from tau2.voice.speech_swift import SpeechSwiftBackend

        backend = SpeechSwiftBackend(native_opener_url)
    monkeypatch.setattr(local_run, "SpeechSwiftBackend", lambda *_: backend)
    greeting = f"Welcome to the {greeting_source}."
    if greeting_source == "task":
        original_context = local_run.task_context

        def custom_context(*args, **kwargs):
            task, loaded, language = original_context(*args, **kwargs)
            return task.model_copy(update={"agent_opener": greeting}), loaded, language

        monkeypatch.setattr(local_run, "task_context", custom_context)
    elif greeting_source == "context":
        monkeypatch.setattr(
            orchestrator_base.BaseOrchestrator,
            "initial_agent_message",
            property(lambda _: AssistantMessage(role="assistant", content=greeting)),
        )

    class ClockedToneCaller(RecordedToneCaller):
        def get_next_chunk(self, *args, **kwargs):
            if setup_delay:
                wall.now += 0.05
            return super().get_next_chunk(*args, **kwargs)

    monkeypatch.setattr(
        local_run, "VoiceStreamingUserSimulator", lambda **_: ClockedToneCaller()
    )
    monkeypatch.setattr(
        local_run,
        "require_qualification",
        lambda *_: {"functional_check_only": True, "caller": "synthetic PCM fixture"},
    )
    peer = Peer(provider, model)
    from tau2.registry import registry

    environment = registry.get_env_constructor("mock")()
    preset = eleven_config(
        agent_instructions(environment.get_policy()),
        environment.get_tools(),
        model,
        "voice",
        60,
    )
    digest = preset_fingerprint({"conversation_config": preset})

    class HTTP:
        def __init__(self, **_):
            pass

        async def get(self, url):
            if "/agents/" in url:
                obj = {"conversation_config": copy.deepcopy(preset)}
            elif "get-signed-url" in url:
                obj = {
                    "signed_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_fixture&token=not-a-real-token"
                }
            else:
                obj = {
                    "status": "done",
                    "metadata": {"cost": 10, "call_duration_secs": 2},
                }
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: obj)

        async def aclose(self):
            pass

    settings = ProviderSettings(
        provider,
        model,
        "voice",
        "VOICE_TEST_KEY",
        max_seconds=60 if provider == "elevenlabs" else 10,
        backend_model="gpt-6-luna" if provider == "openai-live" else None,
        agent_id="agent_fixture" if provider == "elevenlabs" else None,
        preset_sha256=digest if provider == "elevenlabs" else None,
        usd_per_credit=0.00001 if provider == "elevenlabs" else None,
        max_usd_per_minute=0.1 if provider == "elevenlabs" else None,
    )
    cls = session_class(provider)

    def session_factory(settings):
        return cls(
            settings,
            connector=peer.connect,
            **({"http_factory": HTTP} if provider == "elevenlabs" else {}),
        )

    factory = ProviderFactory(
        settings,
        ProviderBudget(tmp_path / "budget.json", 1),
        session_factory=session_factory,
    )
    output = tmp_path / "call"
    result = local_run.run_local_task(
        domain="mock",
        task_id="create_task_1",
        base_url="http://127.0.0.1:8787",
        output=output,
        seed=70,
        max_seconds=10,
        wall_timeout=10,
        profile_path=profile_path,
        agent_factory=factory,
        external_agent=factory.descriptor,
    )
    assert result["reward"] == 1 and result["termination"] == "user_stop"
    setup = json.loads((output / "setup.json").read_text())
    assert setup["mode"] == "prepared-opener-v1"
    if greeting_source:
        assert (
            setup["opener_text_sha256"] == hashlib.sha256(greeting.encode()).hexdigest()
        )
    if setup_delay:
        assert result["setup_seconds"] == setup_delay
        assert result["call_wall_seconds"] < 10
        assert result["wall_seconds"] >= setup_delay
    simulation = SimulationRun.model_validate_json(
        (output / "simulation.json").read_text()
    )
    assert simulation.reward_info.db_check.db_match
    calls = [call for tick in simulation.ticks for call in tick.agent_tool_calls]
    assert len(calls) == 1 and calls[0].id == "wire-call"
    assert calls[0].arguments == {"user_id": "user_1", "title": "Important Meeting"}
    assert "PRIVATE-CALLER-GOLD" not in json.dumps(peer.client_events)
    assert "private-test-key" not in (output / "agent-events.json").read_text()
    wrong = simulation.model_copy(deep=True)
    next(c for tick in wrong.ticks for c in tick.agent_tool_calls).arguments[
        "title"
    ] = "Wrong title"
    task, _, _ = local_run.task_context("mock", "create_task_1", 70)
    negative = evaluate_simulation(
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
    assert negative.reward == 0
    if provider in {"gemini-live", "elevenlabs"}:
        assert (
            simulation.agent_cost is None
        )  # Partial/credit-only billing is never labeled free.


@pytest.mark.parametrize("source", ["task", "context"])
def test_prepared_opener_matches_the_orchestrator_greeting(
    tmp_path, monkeypatch, source
):
    test_offline_provider_wire_through_production_runner_and_frozen_grader(
        tmp_path,
        monkeypatch,
        "openai-realtime",
        "gpt-realtime-2.1",
        0,
        greeting_source=source,
    )


@pytest.mark.local_models
def test_real_local_opener_prepares_before_offline_provider_control(
    tmp_path, monkeypatch
):
    url = os.environ.get("SPEECH_SWIFT_INFERENCE_URL")
    if os.environ.get("TAU_SETUP_NATIVE_E2E") != "1" or not url:
        pytest.skip(
            "Requires explicit native setup check and the single local runtime; no provider calls"
        )
    test_offline_provider_wire_through_production_runner_and_frozen_grader(
        tmp_path,
        monkeypatch,
        "openai-realtime",
        "gpt-realtime-2.1",
        0,
        native_opener_url=url,
    )


def test_greeting_failure_is_unscored_and_releases_unstarted_provider_reservation(
    tmp_path, monkeypatch
):
    with pytest.raises(RuntimeError, match="greeting synthesis"):
        test_offline_provider_wire_through_production_runner_and_frozen_grader(
            tmp_path,
            monkeypatch,
            "openai-realtime",
            "gpt-realtime-2.1",
            0,
            setup_failure=True,
        )
    failure = json.loads((tmp_path / "call/failed-run.json").read_text())
    assert failure["phase"] == "setup" and failure["scored"] is False
    assert not (tmp_path / "call/simulation.json").exists()
    row = json.loads((tmp_path / "budget.json").read_text())["reservations"][0]
    assert row["status"] == "metered" and row["charged_usd"] == 0

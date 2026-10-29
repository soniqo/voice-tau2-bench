"""Opt-in real Nova conversation through the production task runner and grader."""

import audioop
import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.user.user_simulator_base import FullDuplexUser
from tau2.voice.baseline import repository_root
from tau2.voice.nova import NovaSettings
from tau2.voice.nova_run import NovaBudget, NovaFactory

pytestmark = pytest.mark.local_models


def test_real_nova_complete_audio_tools_and_database_grading(tmp_path, monkeypatch):
    if os.environ.get("TAU_NOVA_E2E") != "1" or not os.environ.get(
        "SPEECH_SWIFT_INFERENCE_URL"
    ):
        pytest.skip("Requires explicit TAU_NOVA_E2E and the single local runtime")
    from tau2.voice import local_run

    request = Path(os.environ["TAU_NOVA_REQUEST_WAV"])
    with wave.open(str(request), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        pcm, _ = audioop.ratecv(
            wav.readframes(wav.getnframes()), 2, 1, wav.getframerate(), 8000, None
        )
    speech = audioop.lin2ulaw(pcm, 2)
    clips = [speech]
    default_followup = (
        Path(os.environ.get("TAU_NOVA_E2E_OUTPUT", str(tmp_path / "run"))).parent
        / "thanks.wav"
    )
    followup_path = (
        Path(os.environ["TAU_NOVA_FOLLOWUP_WAV"])
        if os.environ.get("TAU_NOVA_FOLLOWUP_WAV")
        else default_followup
    )
    if followup_path.exists():
        with wave.open(str(followup_path), "rb") as wav:
            followup, _ = audioop.ratecv(
                wav.readframes(wav.getnframes()), 2, 1, wav.getframerate(), 8000, None
            )
            clips.append(audioop.lin2ulaw(followup, 2))
    opening = None
    if os.environ.get("TAU_NOVA_OPENER_WAV"):
        with wave.open(os.environ["TAU_NOVA_OPENER_WAV"], "rb") as wav:
            opening = AudioData(
                data=wav.readframes(wav.getnframes()),
                format=AudioFormat(
                    encoding=AudioEncoding.PCM_S16LE, sample_rate=wav.getframerate()
                ),
            )

    class RecordedCaller(FullDuplexUser):
        tools = []
        stale_generation_count = 0

        def get_init_state(self, message_history=None):
            return {"phase": "opener", "offset": 0, "heard": "", "turn": 0}

        def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
            if participant_chunk is not None:
                state["heard"] += participant_chunk.content or ""
                if (
                    participant_chunk.contains_speech
                    and participant_chunk.is_final_chunk
                ):
                    if state["phase"] == "opener":
                        state.update(phase="send", heard="")
                    elif state["phase"] == "wait" and (
                        state["turn"] > 0
                        or any(
                            word in state["heard"].lower()
                            for word in ("created", "added", "saved", "done")
                        )
                    ):
                        if state["turn"] + 1 < len(clips):
                            state.update(
                                phase="send", offset=0, heard="", turn=state["turn"] + 1
                            )
                        else:
                            return self.stop_message(), state
            active = state["phase"] == "send"
            current = clips[state["turn"]]
            frame = current[state["offset"] : state["offset"] + 400] if active else b""
            if active:
                state["offset"] += len(frame)
                if state["offset"] >= len(current):
                    state["phase"] = "wait"
            return UserMessage(
                role="user",
                is_audio=True,
                contains_speech=bool(frame),
                audio_content=audio_bytes_to_string(frame.ljust(400, b"\xff")),
                audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
                is_final_chunk=active and state["phase"] == "wait",
            ), state

        @staticmethod
        def stop_message():
            return UserMessage(role="user", content="###STOP###", contains_speech=False)

    from tau2.data_model.message import UserMessage

    output = Path(os.environ.get("TAU_NOVA_E2E_OUTPUT", str(tmp_path / "run")))
    ledger = Path(
        os.environ.get("TAU_NOVA_BUDGET_LEDGER", str(tmp_path / "budget.json"))
    )
    settings = NovaSettings(
        profile=os.environ["TAU_NOVA_AWS_PROFILE"], max_seconds=150, stop_usd=0.25
    )
    factory = NovaFactory(
        settings,
        NovaBudget(
            ledger,
            float(os.environ.get("TAU_NOVA_BUDGET_USD", "1")),
            recover_interrupted=os.environ.get("TAU_NOVA_RECOVER_INTERRUPTED") == "1",
        ),
    )

    def cached_opener_factory(tools, policy, **kwargs):
        if opening is not None:
            kwargs["opener_backend"] = SimpleNamespace(synthesize=lambda *args: opening)
        return factory(tools, policy, **kwargs)

    monkeypatch.setattr(
        local_run, "VoiceStreamingUserSimulator", lambda **kwargs: RecordedCaller()
    )
    monkeypatch.setattr(
        local_run,
        "require_qualification",
        lambda *args: {
            "functional_check_only": True,
            "caller_mode": "recorded_audio",
            "opener_mode": "cached_voxcpm2_audio"
            if opening is not None
            else "live_voxcpm2_audio",
            "standard_leaderboard": False,
        },
    )
    result = local_run.run_local_task(
        domain="mock",
        task_id="create_task_1",
        base_url=os.environ["SPEECH_SWIFT_INFERENCE_URL"],
        output=output,
        seed=70,
        max_seconds=120,
        wall_timeout=120,
        profile_path=repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json",
        agent_factory=cached_opener_factory,
        external_agent={
            **factory.descriptor,
            "functional_check_only": True,
            "caller_mode": "recorded_audio",
        },
    )
    assert result["reward"] == 1, result
    simulation = SimulationRun.model_validate_json(
        (output / "simulation.json").read_text()
    )
    calls = [c for tick in simulation.ticks for c in tick.agent_tool_calls]
    assert any(
        c.name == "create_task"
        and c.arguments == {"user_id": "user_1", "title": "Important Meeting"}
        for c in calls
    )
    assert simulation.agent_usage and simulation.agent_cost > 0
    assert simulation.info["local_metrics"]["usage_finalized"]
    wrong = simulation.model_copy(deep=True)
    next(
        c
        for tick in wrong.ticks
        for c in tick.agent_tool_calls
        if c.name == "create_task"
    ).arguments["title"] = "Deliberately wrong title"
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
    (output / "negative-control.json").write_text(
        negative.model_dump_json(indent=2) + "\n"
    )
    (output / "functional-result.json").write_text(
        json.dumps(
            {**result, "functional_check_only": True, "caller_audio_turns": len(clips)},
            indent=2,
        )
        + "\n"
    )

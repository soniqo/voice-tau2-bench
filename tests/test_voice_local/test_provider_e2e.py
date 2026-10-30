"""Explicit opt-in real-provider test; never part of offline adapter validation."""

import audioop
import hashlib
import json
import os
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.local_models


def test_real_provider_audio_tools_and_frozen_task_grader(tmp_path, monkeypatch):
    if os.environ.get("TAU_PROVIDER_E2E") != "1" or not os.environ.get(
        "SPEECH_SWIFT_INFERENCE_URL"
    ):
        pytest.skip("Requires explicit TAU_PROVIDER_E2E and the single local runtime")
    from tau2.voice import local_run
    from tau2.voice.baseline import repository_root
    from tau2.voice.local_run import run_local_task
    from tau2.voice.provider_run import ProviderBudget, ProviderFactory
    from tau2.voice.providers.base import ProviderSettings

    provider = os.environ["TAU_PROVIDER"]
    max_seconds = float(os.environ.get("TAU_PROVIDER_MAX_SECONDS", "90"))
    output = Path(os.environ.get("TAU_PROVIDER_OUTPUT", str(tmp_path / "run")))
    settings = ProviderSettings(
        provider=provider,
        model=os.environ["TAU_PROVIDER_MODEL"],
        voice=os.environ["TAU_PROVIDER_VOICE"],
        key_env=os.environ["TAU_PROVIDER_KEY_ENV"],
        max_seconds=float(
            os.environ.get("TAU_PROVIDER_SESSION_SECONDS", str(max_seconds))
        ),
        stop_usd=float(os.environ.get("TAU_PROVIDER_STOP_USD", "0.25")),
        backend_model=os.environ.get("TAU_PROVIDER_BACKEND_MODEL"),
        agent_id=os.environ.get("TAU_PROVIDER_AGENT_ID"),
        preset_sha256=os.environ.get("TAU_PROVIDER_PRESET_SHA256"),
        vertex_project=os.environ.get("TAU_PROVIDER_VERTEX_PROJECT"),
        vertex_location=os.environ.get("TAU_PROVIDER_VERTEX_LOCATION"),
        usd_per_credit=float(os.environ["TAU_PROVIDER_USD_PER_CREDIT"])
        if provider == "elevenlabs"
        else None,
        max_usd_per_minute=float(os.environ["TAU_PROVIDER_MAX_USD_PER_MINUTE"])
        if provider == "elevenlabs"
        else None,
    )
    factory = ProviderFactory(
        settings,
        ProviderBudget(
            Path(
                os.environ.get(
                    "TAU_PROVIDER_BUDGET_LEDGER", str(tmp_path / "budget.json")
                )
            ),
            float(os.environ.get("TAU_PROVIDER_BUDGET_USD", "1")),
        ),
    )
    descriptor = factory.descriptor
    agent_factory = factory
    if os.environ.get("TAU_PROVIDER_OPENER_WAV"):
        from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat

        opener_path = Path(os.environ["TAU_PROVIDER_OPENER_WAV"])
        with wave.open(str(opener_path), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            opening = AudioData(
                data=wav.readframes(wav.getnframes()),
                format=AudioFormat(
                    encoding=AudioEncoding.PCM_S16LE, sample_rate=wav.getframerate()
                ),
            )

        def cached_factory(tools, policy, **kwargs):
            kwargs["opener_backend"] = SimpleNamespace(synthesize=lambda *args: opening)
            return factory(tools, policy, **kwargs)

        agent_factory = cached_factory
        descriptor = {
            **descriptor,
            "opener_mode": "cached_voxcpm2_audio",
            "opener_sha256": hashlib.sha256(opener_path.read_bytes()).hexdigest(),
        }
    recorded = os.environ.get("TAU_PROVIDER_RECORDED_CONTROL") == "1"
    if recorded:
        from tau2.data_model.audio import (
            AudioEncoding,
            AudioFormat,
            audio_bytes_to_string,
        )
        from tau2.data_model.message import UserMessage
        from tau2.user.user_simulator_base import FullDuplexUser

        request = Path(os.environ["TAU_PROVIDER_REQUEST_WAV"])
        with wave.open(str(request), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
            pcm, _ = audioop.ratecv(
                wav.readframes(wav.getnframes()), 2, 1, wav.getframerate(), 8000, None
            )
        speech = audioop.lin2ulaw(pcm, 2)

        class RecordedCaller(FullDuplexUser):
            tools = []
            stale_generation_count = 0

            def get_init_state(self, message_history=None):
                return {"phase": "opener", "offset": 0, "heard": ""}

            def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
                if participant_chunk is not None:
                    state["heard"] += participant_chunk.content or ""
                    if (
                        participant_chunk.contains_speech
                        and participant_chunk.is_final_chunk
                    ):
                        if state["phase"] == "opener":
                            state.update(phase="send", heard="")
                        elif state["phase"] == "wait" and any(
                            word in state["heard"].lower()
                            for word in ("created", "added", "saved", "done")
                        ):
                            return self.stop_message(), state
                active = state["phase"] == "send"
                frame = (
                    speech[state["offset"] : state["offset"] + 400] if active else b""
                )
                if active:
                    state["offset"] += len(frame)
                    if state["offset"] >= len(speech):
                        state["phase"] = "wait"
                return UserMessage(
                    role="user",
                    is_audio=True,
                    contains_speech=bool(frame),
                    audio_content=audio_bytes_to_string(frame.ljust(400, b"\xff")),
                    audio_format=AudioFormat(
                        encoding=AudioEncoding.ULAW, sample_rate=8000
                    ),
                    is_final_chunk=active and state["phase"] == "wait",
                ), state

            @staticmethod
            def stop_message():
                return UserMessage(
                    role="user", content="###STOP###", contains_speech=False
                )

        monkeypatch.setattr(
            local_run, "VoiceStreamingUserSimulator", lambda **kwargs: RecordedCaller()
        )
        marker = {
            "functional_check_only": True,
            "caller_mode": "recorded_audio",
            "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
            "standard_leaderboard": False,
        }
        monkeypatch.setattr(local_run, "require_qualification", lambda *args: marker)
        descriptor = {**descriptor, **marker}
    result = run_local_task(
        domain="mock",
        task_id="create_task_1",
        seed=70,
        base_url=os.environ["SPEECH_SWIFT_INFERENCE_URL"],
        output=output,
        max_seconds=max_seconds,
        wall_timeout=max_seconds,
        qualification_path=Path(os.environ["TAU_PROVIDER_QUALIFICATION"])
        if not recorded
        else None,
        profile_path=repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json",
        agent_factory=agent_factory,
        external_agent=descriptor,
    )
    (output / "functional-result.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["reward"] == 1, result

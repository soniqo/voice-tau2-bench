import asyncio
import base64
import importlib.util
import json
import wave
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.environment.tool import Tool
from tau2.voice.nova import (
    NovaSession,
    NovaSettings,
    NovaUsage,
    _identity_values,
    initial_events,
)


@pytest.mark.parametrize(
    "token,expected", [(None, None), ("", None), ("temporary-token", "temporary-token")]
)
def test_empty_session_token_is_omitted_from_signing(token, expected):
    parameters = _identity_values(
        SimpleNamespace(access_key="fixture", secret_key="fixture", token=token)
    )
    assert parameters["session_token"] == expected


def usage(count):
    return {
        "details": {
            "total": {
                d: {m: count for m in ("speechTokens", "textTokens")}
                for d in ("input", "output")
            }
        }
    }


def test_usage_uses_cumulative_modality_counts_without_double_billing():
    meter = NovaUsage()
    assert meter.usd is None
    meter.update(usage(100))
    meter.update(usage(100))
    assert meter.usd == Decimal(".001808")
    meter.update(usage(200))
    assert meter.usd == Decimal(".003616")
    with pytest.raises(ValueError, match="backwards"):
        meter.update(usage(100))


@pytest.mark.parametrize(
    "bad", [{}, {"details": {"total": {}}}, usage(-1), usage(True)]
)
def test_unpriced_usage_never_becomes_zero(bad):
    with pytest.raises(ValueError):
        NovaUsage().update(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "unknown"},
        {"region": "eu-central-1"},
        {"stop_usd": 0},
        {"stop_usd": 6},
        {"stop_usd": float("nan")},
        {"max_seconds": 451},
        {"max_seconds": float("inf")},
    ],
)
def test_invalid_pilot_settings_fail_before_credential_resolution(kwargs):
    with pytest.raises(ValueError):
        NovaSettings(profile="fixture", **kwargs)


def test_protocol_has_only_public_policy_and_declared_tools():
    def lookup(reference: str) -> str:
        """Look up a reference."""
        return reference

    events = initial_events("p", "public policy", [Tool(lookup)], "tiffany")
    schema = events[1]["promptStart"]["toolConfiguration"]["tools"][0]["toolSpec"]
    assert (
        json.loads(schema["inputSchema"]["json"])["properties"]["reference"]["type"]
        == "string"
    )
    assert events[3]["textInput"]["content"] == "public policy"
    assert (
        events[-1]["contentStart"]["audioInputConfiguration"]["sampleRateHertz"]
        == 16000
    )


def test_audio_limit_and_alignment_fail_without_network():
    session = NovaSession(NovaSettings(profile="fixture", max_seconds=1))
    for pcm in (b"x", bytes(32002)):
        with pytest.raises(ValueError):
            asyncio.run(session.audio(pcm))


@pytest.mark.parametrize(
    "title,success", [("Important Meeting", True), ("Wrong title", False)]
)
def test_probe_replays_output_ids_and_checks_real_database(
    tmp_path, monkeypatch, title, success
):
    path = Path(__file__).resolve().parents[2] / "scripts/probe_nova.py"
    spec = importlib.util.spec_from_file_location("probe_nova", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(1600))
    events = [
        {"usageEvent": usage(100)},
        {"contentEnd": {"contentId": "transcript"}},
        {
            "toolUse": {
                "contentId": "tool",
                "toolName": "create_task",
                "toolUseId": "call",
                "content": json.dumps({"user_id": "user_1", "title": title}),
            }
        },
        {"contentEnd": {"contentId": "tool"}},
        {"audioOutput": {"content": base64.b64encode(bytes(480)).decode()}},
        {
            "contentStart": {
                "contentId": "final",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        },
        {"contentEnd": {"contentId": "final", "stopReason": "END_TURN"}},
    ]

    class FixtureSession:
        def __init__(self, settings):
            self.usage = NovaUsage()
            self.events = iter(events)

        async def connect(self, system, tools):
            assert "Important Meeting" not in system

        async def receive(self):
            event = next(self.events)
            if "usageEvent" in event:
                self.usage.update(event["usageEvent"])
            return event

        async def tool_result(self, call_id, result):
            assert call_id == "call"
            assert result["title"] == title

        async def close(self):
            pass

    monkeypatch.setattr(module, "NovaSession", FixtureSession)
    args = SimpleNamespace(
        input=audio,
        profile="fixture",
        max_seconds=10,
        stop_usd=0.1,
        output=tmp_path / "run",
    )
    code = asyncio.run(module.probe(args))
    report = json.loads((args.output / "result.json").read_text())
    assert code == (0 if success else 1)
    assert report["success"] is success
    assert report["database_control"]["reward"] == float(success)
    assert report["negative_control"]["reward"] == 0
    assert report["domain_score"] is None

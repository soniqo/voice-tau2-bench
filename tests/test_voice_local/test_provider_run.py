import json
from types import SimpleNamespace

import pytest

from tau2.voice.provider_cli import main, session_window
from tau2.voice.provider_run import ProviderBudget, ProviderFactory
from tau2.voice.providers.base import ProviderSettings


def settings(**kw):
    return ProviderSettings(
        **(
            dict(
                provider="openai-realtime",
                model="gpt-realtime-2.1",
                voice="marin",
                key_env="TEST_PROVIDER_KEY",
            )
            | kw
        )
    )


@pytest.mark.parametrize(
    "call,requested,eleven,expected",
    [
        (45, None, True, 60),
        (90, None, False, 105),
        (240, None, False, 255),
        (405, None, True, 420),
        (45, 75, True, 75),
    ],
)
def test_session_window_separates_conversation_from_provider_lifetime(
    call, requested, eleven, expected
):
    assert session_window(call, requested, eleven=eleven) == expected


@pytest.mark.parametrize(
    "call,requested",
    [
        (0, None),
        (float("nan"), None),
        (406, None),
        (90, 90),
        (45, 59),
        (45, float("inf")),
    ],
)
def test_session_window_rejects_unsupported_limits_before_inference(call, requested):
    with pytest.raises(ValueError):
        session_window(call, requested, eleven=True)


@pytest.mark.parametrize(
    "kw",
    [
        {"provider": "missing"},
        {"key_env": "actual-key-value?"},
        {"key_env": "1BAD"},
        {"max_seconds": 421},
        {"stop_usd": float("nan")},
        {"provider": "openai-live", "model": "gpt-live-1"},
        {"provider": "elevenlabs", "model": "configured-llm"},
        {"vertex_project": "benchmark-project", "vertex_location": "us-central1"},
        {
            "provider": "gemini-live",
            "model": "gemini-3.8-live",
            "vertex_project": "bad/project",
            "vertex_location": "us-central1",
        },
    ],
)
def test_provider_settings_reject_unsupported_or_unbounded_conditions(kw):
    with pytest.raises(ValueError):
        settings(**kw)


def test_provider_factory_rejects_unpriced_models_before_reserving_or_network(tmp_path):
    budget = ProviderBudget(tmp_path / "budget.json", 1)
    with pytest.raises(ValueError, match="verified pricing"):
        ProviderFactory(settings(model="unknown"), budget)
    with pytest.raises(ValueError, match="Live backend"):
        ProviderFactory(
            settings(
                provider="openai-live", model="gpt-live-1", backend_model="unknown"
            ),
            budget,
        )
    assert not budget.path.exists()


def test_provider_ledger_retains_unknown_and_does_not_assume_credits_are_dollars(
    tmp_path,
):
    budget = ProviderBudget(tmp_path / "budget.json", 0.4)
    ident = budget.reserve(0.25)
    budget.finish(
        ident,
        SimpleNamespace(
            _thread=True, metrics={"usage_finalized": True, "metered_usd": None}
        ),
    )
    row = json.loads(budget.path.read_text())["reservations"][0]
    assert row["status"] == "unknown" and row["charged_usd"] == 0.35
    with pytest.raises(ValueError, match="voice-provider budget"):
        budget.reserve(0.25)


def test_provider_constructor_failure_releases_unstarted_reservation(tmp_path):
    budget = ProviderBudget(tmp_path / "budget.json", 1)
    factory = ProviderFactory(settings(), budget)
    with pytest.raises(TypeError):
        factory([], "PUBLIC", missing_argument=True)
    row = json.loads(budget.path.read_text())["reservations"][0]
    assert row["status"] == "metered" and row["charged_usd"] == 0


def test_cli_requires_explicit_cloud_optin_before_any_trial(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--provider",
                "openai-realtime",
                "--model",
                "gpt-realtime-2.1",
                "--domain",
                "mock",
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert caught.value.code == 2
    assert "explicit --allow-cloud" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "extra", [["--trials", "2"], ["--resume"], ["--num-tasks", "2"]]
)
def test_mock_cli_rejects_batch_options_instead_of_silently_ignoring_them(
    tmp_path, extra, capsys
):
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--allow-cloud",
                "--provider",
                "openai-realtime",
                "--model",
                "gpt-realtime-2.1",
                "--domain",
                "mock",
                "--qualification",
                str(tmp_path / "qualified.json"),
                "--output",
                str(tmp_path / "run"),
                *extra,
            ]
        )
    assert caught.value.code == 2 and "one fresh task/trial" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_provider_cli_dispatches_named_stack_without_credentials_or_reservation(
    tmp_path, monkeypatch
):
    from tau2.voice import batch

    observed = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        batch, "run_batch", lambda **kw: observed.update(kw) or {"complete": True}
    )
    assert (
        main(
            [
                "--allow-cloud",
                "--provider",
                "openai-live",
                "--model",
                "gpt-live-1",
                "--backend-model",
                "gpt-6-luna",
                "--domain",
                "all",
                "--num-tasks",
                "1",
                "--qualification",
                str(tmp_path / "qualified.json"),
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    descriptor = observed["external_agent"]
    assert descriptor["backend_model"] == "gpt-6-luna"
    assert descriptor["endpointing"]["mode"] == "reflected_audio_noise_gate"
    assert observed["wall_timeout"] == observed["max_seconds"] == 240
    assert descriptor["native_roles"] == ["caller", "listener", "judge", "caller_tts"]
    assert descriptor["session_seconds"] == 255
    assert descriptor["setup_mode"] == "prepared-opener-v1"
    assert not (tmp_path / "run-budget.json").exists()


def test_eleven_export_is_offline_public_policy_and_exact_domain_tools(
    tmp_path, monkeypatch
):
    from tau2.registry import registry

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    path = tmp_path / "preset.json"
    assert (
        main(
            [
                "--export-eleven-config",
                str(path),
                "--domain",
                "mock",
                "--model",
                "gpt-4.1",
                "--voice",
                "chosen-voice",
            ]
        )
        == 0
    )
    config = json.loads(path.read_text())["conversation_config"]
    prompt = config["agent"]["prompt"]
    expected = registry.get_env_constructor("mock")().get_tools()
    assert {t["name"] for t in prompt["tools"]} == {t.name for t in expected}
    assert config["agent"]["first_message"] == "" and prompt["llm"] == "gpt-4.1"
    assert "agent_response_complete" in config["conversation"]["client_events"]
    assert "Important Meeting" not in prompt["prompt"]
    with pytest.raises(SystemExit):
        main(
            [
                "--export-eleven-config",
                str(path),
                "--domain",
                "mock",
                "--model",
                "gpt-4.1",
                "--voice",
                "chosen-voice",
            ]
        )

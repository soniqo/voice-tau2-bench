import json
from types import SimpleNamespace

import pytest

from tau2.voice.baseline import repository_root
from tau2.voice.nova_cli import main
from tau2.voice.nova_run import NovaBudget
from tau2.voice.runtime_profile import load_profile


def test_budget_is_shared_across_trials_and_unknown_usage_stays_reserved(tmp_path):
    budget = NovaBudget(tmp_path / "budget.json", 1)
    key = budget.reserve(0.25)
    with pytest.raises(ValueError, match="already owns"):
        budget.reserve(0.25)
    budget.finish(key, SimpleNamespace(_thread=True, metrics={}))
    data = json.loads(budget.path.read_text())
    assert data["reservations"][0]["status"] == "unknown"
    assert data["reservations"][0]["charged_usd"] == 0.35
    key = budget.reserve(0.25)
    budget.finish(
        key,
        SimpleNamespace(
            _thread=True, metrics={"metered_usd": 0.05, "usage_finalized": True}
        ),
    )
    assert budget.reserve(0.25)


def test_budget_rejects_new_session_when_reservations_consume_balance(tmp_path):
    budget = NovaBudget(tmp_path / "budget.json", 0.4)
    key = budget.reserve(0.25)
    budget.finish(key, SimpleNamespace(_thread=True, metrics={}))
    with pytest.raises(ValueError, match="Insufficient"):
        budget.reserve(0.25)
    with pytest.raises(ValueError, match="different budget"):
        NovaBudget(budget.path, 1).reserve(0.25)


def test_no_connection_releases_reservation_without_claiming_unknown_cost_zero(
    tmp_path,
):
    budget = NovaBudget(tmp_path / "budget.json", 1)
    key = budget.reserve(0.25)
    budget.finish(key, SimpleNamespace(_thread=None, metrics={}))
    assert json.loads(budget.path.read_text())["reservations"][0]["charged_usd"] == 0


def test_v6_uses_real_pacing_but_the_same_qualified_caller_features():
    profile = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json"
    )
    assert profile.execution.mode == "realtime-v1"
    assert profile.execution.routing_focus and profile.execution.question_focus
    assert profile.execution.tool_guard and profile.execution.require_qualification
    assert profile.budgets() == (240, 240)


def test_cli_dispatches_external_agent_and_records_its_own_model(
    tmp_path, monkeypatch, capsys
):
    from tau2.voice import batch

    observed = {}

    def run(**kwargs):
        observed.update(kwargs)
        return {"complete": True}

    monkeypatch.setattr(batch, "run_batch", run)
    assert (
        main(
            [
                "--domain",
                "all",
                "--num-tasks",
                "1",
                "--aws-profile",
                "fixture",
                "--qualification",
                str(tmp_path / "q.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert observed["external_agent"]["model_id"] == "amazon.nova-2-5-sonic"
    assert observed["max_seconds"] == observed["wall_timeout"] == 240
    assert observed["agent_factory"].settings.profile == "fixture"
    assert not (tmp_path / "out-budget.json").exists(), (
        "No reservation before a real trial"
    )


def test_cli_rejects_calls_exceeding_native_connection_before_any_sdk_import(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--domain",
                "airline",
                "--aws-profile",
                "fixture",
                "--max-seconds",
                "600",
                "--qualification",
                str(tmp_path / "q.json"),
                "--output",
                str(tmp_path / "out"),
            ]
        )


def test_resume_recovers_only_dead_clients_and_keeps_reserved_cost(
    tmp_path, monkeypatch
):
    import os

    budget = NovaBudget(tmp_path / "budget.json", 1)
    budget.reserve(0.25)
    resume = NovaBudget(budget.path, 1, recover_interrupted=True)
    with pytest.raises(ValueError, match="already owns"):
        resume.reserve(0.25)

    def missing(pid, signal):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "kill", missing)
    resume.reserve(0.25)
    saved = json.loads(budget.path.read_text())["reservations"]
    assert saved[0]["status"] == "unknown" and saved[0]["charged_usd"] == 0.35
    assert saved[1]["status"] == "active"

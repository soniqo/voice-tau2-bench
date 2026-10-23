"""Exercise task construction for real domains without loading any models."""

from types import SimpleNamespace

import pytest

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.voice import local_run
from tau2.voice.runtime_profile import LocalVoiceProfile, RuntimeCapabilities


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
def test_runner_constructs_domains_with_optional_caller_tools(
    monkeypatch, tmp_path, domain
):
    profile = LocalVoiceProfile.model_validate_json(
        (local_run.repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )
    models = {}
    for requirement in profile.roles.values():
        if requirement.model_id in models:
            models[requirement.model_id].features |= requirement.features
        else:
            models[requirement.model_id] = requirement.model_copy(deep=True)
    backend = SimpleNamespace(
        capabilities=lambda: RuntimeCapabilities(
            runtime="fixture", revision="test", models=models
        ),
        provenance=lambda: {
            "configuration": {"assets": {"fixture": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
        events=[],
        close=lambda: None,
    )
    monkeypatch.setattr(local_run, "SpeechSwiftBackend", lambda _: backend)
    observed = {}

    class Orchestrator:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def run(self):
            return SimulationRun(
                id="construction",
                task_id=observed["task"].id,
                start_time="2026-10-22T00:00:00",
                end_time="2026-10-22T00:00:00",
                duration=0,
                termination_reason=TerminationReason.USER_STOP,
                ticks=[],
            )

    monkeypatch.setattr(local_run, "FullDuplexOrchestrator", Orchestrator)
    monkeypatch.setattr(
        local_run, "evaluate_simulation", lambda **_: RewardInfo(reward=1)
    )
    result = local_run.run_local_task(
        domain=domain,
        task_id=None,
        base_url="http://localhost:8080",
        output=tmp_path,
    )
    assert result["domain"] == domain
    assert bool(observed["user"].tools) == (domain == "telecom")
    assert (tmp_path / "simulation.json").exists()
    observed["agent"].stop()

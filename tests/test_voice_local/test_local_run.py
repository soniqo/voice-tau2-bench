"""Exercise task construction for real domains without loading any models."""

from types import SimpleNamespace

import pytest

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.voice import local_run
from tau2.voice.runtime_profile import LocalVoiceProfile, RuntimeCapabilities


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
@pytest.mark.parametrize(
    "profile_name",
    ["soniqo-local-v1", "soniqo-local-gemma12b-v1", "soniqo-local-gemma12b-v2"],
)
def test_runner_constructs_domains_with_optional_caller_tools(
    monkeypatch, tmp_path, domain, profile_name
):
    profile_path = local_run.repository_root() / f"configs/voice/{profile_name}.json"
    profile = LocalVoiceProfile.model_validate_json(profile_path.read_text())
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
            self.tool_guards = {"user": SimpleNamespace(events=[])}

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
    monkeypatch.setattr(local_run, "LocalV2Orchestrator", Orchestrator)
    monkeypatch.setattr(
        local_run, "require_qualification", lambda *args: {"fixture": True}
    )
    monkeypatch.setattr(
        local_run, "evaluate_simulation", lambda **_: RewardInfo(reward=1)
    )
    result = local_run.run_local_task(
        domain=domain,
        task_id=None,
        base_url="http://localhost:8080",
        output=tmp_path,
        profile_path=profile_path,
        seed=43,
    )
    assert result["domain"] == domain
    assert bool(observed["user"].tools) == (domain == "telecom")
    assert observed["agent"].model_id == profile.roles["agent"].model_id
    assert observed["agent"].seed == 43
    assert profile.roles["caller"].model_id == "gemma4-e4b-it"
    assert profile.roles["judge"].model_id == "gemma4-e4b-it"
    v2 = profile_name.endswith("-v2")
    assert observed["user"].realtime_generation is (not v2)
    assert observed["max_steps"] == (47999 if v2 else 12000)
    assert observed["timeout"] == (7200 if v2 else 600)
    assert observed["user"].wait_to_respond_threshold_self == (300 if v2 else 100)
    assert (tmp_path / "simulation.json").exists()
    observed["agent"].stop()

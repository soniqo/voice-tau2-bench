"""Exercise task construction for real domains without loading any models."""

from types import SimpleNamespace

import pytest

from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.voice import local_run
from tau2.voice.diagnostics import InferenceRunError
from tau2.voice.runtime_profile import LocalVoiceProfile, RuntimeCapabilities


@pytest.mark.parametrize("domain", ["airline", "retail", "telecom"])
@pytest.mark.parametrize("late_inference_error", [False, True])
@pytest.mark.parametrize(
    "profile_name",
    ["soniqo-local-v1", "soniqo-local-gemma12b-v1", "soniqo-local-gemma12b-v2"],
)
def test_runner_constructs_domains_with_optional_caller_tools(
    monkeypatch, tmp_path, domain, profile_name, late_inference_error
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
        events=[{"request_id": "late-error", "success": False}]
        if late_inference_error
        else [],
        close=lambda: None,
    )
    monkeypatch.setattr(local_run, "SpeechSwiftBackend", lambda _: backend)
    observed = {}

    class Orchestrator:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.tool_guards = {"user": SimpleNamespace(events=[])}
            self.ticks = []

        def run(self):
            return SimulationRun(
                id="construction",
                task_id=observed["task"].id,
                start_time="2026-10-22T00:00:00",
                end_time="2026-10-22T00:00:00",
                duration=0,
                termination_reason=TerminationReason.MAX_STEPS
                if late_inference_error
                else TerminationReason.USER_STOP,
                ticks=[],
            )

    monkeypatch.setattr(local_run, "FullDuplexOrchestrator", Orchestrator)
    monkeypatch.setattr(local_run, "LocalV2Orchestrator", Orchestrator)
    monkeypatch.setattr(
        local_run, "require_qualification", lambda *args: {"fixture": True}
    )

    def grade(**kwargs):
        assert not late_inference_error, "A failed request must prevent grading"
        return RewardInfo(reward=1)

    monkeypatch.setattr(local_run, "evaluate_simulation", grade)
    if late_inference_error:
        with pytest.raises(InferenceRunError, match="late-error"):
            local_run.run_local_task(
                domain=domain,
                task_id=None,
                base_url="http://localhost:8080",
                output=tmp_path,
                profile_path=profile_path,
                seed=43,
            )
        assert not (tmp_path / "simulation.json").exists()
        assert '"scored": false' in (tmp_path / "failed-run.json").read_text()
        observed["agent"].stop()
        return
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


@pytest.mark.parametrize("failed", [False, True])
def test_external_agent_checks_only_local_support_roles_and_rejects_late_errors(
    monkeypatch, tmp_path, failed
):
    from tau2.agent.nova_agent import NovaAgentError
    from tau2.data_model.message import SystemMessage

    path = local_run.repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json"
    profile = LocalVoiceProfile.model_validate_json(path.read_text())
    supporting = ("caller", "listener", "judge", "caller_tts")
    models = {}
    for role in supporting:
        required = profile.roles[role]
        if required.model_id not in models:
            models[required.model_id] = required.model_copy(deep=True)
        else:
            models[required.model_id].features |= required.features
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
    monkeypatch.setattr(
        local_run, "require_qualification", lambda *args: {"fixture": True}
    )
    closed = []
    actor = SimpleNamespace(
        model_id="amazon.nova-2-5-sonic",
        metrics={"cancellations": []},
        get_init_state=lambda: SimpleNamespace(
            messages=[SystemMessage(role="system", content="public policy")]
        ),
        stop=lambda: closed.append(True),
    )

    def validate():
        if failed:
            raise NovaAgentError("late provider failure")

    actor.raise_if_failed = validate

    class Orchestrator:
        def __init__(self, **kwargs):
            assert kwargs["clock"] is None
            self.task = kwargs["task"]
            self.tool_guards = {
                "user": SimpleNamespace(events=[]),
                "assistant": SimpleNamespace(events=[]),
            }
            self.ticks = []

        def run(self):
            return SimulationRun(
                id="external",
                task_id=self.task.id,
                start_time="2026-10-29T00:00:00",
                end_time="2026-10-29T00:00:00",
                duration=0,
                termination_reason=TerminationReason.USER_STOP,
                ticks=[],
            )

    monkeypatch.setattr(local_run, "LocalV2Orchestrator", Orchestrator)

    def grade(**kwargs):
        assert not failed, "A late SDK error must prevent grading"
        return RewardInfo(reward=1)

    monkeypatch.setattr(local_run, "evaluate_simulation", grade)
    params = dict(
        domain="mock",
        task_id="create_task_1",
        base_url="http://localhost:8080",
        output=tmp_path,
        profile_path=path,
        agent_factory=lambda tools, policy, **kwargs: actor,
        external_agent={"model_id": actor.model_id, "model_args": {"temperature": 0.7}},
    )
    if failed:
        with pytest.raises(NovaAgentError, match="late provider"):
            local_run.run_local_task(**params)
        assert not (tmp_path / "simulation.json").exists()
        assert '"scored": false' in (tmp_path / "failed-run.json").read_text()
    else:
        assert local_run.run_local_task(**params)["reward"] == 1
        import json

        settings = json.loads((tmp_path / "settings.json").read_text())
        assert settings["agent_model"] == actor.model_id
        assert settings["agent_model_args"] == {"temperature": 0.7}
        assert "caller_runtime_profile" in json.loads(
            (tmp_path / "profile.json").read_text()
        )
    assert closed

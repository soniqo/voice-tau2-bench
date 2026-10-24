import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tau2.voice import batch
from tau2.voice.baseline import repository_root
from tau2.voice.runtime_profile import LocalVoiceProfile, RuntimeCapabilities


def test_full_plan_keeps_278_frozen_tasks_and_independent_trial_seeds():
    jobs = batch.plan_jobs("all", None, 2, 42)
    assert Counter(job["domain"] for job in jobs) == {
        "airline": 100,
        "retail": 228,
        "telecom": 228,
    }
    assert len({job["case"] for job in jobs}) == 556
    assert jobs[0]["task_id"] == jobs[1]["task_id"]
    assert [job["seed"] for job in jobs[:2]] == [42, 43]


@pytest.mark.parametrize("count,trials", [(0, 1), (-1, 1), (1, 0)])
def test_invalid_trial_plans_are_rejected(count, trials):
    with pytest.raises(ValueError, match="positive"):
        batch.plan_jobs("all", count, trials, 42)


def test_predeclared_pilot_uses_heldout_ids_and_two_independent_seeds():
    plan = json.loads(
        (repository_root() / "configs/pilots/local-v2-heldout.json").read_text()
    )
    jobs = batch.plan_jobs("all", None, 2, 70, plan)
    assert Counter(job["domain"] for job in jobs) == {
        "airline": 6,
        "retail": 6,
        "telecom": 6,
    }
    assert len(jobs) == 18
    assert {job["seed"] for job in jobs} == {70, 71}
    excluded = {
        (entry["domain"], entry["task_id"]) for entry in plan["excluded_debug_tasks"]
    }
    assert not any((job["domain"], job["task_id"]) in excluded for job in jobs)


@pytest.mark.parametrize("change", ["duplicate", "unknown", "count", "domain"])
def test_explicit_plan_rejects_invalid_or_ambiguous_selection(change):
    plan = {"schema_version": 1, "tasks": [{"domain": "airline", "task_id": "1"}]}
    if change == "duplicate":
        plan["tasks"] *= 2
    elif change == "unknown":
        plan["tasks"][0]["task_id"] = "not-in-frame"
    with pytest.raises(ValueError):
        batch.plan_jobs(
            "retail" if change == "domain" else "all",
            1 if change == "count" else None,
            2,
            42,
            plan,
        )


def test_pass_k_uses_reliability_and_excludes_incomplete_tasks():
    jobs = batch.plan_jobs("mock", 2, 3, 42)
    results = [dict(job, reward=0 if i == 2 else 1) for i, job in enumerate(jobs)]
    scores = batch.summarize(jobs, results, 3)
    assert scores["overall"]["pass^k"] == pytest.approx(
        {"1": 5 / 6, "2": 2 / 3, "3": 1 / 2}
    )
    incomplete = batch.summarize(jobs, results[:5], 3)
    assert incomplete["overall"]["tasks"] == 1
    assert not incomplete["complete"]
    assert incomplete["completed_trials"] == 5


@pytest.fixture
def executor(monkeypatch, tmp_path):
    profile = LocalVoiceProfile.model_validate_json(
        (repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )
    models = {}
    for requirement in profile.roles.values():
        if requirement.model_id in models:
            models[requirement.model_id].features |= requirement.features
        else:
            models[requirement.model_id] = requirement.model_copy(deep=True)
    capabilities = RuntimeCapabilities(
        runtime="fixture", revision="test", models=models
    )
    monkeypatch.setattr(
        batch,
        "SpeechSwiftBackend",
        lambda _: SimpleNamespace(
            capabilities=lambda: capabilities,
            close=lambda: None,
        ),
    )
    provenance = {
        "benchmark": {"sha256": "source"},
        "baseline": {"frozen": True},
        "native": {
            "configuration": {"runtimeRevision": "native", "assets": {"model": "hash"}},
            "voxcpm2_parameter_elements": {"float16": 100},
            "pid": 1,
        },
    }
    monkeypatch.setattr(batch, "run_provenance", lambda _: deepcopy(provenance))
    calls = []
    state = SimpleNamespace(fail_trial=None)

    def run(**kwargs):
        calls.append(kwargs["seed"])
        path = kwargs["output"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "simulation.json").write_text('{"complete":true}')
        if kwargs["seed"] == state.fail_trial:
            raise RuntimeError("inference unavailable")
        return {
            "domain": kwargs["domain"],
            "task_id": kwargs["task_id"],
            "reward": 1,
            "termination": "user_stop",
        }

    monkeypatch.setattr(batch, "run_local_task", run)
    return SimpleNamespace(
        calls=calls,
        state=state,
        provenance=provenance,
        output=tmp_path / "batch",
        kwargs=dict(
            domain="mock",
            num_tasks=1,
            trials=2,
            seed=42,
            base_url="http://localhost:8080",
            output=tmp_path / "batch",
        ),
    )


def test_failed_inference_is_unscored_and_resume_skips_completed_trials(executor):
    executor.state.fail_trial = 43
    with pytest.raises(RuntimeError, match="inference unavailable"):
        batch.run_batch(**executor.kwargs)
    summary = json.loads((executor.output / "summary.json").read_text())
    assert summary["completed_trials"] == 1
    assert summary["overall"]["tasks"] == 0
    executor.state.fail_trial = None
    result = batch.run_batch(**executor.kwargs, resume=True)
    assert executor.calls == [42, 43, 43]
    assert result["complete"]
    assert result["overall"]["pass^k"] == {"1": 1, "2": 1}
    assert list((executor.output / "interrupted").iterdir())


@pytest.mark.parametrize("drift", ["source", "models", "artifacts", "settings"])
def test_resume_rejects_drift_before_executing_models(executor, drift):
    batch.run_batch(**executor.kwargs)
    kwargs = dict(executor.kwargs, resume=True)
    if drift == "source":
        executor.provenance["benchmark"]["sha256"] = "changed"
    elif drift == "models":
        executor.provenance["native"]["configuration"]["assets"]["model"] = "changed"
    elif drift == "settings":
        kwargs["max_seconds"] = 12
    else:
        next(executor.output.rglob("simulation.json")).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        batch.run_batch(**kwargs)
    assert executor.calls == [42, 43]


def test_runtime_process_restart_does_not_invalidate_identical_batch(executor):
    batch.run_batch(**executor.kwargs)
    executor.provenance["native"]["pid"] = 123
    assert batch.run_batch(**executor.kwargs, resume=True)["complete"]
    assert executor.calls == [42, 43]


def test_existing_batch_requires_explicit_resume(executor):
    batch.run_batch(**executor.kwargs)
    with pytest.raises(ValueError, match="already exists"):
        batch.run_batch(**executor.kwargs)

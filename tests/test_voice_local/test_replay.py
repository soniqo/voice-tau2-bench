import json
import shutil

import pytest

from tau2.voice.baseline import repository_root, verify_baseline
from tau2.voice.replay import ReplayMismatch, canonical_trace, run_replay


@pytest.mark.parametrize(
    "name,reward", [("voice-create-task", 1.0), ("voice-tool-flow", 0.0)]
)
def test_replay_matches_original_upstream_trace(name, reward):
    fixture = repository_root() / f"data/tau2/replays/{name}.json"
    simulation, deliveries = run_replay(fixture)
    expected = json.loads(fixture.with_suffix(".expected.json").read_text())
    assert canonical_trace(simulation, deliveries) == expected
    assert simulation.reward_info.reward == reward
    assert simulation.info["replay_only"] is True


def test_replay_detects_wrong_tool_delivery_tick(tmp_path):
    fixture = repository_root() / "data/tau2/replays/voice-tool-flow.json"
    script = json.loads(fixture.read_text())
    script["agent"]["4"]["tool_results"] = []
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(script))
    with pytest.raises(ReplayMismatch, match="assistant tick 4"):
        run_replay(changed)


def test_baseline_freezes_all_three_english_task_sets():
    manifest = verify_baseline()
    assert manifest["frame"]["tasks"] == {"airline": 50, "retail": 114, "telecom": 114}
    assert manifest["frame"]["total_tasks"] == 278
    for domain, count in manifest["frame"]["tasks"].items():
        subset = repository_root() / f"data/tau2/task_subsets/{domain}_{count}.json"
        assert len(json.loads(subset.read_text())["task_ids"]) == count


def test_baseline_detects_input_drift(tmp_path):
    manifest = verify_baseline()
    for relative in ["data/tau2/voice_local/soniqo-local-v1.json", *manifest["inputs"]]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root() / relative, destination)
    changed = "data/tau2/domains/airline/policy.md"
    with (tmp_path / changed).open("a") as file:
        file.write("\nchanged\n")
    with pytest.raises(ValueError, match="airline/policy.md"):
        verify_baseline(tmp_path)


def test_baseline_checks_the_data_root_used_by_domain_tools(monkeypatch, tmp_path):
    import tau2.utils.utils as utils

    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="airline/db.json"):
        verify_baseline()

import json
from copy import deepcopy

import pytest

from tau2.voice import qualification
from tau2.voice.baseline import repository_root
from tau2.voice.runtime_profile import load_profile


@pytest.fixture
def certificate(tmp_path):
    profile = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v2.json"
    )
    provenance = {
        "benchmark": {"sha256": "working source", "git_head": "commit"},
        "native": {
            "configuration": {"assets": {"pinned": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
    }
    suite = qualification.qualification_suite()
    report = {
        "identity": qualification.qualification_identity(profile, provenance),
        "passed": True,
        "cases": [
            {"id": case["id"], "passed": True}
            for section in ("caller", "judge")
            for case in suite[section]
        ],
    }
    path = tmp_path / "qualified.json"
    path.write_text(json.dumps(report))
    return profile, provenance, report, path


def test_v2_requires_passing_matching_qualification(certificate):
    profile, provenance, _, path = certificate
    with pytest.raises(ValueError, match="requires --qualification"):
        qualification.require_qualification(None, profile, provenance)
    assert qualification.require_qualification(path, profile, provenance)["sha256"]
    changed = deepcopy(provenance)
    changed["benchmark"]["git_head"] = "documentation-only commit"
    assert qualification.require_qualification(path, profile, changed)
    changed["benchmark"]["sha256"] = "new code"
    with pytest.raises(ValueError, match="no longer matches"):
        qualification.require_qualification(path, profile, changed)


@pytest.mark.parametrize(
    "change", ["failed", "missing", "duplicate", "model", "assets"]
)
def test_invalid_or_partial_qualification_cannot_enable_benchmark(certificate, change):
    profile, provenance, report, path = certificate
    if change == "failed":
        report["cases"][0]["passed"] = False
    elif change == "missing":
        report["cases"].pop()
    elif change == "duplicate":
        report["cases"][0] = report["cases"][1]
    elif change == "model":
        profile.roles["caller"].model_id = "different caller"
    else:
        provenance["native"]["configuration"]["assets"]["pinned"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification failed"):
        qualification.require_qualification(path, profile, provenance)


def test_historical_profiles_do_not_require_new_caller_policy():
    profile = load_profile()
    assert qualification.require_qualification(None, profile, {}) is None

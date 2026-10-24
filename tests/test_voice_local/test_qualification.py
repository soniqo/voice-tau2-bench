import json
from copy import deepcopy

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
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


def test_qualification_checks_answer_after_one_permitted_fixture_read(monkeypatch):
    case = {
        "id": "caller-tool-continuation",
        "scenario": "Create an appointment named Annual Checkup.",
        "history": [["user", "What should it be called?"]],
        "decision": "CONTINUE",
        "reply": {"notification_tools": True, "must_contain": ["annual", "checkup"]},
    }
    monkeypatch.setattr(
        qualification,
        "qualification_suite",
        lambda: {"suite": "fixture", "caller": [case], "judge": []},
    )
    assessment = AssistantMessage(
        role="assistant",
        content=json.dumps(
            {
                "reasoning": "title question remains open",
                "agent_evidence": None,
                "decision": "CONTINUE",
            }
        ),
    )
    replies = iter(
        [
            assessment,
            assessment,
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="read", name="check_notifications", arguments={})
                ],
            ),
            assessment,
            AssistantMessage(role="assistant", content="Annual Checkup, please."),
        ]
    )
    sent = []

    class Backend:
        def generate(self, model, messages, **kwargs):
            sent.append(messages)
            return next(replies)

    profile = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v2.json"
    )
    provenance = {
        "benchmark": {"sha256": "source"},
        "native": {
            "configuration": {"assets": {"fixture": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
    }
    report = qualification.qualify(Backend(), profile, provenance)
    assert report["passed"] is True
    assert report["cases"][0]["fixture_tool_result"]["content"] == "[]"
    assert '"speaker": "customer_tool"' in sent[-1][-1].content


def test_qualification_labels_device_observations_as_customer_tools(monkeypatch):
    case = {
        "id": "device-intermediate-observation",
        "scenario": "Restore my mobile internet.",
        "history": [
            ["user", "Please turn off airplane mode."],
            ["tool", '{"airplane_mode": false, "signal": "Excellent"}'],
        ],
        "decision": "CONTINUE",
    }
    monkeypatch.setattr(
        qualification,
        "qualification_suite",
        lambda: {"suite": "fixture", "caller": [case], "judge": []},
    )
    sent = []

    class Backend:
        def generate(self, model, messages, **kwargs):
            sent.append(json.loads(messages[-1].content))
            return AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "reasoning": "Device setting alone does not prove connectivity.",
                        "agent_evidence": None,
                        "decision": "CONTINUE",
                    }
                ),
            )

    profile = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v2.json"
    )
    provenance = {
        "benchmark": {"sha256": "source"},
        "native": {
            "configuration": {"assets": {"fixture": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
    }
    report = qualification.qualify(Backend(), profile, provenance)
    assert report["passed"] is True
    observation = sent[0]["conversation"][-1]
    assert observation["speaker"] == "customer_tool"
    assert json.loads(observation["content"])["signal"] == "Excellent"
    assert observation["tool_call_id"] == "fixture-tool-1"

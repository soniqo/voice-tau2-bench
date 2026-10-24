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


def terminal(decision="CONTINUE", evidence=None):
    return AssistantMessage(
        role="assistant",
        content=json.dumps(
            {"reasoning": "fixture", "agent_evidence": evidence, "decision": decision}
        ),
    )


def provenance():
    return {
        "benchmark": {"sha256": "source"},
        "native": {
            "configuration": {"assets": {"pinned": True}},
            "voxcpm2_parameter_elements": {"float16": 1},
        },
    }


def single_case(monkeypatch, case):
    monkeypatch.setattr(
        qualification,
        "qualification_suite",
        lambda: {"suite": "fixture", "caller": [case], "judge": []},
    )
    return load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v2.json"
    )


def test_qualification_checks_answer_after_one_permitted_fixture_read(monkeypatch):
    case = {
        "id": "read-before-answer",
        "scenario": "Create an appointment named Annual Checkup.",
        "history": [["user", "What should it be called?"]],
        "decision": "CONTINUE",
        "reply": {"notification_tools": True, "must_contain": ["annual", "checkup"]},
    }
    profile = single_case(monkeypatch, case)
    sent = []

    class Backend:
        def generate(self, model, messages, **kwargs):
            sent.append(messages)
            if kwargs["call_name"].startswith("caller_terminal_v2"):
                return terminal()
            if kwargs["call_name"] == "caller_tool_route_v2":
                history = json.loads(messages[-1].content)["conversation"]
                return AssistantMessage(
                    role="assistant",
                    content="SPEAK"
                    if any(row["speaker"] == "customer_tool" for row in history)
                    else "TOOL",
                )
            if kwargs.get("tool_choice") == "required":
                return AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="read", name="check_notifications", arguments={})
                    ],
                )
            return AssistantMessage(role="assistant", content="Annual Checkup, please.")

    report = qualification.qualify(Backend(), profile, provenance())
    assert report["passed"] is True
    assert report["cases"][0]["fixture_tool_result"]["content"] == "[]"
    assert '"speaker": "customer_tool"' in sent[-1][-1].content


def test_qualification_labels_device_observations_as_customer_tools(monkeypatch):
    case = {
        "id": "observations",
        "scenario": "Restore internet.",
        "history": [
            ["user", "Please turn off airplane mode."],
            ["tool", "Airplane Mode: OFF"],
        ],
        "decision": "CONTINUE",
    }
    profile = single_case(monkeypatch, case)
    received = []

    class Backend:
        def generate(self, model, messages, **kwargs):
            received.append(json.loads(messages[-1].content))
            return terminal()

    report = qualification.qualify(Backend(), profile, provenance())
    assert report["passed"] is True
    assert received[0]["conversation"][-1]["speaker"] == "customer_tool"


def test_qualification_executes_real_device_write_with_full_customer_toolset(
    monkeypatch,
):
    case = {
        "id": "actual-device-write",
        "scenario": "Restore internet.",
        "toolset": "telecom",
        "device": {"airplane_mode": True},
        "history": [["user", "Please turn off Airplane Mode."]],
        "decision": "CONTINUE",
        "reply": {
            "tool": "toggle_airplane_mode",
            "arguments": {},
            "device_after": {"airplane_mode": False},
        },
    }
    profile = single_case(monkeypatch, case)

    class Backend:
        def generate(self, model, messages, **kwargs):
            if kwargs["call_name"].startswith("caller_terminal_v2"):
                return terminal()
            if kwargs["call_name"] == "caller_tool_route_v2":
                return AssistantMessage(role="assistant", content="TOOL")
            assert kwargs["tool_choice"] == "required" and len(kwargs["tools"]) == 30
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="write", name="toggle_airplane_mode", arguments={})
                ],
            )

    report = qualification.qualify(Backend(), profile, provenance())
    assert report["passed"] is True
    assert report["cases"][0]["device_after"] == {"airplane_mode": False}
    assert "Airplane Mode is now OFF" in report["cases"][0]["fixture_tool_result"]


def test_qualification_rejects_terminal_control_after_permitted_fixture_read(
    monkeypatch,
):
    case = {
        "id": "early-stop-after-read",
        "scenario": "More help needed.",
        "history": [["user", "Done."]],
        "decision": "CONTINUE",
        "reply": {"notification_tools": True, "must_contain": []},
    }
    profile = single_case(monkeypatch, case)

    class Backend:
        def generate(self, model, messages, **kwargs):
            if kwargs["call_name"].startswith("caller_terminal_v2"):
                history = json.loads(messages[-1].content)["conversation"]
                return (
                    terminal("COMPLETE", "Done.")
                    if any(row["speaker"] == "customer_tool" for row in history)
                    else terminal()
                )
            if kwargs["call_name"] == "caller_tool_route_v2":
                return AssistantMessage(role="assistant", content="TOOL")
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="read", name="check_notifications", arguments={})
                ],
            )

    report = qualification.qualify(Backend(), profile, provenance())
    assert report["passed"] is False
    assert report["cases"][0]["final_action"] == "stop"

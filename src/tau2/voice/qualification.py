"""Preflight calibration on hand-labelled fixtures outside the frozen task frame."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.inference import InferenceBackends, use_inference_backends
from tau2.user.user_simulator_base import STOP
from tau2.voice.baseline import repository_root
from tau2.voice.caller_policy import LOCAL_CALLER_GUIDELINES, QualifiedCallerBackend


def qualification_suite():
    return json.loads(
        (repository_root() / "data/tau2/voice_local/qualification-v2.json").read_text()
    )


def qualification_identity(profile, provenance):
    suite = qualification_suite()
    return {
        "suite": suite["suite"],
        "suite_sha256": hashlib.sha256(
            json.dumps(suite, sort_keys=True).encode()
        ).hexdigest(),
        "benchmark_source_sha256": provenance["benchmark"]["sha256"],
        "caller_policy": profile.execution.caller_policy,
        "device_grounding": profile.execution.device_grounding,
        "roles": {
            role: profile.roles[role].model_dump(mode="json")
            for role in ("caller", "listener", "judge")
        },
        "native_configuration": deepcopy(provenance["native"]["configuration"]),
        "voxcpm2_parameter_elements": deepcopy(
            provenance["native"]["voxcpm2_parameter_elements"]
        ),
    }


def require_qualification(path: Path | None, profile, provenance):
    if not profile.execution.require_qualification:
        return None
    if path is None:
        raise ValueError(
            "The v2 profile requires --qualification from a passing tau-voice qualify run"
        )
    report = json.loads(path.read_text())
    suite = qualification_suite()
    expected_ids = {
        case["id"] for section in ("caller", "judge") for case in suite[section]
    }
    rows = report.get("cases", [])
    if (
        report.get("identity") != qualification_identity(profile, provenance)
        or report.get("passed") is not True
        or len(rows) != len(expected_ids)
        or {row.get("id") for row in rows} != expected_ids
        or not all(row.get("passed") is True and not row.get("error") for row in rows)
    ):
        raise ValueError(
            "Caller/judge qualification failed or no longer matches these sources, models and policy"
        )
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity": report["identity"],
    }


def toggle_mobile_data(enabled: bool) -> str:
    """Enable or disable mobile data on the customer's phone."""
    return str(enabled)


def run_speed_test() -> str:
    """Run a speed test on the customer's phone."""
    return "20 Mbps"


def check_notifications() -> str:
    """Check notifications on the customer's device."""
    return "[]"


def dismiss_notification(notification_id: str) -> str:
    """Dismiss a notification on the customer's device."""
    return "dismissed"


def fixture_history(history):
    messages = []
    for index, event in enumerate(history):
        role, text, *metadata = event
        if role != "tool":
            speech_metadata = metadata[0] if metadata else {}
            messages.append(
                (UserMessage if role == "user" else AssistantMessage)(
                    role=role,
                    content=text,
                    **{
                        key: speech_metadata[key]
                        for key in ("is_final_chunk", "utterance_ids", "chunk_id")
                        if key in speech_metadata
                    },
                )
            )
            continue
        call_id = f"fixture-tool-{index}"
        details = metadata[0] if metadata else {}
        if "name" in details:
            messages.append(
                AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name=details["name"],
                            arguments=details.get("arguments", {}),
                        )
                    ],
                )
            )
        messages.append(
            ToolMessage(
                role="tool",
                content=text,
                id=call_id,
                requestor="user",
                error=details.get("error", False),
                execution_status=details.get("execution_status", "executed"),
            )
        )
    return messages


def qualify(backend, profile, provenance):
    if profile.execution.caller_policy != "terminal-v2":
        raise ValueError(
            "Qualification applies to the explicit terminal-v2 caller profile"
        )
    suite = qualification_suite()
    rows = []
    for case in suite["caller"]:
        row = {"id": case["id"], "component": "caller", "passed": False}
        try:
            expected = case.get("reply", {})
            fixture_environment = None
            if case.get("toolset") == "telecom":
                from tau2.domains.telecom.environment import get_environment

                fixture_environment = get_environment()
                device = fixture_environment.user_tools.db.device
                fixture_environment.user_tools.db.device = type(device).model_validate(
                    {**device.model_dump(), **case.get("device", {})}
                )
                tools = fixture_environment.get_user_tools()
            else:
                tools = (
                    [Tool(toggle_mobile_data), Tool(run_speed_test)]
                    if "tool" in expected
                    else [Tool(check_notifications), Tool(dismiss_notification)]
                    if expected.get("notification_tools")
                    else []
                )
            caller = QualifiedCallerBackend(
                backend,
                case["scenario"],
                device_grounding=profile.execution.device_grounding
                and case.get("toolset") == "telecom",
            )
            messages = fixture_history(case["history"])
            decision, _ = caller.assess(
                profile.roles["caller"].model_id, messages, tools=tools
            )
            row.update(expected=case["decision"], actual=decision)
            row["assessment"] = caller.decisions[-1]
            row["assessments"] = caller.decisions
            # Both terminal labels produce the same caller STOP control. Qualify
            # the observable action, while retaining its diagnostic label.
            row["expected_action"] = (
                "continue" if case["decision"] == "CONTINUE" else "stop"
            )
            row["actual_action"] = "continue" if decision == "CONTINUE" else "stop"
            row["passed"] = row["actual_action"] == row["expected_action"]
            if "reply" in case and decision == "CONTINUE":
                # Exercise the actual caller generation path as well as its controller.
                from tau2.data_model.message import SystemMessage
                from tau2.user.prompts import build_user_system_prompt

                system = build_user_system_prompt(
                    instructions=case["scenario"],
                    persona_config=PersonaConfig(),
                    mode="voice",
                    guidelines=LOCAL_CALLER_GUIDELINES,
                )
                history = [SystemMessage(role="system", content=system), *messages]
                response = caller.generate(
                    profile.roles["caller"].model_id,
                    history,
                    tools=tools,
                    temperature=0,
                    max_tokens=1024,
                    call_name="user_streaming_response",
                    **{
                        key: value
                        for key, value in case.get("caller_timing", {}).items()
                        if key in ("agent_silence_seconds", "caller_idle_seconds")
                    },
                )
                permitted_read = expected.get("permitted_read") or (
                    "check_notifications"
                    if expected.get("notification_tools")
                    else None
                )
                if permitted_read and response.is_tool_call():
                    calls = response.tool_calls
                    # A permitted read may precede the answer. Exercise its real
                    # fixture result once; invented tools, writes, or loops fail.
                    if (
                        len(calls) == 1
                        and calls[0].name == permitted_read
                        and calls[0].arguments == {}
                    ):
                        row["intermediate_response"] = response.model_dump(mode="json")
                        selected = next(
                            tool for tool in tools if tool.name == permitted_read
                        )
                        observed = selected()
                        result = ToolMessage(
                            role="tool",
                            id=calls[0].id,
                            requestor="user",
                            content=observed
                            if isinstance(observed, str)
                            else json.dumps(observed),
                        )
                        row["fixture_tool_result"] = result.model_dump(mode="json")
                        history = [*history, response, result]
                        response = caller.generate(
                            profile.roles["caller"].model_id,
                            history,
                            tools=tools,
                            temperature=0,
                            max_tokens=1024,
                            call_name="user_streaming_response",
                        )
                row["response"] = response.model_dump(mode="json")
                row["final_action"] = (
                    "stop"
                    if STOP in (response.content or "")
                    else "tool"
                    if response.is_tool_call()
                    else "speak"
                    if (response.content or "").strip()
                    else "wait"
                )
                row["passed"] &= row["final_action"] != "stop"
                if "tool" in expected:
                    calls = response.tool_calls or []
                    row["passed"] &= (
                        len(calls) == 1
                        and calls[0].name == expected["tool"]
                        and calls[0].arguments == expected["arguments"]
                    )
                    if row["passed"] and (
                        "device_after" in expected or "spoken_result" in expected
                    ):
                        selected = next(
                            tool for tool in tools if tool.name == calls[0].name
                        )
                        row["fixture_tool_result"] = selected(**calls[0].arguments)
                        if "device_after" in expected:
                            device = fixture_environment.user_tools.db.device
                            row["device_after"] = {
                                key: getattr(device, key)
                                for key in expected["device_after"]
                            }
                            row["passed"] &= (
                                row["device_after"] == expected["device_after"]
                            )
                        if "spoken_result" in expected:
                            report = caller.generate(
                                profile.roles["caller"].model_id,
                                [
                                    *history,
                                    response,
                                    ToolMessage(
                                        role="tool",
                                        requestor="user",
                                        id=calls[0].id,
                                        content=row["fixture_tool_result"],
                                    ),
                                ],
                                tools=tools,
                                temperature=0,
                                max_tokens=1024,
                                call_name="user_streaming_response",
                            )
                            row["spoken_response"] = report.model_dump(mode="json")
                            row["passed"] &= (
                                not report.is_tool_call()
                                and report.content == expected["spoken_result"]
                            )
                elif expected.get("no_speech"):
                    row["passed"] &= row["final_action"] == "wait"
                elif expected.get("must_speak"):
                    row["passed"] &= row["final_action"] == "speak"
                elif "device_unchanged" in expected:
                    device = fixture_environment.user_tools.db.device
                    row["device_after"] = {
                        key: getattr(device, key)
                        for key in expected["device_unchanged"]
                    }
                    row["passed"] &= (
                        not response.is_tool_call()
                        and bool((response.content or "").strip())
                        and row["device_after"] == expected["device_unchanged"]
                    )
                elif expected.get("unknown_information"):
                    text = response.content or ""
                    row["passed"] &= (
                        not response.is_tool_call()
                        and bool(
                            re.search(
                                r"(?:don.t|not) (?:\w+ ){0,3}(?:know|have|remember)|unsure|unknown",
                                text,
                                re.I,
                            )
                        )
                        and not bool(re.search(r"\d{3,}", text))
                    )
                else:
                    row["passed"] &= not response.is_tool_call() and all(
                        word.lower() in (response.content or "").lower()
                        for word in expected["must_contain"]
                    )
        except Exception as error:
            row["passed"] = False
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    scoring = CommunicateScoring(
        scorer="semantic",
        judge=SemanticCommunicateSettings(
            model=profile.roles["judge"].model_id,
            model_args={"temperature": 0, "max_tokens": 1024},
        ),
    )
    with use_inference_backends(
        InferenceBackends(chat=backend, synthesis=backend, transcription=backend)
    ):
        for case in suite["judge"]:
            row = {
                "id": case["id"],
                "component": "judge",
                "passed": False,
                "expected": case["met"],
            }
            try:
                history = (
                    [AssistantMessage(role="assistant", content=case["said"])]
                    if case["said"]
                    else [UserMessage(role="user", content=case["caller_said"])]
                )
                check = CommunicateEvaluator.evaluate_communicate_info(
                    history, [case["expected"]], scoring=scoring
                )[0]
                row.update(
                    actual=check.met,
                    justification=check.justification,
                    passed=check.met == case["met"],
                )
            except Exception as error:
                row["passed"] = False
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
    return {
        "schema_version": 1,
        "identity": qualification_identity(profile, provenance),
        "passed": all(row["passed"] for row in rows),
        "cases": rows,
    }

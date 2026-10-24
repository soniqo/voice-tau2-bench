import json

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.voice.caller_policy import CallerPolicyError, QualifiedCallerBackend


class Backend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def generate(self, model, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if (
            kwargs.get("call_name", "").startswith("caller_terminal_v2")
            and isinstance(reply, str)
            and reply in {"CONTINUE", "COMPLETE", "REFUSED"}
        ):
            reply = json.dumps(
                {
                    "reasoning": "fixture",
                    "agent_evidence": None if reply == "CONTINUE" else "Done.",
                    "decision": reply,
                    "customer_action": None,
                }
            )
        return (
            reply
            if isinstance(reply, AssistantMessage)
            else AssistantMessage(role="assistant", content=reply)
        )


@pytest.mark.parametrize("decision", ["COMPLETE", "REFUSED"])
def test_only_model_assessed_terminal_state_emits_caller_stop(decision):
    backend = Backend([decision])
    caller = QualifiedCallerBackend(backend, "customer-private goal")
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content="Done.")],
        call_name="user_streaming_response",
    )
    assert reply.content == "###STOP###"
    assert len(backend.calls) == 1
    assert "customer-private goal" in backend.calls[0][0][-1].content
    assert reply.raw_data["decision"] == decision


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "probably done",
        "COMPLETE because it is done",
        AssistantMessage(
            role="assistant", tool_calls=[ToolCall(name="write", arguments={})]
        ),
    ],
)
def test_malformed_terminal_assessment_propagates_without_fallback(reply):
    caller = QualifiedCallerBackend(Backend([reply]), "scenario")
    with pytest.raises(CallerPolicyError, match="Invalid caller terminal"):
        caller.assess("model", [])


def test_continue_preserves_dialogue_and_coalesces_caller_system_instructions():
    backend = Backend(["CONTINUE", "I do not know that number."])
    caller = QualifiedCallerBackend(backend, "scenario")
    history = [
        SystemMessage(role="system", content="caller instructions"),
        AssistantMessage(role="assistant", content="Please help."),
        UserMessage(role="user", content="Your number?"),
        SystemMessage(role="system", content="role reminder"),
    ]
    reply = caller.generate(
        "model", history, tools=[], call_name="user_streaming_response"
    )
    assert reply.content == "I do not know that number."
    messages, args = backend.calls[-1]
    assert [message.role for message in messages] == ["system", "user"]
    assert '"speaker": "customer", "content": "Please help."' in messages[1].content
    assert (
        '"speaker": "service_agent", "content": "Your number?"' in messages[1].content
    )
    assert (
        "caller instructions" in messages[0].content
        and "role reminder" in messages[0].content
    )
    assert args["tools"] == []


def test_premature_caller_stop_invalidates_trial():
    caller = QualifiedCallerBackend(Backend(["CONTINUE", "###STOP###"]), "scenario")
    with pytest.raises(CallerPolicyError, match="outstanding"):
        caller.generate("model", [], call_name="user_streaming_response")


def test_listener_and_judge_calls_bypass_terminal_controller():
    backend = Backend(["judge response"])
    caller = QualifiedCallerBackend(backend, "private goal")
    messages = [UserMessage(role="user", content="fact check")]
    caller.generate("model", messages, call_name="communicate_info_eval")
    assert backend.calls[0][0] is messages
    assert len(backend.calls) == 1
    assert not caller.decisions


@pytest.mark.parametrize(
    "evidence",
    [
        "I will cancel it now.",
        "We can change it.",
        "I'll update it later.",
        "I am going to update it.",
    ],
)
def test_future_offer_cannot_end_call_even_when_model_proposes_completion(evidence):
    response = json.dumps(
        {
            "reasoning": "fixture false completion",
            "agent_evidence": evidence,
            "decision": "COMPLETE",
        }
    )
    caller = QualifiedCallerBackend(Backend([response]), "Cancel my subscription")
    decision, _ = caller.assess("model", [UserMessage(role="user", content=evidence)])
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["proposed_decision"] == "COMPLETE"
    assert caller.decisions[-1]["guard"] == "pending_agent_offer"


def test_customer_only_quote_cannot_support_terminal_control():
    response = json.dumps(
        {
            "reasoning": "fixture false completion",
            "agent_evidence": "It was cancelled.",
            "decision": "COMPLETE",
        }
    )
    backend = Backend([response, response])
    caller = QualifiedCallerBackend(backend, "Cancel my subscription")
    with pytest.raises(CallerPolicyError, match="quote from the service agent"):
        caller.assess(
            "model",
            [
                UserMessage(role="user", content="I am checking."),
                AssistantMessage(role="assistant", content="It was cancelled."),
            ],
        )
    assert len(backend.calls) == 2
    assert not caller.decisions


def assessment(decision, evidence=None, customer_action=None, **metadata):
    return AssistantMessage(
        role="assistant",
        content=json.dumps(
            {
                "reasoning": "fixture assessment",
                "agent_evidence": evidence,
                "decision": decision,
                "customer_action": customer_action,
            }
        ),
        **metadata,
    )


def test_missing_terminal_quote_is_reassessed_once_without_forcing_stop():
    backend = Backend(
        [
            assessment("COMPLETE"),
            "CONTINUE",
            "Airplane mode is off. Internet still does not work.",
        ]
    )
    caller = QualifiedCallerBackend(backend, "Restore my mobile internet.")
    history = [UserMessage(role="user", content="Please turn off airplane mode.")]
    reply = caller.generate("model", history, call_name="user_streaming_response")
    assert "still does not work" in reply.content
    assert [args["call_name"] for _, args in backend.calls] == [
        "caller_terminal_v2",
        "caller_terminal_v2_repair",
        "user_streaming_response",
    ]
    first = json.loads(backend.calls[0][0][-1].content)
    repair = json.loads(backend.calls[1][0][-1].content)
    assert repair["scenario"] == first["scenario"]
    assert repair["conversation"] == first["conversation"]
    assert repair["rejected_assessment"]["decision"] == "COMPLETE"
    assert "Do not invent a quote" in repair["feedback"]
    assert caller.decisions[-1]["decision"] == "CONTINUE"
    assert len(caller.decisions[-1]["rejected_assessments"]) == 1


def test_reassessment_may_stop_only_with_heard_agent_evidence_and_counts_both_calls():
    backend = Backend(
        [
            assessment(
                "COMPLETE",
                cost=0.2,
                usage={"completion_tokens": 15},
                generation_time_seconds=1.25,
            ),
            assessment(
                "COMPLETE",
                "Your subscription has been cancelled.",
                cost=0.3,
                usage={"completion_tokens": 20},
                generation_time_seconds=2.5,
            ),
        ]
    )
    caller = QualifiedCallerBackend(backend, "Cancel my subscription.")
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content="Your subscription has been cancelled.")],
        call_name="user_streaming_response",
    )
    assert reply.content == "###STOP###"
    assert reply.cost == pytest.approx(0.5)
    assert reply.usage == {"completion_tokens": 35}
    assert reply.generation_time_seconds == 3.75
    assert len(backend.calls) == 2


@pytest.mark.parametrize("invalid", ["not JSON", RuntimeError("native failed")])
@pytest.mark.parametrize("during_reassessment", [False, True])
def test_malformed_or_failed_inference_is_not_retried(invalid, during_reassessment):
    backend = Backend(
        [assessment("COMPLETE"), invalid] if during_reassessment else [invalid]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    expected = CallerPolicyError if isinstance(invalid, str) else RuntimeError
    with pytest.raises(expected):
        caller.assess("model", [])
    assert len(backend.calls) == 1 + during_reassessment
    assert not caller.decisions


def test_repaired_future_offer_still_cannot_end_call():
    backend = Backend(
        [assessment("COMPLETE"), assessment("COMPLETE", "I will cancel it now.")]
    )
    caller = QualifiedCallerBackend(backend, "Cancel my subscription.")
    decision, _ = caller.assess(
        "model", [UserMessage(role="user", content="I will cancel it now.")]
    )
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "pending_agent_offer"
    assert len(caller.decisions[-1]["rejected_assessments"]) == 1


def test_quote_cannot_be_assembled_across_separate_agent_statements():
    invalid = assessment("COMPLETE", "cancelled. Your refund")
    backend = Backend([invalid, invalid])
    caller = QualifiedCallerBackend(backend, "Cancel and refund.")
    with pytest.raises(CallerPolicyError, match="after one reassessment"):
        caller.assess(
            "model",
            [
                UserMessage(role="user", content="Your subscription was cancelled."),
                AssistantMessage(role="assistant", content="What about the refund?"),
                UserMessage(role="user", content="Your refund is pending."),
            ],
        )
    assert len(backend.calls) == 2


def set_mobile_data(enabled: bool) -> str:
    """Set mobile data to the requested state on the customer's phone."""
    return f"Mobile data enabled: {enabled}"


def test_controller_action_returns_real_tool_call_without_claiming_execution():
    executions = []

    def toggle_airplane_mode() -> str:
        """Toggle airplane mode on the customer's phone."""
        executions.append("executed")
        return "Airplane Mode is now OFF."

    instruction = "Please turn off airplane mode."
    backend = Backend(
        [
            assessment(
                "CONTINUE",
                "Switch airplane mode off, please.",
                customer_action={
                    "name": "toggle_airplane_mode",
                    "arguments": {},
                    "instruction_id": 0,
                },
                usage={"completion_tokens": 31},
            )
        ]
    )
    caller = QualifiedCallerBackend(backend, "Restore mobile internet.")
    tool = Tool(toggle_airplane_mode)
    reply = caller.generate(
        "model",
        [
            SystemMessage(role="system", content="caller reminder"),
            UserMessage(role="user", content=instruction),
        ],
        tools=[tool],
        call_name="user_streaming_response",
    )
    assert not executions
    assert reply.content is None
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.id and call.name == tool.name and call.arguments == {}
    assert reply.usage == {"completion_tokens": 31}
    assert len(backend.calls) == 1
    assert tool(**call.arguments) == "Airplane Mode is now OFF."
    assert executions == ["executed"]
    assert caller.decisions[-1]["customer_action_evidence"] == {
        "message_id": 0,
        "quote": instruction,
    }
    assert "DECLARED_CUSTOMER_TOOLS" in backend.calls[0][0][0].content


@pytest.mark.parametrize(
    "decision,name,arguments,instruction_id,error",
    [
        ("COMPLETE", "set_mobile_data", {"enabled": True}, 0, "terminal decision"),
        ("CONTINUE", "agent_only_tool", {}, 0, "not declared"),
        ("CONTINUE", "set_mobile_data", {"enabled": True}, 1, "heard agent"),
        ("CONTINUE", "set_mobile_data", {"enabled": True}, 99, "heard agent"),
        ("CONTINUE", "set_mobile_data", {"enabled": True}, True, "valid integer"),
        ("CONTINUE", "set_mobile_data", {"enabled": "yes"}, 0, "valid boolean"),
        ("CONTINUE", "set_mobile_data", {}, 0, "required"),
        (
            "CONTINUE",
            "set_mobile_data",
            {"enabled": True, "extra": 1},
            0,
            "undeclared arguments",
        ),
    ],
)
def test_invalid_customer_action_is_unscored_without_repair(
    decision, name, arguments, instruction_id, error
):
    backend = Backend(
        [
            assessment(
                decision,
                "Done.",
                {
                    "name": name,
                    "arguments": arguments,
                    "instruction_id": instruction_id,
                },
            )
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match=error):
        caller.assess(
            "model",
            [
                UserMessage(role="user", content="Done."),
                AssistantMessage(role="assistant", content="Customer claim."),
            ],
            tools=[Tool(set_mobile_data)],
        )
    assert len(backend.calls) == 1
    assert not caller.decisions


@pytest.mark.parametrize(
    "source",
    [
        ToolMessage(role="tool", id="device", requestor="user", content="Done."),
        UserMessage(role="user", content=None),
    ],
)
def test_customer_action_cannot_cite_a_tool_or_empty_message(source):
    backend = Backend(
        [
            assessment(
                "CONTINUE",
                customer_action={
                    "name": "set_mobile_data",
                    "arguments": {"enabled": True},
                    "instruction_id": 0,
                },
            )
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="heard agent instruction"):
        caller.assess("model", [source], tools=[Tool(set_mobile_data)])
    assert not caller.decisions


def test_spoken_response_cannot_bypass_controller_with_an_unplanned_tool():
    backend = Backend(
        [
            "CONTINUE",
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(name="set_mobile_data", arguments={"enabled": True})
                ],
            ),
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="outside its assessed action"):
        caller.generate(
            "model",
            [],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )
    assert backend.calls[-1][1]["tools"] == []
    assert backend.calls[-1][1]["tool_choice"] == "none"


def test_spoken_response_receives_actual_device_result_and_latest_question():
    backend = Backend(["CONTINUE", "Yes, airplane mode is off."])
    caller = QualifiedCallerBackend(backend, "Restore mobile internet.")
    caller.generate(
        "model",
        [
            UserMessage(role="user", content="Is airplane mode off now?"),
            ToolMessage(
                role="tool",
                id="actual-result",
                requestor="user",
                content="Airplane Mode: OFF",
            ),
        ],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    payload = json.loads(backend.calls[-1][0][-1].content.split("\n", 1)[1])
    assert (
        payload["latest_service_agent_statement"]["content"]
        == "Is airplane mode off now?"
    )
    assert payload["latest_customer_tool_result"]["content"] == "Airplane Mode: OFF"
    assert payload["latest_customer_tool_result"] == payload["conversation"][-1]


def test_customer_tools_require_an_explicit_action_or_null_decision():
    backend = Backend(
        [
            json.dumps(
                {"reasoning": "pending", "agent_evidence": None, "decision": "CONTINUE"}
            )
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="customer action decision is required"):
        caller.assess("model", [], tools=[Tool(set_mobile_data)])
    assert len(backend.calls) == 1
    assert not caller.decisions


@pytest.mark.parametrize(
    "question",
    [
        "Is Airplane Mode off now?",
        "Did the connection work?",
        "Have you checked it?",
        "Could you confirm it is off?",
    ],
)
def test_agent_confirmation_question_cannot_prove_task_completion(question):
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", question)]), "Restore internet."
    )
    decision, _ = caller.assess("model", [UserMessage(role="user", content=question)])
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "agent_confirmation_question"
    assert caller.decisions[-1]["proposed_decision"] == "COMPLETE"


def test_completed_action_with_following_question_can_still_end_call():
    confirmation = "Your subscription has been cancelled. Anything else?"
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", confirmation)]),
        "Cancel my subscription. No other questions.",
    )
    decision, _ = caller.assess(
        "model", [UserMessage(role="user", content=confirmation)]
    )
    assert decision == "COMPLETE"

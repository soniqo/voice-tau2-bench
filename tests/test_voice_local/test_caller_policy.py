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


def assessment(decision, evidence=None, **metadata):
    return AssistantMessage(
        role="assistant",
        content=json.dumps(
            {
                "reasoning": "fixture assessment",
                "agent_evidence": evidence,
                "decision": decision,
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


def set_mobile_data(enabled: bool) -> str:
    """Set mobile data on the customer's phone."""
    return str(enabled)


def native_call(name="set_mobile_data", arguments=None):
    return AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="native-id",
                name=name,
                arguments={"enabled": True} if arguments is None else arguments,
            )
        ],
    )


@pytest.mark.parametrize("route", ["TOOL", "TOOL: run_speed_test", "TOOL."])
def test_required_customer_action_uses_native_function_call_before_speaking(route):
    backend = Backend(["CONTINUE", route, native_call()])
    caller = QualifiedCallerBackend(backend, "Restore internet.")
    tool = Tool(set_mobile_data)
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content="Please enable mobile data.")],
        tools=[tool],
        call_name="user_streaming_response",
    )
    assert reply.is_tool_call() and reply.content is None
    assert reply.tool_calls[0].arguments == {"enabled": True}
    assert backend.calls[-1][1]["tool_choice"] == "required"
    assert backend.calls[-1][1]["tools"] == [tool]
    assert caller.decisions[-1]["customer_tool_route"]["mode"] == "TOOL"


def test_unexecuted_action_claim_gets_one_native_tool_reassessment():
    backend = Backend(
        ["CONTINUE", "TOOL", "I have enabled mobile data.", native_call()]
    )
    caller = QualifiedCallerBackend(backend, "Restore internet.")
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content="Please enable mobile data.")],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    assert reply.is_tool_call()
    assert backend.calls[-1][1]["call_name"] == "caller_tool_response_repair_v2"
    assert caller.decisions[-1]["rejected_customer_responses"] == [
        "I have enabled mobile data."
    ]
    first = json.loads(
        backend.calls[-2][0][-1].content.split("Customer context:\n", 1)[1]
    )
    retry = json.loads(
        backend.calls[-1][0][-1].content.split("Customer context:\n", 1)[1]
    )
    assert first["conversation"] == retry["conversation"]
    assert backend.calls[-2][0][-1].content.startswith(
        "The service agent asks you: Please enable mobile data.\n"
    )


def test_repeated_action_claim_without_tool_is_unscored():
    backend = Backend(["CONTINUE", "TOOL", "Done.", "Done."])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="after one reassessment"):
        caller.generate(
            "model",
            [UserMessage(role="user", content="Enable data.")],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )
    assert len(backend.calls) == 4


@pytest.mark.parametrize(
    "reply",
    [
        native_call("agent_only"),
        native_call(arguments={"enabled": "yes"}),
        native_call(arguments={"enabled": True, "extra": 1}),
    ],
)
def test_invalid_native_customer_tool_never_gets_repaired_or_executed(reply):
    backend = Backend(["CONTINUE", "TOOL", reply])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="Invalid customer tool"):
        caller.generate(
            "model",
            [UserMessage(role="user", content="Enable data.")],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )
    assert len(backend.calls) == 3


@pytest.mark.parametrize("label", ["", "maybe", "TOOL/SPEAK"])
def test_unknown_tool_route_remains_unscored(label):
    backend = Backend(["CONTINUE", label])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="TOOL/SPEAK route"):
        caller.generate(
            "model",
            [UserMessage(role="user", content="Enable data.")],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )


def test_spoken_reply_uses_recent_result_and_cannot_invent_a_tool():
    backend = Backend(["CONTINUE", "SPEAK", native_call()])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="unplanned customer tool"):
        caller.generate(
            "model",
            [
                UserMessage(role="user", content="Is data enabled?"),
                ToolMessage(
                    role="tool",
                    id="actual",
                    requestor="user",
                    content="Data is enabled.",
                ),
            ],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )
    payload = json.loads(backend.calls[-1][0][-1].content.split("\n", 1)[1])
    assert payload["latest_customer_tool_result"]["content"] == "Data is enabled."
    assert backend.calls[-1][1]["tools"] == []


def test_agent_offer_has_no_customer_tools_and_does_not_prioritize_an_old_report():
    def run_speed_test() -> str:
        """Test the current connection speed."""
        return "Speed test failed: No Connection."

    backend = Backend(["CONTINUE", "Yes, please check that."])
    caller = QualifiedCallerBackend(
        backend, "Restore my internet.", device_grounding=True
    )
    offer = "Since the speed test shows no connection, I can check if your data is restricted. Would you like me to check that?"
    reply = caller.generate(
        "model",
        [
            UserMessage(role="user", content="Please run a speed test."),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="speed", name="run_speed_test", arguments={})],
            ),
            ToolMessage(
                role="tool",
                requestor="user",
                id="speed",
                content="Speed test failed: No Connection.",
            ),
            AssistantMessage(
                role="assistant", content="Speed test failed: No Connection."
            ),
            UserMessage(role="user", content=offer),
        ],
        tools=[Tool(run_speed_test)],
        call_name="user_streaming_response",
    )
    assert reply.content == "Yes, please check that." and not reply.is_tool_call()
    assert [kwargs["call_name"] for _, kwargs in backend.calls] == [
        "caller_terminal_v2",
        "user_streaming_response",
    ]
    payload = json.loads(backend.calls[-1][0][-1].content.split("\n", 1)[1])
    assert payload["latest_customer_tool_result"] is None
    assert any(row["speaker"] == "customer_tool" for row in payload["conversation"])
    assert offer in backend.calls[-1][0][-1].content.split("\n", 1)[0]
    assert backend.calls[-1][1]["tools"] == []
    assert caller.decisions[-1]["customer_tool_route"]["guard"] == "agent_owns_action"


def test_terminal_evidence_can_span_streaming_fragments_of_one_heard_turn():
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", "Your subscription has been cancelled.")]),
        "Cancel my subscription.",
    )
    decision, _ = caller.assess(
        "model",
        [
            UserMessage(role="user", content=text)
            for text in ("Your ", "subscription ", "has been ", "cancelled.")
        ],
    )
    assert decision == "COMPLETE"


def test_tool_route_gets_whole_heard_request_and_preserves_tool_boundaries():
    backend = Backend(["CONTINUE", "TOOL", native_call()])
    caller = QualifiedCallerBackend(backend, "Restore internet.")
    history = [
        UserMessage(role="user", content=text)
        for text in ("Could you ", "please ", "enable mobile ", "data?")
    ]
    caller.generate(
        "model",
        history,
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    routed = json.loads(backend.calls[1][0][-1].content)
    assert (
        routed["latest_service_agent_statement"]
        == "Could you please enable mobile data?"
    )
    assert len(routed["conversation"]) == 1
    history.extend(
        [
            AssistantMessage(role="assistant", content="Okay."),
            native_call(),
            ToolMessage(
                role="tool", id="native-id", requestor="user", content="Data enabled."
            ),
            UserMessage(role="user", content="Is it working?"),
        ]
    )
    turns = caller.conversation(history)
    assert [row["speaker"] for row in turns] == [
        "service_agent",
        "customer",
        "customer",
        "customer_tool",
        "service_agent",
    ]
    assert turns[2]["tool_calls"] and turns[3]["tool_call_id"] == "native-id"


@pytest.mark.parametrize(
    "bad",
    [
        {
            "reasoning": "Work is still pending.",
            "agent_evidence": None,
            "decision": "CONTINE",
        },
        {"reasoning": None, "agent_evidence": None, "decision": "CONTINUE"},
    ],
)
def test_invalid_control_fields_get_one_model_reassessment(bad):
    backend = Backend([json.dumps(bad), "CONTINUE"])
    caller = QualifiedCallerBackend(backend, "private scenario")
    decision, _ = caller.assess(
        "model", [UserMessage(role="user", content="I am checking.")]
    )
    assert decision == "CONTINUE"
    assert len(backend.calls) == 2
    initial = json.loads(backend.calls[0][0][-1].content)
    retry = json.loads(backend.calls[1][0][-1].content)
    assert retry["scenario"] == initial["scenario"]
    assert retry["conversation"] == initial["conversation"]
    assert retry["required_schema"]["properties"]["decision"]["enum"] == [
        "CONTINUE",
        "COMPLETE",
        "REFUSED",
    ]
    assert caller.decisions[-1]["rejected_assessments"][0][
        "invalid_response"
    ] == json.dumps(bad)


def test_repeated_invalid_control_fields_remain_unscored():
    invalid = json.dumps(
        {"reasoning": "pending", "agent_evidence": None, "decision": "CONTINE"}
    )
    backend = Backend([invalid, invalid])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="rejected assessments"):
        caller.assess("model", [])
    assert len(backend.calls) == 2
    assert not caller.decisions


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know if I've reached my data limit.",
        "I'm not sure whether I reached the limit.",
        "I cannot access my plan usage.",
    ],
)
def test_honest_unknown_answer_does_not_require_an_unavailable_customer_tool(answer):
    backend = Backend(["CONTINUE", "SPEAK", answer])
    caller = QualifiedCallerBackend(backend, "scenario")
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content="Have you reached your plan data limit?")],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    assert reply.content == answer and not reply.is_tool_call()
    assert backend.calls[-1][1]["tools"] == []
    assert len(backend.calls) == 3


@pytest.mark.parametrize(
    "answer",
    ["I don't know how to enable mobile data.", "I don't know if mobile data is on."],
)
def test_unknown_device_state_does_not_replace_an_available_requested_tool(answer):
    backend = Backend(["CONTINUE", "TOOL", answer, answer])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="after one reassessment"):
        caller.generate(
            "model",
            [UserMessage(role="user", content="Please enable mobile data.")],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )


def test_empty_heard_record_does_not_replace_pending_device_instruction():
    backend = Backend(["CONTINUE", "TOOL", native_call()])
    caller = QualifiedCallerBackend(backend, "scenario")
    caller.generate(
        "model",
        [
            UserMessage(role="user", content="Please enable mobile data."),
            ToolMessage(role="tool", content="Mobile data is off.", id="observed"),
            UserMessage(role="user", content=None),
        ],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    assert backend.calls[-1][0][-1].content.startswith(
        "The service agent asks you: Please enable mobile data.\n"
    )


def test_unknown_answer_cannot_hide_an_unexecuted_action_claim():
    claim = "I don't know, but I have turned off airplane mode."
    backend = Backend(["CONTINUE", "TOOL", claim, claim])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="after one reassessment"):
        caller.generate(
            "model",
            [UserMessage(role="user", content="Please disable airplane mode.")],
            tools=[Tool(set_mobile_data)],
            call_name="user_streaming_response",
        )

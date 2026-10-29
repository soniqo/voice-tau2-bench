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


@pytest.mark.parametrize("enabled", [False, True])
def test_question_focus_exposes_only_customer_scenario_and_heard_context(enabled):
    backend = Backend(["CONTINUE", "I am moving to another city."])
    scenario = "Cancel membership because I am moving. My account ID is unknown."
    caller = QualifiedCallerBackend(backend, scenario, question_focus=enabled)
    history = [AssistantMessage(role="assistant", content="Please cancel it.")]
    for _ in range(8):
        history += [
            UserMessage(role="user", content="Why are you cancelling?"),
            AssistantMessage(role="assistant", content="Please cancel it."),
        ]
    history.append(UserMessage(role="user", content="Why are you cancelling?"))
    reply = caller.generate(
        "model", history, tools=[], call_name="user_streaming_response"
    )
    assert reply.content == "I am moving to another city."
    sent = backend.calls[-1][0][-1].content
    context = json.loads(sent[sent.index("{") :])
    assert (
        context["latest_service_agent_statement"]["content"]
        == "Why are you cancelling?"
    )
    assert context.get("customer_scenario") == (scenario if enabled else None)
    assert all(
        row["speaker"] in {"service_agent", "customer"}
        for row in context["conversation"]
    )


@pytest.mark.parametrize("proposed", ["COMPLETE", "CONTINUE"])
@pytest.mark.parametrize("enabled", [False, True])
def test_price_guard_keeps_unquoted_capped_change_open(proposed, enabled):
    heard = "I have successfully upgraded the booking. Is there anything else I can help you with?"
    assessment = json.dumps(
        {"reasoning": "fixture", "agent_evidence": heard, "decision": proposed}
    )
    backend = Backend([assessment, "No, that is all. Thank you."])
    caller = QualifiedCallerBackend(
        backend, "Upgrade for up to $650.", price_grounding=enabled
    )
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content=heard)],
        tools=[],
        call_name="user_streaming_response",
    )
    if enabled:
        assert "total cost" in reply.content
        assert caller.decisions[-1]["decision"] == "CONTINUE"
        assert len(backend.calls) == 1
        assert reply.raw_data["caller_turn_control"]["guard"] == "missing_price_quote"
    else:
        assert reply.content == (
            "###STOP###" if proposed == "COMPLETE" else "No, that is all. Thank you."
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_quoted_offer_cannot_complete_a_price_controlled_call(enabled):
    heard = "The total cost is $598.00, within your $650 budget. Would you like me to proceed?"
    backend = Backend(
        [
            json.dumps(
                {
                    "reasoning": "The quoted price meets the limit.",
                    "agent_evidence": heard,
                    "decision": "COMPLETE",
                }
            ),
            "Yes, please proceed.",
        ]
    )
    caller = QualifiedCallerBackend(
        backend, "Upgrade for up to $650.", price_grounding=enabled
    )
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content=heard)],
        tools=[],
        call_name="user_streaming_response",
    )
    if enabled:
        assert reply.content == "Yes, please proceed."
        assert caller.decisions[-1]["guard"] == "pending_price_consent"
        assert caller.decisions[-1]["decision"] == "CONTINUE"
    else:
        assert reply.content == "###STOP###"


@pytest.mark.parametrize("enabled", [False, True])
def test_price_controls_preserve_the_recorded_caller_prompts(enabled):
    backend = Backend(["CONTINUE", "My number is 555-123-2002."])
    caller = QualifiedCallerBackend(
        backend, "Check my account.", price_grounding=enabled
    )
    caller.generate(
        "model",
        [UserMessage(role="user", content="What is your phone number?")],
        tools=[],
        call_name="user_streaming_response",
    )
    assessment_prompt = backend.calls[0][0][0].content
    caller_prompt = backend.calls[1][0][0].content
    from tau2.voice.caller_policy import TERMINAL_PROMPT

    assert assessment_prompt == TERMINAL_PROMPT
    assert "Before answering any consent question" not in caller_prompt


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
    ],
)
def test_malformed_terminal_assessment_propagates_without_fallback(reply):
    backend = Backend([reply, reply])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="Invalid caller terminal"):
        caller.assess("model", [])
    assert len(backend.calls) == 2
    assert not caller.decisions


def test_assessment_tool_calls_fail_immediately_without_reassessment():
    backend = Backend(
        [
            AssistantMessage(
                role="assistant", tool_calls=[ToolCall(name="write", arguments={})]
            )
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="assessment cannot call tools"):
        caller.assess("model", [])
    assert len(backend.calls) == 1
    assert not caller.decisions


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


@pytest.mark.parametrize("grounding", [False, True])
@pytest.mark.parametrize("empty_frame", [False, True])
@pytest.mark.parametrize(
    "silence,text,action",
    [(1, "", "wait_for_agent_turn"), (15, "Are you still there?", "silence_check_in")],
)
def test_answered_turn_waits_instead_of_repeating_the_answer(
    grounding, empty_frame, silence, text, action
):
    backend = Backend(["CONTINUE", "My user ID is account_7."])
    caller = QualifiedCallerBackend(
        backend, "Change my flight.", device_grounding=grounding
    )
    history = [
        UserMessage(role="user", content="Could you provide your user ID?"),
        AssistantMessage(role="assistant", content="My user ID is account_7."),
    ]
    if empty_frame:
        history.append(UserMessage(role="user", content=None, is_final_chunk=False))
    reply = caller.generate(
        "model",
        history,
        tools=[],
        call_name="user_streaming_response",
        agent_silence_seconds=silence,
        caller_idle_seconds=15,
    )
    assert reply.content == text
    control = (
        reply.raw_data.get("caller_turn_control") or reply.raw_data["device_grounding"]
    )
    assert control["action"] == action
    assert len(backend.calls) == (0 if grounding and not empty_frame else 1)


def test_answered_turn_wait_reuses_only_an_unchanged_visible_assessment():
    backend = Backend(["CONTINUE", "My user ID is account_7."])
    caller = QualifiedCallerBackend(backend, "Change my flight.")
    history = [
        UserMessage(role="user", content="Your user ID?"),
        AssistantMessage(role="assistant", content="account_7"),
    ]
    first = caller.generate(
        "model", history, call_name="user_streaming_response", agent_silence_seconds=1
    )
    second = caller.generate(
        "model",
        [*history, UserMessage(role="user", content=None)],
        call_name="user_streaming_response",
        agent_silence_seconds=15,
    )
    assert first.content == "" and second.content == "Are you still there?"
    assert len(backend.calls) == 1
    assert second.cost == 0 and second.generation_time_seconds == 0
    assert second.raw_data["caller_turn_control"]["cached_assessment"] is True


@pytest.mark.parametrize("terminal", ["COMPLETE", "REFUSED"])
def test_answered_turn_still_checks_terminal_state_after_customer_acknowledgment(
    terminal,
):
    evidence = (
        "Your subscription has been cancelled."
        if terminal == "COMPLETE"
        else "Policy prohibits cancellation."
    )
    backend = Backend([assessment(terminal, evidence)])
    caller = QualifiedCallerBackend(backend, "Cancel my subscription.")
    reply = caller.generate(
        "model",
        [
            UserMessage(role="user", content=evidence),
            AssistantMessage(role="assistant", content="Thank you."),
        ],
        call_name="user_streaming_response",
        agent_silence_seconds=1,
    )
    assert reply.content == "###STOP###" and len(backend.calls) == 1


def test_answered_turn_wait_preserves_assessment_accounting():
    backend = Backend(
        [
            assessment(
                "CONTINUE",
                cost=0.2,
                usage={"completion_tokens": 5},
                generation_time_seconds=1.25,
            ),
            "repeat",
        ]
    )
    caller = QualifiedCallerBackend(backend, "Change my flight.")
    reply = caller.generate(
        "model",
        [
            UserMessage(role="user", content="Your user ID?"),
            AssistantMessage(role="assistant", content="account_7"),
        ],
        call_name="user_streaming_response",
        agent_silence_seconds=1,
    )
    assert reply.content == "" and reply.cost == 0.2
    assert (
        reply.usage == {"completion_tokens": 5}
        and reply.generation_time_seconds == 1.25
    )


def test_new_heard_turn_after_wait_is_assessed_and_can_complete():
    evidence = "Your subscription has been cancelled."
    backend = Backend(["CONTINUE", assessment("COMPLETE", evidence)])
    caller = QualifiedCallerBackend(backend, "Cancel my subscription.")
    history = [
        UserMessage(role="user", content="Your user ID?"),
        AssistantMessage(role="assistant", content="account_7"),
    ]
    first = caller.generate(
        "model", history, call_name="user_streaming_response", agent_silence_seconds=1
    )
    reply = caller.generate(
        "model",
        [*history, UserMessage(role="user", content=evidence)],
        call_name="user_streaming_response",
        agent_silence_seconds=0,
    )
    assert first.content == "" and reply.content == "###STOP###"
    assert len(backend.calls) == 2


def test_answered_turn_does_not_hide_a_pending_customer_tool_result():
    backend = Backend(["CONTINUE", "The device check returned off."])
    caller = QualifiedCallerBackend(backend, "Restore my connection.")
    reply = caller.generate(
        "model",
        [
            UserMessage(role="user", content="Please check your device."),
            AssistantMessage(role="assistant", content="Okay."),
            ToolMessage(
                role="tool", requestor="user", id="read-1", content="The device is off."
            ),
        ],
        tools=[],
        call_name="user_streaming_response",
        agent_silence_seconds=1,
    )
    assert reply.content == "The device check returned off." and len(backend.calls) == 2


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


@pytest.mark.parametrize("during_reassessment", [False, True])
def test_failed_inference_is_not_retried(during_reassessment):
    invalid = RuntimeError("native failed")
    backend = Backend(
        [assessment("COMPLETE"), invalid] if during_reassessment else [invalid]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(RuntimeError, match="native failed"):
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


@pytest.mark.parametrize(
    "confirmation",
    [
        "I have successfully updated your reservation to business class and added two checked bags. Is there anything else I can help you with?",
        "Your subscription has been cancelled. Is there anything else that we can assist you with?",
        "Your subscription has been cancelled! Is there anything else I can do for you?",
    ],
)
def test_completed_action_with_closing_help_offer_preserves_model_stop(confirmation):
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", confirmation)]), "Complete the requested work."
    )
    reply = caller.generate(
        "model",
        [
            UserMessage(role="user", content=confirmation),
            AssistantMessage(role="assistant", content="Thank you."),
        ],
        call_name="user_streaming_response",
        agent_silence_seconds=1,
    )
    assert reply.content == "###STOP###"
    assert caller.decisions[-1]["decision"] == "COMPLETE"
    assert "guard" not in caller.decisions[-1]


@pytest.mark.parametrize(
    "evidence",
    [
        "I will cancel your subscription. Is there anything else I can help you with?",
        "I have updated your reservation. I will add the bags now. Is there anything else I can help you with?",
        "I have updated your reservation. Is there a bag I can add for you?",
    ],
)
def test_closing_question_does_not_hide_actual_pending_work(evidence):
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", evidence)]), "Complete the requested work."
    )
    decision, _ = caller.assess("model", [UserMessage(role="user", content=evidence)])
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "pending_agent_offer"


def test_closing_help_question_alone_cannot_prove_completion():
    evidence = "Is there anything else I can help you with?"
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", evidence)]), "Cancel my subscription."
    )
    decision, _ = caller.assess("model", [UserMessage(role="user", content=evidence)])
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "agent_confirmation_question"


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


@pytest.mark.parametrize(
    "statement",
    [
        "I've checked your usage. Since the speed is still slow, I can check if your phone is using an older network technology.",
        "I can check if your data plan limit has been reached.",
    ],
)
def test_v5_agent_offer_does_not_require_a_customer_tool(statement):
    backend = Backend(["CONTINUE", "Yes, please check that."])
    caller = QualifiedCallerBackend(
        backend, "Restore my mobile data.", device_grounding=True, routing_focus=True
    )
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content=statement)],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    assert reply.content == "Yes, please check that." and not reply.is_tool_call()
    assert len(backend.calls) == 2
    assert caller.decisions[-1]["customer_tool_route"]["guard"] == "agent_owns_action"


@pytest.mark.parametrize(
    "statement",
    [
        "I see roaming is already enabled on your line. I will check your data usage. Could you please tell me if you have reached your data limit?",
        "Since you have no connection, please check if your data usage has exceeded your plan limit.",
    ],
)
def test_v5_account_usage_unknown_is_a_spoken_answer_not_a_device_action(statement):
    backend = Backend(["CONTINUE", "I don't know if I've reached my data limit."])
    caller = QualifiedCallerBackend(
        backend,
        "Restore my mobile data. I do not know my usage.",
        device_grounding=True,
        routing_focus=True,
    )
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content=statement)],
        tools=[Tool(set_mobile_data)],
        call_name="user_streaming_response",
    )
    assert not reply.is_tool_call() and "don't know" in reply.content
    assert (
        caller.decisions[-1]["customer_tool_route"]["guard"] == "account_usage_question"
    )
    assert len(backend.calls) == 2


@pytest.mark.parametrize(
    "idle,silence,expected",
    [
        (15, 0, ""),
        (15, 14.95, ""),
        (15, 15, "Are you still there?"),
        (30, 15, ""),
        (30, 29.95, ""),
        (30, 30, "Are you still there?"),
    ],
)
def test_non_phone_caller_waits_for_a_complete_turn_and_retains_check_in(
    idle, silence, expected
):
    backend = Backend([])
    caller = QualifiedCallerBackend(backend, "Change my flight.")
    reply = caller.generate(
        "model",
        [
            UserMessage(
                role="user",
                content="Could you please tell me your ",
                is_final_chunk=False,
            )
        ],
        call_name="user_streaming_response",
        agent_silence_seconds=silence,
        caller_idle_seconds=idle,
    )
    assert reply.content == expected and not reply.is_tool_call()
    assert not backend.calls


def test_non_phone_caller_can_answer_a_completed_question_during_speech():
    backend = Backend(["CONTINUE", "My name is John."])
    caller = QualifiedCallerBackend(backend, "My name is John. Change my flight.")
    reply = caller.generate(
        "model",
        [
            UserMessage(
                role="user",
                content="What is your name? While I look up",
                is_final_chunk=False,
            )
        ],
        call_name="user_streaming_response",
        agent_silence_seconds=0,
        caller_idle_seconds=15,
    )
    assert reply.content == "My name is John."
    assert len(backend.calls) == 2


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
    "invalid",
    [
        '```json\n{\nen: "The conversation is ongoing.",\nagent_evidence: null,\ndecision: "CONTINUE"\n}\n```',
        '{"reasoning": "pending", "agent_evidence": null, "decision":',
        "not JSON",
        "",
    ],
)
def test_json_syntax_error_gets_one_strict_reassessment(invalid):
    backend = Backend([invalid, "CONTINUE"])
    caller = QualifiedCallerBackend(backend, "private scenario")
    decision, _ = caller.assess(
        "model", [UserMessage(role="user", content="I am checking your account.")]
    )
    assert decision == "CONTINUE" and len(backend.calls) == 2
    initial = json.loads(backend.calls[0][0][-1].content)
    retry = json.loads(backend.calls[1][0][-1].content)
    assert retry["scenario"] == initial["scenario"]
    assert retry["conversation"] == initial["conversation"]
    assert retry["required_schema"]["additionalProperties"] is False
    assert retry["rejected_assessment"]["invalid_response"] == invalid
    assert (
        caller.decisions[-1]["rejected_assessments"][0]["invalid_response"] == invalid
    )
    assert backend.calls[1][1]["call_name"] == "caller_terminal_v2_repair"


@pytest.mark.parametrize("first", ['{unquoted: "CONTINUE"}', assessment("COMPLETE")])
def test_json_syntax_reassessment_never_exceeds_two_total_calls(first):
    backend = Backend([first, '{still_unquoted: "COMPLETE"}'])
    caller = QualifiedCallerBackend(backend, "scenario")
    with pytest.raises(CallerPolicyError, match="rejected assessments"):
        caller.assess("model", [])
    assert len(backend.calls) == 2 and not caller.decisions


def test_json_syntax_reassessment_accounts_for_rejected_inference():
    backend = Backend(
        [
            AssistantMessage(
                role="assistant",
                content="{unquoted: true}",
                cost=0.1,
                usage={"completion_tokens": 3},
                generation_time_seconds=0.2,
            ),
            assessment(
                "CONTINUE",
                cost=0.2,
                usage={"completion_tokens": 5},
                generation_time_seconds=0.3,
            ),
        ]
    )
    caller = QualifiedCallerBackend(backend, "scenario")
    decision, reply = caller.assess("model", [])
    assert decision == "CONTINUE"
    assert reply.cost == pytest.approx(0.3)
    assert reply.usage == {"completion_tokens": 8}
    assert reply.generation_time_seconds == pytest.approx(0.5)


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


@pytest.mark.parametrize(
    "result,error,expected",
    [
        (
            "Speed Test Result: 200.00 Mbps (Excellent). Connection is very fast.",
            False,
            "COMPLETE",
        ),
        ("Speed Test Result: 80.00 Mbps (Good).", False, "CONTINUE"),
        ("Speed test failed: No Connection.", False, "CONTINUE"),
        ("Speed Test Result: 200.00 Mbps (Excellent).", True, "CONTINUE"),
    ],
)
def test_explicit_speed_acceptance_requires_successful_speed_tool_evidence(
    result, error, expected
):
    from tau2.domains.telecom.environment import get_environment

    quote = "Your mobile internet is restored."
    backend = Backend(
        [
            json.dumps(
                {
                    "decision": "COMPLETE",
                    "agent_evidence": quote,
                    "reasoning": "fixture",
                }
            )
        ]
    )
    caller = QualifiedCallerBackend(
        backend,
        "You will consider the issue resolved only when speed test returns excellent internet speed and nothing else.",
        device_grounding=True,
    )
    history = [
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="speed", name="run_speed_test", arguments={})],
        ),
        ToolMessage(
            role="tool", id="speed", requestor="user", content=result, error=error
        ),
        UserMessage(role="user", content=quote),
    ]
    decision, _ = caller.assess(
        "model", history, tools=get_environment().get_user_tools()
    )
    assert decision == expected
    if expected == "CONTINUE":
        assert caller.decisions[-1]["guard"] == "caller_goal_result_not_verified"


def test_excellent_radio_signal_is_not_evidence_of_excellent_internet_speed():
    from tau2.domains.telecom.environment import get_environment

    quote = "Your mobile internet is restored."
    caller = QualifiedCallerBackend(
        Backend(
            [
                json.dumps(
                    {
                        "decision": "COMPLETE",
                        "agent_evidence": quote,
                        "reasoning": "fixture",
                    }
                )
            ]
        ),
        "You will consider the issue resolved only when speed test returns excellent internet speed and nothing else.",
        device_grounding=True,
    )
    history = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="network", name="check_network_status", arguments={})
            ],
        ),
        ToolMessage(
            role="tool",
            id="network",
            requestor="user",
            content="Cellular Signal: excellent\nMobile Data Enabled: Yes",
        ),
        UserMessage(role="user", content=quote),
    ]
    decision, _ = caller.assess(
        "model", history, tools=get_environment().get_user_tools()
    )
    assert decision == "CONTINUE"


@pytest.mark.parametrize("proposal", ["COMPLETE", "REFUSED"])
def test_terminal_assessment_cannot_skip_a_pending_customer_device_response(proposal):
    from tau2.domains.telecom.environment import get_environment

    instruction = "Please turn off Airplane Mode on your device. This will restore your cellular and data connections. Would you like me to check anything else?"
    backend = Backend([assessment(proposal, instruction)])
    caller = QualifiedCallerBackend(
        backend,
        "Restore mobile internet and follow the agent instructions.",
        device_grounding=True,
    )
    env = get_environment()
    env.user_tools.db.device.airplane_mode = True
    before = env.user_tools.db.device.model_dump()
    decision, _ = caller.assess(
        "model",
        [UserMessage(role="user", content=instruction)],
        tools=env.get_user_tools(),
    )
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "pending_customer_device_response"
    assert caller.decisions[-1]["proposed_decision"] == proposal
    assert env.user_tools.db.device.model_dump() == before
    assert len(backend.calls) == 1
    assert backend.calls[0][1]["tools"] is None


def test_device_response_guard_allows_a_later_completed_confirmation():
    from tau2.domains.telecom.environment import get_environment

    quote = "Your mobile internet has been restored."
    caller = QualifiedCallerBackend(
        Backend([assessment("COMPLETE", quote)]),
        "Restore mobile internet and follow the agent instructions.",
        device_grounding=True,
    )
    decision, _ = caller.assess(
        "model",
        [UserMessage(role="user", content=quote)],
        tools=get_environment().get_user_tools(),
    )
    assert decision == "COMPLETE"
    assert "guard" not in caller.decisions[-1]


@pytest.mark.parametrize("decision", ["COMPLETE", "REFUSED"])
def test_remaining_request_prevents_terminal_control_after_partial_refusal(decision):
    quote = "I cannot change the delivery address."
    backend = Backend(
        [
            json.dumps(
                {
                    "reasoning": "One request is refused.",
                    "agent_evidence": quote,
                    "decision": decision,
                    "remaining_requests": ["Cancel my recurring subscription."],
                }
            )
        ]
    )
    caller = QualifiedCallerBackend(
        backend,
        "Change the address and cancel my subscription even if the address cannot change.",
        goal_progress=True,
    )
    actual, _ = caller.assess("model", [UserMessage(role="user", content=quote)])
    assert actual == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "remaining_customer_requests"


def test_progress_request_list_is_required_without_silent_legacy_fallback():
    heard = "I cannot cancel that subscription."
    reply = json.dumps(
        {"reasoning": "fixture", "agent_evidence": heard, "decision": "REFUSED"}
    )
    caller = QualifiedCallerBackend(
        Backend([reply, reply]), "scenario", goal_progress=True
    )
    with pytest.raises(CallerPolicyError, match="remaining_requests"):
        caller.assess("model", [UserMessage(role="user", content=heard)])


def test_completed_progress_allows_terminal_control_with_empty_requests():
    quote = "I cannot change the address. Your subscription is cancelled."
    caller = QualifiedCallerBackend(
        Backend(
            [
                json.dumps(
                    {
                        "reasoning": "All requests addressed.",
                        "agent_evidence": quote,
                        "decision": "COMPLETE",
                        "remaining_requests": [],
                    }
                )
            ]
        ),
        "Cancel my subscription.",
        goal_progress=True,
    )
    decision, _ = caller.assess("model", [UserMessage(role="user", content=quote)])
    assert decision == "COMPLETE"


@pytest.mark.parametrize(
    "heard,has_pending,redirect",
    [
        ("I cannot upgrade just one passenger.", True, True),
        ("I cannot change the delivery address.", True, True),
        (
            "I cannot upgrade just one passenger. Is there anything else I can help you with?",
            True,
            True,
        ),
        ("I cannot find your account. Please provide your ID.", True, False),
        (
            "I cannot upgrade one passenger. Would you like me to add the bags?",
            True,
            False,
        ),
        ("I cannot change that. Please confirm your account ID.", True, False),
        ("I cannot upgrade just one passenger.", False, False),
    ],
)
def test_progress_redirects_only_unanswered_independent_requests(
    heard, has_pending, redirect
):
    data = {
        "reasoning": "The upgrade was denied; another request may remain.",
        "agent_evidence": heard,
        "decision": "CONTINUE",
    }
    if QualifiedCallerBackend.followup_after_refusal(
        [{"speaker": "service_agent", "content": heard}]
    ):
        data["remaining_requests"] = ["Add two checked bags."] if has_pending else []
    backend = Backend([json.dumps(data), "Could you please clarify?"])
    caller = QualifiedCallerBackend(
        backend, "Upgrade and add two bags.", goal_progress=True
    )
    reply = caller.generate(
        "model",
        [UserMessage(role="user", content=heard)],
        tools=[],
        call_name="user_streaming_response",
    )
    if redirect:
        assert reply.content == "Then, Add two checked bags."
        assert (
            reply.raw_data["caller_turn_control"]["request"] == "Add two checked bags"
        )
        assert len(backend.calls) == 1
    else:
        assert reply.content == "Could you please clarify?"
        assert len(backend.calls) == 2


@pytest.mark.parametrize(
    "heard",
    [
        "I am checking your subscription now.",
        "I'm still processing the cancellation.",
        "We are updating your reservation.",
        "We're adding the bags now.",
    ],
)
@pytest.mark.parametrize("proposed", ["COMPLETE", "REFUSED"])
def test_ongoing_agent_work_never_confirms_completion(heard, proposed):
    caller = QualifiedCallerBackend(
        Backend(
            [
                json.dumps(
                    {
                        "reasoning": "fixture",
                        "agent_evidence": heard,
                        "decision": proposed,
                    }
                )
            ]
        ),
        "Complete the requested change.",
    )
    decision, _ = caller.assess(
        "model",
        [
            UserMessage(role="user", content=heard),
            AssistantMessage(role="assistant", content="It has been completed."),
        ],
    )
    assert decision == "CONTINUE"
    assert caller.decisions[-1]["guard"] == "pending_agent_offer"

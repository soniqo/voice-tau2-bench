import json

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.voice.caller_policy import CallerPolicyError, QualifiedCallerBackend


class Backend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def generate(self, model, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = next(self.replies)
        if (
            kwargs.get("call_name") == "caller_terminal_v2"
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
    caller = QualifiedCallerBackend(Backend([response]), "Cancel my subscription")
    with pytest.raises(CallerPolicyError, match="quote from the service agent"):
        caller.assess(
            "model",
            [
                UserMessage(role="user", content="I am checking."),
                AssistantMessage(role="assistant", content="It was cancelled."),
            ],
        )

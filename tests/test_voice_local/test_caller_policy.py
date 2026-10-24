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

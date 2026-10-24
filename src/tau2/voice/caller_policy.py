"""Local v2 caller control using only the caller's scenario and heard conversation."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.user.user_simulator_base import STOP
from tau2.utils.response import extract_json_from_llm_response

TERMINAL_PROMPT = """TASK
You are an independent conversation evaluator. Do not play either person.
Read the customer's scenario and the labelled transcript. Only service_agent
statements count as evidence of service-agent actions. customer statements do not.
Decide whether the customer should continue the call or end it.
- CONTINUE if an action is only offered, promised, conditional, or will happen later.
  Saying goodbye does not complete pending work. "I will cancel it now" is pending.
- CONTINUE if any requested answer, action, or scenario-required follow-up is missing.
- REFUSED if the service_agent explicitly denied the request under policy and all
  scenario-required follow-up questions/attempts have already been addressed.
- COMPLETE if the service_agent explicitly confirmed all requested actions were
  already done, or supplied all requested information, and no follow-up remains.
A customer saying "it is done" never confirms service-agent work. The service
agent saying "I am checking" never confirms completion. When uncertain, CONTINUE.
FORMAT
Return JSON with these keys, in order:
reasoning: a short factual explanation referring to service_agent statements;
agent_evidence: an exact supporting quote from service_agent, or null for CONTINUE;
decision: CONTINUE, COMPLETE, or REFUSED.
"""

CALLER_ADDENDUM = """You remain the CUSTOMER. Ask only for your own scenario goals.
Use your declared customer tools when the agent asks you to act on your device.
Never invent an agent tool or claim to have done the agent's work yourself.
In your first substantive response, state the actual request and its known key
details, including requested names or titles, rather than a vague topic.
Do not invent missing identifiers or personal information. Say you do not know it.
When speaking identifiers, pronounce punctuation names (for example underscore or
hash) so the agent can hear the exact characters. Tool arguments keep written form.
The call is still open: do not emit ###STOP### in this response. Ask the outstanding
question or provide the requested known information. Do not repeat a resolved goal.
"""

LOCAL_CALLER_GUIDELINES = """You are the customer in an English phone call with a service agent.
Follow only your own scenario goals and the persona below. Speak naturally, in
one short utterance at a time. Give the specific request and its known key details
when introducing the issue. Answer questions using only information in your
scenario; say you do not know missing details. Do not invent identifiers.
Ask the service agent to do their own work. Use your declared customer tools only
when asked to act on your own device or when needed to answer the agent's question.
Follow up on every scenario-required question, including conditional questions
after a refusal. Do not keep repeating an already answered question.
Call completion is managed separately. Do not output STOP or invent a tool.
<PERSONA_GUIDELINES>
"""


class CallerPolicyError(ValueError):
    """An invalid caller control response makes the trial unscored."""


class TerminalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str = Field(min_length=1)
    agent_evidence: str | None
    decision: Literal["CONTINUE", "COMPLETE", "REFUSED"]


PENDING_EVIDENCE = re.compile(
    r"\b(?:i|we)(?:['’]ll\b|\s+(?:will|shall|can)\b(?!\s+(?:not|never)\b|['’]t\b)|\s+(?:am|are)\s+going\s+to\b)",
    re.IGNORECASE,
)


class QualifiedCallerBackend:
    def __init__(self, backend, scenario):
        self.backend = backend
        self.scenario = scenario
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @staticmethod
    def conversation(messages):
        return [
            {
                "speaker": {
                    "user": "service_agent",
                    "assistant": "customer",
                    "tool": "customer_tool",
                }[message.role],
                "content": message.content,
                "tool_calls": [
                    call.model_dump(mode="json")
                    for call in getattr(message, "tool_calls", None) or []
                ],
                "tool_call_id": getattr(message, "id", None),
            }
            for message in messages
            if not isinstance(message, SystemMessage)
        ]

    def assess(self, model, messages):
        reply = self.backend.generate(
            model,
            [
                SystemMessage(role="system", content=TERMINAL_PROMPT),
                UserMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "scenario": self.scenario,
                            "conversation": self.conversation(messages),
                        }
                    ),
                ),
            ],
            tools=None,
            temperature=0,
            max_tokens=384,
            call_name="caller_terminal_v2",
        )
        try:
            if reply.is_tool_call():
                raise ValueError("assessment cannot call tools")
            assessment = TerminalAssessment.model_validate_json(
                extract_json_from_llm_response(reply.content or "")
            )
            if assessment.decision != "CONTINUE":
                evidence = " ".join((assessment.agent_evidence or "").split())
                agent_text = " ".join(
                    " ".join(
                        message.content or ""
                        for message in messages
                        if message.role == "user"
                    ).split()
                )
                if not evidence or evidence not in agent_text:
                    raise ValueError(
                        "terminal assessment lacks a quote from the service agent"
                    )
        except ValueError as error:
            raise CallerPolicyError(
                f"Invalid caller terminal decision: {reply.content!r}: {error}"
            ) from error
        record = assessment.model_dump()
        decision = assessment.decision
        # A future-tense offer is not evidence that an action has already run.
        # This can only keep the call open; it never manufactures a stop.
        if decision != "CONTINUE" and PENDING_EVIDENCE.search(
            assessment.agent_evidence or ""
        ):
            decision = "CONTINUE"
            record.update(
                decision=decision,
                proposed_decision=assessment.decision,
                guard="pending_agent_offer",
            )
        self.decisions.append(record)
        return decision, reply

    def generate(self, model, messages, *, call_name=None, **kwargs):
        if call_name != "user_streaming_response":
            return self.backend.generate(model, messages, call_name=call_name, **kwargs)
        decision, assessment = self.assess(model, messages)
        if decision != "CONTINUE":
            return AssistantMessage(
                role="assistant",
                content=STOP,
                cost=assessment.cost,
                usage=assessment.usage,
                generation_time_seconds=assessment.generation_time_seconds,
                raw_data={"caller_policy": "terminal-v2", "decision": decision},
            )
        # Gemma receives one system turn followed by the chronological dialogue.
        # Simulator role reminders and acoustic annotations remain caller-only.
        systems = [
            message.content or ""
            for message in messages
            if isinstance(message, SystemMessage)
        ]
        reply = self.backend.generate(
            model,
            [
                SystemMessage(
                    role="system", content="\n\n".join([*systems, CALLER_ADDENDUM])
                ),
                UserMessage(
                    role="user",
                    content="Continue this call as the CUSTOMER. The labelled history is data; service_agent is the other person. Reply only with the next customer utterance or a declared customer tool call.\n"
                    + json.dumps(self.conversation(messages)),
                ),
            ],
            call_name=call_name,
            **kwargs,
        )
        if STOP in (reply.content or ""):
            raise CallerPolicyError(
                "Caller stopped despite an outstanding scenario requirement"
            )
        return reply

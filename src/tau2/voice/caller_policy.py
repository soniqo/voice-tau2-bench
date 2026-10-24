"""Local v2 caller control using only the caller's scenario and heard conversation."""

import json
import re
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.user.user_simulator_base import STOP
from tau2.utils.response import extract_json_from_llm_response

TERMINAL_PROMPT = """TASK
You are an independent conversation controller. Do not play either person.
Read the customer's scenario and the labelled transcript. Only service_agent
statements count as evidence of service-agent actions. customer statements do not.
customer_tool records describe observations on the customer's device; they are
not service-agent confirmations. A changed setting or strong signal alone does
not establish that the customer's requested service now works end to end.
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
An instruction to try a fix or a question asking whether it worked is not a
completion confirmation. If no exact supporting service_agent quote exists,
choose CONTINUE with null evidence; never return COMPLETE or REFUSED with null.
CUSTOMER ACTIONS
When CONTINUE, select the next declared customer tool needed to follow an
outstanding service_agent instruction or answer their question. Cite the integer
message_id of that heard service_agent record as instruction_id. Select only one
tool with its actual arguments. Never cite a customer or customer_tool record.
Respect the customer's scenario and persona. If they require clarification or
reassurance before acting, use null so the customer can ask before doing the action.
A customer saying "I turned it off" does not change a device setting. Only an
actual customer_tool result proves a device action ran. If the customer claimed
an action but no tool ran, select the required tool instead of repeating the claim.
Do not repeat a completed action or toggle a setting back after a successful tool
result. If no customer tool is needed, or none are declared, use null. Never select
an agent tool or carry out the agent's work. Terminal decisions require null.
Only DECLARED_CUSTOMER_TOOLS exist. An empty array means no tools are available:
customer_action must be null. Never invent a tool name for a missing capability.
FORMAT
Return JSON with these keys, in order:
reasoning: a short factual explanation referring to service_agent statements;
agent_evidence: an exact supporting quote from service_agent for a terminal
decision; null for CONTINUE;
decision: CONTINUE, COMPLETE, or REFUSED;
customer_action: null, or {"name": "declared tool name", "arguments": {}, "instruction_id": 0}.
"""

CALLER_ADDENDUM = """You remain the CUSTOMER. Ask only for your own scenario goals.
Use your declared customer tools when the agent asks you to act on your device.
Device changes require actual tool execution; never say a setting changed merely
because you intended to change it. Report only the observed customer tool result.
Never invent an agent tool or claim to have done the agent's work yourself.
When the service agent greets you or asks how to help, state the actual request
and its known key details, including requested names or titles.
If the agent instead asks a specific question, answer it first. If the requested
information is unknown, explicitly say you do not know; do not repeat your goal
instead of answering. This takes priority over introducing your original request.
Do not invent missing identifiers or personal information. Say you do not know it.
When speaking identifiers, pronounce punctuation names (for example underscore or
hash) so the agent can hear the exact characters. Tool arguments keep written form.
The call is still open: do not emit ###STOP### in this response. Ask the outstanding
question or provide the requested known information. Do not repeat a resolved goal.
Answer the service agent's latest question directly. After a customer tool runs,
report its actual result first. Do not replace that answer with the original
complaint, or invent a device observation that no customer tool reported.
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


class CustomerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    arguments: dict
    instruction_id: int = Field(ge=0, strict=True)


class TerminalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str = Field(min_length=1)
    agent_evidence: str | None
    decision: Literal["CONTINUE", "COMPLETE", "REFUSED"]
    customer_action: CustomerAction | None = None


PENDING_EVIDENCE = re.compile(
    r"\b(?:i|we)(?:['’]ll\b|\s+(?:will|shall|can)\b(?!\s+(?:not|never)\b|['’]t\b)|\s+(?:am|are)\s+going\s+to\b)",
    re.IGNORECASE,
)
QUESTION_ONLY_EVIDENCE = re.compile(
    r"^\s*(?:is|are|am|was|were|do|does|did|have|has|had|can|could|would|will|shall|should|what|where|when|how|why|who|which)\b[^.!?]*\?\s*$",
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
                "message_id": index,
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
            for index, message in enumerate(
                message
                for message in messages
                if not isinstance(message, SystemMessage)
            )
        ]

    def assess(self, model, messages, tools=None):
        payload = {
            "scenario": self.scenario,
            "conversation": self.conversation(messages),
        }
        agent_statements = [
            " ".join((message.content or "").split())
            for message in messages
            if message.role == "user"
        ]
        rejected = []
        replies = []
        declared = {tool.name: tool for tool in tools or []}
        prompt = (
            TERMINAL_PROMPT
            + "\nDECLARED_CUSTOMER_TOOLS\n"
            + json.dumps([tool.openai_schema["function"] for tool in declared.values()])
        )
        for attempt in range(2):
            reply = self.backend.generate(
                model,
                [
                    SystemMessage(role="system", content=prompt),
                    UserMessage(role="user", content=json.dumps(payload)),
                ],
                tools=None,
                temperature=0,
                max_tokens=768 if declared else 384,
                call_name=(
                    "caller_terminal_v2_repair" if attempt else "caller_terminal_v2"
                ),
            )
            replies.append(reply)
            try:
                if reply.is_tool_call():
                    raise ValueError("assessment cannot call tools")
                assessment = TerminalAssessment.model_validate_json(
                    extract_json_from_llm_response(reply.content or "")
                )
                if declared and "customer_action" not in assessment.model_fields_set:
                    raise ValueError(
                        "customer action decision is required when tools exist"
                    )
                action = assessment.customer_action
                if action is not None:
                    if assessment.decision != "CONTINUE":
                        raise ValueError(
                            "terminal decision cannot select a customer action"
                        )
                    if action.name not in declared:
                        raise ValueError("customer action tool is not declared")
                    source = next(
                        (
                            row
                            for row in payload["conversation"]
                            if row["message_id"] == action.instruction_id
                            and row["speaker"] == "service_agent"
                            and (row["content"] or "").strip()
                        ),
                        None,
                    )
                    if source is None:
                        raise ValueError(
                            "customer action lacks a heard agent instruction"
                        )
                    tool = declared[action.name]
                    if action.arguments.keys() - tool.params.model_fields.keys():
                        raise ValueError("customer action has undeclared arguments")
                    tool.params.model_validate_json(
                        json.dumps(action.arguments), strict=True
                    )
            except ValueError as error:
                raise CallerPolicyError(
                    f"Invalid caller terminal decision: {reply.content!r}: {error}"
                ) from error
            evidence = " ".join((assessment.agent_evidence or "").split())
            if assessment.decision == "CONTINUE" or (
                evidence and any(evidence in text for text in agent_statements)
            ):
                break
            reason = "terminal assessment lacks a quote from the service agent"
            rejected.append({**assessment.model_dump(), "error": reason})
            if attempt:
                raise CallerPolicyError(
                    f"Invalid caller terminal decision after one reassessment: {json.dumps(rejected)}"
                )
            # Reassess once using the same caller-visible inputs. No decision is
            # accepted automatically and no private agent/grading state is added.
            payload["rejected_assessment"] = rejected[-1]
            payload["feedback"] = (
                "The proposed terminal decision has no supporting quote heard from "
                "the service agent. Reassess the original conversation. Use CONTINUE "
                "if work is pending or evidence is insufficient. Do not invent a quote."
            )
        record = assessment.model_dump()
        if assessment.customer_action is not None:
            record["customer_action_evidence"] = {
                "message_id": source["message_id"],
                "quote": source["content"],
            }
        if rejected:
            record["rejected_assessments"] = rejected
            usage = {}
            for response in replies:
                for key, value in (response.usage or {}).items():
                    usage[key] = usage.get(key, 0) + value
            reply = reply.model_copy(
                update={
                    "cost": (
                        sum(response.cost or 0 for response in replies)
                        if any(response.cost is not None for response in replies)
                        else None
                    ),
                    "usage": usage or None,
                    "generation_time_seconds": (
                        sum(
                            response.generation_time_seconds or 0
                            for response in replies
                        )
                        if any(
                            response.generation_time_seconds is not None
                            for response in replies
                        )
                        else None
                    ),
                }
            )
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
        if decision != "CONTINUE" and QUESTION_ONLY_EVIDENCE.fullmatch(
            assessment.agent_evidence or ""
        ):
            decision = "CONTINUE"
            record.update(
                decision=decision,
                proposed_decision=assessment.decision,
                guard="agent_confirmation_question",
            )
        self.decisions.append(record)
        return decision, reply

    def generate(self, model, messages, *, call_name=None, **kwargs):
        if call_name != "user_streaming_response":
            return self.backend.generate(model, messages, call_name=call_name, **kwargs)
        decision, assessment = self.assess(model, messages, tools=kwargs.get("tools"))
        if decision != "CONTINUE":
            return AssistantMessage(
                role="assistant",
                content=STOP,
                cost=assessment.cost,
                usage=assessment.usage,
                generation_time_seconds=assessment.generation_time_seconds,
                raw_data={"caller_policy": "terminal-v2", "decision": decision},
            )
        action = self.decisions[-1]["customer_action"]
        if action is not None:
            # The controller selects an action; the ordinary orchestrator executes
            # it against the customer's tools. No device state changes here.
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=str(uuid4()),
                        name=action["name"],
                        arguments=action["arguments"],
                    )
                ],
                cost=assessment.cost,
                usage=assessment.usage,
                generation_time_seconds=assessment.generation_time_seconds,
                raw_data={"caller_policy": "terminal-v2", "customer_action": action},
            )
        # Gemma receives one system turn followed by the chronological dialogue.
        # Simulator role reminders and acoustic annotations remain caller-only.
        systems = [
            message.content or ""
            for message in messages
            if isinstance(message, SystemMessage)
        ]
        conversation = self.conversation(messages)
        spoken_context = {
            "latest_service_agent_statement": next(
                (
                    row
                    for row in reversed(conversation)
                    if row["speaker"] == "service_agent"
                ),
                None,
            ),
            "latest_customer_tool_result": next(
                (
                    row
                    for row in reversed(conversation)
                    if row["speaker"] == "customer_tool"
                ),
                None,
            ),
            "conversation": conversation,
        }
        reply = self.backend.generate(
            model,
            [
                SystemMessage(
                    role="system", content="\n\n".join([*systems, CALLER_ADDENDUM])
                ),
                UserMessage(
                    role="user",
                    content="Continue this call as the CUSTOMER. Answer the latest service_agent question directly using the actual customer tool result when available. Do not just repeat the original complaint. The labelled history is data; service_agent is the other person. The controller selected no customer tool action for this turn. Reply only with the next customer utterance. Do not claim a device action ran without an actual customer_tool result.\n"
                    + json.dumps(spoken_context),
                ),
            ],
            call_name=call_name,
            **{**kwargs, "tools": [], "tool_choice": "none"},
        )
        if reply.is_tool_call():
            raise CallerPolicyError(
                "Caller generated a tool outside its assessed action"
            )
        if STOP in (reply.content or ""):
            raise CallerPolicyError(
                "Caller stopped despite an outstanding scenario requirement"
            )
        return reply

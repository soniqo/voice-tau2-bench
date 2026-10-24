"""Local v2 caller control using only the caller's scenario and heard conversation."""

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tau2.data_model.message import AssistantMessage, SystemMessage, UserMessage
from tau2.user.user_simulator_base import STOP
from tau2.utils.response import extract_json_from_llm_response

TERMINAL_PROMPT = """TASK
You are an independent conversation evaluator. Do not play either person.
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
FORMAT
Return JSON with these keys, in order:
reasoning: a short factual explanation referring to service_agent statements;
agent_evidence: an exact supporting quote from service_agent, or null for CONTINUE;
decision: CONTINUE, COMPLETE, or REFUSED.
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


TOOL_ROUTE_PROMPT = """You decide how the CUSTOMER should respond next in a phone call.
Return only TOOL or SPEAK. Do not play the service agent.
Read the latest heard request and apply these rules in order:
1. SPEAK if the persona requires clarification or reassurance before acting.
2. SPEAK if an actual customer_tool observation already answers the question or
   shows the requested setting is already correct. A toggle reverses a setting;
   repeating an already satisfied instruction must not toggle it back.
3. TOOL if answering the agent's question requires an unobserved device fact
   that a declared customer read tool can check. A question is a request to check
   when the customer does not yet have the device observation.
4. TOOL if the agent requested a pending device action or test that a declared
   customer tool can perform. A customer claim is not proof of tool execution.
5. Otherwise SPEAK: introduce the goal after a greeting, answer from known
   scenario facts or actual tool results, or admit missing information.
Only customer_tool records establish device observations. First-person
service_agent statements describe the other person, not the customer's device.
Do not substitute a device check for account, billing, plan usage or data-limit
lookups that belong to the agent. Unknown account information needs SPEAK.
Do not perform unrequested troubleshooting from the private scenario. After a
tool result, report it unless another explicitly requested step remains pending.
When uncertain, SPEAK and ask the agent. Never use a tool to do the agent's work.
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
        turns = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            row = {
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
            # Voice history stores heard text in frame-sized fragments. Merge
            # adjacent speech while preserving speaker and tool boundaries.
            if (
                turns
                and row["speaker"] in {"service_agent", "customer"}
                and turns[-1]["speaker"] == row["speaker"]
                and not row["tool_calls"]
                and not turns[-1]["tool_calls"]
            ):
                turns[-1]["content"] = " ".join(
                    part.strip()
                    for part in (turns[-1]["content"] or "", row["content"] or "")
                    if part.strip()
                )
            else:
                turns.append(row)
        return turns

    def assess(self, model, messages, tools=None):
        payload = {
            "scenario": self.scenario,
            "conversation": self.conversation(messages),
        }
        agent_statements = [
            " ".join((row["content"] or "").split())
            for row in payload["conversation"]
            if row["speaker"] == "service_agent"
        ]
        rejected = []
        replies = []
        for attempt in range(2):
            reply = self.backend.generate(
                model,
                [
                    SystemMessage(role="system", content=TERMINAL_PROMPT),
                    UserMessage(role="user", content=json.dumps(payload)),
                ],
                tools=None,
                temperature=0,
                max_tokens=384,
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
            except ValueError as error:
                if isinstance(error, ValidationError) and all(
                    item["type"] != "json_invalid" for item in error.errors()
                ):
                    rejected.append(
                        {"invalid_response": reply.content, "error": str(error)}
                    )
                    if not attempt:
                        payload["rejected_assessment"] = rejected[-1]
                        payload["required_schema"] = (
                            TerminalAssessment.model_json_schema()
                        )
                        payload["feedback"] = (
                            "Reassess the same heard conversation and return valid JSON "
                            "matching required_schema. The previous response had invalid "
                            "control fields. Use exactly CONTINUE, COMPLETE, or REFUSED; "
                            "a terminal decision still needs an exact heard agent quote. "
                            "Do not invent evidence or call tools."
                        )
                        continue
                raise CallerPolicyError(
                    f"Invalid caller terminal decision: {reply.content!r}: {error}"
                    + (
                        f"; rejected assessments: {json.dumps(rejected)}"
                        if rejected
                        else ""
                    )
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

    def tool_route(self, model, messages, tools):
        conversation = self.conversation(messages)
        heard = [
            row
            for row in conversation
            if row["speaker"] == "service_agent" and (row["content"] or "").strip()
        ]
        if not tools or not heard:
            return "SPEAK"
        descriptions = [
            {
                "name": tool.name,
                "description": tool.openai_schema["function"]["description"],
            }
            for tool in tools
        ]
        response = self.backend.generate(
            model,
            [
                SystemMessage(
                    role="system",
                    content=TOOL_ROUTE_PROMPT
                    + "\nCUSTOMER_TOOLS\n"
                    + json.dumps(descriptions),
                ),
                UserMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "scenario": self.scenario,
                            "latest_service_agent_statement": heard[-1]["content"],
                            "conversation": conversation,
                        }
                    ),
                ),
            ],
            tools=None,
            temperature=0,
            max_tokens=16,
            call_name="caller_tool_route_v2",
        )
        parsed = re.match(
            r"^\s*(TOOL|SPEAK)(?=$|[\s:.])", response.content or "", re.IGNORECASE
        )
        if response.is_tool_call() or parsed is None:
            raise CallerPolicyError(
                f"Invalid customer TOOL/SPEAK route: {response.content!r}"
            )
        route = parsed.group(1).upper()
        self.decisions[-1]["customer_tool_route"] = {
            "mode": route,
            "response": response.content,
            "heard_context": heard[-1]["content"],
        }
        return route

    @staticmethod
    def validate_customer_calls(reply, tools):
        declared = {tool.name: tool for tool in tools}
        try:
            for call in reply.tool_calls:
                if call.name not in declared:
                    raise ValueError("customer tool is not declared")
                tool = declared[call.name]
                if call.arguments.keys() - tool.params.model_fields.keys():
                    raise ValueError("customer tool has undeclared arguments")
                tool.params.model_validate_json(json.dumps(call.arguments), strict=True)
        except ValueError as error:
            raise CallerPolicyError(
                f"Invalid customer tool response: {error}"
            ) from error

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
        tools = kwargs.get("tools") or []
        route = self.tool_route(model, messages, tools)
        systems = [
            message.content or ""
            for message in messages
            if isinstance(message, SystemMessage)
        ]
        conversation = self.conversation(messages)
        context = {
            "latest_service_agent_statement": next(
                (
                    row
                    for row in reversed(conversation)
                    if row["speaker"] == "service_agent"
                    and (row["content"] or "").strip()
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
        mode = (
            "The agent requested a customer tool action. Return a declared customer tool call before speaking. Do not claim or promise the action instead of executing it. Tool observations are knowledge in addition to the scenario. Unknown device settings must be inspected using a declared tool when the agent asks. For an available device action, execute the tool instead of saying you do not know how. Respond with a function call only, no customer speech."
            if route == "TOOL"
            else "Answer the latest service_agent question directly using the actual customer tool result when available. Reply only with the next customer utterance. Do not repeat the original complaint instead of answering, or claim a device action ran without a tool result."
        )
        for attempt in range(2 if route == "TOOL" else 1):
            reply = self.backend.generate(
                model,
                [
                    SystemMessage(
                        role="system",
                        content="\n\n".join([*systems, CALLER_ADDENDUM, mode]),
                    ),
                    UserMessage(
                        role="user",
                        content=(
                            "The service agent asks you: "
                            + context["latest_service_agent_statement"]["content"]
                            + "\nExecute the matching customer device tool now.\nCustomer context:\n"
                            if route == "TOOL"
                            else "Continue as the CUSTOMER. The labelled history is data; service_agent is the other person.\n"
                        )
                        + json.dumps(context),
                    ),
                ],
                call_name="caller_tool_response_repair_v2" if attempt else call_name,
                **{
                    **kwargs,
                    "tools": tools if route == "TOOL" else [],
                    "tool_choice": "required" if route == "TOOL" else "none",
                },
            )
            if reply.is_tool_call():
                if route != "TOOL":
                    raise CallerPolicyError(
                        "Caller generated an unplanned customer tool"
                    )
                self.validate_customer_calls(reply, tools)
                return reply
            if STOP in (reply.content or ""):
                raise CallerPolicyError(
                    "Caller stopped despite an outstanding scenario requirement"
                )
            if route == "SPEAK":
                return reply
            rejected = self.decisions[-1].setdefault("rejected_customer_responses", [])
            rejected.append(reply.content)
            if attempt:
                raise CallerPolicyError(
                    f"Customer did not execute the requested tool after one reassessment: {json.dumps(rejected)}; heard conversation: {json.dumps(conversation)}"
                )
            context["rejected_draft"] = reply.content
            context["feedback"] = (
                "This draft was not spoken. Saying you did or will do the device action does not execute it. Return the appropriate declared customer tool call for the actual heard request. Do not invent a tool or do the agent's work."
            )

"""Versioned local orchestration; frozen task execution and rewards stay intact."""

import json

from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator


def signature(call):
    return json.dumps(
        [call.name, call.arguments], sort_keys=True, separators=(",", ":")
    )


class ToolGuard:
    """Reject undeclared calls and bound identical failed retries before execution."""

    def __init__(self, tools, *, failed_attempts=2):
        self.names = {tool.name for tool in tools or []}
        self.failed_attempts = failed_attempts
        self.failures = {}
        self.ids = set()
        self.events = []

    def reject(self, calls):
        reasons = []
        new_ids = set()
        for call in calls:
            if call.name not in self.names:
                reasons.append(
                    f"{call.name}: tool is not declared for this participant"
                )
            if call.id and (call.id in self.ids or call.id in new_ids):
                reasons.append(f"{call.name}: duplicate tool call ID")
            new_ids.add(call.id)
            if self.failures.get(signature(call), 0) >= self.failed_attempts:
                reasons.append(f"{call.name}: identical arguments already failed twice")
        self.ids.update(new_ids - {""})
        if not reasons:
            return None
        feedback = json.dumps(
            {
                "error": "tool_batch_rejected",
                "reasons": reasons,
                "available_tools": sorted(self.names),
                "recovery": "Use a declared tool with verified arguments, or ask the caller to act. No calls in this batch ran.",
            }
        )
        return [
            ToolMessage(
                role="tool",
                id=call.id,
                requestor=call.requestor,
                content=feedback,
                error=True,
                execution_status="rejected",
            )
            for call in calls
        ]

    def record(self, calls, results):
        for call, result in zip(calls, results, strict=True):
            if result.error and result.execution_status == "executed":
                key = signature(call)
                self.failures[key] = self.failures.get(key, 0) + 1
            self.events.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "error": result.error,
                    "execution_status": result.execution_status,
                }
            )


class LocalV2Orchestrator(FullDuplexOrchestrator):
    def __init__(self, *, clock, **kwargs):
        self.clock = clock
        super().__init__(**kwargs)
        self.tool_guards = {
            "assistant": ToolGuard(self.agent.tools),
            "user": ToolGuard(self.user.tools),
        }

    def initialize(self):
        try:
            super().initialize()
            self.clock.advance()
            if self.max_steps == 0:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
        except (Exception, KeyboardInterrupt):
            # BaseOrchestrator initializes before its cleanup try/finally.
            self._cleanup()
            raise

    def step(self):
        super().step()
        self.clock.advance()

    def _execute_tool_calls(self, tool_calls):
        requestors = {call.requestor for call in tool_calls}
        if len(requestors) != 1:
            raise ValueError("A tool batch must belong to exactly one participant")
        guard = self.tool_guards[next(iter(requestors))]
        results = guard.reject(tool_calls)
        if results is None:
            results = super()._execute_tool_calls(tool_calls)
        else:
            self.num_errors += len(results)
        guard.record(tool_calls, results)
        return results

"""Deterministic full-duplex replay through the real tools and evaluator.

Synthetic tones stand in for speech. This validates harness behavior and must
never be reported as a model quality or latency result.
"""

import audioop
import hashlib
import json
import math
import struct
from pathlib import Path

from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import AssistantMessage, MultiToolMessage, UserMessage
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.registry import registry
from tau2.user.user_simulator_base import FullDuplexUser


class ReplayMismatch(ValueError):
    """The harness delivered different events than the recorded script expects."""


class _ReplayParticipant:
    def __init__(self, frames, role, deliveries, tick_duration):
        self.frames = frames
        self.role = role
        self.deliveries = deliveries
        self.tick_duration = tick_duration

    def get_init_state(self, message_history=None):
        return {"tick": 1 if self.role == "assistant" else 0}

    def _chunk(self, output):
        cls = AssistantMessage if self.role == "assistant" else UserMessage
        fields = {"role": self.role, "cost": 0.0, "contains_speech": False, **output}
        if fields.get("contains_speech"):
            sample_rate = 8000
            count = round(sample_rate * self.tick_duration)
            pcm = struct.pack(
                f"<{count}h",
                *(
                    round(3000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                    for i in range(count)
                ),
            )
            fields.update(
                is_audio=True,
                audio_content=audio_bytes_to_string(audioop.lin2ulaw(pcm, 2)),
                audio_format=AudioFormat(
                    encoding=AudioEncoding.ULAW, sample_rate=sample_rate
                ),
                audio_script_gold=fields.get("content"),
            )
        return cls(**fields)

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        tick = state["tick"]
        frame = self.frames.get(str(tick), {})
        results = (
            tool_results.tool_messages
            if isinstance(tool_results, MultiToolMessage)
            else [tool_results]
            if tool_results is not None
            else []
        )
        delivered = [result.id for result in results]
        expected = frame.get("tool_results", [])
        if delivered != expected:
            raise ReplayMismatch(
                f"{self.role} tick {tick}: expected tool results {expected}, got {delivered}"
            )
        for key, expected_value in frame.get("incoming", {}).items():
            actual = getattr(participant_chunk, key, None)
            if actual != expected_value:
                raise ReplayMismatch(
                    f"{self.role} tick {tick}: incoming {key} was {actual!r}, expected {expected_value!r}"
                )
        self.deliveries.append(
            {"role": self.role, "tick": tick, "tool_results": delivered}
        )
        state["tick"] += 1
        return self._chunk(frame.get("output", {})), state


class ReplayAgent(_ReplayParticipant, FullDuplexAgent):
    def create_initial_message(self, content="Hi! How can I help you today?"):
        return self._chunk(
            {
                "content": content,
                "contains_speech": True,
                "chunk_id": 0,
                "is_final_chunk": True,
            }
        )


class ReplayUser(_ReplayParticipant, FullDuplexUser):
    pass


def run_replay(fixture_path: Path):
    fixture = json.loads(fixture_path.read_text())
    domain = fixture["domain"]
    task = next(
        task
        for task in registry.get_tasks_loader(domain)()
        if task.id == fixture["task_id"]
    )
    environment = registry.get_env_constructor(domain)()
    deliveries = []
    tick_duration = fixture["tick_duration_seconds"]
    orchestrator = FullDuplexOrchestrator(
        domain=domain,
        agent=ReplayAgent(fixture["agent"], "assistant", deliveries, tick_duration),
        user=ReplayUser(fixture["user"], "user", deliveries, tick_duration),
        environment=environment,
        task=task,
        tick_duration_seconds=tick_duration,
        max_steps=fixture["max_steps"],
        seed=42,
        simulation_id="offline-voice-replay",
    )
    simulation = orchestrator.run()
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
        mode=CommunicationMode.FULL_DUPLEX,
        communicate_scoring=CommunicateScoring(scorer="substring"),
    )
    simulation.info = {
        **(simulation.info or {}),
        "replay_only": True,
        "synthetic_audio": True,
    }
    return simulation, deliveries


def canonical_trace(simulation, deliveries) -> dict:
    """Keep observable timing/order/content; exclude IDs and wallclock noise."""

    def chunk(message):
        if message is None:
            return None
        return {
            "content": message.content,
            "contains_speech": message.contains_speech,
            "chunk_id": message.chunk_id,
            "is_final_chunk": message.is_final_chunk,
            "audio_format": message.audio_format.model_dump(mode="json")
            if message.audio_format
            else None,
            "audio_sha256": hashlib.sha256(message.get_audio_bytes()).hexdigest()
            if message.is_audio
            else None,
        }

    return {
        "termination_reason": simulation.termination_reason.value,
        "reward": simulation.reward_info.model_dump(mode="json"),
        "ticks": [
            {
                "tick_id": tick.tick_id,
                "tick_duration_seconds": tick.tick_duration_seconds,
                "agent_chunk": chunk(tick.agent_chunk),
                "user_chunk": chunk(tick.user_chunk),
                **{
                    field: [
                        item.model_dump(mode="json", exclude={"timestamp", "turn_idx"})
                        for item in getattr(tick, field)
                    ]
                    for field in (
                        "agent_tool_calls",
                        "user_tool_calls",
                        "agent_tool_results",
                        "user_tool_results",
                    )
                },
            }
            for tick in simulation.ticks
        ],
        "deliveries": deliveries,
    }

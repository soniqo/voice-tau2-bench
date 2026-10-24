import json
from types import SimpleNamespace

import pytest

from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator import base
from tau2.orchestrator.modes import CommunicationMode
from tau2.registry import registry
from tau2.voice.baseline import repository_root
from tau2.voice.execution import SimulationClock
from tau2.voice.local_orchestrator import LocalV2Orchestrator
from tau2.voice.replay import ReplayAgent, ReplayUser


def run(monkeypatch, *, caller_delay, timeout=10000, max_steps=200):
    fixture = json.loads(
        (repository_root() / "data/tau2/replays/voice-create-task.json").read_text()
    )
    environment = registry.get_env_constructor("mock")()
    task = next(
        task
        for task in registry.get_tasks_loader("mock")()
        if task.id == fixture["task_id"]
    )
    wall = SimpleNamespace(now=0)
    monkeypatch.setattr(base, "time", SimpleNamespace(perf_counter=lambda: wall.now))

    class SlowCaller(ReplayUser):
        def get_next_chunk(self, *args, **kwargs):
            wall.now += caller_delay
            return super().get_next_chunk(*args, **kwargs)

    agent = ReplayAgent(fixture["agent"], "assistant", [], 0.05)
    agent.tools = environment.get_tools()
    caller = SlowCaller(fixture["user"], "user", [], 0.05)
    caller.tools = None
    clock = SimulationClock()
    orchestrator = LocalV2Orchestrator(
        clock=clock,
        domain="mock",
        agent=agent,
        user=caller,
        environment=environment,
        task=task,
        max_steps=max_steps,
        tick_duration_seconds=0.05,
        timeout=timeout,
    )
    simulation = orchestrator.run()
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ENV,
        solo_mode=False,
        domain="mock",
        mode=CommunicationMode.FULL_DUPLEX,
        communicate_scoring=CommunicateScoring(scorer="substring"),
    )
    return simulation, clock


def test_caller_delay_leaves_simulation_tools_reward_and_audio_duration_unchanged(
    monkeypatch,
):
    fast, fast_clock = run(monkeypatch, caller_delay=0)
    slow, slow_clock = run(monkeypatch, caller_delay=100)
    assert (
        fast.termination_reason
        == slow.termination_reason
        == TerminationReason.USER_STOP
    )
    assert fast.reward_info.reward == slow.reward_info.reward == 1
    assert len(fast.ticks) == len(slow.ticks)
    assert fast_clock.now == slow_clock.now == len(slow.ticks) * 0.05
    assert [[call.name for call in tick.agent_tool_calls] for tick in fast.ticks] == [
        [call.name for call in tick.agent_tool_calls] for tick in slow.ticks
    ]


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"timeout": 1}, TerminationReason.TIMEOUT),
        ({"max_steps": 1}, TerminationReason.MAX_STEPS),
    ],
)
def test_wall_watchdog_and_call_cap_remain_distinct_failures(
    monkeypatch, kwargs, reason
):
    simulation, _ = run(monkeypatch, caller_delay=2, **kwargs)
    assert simulation.termination_reason == reason
    assert simulation.reward_info.reward == 0

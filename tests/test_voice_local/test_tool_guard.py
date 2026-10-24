from types import SimpleNamespace

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.voice.local_orchestrator import LocalV2Orchestrator, ToolGuard


def call(name="lookup", id="one", **arguments):
    return ToolCall(id=id, name=name, arguments=arguments)


def test_mixed_undeclared_batch_is_entirely_rejected_without_side_effects():
    orchestrator = LocalV2Orchestrator.__new__(LocalV2Orchestrator)
    orchestrator.tool_guards = {
        "assistant": ToolGuard([SimpleNamespace(name="lookup")])
    }
    orchestrator.num_errors = 0
    orchestrator.environment = SimpleNamespace(
        get_response=lambda _: (_ for _ in ()).throw(AssertionError("must not execute"))
    )
    results = orchestrator._execute_tool_calls(
        [call(), call("run_speed_test", id="two")]
    )
    assert all(
        result.execution_status == "rejected" and result.error for result in results
    )
    assert [result.id for result in results] == ["one", "two"]
    assert orchestrator.num_errors == 2
    assert len(orchestrator.tool_guards["assistant"].events) == 2


def test_identical_failed_retries_are_bounded_but_repaired_arguments_can_execute():
    guard = ToolGuard([SimpleNamespace(name="lookup")])
    for id in ("a", "b"):
        attempt = call(id=id, account="wrong")
        assert guard.reject([attempt]) is None
        guard.record(
            [attempt],
            [ToolMessage(role="tool", id=id, error=True, content="not found")],
        )
    assert (
        guard.reject([call(id="c", account="wrong")])[0].execution_status == "rejected"
    )
    assert guard.reject([call(id="d", account="verified")]) is None


def test_successful_new_call_id_can_repeat_but_duplicate_ids_cannot_execute():
    guard = ToolGuard([SimpleNamespace(name="lookup")])
    attempt = call()
    assert guard.reject([attempt]) is None
    guard.record([attempt], [ToolMessage(role="tool", id=attempt.id, content="found")])
    assert guard.reject([attempt])[0].execution_status == "rejected"
    assert guard.reject([call(id="new")]) is None


def test_valid_call_rejected_with_invalid_sibling_can_be_retried_alone():
    guard = ToolGuard([SimpleNamespace(name="lookup")])
    for index in range(2):
        batch = [call(id=f"valid-{index}"), call("undeclared", id=f"invalid-{index}")]
        rejected = guard.reject(batch)
        guard.record(batch, rejected)
    assert guard.reject([call(id="valid-alone")]) is None

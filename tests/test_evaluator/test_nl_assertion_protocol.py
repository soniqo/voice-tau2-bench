import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import EvaluationCriteria, RewardType, Task, UserScenario
from tau2.evaluator import evaluator_nl_assertions as module

EVALUATORS = [module.NLAssertionsEvaluator, module.FullDuplexNLAssertionsEvaluator]
OUTCOMES = ["The agent stated the refund amount.", "The agent explained the wait."]


def task(outcomes=None):
    return Task(
        id="judge-protocol",
        user_scenario=UserScenario(instructions="Ask about a refund."),
        evaluation_criteria=EvaluationCriteria(
            nl_assertions=OUTCOMES if outcomes is None else outcomes,
            reward_basis=[RewardType.NL_ASSERTION],
        ),
    )


def response(rows):
    return {"results": rows}


def result(outcome, met=True):
    return {"expectedOutcome": outcome, "metExpectation": met, "reasoning": "fixture"}


def backend(monkeypatch, payload):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(role="assistant", content=json.dumps(payload))

    monkeypatch.setattr(module, "generate", generate)
    return calls


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_missing_assertion_results_cannot_be_a_success(monkeypatch, evaluator):
    backend(monkeypatch, response([]))
    with pytest.raises(ValueError, match="judge"):
        evaluator.calculate_reward(task(), [])


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        None,
        {"results": None},
        {"results": {}},
        response([result(OUTCOMES[0])]),
        response([result(OUTCOMES[0]), result(OUTCOMES[0])]),
        response([result(OUTCOMES[0]), result("An unrequested outcome.")]),
        response(
            [result(OUTCOMES[0]), result(OUTCOMES[1]), result("An extra outcome.")]
        ),
        response([result(OUTCOMES[0]), result(OUTCOMES[1], "false")]),
        response([result(OUTCOMES[0]), result(OUTCOMES[1], 1)]),
        response(
            [
                result(OUTCOMES[0]),
                {"expectedOutcome": OUTCOMES[1], "metExpectation": False},
            ]
        ),
        response(
            [
                result(OUTCOMES[0]),
                {"expectedOutcome": 1, "metExpectation": True, "reasoning": "fixture"},
            ]
        ),
    ],
)
def test_malformed_or_mismatched_results_are_unscored(monkeypatch, evaluator, payload):
    backend(monkeypatch, payload)
    with pytest.raises(ValueError, match="judge"):
        evaluator.calculate_reward(task(), [])


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("second_met", [False, True])
def test_valid_results_keep_the_existing_product_and_allow_reordering(
    monkeypatch, evaluator, second_met
):
    calls = backend(
        monkeypatch,
        {
            "results": [result(OUTCOMES[1], second_met), result(OUTCOMES[0])],
            "metadata": "ignored",
        },
    )
    reward = evaluator.calculate_reward(task(), [])
    assert reward.reward == float(second_met)
    assert [check.nl_assertion for check in reward.nl_assertions] == [
        OUTCOMES[1],
        OUTCOMES[0],
    ]
    assert len(calls) == 1 and calls[0]["call_name"] == "nl_assertions_eval"


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_repeated_requested_outcome_keeps_its_multiplicity(monkeypatch, evaluator):
    backend(monkeypatch, response([result(OUTCOMES[0]), result(OUTCOMES[0], False)]))
    reward = evaluator.calculate_reward(task([OUTCOMES[0], OUTCOMES[0]]), [])
    assert reward.reward == 0 and len(reward.nl_assertions) == 2


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_absent_assertions_stay_neutral_without_inference(monkeypatch, evaluator):
    calls = backend(monkeypatch, response([]))
    reward = evaluator.calculate_reward(task([]), [])
    assert reward.reward == 1 and reward.nl_assertions == [] and calls == []


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_tool_call_cannot_be_an_assertion_result(monkeypatch, evaluator):
    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="judge-call", name="unsupported", arguments={})],
        ),
    )
    with pytest.raises(ValueError, match="judge"):
        evaluator.calculate_reward(task(), [])


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_invalid_json_is_unscored(monkeypatch, evaluator):
    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: AssistantMessage(role="assistant", content="not JSON"),
    )
    with pytest.raises(ValueError):
        evaluator.calculate_reward(task(), [])


@pytest.mark.parametrize("evaluator", EVALUATORS)
def test_native_failure_propagates_without_retry_or_a_score(monkeypatch, evaluator):
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("native unavailable")

    monkeypatch.setattr(module, "generate", unavailable)
    with pytest.raises(RuntimeError, match="native unavailable"):
        evaluator.calculate_reward(task(), [])
    assert len(calls) == 1


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize("tag", ["json", ""])
def test_complete_code_fence_keeps_the_actual_json_values(monkeypatch, evaluator, tag):
    payload = json.dumps(response([result(OUTCOMES[0]), result(OUTCOMES[1], False)]))
    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: AssistantMessage(
            role="assistant", content=f"```{tag}\n{payload}\n```"
        ),
    )
    reward = evaluator.calculate_reward(task(), [])
    assert reward.reward == 0
    assert [check.met for check in reward.nl_assertions] == [True, False]


@pytest.mark.parametrize("evaluator", EVALUATORS)
@pytest.mark.parametrize(
    "wrapper",
    [
        "Here is the result:\n```json\nPAYLOAD\n```",
        "```json\nPAYLOAD\n```\nMore text",
        "PAYLOAD\nPAYLOAD",
        "```json\nPAYLOAD\n```\n```json\nPAYLOAD\n```",
    ],
)
def test_ambiguous_or_prose_wrapped_json_is_unscored(monkeypatch, evaluator, wrapper):
    content = wrapper.replace(
        "PAYLOAD", json.dumps(response([result(o) for o in OUTCOMES]))
    )
    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: AssistantMessage(role="assistant", content=content),
    )
    with pytest.raises(ValueError, match="judge"):
        evaluator.calculate_reward(task(), [])

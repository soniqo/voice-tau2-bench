import pytest

from tau2.voice.baseline import repository_root
from tau2.voice.runtime_profile import (
    LocalVoiceProfile,
    RuntimeCapabilities,
    load_profile,
)


def test_v3_has_explicit_patience_and_requires_reasoning_history():
    from tau2.voice.baseline import repository_root

    root = repository_root() / "configs/voice"
    previous = load_profile(root / "soniqo-local-gemma12b-v2.json")
    requested = load_profile(root / "soniqo-local-gemma12b-v3.json")
    assert previous.execution.caller_idle_seconds == 15
    assert requested.execution.caller_idle_seconds == 30
    assert requested.execution.price_grounding is True
    assert previous.execution.price_grounding is False
    assert requested.execution.goal_progress is True
    assert previous.execution.goal_progress is False
    assert requested.budgets() == (600, 1200)
    for role in requested.roles:
        if role != "agent":
            assert requested.roles[role] == previous.roles[role]
    assert requested.roles["agent"].model_id == "gemma4-12b-it"
    actual = capabilities()
    actual.models["gemma4-12b-it"] = previous.roles["agent"].model_copy(deep=True)
    with pytest.raises(ValueError, match="reasoning_history"):
        requested.check_runtime(actual)
    actual.models["gemma4-12b-it"].features.add("reasoning_history")
    with pytest.raises(ValueError, match="non_thinking_tool_continuation"):
        requested.check_runtime(actual)
    actual.models["gemma4-12b-it"].features.add("non_thinking_tool_continuation")
    requested.check_runtime(actual)


def profile():
    return LocalVoiceProfile.model_validate_json(
        (repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )


def test_question_focus_requires_separate_v4_profile():
    requested = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v4.json"
    )
    assert requested.execution.question_focus
    assert requested.budgets() == (600, 1200)
    data = requested.model_dump(mode="json")
    data["profile"] = "soniqo-local-gemma12b-v3"
    with pytest.raises(ValueError, match="separate v4"):
        LocalVoiceProfile.model_validate(data)


def test_routing_focus_requires_separate_v5_profile():
    requested = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v5.json"
    )
    assert requested.execution.routing_focus
    data = requested.model_dump(mode="json")
    data["profile"] = "soniqo-local-gemma12b-v4"
    with pytest.raises(ValueError, match="separate v5"):
        LocalVoiceProfile.model_validate(data)


def capabilities():
    models = {}
    for required in profile().roles.values():
        if required.model_id in models:
            models[required.model_id].features |= required.features
        else:
            models[required.model_id] = required.model_copy(deep=True)
    return RuntimeCapabilities(runtime="fixture-only", revision="test", models=models)


def test_complete_capabilities_satisfy_all_roles():
    profile().check_runtime(capabilities())


def test_inventory_features_have_stable_order_for_cross_process_resume():
    snapshots = [
        profile().model_dump(mode="json")["roles"],
        capabilities().model_dump(mode="json")["models"],
    ]
    for snapshot in snapshots:
        for model in snapshot.values():
            assert model["features"] == sorted(model["features"])


@pytest.mark.parametrize("precision", ["int8", "bf16", "fp32"])
def test_voxcpm_precision_cannot_silently_change(precision):
    actual = capabilities()
    actual.models["voxcpm2"].precision = precision
    with pytest.raises(ValueError, match=f"requires fp16, runtime reports {precision}"):
        profile().check_runtime(actual)


def test_whisper_turbo_cannot_replace_requested_large_v3():
    actual = capabilities()
    actual.models["whisper-large-v3-turbo"] = actual.models.pop("whisper-large-v3")
    with pytest.raises(ValueError, match="whisper-large-v3.*unavailable"):
        profile().check_runtime(actual)


@pytest.mark.parametrize(
    "feature", ["tool_calls", "tool_history", "cancel", "json_output"]
)
def test_gemma_requires_tools_history_cancellation_and_judging(feature):
    actual = capabilities()
    actual.models["gemma4-e4b-it"].features.remove(feature)
    with pytest.raises(ValueError, match=feature):
        profile().check_runtime(actual)


def test_listener_and_judge_roles_cannot_be_omitted():
    requested = profile()
    del requested.roles["listener"]
    del requested.roles["judge"]
    with pytest.raises(ValueError, match="missing role: judge"):
        requested.check_runtime(capabilities())


def test_v2_budget_defaults_keep_call_and_wall_time_separate():
    requested = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v2.json"
    )
    assert requested.budgets() == (2400, 7200)
    assert requested.budgets(600) == (600, 7200)
    assert profile().budgets() == (600, 600)
    assert profile().budgets(1200) == (1200, 1200)
    assert requested.execution.device_grounding is True
    assert profile().execution.device_grounding is False


def test_historical_profiles_cannot_enable_new_device_grounding():
    data = profile().model_dump(mode="json")
    data["execution"]["device_grounding"] = True
    with pytest.raises(ValueError, match="Historical v1"):
        LocalVoiceProfile.model_validate(data)


@pytest.mark.parametrize(
    "call,wall", [(0, None), (float("nan"), None), (float("inf"), None), (600, -1)]
)
def test_invalid_clock_limits_fail_before_inference(call, wall):
    with pytest.raises(ValueError):
        profile().budgets(call, wall)

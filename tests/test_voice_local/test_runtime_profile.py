import pytest

from tau2.voice.baseline import repository_root
from tau2.voice.runtime_profile import LocalVoiceProfile, RuntimeCapabilities


def profile():
    return LocalVoiceProfile.model_validate_json(
        (repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )


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

"""Explicit opt-in real-provider test; never part of offline adapter validation."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.local_models


def test_real_provider_audio_tools_and_frozen_task_grader(tmp_path):
    if os.environ.get("TAU_PROVIDER_E2E") != "1" or not os.environ.get(
        "SPEECH_SWIFT_INFERENCE_URL"
    ):
        pytest.skip("Requires explicit TAU_PROVIDER_E2E and the single local runtime")
    from tau2.voice.baseline import repository_root
    from tau2.voice.local_run import run_local_task
    from tau2.voice.provider_run import ProviderBudget, ProviderFactory
    from tau2.voice.providers.base import ProviderSettings

    provider = os.environ["TAU_PROVIDER"]
    settings = ProviderSettings(
        provider=provider,
        model=os.environ["TAU_PROVIDER_MODEL"],
        voice=os.environ["TAU_PROVIDER_VOICE"],
        key_env=os.environ["TAU_PROVIDER_KEY_ENV"],
        max_seconds=90,
        stop_usd=0.25,
        backend_model=os.environ.get("TAU_PROVIDER_BACKEND_MODEL"),
        agent_id=os.environ.get("TAU_PROVIDER_AGENT_ID"),
        preset_sha256=os.environ.get("TAU_PROVIDER_PRESET_SHA256"),
        usd_per_credit=float(os.environ["TAU_PROVIDER_USD_PER_CREDIT"])
        if provider == "elevenlabs"
        else None,
        max_usd_per_minute=float(os.environ["TAU_PROVIDER_MAX_USD_PER_MINUTE"])
        if provider == "elevenlabs"
        else None,
    )
    factory = ProviderFactory(settings, ProviderBudget(tmp_path / "budget.json", 1))
    result = run_local_task(
        domain="mock",
        task_id="create_task_1",
        seed=70,
        base_url=os.environ["SPEECH_SWIFT_INFERENCE_URL"],
        output=tmp_path / "run",
        max_seconds=90,
        wall_timeout=90,
        qualification_path=Path(os.environ["TAU_PROVIDER_QUALIFICATION"]),
        profile_path=repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json",
        agent_factory=factory,
        external_agent=factory.descriptor,
    )
    assert result["reward"] == 1, result

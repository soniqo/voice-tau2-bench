"""Factories and persistent reservations for explicitly selected cloud agents."""

import hashlib

from tau2.agent.provider_agent import ProviderAgent
from tau2.voice.nova_run import NovaBudget
from tau2.voice.pricing import PRICING, PRICING_VERSION
from tau2.voice.providers.base import ProviderSettings


class ProviderBudget(NovaBudget):
    def reserve(self, stop_usd):
        try:
            return super().reserve(stop_usd)
        except ValueError as error:
            raise ValueError(str(error).replace("Nova", "voice-provider")) from None


def session_class(provider):
    if provider == "openai-realtime":
        from tau2.voice.providers.realtime import RealtimeSession

        return RealtimeSession
    if provider == "openai-live":
        from tau2.voice.providers.live import LiveSession

        return LiveSession
    if provider == "gemini-live":
        from tau2.voice.providers.gemini import GeminiSession

        return GeminiSession
    if provider == "elevenlabs":
        from tau2.voice.providers.elevenlabs import ElevenSession

        return ElevenSession
    raise ValueError("Unsupported voice provider")


class ProviderFactory:
    def __init__(
        self,
        settings: ProviderSettings,
        budget: ProviderBudget,
        *,
        session_factory=None,
    ):
        self.settings, self.budget = settings, budget
        self.session_factory = session_factory or session_class(settings.provider)
        provider = "gemini" if settings.provider == "gemini-live" else "openai"
        if (
            settings.provider != "elevenlabs"
            and (provider, settings.model) not in PRICING
        ):
            raise ValueError(
                "Explicit verified pricing is required for this provider model"
            )
        if (
            settings.provider == "openai-live"
            and ("openai", settings.backend_model) not in PRICING
        ):
            raise ValueError(
                "Explicit verified pricing is required for the Live backend"
            )

    @property
    def descriptor(self):
        s = self.settings
        return {
            "provider": s.provider,
            "model_id": s.model,
            "voice": s.voice,
            "backend_model": s.backend_model,
            "model_args": {"delegation": "responses"}
            if s.provider == "openai-live"
            else {"preset_sha256": s.preset_sha256}
            if s.provider == "elevenlabs"
            else {"max_output_tokens": 1024},
            "session_seconds": s.max_seconds,
            "stop_usd": s.stop_usd,
            "budget_usd": self.budget.limit,
            "budget_ledger_path_sha256": hashlib.sha256(
                str(self.budget.path.resolve()).encode()
            ).hexdigest(),
            "agent_id_sha256": hashlib.sha256(s.agent_id.encode()).hexdigest()
            if s.agent_id
            else None,
            "preset_sha256": s.preset_sha256,
            "usd_per_credit": s.usd_per_credit,
            "max_usd_per_minute": s.max_usd_per_minute,
            "native_roles": ["caller", "listener", "judge", "caller_tts"],
            "fixed_opener": "speech-swift/voxcpm2",
            "endpointing": {
                "mode": "reflected_audio_noise_gate",
                "rms_threshold": 180,
                "silence_ms": 800,
                "transcript_grace_ms": 500,
            }
            if s.provider == "openai-live"
            else {"mode": "provider_completion_event"},
            "billing": "retain reservation when session-final usage is unconfirmed",
            "pricing_revision": PRICING_VERSION,
        }

    def __call__(self, tools, policy, **kwargs):
        key = self.budget.reserve(self.settings.stop_usd)
        budget = self.budget

        class BudgetedProviderAgent(ProviderAgent):
            def stop(self, *args, **kw):
                try:
                    return super().stop(*args, **kw)
                finally:
                    budget.finish(key, self)

        try:
            return BudgetedProviderAgent(
                tools,
                policy,
                settings=self.settings,
                session_factory=self.session_factory,
                **kwargs,
            )
        except BaseException:
            from types import SimpleNamespace

            budget.finish(key, SimpleNamespace(_thread=None, metrics={}))
            raise

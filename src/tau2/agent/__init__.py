"""Participant exports loaded on demand."""

import warnings
from importlib import import_module

_EXPORTS = {
    "LLMConfigMixin": ("tau2.agent.base.llm_config", "LLMConfigMixin"),
    "FullDuplexParticipant": ("tau2.agent.base.participant", "FullDuplexParticipant"),
    "HalfDuplexParticipant": ("tau2.agent.base.participant", "HalfDuplexParticipant"),
    "StreamingMixin": ("tau2.agent.base.streaming", "StreamingMixin"),
    "StreamingState": ("tau2.agent.base.streaming", "StreamingState"),
    "FullDuplexAgent": ("tau2.agent.base_agent", "FullDuplexAgent"),
    "FullDuplexVoiceAgent": ("tau2.agent.base_agent", "FullDuplexVoiceAgent"),
    "HalfDuplexAgent": ("tau2.agent.base_agent", "HalfDuplexAgent"),
    "HalfDuplexVoiceAgent": ("tau2.agent.base_agent", "HalfDuplexVoiceAgent"),
    "ValidAgentInputMessage": ("tau2.agent.base_agent", "ValidAgentInputMessage"),
}

_ALIASES = {
    "BaseConversationParticipant": "HalfDuplexParticipant",
    "BaseStreamingParticipant": "FullDuplexParticipant",
    "BaseAgent": "HalfDuplexAgent",
    "LocalAgent": "HalfDuplexAgent",
    "BaseStreamingAgent": "FullDuplexAgent",
    "BaseVoiceAgent": "HalfDuplexVoiceAgent",
}


def __getattr__(name: str):
    if name in _ALIASES:
        replacement = _ALIASES[name]
        warnings.warn(
            f"{name} is deprecated, use {replacement} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return __getattr__(replacement)
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = list(_EXPORTS) + list(_ALIASES)

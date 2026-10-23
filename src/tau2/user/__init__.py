"""Participant exports loaded on demand."""

import warnings
from importlib import import_module

_EXPORTS = {
    "DummyUser": ("tau2.user.user_simulator", "DummyUser"),
    "UserSimulator": ("tau2.user.user_simulator", "UserSimulator"),
    "FullDuplexUser": ("tau2.user.user_simulator_base", "FullDuplexUser"),
    "FullDuplexVoiceUser": ("tau2.user.user_simulator_base", "FullDuplexVoiceUser"),
    "HalfDuplexUser": ("tau2.user.user_simulator_base", "HalfDuplexUser"),
    "HalfDuplexVoiceUser": ("tau2.user.user_simulator_base", "HalfDuplexVoiceUser"),
    "UserState": ("tau2.user.user_simulator_base", "UserState"),
    "ValidUserInputMessage": ("tau2.user.user_simulator_base", "ValidUserInputMessage"),
    "VoiceUserSimulator": ("tau2.user.user_simulator_voice", "VoiceUserSimulator"),
    "VoiceStreamingUserSimulator": (
        "tau2.user.user_simulator_streaming",
        "VoiceStreamingUserSimulator",
    ),
}

_ALIASES = {
    "BaseUser": "HalfDuplexUser",
    "BaseStreamingUser": "FullDuplexUser",
    "BaseVoiceUser": "HalfDuplexVoiceUser",
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


__all__ = [
    # Base classes
    "HalfDuplexUser",
    "FullDuplexUser",
    "HalfDuplexVoiceUser",
    "FullDuplexVoiceUser",
    "UserState",
    "ValidUserInputMessage",
    # User simulators
    "UserSimulator",
    "DummyUser",
    # Voice users (lazy imports)
    "VoiceUserSimulator",
    "VoiceStreamingUserSimulator",
    # Deprecated aliases (kept for backward compatibility)
    "BaseUser",
    "BaseStreamingUser",
    "BaseVoiceUser",
]

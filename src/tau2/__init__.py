"""τ-bench public API, loaded on demand to keep optional runtimes isolated."""

import warnings
from importlib import import_module

# Preserve the public singleton even after Python imports the registry submodule.
# Registry entries themselves remain lazy and do not load domains or providers.
from tau2.registry import Registry as Registry
from tau2.registry import registry as registry

# Importing a domain or data model must not initialize a model provider.
_EXPORTS = {
    "FullDuplexOrchestrator": (
        "tau2.orchestrator.full_duplex_orchestrator",
        "FullDuplexOrchestrator",
    ),
    "SpeechSwiftAgent": ("tau2.agent.speech_swift_agent", "SpeechSwiftAgent"),
    "LLMConfigMixin": ("tau2.agent.base.llm_config", "LLMConfigMixin"),
    "HalfDuplexAgent": ("tau2.agent.base_agent", "HalfDuplexAgent"),
    "FullDuplexAgent": ("tau2.agent.base_agent", "FullDuplexAgent"),
    "Task": ("tau2.data_model.tasks", "Task"),
    "Environment": ("tau2.environment.environment", "Environment"),
    "EvaluationType": ("tau2.evaluator.evaluator", "EvaluationType"),
    "evaluate_simulation": ("tau2.evaluator.evaluator", "evaluate_simulation"),
    "CommunicationMode": ("tau2.orchestrator.modes", "CommunicationMode"),
    "Registry": ("tau2.registry", "Registry"),
    "registry": ("tau2.registry", "registry"),
    "UserSimulator": ("tau2.user.user_simulator", "UserSimulator"),
    "HalfDuplexUser": ("tau2.user.user_simulator_base", "HalfDuplexUser"),
    "FullDuplexUser": ("tau2.user.user_simulator_base", "FullDuplexUser"),
    "VoiceStreamingUserSimulator": (
        "tau2.user.user_simulator_streaming",
        "VoiceStreamingUserSimulator",
    ),
    "BaseRunConfig": ("tau2.data_model.simulation", "BaseRunConfig"),
    "RunConfig": ("tau2.data_model.simulation", "RunConfig"),
    "SimulationRun": ("tau2.data_model.simulation", "SimulationRun"),
    "TextRunConfig": ("tau2.data_model.simulation", "TextRunConfig"),
    "VoiceRunConfig": ("tau2.data_model.simulation", "VoiceRunConfig"),
}

_ALIASES = {
    "BaseAgent": "HalfDuplexAgent",
    "LocalAgent": "HalfDuplexAgent",
    "BaseStreamingAgent": "FullDuplexAgent",
    "BaseUser": "HalfDuplexUser",
    "BaseStreamingUser": "FullDuplexUser",
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

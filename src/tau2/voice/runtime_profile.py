"""Requested model roles and explicit runtime capability validation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["realtime-v1", "simulation-v2"] = "realtime-v1"
    call_seconds: float = Field(default=600, ge=0.05, allow_inf_nan=False)
    wall_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    caller_policy: Literal["upstream", "terminal-v2"] = "upstream"
    require_qualification: bool = False
    tool_guard: bool = False
    caller_idle_seconds: float = Field(default=5, ge=1, allow_inf_nan=False)


class ModelRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    precision: str | None = None
    features: set[str] = Field(default_factory=set)

    @field_serializer("features")
    def stable_features(self, value: set[str]) -> list[str]:
        """Keep run inventories identical across processes and hash seeds."""
        return sorted(value)


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    runtime: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    models: dict[str, ModelRequirement]


class LocalVoiceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profile: Literal[
        "soniqo-local-v1", "soniqo-local-gemma12b-v1", "soniqo-local-gemma12b-v2"
    ] = "soniqo-local-v1"
    language: Literal["en"] = "en"
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    roles: dict[
        Literal[
            "agent",
            "caller",
            "listener",
            "judge",
            "agent_tts",
            "caller_tts",
            "asr",
            "vad",
        ],
        ModelRequirement,
    ]

    @model_validator(mode="after")
    def validate_execution_version(self):
        if self.profile.endswith("-v2"):
            if (
                self.execution.mode != "simulation-v2"
                or self.execution.caller_policy != "terminal-v2"
                or not self.execution.require_qualification
                or not self.execution.tool_guard
            ):
                raise ValueError(
                    "The v2 profile requires simulation time, qualified terminal-v2 caller, and tool guard"
                )
        elif self.execution != ExecutionSettings():
            raise ValueError(
                "Historical v1 profiles retain their original execution settings"
            )
        return self

    def budgets(self, max_seconds=None, wall_timeout=None):
        import math

        call = self.execution.call_seconds if max_seconds is None else max_seconds
        wall = (
            self.execution.wall_timeout_seconds
            if wall_timeout is None
            else wall_timeout
        )
        if not math.isfinite(call) or call < 0.05:
            raise ValueError(
                "max_seconds must be finite and at least one 0.05-second tick"
            )
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError("wall_timeout must be finite and positive")
        if self.execution.mode == "realtime-v1" and wall_timeout is None:
            wall = call
        return call, wall

    def check_runtime(self, capabilities: RuntimeCapabilities) -> None:
        """Fail before a run if a role, model, precision, or feature is missing."""
        required_roles = {
            "agent",
            "caller",
            "listener",
            "judge",
            "agent_tts",
            "caller_tts",
            "asr",
            "vad",
        }
        errors = [
            f"missing role: {role}"
            for role in sorted(required_roles - self.roles.keys())
        ]
        for role, required in self.roles.items():
            actual = capabilities.models.get(required.model_id)
            if actual is None:
                errors.append(f"{role}: model {required.model_id!r} unavailable")
                continue
            if actual.model_id != required.model_id or actual.engine != required.engine:
                errors.append(f"{role}: model identity or engine mismatch")
            if (
                required.precision is not None
                and actual.precision != required.precision
            ):
                errors.append(
                    f"{role}: requires {required.precision}, runtime reports {actual.precision}"
                )
            missing = required.features - actual.features
            if missing:
                errors.append(f"{role}: missing features {', '.join(sorted(missing))}")
        if errors:
            raise ValueError("Runtime cannot run this profile:\n" + "\n".join(errors))


def load_profile(path: Path | None = None) -> LocalVoiceProfile:
    from tau2.voice.baseline import repository_root

    return LocalVoiceProfile.model_validate_json(
        (path or repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )

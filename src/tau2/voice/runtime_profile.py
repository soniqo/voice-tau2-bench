"""Requested model roles and explicit runtime capability validation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


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
    profile: Literal["soniqo-local-v1", "soniqo-local-gemma12b-v1"] = "soniqo-local-v1"
    language: Literal["en"] = "en"
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

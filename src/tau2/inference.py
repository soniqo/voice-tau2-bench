"""Run-scoped inference interfaces shared by caller, agent, and evaluators.

A bound run routes every inference call through the supplied backends. Legacy
provider loading is deferred until an unbound caller explicitly generates.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol

from loguru import logger
from pydantic import BaseModel, ValidationError

from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.environment.tool import Tool
from tau2.utils.response import extract_json_from_llm_response

if TYPE_CHECKING:
    from tau2.data_model.audio import AudioData
    from tau2.data_model.voice import (
        RuntimeTranscriptionConfig,
        RuntimeTTSConfig,
        TranscriptionResult,
    )


class ChatBackend(Protocol):
    def generate(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: Optional[list[Tool]] = None,
        tool_choice: Optional[str] = None,
        call_name: Optional[str] = None,
        **kwargs: Any,
    ) -> AssistantMessage | UserMessage: ...


class SynthesisBackend(Protocol):
    def synthesize(self, text: str, config: "RuntimeTTSConfig") -> "AudioData": ...


class TranscriptionBackend(Protocol):
    def transcribe(
        self, audio: "AudioData", config: "RuntimeTranscriptionConfig"
    ) -> "TranscriptionResult": ...


@dataclass(frozen=True)
class InferenceBackends:
    """Explicit services for all inference in one run, with no provider fallback."""

    chat: ChatBackend
    synthesis: SynthesisBackend
    transcription: TranscriptionBackend


_BACKENDS: ContextVar[InferenceBackends | None] = ContextVar(
    "inference_backends", default=None
)


def get_inference_backends() -> InferenceBackends | None:
    return _BACKENDS.get()


@contextmanager
def use_inference_backends(backends: InferenceBackends) -> Iterator[None]:
    """Bind inference for this run; background work must copy the context."""
    token = _BACKENDS.set(backends)
    try:
        yield
    finally:
        _BACKENDS.reset(token)


def _generate_once(model, messages, **kwargs):
    backends = get_inference_backends()
    if backends is not None:
        return backends.chat.generate(model, messages, **kwargs)
    raise RuntimeError("Inference requires explicitly bound local backends")


class LLMResponseValidationError(ValueError):
    """A model reply remained invalid after response validation retries."""

    def __init__(self, model: str, raw_responses: list[str], cause: Exception):
        self.model = model
        self.raw_responses = raw_responses
        super().__init__(f"Invalid {model} response: {cause}")


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    response_model: type[BaseModel] | None = None,
    response_validation_retries: int = 0,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """Generate a reply, retrying invalid structured responses in this boundary.

    The response model validates an extracted JSON object. On a validation
    failure, the model sees its prior reply and one short correction request.
    All invalid replies are retained on the raised exception for diagnosis.
    """
    if response_validation_retries < 0:
        raise ValueError("response_validation_retries must be nonnegative")
    if response_validation_retries and response_model is None:
        raise ValueError("response_validation_retries requires response_model")

    attempt_messages = list(messages)
    raw_responses: list[str] = []
    total_cost = 0.0
    has_cost = False
    for attempt in range(response_validation_retries + 1):
        reply = _generate_once(
            model,
            attempt_messages,
            tools=tools,
            tool_choice=tool_choice,
            call_name=call_name,
            **kwargs,
        )
        if response_model is None:
            return reply
        if reply.cost is not None:
            total_cost += reply.cost
            has_cost = True
        raw_responses.append(reply.content or "")
        try:
            response_model.model_validate_json(
                extract_json_from_llm_response(reply.content or "")
            )
            if has_cost:
                reply.cost = total_cost
            return reply
        except (ValidationError, ValueError) as exc:
            logger.warning(
                f"Invalid structured response for {call_name or model} "
                f"(attempt {attempt + 1}/{response_validation_retries + 1}): {exc}"
            )
            if attempt == response_validation_retries:
                raise LLMResponseValidationError(
                    response_model.__name__, raw_responses, exc
                ) from exc
            attempt_messages.extend(
                [
                    reply,
                    UserMessage(
                        role="user",
                        content=(
                            "Your previous reply was invalid JSON or did not match "
                            "the requested schema. Return only a corrected JSON "
                            "object. Escape quotation marks and newlines in strings."
                        ),
                    ),
                ]
            )
    raise AssertionError("response validation loop did not return or raise")

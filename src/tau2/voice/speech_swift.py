"""Strict loopback transport for the native speech-swift runtime."""

import audioop
import base64
import json
import threading
import time
import uuid
from collections.abc import Iterator
from urllib.parse import urlparse

import httpx

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.voice import TranscriptionResult
from tau2.voice.runtime_profile import RuntimeCapabilities


class SpeechSwiftError(RuntimeError):
    pass


def pcm16(audio: AudioData, sample_rate: int | None = None) -> AudioData:
    """Normalize the actual received audio; gold transcripts never enter ASR."""
    data = audio.data
    fmt = audio.format
    if fmt.encoding == AudioEncoding.ULAW:
        data = audioop.ulaw2lin(data, 2)
    elif fmt.encoding == AudioEncoding.ALAW:
        data = audioop.alaw2lin(data, 2)
    else:
        if fmt.encoding == AudioEncoding.PCM_U8:
            data = audioop.bias(data, 1, -128)
        if fmt.sample_width != 2:
            data = audioop.lin2lin(data, fmt.sample_width, 2)
    if fmt.channels == 2:
        data = audioop.tomono(data, 2, 0.5, 0.5)
    target_rate = sample_rate or fmt.sample_rate
    if target_rate != fmt.sample_rate:
        data, _ = audioop.ratecv(data, 2, 1, fmt.sample_rate, target_rate, None)
    return AudioData(
        data=data,
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=target_rate),
    )


class SpeechSwiftBackend:
    def __init__(self, base_url: str, *, transport=None, timeout: float = 300):
        url = urlparse(base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("speech-swift must use an HTTP loopback URL")
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self.events: list[dict] = []

    def capabilities(self) -> RuntimeCapabilities:
        reply = self.client.get("/v1/inference/capabilities")
        reply.raise_for_status()
        return RuntimeCapabilities.model_validate(reply.json())

    def provenance(self) -> dict:
        """Read the native source revision, asset pins, and actual parameter dtypes."""
        reply = self.client.get("/v1/inference/provenance")
        reply.raise_for_status()
        return reply.json()

    def cancel(self, request_id: str) -> None:
        response = self.client.post(
            "/v1/inference/cancel", json={"requestID": request_id}
        )
        response.raise_for_status()

    def close(self):
        with self._lock:
            active = list(self._active)
        for request_id in active:
            self.cancel(request_id)
        self.client.close()

    def _request(self, operation: str, model: str, *, request_id=None, **fields):
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        success = False
        failure = None
        metadata = {}
        with self._lock:
            self._active.add(request_id)
        request = None
        response = None
        try:
            request = self.client.build_request(
                "POST",
                "/v1/inference",
                json={
                    "operation": operation,
                    "model": model,
                    "requestID": request_id,
                    **fields,
                },
            )
            response = self.client.send(request)
            response.raise_for_status()
            result = response.json()
            if operation == "chat" and "cachedPrefixTokens" in result:
                cached_tokens = result["cachedPrefixTokens"]
                if type(cached_tokens) is not int or cached_tokens < 0:
                    raise SpeechSwiftError("Invalid native cached prefix token count")
                metadata["cached_prefix_tokens"] = cached_tokens
            success = True
            return result
        except Exception as error:
            if operation == "chat" and request is not None:
                failure = {
                    "request_body": request.content.decode("utf-8"),
                    "response_status": response.status_code
                    if response is not None
                    else None,
                    "response_body": response.text if response is not None else None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            raise
        finally:
            with self._lock:
                self._active.discard(request_id)
                self.events.append(
                    {
                        "operation": operation,
                        "model": model,
                        "request_id": request_id,
                        "wall_seconds": time.perf_counter() - started,
                        "success": success,
                        **metadata,
                        **({"failure": failure} if failure is not None else {}),
                    }
                )

    def generate(
        self,
        model,
        messages,
        *,
        tools=None,
        tool_choice=None,
        call_name=None,
        request_id=None,
        **kwargs,
    ):
        if model not in {"gemma4-e4b-it", "gemma4-12b-it"}:
            raise ValueError(f"Local chat model unavailable: {model}")
        if tool_choice not in {None, "auto", "none", "required"}:
            raise ValueError(f"Unsupported tool choice: {tool_choice}")
        history = []
        for message in messages:
            entry = {"role": message.role, "content": message.content}
            if isinstance(message, ToolMessage):
                entry["toolCallID"] = message.id
            calls = getattr(message, "tool_calls", None)
            if calls:
                entry["toolCalls"] = [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ]
                entry["role"] = "assistant"
            if (
                entry["role"] in {"user", "assistant"}
                and not entry.get("toolCalls")
                and history
                and history[-1]["role"] == entry["role"]
                and not history[-1].get("toolCalls")
            ):
                history[-1]["content"] = " ".join(
                    part.strip()
                    for part in (
                        history[-1].get("content") or "",
                        entry.get("content") or "",
                    )
                    if part.strip()
                )
            else:
                history.append(entry)
        declarations = (
            [tool.openai_schema["function"] for tool in tools or []]
            if tool_choice != "none"
            else []
        )
        if tool_choice == "required":
            history.insert(
                0,
                {
                    "role": "system",
                    "content": "Use one of the supplied tools for this response.",
                },
            )
        started = time.perf_counter()
        result = self._request(
            "chat",
            model,
            request_id=request_id,
            messages=history,
            tools=declarations,
            maxTokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0),
        )
        calls = result.get("toolCalls") or None
        # Task messages carry exactly one of text and calls.
        return AssistantMessage(
            role="assistant",
            content=None if calls else result["content"],
            tool_calls=calls,
            cost=0.0,
            usage={"completion_tokens": result["generatedTokens"]},
            raw_data={
                "runtime": "speech-swift",
                "call_name": call_name,
                **(
                    {"cached_prefix_tokens": result["cachedPrefixTokens"]}
                    if "cachedPrefixTokens" in result
                    else {}
                ),
            },
            generation_time_seconds=time.perf_counter() - started,
        )

    def synthesize(self, text, config):
        result = self._request(
            "speech",
            config.model_id,
            text=text,
            voiceInstruction=config.voice_id,
            maxTokens=config.model_args.get("max_tokens", 1500),
            seed=config.seed,
        )
        return AudioData(
            data=base64.b64decode(result["audioBase64"], validate=True),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=result["sampleRate"]
            ),
        )

    def stream_speech(
        self, text: str, *, request_id: str, voice: str, seed: int = 42
    ) -> Iterator[AudioData]:
        started = time.perf_counter()
        first = None
        done = False
        with self._lock:
            self._active.add(request_id)
        try:
            with self.client.stream(
                "POST",
                "/v1/inference",
                json={
                    "operation": "speech",
                    "model": "voxcpm2",
                    "requestID": request_id,
                    "text": text,
                    "voiceInstruction": voice,
                    "stream": True,
                    "seed": seed,
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event["event"] == "error":
                        raise SpeechSwiftError(event["message"])
                    if event["event"] == "done":
                        done = True
                        break
                    if event["event"] != "audio" or event.get("sampleRate") != 48000:
                        raise SpeechSwiftError("Invalid native speech stream")
                    if first is None:
                        first = time.perf_counter() - started
                    yield AudioData(
                        data=base64.b64decode(event["audioBase64"], validate=True),
                        format=AudioFormat(
                            encoding=AudioEncoding.PCM_S16LE, sample_rate=48000
                        ),
                    )
            if not done:
                raise SpeechSwiftError(
                    "Native speech stream ended without a completion event"
                )
        finally:
            with self._lock:
                self._active.discard(request_id)
                self.events.append(
                    {
                        "operation": "stream_speech",
                        "model": "voxcpm2",
                        "request_id": request_id,
                        "first_audio_seconds": first,
                        "wall_seconds": time.perf_counter() - started,
                        "success": done,
                    }
                )

    def transcribe(self, audio, config, *, request_id=None):
        normalized = pcm16(audio)
        result = self._request(
            "transcribe",
            config.model,
            request_id=request_id,
            audioBase64=base64.b64encode(normalized.data).decode(),
            sampleRate=normalized.format.sample_rate,
        )
        return TranscriptionResult(transcript=result["text"])

    def vad(
        self, pcm: bytes, stream_id: str, *, reset=False, end_stream=False
    ) -> list[float]:
        result = self._request(
            "vad",
            "silero-vad",
            audioBase64=base64.b64encode(pcm).decode(),
            sampleRate=16000,
            streamID=stream_id,
            reset=reset,
            endStream=end_stream,
        )
        return result["probabilities"]

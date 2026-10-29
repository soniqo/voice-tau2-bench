"""Shared event contract, bounded transport and usage for optional voice agents."""

import asyncio
import base64
import json
import math
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass

from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.voice.pricing import build_session_usage

PROVIDERS = {"openai-realtime", "openai-live", "gemini-live", "elevenlabs"}


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    voice: str
    key_env: str
    max_seconds: float = 240
    stop_usd: float = 0.25
    backend_model: str | None = None
    agent_id: str | None = None
    usd_per_credit: float | None = None
    max_usd_per_minute: float | None = None
    preset_sha256: str | None = None

    def __post_init__(self):
        if self.provider not in PROVIDERS or not self.model or not self.voice:
            raise ValueError(
                "A supported provider, explicit model and voice are required"
            )
        if not self.key_env or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.key_env
        ):
            raise ValueError("key_env must name an environment variable")
        if not math.isfinite(self.max_seconds) or not 1 <= self.max_seconds <= 420:
            raise ValueError("Calls must be 1–420 seconds")
        if not math.isfinite(self.stop_usd) or not 0 < self.stop_usd <= 5:
            raise ValueError("Spending threshold must be positive and at most 5")
        if self.provider == "openai-live" and not self.backend_model:
            raise ValueError("GPT-Live requires an explicitly named Responses backend")
        if self.provider == "elevenlabs":
            if not self.agent_id:
                raise ValueError("ElevenLabs requires a configured agent ID")
            if (
                not self.preset_sha256
                or len(self.preset_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.preset_sha256)
            ):
                raise ValueError("ElevenLabs requires its verified preset SHA256")
            for number in (self.usd_per_credit, self.max_usd_per_minute):
                if number is None or not math.isfinite(number) or number <= 0:
                    raise ValueError(
                        "ElevenLabs requires credit conversion and maximum per-minute rates"
                    )
            if self.max_seconds * self.max_usd_per_minute / 60 > self.stop_usd:
                raise ValueError(
                    "Declared maximum ElevenLabs call charge exceeds the threshold"
                )


@dataclass(frozen=True)
class ProviderEvent:
    kind: str
    id: str = ""
    text: str = ""
    audio: bytes = b""
    sample_rate: int = 24000
    calls: tuple[ToolCall, ...] = ()
    cancelled_ids: tuple[str, ...] = ()
    final: bool = True

    def public(self):
        return {
            "kind": self.kind,
            "id": self.id,
            "text": self.text,
            "audio_bytes": len(self.audio),
            "sample_rate": self.sample_rate,
            "calls": [c.model_dump(mode="json") for c in self.calls],
            "cancelled_ids": list(self.cancelled_ids),
            "final": self.final,
        }


def decoded_pcm(value):
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ProviderError("Invalid provider audio encoding") from error
    if len(data) % 2:
        raise ProviderError("Unaligned provider PCM16 audio")
    return data


def count(value, name):
    if type(value) is not int or value < 0:
        raise ProviderError(f"Invalid usage meter: {name}")
    return value


class JSONSocket:
    """Lazy optional aiohttp dependency; no URLs or authentication in artifacts."""

    @classmethod
    async def open(cls, url, headers):
        try:
            import aiohttp
        except ImportError as error:
            raise ProviderError("Install the cloud-voice dependency extra") from error
        instance = cls()
        instance.client = aiohttp.ClientSession()
        try:
            instance.ws = await instance.client.ws_connect(
                url,
                headers=headers,
                max_msg_size=2 * 1024 * 1024,
            )
        except BaseException:
            await instance.client.close()
            raise ProviderError("Provider WebSocket connection failed") from None
        instance.lock = asyncio.Lock()
        return instance

    async def send(self, value):
        async with self.lock:
            await self.ws.send_json(value)

    async def recv(self):
        import aiohttp

        while True:
            message = await self.ws.receive()
            if message.type in {aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY}:
                result = json.loads(message.data)
                if not isinstance(result, dict):
                    raise ProviderError("Provider event must be an object")
                return result
            if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                raise EOFError("Provider WebSocket ended")
            if message.type == aiohttp.WSMsgType.ERROR:
                raise ProviderError("Provider WebSocket failed")

    async def close(self):
        try:
            await self.ws.close()
        finally:
            await self.client.close()


class ProviderSession:
    input_rate = 24000

    def __init__(self, settings, *, connector=JSONSocket.open):
        self.settings, self.connector = settings, connector
        self.id = uuid.uuid4().hex
        self.socket = None
        self.events = deque()
        self.records = []
        self.played_ms = {}
        self.connected = self.closing = self.closed = False
        self.usage_finalized = False
        self._record_ids = set()
        self._key = None
        self._input_samples = 0
        self._input_peak = 0

    def credential(self):
        key = os.environ.get(self.settings.key_env)
        if not key:
            raise ProviderError(
                f"Missing credential environment variable: {self.settings.key_env}"
            )
        self._key = key
        return key

    @property
    def metered_usd(self):
        return build_session_usage(self.records).cost if self.records else None

    def usage(self, record):
        if record.scope_id and record.semantics == "delta":
            key = (record.provider, record.model, record.scope_id)
            if key in self._record_ids:
                raise ProviderError("Duplicate provider billing scope")
            self._record_ids.add(key)
        self.records.append(record)
        if self.metered_usd is not None and self.metered_usd >= self.settings.stop_usd:
            raise ProviderError("Provider spending threshold reached")

    def speech(self, ident, pcm, text, rate=24000, *, final=True):
        if not pcm or not text or not text.strip():
            raise ProviderError(
                "Completed speech requires audio and a final transcript"
            )
        if len(pcm) > rate * 2 * 90:
            raise ProviderError("Provider exceeded 90 seconds of buffered speech")
        self.events.append(
            ProviderEvent("speech", ident, text, bytes(pcm), rate, final=final)
        )

    async def receive(self):
        while not self.events:
            message = await self.socket.recv()
            await self.parse(message)
        return self.events.popleft()

    async def playback(self, ident, milliseconds):
        self.played_ms[ident] = milliseconds

    async def audio(self, pcm):
        self._input_samples += len(pcm) // 2
        if self._input_samples > (self.settings.max_seconds + 10) * self.input_rate:
            raise ProviderError("Provider input audio limit reached")
        await self.send_audio(pcm)

    async def end_input(self):
        self.closing = True

    async def close(self):
        if self.closed:
            return
        if self.socket:
            await self.socket.close()
        self.closed = True

    def usage_records(self):
        records = list(self.records)
        if not self.usage_finalized or not records:
            records.append(
                UsageRecord(
                    provider="unconfirmed-usage",
                    model=self.settings.model,
                    raw={"final_usage_confirmed": False},
                )
            )
        return records


def function_tools(tools):
    return [{"type": "function", **t.openai_schema["function"]} for t in tools]


def agent_instructions(policy):
    return (
        "You are the customer service agent in an English voice call. Speak briefly. "
        "Follow the domain policy and use the declared tools. Ask the caller to "
        "perform actions on their own phone. Confirm actions only after tool results "
        "establish success. A common opening greeting is delivered by the application; "
        "wait for the caller's request.\n\n" + policy
    )

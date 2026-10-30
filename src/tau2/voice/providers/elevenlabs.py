"""Configured ElevenLabs Agents with verified client tools and completion events."""

import asyncio
import base64
import hashlib
import json
import math
import re
from urllib.parse import parse_qs, quote, urlparse

import httpx

from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.voice.providers.base import (
    ProviderError,
    ProviderEvent,
    ProviderSession,
    count,
    decoded_pcm,
)

REQUIRED_EVENTS = {
    "audio",
    "agent_response",
    "agent_response_complete",
    "user_transcript",
    "interruption",
    "client_tool_call",
}


def preset_fingerprint(preset):
    relevant = {
        name: preset.get(name)
        for name in ("conversation_config", "platform_settings", "workflow")
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def eleven_config(system, tools, model, voice, max_seconds):
    return {
        "agent": {
            "first_message": "",
            "language": "en",
            "prompt": {
                "prompt": system,
                "llm": model,
                "tools": [
                    {
                        "type": "client",
                        "name": t.name,
                        "description": t.openai_schema["function"]["description"],
                        "parameters": t.openai_schema["function"]["parameters"],
                        "expects_response": True,
                    }
                    for t in tools
                ],
            },
        },
        "tts": {"voice_id": voice, "agent_output_audio_format": "pcm_24000"},
        "asr": {"user_input_audio_format": "pcm_16000"},
        "conversation": {
            "max_duration_seconds": max_seconds,
            "client_events": sorted(REQUIRED_EVENTS),
        },
    }


def verify_preset(config, expected):
    agent = config.get("agent", {})
    prompt = agent.get("prompt", {})
    desired = expected["agent"]["prompt"]
    if prompt.get("prompt") != desired["prompt"] or prompt.get("llm") != desired["llm"]:
        raise ProviderError(
            "ElevenLabs preset policy or reasoning model differs from the declared condition"
        )
    if agent.get("first_message") or agent.get("language") != "en":
        raise ProviderError(
            "ElevenLabs preset must disable its greeting and use English"
        )
    if (
        prompt.get("knowledge_base")
        or prompt.get("built_in_tools")
        or prompt.get("custom_llm")
    ):
        raise ProviderError(
            "ElevenLabs preset contains undeclared backend capabilities"
        )
    if config.get("language_presets"):
        raise ProviderError(
            "ElevenLabs language overrides are outside this English condition"
        )
    actual = prompt.get("tools", [])
    if len(actual) != len(desired["tools"]):
        raise ProviderError(
            "ElevenLabs preset must expose exactly the domain client tools"
        )
    by_name = {}
    for item in actual:
        if item.get("type") != "client" or item.get("expects_response") is not True:
            raise ProviderError(
                "Only response-bearing ElevenLabs client tools are supported"
            )
        if item.get("name") in by_name:
            raise ProviderError("Duplicate ElevenLabs preset tool")
        by_name[item.get("name")] = item
    for tool in desired["tools"]:
        actual = by_name.get(tool["name"], {})
        if (
            actual.get("parameters") != tool["parameters"]
            or actual.get("description") != tool["description"]
        ):
            raise ProviderError(
                "ElevenLabs client tool schema differs from the benchmark"
            )
    conversation = config.get("conversation", {})
    if not REQUIRED_EVENTS <= set(conversation.get("client_events", [])):
        raise ProviderError(
            "Enable the required ElevenLabs completion/audio/tool events"
        )
    if (
        not 1
        <= conversation.get("max_duration_seconds", 0)
        <= expected["conversation"]["max_duration_seconds"]
    ):
        raise ProviderError("ElevenLabs preset exceeds the declared duration")
    if config.get("tts", {}).get("voice_id") != expected["tts"]["voice_id"]:
        raise ProviderError("ElevenLabs voice differs from the declared condition")
    if (
        config.get("tts", {}).get("agent_output_audio_format") != "pcm_24000"
        or config.get("asr", {}).get("user_input_audio_format") != "pcm_16000"
    ):
        raise ProviderError("ElevenLabs preset must use PCM16k input and PCM24k output")


class ElevenSession(ProviderSession):
    input_rate = 16000

    def __init__(self, *args, http_factory=httpx.AsyncClient, **kwargs):
        super().__init__(*args, **kwargs)
        self.http_factory = http_factory
        self.http = None
        self.pcm, self.text = bytearray(), []
        self.rate = 24000
        self.pending = set()
        self.credits = None
        self.configuration_sha256 = None
        self._last_response_id = None
        self._last_interrupt_id = -1

    async def get(self, path):
        response = await self.http.get("https://api.elevenlabs.io/v1/convai/" + path)
        response.raise_for_status()
        return response.json()

    async def connect(self, system, tools):
        key = self.credential()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", self.settings.agent_id):
            raise ValueError("Invalid ElevenLabs agent ID")
        self.http = self.http_factory(
            headers={"xi-api-key": key}, timeout=15, trust_env=False
        )
        preset = await self.get("agents/" + quote(self.settings.agent_id))
        if preset.get("workflow"):
            raise ProviderError(
                "ElevenLabs multi-agent workflows are not supported by this adapter"
            )
        config = preset["conversation_config"]
        prompt = config.get("agent", {}).get("prompt", {})
        # Tool references are resolved read-only; no remote agent is created or edited.
        ids = prompt.get("tool_ids", [])
        if ids:
            if prompt.get("tools"):
                raise ProviderError("Ambiguous ElevenLabs tool definitions")
            resolved = []
            for ident in ids:
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", ident):
                    raise ProviderError("Invalid ElevenLabs tool reference")
                item = await self.get("tools/" + ident)
                resolved.append(item.get("tool_config", item))
            prompt["tools"] = resolved
        expected = eleven_config(
            system,
            tools,
            self.settings.model,
            self.settings.voice,
            self.settings.max_seconds,
        )
        verify_preset(config, expected)
        self.configuration_sha256 = preset_fingerprint(preset)
        if self.configuration_sha256 != self.settings.preset_sha256:
            raise ProviderError("ElevenLabs preset fingerprint changed")
        result = await self.get(
            "conversation/get-signed-url?agent_id=" + quote(self.settings.agent_id)
        )
        url = result["signed_url"]
        parsed = urlparse(url)
        if (
            parsed.scheme != "wss"
            or parsed.hostname != "api.elevenlabs.io"
            or parsed.username
            or parsed.password
            or parsed.path != "/v1/convai/conversation"
            or parse_qs(parsed.query).get("agent_id") != [self.settings.agent_id]
        ):
            raise ProviderError("Unexpected ElevenLabs signed WebSocket endpoint")
        self.socket = await self.connector(url, {})
        await self.socket.send({"type": "conversation_initiation_client_data"})
        async with asyncio.timeout(15):
            while True:
                message = await self.socket.recv()
                if message.get("type") == "conversation_initiation_metadata":
                    info = message["conversation_initiation_metadata_event"]
                    if (
                        info["user_input_audio_format"] != "pcm_16000"
                        or info["agent_output_audio_format"] != "pcm_24000"
                    ):
                        raise ProviderError(
                            "ElevenLabs negotiated unexpected audio formats"
                        )
                    self.id = info["conversation_id"]
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", self.id):
                        raise ProviderError("Invalid ElevenLabs conversation ID")
                    break
                await self.parse(message)
        self.connected = True

    async def send_audio(self, pcm):
        await self.socket.send({"user_audio_chunk": base64.b64encode(pcm).decode()})

    async def assistant_history(self, text):
        await self.socket.send(
            {
                "type": "contextual_update",
                "text": "The assistant already delivered this greeting: " + text,
            }
        )

    async def tool_result(self, call_id, result):
        if call_id not in self.pending:
            raise ProviderError("ElevenLabs result has no pending call")
        self.pending.remove(call_id)
        await self.socket.send(
            {
                "type": "client_tool_result",
                "tool_call_id": call_id,
                "result": json.dumps(result),
                "is_error": bool(result.get("error")),
            }
        )

    async def parse(self, message):
        kind = message.get("type")
        if kind == "ping":
            await self.socket.send(
                {"type": "pong", "event_id": message["ping_event"]["event_id"]}
            )
        elif kind in {"client_error", "error", "guardrail_triggered"}:
            raise ProviderError("ElevenLabs returned an inference/error event")
        elif kind == "queue_status":
            if message["queue_status_event"]["status"] != "admitted":
                raise ProviderError(
                    "ElevenLabs agent unavailable; hold audio cannot be scored"
                )
        elif kind == "interruption":
            self._last_interrupt_id = count(
                message["interruption_event"]["event_id"], "interruption event ID"
            )
            self.pcm.clear()
            self.text.clear()
            self.events.append(ProviderEvent("interrupt"))
        elif kind == "audio":
            item = message["audio_event"]
            if count(item["event_id"], "audio event ID") <= self._last_interrupt_id:
                return
            self.pcm.extend(decoded_pcm(item["audio_base_64"]))
            if len(self.pcm) > 24000 * 2 * 90:
                raise ProviderError("ElevenLabs buffered speech exceeded its bound")
        elif kind == "agent_response":
            item = message["agent_response_event"]
            ident = item.get("event_id")
            if ident is not None and ident == self._last_response_id:
                raise ProviderError("Duplicate ElevenLabs response")
            self._last_response_id = ident
            self.text.append(item["agent_response"])
        elif kind == "agent_response_complete":
            if self.pending:
                raise ProviderError("ElevenLabs completed with unresolved tools")
            if self.pcm or self.text:
                self.speech(
                    str(message["agent_response_complete_event"]["event_id"]),
                    self.pcm,
                    " ".join(self.text),
                )
            self.pcm.clear()
            self.text.clear()
        elif kind == "user_transcript":
            self.events.append(
                ProviderEvent(
                    "transcript",
                    text=message["user_transcription_event"]["user_transcript"],
                )
            )
        elif kind == "client_tool_call":
            item = message["client_tool_call"]
            # Some documented events omit this flag; the verified preset already
            # requires response-bearing client tools. Reject explicit conflicts.
            if (
                item.get("expects_response", True) is not True
                or item["tool_call_id"] in self.pending
            ):
                raise ProviderError("Invalid ElevenLabs client tool call")
            self.pending.add(item["tool_call_id"])
            self.events.append(
                ProviderEvent(
                    "tools",
                    calls=(
                        ToolCall(
                            id=item["tool_call_id"],
                            name=item["tool_name"],
                            arguments=item["parameters"],
                        ),
                    ),
                )
            )
        elif kind == "agent_tool_response":
            if message.get("agent_tool_response", {}).get("tool_type") not in {
                None,
                "client",
            }:
                raise ProviderError("ElevenLabs executed an undeclared server tool")

    @property
    def metered_usd(self):
        # Conversation credits exclude some charges (including Voice Library
        # surcharges). A plan conversion is not a final dollar invoice.
        return None

    async def end_input(self):
        self.closing = True
        if self.pending:
            raise ProviderError("ElevenLabs shutdown has unresolved tool calls")
        await self.socket.close()
        async with asyncio.timeout(15):
            while True:
                details = await self.get("conversations/" + quote(self.id))
                if details.get("status") == "done":
                    metadata = details.get("metadata", {})
                    credit = metadata.get("cost")
                    if (
                        type(credit) not in {int, float}
                        or not math.isfinite(credit)
                        or credit < 0
                    ):
                        raise ProviderError("ElevenLabs final credit meter is invalid")
                    self.credits = credit
                    duration = metadata.get("call_duration_secs")
                    if (
                        type(duration) not in {int, float}
                        or not math.isfinite(duration)
                        or duration < 0
                    ):
                        raise ProviderError("ElevenLabs final duration is invalid")
                    self.records.append(
                        UsageRecord(
                            provider="elevenlabs-agent",
                            model=self.settings.model,
                            audio_input_seconds=duration,
                            raw={
                                "credits": credit,
                                "charging": metadata.get("charging"),
                                "usd_per_credit": self.settings.usd_per_credit,
                                "credit_subtotal_usd": credit
                                * self.settings.usd_per_credit,
                                "final_usd_confirmed": False,
                            },
                        )
                    )
                    self.usage_finalized = not self.pending
                    break
                await asyncio.sleep(0.25)

    async def close(self):
        try:
            await super().close()
        finally:
            if self.http:
                await self.http.aclose()

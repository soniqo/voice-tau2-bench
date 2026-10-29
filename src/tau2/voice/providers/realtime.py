"""OpenAI Realtime GA audio, function calls and per-response usage."""

import asyncio
import base64
import json
from urllib.parse import urlencode

from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.voice.providers.base import (
    ProviderError,
    ProviderEvent,
    ProviderSession,
    count,
    decoded_pcm,
    function_tools,
)


def realtime_usage(usage, model, scope):
    for key in ("input_tokens", "output_tokens"):
        count(usage.get(key), key)
    record = UsageRecord.from_openai_realtime_usage(
        usage,
        provider="openai",
        model=model,
        scope_id=scope,
    )
    for side in ("input", "output"):
        total = getattr(record, side + "_tokens")
        text = getattr(record, side + "_text_tokens")
        audio = getattr(record, side + "_audio_tokens")
        if text is None or audio is None:
            raise ProviderError("Realtime usage lacks its modality split")
        count(text, side + " text")
        count(audio, side + " audio")
        if text + audio != total:
            raise ProviderError("Unsupported or inconsistent Realtime usage modality")
    for modality in ("text", "audio"):
        cached = getattr(record, "input_cached_" + modality + "_tokens")
        if cached is not None:
            count(cached, "cached " + modality)
        if cached is not None and not 0 <= cached <= getattr(
            record, "input_" + modality + "_tokens"
        ):
            raise ProviderError("Invalid cached Realtime usage")
    return record


class RealtimeSession(ProviderSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_parts = {}
        self.transcripts = {}
        self.active = set()
        self.cancelled = set()
        self.pending = set()
        self.output_items = {}
        self._clear_ack = asyncio.Event()
        self.awaiting_response = False

    async def connect(self, system, tools):
        key = self.credential()
        self.socket = await self.connector(
            "wss://api.openai.com/v1/realtime?"
            + urlencode({"model": self.settings.model}),
            {"Authorization": "Bearer " + key},
        )
        await self.socket.send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.settings.model,
                    "instructions": system,
                    "output_modalities": ["audio"],
                    "max_output_tokens": 1024,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {"type": "server_vad"},
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": self.settings.voice,
                        },
                    },
                    "tools": function_tools(tools),
                    "tool_choice": "auto",
                },
            }
        )
        async with asyncio.timeout(15):
            while True:
                event = await self.socket.recv()
                if event.get("type") == "session.updated":
                    model = event.get("session", {}).get("model")
                    if model != self.settings.model:
                        raise ProviderError("Realtime acknowledged a different model")
                    self.id = event["session"]["id"]
                    break
                await self.parse(event)
        self.connected = True

    async def send_audio(self, pcm):
        await self.socket.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )

    async def assistant_history(self, text):
        await self.socket.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            }
        )

    async def tool_result(self, call_id, result):
        if call_id not in self.pending:
            raise ProviderError("Realtime tool result has no matching pending call")
        self.pending.remove(call_id)
        await self.socket.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        if not self.pending:
            self.awaiting_response = True
            await self.socket.send({"type": "response.create"})

    async def parse(self, event):
        kind = event.get("type")
        if kind == "error":
            raise ProviderError("Realtime returned an error event")
        if kind == "response.created":
            self.awaiting_response = False
            self.active.add(event["response"]["id"])
        elif kind == "input_audio_buffer.speech_started":
            for ident in self.active:
                self.cancelled.add(ident)
            for key, pcm in self.audio_parts.items():
                self.output_items.setdefault(
                    f"{key[1]}:{key[2]}", (*key, len(pcm) / 48)
                )
            for segment_id, (response_id, item_id, content_index, duration_ms) in list(
                self.output_items.items()
            ):
                played = self.played_ms.get(segment_id, 0)
                if played >= duration_ms:
                    continue
                self.cancelled.add(response_id)
                await self.socket.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": item_id,
                        "content_index": content_index,
                        "audio_end_ms": int(played),
                    }
                )
                self.output_items.pop(segment_id)
            for key in list(self.audio_parts):
                if key[0] in self.cancelled:
                    self.audio_parts.pop(key)
            self.events.append(ProviderEvent("interrupt"))
        elif kind == "response.output_audio.delta":
            if event["response_id"] in self.cancelled:
                return
            key = (event["response_id"], event["item_id"], event["content_index"])
            pcm = self.audio_parts.setdefault(key, bytearray())
            pcm.extend(decoded_pcm(event["delta"]))
            if sum(map(len, self.audio_parts.values())) > 24000 * 2 * 90:
                raise ProviderError("Realtime buffered speech exceeded its bound")
        elif kind == "response.output_audio_transcript.done":
            key = (event["response_id"], event["item_id"], event["content_index"])
            self.transcripts[key] = event["transcript"]
        elif kind == "response.done":
            self.awaiting_response = False
            response = event["response"]
            ident = response["id"]
            self.active.discard(ident)
            if response.get("usage") is None:
                raise ProviderError("Realtime response lacks usage")
            self.usage(realtime_usage(response["usage"], self.settings.model, ident))
            if response["status"] == "cancelled" or ident in self.cancelled:
                self._discard(ident)
                return
            if response["status"] != "completed":
                raise ProviderError("Realtime inference did not complete")
            calls, spoken = [], []
            for item in response.get("output", []):
                if item["type"] == "function_call":
                    call = ToolCall(
                        id=item["call_id"],
                        name=item["name"],
                        arguments=json.loads(item["arguments"]),
                    )
                    if call.id in self.pending:
                        raise ProviderError("Duplicate Realtime function call")
                    self.pending.add(call.id)
                    calls.append(call)
                elif item["type"] == "message":
                    for index, part in enumerate(item.get("content", [])):
                        key = (ident, item["id"], index)
                        if part.get("type") not in {"audio", "output_audio"}:
                            continue
                        pcm = self.audio_parts.pop(key, b"")
                        text = self.transcripts.pop(key, None) or part.get("transcript")
                        segment_id = f"{item['id']}:{index}"
                        self.output_items[segment_id] = (*key, len(pcm) / 48)
                        spoken.append((segment_id, pcm, text))
            for index, (segment_id, pcm, text) in enumerate(spoken):
                self.speech(
                    segment_id, pcm, text, final=index == len(spoken) - 1 and not calls
                )
            self._discard(ident)
            if calls:
                self.events.append(ProviderEvent("tools", calls=tuple(calls)))
        elif kind == "input_audio_buffer.cleared":
            self._clear_ack.set()

    def _discard(self, ident):
        for collection in (self.audio_parts, self.transcripts):
            for key in list(collection):
                if key[0] == ident:
                    collection.pop(key)

    async def end_input(self):
        self.closing = True
        await self.socket.send({"type": "input_audio_buffer.clear"})
        async with asyncio.timeout(10):
            await self._clear_ack.wait()
            while self.active or self.awaiting_response:
                await asyncio.sleep(0.01)
        if self.pending:
            raise ProviderError("Realtime shutdown has unresolved tool calls")
        self.usage_finalized = bool(self.records) and not self.pending

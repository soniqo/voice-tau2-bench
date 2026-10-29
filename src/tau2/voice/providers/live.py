"""GPT-Live Responses delegation and timestamped reflected audio.

Live has no output-turn-complete event. The declared noise gate establishes
speech boundaries, not proof that a business answer is complete.
"""

import asyncio
import audioop
import base64
import json
import math
import time
from urllib.parse import quote

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


class TimedSpeech:
    silence_ms = 800
    transcript_grace_ms = 500
    rms_threshold = 180

    def __init__(self, emit):
        self.emit = emit
        self.audio = []
        self.captions = []
        self.preroll = []
        self.last_voice = self.latest = self.last_emitted = 0
        self.received_at = None
        self.sequence = 0

    def add_audio(self, event):
        start, end = event["start_ms"], event["end_ms"]
        pcm = decoded_pcm(event["delta"])
        if (
            start < self.latest
            or end <= start
            or abs(len(pcm) - (end - start) * 48) > 48
        ):
            raise ProviderError("Invalid or overlapping Live reflected audio timing")
        self.latest, self.received_at = end, time.monotonic()
        chunk = (start, end, pcm)
        voiced = audioop.rms(pcm, 2) >= self.rms_threshold
        if voiced:
            if not self.audio:
                self.audio.extend(self.preroll)
            self.last_voice = end
        if self.audio or voiced or self.captions:
            self.audio.append(chunk)
        self.preroll.append(chunk)
        self.preroll = [x for x in self.preroll if x[1] > end - 200]
        if sum(len(x[2]) for x in self.audio) > 24000 * 2 * 90:
            raise ProviderError("Live buffered speech exceeded its bound")
        self.flush()

    def add_caption(self, event):
        start, end, text = event["start_ms"], event["end_ms"], event["delta"]
        if end < start or start < 0:
            raise ProviderError("Invalid Live transcript timing")
        if start == end:
            return  # Context append acknowledgments are not audible speech.
        if start < self.last_emitted:
            raise ProviderError("Live transcript arrived after its audio was released")
        self.captions.append((start, end, text))
        if sum(len(x[2]) for x in self.captions) > 16000:
            raise ProviderError("Live transcript buffer exceeded its bound")
        self.flush()

    def flush(self, now_ms=None):
        if not self.audio or self.received_at is None:
            return
        now = (
            self.latest + (time.monotonic() - self.received_at) * 1000
            if now_ms is None
            else now_ms
        )
        speech_end = max(self.last_voice, max((x[1] for x in self.captions), default=0))
        if now < speech_end + self.silence_ms + self.transcript_grace_ms:
            return
        if not self.captions:
            raise ProviderError("Live speech ended without its transcript")
        # A reflected-frame gap within a text interval must not become fabricated audio.
        for start, end, _ in self.captions:
            cursor = start
            for a, b, _pcm in self.audio:
                if b <= cursor:
                    continue
                if a > cursor + 1:
                    break
                cursor = max(cursor, b)
                if cursor >= end:
                    break
            if cursor + 1 < end:
                raise ProviderError("Live transcript covers missing reflected audio")
        first = self.audio[0][0]
        last = max(speech_end, self.audio[-1][1])
        if last - first > 90000:
            raise ProviderError("Live speech segment exceeded its duration bound")
        pcm = bytearray(int(last - first) * 48)
        for start, end, data in self.audio:
            offset = int(start - first) * 48
            pcm[offset : offset + len(data)] = data
        self.emit(
            f"live-speech-{self.sequence}", pcm, "".join(x[2] for x in self.captions)
        )
        self.sequence += 1
        self.last_emitted = last
        self.audio.clear()
        self.captions.clear()
        self.preroll.clear()
        self.last_voice = last


class LiveSession(ProviderSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary = None
        self.primary_reader = None
        self.finished = asyncio.Event()
        self.active = set()
        self.pending = set()
        self.awaiting_response = False
        self.seconds = 0
        self.speech_buffer = TimedSpeech(self.speech)
        self._voice_record = None

    async def connect(self, system, tools):
        key = self.credential()
        headers = {"Authorization": "Bearer " + key}
        self.primary = await self.connector(
            "wss://api.openai.com/v1/live/sessions", headers
        )
        declared = [{**tool, "strict": False} for tool in function_tools(tools)]
        await self.primary.send(
            {
                "type": "session.start",
                "session": {
                    "model": self.settings.model,
                    "instructions": "Handle the English spoken conversation briefly. The application delivers the opening greeting. Wait for the caller. Delegate every business request to the backend and report only its verified results.",
                    "audio": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "output": {"voice": self.settings.voice},
                    },
                    "delegation": {
                        "type": "responses",
                        "responses": {
                            "model": self.settings.backend_model,
                            "instructions": system,
                            "tools": declared,
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                        },
                    },
                },
            }
        )
        async with asyncio.timeout(15):
            while True:
                event = await self.primary.recv()
                if event.get("type") == "session.started":
                    session = event["session"]
                    if session["model"] != self.settings.model:
                        raise ProviderError("Live acknowledged a different voice model")
                    self.id = session["id"]
                    break
                if event.get("type") == "error":
                    raise ProviderError("Live rejected session configuration")
        if not self.id or not all(c.isalnum() or c in "_-" for c in self.id):
            raise ProviderError("Invalid Live session identifier")
        self.socket = await self.connector(
            "wss://api.openai.com/v1/live/sessions/" + quote(self.id) + "/attach",
            headers,
        )
        self.primary_reader = asyncio.create_task(self._drain_primary())
        self.connected = True

    async def _drain_primary(self):
        while True:
            try:
                event = await self.primary.recv()
            except EOFError:
                if self.closing:
                    return
                raise ProviderError(
                    "Live primary connection ended unexpectedly"
                ) from None
            if event.get("type") == "error":
                raise ProviderError("Live primary connection returned an error")

    async def receive(self):
        while not self.events:
            receiving = asyncio.create_task(self.socket.recv())
            try:
                watching = {receiving}
                if self.primary_reader and not self.primary_reader.done():
                    watching.add(self.primary_reader)
                if self.primary_reader and self.primary_reader.done():
                    self.primary_reader.result()
                await asyncio.wait(watching, return_when=asyncio.FIRST_COMPLETED)
                if self.primary_reader and self.primary_reader.done():
                    self.primary_reader.result()
                event = await receiving
                await self.parse(event)
            finally:
                if not receiving.done():
                    receiving.cancel()
                    await asyncio.gather(receiving, return_exceptions=True)
        return self.events.popleft()

    async def send_audio(self, pcm):
        await self.primary.send(
            {
                "type": "session.input_audio.append",
                "audio": base64.b64encode(pcm).decode(),
            }
        )
        self.speech_buffer.flush()

    async def assistant_history(self, text):
        await self.primary.send(
            {
                "type": "session.thinking.append",
                "event_id": "delivered-opener",
                "delegation_id": None,
                "content": "The assistant already delivered this greeting: " + text,
            }
        )

    async def tool_result(self, call_id, result):
        if call_id not in self.pending:
            raise ProviderError("Live backend result has no pending call")
        self.pending.remove(call_id)
        await self.primary.send(
            {
                "type": "response.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        if not self.pending:
            self.awaiting_response = True
            await self.primary.send({"type": "response.create"})

    def voice_usage(self, seconds):
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or seconds < self.seconds
        ):
            raise ProviderError("Invalid or decreasing Live duration meter")
        self.seconds = seconds
        self._voice_record = UsageRecord(
            provider="openai",
            model=self.settings.model,
            audio_input_seconds=seconds,
            semantics="cumulative",
            scope_id=self.id,
            raw={"seconds": seconds},
        )
        self.records = [r for r in self.records if r.scope_id != self.id]
        self.usage(self._voice_record)

    async def parse(self, event):
        kind = event.get("type")
        if kind == "error":
            raise ProviderError("Live returned an error event")
        if kind == "session.output_audio.delta":
            self.speech_buffer.add_audio(event)
        elif kind == "session.output_transcript.delta":
            self.speech_buffer.add_caption(event)
        elif kind == "session.input_transcript.delta":
            self.events.append(ProviderEvent("transcript", text=event["delta"]))
        elif kind == "session.usage.updated":
            self.voice_usage(event["usage"]["seconds"])
        elif kind == "response.event":
            item = event["event"]
            if item["type"] == "response.created":
                self.awaiting_response = False
                self.active.add(item["response"]["id"])
            elif item["type"] == "response.output_item.done":
                output = item["item"]
                if output["type"] == "function_call":
                    if not event.get("delegation_id"):
                        raise ProviderError("Live function call lacks a delegation ID")
                    call = ToolCall(
                        id=output["call_id"],
                        name=output["name"],
                        arguments=json.loads(output["arguments"]),
                    )
                    if call.id in self.pending:
                        raise ProviderError("Duplicate Live backend call")
                    self.pending.add(call.id)
                    self.events.append(ProviderEvent("tools", calls=(call,)))
            elif item["type"] in {
                "response.completed",
                "response.failed",
                "response.incomplete",
                "response.cancelled",
            }:
                response = item["response"]
                self.awaiting_response = False
                self.active.discard(response["id"])
                usage = response.get("usage")
                if usage is None:
                    raise ProviderError("Live backend response lacks usage")
                for name in ("input_tokens", "output_tokens"):
                    count(usage.get(name), name)
                cached = usage.get("input_tokens_details", {}).get("cached_tokens")
                if cached is not None:
                    count(cached, "cached tokens")
                    if cached > usage["input_tokens"]:
                        raise ProviderError("Live cached tokens exceed input usage")
                if (
                    response.get("model", self.settings.backend_model)
                    != self.settings.backend_model
                ):
                    raise ProviderError(
                        "Live delegated inference used a different backend"
                    )
                self.usage(
                    UsageRecord(
                        provider="openai",
                        model=self.settings.backend_model,
                        component="llm",
                        scope_id=response["id"],
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        input_cached_tokens=cached,
                        raw=usage,
                    )
                )
                if item["type"] != "response.completed":
                    raise ProviderError("Live delegated inference did not complete")
        elif kind == "session.closed":
            self.voice_usage(event["usage"]["seconds"])
            if self.active or self.pending or self.awaiting_response:
                raise ProviderError("Live closed with unresolved delegated inference")
            self.usage_finalized = True
            self.finished.set()
            self.events.append(ProviderEvent("closed"))
            if not self.closing or event["reason"] != "close_requested":
                raise ProviderError("Live session ended before requested closure")

    async def end_input(self):
        self.closing = True
        async with asyncio.timeout(15):
            while self.active or self.pending or self.awaiting_response:
                await asyncio.sleep(0.01)
            await self.primary.send({"type": "session.close"})
            await self.finished.wait()

    async def close(self):
        try:
            await super().close()
        finally:
            if self.primary:
                await self.primary.close()
            if self.primary_reader:
                self.primary_reader.cancel()
                await asyncio.gather(self.primary_reader, return_exceptions=True)

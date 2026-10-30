"""Gemini Live JSON protocol with independently arriving audio and transcripts."""

import asyncio
import base64
from urllib.parse import urlencode

from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from tau2.voice.providers.base import (
    ProviderError,
    ProviderEvent,
    ProviderSession,
    count,
    decoded_pcm,
)


class GeminiSession(ProviderSession):
    input_rate = 16000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pcm, self.text = bytearray(), ""
        self.active = False
        self.turn = 0
        self.pending = {}
        self.cancelled = set()
        self.dropping = False
        self.partial_usage = False

    @property
    def metered_usd(self):
        return None if self.partial_usage else super().metered_usd

    async def connect(self, system, tools):
        key = self.credential()
        self.system = system
        model = "models/" + self.settings.model
        url = (
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?"
            + urlencode({"key": key})
        )
        headers = {}
        if self.settings.vertex_project:
            location = self.settings.vertex_location
            host = (
                "aiplatform.googleapis.com"
                if location == "global"
                else location + "-aiplatform.googleapis.com"
            )
            url = f"wss://{host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
            headers = {"Authorization": "Bearer " + key}
            model = f"projects/{self.settings.vertex_project}/locations/{location}/publishers/google/models/{self.settings.model}"
        self.socket = await self.connector(url, headers)
        await self.socket.send(
            {
                "setup": {
                    "model": model,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "maxOutputTokens": 1024,
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": self.settings.voice
                                }
                            }
                        },
                    },
                    "systemInstruction": {"parts": [{"text": system}]},
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "name": t.name,
                                    "description": t.openai_schema["function"][
                                        "description"
                                    ],
                                    "parametersJsonSchema": t.openai_schema["function"][
                                        "parameters"
                                    ],
                                }
                                for t in tools
                            ]
                        }
                    ],
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                }
            }
        )
        async with asyncio.timeout(15):
            while True:
                message = await self.socket.recv()
                if "setupComplete" in message:
                    break
                await self.parse(message)
        self.connected = True

    async def send_audio(self, pcm):
        await self.socket.send(
            {
                "realtimeInput": {
                    "audio": {
                        "mimeType": "audio/pcm;rate=16000",
                        "data": base64.b64encode(pcm).decode(),
                    }
                }
            }
        )

    async def assistant_history(self, text):
        if self.settings.vertex_project and self.settings.model.startswith(
            "gemini-3.8-live"
        ):
            await self.socket.send(
                {
                    "contextUpdate": {
                        "systemInstruction": {
                            "parts": [
                                {
                                    "text": self.system
                                    + "\n\nThe assistant already delivered this greeting: "
                                    + text
                                }
                            ]
                        }
                    }
                }
            )
            return
        await self.socket.send(
            {
                "clientContent": {
                    "turns": [{"role": "model", "parts": [{"text": text}]}],
                    "turnComplete": False,
                }
            }
        )

    async def tool_result(self, call_id, result):
        if call_id in self.cancelled:
            return  # Already executed tools remain in the benchmark's atomic trace.
        if call_id not in self.pending:
            raise ProviderError("Gemini result has no matching call")
        name = self.pending.pop(call_id)
        await self.socket.send(
            {
                "toolResponse": {
                    "functionResponses": [
                        {"id": call_id, "name": name, "response": result},
                    ]
                }
            }
        )

    async def parse(self, message):
        if "error" in message:
            raise ProviderError("Gemini returned an error event")
        if "goAway" in message:
            raise ProviderError(
                "Gemini requested a reconnect; continuation is not automatic"
            )
        if "usageMetadata" in message:
            raw = message["usageMetadata"]
            response_key = (
                "responseTokenCount"
                if "responseTokenCount" in raw
                else "candidatesTokenCount"
            )
            response_details = (
                "responseTokensDetails"
                if "responseTokensDetails" in raw
                else "candidatesTokensDetails"
            )
            for key in ("promptTokenCount", response_key):
                if raw.get(key) is not None:
                    count(raw[key], key)

            def details(name):
                return [
                    {
                        "modality": d["modality"],
                        "token_count": count(d["tokenCount"], name),
                    }
                    for d in (raw.get(name) or [])
                ]

            for name in ("thoughtsTokenCount", "cachedContentTokenCount"):
                if raw.get(name) is not None:
                    count(raw[name], name)
            # Tool-only responses can omit counters/modality fields. Preserve
            # the observation and unknown billing without failing inference.
            if (
                any(raw.get(key) is None for key in ("promptTokenCount", response_key))
                or not raw.get("promptTokensDetails")
                or not raw.get(response_details)
            ):
                self.partial_usage = True
            # These are response observations, not one last-value session bill.
            self.usage(
                UsageRecord.from_gemini_usage_metadata(
                    model=self.settings.model,
                    prompt_token_count=raw.get("promptTokenCount"),
                    response_token_count=raw.get(response_key),
                    cached_content_token_count=raw.get("cachedContentTokenCount"),
                    thoughts_token_count=raw.get("thoughtsTokenCount"),
                    prompt_tokens_details=details("promptTokensDetails"),
                    response_tokens_details=details(response_details),
                    raw=raw,
                    scope_id=f"{self.id}-usage-{len(self.records)}",
                )
            )
        if "toolCallCancellation" in message:
            ids = tuple(message["toolCallCancellation"].get("ids", []))
            self.cancelled.update(ids)
            for ident in ids:
                self.pending.pop(ident, None)
            self.events.append(ProviderEvent("cancel_tools", cancelled_ids=ids))
        if "toolCall" in message:
            calls = []
            for item in message["toolCall"]["functionCalls"]:
                ident = item["id"]
                if ident in self.pending or ident in self.cancelled:
                    raise ProviderError("Duplicate Gemini function call")
                self.pending[ident] = item["name"]
                calls.append(
                    ToolCall(
                        id=ident, name=item["name"], arguments=item.get("args", {})
                    )
                )
            self.events.append(ProviderEvent("tools", calls=tuple(calls)))
        content = message.get("serverContent", {})
        if message.get("voiceActivity", {}).get("type") == "ACTIVITY_START":
            self.events.append(ProviderEvent("interrupt"))
        if content.get("interrupted"):
            self.pcm.clear()
            self.text = ""
            self.dropping = True
            self.events.append(ProviderEvent("interrupt"))
        if content.get("inputTranscription", {}).get("text"):
            self.events.append(
                ProviderEvent("transcript", text=content["inputTranscription"]["text"])
            )
        if self.dropping:
            if content.get("turnComplete"):
                self.dropping = False
                self.active = False
            return
        for part in content.get("modelTurn", {}).get("parts", []):
            blob = part.get("inlineData")
            if blob:
                if blob.get("mimeType") not in {
                    "audio/pcm;rate=24000",
                    "audio/pcm;rate=24000;channels=1",
                }:
                    raise ProviderError("Unexpected Gemini audio format")
                self.active = True
                self.pcm.extend(decoded_pcm(blob["data"]))
        if len(self.pcm) > 24000 * 2 * 90:
            raise ProviderError("Gemini buffered speech exceeded its bound")
        if content.get("outputTranscription", {}).get("text"):
            self.active = True
            self.text += content["outputTranscription"]["text"]
        if content.get("turnComplete"):
            if self.pcm or self.text:
                self.speech(f"gemini-{self.turn}", self.pcm, self.text)
            self.pcm.clear()
            self.text = ""
            self.turn += 1
            self.active = False

    async def end_input(self):
        self.closing = True
        await self.socket.send({"realtimeInput": {"audioStreamEnd": True}})
        async with asyncio.timeout(10):
            while self.active or self.pending:
                await asyncio.sleep(0.01)
        # Live provides response meters, not a session-final usage event.
        # Keep session billing unconfirmed and its reservation intact.
        self.usage_finalized = False

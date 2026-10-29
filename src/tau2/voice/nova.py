"""Explicit, optional Nova Sonic transport. Importing this module is offline.

Protocol: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
Rates: AWS public US East (N. Virginia) price list, 2026-10-28.
"""

import asyncio
import base64
import json
import math
import uuid
from dataclasses import dataclass, field
from decimal import Decimal


def _identity_values(credentials):
    # Boto accepts an empty configured token; Smithy signs it unless it is None.
    return {
        "access_key_id": credentials.access_key,
        "secret_access_key": credentials.secret_key,
        "session_token": credentials.token or None,
    }


@dataclass(frozen=True)
class NovaSettings:
    profile: str
    region: str = "us-east-1"
    model_id: str = "amazon.nova-2-5-sonic"
    voice: str = "tiffany"
    stop_usd: float = 0.25
    max_seconds: float = 60

    def __post_init__(self):
        if self.model_id != "amazon.nova-2-5-sonic":
            raise ValueError("Only the explicitly priced Nova 2.5 Sonic is supported")
        if self.region != "us-east-1":
            raise ValueError("This pilot's verified price list is for us-east-1")
        if not self.profile:
            raise ValueError("An explicit AWS profile is required")
        if not math.isfinite(self.stop_usd) or not 0 < self.stop_usd <= 5:
            raise ValueError("stop_usd must be positive and at most 5")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 450:
            raise ValueError(
                "Session duration must be positive and at most 450 seconds"
            )


@dataclass
class NovaUsage:
    """Use cumulative meters once, never sum successive cumulative events."""

    totals: dict = field(default_factory=dict)
    events: int = 0

    def update(self, event):
        total = event.get("details", {}).get("total")
        if total is None:
            raise ValueError("Unpriced Nova usage: modality totals are missing")
        parsed = {}
        for direction in ("input", "output"):
            parsed[direction] = {}
            for modality in ("speechTokens", "textTokens"):
                value = total.get(direction, {}).get(modality)
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid Nova usage meter")
                if value < self.totals.get(direction, {}).get(modality, 0):
                    raise ValueError("Nova cumulative usage moved backwards")
                parsed[direction][modality] = value
        self.totals = parsed
        self.events += 1

    @property
    def usd(self):
        if not self.events:
            return None
        rates = {
            "input": {"speechTokens": "3", "textTokens": ".33"},
            "output": {"speechTokens": "12", "textTokens": "2.75"},
        }
        return sum(
            Decimal(value) * Decimal(rates[d][m]) / Decimal(1_000_000)
            for d, meters in self.totals.items()
            for m, value in meters.items()
        )


def initial_events(prompt, system, tools, voice):
    config = {
        "promptName": prompt,
        "textOutputConfiguration": {"mediaType": "text/plain"},
        "audioOutputConfiguration": {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 24000,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "voiceId": voice,
            "encoding": "base64",
            "audioType": "SPEECH",
        },
    }
    if tools:
        config["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
        config["toolConfiguration"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.openai_schema["function"]["description"],
                        "inputSchema": {
                            "json": json.dumps(
                                t.openai_schema["function"]["parameters"]
                            )
                        },
                    }
                }
                for t in tools
            ]
        }
    return [
        {
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7,
                }
            }
        },
        {"promptStart": config},
        {
            "contentStart": {
                "promptName": prompt,
                "contentName": "system",
                "type": "TEXT",
                "role": "SYSTEM",
                "interactive": True,
                "textInputConfiguration": {"mediaType": "text/plain"},
            }
        },
        {
            "textInput": {
                "promptName": prompt,
                "contentName": "system",
                "content": system,
            }
        },
        {"contentEnd": {"promptName": prompt, "contentName": "system"}},
        {
            "contentStart": {
                "promptName": prompt,
                "contentName": "audio",
                "type": "AUDIO",
                "role": "USER",
                "interactive": True,
                "audioInputConfiguration": {
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": 16000,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "audioType": "SPEECH",
                    "encoding": "base64",
                },
            }
        },
    ]


class NovaSession:
    def __init__(self, settings):
        self.settings = settings
        self.prompt = uuid.uuid4().hex
        self.usage = NovaUsage()
        self.client = self.stream = self.output = None
        self.sent_audio_bytes = 0
        self.initialized = False
        self.usage_finalized = False
        self.input_closed = False

    async def connect(self, system, tools):
        # SDK imports and credential resolution happen only on explicit connect.
        import boto3
        from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import AsyncBedrockRuntimeConfig
        from aws_sdk_bedrock_runtime.models import (
            InvokeModelWithBidirectionalStreamOperationInput,
        )
        from smithy_aws_core.identity import AWSCredentialsIdentity
        from smithy_core.retries import RetryStrategyOptions
        from smithy_http.aio.crt import AWSCRTHTTPClient

        credentials = boto3.Session(
            profile_name=self.settings.profile
        ).get_credentials()
        if credentials is None:
            raise ValueError("No credentials available for the requested AWS profile")
        frozen = credentials.get_frozen_credentials()

        class Resolver:
            async def get_identity(self, **kwargs):
                return AWSCredentialsIdentity(**_identity_values(frozen))

        config = await AsyncBedrockRuntimeConfig.resolve(
            profile=self.settings.profile,
            region=self.settings.region,
            transport=AWSCRTHTTPClient(),
            aws_credentials_identity_resolver=Resolver(),
            retry_strategy=RetryStrategyOptions(max_attempts=1),
        )
        self.client = AsyncBedrockRuntimeClient(config=config)
        await self.client.__aenter__()
        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(
                model_id=self.settings.model_id
            )
        )
        for event in initial_events(self.prompt, system, tools, self.settings.voice):
            await self.send(event)
        self.initialized = True

    async def send(self, event):
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )

        await self.stream.input_stream.send(
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(
                    bytes_=json.dumps({"event": event}).encode()
                )
            )
        )

    async def audio(self, pcm):
        if len(pcm) % 2:
            raise ValueError("Nova requires aligned PCM16 samples")
        if self.sent_audio_bytes + len(pcm) > self.settings.max_seconds * 32000:
            raise ValueError("Nova audio duration limit reached")
        await self.send(
            {
                "audioInput": {
                    "promptName": self.prompt,
                    "contentName": "audio",
                    "content": base64.b64encode(pcm).decode("ascii"),
                }
            }
        )
        self.sent_audio_bytes += len(pcm)

    async def tool_result(self, call_id, result):
        name = uuid.uuid4().hex
        await self.send(
            {
                "contentStart": {
                    "promptName": self.prompt,
                    "contentName": name,
                    "type": "TOOL",
                    "role": "TOOL",
                    "interactive": False,
                    "toolResultInputConfiguration": {
                        "toolUseId": call_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        )
        await self.send(
            {
                "toolResult": {
                    "promptName": self.prompt,
                    "contentName": name,
                    "content": json.dumps({"result": result}),
                }
            }
        )
        await self.send(
            {"contentEnd": {"promptName": self.prompt, "contentName": name}}
        )

    async def assistant_history(self, text):
        """Commit the already delivered fixed greeting, never customer text."""
        name = uuid.uuid4().hex
        await self.send(
            {
                "contentStart": {
                    "promptName": self.prompt,
                    "contentName": name,
                    "type": "TEXT",
                    "role": "ASSISTANT",
                    "interactive": False,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        )
        await self.send(
            {
                "textInput": {
                    "promptName": self.prompt,
                    "contentName": name,
                    "content": text,
                }
            }
        )
        await self.send(
            {"contentEnd": {"promptName": self.prompt, "contentName": name}}
        )

    async def end_input(self):
        if self.stream is not None and not self.input_closed:
            if self.initialized:
                await self.send(
                    {"contentEnd": {"promptName": self.prompt, "contentName": "audio"}}
                )
                await self.send({"promptEnd": {"promptName": self.prompt}})
                await self.send({"sessionEnd": {}})
            await self.stream.input_stream.close()
            self.input_closed = True

    async def receive(self):
        from aws_sdk_bedrock_runtime.models import (
            InvokeModelWithBidirectionalStreamOutputChunk,
        )

        if self.output is None:
            _, self.output = await self.stream.await_output()
        if self.output is None:
            raise RuntimeError("Nova returned no output stream")
        result = await self.output.receive()
        if result is None:
            raise EOFError("Nova stream ended")
        if not isinstance(result, InvokeModelWithBidirectionalStreamOutputChunk):
            raise RuntimeError("Nova stream error: " + type(result).__name__)
        event = json.loads(result.value.bytes_)["event"]
        if "usageEvent" in event:
            self.usage.update(event["usageEvent"])
            if self.usage.usd >= Decimal(str(self.settings.stop_usd)):
                raise RuntimeError("Nova spending threshold reached")
        return event

    async def close(self):
        try:
            if self.stream is not None:
                if self.initialized:
                    async with asyncio.timeout(3):
                        await self.end_input()
                        while not self.usage_finalized:
                            try:
                                await self.receive()
                            except EOFError:
                                self.usage_finalized = True
                                break
                else:
                    await self.end_input()
        finally:
            if self.client is not None:
                await self.client.__aexit__(None, None, None)

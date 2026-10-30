import asyncio
import base64
import copy
import json
from types import SimpleNamespace

import pytest

from tau2.data_model.usage import UsageRecord
from tau2.voice.pricing import build_session_usage
from tau2.voice.providers.base import ProviderError, ProviderSettings
from tau2.voice.providers.elevenlabs import (
    ElevenSession,
    eleven_config,
    preset_fingerprint,
    verify_preset,
)
from tau2.voice.providers.gemini import GeminiSession
from tau2.voice.providers.live import LiveSession, TimedSpeech
from tau2.voice.providers.realtime import RealtimeSession, realtime_usage


def settings(provider="openai-realtime", **kw):
    values = dict(
        provider=provider,
        model={
            "openai-realtime": "gpt-realtime-2.1",
            "gemini-live": "gemini-3.8-live",
            "openai-live": "gpt-live-1",
            "elevenlabs": "gpt-4.1",
        }[provider],
        voice="voice",
        key_env="VOICE_TEST_KEY",
        stop_usd=1,
    )
    if provider == "openai-live":
        values["backend_model"] = "gpt-6-luna"
    if provider == "elevenlabs":
        values.update(
            agent_id="agent_test",
            preset_sha256="a" * 64,
            usd_per_credit=0.00001,
            max_usd_per_minute=0.1,
        )
    return ProviderSettings(**(values | kw))


def usage():
    return {
        "input_tokens": 3,
        "output_tokens": 3,
        "input_token_details": {
            "text_tokens": 1,
            "audio_tokens": 2,
            "cached_tokens": 0,
            "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
        },
        "output_token_details": {"text_tokens": 1, "audio_tokens": 2},
    }


class Socket:
    def __init__(self, events=()):
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.sent = []
        self.closed = False

    async def send(self, event):
        self.sent.append(event)

    async def recv(self):
        event = await self.events.get()
        if isinstance(event, BaseException):
            raise event
        return event

    async def close(self):
        if not self.closed:
            self.closed = True
            self.events.put_nowait(EOFError())


def tools():
    return [
        SimpleNamespace(
            name="lookup",
            openai_schema={
                "function": {
                    "name": "lookup",
                    "description": "Read a record",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                }
            },
        )
    ]


def realtime_reply(ident="resp", item="item", *, text="Done.", status="completed"):
    return {
        "type": "response.done",
        "response": {
            "id": ident,
            "status": status,
            "usage": usage(),
            "output": [
                {
                    "id": item,
                    "type": "message",
                    "content": [{"type": "audio", "transcript": text}],
                },
            ],
        },
    }


def test_realtime_handshake_contains_only_public_policy_and_tools(monkeypatch):
    async def run():
        socket = Socket(
            [
                {"type": "session.created"},
                {
                    "type": "session.updated",
                    "session": {"id": "session", "model": "gpt-realtime-2.1"},
                },
            ]
        )
        connections = []

        async def connect(url, headers):
            connections.append((url, headers))
            return socket

        session = RealtimeSession(settings(), connector=connect)
        await session.connect("PUBLIC POLICY", tools())
        assert session.connected
        assert connections[0][1]["Authorization"] == "Bearer private-test-key"
        config = socket.sent[0]["session"]
        assert config["instructions"] == "PUBLIC POLICY"
        assert config["tools"][0]["name"] == "lookup"
        assert config["audio"]["input"]["format"]["rate"] == 24000
        assert "private-test-key" not in json.dumps(config)
        await session.assistant_history("Only delivered greeting")
        assert socket.sent[-1]["item"]["role"] == "assistant"
        await session.close()

    monkeypatch.setenv("VOICE_TEST_KEY", "private-test-key")
    asyncio.run(run())


def test_realtime_buffers_final_audio_and_truncates_completed_but_unplayed_reply():
    async def run():
        s = RealtimeSession(settings())
        s.socket = Socket()
        pcm = b"\x00\x10" * 2400
        await s.parse({"type": "response.created", "response": {"id": "resp"}})
        await s.parse(
            {
                "type": "response.output_audio.delta",
                "response_id": "resp",
                "item_id": "item",
                "content_index": 0,
                "delta": base64.b64encode(pcm).decode(),
            }
        )
        assert not s.events
        await s.parse(realtime_reply())
        speech = s.events.popleft()
        assert speech.audio == pcm and speech.text == "Done." and speech.final
        assert s.metered_usd == pytest.approx(0.00022)
        await s.playback("item:0", 50)
        await s.parse({"type": "input_audio_buffer.speech_started"})
        assert s.socket.sent[-1] == {
            "type": "conversation.item.truncate",
            "item_id": "item",
            "content_index": 0,
            "audio_end_ms": 50,
        }
        assert s.events[-1].kind == "interrupt"

    asyncio.run(run())


def test_realtime_cancelled_response_still_bills_but_never_plays():
    async def run():
        s = RealtimeSession(settings())
        await s.parse(realtime_reply(status="cancelled"))
        assert not s.events and len(s.records) == 1
        with pytest.raises(ProviderError, match="Duplicate provider billing"):
            await s.parse(realtime_reply(status="cancelled"))

    asyncio.run(run())


def test_realtime_tool_batch_preserves_ids_and_continues_after_all_results():
    async def run():
        s = RealtimeSession(settings())
        s.socket = Socket()
        event = realtime_reply()
        event["response"]["output"] = [
            {
                "type": "function_call",
                "call_id": key,
                "name": "lookup",
                "arguments": '{"id":"exact"}',
            }
            for key in ("one", "two")
        ]
        await s.parse(event)
        assert [c.id for c in s.events[-1].calls] == ["one", "two"]
        await s.tool_result("one", {"content": "first"})
        assert not any(e["type"] == "response.create" for e in s.socket.sent)
        await s.tool_result("two", {"content": "second"})
        assert s.socket.sent[-1]["type"] == "response.create"
        with pytest.raises(ProviderError, match="no matching"):
            await s.tool_result("two", {})

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda u: u.pop("output_token_details"),
        lambda u: u.update(input_tokens=-1),
        lambda u: u["input_token_details"].update(audio_tokens=100),
        lambda u: u["input_token_details"]["cached_tokens_details"].update(
            audio_tokens=3
        ),
    ],
)
def test_realtime_rejects_unpriceable_or_invalid_meters(mutate):
    u = usage()
    mutate(u)
    with pytest.raises(ProviderError):
        realtime_usage(u, "gpt-realtime-2.1", "response")


def gemini_usage(prompt=10):
    return {
        "usageMetadata": {
            "promptTokenCount": prompt,
            "responseTokenCount": 5,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": prompt - 6},
                {"modality": "AUDIO", "tokenCount": 6},
            ],
            "responseTokensDetails": [
                {"modality": "TEXT", "tokenCount": 1},
                {"modality": "AUDIO", "tokenCount": 4},
            ],
        }
    }


def test_gemini_audio_transcript_order_and_actual_turn_end():
    async def run():
        s = GeminiSession(settings("gemini-live"))
        s.socket = Socket()
        pcm = b"\x00\x10" * 2400
        await s.parse(
            {"serverContent": {"outputTranscription": {"text": "A final reply."}}}
        )
        await s.parse(
            {
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/pcm;rate=24000",
                                    "data": base64.b64encode(pcm).decode(),
                                }
                            }
                        ]
                    },
                    "generationComplete": True,
                }
            }
        )
        assert not s.events
        await s.parse({"serverContent": {"turnComplete": True}})
        assert s.events[0].text == "A final reply." and s.events[0].audio == pcm
        await s.parse(gemini_usage(10))
        await s.parse(gemini_usage(20))
        aggregate = build_session_usage(s.records)
        assert (
            aggregate.records[0].input_tokens == 30
        )  # Context is re-billed on the next response.
        assert (
            len(s.records) == 2 and s.records[0].raw == gemini_usage()["usageMetadata"]
        )

    asyncio.run(run())


def test_gemini_interrupt_discards_late_old_generation_and_tools():
    async def run():
        s = GeminiSession(settings("gemini-live"))
        s.socket = Socket()
        await s.parse(
            {
                "toolCall": {
                    "functionCalls": [
                        {"id": "call", "name": "lookup", "args": {"id": "exact"}}
                    ]
                }
            }
        )
        await s.parse({"toolCallCancellation": {"ids": ["call"]}})
        await s.tool_result("call", {"content": "Atomic action already executed"})
        assert not s.socket.sent
        await s.parse({"serverContent": {"interrupted": True}})
        await s.parse(
            {
                "serverContent": {
                    "outputTranscription": {"text": "Stale"},
                    "turnComplete": True,
                }
            }
        )
        assert not any(e.kind == "speech" for e in s.events)
        with pytest.raises(ProviderError, match="reconnect"):
            await s.parse({"goAway": {"timeLeft": "1s"}})

    asyncio.run(run())


def test_live_uses_timed_reflected_audio_and_does_not_release_without_transcript():
    spoken = []
    timed = TimedSpeech(lambda *args: spoken.append(args))
    tone = b"\x00\x10" * 2400
    timed.add_audio(
        {"start_ms": 100, "end_ms": 200, "delta": base64.b64encode(tone).decode()}
    )
    assert not spoken
    timed.add_caption({"start_ms": 100, "end_ms": 200, "delta": "A reply."})
    timed.add_audio(
        {
            "start_ms": 200,
            "end_ms": 1600,
            "delta": base64.b64encode(bytes(1400 * 48)).decode(),
        }
    )
    assert len(spoken) == 1 and spoken[0][2] == "A reply."
    assert spoken[0][1].startswith(tone)
    with pytest.raises(ProviderError, match="after its audio"):
        timed.add_caption({"start_ms": 100, "end_ms": 200, "delta": "Late text"})
    missing = TimedSpeech(lambda *_: None)
    missing.add_audio(
        {"start_ms": 0, "end_ms": 100, "delta": base64.b64encode(tone).decode()}
    )
    with pytest.raises(ProviderError, match="without its transcript"):
        missing.flush(now_ms=2000)


def test_live_reflected_gap_inside_transcript_is_an_error():
    timed = TimedSpeech(lambda *_: pytest.fail("Missing audio must not play"))
    timed.add_audio(
        {
            "start_ms": 0,
            "end_ms": 100,
            "delta": base64.b64encode(b"\x00\x10" * 2400).decode(),
        }
    )
    timed.add_caption(
        {"start_ms": 0, "end_ms": 250, "delta": "Contains a missing word"}
    )
    with pytest.raises(ProviderError, match="missing reflected audio"):
        timed.add_audio(
            {
                "start_ms": 200,
                "end_ms": 1800,
                "delta": base64.b64encode(bytes(1600 * 48)).decode(),
            }
        )


def test_live_nested_functions_and_cumulative_duration_are_separate_meters():
    async def run():
        s = LiveSession(settings("openai-live"))
        s.primary = Socket()
        await s.parse({"type": "session.usage.updated", "usage": {"seconds": 12}})
        await s.parse({"type": "session.usage.updated", "usage": {"seconds": 15}})
        assert len(s.records) == 1 and s.records[0].audio_input_seconds == 15
        function = {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "exact-id",
                "name": "lookup",
                "arguments": '{"id":"exact"}',
            },
        }
        await s.parse(
            {"type": "response.event", "delegation_id": "delegation", "event": function}
        )
        assert s.events[-1].calls[0].id == "exact-id"
        await s.tool_result("exact-id", {"content": "verified"})
        assert s.primary.sent[0]["item"]["call_id"] == "exact-id"
        assert s.primary.sent[1]["type"] == "response.create"
        await s.parse(
            {
                "type": "response.event",
                "event": {
                    "type": "response.completed",
                    "response": {
                        "id": "backend",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 5,
                            "input_tokens_details": {"cached_tokens": 15},
                        },
                    },
                },
            }
        )
        assert s.metered_usd == pytest.approx(15 / 60 * 0.05 + 0.00001115)
        assert build_session_usage(s.usage_records()).cost is None
        s.closing = True
        await s.parse(
            {
                "type": "session.closed",
                "reason": "close_requested",
                "usage": {"seconds": 15},
            }
        )
        assert s.usage_finalized and len(s.records) == 2

    asyncio.run(run())


def test_eleven_preset_must_have_exact_client_tools_and_completion_events():
    wanted = eleven_config("PUBLIC", tools(), "gpt-4.1", "voice", 240)
    verify_preset(wanted, wanted)
    for mutate in [
        lambda c: c["agent"]["prompt"].update(prompt="Different policy"),
        lambda c: c["agent"]["prompt"]["tools"][0].update(type="webhook"),
        lambda c: c["conversation"].update(client_events=["audio"]),
        lambda c: c["tts"].update(voice_id="different"),
        lambda c: c.update(language_presets={"de": {}}),
        lambda c: c["agent"]["prompt"]["tools"][0].update(description="Different tool"),
    ]:
        broken = copy.deepcopy(wanted)
        mutate(broken)
        with pytest.raises(ProviderError):
            verify_preset(broken, wanted)


def test_eleven_fingerprint_covers_platform_workflow_and_resolved_tools():
    preset = {
        "conversation_config": eleven_config("PUBLIC", tools(), "gpt-4.1", "voice", 90)
    }
    original = preset_fingerprint(preset)
    for key in ("platform_settings", "workflow"):
        changed = copy.deepcopy(preset)
        changed[key] = {"changed": True}
        assert preset_fingerprint(changed) != original
    changed = copy.deepcopy(preset)
    changed["conversation_config"]["agent"]["prompt"]["tools"][0]["description"] = (
        "Changed"
    )
    assert preset_fingerprint(changed) != original


def test_eleven_preserves_credit_and_surcharge_details_without_claiming_final_usd():
    async def run():
        s = ElevenSession(settings("elevenlabs"))
        s.socket = Socket()
        metadata = {
            "cost": 40,
            "call_duration_secs": 20,
            "charging": {"voice_surcharge_charge": 7, "llm_charge": 5},
        }

        async def get(_):
            return {"status": "done", "metadata": metadata}

        s.get = get
        await s.end_input()
        assert s.usage_finalized and s.metered_usd is None
        assert s.records[0].raw["credits"] == 40
        assert s.records[0].raw["charging"] == metadata["charging"]
        assert s.records[0].raw["final_usd_confirmed"] is False
        assert build_session_usage(s.usage_records()).cost is None

    asyncio.run(run())


@pytest.mark.parametrize("seconds", [-1, float("nan"), True])
def test_live_rejects_invalid_or_decreasing_session_duration(seconds):
    s = LiveSession(settings("openai-live"))
    s.voice_usage(2)
    with pytest.raises(ProviderError, match="duration meter"):
        s.voice_usage(seconds)


@pytest.mark.parametrize("provider", ["openai-realtime", "openai-live"])
def test_shutdown_waits_for_requested_response_before_its_created_event(provider):
    async def run():
        s = (
            RealtimeSession(settings())
            if provider == "openai-realtime"
            else LiveSession(settings("openai-live"))
        )
        s.socket = Socket()
        if provider == "openai-live":
            s.primary = Socket()
        else:
            s._clear_ack.set()
        s.pending.add("pending")
        await s.tool_result("pending", {"content": "atomic result"})
        assert s.awaiting_response
        shutting = asyncio.create_task(s.end_input())
        await asyncio.sleep(0)
        assert not shutting.done()
        if provider == "openai-realtime":
            await s.parse(
                {"type": "response.created", "response": {"id": "continuation"}}
            )
            event = realtime_reply("continuation")
            event["response"]["output"] = []
            await s.parse(event)
        else:
            await s.parse(
                {
                    "type": "response.event",
                    "event": {
                        "type": "response.created",
                        "response": {"id": "continuation"},
                    },
                }
            )
            await s.parse(
                {
                    "type": "response.event",
                    "event": {
                        "type": "response.completed",
                        "response": {
                            "id": "continuation",
                            "usage": {"input_tokens": 3, "output_tokens": 1},
                        },
                    },
                }
            )
            await asyncio.sleep(0.02)
            assert s.primary.sent[-1]["type"] == "session.close"
            await s.parse(
                {
                    "type": "session.closed",
                    "reason": "close_requested",
                    "usage": {"seconds": 2},
                }
            )
        await asyncio.wait_for(shutting, 0.5)
        assert s.usage_finalized

    asyncio.run(run())


@pytest.mark.parametrize("response_flag", [{}, {"expects_response": True}])
def test_eleven_completion_event_waits_for_audio_text_and_tool_results(response_flag):
    async def run():
        s = ElevenSession(settings("elevenlabs"))
        s.socket = Socket()
        await s.parse(
            {
                "type": "client_tool_call",
                "client_tool_call": {
                    **response_flag,
                    "tool_call_id": "client",
                    "tool_name": "lookup",
                    "parameters": {"id": "exact"},
                },
            }
        )
        with pytest.raises(ProviderError, match="unresolved tools"):
            await s.parse(
                {
                    "type": "agent_response_complete",
                    "agent_response_complete_event": {"event_id": 1},
                }
            )
        await s.tool_result("client", {"content": "actual", "error": False})
        assert s.socket.sent[-1]["tool_call_id"] == "client"
        await s.parse(
            {
                "type": "audio",
                "audio_event": {
                    "audio_base_64": base64.b64encode(b"\x00\x10" * 2400).decode(),
                    "event_id": 2,
                },
            }
        )
        await s.parse(
            {
                "type": "agent_response",
                "agent_response_event": {"agent_response": "Verified.", "event_id": 2},
            }
        )
        assert not any(e.kind == "speech" for e in s.events)
        await s.parse(
            {
                "type": "agent_response_complete",
                "agent_response_complete_event": {"event_id": 2},
            }
        )
        assert s.events[-1].kind == "speech" and s.events[-1].text == "Verified."
        await s.parse({"type": "ping", "ping_event": {"event_id": 55}})
        assert s.socket.sent[-1] == {"type": "pong", "event_id": 55}
        with pytest.raises(ProviderError, match="undeclared server tool"):
            await s.parse(
                {
                    "type": "agent_tool_response",
                    "agent_tool_response": {"tool_type": "webhook"},
                }
            )

    asyncio.run(run())


def test_eleven_interrupted_audio_ids_are_discarded_before_next_response():
    async def run():
        s = ElevenSession(settings("elevenlabs"))
        old = b"\x00\x10" * 2400
        fresh = b"\x00\x20" * 2400

        async def audio(ident, pcm):
            await s.parse(
                {
                    "type": "audio",
                    "audio_event": {
                        "event_id": ident,
                        "audio_base_64": base64.b64encode(pcm).decode(),
                    },
                }
            )

        await audio(1, old)
        await s.parse({"type": "interruption", "interruption_event": {"event_id": 2}})
        await audio(1, old)
        await audio(2, old)
        assert not s.pcm and s.events[-1].kind == "interrupt"
        await audio(3, fresh)
        await s.parse(
            {
                "type": "agent_response",
                "agent_response_event": {"agent_response": "New response."},
            }
        )
        await s.parse(
            {
                "type": "agent_response_complete",
                "agent_response_complete_event": {"event_id": 3},
            }
        )
        assert s.events[-1].audio == fresh
        assert s.events[-1].text == "New response."

    asyncio.run(run())


def test_eleven_rejects_explicit_nonresponse_tools_even_with_legacy_event_support():
    async def run():
        s = ElevenSession(settings("elevenlabs"))
        with pytest.raises(ProviderError, match="Invalid ElevenLabs client tool call"):
            await s.parse(
                {
                    "type": "client_tool_call",
                    "client_tool_call": {
                        "expects_response": False,
                        "tool_call_id": "invalid",
                        "tool_name": "lookup",
                        "parameters": {},
                    },
                }
            )
        assert not s.pending and not s.events

    asyncio.run(run())


def test_eleven_readonly_preflight_pins_config_and_rejects_signed_url_redirect(
    monkeypatch,
):
    async def run():
        config = eleven_config("PUBLIC", tools(), "gpt-4.1", "voice", 240)
        digest = preset_fingerprint({"conversation_config": config})
        requests = []

        class HTTP:
            def __init__(self, **kwargs):
                assert kwargs["headers"] == {"xi-api-key": "private-test-key"}

            async def get(self, url):
                requests.append(url)
                obj = (
                    {"conversation_config": config}
                    if "/agents/" in url
                    else {"signed_url": "wss://untrusted.invalid/steal"}
                )
                return SimpleNamespace(
                    raise_for_status=lambda: None, json=lambda: copy.deepcopy(obj)
                )

            async def aclose(self):
                pass

        async def connect(*_):
            pytest.fail("An untrusted endpoint must not receive audio or credentials")

        s = ElevenSession(
            settings("elevenlabs", preset_sha256=digest),
            connector=connect,
            http_factory=HTTP,
        )
        with pytest.raises(ProviderError, match="Unexpected ElevenLabs"):
            await s.connect("PUBLIC", tools())
        assert len(requests) == 2
        await s.close()

    monkeypatch.setenv("VOICE_TEST_KEY", "private-test-key")
    asyncio.run(run())


@pytest.mark.parametrize(
    "provider", ["openai-realtime", "openai-live", "gemini-live", "elevenlabs"]
)
def test_unconfirmed_usage_never_becomes_free(provider):
    cls = {
        "openai-realtime": RealtimeSession,
        "openai-live": LiveSession,
        "gemini-live": GeminiSession,
        "elevenlabs": ElevenSession,
    }[provider]
    s = cls(settings(provider))
    s.records.append(
        UsageRecord(provider="openai", model="gpt-live-1", audio_input_seconds=10)
    )
    assert build_session_usage(s.usage_records()).cost is None

import asyncio
import audioop
from types import SimpleNamespace

import pytest

from tau2.agent.provider_agent import ProviderAgent
from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.voice.providers.base import ProviderError, ProviderEvent, ProviderSettings


def agent(**kwargs):
    return ProviderAgent(
        [],
        "PUBLIC POLICY",
        settings=ProviderSettings(
            "openai-realtime", "gpt-realtime-2.1", "marin", "TEST_KEY"
        ),
        opener_backend=kwargs.pop("opener_backend", None),
        session_factory=kwargs.pop("session_factory", None),
        pace=False,
        **kwargs,
    )


def commands(a):
    items = []
    while not a._commands.empty():
        items.append(a._commands.get_nowait())
    return items


def test_provider_output_conversion_is_stateful_and_history_is_delivered_only():
    a = agent()
    pcm = b"\x00\x10" * 3600
    a._process(ProviderEvent("speech", "first", "First. ", pcm[:3600], final=False))
    a._process(ProviderEvent("speech", "last", "Last.", pcm[3600:]))
    state = a.get_init_state()
    assert len(state.messages) == 1
    frames = []
    for _ in range(4):
        reply, state = a.get_next_chunk(state)
        frames.append(reply)
        commands(a)
    converted, _ = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)
    # Each segment has a final padded frame. Padding is telephony silence.
    raw = audioop.lin2ulaw(converted, 2)
    assert frames[0].get_audio_bytes() == raw[:400]
    assert frames[1].get_audio_bytes() == raw[400:600].ljust(400, b"\xff")
    assert frames[2].get_audio_bytes() == raw[600:1000]
    assert frames[3].get_audio_bytes() == raw[1000:].ljust(400, b"\xff")
    assert [f.is_final_chunk for f in frames] == [False, False, False, True]
    assert "".join(f.content or "" for f in frames) == "First. Last."
    assert "".join(m.content or "" for m in state.messages[1:]) == "First. Last."


def test_provider_input_is_stateful_audio_without_private_text_or_speech_labels():
    a = agent()
    a._session = SimpleNamespace(input_rate=16000)
    source = audioop.lin2ulaw(b"\x00\x20" * 800, 2)
    state = a.get_init_state()
    sent = []
    for offset in (0, 400):
        chunk = UserMessage(
            role="user",
            content="PRIVATE GOAL",
            contains_speech=False,
            audio_script_gold="REFERENCE ACTION",
            is_audio=True,
            audio_content=audio_bytes_to_string(source[offset : offset + 400]),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        )
        a.get_next_chunk(state, chunk)
        sent.extend(value for kind, value in commands(a) if kind == "audio")
    expected, _ = audioop.ratecv(audioop.ulaw2lin(source, 2), 2, 1, 8000, 16000, None)
    assert b"".join(sent) == expected
    assert len(state.messages) == 1 and not a.metrics["asr"]


@pytest.mark.parametrize("size,rate", [(399, 8000), (401, 8000), (400, 16000)])
def test_provider_rejects_invalid_telephony_frames(size, rate):
    a = agent()
    message = UserMessage(
        role="user",
        is_audio=True,
        audio_content=audio_bytes_to_string(b"\xff" * size),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=rate),
    )
    with pytest.raises(ValueError, match="400-byte"):
        a.get_next_chunk(a.get_init_state(), message)
    assert not commands(a)


def test_provider_interruption_cancels_only_unissued_tools_and_unplayed_speech():
    a = agent()
    call = ToolCall(id="issued", name="lookup", arguments={})
    a._process(ProviderEvent("tools", calls=(call,)))
    reply, state = a.get_next_chunk(a.get_init_state())
    assert reply.tool_calls[0].id == "issued"
    commands(a)
    a._process(ProviderEvent("speech", "reply", "Undelivered.", b"\x00\x10" * 2400))
    a._process(
        ProviderEvent(
            "tools", calls=(ToolCall(id="queued", name="write", arguments={}),)
        )
    )
    a._process(ProviderEvent("interrupt"))
    assert not a._ready and not a._calls and a._pending == {"issued"}
    assert a.metrics["discarded_samples"] > 0
    cancelled = commands(a)
    assert cancelled[0][1][0] == "queued"
    assert cancelled[0][1][1]["execution_status"] == "cancelled"
    a._results(ToolMessage(role="tool", id="issued", content="Actual result"))
    assert commands(a)[0][1] == (
        "issued",
        {"content": "Actual result", "error": False, "execution_status": "executed"},
    )
    assert not a._pending
    assert len(state.messages) == 2  # No undelivered speech in history.


def test_provider_duplicate_call_and_result_ids_fail_before_execution_or_send():
    a = agent()
    event = ProviderEvent(
        "tools", calls=(ToolCall(id="same", name="lookup", arguments={}),)
    )
    a._process(event)
    with pytest.raises(ProviderError, match="Duplicate"):
        a._process(event)
    a._pending = {"same"}
    message = ToolMessage(role="tool", id="same", content="Actual")
    with pytest.raises(ProviderError, match="do not match"):
        a._results(MultiToolMessage(role="tool", tool_messages=[message, message]))
    assert not commands(a) and a._pending == {"same"}


def test_provider_queue_overflow_is_explicit_and_does_not_drop_input():
    a = agent()
    for _ in range(a._commands.maxsize):
        a.command("audio", b"pcm")
    with pytest.raises(ProviderError, match="no audio was dropped"):
        a.command("audio", b"last")
    assert len(commands(a)) == 120


def test_provider_inference_error_is_propagated_without_secret_transport_details():
    a = agent()
    a._failure = ExceptionGroup(
        "group",
        [RuntimeError("wss://secret?key=private"), ProviderError("Safe failure")],
    )
    with pytest.raises(ProviderError) as caught:
        a.get_next_chunk(a.get_init_state())
    assert "RuntimeError; Safe failure" in str(caught.value)
    assert "secret" not in str(caught.value) and "private" not in str(caught.value)


@pytest.mark.parametrize("fail", [False, True])
def test_provider_worker_closes_connections_and_collects_usage_on_failure(fail):
    sessions = []

    class Session:
        input_rate = 24000
        records = []
        usage_finalized = False
        metered_usd = None

        def __init__(self, settings):
            self.queue = asyncio.Queue()
            self.closed = False
            sessions.append(self)

        async def connect(self, policy, tools):
            assert policy.endswith("PUBLIC POLICY") and tools == []
            if fail:
                raise ProviderError("fixture startup failure")

        async def receive(self):
            await self.queue.get()
            raise EOFError()

        async def end_input(self):
            self.queue.put_nowait(None)

        async def close(self):
            self.closed = True

        def usage_records(self):
            return []

    opener = SimpleNamespace(
        synthesize=lambda *_: AudioData(
            data=b"\x00\x10" * 2400,
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=24000),
        )
    )
    a = agent(session_factory=Session, opener_backend=opener)
    if fail:
        with pytest.raises(ProviderError, match="startup failure"):
            a.create_initial_message()
        with pytest.raises(ProviderError, match="startup failure"):
            a.stop()
    else:
        a.create_initial_message()
        a.stop()
        a.stop()  # Idempotent cleanup.
    assert sessions[0].closed and not a._thread.is_alive()
    assert a.metrics["usage_finalized"] is False and a.metrics["metered_usd"] is None

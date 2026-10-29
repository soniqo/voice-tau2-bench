import asyncio
import audioop
import base64
import threading
from types import SimpleNamespace

import pytest

from tau2.agent.nova_agent import NovaAgent, NovaAgentError
from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import ToolMessage, UserMessage
from tau2.voice.nova import NovaSettings, NovaUsage


def agent(**kwargs):
    return NovaAgent(
        [],
        "public policy only",
        settings=NovaSettings(profile="fixture"),
        opener_backend=None,
        pace=False,
        **kwargs,
    )


def feed_turn(
    a,
    *,
    completion="c",
    text="The task is complete.",
    samples=3600,
    split=166,
    prefix=None,
    audio_stop="END_TURN",
    text_stop="END_TURN",
):
    suffix = prefix or completion
    a._process(
        {
            "contentStart": {
                "contentId": "spec-" + suffix,
                "completionId": completion,
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
            }
        }
    )
    a._process(
        {"contentEnd": {"contentId": "spec-" + suffix, "stopReason": "PARTIAL_TURN"}}
    )
    a._process(
        {
            "contentStart": {
                "contentId": "audio-" + suffix,
                "completionId": completion,
                "type": "AUDIO",
                "role": "ASSISTANT",
                "audioOutputConfiguration": {
                    "sampleRateHertz": 24000,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                },
            }
        }
    )
    pcm = b"\x00\x10" * samples
    for start in range(0, len(pcm), split):
        a._process(
            {
                "audioOutput": {
                    "contentId": "audio-" + suffix,
                    "completionId": completion,
                    "content": base64.b64encode(pcm[start : start + split]).decode(),
                }
            }
        )
    a._process(
        {"contentEnd": {"contentId": "audio-" + suffix, "stopReason": audio_stop}}
    )
    a._process(
        {
            "contentStart": {
                "contentId": "text-" + suffix,
                "completionId": completion,
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        }
    )
    a._process(
        {
            "textOutput": {
                "contentId": "text-" + suffix,
                "completionId": completion,
                "content": text,
            }
        }
    )
    a._process({"contentEnd": {"contentId": "text-" + suffix, "stopReason": text_stop}})
    return pcm


def next_frame(a, state):
    result = a.get_next_chunk(state)
    while not a._commands.empty():
        a._commands.get_nowait()
    return result[0]


def test_audio_conversion_is_stateful_and_history_contains_only_delivered_text():
    a = agent()
    pcm = feed_turn(a)
    state = a.get_init_state()
    messages = [next_frame(a, state) for _ in range(3)]
    reference, _ = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)
    assert b"".join(m.get_audio_bytes() for m in messages) == audioop.lin2ulaw(
        reference, 2
    )
    assert [m.is_final_chunk for m in messages] == [False, False, True]
    assert "".join(m.content or "" for m in messages) == "The task is complete."
    assert "".join(m.content for m in state.messages[1:]) == "The task is complete."


def test_multi_segment_reply_ends_at_audio_boundary_not_final_text_marker():
    # Live Sonic emitted END_TURN on the first FINAL text block and
    # PARTIAL_TURN on the last. AUDIO boundaries identify the actual spoken turn.
    a = agent()
    first = "Airplane mode is on."
    second = " Please turn it off and test your connection."
    feed_turn(
        a,
        prefix="first",
        text=first,
        audio_stop="PARTIAL_TURN",
        text_stop="END_TURN",
    )
    feed_turn(
        a,
        prefix="last",
        text=second,
        audio_stop="END_TURN",
        text_stop="PARTIAL_TURN",
    )
    state = a.get_init_state()
    frames = [next_frame(a, state) for _ in range(6)]
    assert [m.is_final_chunk for m in frames] == [False] * 5 + [True]
    assert "".join(m.content or "" for m in frames) == first + second


def test_input_ignores_text_gold_scripts_and_speech_labels():
    a = agent()
    data = audioop.lin2ulaw(b"\x00\x20" * 400, 2)
    message = UserMessage(
        role="user",
        content="PRIVATE-GOAL",
        audio_script_gold="SECRET",
        contains_speech=False,
        is_audio=True,
        audio_content=audio_bytes_to_string(data),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    a.get_next_chunk(a.get_init_state(), participant_chunk=message)
    kind, pcm = a._commands.get_nowait()
    assert kind == "audio" and len(pcm) > 0 and any(pcm)
    assert "SECRET" not in str(a.metrics) and "PRIVATE-GOAL" not in str(a.metrics)


@pytest.mark.parametrize(
    "encoding,rate,size",
    [
        (AudioEncoding.ULAW, 16000, 400),
        (AudioEncoding.PCM_S16LE, 8000, 400),
        (AudioEncoding.ULAW, 8000, 399),
    ],
)
def test_invalid_benchmark_frames_are_rejected(encoding, rate, size):
    a = agent()
    message = UserMessage(
        role="user",
        is_audio=True,
        contains_speech=False,
        audio_content=audio_bytes_to_string(bytes(size)),
        audio_format=AudioFormat(encoding=encoding, sample_rate=rate),
    )
    with pytest.raises(ValueError, match="400-byte"):
        a.get_next_chunk(a.get_init_state(), participant_chunk=message)


def test_barge_in_drops_unplayed_audio_and_pending_tools_but_keeps_delivered_history():
    a = agent()
    feed_turn(a, text="This sentence has not finished playing.")
    state = a.get_init_state()
    first = next_frame(a, state)
    delivered = first.content
    a._tools.append(SimpleNamespace(id="stale"))
    a._process({"userSpeechStart": {}})
    assert not a._ready and not a._tools
    assert "".join(m.content for m in state.messages[1:]) == delivered
    assert a.metrics["discarded_samples"] == 800
    # Already cancelled completions cannot re-emit audio.
    a._process(
        {
            "audioOutput": {
                "contentId": "audio-c",
                "completionId": "c",
                "content": "AAAA",
            }
        }
    )
    assert not a._ready
    feed_turn(a, completion="new", text="A new reply.")
    assert next_frame(a, state).content


def tool_event(a, ident="call"):
    a._process(
        {
            "contentStart": {
                "contentId": ident,
                "completionId": "c",
                "role": "TOOL",
                "type": "TOOL",
            }
        }
    )
    a._process(
        {
            "toolUse": {
                "contentId": ident,
                "completionId": "c",
                "toolName": "lookup",
                "toolUseId": ident,
                "content": '{"id":"#W123"}',
            }
        }
    )
    a._process({"contentEnd": {"contentId": ident, "stopReason": "TOOL_USE"}})


def test_tools_use_ordinary_orchestrator_channel_and_preserve_ids_and_arguments():
    a = agent()
    tool_event(a)
    state = a.get_init_state()
    message = next_frame(a, state)
    assert message.content is None
    assert message.tool_calls[0].arguments == {"id": "#W123"}
    assert a._pending_ids == {"call"}
    a._results(
        ToolMessage(
            role="tool", requestor="assistant", id="call", content="actual tool result"
        )
    )
    kind, result = a._commands.get_nowait()
    assert (
        kind == "tool"
        and result[0] == "call"
        and result[1]["content"] == "actual tool result"
    )
    assert not a._pending_ids
    with pytest.raises(NovaAgentError, match="Duplicate"):
        tool_event(a)
    with pytest.raises(NovaAgentError, match="do not match"):
        a._results(
            ToolMessage(role="tool", requestor="assistant", id="wrong", content="none")
        )


def test_interruption_does_not_undo_an_already_emitted_atomic_tool_call():
    a = agent()
    tool_event(a)
    next_frame(a, a.get_init_state())
    a._process({"userSpeechStart": {}})
    a._results(
        ToolMessage(role="tool", requestor="assistant", id="call", content="executed")
    )
    assert a._commands.get_nowait()[0] == "tool"


def test_provider_failure_never_becomes_a_silent_zero():
    a = agent()
    a._failure = RuntimeError("transport failed")
    with pytest.raises(NovaAgentError, match="inference failed"):
        a.get_next_chunk(a.get_init_state())
    a._failure = None
    with pytest.raises(NovaAgentError, match="usage"):
        a.raise_if_failed()


def test_worker_closes_both_streams_drains_usage_and_never_sends_customer_text():
    ready = threading.Event()
    commands = []

    class Session:
        def __init__(self, settings):
            self.usage = NovaUsage()
            self.usage_finalized = False
            self.prompt = "fixture"

        async def connect(self, system, tools):
            self.events = asyncio.Queue()
            self.usage.update(
                {
                    "details": {
                        "total": {
                            d: {m: 1 for m in ("speechTokens", "textTokens")}
                            for d in ("input", "output")
                        }
                    }
                }
            )
            ready.set()

        async def audio(self, data):
            commands.append(("audio", len(data)))

        async def assistant_history(self, text):
            commands.append(("assistant", text))

        async def receive(self):
            await self.events.get()
            raise EOFError()

        async def end_input(self):
            await self.events.put(None)

        async def close(self):
            commands.append(("closed",))

    a = agent(session_factory=Session)
    a.opener_backend = SimpleNamespace(
        synthesize=lambda *args: AudioData(
            data=bytes(2 * 400),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=8000),
        )
    )
    a.create_initial_message("Hello.")
    assert ready.wait(2)
    a.get_next_chunk(a.get_init_state())
    a.stop()
    assert not a._thread.is_alive()
    assert ("assistant", "Hello.") in commands
    assert ("closed",) in commands
    assert a.metrics["usage_finalized"] is True
    assert len(a.get_usage_records()) == 1


def test_interrupted_content_does_not_cancel_later_turns_reusing_completion_id():
    a = agent()
    feed_turn(a, completion="shared-session", prefix="old")
    state = a.get_init_state()
    next_frame(a, state)
    a._process({"userSpeechStart": {}})
    assert "audio-old" in a._cancelled
    assert "shared-session" not in a._cancelled
    # A final transcript emitted for interrupted generation is not delivered.
    a._process(
        {
            "contentStart": {
                "contentId": "old-final",
                "completionId": "shared-session",
                "role": "ASSISTANT",
                "type": "TEXT",
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        }
    )
    a._process(
        {"textOutput": {"contentId": "old-final", "content": "Unplayed old words."}}
    )
    a._process({"contentEnd": {"contentId": "old-final", "stopReason": "END_TURN"}})
    feed_turn(
        a, completion="shared-session", prefix="new", text="This is the new response."
    )
    messages = [next_frame(a, state) for _ in range(3)]
    assert "".join(m.content or "" for m in messages) == "This is the new response."
    assert "Unplayed old words" not in str(state.messages)

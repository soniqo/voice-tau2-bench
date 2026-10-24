import base64
import json

import httpx
import pytest

from tau2.agent.speech_swift_agent import SpeechSwiftAgent
from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.voice.speech_swift import SpeechSwiftBackend, SpeechSwiftError


def test_failed_chat_keeps_exact_sent_body_and_error_response():
    received = []

    def server(request):
        received.append(request.content)
        return httpx.Response(500, text="native parser failure")

    backend = SpeechSwiftBackend(
        "http://127.0.0.1:8080", transport=httpx.MockTransport(server)
    )
    with pytest.raises(httpx.HTTPStatusError):
        backend.generate(
            "gemma4-12b-it",
            [UserMessage(role="user", content="Read record #123: café")],
            request_id="failed-chat",
        )
    event = backend.events[-1]
    assert event["success"] is False
    assert event["failure"]["request_body"].encode() == received[0]
    assert event["failure"]["response_status"] == 500
    assert event["failure"]["response_body"] == "native parser failure"
    assert not backend._active
    backend.close()


def test_successful_requests_and_failed_audio_do_not_retain_request_bodies():
    def server(request):
        payload = json.loads(request.content)
        if payload["operation"] == "chat":
            return httpx.Response(
                200, json={"content": "Ready", "generatedTokens": 1, "toolCalls": []}
            )
        return httpx.Response(500, text="unavailable")

    backend = SpeechSwiftBackend(
        "http://127.0.0.1:8080", transport=httpx.MockTransport(server)
    )
    backend.generate("gemma4-12b-it", [UserMessage(role="user", content="Hello")])
    with pytest.raises(httpx.HTTPStatusError):
        backend._request("transcribe", "whisper-large-v3", audioBase64="PRIVATE AUDIO")
    assert all("failure" not in event for event in backend.events)
    backend.close()


def test_transport_failure_keeps_sent_chat_body_without_masking_exception():
    def server(request):
        raise httpx.ReadTimeout("native request timed out", request=request)

    backend = SpeechSwiftBackend(
        "http://127.0.0.1:8080", transport=httpx.MockTransport(server)
    )
    with pytest.raises(httpx.ReadTimeout):
        backend.generate("gemma4-12b-it", [UserMessage(role="user", content="Hello")])
    failure = backend.events[-1]["failure"]
    assert json.loads(failure["request_body"])["messages"][0]["content"] == "Hello"
    assert failure["response_status"] is None
    assert failure["response_body"] is None
    assert not backend._active
    backend.close()


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1",
        "http://example.com",
        "http://127.0.0.1@evil.test",
        "http://localhost?remote=1",
    ],
)
def test_transport_rejects_nonlocal_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError, match="loopback"):
        SpeechSwiftBackend(url)


@pytest.mark.parametrize("model", ["gemma4-e4b-it", "gemma4-12b-it"])
def test_chat_preserves_native_tool_ids_arguments_and_results(model):
    received = []

    def server(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": "",
                "generatedTokens": 12,
                "toolCalls": [
                    {
                        "id": "new",
                        "name": "update",
                        "arguments": {"text": "a,b}: café", "done": False},
                    }
                ],
            },
        )

    backend = SpeechSwiftBackend(
        "http://127.0.0.1:8080", transport=httpx.MockTransport(server)
    )
    reply = backend.generate(
        model,
        [
            SystemMessage(role="system", content="policy"),
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="old", name="lookup", arguments={"id": 3})],
            ),
            ToolMessage(
                role="tool", id="old", content='{"value":4}', requestor="assistant"
            ),
        ],
    )
    assert received[0]["messages"][-1]["toolCallID"] == "old"
    assert received[0]["model"] == model
    assert received[0]["messages"][-2]["toolCalls"][0]["arguments"] == {"id": 3}
    assert reply.content is None
    assert reply.tool_calls[0].arguments == {"text": "a,b}: café", "done": False}
    backend.close()


@pytest.mark.parametrize(
    "tail", ["", '{"event":"error","message":"inference failed"}\n']
)
def test_stream_failure_or_truncation_never_becomes_success(tail):
    audio = base64.b64encode(b"\x00\x01" * 512).decode()
    content = (
        json.dumps({"event": "audio", "audioBase64": audio, "sampleRate": 48000})
        + "\n"
        + tail
    )
    backend = SpeechSwiftBackend(
        "http://localhost:8080",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content)),
    )
    with pytest.raises(SpeechSwiftError):
        list(backend.stream_speech("hello", request_id="test", voice="local"))
    backend.close()


def test_spoken_fragments_form_one_participant_turn_for_native_chat():
    captured = []

    def server(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200, json={"content": "Understood", "generatedTokens": 2, "toolCalls": []}
        )

    backend = SpeechSwiftBackend(
        "http://localhost:8080", transport=httpx.MockTransport(server)
    )
    backend.generate(
        "gemma4-e4b-it",
        [
            UserMessage(role="user", content="What is the "),
            UserMessage(role="user", content="task title?"),
        ],
    )
    assert captured[0]["messages"] == [
        {"role": "user", "content": "What is the task title?"}
    ]
    backend.close()


def test_agent_never_reads_gold_text_or_speech_labels():
    class Backend:
        def vad(self, frames, stream_id, reset=False, **kwargs):
            self.received = frames
            return [0] * (len(frames) // 1024)

    backend = Backend()
    agent = SpeechSwiftAgent([], "policy", backend=backend)
    state = agent.get_init_state()
    incoming = UserMessage(
        role="user",
        content="SECRET GOLD TEXT",
        audio_script_gold="MORE SECRET",
        contains_speech=True,
        is_audio=True,
        audio_content=audio_bytes_to_string(b"\xff" * 400),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    agent._receive_audio(state, incoming)
    assert backend.received == b"\x00" * 1024
    assert len(state.messages) == 1
    assert state.messages[0].content.endswith("policy")
    assert "SECRET" not in state.messages[0].content
    assert not agent._in_speech
    agent.stop()


def test_cancelled_epoch_audio_and_tools_are_discarded():
    class Backend:
        pass

    agent = SpeechSwiftAgent([], "policy", backend=Backend())
    state = agent.get_init_state()
    agent._epoch = 2
    agent._events.put(("audio", 1, b"stale"))
    agent._events.put(
        (
            "tools",
            1,
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="stale", name="write", arguments={})],
            ),
        )
    )
    assert agent._drain_events(state) is None
    assert not agent._output
    assert len(state.messages) == 1
    agent.stop()


@pytest.mark.parametrize("asr_failure", [False, True])
def test_resumed_caller_speech_preserves_inflight_recognition(asr_failure):
    import threading

    from tau2.data_model.voice import TranscriptionResult

    entered = threading.Event()
    release = threading.Event()

    class Backend:
        cancelled = []

        def transcribe(self, audio, config, *, request_id):
            entered.set()
            assert release.wait(2)
            if asr_failure:
                raise SpeechSwiftError("ASR unavailable")
            return TranscriptionResult(transcript="My reservation is ABC123.")

        def generate(self, *args, **kwargs):
            pytest.fail("A superseded response must not be generated")

        def cancel(self, request_id):
            self.cancelled.append(request_id)

    backend = Backend()
    agent = SpeechSwiftAgent([], "policy", backend=backend)
    state = agent.get_init_state()
    try:
        agent._start_worker(list(state.messages), audio=b"\x00" * 1024)
        assert entered.wait(2)
        agent._interrupt(state)
        assert not backend.cancelled
        release.set()
        agent._future.result(timeout=2)
        if asr_failure:
            with pytest.raises(SpeechSwiftError, match="ASR unavailable"):
                agent._drain_events(state)
        else:
            agent._drain_events(state)
            assert state.messages[-1].content == "My reservation is ABC123."
            assert state.recognized_utterances == ["My reservation is ABC123."]
    finally:
        release.set()
        agent.stop()


def test_barge_in_cancels_reply_requests_and_shutdown_cancels_all_requests():
    cancelled = []

    class Backend:
        def cancel(self, request_id):
            cancelled.append(request_id)

    agent = SpeechSwiftAgent([], "policy", backend=Backend())
    agent._request_ids = {"hearing": "asr", "thinking": "chat", "speaking": "tts"}
    agent._speech_text = "Hello"
    agent._interrupt(agent.get_init_state())
    assert set(cancelled) == {"thinking", "speaking"}
    agent.stop()
    assert "hearing" in cancelled


def test_agent_rejects_frames_that_do_not_match_the_tick_clock():
    agent = SpeechSwiftAgent([], "policy", backend=object())
    incoming = UserMessage(
        role="user",
        is_audio=True,
        audio_content=audio_bytes_to_string(b"\xff" * 200),
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    with pytest.raises(ValueError, match="expected 800"):
        agent._receive_audio(agent.get_init_state(), incoming)
    agent.stop()


def test_partial_playback_never_adds_unheard_text_to_agent_history():
    agent = SpeechSwiftAgent([], "policy", backend=object())
    state = agent.get_init_state()
    agent._speech_text = "I will cancel the order after you confirm."
    agent._spoken_chars = len("I will ")
    agent._finish_spoken_history(state, interrupted=True)
    assert state.messages[-1].content == "I will "
    assert agent.metrics["interruptions"] == 1
    agent.stop()


def test_audio_frames_preserve_one_utterance_and_complete_its_text():
    agent = SpeechSwiftAgent([], "policy", backend=object())
    state = agent.get_init_state()
    text = "Your user ID, please."
    agent._speech_text = text
    agent._speech_done = True
    agent._generated_samples = 800
    agent._output.extend(b"\xff" * 800)
    first = agent._play_audio(state)
    second = agent._play_audio(state)
    assert first.utterance_ids == second.utterance_ids
    assert first.chunk_id + 1 == second.chunk_id
    assert not first.is_final_chunk and second.is_final_chunk
    assert (first.content or "") + (second.content or "") == text
    assert len(first.get_audio_bytes()) == len(second.get_audio_bytes()) == 400
    assert state.messages[-1].content == text
    agent.stop()


def test_caller_discards_a_reply_superseded_before_it_was_spoken():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from tau2.data_model.message import TurnTakingAction
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    caller = object.__new__(VoiceStreamingUserSimulator)
    caller.realtime_generation = True
    caller._generation = Future()
    caller._generation.set_result(
        (UserMessage(role="user", content="stale reply"), object())
    )
    caller._generation_input_count = 1
    caller._generation_from_tool = False
    caller._generation_started_during_speech = False
    caller.stale_generation_count = 0
    state = SimpleNamespace(
        input_turn_taking_buffer=[
            AssistantMessage(role="assistant", contains_speech=False),
            AssistantMessage(
                role="assistant", content="What is your user ID?", contains_speech=True
            ),
        ]
    )
    waiting = UserMessage(role="user", contains_speech=False)
    caller._emit_waiting_chunk = lambda current: (waiting, current)
    reply, result = caller._perform_turn_taking_action(
        state, TurnTakingAction(action="generate_message")
    )
    assert reply is waiting and result is state
    assert caller._generation is None
    assert caller.stale_generation_count == 1
    assert len(state.input_turn_taking_buffer) == 2

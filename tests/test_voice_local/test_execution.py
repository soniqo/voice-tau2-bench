import queue
import threading
from concurrent.futures import CancelledError, Future
from types import SimpleNamespace

import pytest

from tau2.agent.speech_swift_agent import SpeechSwiftAgent
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.voice import TranscriptionResult
from tau2.voice.execution import InferenceLane, SimulationClock, TimedEventQueue


class Timer:
    def __init__(self):
        self.seconds = 0

    def __call__(self):
        return self.seconds

    def add(self, seconds):
        self.seconds += seconds


def advance(clock, seconds):
    for _ in range(round(seconds / clock.tick_seconds)):
        clock.advance()


def test_completed_cancelled_reply_does_not_hold_the_agent_to_its_old_deadline():
    clock = SimulationClock()
    agent = SpeechSwiftAgent([], "policy", backend=object(), clock=clock)
    state = agent.get_init_state()
    agent._work_timeline = clock.new_work()
    agent._work_timeline.ready_at = 100
    agent._future = Future()
    agent._future.set_result(None)
    agent._cancelled.set()
    try:
        agent._drain_events(state)
        assert clock.now == 0
        assert agent._future is None
    finally:
        agent.stop()


def test_cancelled_reply_still_delivers_latency_gated_hearing_before_next_work():
    clock = SimulationClock()
    agent = SpeechSwiftAgent([], "policy", backend=object(), clock=clock)
    state = agent.get_init_state()
    agent._hearing_epochs = {0}
    hearing = clock.new_work()
    hearing.ready_at = 5
    with clock.activate(hearing):
        agent._events.put(("transcript", 0, ("Retain these heard words.", 5)))
    obsolete = clock.new_work()
    obsolete.ready_at = 100
    with clock.activate(obsolete):
        agent._events.put(("audio", 0, b"discard obsolete audio"))
    agent._work_timeline = obsolete
    agent._future = Future()
    agent._future.set_result(None)
    agent._cancelled.set()
    agent._epoch = 1
    try:
        agent._drain_events(state)
        assert agent._future is not None
        assert not state.recognized_utterances
        advance(clock, 5)
        agent._drain_events(state)
        assert state.recognized_utterances == ["Retain these heard words."]
        assert agent._future is None and not agent._output
    finally:
        agent.stop()


def test_uncancelled_completed_work_retains_its_measured_deadline():
    clock = SimulationClock()
    agent = SpeechSwiftAgent([], "policy", backend=object(), clock=clock)
    state = agent.get_init_state()
    agent._work_timeline = clock.new_work()
    agent._work_timeline.ready_at = 5
    agent._future = Future()
    agent._future.set_result(None)
    try:
        agent._drain_events(state)
        assert agent._future is not None
        advance(clock, 5)
        agent._drain_events(state)
        assert agent._future is None
    finally:
        agent.stop()


def test_worker_registers_hearing_until_the_transcript_is_delivered():
    entered = threading.Event()
    release = threading.Event()

    class Backend:
        def transcribe(self, *args, **kwargs):
            entered.set()
            assert release.wait(2)
            return TranscriptionResult(transcript="Actual heard words.")

        def generate(self, *args, **kwargs):
            return AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="read", name="lookup", arguments={})],
            )

    agent = SpeechSwiftAgent([], "policy", backend=Backend())
    state = agent.get_init_state()
    try:
        agent._start_worker(list(state.messages), audio=b"\x00" * 1024)
        assert entered.wait(2)
        assert agent._hearing_epochs == {0}
        release.set()
        agent._future.result(timeout=2)
        agent._drain_events(state)
        assert not agent._hearing_epochs
        assert state.recognized_utterances == ["Actual heard words."]
    finally:
        release.set()
        agent.stop()


@pytest.mark.parametrize("caller_seconds", [0, 100, 1000])
def test_caller_compute_does_not_advance_call_or_hide_agent_latency(caller_seconds):
    timer = Timer()
    clock = SimulationClock()

    class Backend:
        def generate(self, model, messages, **kwargs):
            timer.add(2 if model == "agent" else caller_seconds)
            return "response"

    lane = InferenceLane(Backend(), clock, timer=timer)
    work = clock.new_work()
    events = TimedEventQueue(clock)
    with clock.activate(work):
        lane.for_role("agent").generate("agent", [])
        events.put(("audio", 0, b"result"))
    lane.for_role("caller").generate("caller", [])
    assert clock.now == 0
    assert work.ready_at == 2
    advance(clock, 1.95)
    with pytest.raises(queue.Empty):
        events.get_nowait()
    clock.advance()
    assert events.get_nowait() == ("audio", 0, b"result")


def test_agent_waiting_for_caller_lane_is_measured_but_not_charged():
    timer = Timer()
    clock = SimulationClock()
    queued = threading.Event()
    work = clock.new_work()

    class Backend:
        def generate(self, *args, **kwargs):
            timer.add(3)
            return None

    def observed_timer():
        value = timer()
        queued.set()
        return value

    lane = InferenceLane(Backend(), clock, timer=observed_timer)

    def run():
        with clock.activate(work):
            lane.for_role("agent").generate("agent", [])

    lane.lock.acquire()
    thread = threading.Thread(target=run)
    try:
        thread.start()
        assert queued.wait(2)
        timer.add(100)
    finally:
        lane.lock.release()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert work.ready_at == 3
    assert lane.events[0]["queue_seconds"] == 100
    assert lane.events[0]["execution_seconds"] == 3


def test_streaming_audio_retains_first_and_subsequent_chunk_latency():
    timer = Timer()
    clock = SimulationClock()
    events = TimedEventQueue(clock)

    class Backend:
        def stream_speech(self, *args, **kwargs):
            timer.add(0.2)
            yield b"first"
            timer.add(0.3)
            yield b"second"

    lane = InferenceLane(Backend(), clock, timer=timer)
    work = clock.new_work()
    with clock.activate(work):
        for chunk in lane.for_role("agent").stream_speech("hello", request_id="tts"):
            events.put(chunk)
    advance(clock, 0.15)
    with pytest.raises(queue.Empty):
        events.get_nowait()
    clock.advance()
    assert events.get_nowait() == b"first"
    with pytest.raises(queue.Empty):
        events.get_nowait()
    advance(clock, 0.3)
    assert events.get_nowait() == b"second"
    assert work.ready_at == pytest.approx(0.5)


@pytest.mark.parametrize("interrupt", [False, True])
def test_physically_finished_agent_stays_pending_until_simulated_readiness(interrupt):
    timer = Timer()
    clock = SimulationClock()

    class Backend:
        def generate(self, *args, **kwargs):
            timer.add(2)
            return AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="write", name="save", arguments={})],
            )

    lane = InferenceLane(Backend(), clock, timer=timer)
    agent = SpeechSwiftAgent([], "policy", backend=lane.for_role("agent"), clock=clock)
    state = agent.get_init_state()
    try:
        agent._start_worker(list(state.messages))
        agent._future.result(timeout=2)
        assert agent._drain_events(state) is None
        assert agent._future is not None
        if interrupt:
            agent._interrupt(state)
        advance(clock, 2)
        reply = agent._drain_events(state)
        assert (reply is None) == interrupt
        assert agent._future is None
        assert len(state.messages) == (1 if interrupt else 2)
    finally:
        agent.stop()


def test_interrupted_asr_retains_transcript_only_after_its_measured_latency():
    timer = Timer()
    clock = SimulationClock()

    class Backend:
        def transcribe(self, *args, **kwargs):
            timer.add(1)
            return TranscriptionResult(transcript="my account")

        def generate(self, *args, **kwargs):
            return AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="read", name="lookup", arguments={})],
            )

    lane = InferenceLane(Backend(), clock, timer=timer)
    agent = SpeechSwiftAgent([], "policy", backend=lane.for_role("agent"), clock=clock)
    state = agent.get_init_state()
    try:
        agent._start_worker(list(state.messages), audio=b"audio")
        agent._future.result(timeout=2)
        agent._interrupt(state)
        agent._drain_events(state)
        assert not state.recognized_utterances
        advance(clock, 1)
        assert agent._drain_events(state) is None
        assert state.recognized_utterances == ["my account"]
        assert len(state.messages) == 2
    finally:
        agent.stop()


def test_ready_event_is_not_blocked_by_a_future_event():
    clock = SimulationClock()
    events = TimedEventQueue(clock)
    with clock.activate(SimpleNamespace(ready_at=10)):
        events.put("future")
    events.put("now")
    assert events.get_nowait() == "now"


def test_cancelled_queued_request_never_starts_native_inference():
    queued = threading.Event()
    errors = []
    cancelled = []

    class Backend:
        def generate(self, *args, **kwargs):
            pytest.fail(
                "A queued request cancelled before execution must never load a model"
            )

        def cancel(self, request_id):
            cancelled.append(request_id)

    def timer():
        queued.set()
        return 0

    lane = InferenceLane(Backend(), SimulationClock(), timer=timer)
    backend = lane.for_role("agent")

    def run():
        try:
            backend.generate("model", [], request_id="queued")
        except CancelledError as error:
            errors.append(error)

    lane.lock.acquire()
    thread = threading.Thread(target=run)
    try:
        thread.start()
        assert queued.wait(2)
        backend.cancel("queued")
    finally:
        lane.lock.release()
        thread.join(timeout=2)
    assert len(errors) == 1
    assert cancelled == ["queued"]
    assert lane.events[0]["cancelled_before_execution"] is True
    assert not backend._requests and not backend._cancelled

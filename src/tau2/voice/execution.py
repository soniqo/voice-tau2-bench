"""Simulation time and inference accounting for the opt-in local v2 profile."""

import heapq
import queue
import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class WorkTimeline:
    ready_at: float


_WORK: ContextVar[WorkTimeline | None] = ContextVar("voice_agent_work", default=None)


class SimulationClock:
    """Only committed audio ticks advance the call; inference uses real durations."""

    def __init__(self, tick_seconds=0.05):
        self.tick_seconds = tick_seconds
        self._ticks = 0
        self._lock = threading.Lock()

    @property
    def now(self):
        with self._lock:
            return self._ticks * self.tick_seconds

    def advance(self):
        with self._lock:
            self._ticks += 1

    def new_work(self):
        return WorkTimeline(self.now)

    @contextmanager
    def activate(self, work):
        token = _WORK.set(work)
        try:
            yield
        finally:
            _WORK.reset(token)


class TimedEventQueue:
    """Hold completed work until its measured agent latency elapses in call time."""

    def __init__(self, clock):
        self.clock = clock
        self._items = []
        self._sequence = 0
        self._lock = threading.Lock()

    def put(self, item):
        work = _WORK.get()
        ready_at = work.ready_at if work is not None else self.clock.now
        with self._lock:
            heapq.heappush(self._items, (ready_at, self._sequence, item))
            self._sequence += 1

    def get_nowait(self):
        with self._lock:
            if not self._items or self._items[0][0] > self.clock.now + 1e-9:
                raise queue.Empty
            return heapq.heappop(self._items)[2]


class InferenceLane:
    """One heavy operation at a time, with caller queue waits charged separately."""

    def __init__(self, backend, clock, *, timer=time.perf_counter):
        self.backend = backend
        self.clock = clock
        self.timer = timer
        self.lock = threading.Lock()
        self.events = []

    def for_role(self, role):
        return AccountedBackend(self, role)


class AccountedBackend:
    def __init__(self, lane, role):
        self.lane = lane
        self.role = role
        self._requests = set()
        self._cancelled = set()
        self._request_lock = threading.Lock()

    def __getattr__(self, name):
        # VAD and cancellation must remain usable during heavy inference.
        return getattr(self.lane.backend, name)

    def cancel(self, request_id):
        with self._request_lock:
            if request_id in self._requests:
                self._cancelled.add(request_id)
        self.lane.backend.cancel(request_id)

    @contextmanager
    def _operation(self, operation, request_id=None, call_name=None):
        with self._request_lock:
            if request_id:
                self._requests.add(request_id)
        try:
            with self._account_operation(operation, request_id, call_name) as charge:
                yield charge
        finally:
            with self._request_lock:
                self._requests.discard(request_id)
                self._cancelled.discard(request_id)

    @contextmanager
    def _account_operation(self, operation, request_id, call_name):
        lane = self.lane
        queued = lane.timer()
        with lane.lock:
            started = lane.timer()
            work = _WORK.get() if self.role == "agent" else None
            if work is not None:
                work.ready_at = max(work.ready_at, lane.clock.now)
            event = {
                "actor": "judge" if call_name == "communicate_info_eval" else self.role,
                "operation": operation,
                "request_id": request_id,
                "call_name": call_name,
                "queue_seconds": started - queued,
                "simulation_started": lane.clock.now,
                "success": False,
            }
            charged = 0.0

            def charge():
                nonlocal charged
                elapsed = lane.timer() - started
                if work is not None:
                    work.ready_at += max(0.0, elapsed - charged)
                charged = elapsed

            try:
                with self._request_lock:
                    cancelled = request_id in self._cancelled
                if cancelled:
                    event["cancelled_before_execution"] = True
                    raise CancelledError(
                        "Agent request cancelled while waiting for inference lane"
                    )
                yield charge
                event["success"] = True
            finally:
                charge()
                event["execution_seconds"] = charged
                event["simulation_ready"] = work.ready_at if work else None
                lane.events.append(event)

    def generate(self, model, messages, **kwargs):
        with self._operation("chat", kwargs.get("request_id"), kwargs.get("call_name")):
            return self.lane.backend.generate(model, messages, **kwargs)

    def synthesize(self, text, config):
        with self._operation("speech"):
            return self.lane.backend.synthesize(text, config)

    def transcribe(self, audio, config, **kwargs):
        with self._operation("asr", kwargs.get("request_id")):
            return self.lane.backend.transcribe(audio, config, **kwargs)

    def stream_speech(self, text, **kwargs):
        with self._operation("stream_speech", kwargs.get("request_id")) as charge:
            for chunk in self.lane.backend.stream_speech(text, **kwargs):
                charge()
                yield chunk

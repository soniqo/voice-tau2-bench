"""Common optional-provider bridge for the ordinary audio/tool orchestrator."""

import asyncio
import audioop
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from pydantic import Field

from tau2.agent.base.streaming import StreamingState
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage
from tau2.data_model.voice import RuntimeTTSConfig
from tau2.voice.providers.base import ProviderError, agent_instructions
from tau2.voice.speech_swift import pcm16


class ProviderState(StreamingState):
    messages: list = Field(default_factory=list)


@dataclass
class PlayingSpeech:
    id: str
    audio: bytes
    text: str
    final: bool = True
    offset: int = 0
    chars: int = 0
    opener: bool = False


class ProviderAgent(FullDuplexAgent[ProviderState]):
    def __init__(
        self,
        tools,
        domain_policy,
        *,
        settings,
        opener_backend,
        session_factory,
        seed=42,
        pace=True,
    ):
        super().__init__(tools, domain_policy)
        self.settings, self.opener_backend = settings, opener_backend
        self.session_factory, self.seed, self.pace = session_factory, seed, pace
        self.model_id = settings.model
        self._commands = queue.Queue(maxsize=120)
        self._events = queue.Queue(maxsize=64)
        self._ready = deque()
        self._calls = []
        self._pending, self._seen = set(), set()
        self._thread = self._loop = self._task = self._session = None
        self._closing, self._connected = threading.Event(), threading.Event()
        self._failure = None
        self._input_rate = self._output_rate = None
        self._tick = None
        self._opening_text = ""
        self.metrics = {
            "provider": settings.provider,
            "events": [],
            "asr": [],
            "chat": [],
            "cancellations": [],
            "interruptions": 0,
            "delivered_samples": 0,
            "discarded_samples": 0,
            "fixed_opener": "speech-swift/voxcpm2",
            "text_alignment": "proportional_provider_transcript",
        }

    def set_seed(self, seed):
        self.seed = seed

    def get_init_state(self, message_history=None):
        return ProviderState(
            messages=[
                SystemMessage(
                    role="system", content=agent_instructions(self.domain_policy)
                )
            ]
        )

    def silence(self):
        return AssistantMessage(
            role="assistant",
            contains_speech=False,
            is_audio=True,
            audio_content=audio_bytes_to_string(b"\xff" * 400),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            is_final_chunk=False,
            cost=0,
        )

    def create_initial_message(
        self, content="Hi, thanks for calling. How can I help you today?"
    ):
        if self._thread:
            raise ProviderError("Provider agent cannot be reused across trials")
        audio = self.opener_backend.synthesize(
            content,
            RuntimeTTSConfig(
                model_id="voxcpm2",
                voice_id="A clear, calm adult female voice speaking American English.",
                seed=self.seed,
            ),
        )
        self._ready.append(
            PlayingSpeech(
                uuid.uuid4().hex,
                audioop.lin2ulaw(pcm16(audio, 8000).data, 2),
                content,
                opener=True,
            )
        )
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="voice-provider"
        )
        self._thread.start()
        deadline = time.monotonic() + 25
        while not self._connected.wait(0.05):
            self.raise_if_failed()
            if time.monotonic() >= deadline:
                raise ProviderError("Provider startup did not finish")
        return self.silence()

    def command(self, kind, value):
        try:
            self._commands.put_nowait((kind, value))
        except queue.Full as error:
            raise ProviderError(
                "Provider input queue overflow; no audio was dropped"
            ) from error

    def _worker(self):
        try:
            asyncio.run(self._exchange())
        except BaseException as error:
            self._failure = error

    async def _exchange(self):
        self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
        self._session = session = self.session_factory(self.settings)
        try:
            async with asyncio.timeout(self.settings.max_seconds + 40):
                await session.connect(
                    self.get_init_state().messages[0].content, self.tools
                )
                self._connected.set()

                async def send():
                    while True:
                        try:
                            kind, value = self._commands.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.005)
                            continue
                        if kind == "audio":
                            await session.audio(value)
                        elif kind == "history":
                            await session.assistant_history(value)
                        elif kind == "tool":
                            await session.tool_result(*value)
                        elif kind == "playback":
                            await session.playback(*value)
                        elif kind == "end":
                            await session.end_input()
                            await session.close()
                            return

                async def receive():
                    while True:
                        try:
                            event = await session.receive()
                        except EOFError:
                            if not self._closing.is_set():
                                raise
                            return
                        if event.kind == "closed":
                            if not self._closing.is_set():
                                raise ProviderError(
                                    "Provider ended before benchmark closure"
                                )
                            return
                        try:
                            self._events.put_nowait(event)
                        except queue.Full as error:
                            raise ProviderError(
                                "Provider output event bound exceeded"
                            ) from error

                async with asyncio.TaskGroup() as group:
                    group.create_task(send())
                    group.create_task(receive())
        finally:
            try:
                await asyncio.wait_for(session.close(), 5)
            finally:
                self.metrics.update(
                    usage=[r.model_dump(mode="json") for r in session.records],
                    usage_finalized=session.usage_finalized,
                    metered_usd=session.metered_usd,
                )
                if hasattr(session, "configuration_sha256"):
                    self.metrics["provider_configuration_sha256"] = (
                        session.configuration_sha256
                    )

    def _process(self, event):
        self.metrics["events"].append(event.public())
        if event.kind == "speech":
            pcm, self._output_rate = audioop.ratecv(
                event.audio, 2, 1, event.sample_rate, 8000, self._output_rate
            )
            self._ready.append(
                PlayingSpeech(
                    event.id, audioop.lin2ulaw(pcm, 2), event.text, final=event.final
                )
            )
            if sum(len(s.audio) - s.offset for s in self._ready) > 8000 * 120:
                raise ProviderError("Provider playback buffer exceeded two minutes")
        elif event.kind == "transcript":
            self.metrics["asr"].append(
                {"provider": self.settings.provider, "text": event.text}
            )
        elif event.kind == "tools":
            for call in event.calls:
                if call.id in self._seen:
                    raise ProviderError("Duplicate provider function call ID")
                self._seen.add(call.id)
                self._calls.append(call)
        elif event.kind == "cancel_tools":
            self._calls = [c for c in self._calls if c.id not in event.cancelled_ids]
        elif event.kind == "interrupt":
            self.metrics["interruptions"] += 1
            self.metrics["discarded_samples"] += sum(
                len(s.audio) - s.offset for s in self._ready
            )
            self._ready.clear()
            self._output_rate = None
            if self._opening_text:
                self.command("history", self._opening_text)
                self._opening_text = ""
            for call in self._calls:
                self.command(
                    "tool",
                    (
                        call.id,
                        {
                            "error": True,
                            "execution_status": "cancelled",
                            "content": "Caller interrupted before execution; no action was taken.",
                        },
                    ),
                )
            self._calls.clear()

    def _results(self, results):
        if results is None:
            return
        rows = (
            results.tool_messages
            if isinstance(results, MultiToolMessage)
            else [results]
        )
        ids = {r.id for r in rows}
        if not ids or len(ids) != len(rows) or not ids <= self._pending:
            raise ProviderError("Provider tool results do not match emitted calls")
        for result in rows:
            self.command(
                "tool",
                (
                    result.id,
                    {
                        "content": result.content,
                        "error": result.error,
                        "execution_status": result.execution_status,
                    },
                ),
            )
        self._pending -= ids

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        if self.pace:
            now = time.monotonic()
            if self._tick is not None:
                time.sleep(max(0, self._tick - now))
            self._tick = max(self._tick or now, time.monotonic()) + 0.05
        self.raise_if_failed()
        data = b"\xff" * 400
        if participant_chunk is not None and participant_chunk.audio_content:
            fmt = participant_chunk.audio_format
            data = participant_chunk.get_audio_bytes()
            if (
                fmt is None
                or fmt.encoding != AudioEncoding.ULAW
                or fmt.sample_rate != 8000
                or fmt.channels != 1
                or len(data) != 400
            ):
                raise ValueError(
                    "Provider input must be 400-byte 8 kHz mono mu-law frames"
                )
        # The agent never reads participant text, gold scripts or speech labels.
        rate = self._session.input_rate if self._session else 24000
        pcm, self._input_rate = audioop.ratecv(
            audioop.ulaw2lin(data, 2), 2, 1, 8000, rate, self._input_rate
        )
        self.command("audio", pcm)
        self._results(tool_results)
        while True:
            try:
                self._process(self._events.get_nowait())
            except queue.Empty:
                break
        state.tick_count += 1
        if self._calls and not self._pending:
            calls, self._calls = self._calls, []
            self._pending = {c.id for c in calls}
            reply = AssistantMessage(
                role="assistant", tool_calls=calls, contains_speech=False, cost=0
            )
            state.messages.append(reply)
            return reply, state
        if not self._ready:
            return self.silence(), state
        segment = self._ready[0]
        frame = segment.audio[segment.offset : segment.offset + 400]
        segment.offset += len(frame)
        done = segment.offset >= len(segment.audio)
        chars = (
            len(segment.text)
            if done
            else int(len(segment.text) * segment.offset / len(segment.audio))
        )
        text = segment.text[segment.chars : chars]
        segment.chars = chars
        self.metrics["delivered_samples"] += len(frame)
        if segment.opener:
            self._opening_text += text
        else:
            self.command("playback", (segment.id, segment.offset / 8))
        if done:
            self._ready.popleft()
            if segment.opener:
                self.command("history", self._opening_text)
                self._opening_text = ""
        if text:
            state.messages.append(AssistantMessage(role="assistant", content=text))
        return AssistantMessage(
            role="assistant",
            content=text or None,
            contains_speech=True,
            is_audio=True,
            audio_content=audio_bytes_to_string(frame.ljust(400, b"\xff")),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            utterance_ids=[segment.id],
            is_final_chunk=done and segment.final,
            cost=0,
        ), state

    def raise_if_failed(self):
        if self._failure is not None:

            def details(error):
                if isinstance(error, ProviderError):
                    return str(error)
                children = getattr(error, "exceptions", ())
                return (
                    "; ".join(details(e) for e in children)
                    if children
                    else type(error).__name__
                )

            raise ProviderError(
                "Provider inference failed: " + details(self._failure)
            ) from self._failure

    def stop(self, participant_chunk=None, state=None, tool_results=None):
        if self._thread is None or self._closing.is_set():
            return
        self._closing.set()
        try:
            self._results(tool_results)
            self.command("end", None)
        except Exception as error:
            self._failure = error
        self._thread.join(timeout=25)
        if self._thread.is_alive():
            if self._loop and self._task:
                self._loop.call_soon_threadsafe(self._task.cancel)
            self._thread.join(timeout=6)
            self._failure = ProviderError("Provider shutdown did not complete")
        self.raise_if_failed()

    def get_usage_records(self):
        return self._session.usage_records() if self._session else []

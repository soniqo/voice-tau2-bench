"""Nova's audio-native agent behind the benchmark's ordinary audio/tool interface.

Only real caller audio reaches the provider. The fixed benchmark greeting uses
the local synthesizer; its delivered text is committed as assistant history.
"""

import asyncio
import audioop
import base64
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from pydantic import Field

from tau2.agent.base.streaming import StreamingState
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.audio import AudioEncoding, AudioFormat, audio_bytes_to_string
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
)
from tau2.data_model.usage import UsageRecord
from tau2.data_model.voice import RuntimeTTSConfig
from tau2.voice.nova import NovaSession
from tau2.voice.speech_swift import pcm16


class NovaAgentError(RuntimeError):
    pass


class NovaState(StreamingState):
    messages: list = Field(default_factory=list)


@dataclass
class Segment:
    completion: str
    data: bytearray = field(default_factory=bytearray)
    text: str | None = None
    done: bool = False
    final: bool = False
    offset: int = 0
    chars: int = 0
    queued: bool = False
    opener: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class NovaAgent(FullDuplexAgent[NovaState]):
    def __init__(
        self,
        tools,
        domain_policy,
        *,
        settings,
        opener_backend,
        seed=42,
        session_factory=NovaSession,
        pace=True,
    ):
        super().__init__(tools, domain_policy)
        self.settings, self.opener_backend = settings, opener_backend
        self.session_factory, self.pace, self.seed = session_factory, pace, seed
        self.model_id = settings.model_id
        self._commands = queue.Queue(maxsize=40)
        self._events = queue.Queue(maxsize=2048)
        self._thread = None
        self._loop = self._task = None
        self._failure = None
        self._next_tick = None
        self._session = None
        self._closing = threading.Event()
        self._connected = threading.Event()
        self._blocks = {}
        self._segments = {}
        self._ready = deque()
        self._cancelled = set()
        self._epoch = 0
        self._drop_final_until_spec = False
        self._tool_blocks = {}
        self._active_completion = None
        self._tools = []
        self._pending_ids = set()
        self._seen_ids = set()
        self._in_rate = self._out_rate = None
        self._opening_history = ""
        self.metrics = {
            "cancellations": [],
            "asr": [],
            "chat": [],
            "interruptions": 0,
            "provider": "nova",
            "input_samples": 0,
            "output_samples": 0,
            "delivered_samples": 0,
            "discarded_samples": 0,
            "text_alignment": "proportional_final_transcript",
            "fixed_opener": "speech-swift/voxcpm2",
            "events": [],
        }

    def set_seed(self, seed):
        self.seed = seed

    def get_init_state(self, message_history=None):
        return NovaState(
            messages=[
                SystemMessage(
                    role="system",
                    content=(
                        "You are the customer service agent in a live voice call. Follow the domain policy. "
                        "Speak briefly and naturally. Use your declared tools to carry out requests. "
                        "Ask the caller to perform actions on their own phone. Never claim a tool action "
                        "succeeded before its result confirms success.\n\n"
                        + self.domain_policy
                    ),
                )
            ]
        )

    def _silence(self):
        return AssistantMessage(
            role="assistant",
            contains_speech=False,
            is_audio=True,
            content=None,
            audio_content=audio_bytes_to_string(b"\xff" * 400),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            is_final_chunk=False,
            cost=0,
        )

    def create_initial_message(
        self, content="Hi, thanks for calling. How can I help you today?"
    ):
        if self._thread is not None:
            raise NovaAgentError("A Nova agent cannot be reused across trials")
        audio = self.opener_backend.synthesize(
            content,
            RuntimeTTSConfig(
                model_id="voxcpm2",
                voice_id="A clear, calm adult female voice speaking American English.",
                seed=self.seed,
            ),
        )
        data = audioop.lin2ulaw(pcm16(audio, 8000).data, 2)
        opening = Segment("opener", bytearray(data), content, True, True, opener=True)
        self._ready.append(opening)
        self._thread = threading.Thread(
            target=self._worker, name="nova-audio", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self._connected.wait(0.05):
            self.raise_if_failed(final=False)
            if time.monotonic() >= deadline:
                raise NovaAgentError("Nova connection did not become ready")
        return self._silence()

    def _command(self, name, value):
        try:
            self._commands.put_nowait((name, value))
        except queue.Full as error:
            raise NovaAgentError(
                "Nova input queue exceeded two seconds; refusing to drop audio"
            ) from error

    def _worker(self):
        try:
            asyncio.run(self._exchange())
        except BaseException as error:
            self._failure = error

    async def _exchange(self):
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._session = session = self.session_factory(self.settings)
        try:
            async with asyncio.timeout(self.settings.max_seconds):
                await session.connect(
                    self.get_init_state().messages[0].content, self.tools
                )

                self._connected.set()

                async def send():
                    while True:
                        try:
                            name, value = self._commands.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.005)
                            continue
                        if name == "audio":
                            await session.audio(value)
                        elif name == "tool":
                            await session.tool_result(*value)
                        elif name == "history":
                            await session.assistant_history(value)
                        elif name == "end":
                            await session.end_input()
                            return
                        else:
                            raise NovaAgentError("Unknown Nova input command")

                async def receive():
                    while True:
                        try:
                            event = await session.receive()
                        except EOFError:
                            if not self._closing.is_set():
                                raise
                            session.usage_finalized = True
                            return
                        try:
                            self._events.put_nowait(event)
                        except queue.Full as error:
                            raise NovaAgentError(
                                "Nova output queue exceeded its bound"
                            ) from error

                async with asyncio.TaskGroup() as group:
                    group.create_task(send())
                    group.create_task(receive())
        finally:
            try:
                await asyncio.wait_for(session.close(), timeout=5)
            finally:
                self.metrics["usage"] = session.usage.totals
                self.metrics["usage_finalized"] = session.usage_finalized
                self.metrics["metered_usd"] = (
                    float(session.usage.usd) if session.usage.usd is not None else None
                )

    def _interrupt(self):
        active = bool(
            self._ready
            or self._segments
            or self._tools
            or self._pending_ids
            or any(b.get("role") == "ASSISTANT" for b in self._blocks.values())
        )
        if active:
            self._epoch += 1
            self._drop_final_until_spec = True
        # completionId can span the whole connection. Cancel content IDs only.
        self._cancelled.update(self._segments)
        self._cancelled.update(
            k for k, b in self._blocks.items() if b.get("role") in {"ASSISTANT", "TOOL"}
        )
        self._cancelled.update(
            self._tool_blocks[c.id] for c in self._tools if c.id in self._tool_blocks
        )
        dropped = sum(len(s.data) - s.offset for s in self._segments.values())
        dropped += sum(len(s.data) - s.offset for s in self._ready if s.opener)
        self.metrics["discarded_samples"] += dropped
        self.metrics["interruptions"] += int(active)
        if self._opening_history:
            self._command("history", self._opening_history)
            self._opening_history = ""
        self._segments.clear()
        self._blocks = {
            k: v for k, v in self._blocks.items() if k not in self._cancelled
        }
        self._ready.clear()
        for call in self._tools:
            self._command(
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
        self._tools.clear()
        self._out_rate = None

    def _process(self, event):
        saved = json.loads(json.dumps(event))
        if "audioOutput" in saved:
            saved["audioOutput"]["content"] = {
                "base64_bytes": len(saved["audioOutput"]["content"])
            }
        self.metrics["events"].append(saved)
        if any(
            isinstance(value, dict) and value.get("contentId") in self._cancelled
            for key, value in event.items()
            if key != "usageEvent"
        ):
            return
        if "userSpeechStart" in event:
            self._interrupt()
        if "contentStart" in event:
            start = event["contentStart"]
            fields = start.get("additionalModelFields", "{}")
            block = {
                **start,
                "stage": json.loads(fields).get("generationStage"),
                "text": "",
                "epoch": self._epoch,
            }
            if block.get("role") == "ASSISTANT" and block["stage"] == "SPECULATIVE":
                self._drop_final_until_spec = False
            self._blocks[start["contentId"]] = block
            if start.get("role") == "ASSISTANT":
                self._active_completion = start["completionId"]
            if start.get("type") == "AUDIO":
                config = start.get("audioOutputConfiguration", {})
                if (
                    config.get("sampleRateHertz"),
                    config.get("sampleSizeBits"),
                    config.get("channelCount"),
                ) != (24000, 16, 1):
                    raise NovaAgentError("Unexpected Nova audio format")
                self._segments[start["contentId"]] = Segment(
                    start["completionId"], id=start["contentId"]
                )
        if "audioOutput" in event:
            value = event["audioOutput"]
            if value["contentId"] in self._cancelled:
                return
            segment = self._segments[value["contentId"]]
            pcm = base64.b64decode(value["content"], validate=True)
            if len(pcm) % 2:
                raise NovaAgentError("Unaligned Nova PCM16 audio")
            pcm, self._out_rate = audioop.ratecv(pcm, 2, 1, 24000, 8000, self._out_rate)
            data = audioop.lin2ulaw(pcm, 2)
            segment.data.extend(data)
            self.metrics["output_samples"] += len(data)
            if sum(len(s.data) - s.offset for s in self._segments.values()) > 8000 * 90:
                raise NovaAgentError(
                    "Nova generated more than 90 seconds of buffered speech"
                )
        if "textOutput" in event:
            value = event["textOutput"]
            text = value.get("content", "")
            notice = None
            if text.lstrip().startswith("{"):
                try:
                    notice = json.loads(text)
                except ValueError:
                    pass
            if isinstance(notice, dict) and notice.get("interrupted") is True:
                self._blocks.pop(value["contentId"], None)
                self._interrupt()
                return
            self._blocks[value["contentId"]]["text"] += text
        if "toolUse" in event:
            value = event["toolUse"]
            self._active_completion = value.get("completionId")
            self._blocks[value["contentId"]]["call"] = value
        if "contentEnd" in event:
            end = event["contentEnd"]
            block = self._blocks.pop(end["contentId"], {})
            if end.get("stopReason") == "INTERRUPTED":
                self._interrupt()
                return
            if end["contentId"] in self._cancelled:
                return
            if block.get("call"):
                value = block["call"]
                call = ToolCall(
                    id=value["toolUseId"],
                    name=value["toolName"],
                    arguments=json.loads(value["content"]),
                )
                if call.id in self._seen_ids:
                    raise NovaAgentError("Duplicate provider tool call ID")
                self._seen_ids.add(call.id)
                self._tool_blocks[call.id] = end["contentId"]
                self._tools.append(call)
            if block.get("type") == "AUDIO":
                segment = self._segments[end["contentId"]]
                segment.done = True
                segment.final = end.get("stopReason") == "END_TURN"
            if block.get("stage") == "FINAL" and block.get("type") == "TEXT":
                if block.get("role") == "USER":
                    self.metrics["asr"].append(
                        {"text": block["text"], "provider": "nova"}
                    )
                elif block.get("role") == "ASSISTANT":
                    if self._drop_final_until_spec:
                        return
                    matching = next(
                        (
                            s
                            for s in self._segments.values()
                            if s.completion == block["completionId"] and s.text is None
                        ),
                        None,
                    )
                    if matching is None:
                        raise NovaAgentError(
                            "Nova final transcript has no corresponding audio"
                        )
                    matching.text = block["text"]
            for segment in self._segments.values():
                if segment.done and segment.text is not None and not segment.queued:
                    if not segment.data:
                        raise NovaAgentError("Nova transcript has no speech samples")
                    segment.queued = True
                    self._ready.append(segment)

    def _results(self, results):
        if results is None:
            return
        rows = (
            results.tool_messages
            if isinstance(results, MultiToolMessage)
            else [results]
        )
        ids = {r.id for r in rows}
        if not ids or not ids <= self._pending_ids:
            raise NovaAgentError("Tool results do not match emitted Nova calls")
        for result in rows:
            self._command(
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
        self._pending_ids -= ids

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        if self.pace:
            now = time.monotonic()
            if self._next_tick is not None:
                time.sleep(max(0, self._next_tick - now))
            self._next_tick = max(self._next_tick or now, time.monotonic()) + 0.05
        self.raise_if_failed(final=False)
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
                    "Nova benchmark input must be 400-byte 8 kHz mono mu-law frames"
                )
        # Never inspect participant text, scripts or speech labels.
        pcm, self._in_rate = audioop.ratecv(
            audioop.ulaw2lin(data, 2), 2, 1, 8000, 16000, self._in_rate
        )
        self.metrics["input_samples"] += len(data)
        self._command("audio", pcm)
        self._results(tool_results)
        while True:
            try:
                self._process(self._events.get_nowait())
            except queue.Empty:
                break
        state.tick_count += 1
        if self._tools and not self._pending_ids:
            calls, self._tools = self._tools, []
            self._pending_ids = {call.id for call in calls}
            message = AssistantMessage(
                role="assistant", tool_calls=calls, contains_speech=False, cost=0
            )
            state.messages.append(message)
            return message, state
        if not self._ready:
            return self._silence(), state
        segment = self._ready[0]
        frame = bytes(segment.data[segment.offset : segment.offset + 400])
        segment.offset += len(frame)
        complete = segment.offset >= len(segment.data)
        chars = (
            len(segment.text)
            if complete
            else int(len(segment.text) * segment.offset / len(segment.data))
        )
        text = segment.text[segment.chars : chars]
        segment.chars = chars
        self.metrics["delivered_samples"] += len(frame)
        if segment.opener:
            self._opening_history += text
        if complete:
            self._ready.popleft()
            self._segments.pop(segment.id, None)
            if segment.opener:
                self._command("history", self._opening_history)
                self._opening_history = ""
        message = AssistantMessage(
            role="assistant",
            content=text or None,
            contains_speech=True,
            is_audio=True,
            audio_content=audio_bytes_to_string(frame.ljust(400, b"\xff")),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            is_final_chunk=complete and segment.final,
            utterance_ids=[segment.id],
            cost=0,
        )
        if text:
            state.messages.append(AssistantMessage(role="assistant", content=text))
        return message, state

    def raise_if_failed(self, *, final=True):
        if self._failure is not None:
            raise NovaAgentError(
                "Nova inference failed: " + type(self._failure).__name__
            ) from self._failure
        if final and (
            self._session is None
            or not self._session.usage.events
            or not self._session.usage_finalized
        ):
            raise NovaAgentError(
                "Nova final usage was not confirmed; trial is unscored"
            )

    def stop(self, participant_chunk=None, state=None, tool_results=None):
        if self._thread is None or self._closing.is_set():
            return
        self._closing.set()
        try:
            self._results(tool_results)
            self._command("end", None)
        except Exception as error:
            self._failure = error
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)
                self._thread.join(timeout=6)
            self._failure = NovaAgentError("Nova shutdown did not finish")
        self.raise_if_failed()

    def get_usage_records(self):
        if self._session is None or not self._session.usage.events:
            return []
        totals = self._session.usage.totals
        return [
            UsageRecord(
                provider="nova",
                model=self.model_id,
                semantics="cumulative",
                scope_id=self._session.prompt,
                input_audio_tokens=totals["input"]["speechTokens"],
                input_text_tokens=totals["input"]["textTokens"],
                output_audio_tokens=totals["output"]["speechTokens"],
                output_text_tokens=totals["output"]["textTokens"],
            )
        ]

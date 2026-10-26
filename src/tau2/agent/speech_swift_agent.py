"""Local ASR → tool-capable chat → streaming TTS full-duplex agent."""

import audioop
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from loguru import logger
from pydantic import Field

from tau2.agent.base.streaming import StreamingState
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.audio import (
    AudioData,
    AudioEncoding,
    AudioFormat,
    audio_bytes_to_string,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.data_model.voice import RuntimeTranscriptionConfig
from tau2.voice.execution import TimedEventQueue
from tau2.voice.speech_swift import SpeechSwiftBackend, pcm16
from tau2.voice.synthesis.audio_effects.effects import StreamingTelephonyConverter


class CascadeState(StreamingState):
    messages: list[Message] = Field(default_factory=list)
    recognized_utterances: list[str] = Field(default_factory=list)


class SpeechSwiftAgent(FullDuplexAgent[CascadeState]):
    def __init__(
        self,
        tools,
        domain_policy,
        *,
        backend: SpeechSwiftBackend,
        model_id="gemma4-e4b-it",
        seed=42,
        tick_seconds=0.05,
        voice="A clear, calm adult female voice speaking American English.",
        clock=None,
        tool_guidance=False,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.backend = backend
        self.model_id = model_id
        self.tick_seconds = tick_seconds
        self.voice = voice
        self.seed = seed
        self.clock = clock
        self.tool_guidance = tool_guidance
        self._work_timeline = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="local-agent"
        )
        self._events = TimedEventQueue(clock) if clock else queue.Queue()
        self._future = None
        self._cancelled = threading.Event()
        self._epoch = 0
        self._request_ids = {}
        self._request_lock = threading.Lock()
        self._pending_audio = []
        self._hearing_epochs = set()
        self._need_reply = False
        self._awaiting_tools = set()
        self._transfer_pending = False
        self._output = bytearray()
        self._speech_text = ""
        self._spoken_chars = 0
        self._played_samples = 0
        self._generated_samples = 0
        self._speech_done = False
        self._chunk_id = 0
        self._utterance_id = uuid.uuid4().hex
        self._vad_id = uuid.uuid4().hex
        self._vad_reset = True
        self._vad_buffer = bytearray()
        self._resample_state = None
        self._utterance = bytearray()
        self._preroll = bytearray()
        self._speech_frames = 0
        self._silent_frames = 0
        self._in_speech = False
        self._next_tick = None
        self.metrics = {
            "interruptions": 0,
            "asr": [],
            "chat": [],
            "cancellations": [],
            "text_alignment": "estimated_until_tts_complete",
        }

    def set_seed(self, seed):
        self.seed = seed

    def get_init_state(self, message_history=None):
        instructions = (
            "You are the customer service agent in a live voice call. Follow the domain policy. "
            "Speak in short natural sentences, without markdown or lists. Keep each spoken reply under 30 words. "
            "Use the available tools to carry out requests. "
            "Convert explicitly spoken punctuation and digit names into their written characters in identifier arguments.\n\n"
            + self.domain_policy
        )
        if self.tool_guidance:
            instructions += (
                "\n\nOnly call tools declared for you. Ask the caller to perform actions "
                "on their phone or device. After a lookup error, check a successful "
                "customer/profile lookup for the exact identifier or ask the caller; "
                "do not invent identifiers or repeat an unchanged failing call. "
                "Never claim an action succeeded until its tool result confirms it."
            )
        return CascadeState(
            messages=[SystemMessage(role="system", content=instructions)]
        )

    def _silence(self):
        return AssistantMessage(
            role="assistant",
            content=None,
            contains_speech=False,
            cost=0.0,
            is_audio=False,
            audio_content=audio_bytes_to_string(
                b"\xff" * round(8000 * self.tick_seconds)
            ),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
        )

    def create_initial_message(self, content="Hi! How can I help you today?"):
        # Start the opener asynchronously; tick zero remains a real silent frame.
        self._speech_text = content
        self._start_worker([], initial_text=content)
        return self._silence()

    def _request(self, operation, callback):
        request_id = uuid.uuid4().hex
        with self._request_lock:
            self._request_ids[request_id] = operation
        try:
            return callback(request_id)
        finally:
            with self._request_lock:
                self._request_ids.pop(request_id, None)

    def _start_worker(self, history, audio=None, initial_text=None):
        epoch = self._epoch
        cancelled = self._cancelled = threading.Event()
        if audio is not None:
            self._hearing_epochs.add(epoch)

        def work():
            if audio is not None:
                started = time.perf_counter()
                try:
                    result = self._request(
                        "asr",
                        lambda rid: self.backend.transcribe(
                            AudioData(
                                data=audio,
                                format=AudioFormat(
                                    encoding=AudioEncoding.PCM_S16LE, sample_rate=16000
                                ),
                            ),
                            RuntimeTranscriptionConfig(model="whisper-large-v3"),
                            request_id=rid,
                        ),
                    )
                except Exception as error:
                    self._events.put(("asr_error", epoch, error))
                    return
                self._events.put(
                    (
                        "transcript",
                        epoch,
                        (result.transcript, time.perf_counter() - started),
                    )
                )
                if result.transcript.strip():
                    history.append(UserMessage(role="user", content=result.transcript))
                else:
                    return
            if cancelled.is_set():
                return
            if initial_text is None:
                reply = self._request(
                    "chat",
                    lambda rid: self.backend.generate(
                        self.model_id,
                        history,
                        tools=self.tools,
                        request_id=rid,
                        temperature=0,
                        max_tokens=1024,
                    ),
                )
                self.metrics["chat"].append(
                    {
                        "epoch": epoch,
                        "completion_tokens": (reply.usage or {}).get(
                            "completion_tokens"
                        ),
                        "text_characters": len(reply.content or ""),
                        "text_words": len((reply.content or "").split()),
                        "tool_calls": len(reply.tool_calls or []),
                    }
                )
                if cancelled.is_set():
                    return
                if reply.is_tool_call():
                    self._events.put(("tools", epoch, reply))
                    return
                text = reply.content or ""
            else:
                text = initial_text
            if not text.strip():
                raise ValueError("Native agent produced an empty response")
            self._events.put(("speech_start", epoch, text))
            converter = StreamingTelephonyConverter(input_sample_rate=48000)

            def stream(rid):
                for chunk in self.backend.stream_speech(
                    text, request_id=rid, voice=self.voice, seed=self.seed
                ):
                    if cancelled.is_set():
                        return
                    self._events.put(
                        ("audio", epoch, converter.convert_chunk(chunk).data)
                    )

            self._request("tts", stream)
            if not cancelled.is_set():
                self._events.put(("speech_done", epoch, None))

        if self.clock is None:
            run = work
        else:
            timeline = self._work_timeline = self.clock.new_work()

            def run():
                with self.clock.activate(timeline):
                    work()

        self._future = self._executor.submit(copy_context().run, run)

    def _finish_spoken_history(self, state, *, interrupted=False):
        heard = self._speech_text[: self._spoken_chars]
        was_audible = bool(heard or self._played_samples)
        if heard:
            state.messages.append(AssistantMessage(role="assistant", content=heard))
        self._speech_text = ""
        self._spoken_chars = 0
        self._played_samples = 0
        self._generated_samples = 0
        self._speech_done = False
        if interrupted and was_audible:
            self.metrics["interruptions"] += 1

    def _interrupt(self, state):
        if self._future is None and not self._speech_text:
            return
        self._cancelled.set()
        with self._request_lock:
            # Hearing the caller must finish even when a newer speech segment
            # supersedes the reply. Its transcript is retained across epochs.
            requests = [
                rid
                for rid, operation in self._request_ids.items()
                if operation != "asr"
            ]
        for rid in requests:
            self.metrics["cancellations"].append(
                {"request_id": rid, "reason": "barge_in"}
            )
            self.backend.cancel(rid)
        self._epoch += 1
        self._output.clear()
        self._finish_spoken_history(state, interrupted=True)

    def _receive_audio(self, state, participant_chunk):
        # Read only the audio and format. In particular, content, gold scripts,
        # and contains_speech are never exposed to the evaluated agent.
        if participant_chunk is not None and participant_chunk.is_audio:
            raw = pcm16(
                AudioData(
                    data=participant_chunk.get_audio_bytes(),
                    format=participant_chunk.audio_format,
                )
            )
            expected_bytes = round(raw.format.sample_rate * self.tick_seconds) * 2
            if len(raw.data) != expected_bytes:
                raise ValueError(
                    f"Caller audio frame has {len(raw.data)} PCM bytes; expected {expected_bytes} for one tick"
                )
            normalized, self._resample_state = audioop.ratecv(
                raw.data, 2, 1, raw.format.sample_rate, 16000, self._resample_state
            )
        else:
            normalized = b"\x00" * round(16000 * self.tick_seconds) * 2
        self._vad_buffer.extend(normalized)
        length = len(self._vad_buffer) // 1024 * 1024
        if not length:
            return
        frames = bytes(self._vad_buffer[:length])
        del self._vad_buffer[:length]
        probabilities = self.backend.vad(frames, self._vad_id, reset=self._vad_reset)
        self._vad_reset = False
        for index, probability in enumerate(probabilities):
            frame = frames[index * 1024 : (index + 1) * 1024]
            if probability >= 0.5:
                if not self._in_speech:
                    self._interrupt(state)
                    self._utterance = self._preroll.copy()
                    self._speech_frames = 0
                    self._in_speech = True
                self._speech_frames += 1
                self._silent_frames = 0
            elif self._in_speech:
                self._silent_frames += 1
            if self._in_speech:
                self._utterance.extend(frame)
                if self._silent_frames >= 13 or len(self._utterance) >= 30 * 16000 * 2:
                    if self._speech_frames >= 4:
                        self._pending_audio.append(bytes(self._utterance))
                    self._utterance.clear()
                    self._in_speech = False
                    self._silent_frames = 0
            self._preroll.extend(frame)
            del self._preroll[: -6 * 1024]

    def _drain_events(self, state):
        tool_message = None
        while True:
            try:
                kind, epoch, value = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "asr_error":
                self._hearing_epochs.discard(epoch)
                raise value
            if kind == "transcript":
                self._hearing_epochs.discard(epoch)
                text, elapsed = value
                self.metrics["asr"].append({"text": text, "wall_seconds": elapsed})
                logger.info("Local agent heard: {}", text)
                if text.strip():
                    state.messages.append(UserMessage(role="user", content=text))
                    state.recognized_utterances.append(text)
                continue
            if epoch != self._epoch:
                continue
            if kind == "speech_start":
                self._speech_text = value
                logger.info("Local agent response: {}", value)
                self._chunk_id = 0
                self._utterance_id = uuid.uuid4().hex
            elif kind == "audio":
                self._output.extend(value)
                self._generated_samples += len(value)
            elif kind == "speech_done":
                self._speech_done = True
            elif kind == "tools":
                value = value.model_copy(update={"contains_speech": False})
                state.messages.append(value)
                tool_message = value
                self._transfer_pending = any(
                    call.name == "transfer_to_human_agents" for call in value.tool_calls
                )
        work_ready = (
            self.clock is None
            or self._work_timeline is None
            or self.clock.now + 1e-9 >= self._work_timeline.ready_at
        )
        # Obsolete replies must not reserve their old simulated completion time.
        # Physical work still has to finish, and preserved hearing must reach
        # its own deadline before another response starts.
        if self._cancelled.is_set() and not self._hearing_epochs:
            work_ready = True
        if self._future is not None and self._future.done() and work_ready:
            try:
                self._future.result()
            except Exception:
                if not self._cancelled.is_set():
                    raise
            self._future = None
        return tool_message

    def _play_audio(self, state):
        count = round(8000 * self.tick_seconds)
        if len(self._output) < count and not self._speech_done:
            return self._silence()
        if not self._output:
            return self._silence()
        audio = bytes(self._output[:count])
        del self._output[:count]
        self._played_samples += len(audio)
        final = self._speech_done and not self._output
        duration = (
            self._generated_samples / 8000
            if self._speech_done
            else max(1, len(self._speech_text) / 15)
        )
        fraction = min(
            1 if self._speech_done else 0.85, self._played_samples / 8000 / duration
        )
        end = (
            len(self._speech_text) if final else int(len(self._speech_text) * fraction)
        )
        if not final:
            space = self._speech_text.rfind(" ", self._spoken_chars, end)
            end = max(
                self._spoken_chars, space + 1 if space >= 0 else self._spoken_chars
            )
        content = self._speech_text[self._spoken_chars : end]
        self._spoken_chars = end
        message = AssistantMessage(
            role="assistant",
            content=content or None,
            contains_speech=True,
            cost=0.0,
            is_audio=False,
            audio_content=audio_bytes_to_string(audio.ljust(count, b"\xff")),
            audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
            chunk_id=self._chunk_id,
            utterance_ids=[self._utterance_id],
            is_final_chunk=final,
        )
        self._chunk_id += 1
        if final:
            self._finish_spoken_history(state)
        return message

    def get_next_chunk(self, state, participant_chunk=None, tool_results=None):
        now = time.perf_counter()
        if self._next_tick is not None and self._next_tick > now:
            time.sleep(self._next_tick - now)
        self._next_tick = (
            max(self._next_tick or now, time.perf_counter()) + self.tick_seconds
        )
        self._receive_audio(state, participant_chunk)
        tools = self._drain_events(state)
        if tool_results is not None:
            results = (
                tool_results.tool_messages
                if isinstance(tool_results, MultiToolMessage)
                else [tool_results]
            )
            if (
                self._awaiting_tools
                and {result.id for result in results} != self._awaiting_tools
            ):
                raise ValueError(
                    "Agent received tool results that do not match its pending calls"
                )
            self._awaiting_tools.clear()
            state.messages.extend(results)
            self._need_reply = True
            if self._transfer_pending:
                return AssistantMessage(
                    role="assistant", content="###STOP###", contains_speech=False
                ), state
        if tools is not None:
            self._awaiting_tools = {call.id for call in tools.tool_calls}
            state.tick_count += 1
            return tools, state
        if (
            self._future is None
            and not self._speech_text
            and not self._in_speech
            and not self._awaiting_tools
        ):
            if self._pending_audio:
                audio = self._pending_audio.pop(0)
                self._need_reply = False
                self._start_worker(list(state.messages), audio=audio)
            elif self._need_reply:
                self._need_reply = False
                self._start_worker(list(state.messages))
        state.tick_count += 1
        return tools or self._play_audio(state), state

    @classmethod
    def is_stop(cls, message):
        return message.content == "###STOP###"

    def stop(self, participant_chunk=None, state=None, tool_results=None):
        self.metrics["shutdown_state"] = {
            "pending_audio": len(self._pending_audio),
            "pending_hearing_epochs": sorted(self._hearing_epochs),
            "future_done": self._future.done() if self._future is not None else None,
            "reply_cancelled": self._cancelled.is_set(),
            "ready_at": self._work_timeline.ready_at if self._work_timeline else None,
            "simulation_now": self.clock.now if self.clock else None,
            "remaining_audio_samples": len(self._output),
        }
        self._cancelled.set()
        with self._request_lock:
            requests = list(self._request_ids)
        for rid in requests:
            self.metrics["cancellations"].append(
                {"request_id": rid, "reason": "shutdown"}
            )
            self.backend.cancel(rid)
        self._executor.shutdown(wait=True, cancel_futures=True)
        if not self._vad_reset:
            self.backend.vad(b"\x00" * 1024, self._vad_id, end_stream=True)
            self._vad_reset = True

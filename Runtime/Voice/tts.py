"""Engine-agnostic streaming speech output for Dexter.

This module deliberately does not import a TTS model.  It owns the pieces that
stay the same whichever synthesizer Dexter uses:

* streamed agent text -> sentence-sized speech units;
* deterministic cleanup of Markdown/code/URLs for listening;
* one synthesis worker and one playback worker so generation can stay ahead of
  playback without blocking Qt/Goose;
* bounded queues;
* interruption/cancellation semantics; and
* a small backend contract implemented later by Pocket, Kokoro, etc.

Backends return float32 PCM audio.  The optional SoundDeviceAudioSink uses the
same ``sounddevice``/NumPy stack already present for Dexter's Moonshine input.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)

_SENTENCE_END_RE = re.compile(r'''[.!?](?:["')\]]*)\s+''')
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]+|\([^)]*\))*\)")
_RAW_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_QUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s?")
_LIST_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_WHITESPACE_RE = re.compile(r"\s+")

_STOP = object()


class VoiceOutputState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    IDLE = "idle"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One engine-independent PCM chunk.

    ``samples`` may be mono ``(frames,)`` or interleaved ``(frames, channels)``.
    Backends should provide floating point samples in the conventional
    ``[-1.0, 1.0]`` range.
    """

    samples: np.ndarray
    sample_rate: int


class SpeechBackend(Protocol):
    """Minimal adapter implemented by a concrete TTS engine."""

    def start(self) -> None:
        """Load the model and any reusable voice state."""

    def synthesize(
        self,
        text: str,
        cancel_event: threading.Event,
    ) -> Iterable[AudioChunk]:
        """Yield PCM chunks for one already-cleaned speech unit."""

    def stop(self) -> None:
        """Release model resources."""


class AudioSink(Protocol):
    """Playback boundary, separated so the core is easy to test."""

    def start(self) -> None: ...

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None: ...

    def interrupt(self) -> None: ...

    def stop(self) -> None: ...


class SpeechTextBuffer:
    """Turn arbitrary streamed text fragments into bounded speech units.

    Sentence punctuation is preferred.  Long punctuation-free output is split
    on a newline/whitespace boundary so a pathological model response cannot
    delay speech indefinitely.
    """

    def __init__(self, *, max_chars: int = 320) -> None:
        if max_chars < 80:
            raise ValueError("max_chars must be at least 80")
        self.max_chars = max_chars
        self._buffer = ""

    def reset(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        ready: list[str] = []

        while True:
            match = _SENTENCE_END_RE.search(self._buffer)
            if match is not None:
                end = match.end()
                ready.append(self._buffer[:end].strip())
                self._buffer = self._buffer[end:]
                continue

            if len(self._buffer) <= self.max_chars:
                break

            split_at = self._soft_split_index(self._buffer, self.max_chars)
            ready.append(self._buffer[:split_at].strip())
            self._buffer = self._buffer[split_at:].lstrip()

        return [item for item in ready if item]

    def flush(self) -> list[str]:
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []

    @staticmethod
    def _soft_split_index(text: str, limit: int) -> int:
        window = text[: limit + 1]
        newline = window.rfind("\n")
        if newline >= limit // 2:
            return newline + 1

        whitespace = max(window.rfind(" "), window.rfind("\t"))
        if whitespace >= limit // 2:
            return whitespace + 1

        return limit


class SpeechFormatter:
    """Deterministically make visible assistant text less awful to hear.

    It is intentionally conservative: no LLM rewrite and no invented summary.
    Fenced code is omitted, inline code keeps its words, link labels are kept,
    and raw URLs are replaced by the word ``link``.
    """

    def __init__(self) -> None:
        self._in_code_fence = False

    def reset(self) -> None:
        self._in_code_fence = False

    def format(self, text: str) -> str:
        if not text:
            return ""

        visible: list[str] = []
        parts = text.split("```")
        for index, part in enumerate(parts):
            if not self._in_code_fence:
                visible.append(part)
            if index != len(parts) - 1:
                self._in_code_fence = not self._in_code_fence

        value = " ".join(visible)
        value = _MARKDOWN_LINK_RE.sub(r"\1", value)
        value = _RAW_URL_RE.sub("link", value)
        value = _HEADING_RE.sub("", value)
        value = _QUOTE_RE.sub("", value)
        value = _LIST_RE.sub("", value)
        value = value.replace("`", "")
        value = value.replace("**", "").replace("__", "")
        value = _WHITESPACE_RE.sub(" ", value)
        return value.strip()


@dataclass(frozen=True, slots=True)
class _SpeechUnit:
    response_id: int
    text: str


@dataclass(frozen=True, slots=True)
class _QueuedAudio:
    response_id: int
    chunk: AudioChunk
    cancel_event: threading.Event


class SoundDeviceAudioSink:
    """Low-overhead local playback using Dexter's existing audio dependency."""

    def __init__(self, *, device: int | str | None = None) -> None:
        self.device = device
        self._lock = threading.Lock()
        self._stream = None
        self._sample_rate: int | None = None
        self._channels: int | None = None

    def start(self) -> None:
        # Output is opened lazily when the backend reveals its sample rate.
        pass

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return

        import sounddevice as sd

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[0] == 0:
            return

        channels = int(samples.shape[1])
        with self._lock:
            if cancel_event.is_set():
                return
            if (
                self._stream is None
                or self._sample_rate != chunk.sample_rate
                or self._channels != channels
            ):
                self._close_locked()
                self._stream = sd.OutputStream(
                    samplerate=chunk.sample_rate,
                    channels=channels,
                    dtype="float32",
                    device=self.device,
                )
                self._stream.start()
                self._sample_rate = chunk.sample_rate
                self._channels = channels
            stream = self._stream

        if cancel_event.is_set():
            return
        try:
            stream.write(samples)
        except Exception:
            if not cancel_event.is_set():
                raise

    def interrupt(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._sample_rate = None
            self._channels = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                LOGGER.debug("Audio stream abort failed during interrupt", exc_info=True)
            try:
                stream.close()
            except Exception:
                LOGGER.debug("Audio stream close failed during interrupt", exc_info=True)

    def stop(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        stream = self._stream
        self._stream = None
        self._sample_rate = None
        self._channels = None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()


class VoiceOutput:
    """Non-blocking, engine-independent speech coordinator.

    Public methods are cheap enough for Qt callbacks.  Synthesis and playback
    happen on separate workers, allowing sentence N+1 to synthesize while
    sentence N is still playing. If several complete sentence units are already
    queued when a synthesis request finishes, a small number are coalesced into
    the next request. This preserves first-sentence latency while avoiding a new
    NeuTTS prompt-prefill/TTFA penalty at every sentence boundary.
    """

    def __init__(
        self,
        backend: SpeechBackend,
        *,
        sink: AudioSink | None = None,
        max_text_queue: int = 16,
        max_audio_queue: int = 4,
        max_buffer_chars: int = 320,
        max_coalesced_units: int = 3,
    ) -> None:
        if max_coalesced_units < 1:
            raise ValueError("max_coalesced_units must be positive")
        self.backend = backend
        self.sink = sink or SoundDeviceAudioSink()
        self.buffer = SpeechTextBuffer(max_chars=max_buffer_chars)
        self.formatter = SpeechFormatter()
        self.max_coalesced_units = int(max_coalesced_units)

        self._text_queue: queue.Queue[object] = queue.Queue(maxsize=max_text_queue)
        self._audio_queue: queue.Queue[object] = queue.Queue(maxsize=max_audio_queue)
        self._lock = threading.Lock()
        self._response_id = 0
        self._active_response_id: int | None = None
        self._cancel_event = threading.Event()
        self._state = VoiceOutputState.STOPPED
        self._synthesis_thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: BaseException | None = None

    @property
    def state(self) -> VoiceOutputState:
        with self._lock:
            return self._state

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def start(self) -> None:
        with self._lock:
            if self._state is not VoiceOutputState.STOPPED:
                raise RuntimeError("VoiceOutput is already running")
            self._state = VoiceOutputState.STARTING
            self._error = None

        try:
            self.backend.start()
            self.sink.start()
        except Exception as exc:
            with self._lock:
                self._state = VoiceOutputState.ERROR
                self._error = exc
            raise

        self._stop_event.clear()
        self._synthesis_thread = threading.Thread(
            target=self._synthesis_loop,
            name="DexterTTSSynthesis",
            daemon=True,
        )
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="DexterTTSPlayback",
            daemon=True,
        )
        self._synthesis_thread.start()
        self._playback_thread.start()
        with self._lock:
            self._state = VoiceOutputState.IDLE

    def begin_response(self) -> int:
        """Start accepting streamed text for a new assistant response."""
        self.interrupt()
        with self._lock:
            self._response_id += 1
            response_id = self._response_id
            self._active_response_id = response_id
            self._cancel_event = threading.Event()
            self.buffer.reset()
            self.formatter.reset()
            return response_id

    def add_text(self, text: str) -> None:
        """Accept a streamed text fragment without doing model work."""
        with self._lock:
            response_id = self._active_response_id
        if response_id is None or not text:
            return

        for raw_unit in self.buffer.feed(text):
            self._enqueue_unit(response_id, raw_unit)

    def finish_response(self) -> None:
        """Flush the final partial sentence and stop accepting more text."""
        with self._lock:
            response_id = self._active_response_id
            self._active_response_id = None
        if response_id is None:
            return

        for raw_unit in self.buffer.flush():
            self._enqueue_unit(response_id, raw_unit)

    def interrupt(self) -> None:
        """Stop audible speech now and invalidate queued/current synthesis."""
        with self._lock:
            self._active_response_id = None
            self._cancel_event.set()
            self.buffer.reset()
            self.formatter.reset()
        self._drain_queue(self._text_queue)
        self._drain_queue(self._audio_queue)
        self.sink.interrupt()

    def stop(self) -> None:
        if self.state is VoiceOutputState.STOPPED:
            return

        self.interrupt()
        self._stop_event.set()
        self._put_stop(self._text_queue)
        self._put_stop(self._audio_queue)

        if self._synthesis_thread is not None:
            self._synthesis_thread.join(timeout=5.0)
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=5.0)

        try:
            self.sink.stop()
        finally:
            self.backend.stop()

        with self._lock:
            self._state = VoiceOutputState.STOPPED

    def _enqueue_unit(self, response_id: int, raw_text: str) -> None:
        spoken = self.formatter.format(raw_text)
        if not spoken:
            return

        with self._lock:
            if response_id != self._response_id or self._cancel_event.is_set():
                return

        unit = _SpeechUnit(response_id, spoken)
        try:
            self._text_queue.put_nowait(unit)
            return
        except queue.Full:
            pass

        # Never block Qt/Goose. If synthesis falls behind far enough to fill the
        # bounded queue, discard the oldest *pending* sentence and keep the
        # newest speech. The sentence already being synthesized is untouched,
        # and queue pressure must never cancel the rest of the assistant turn.
        try:
            dropped = self._text_queue.get_nowait()
        except queue.Empty:
            # The synthesis worker drained the queue between Full and get().
            # Retry once without blocking; a concurrent refill can still win.
            try:
                self._text_queue.put_nowait(unit)
            except queue.Full:
                LOGGER.warning(
                    "TTS text queue refilled concurrently; dropping newest speech unit: %r",
                    spoken[:80],
                )
            return

        if dropped is _STOP:
            # Shutdown won the race. Preserve the stop sentinel and ignore new
            # speech rather than allowing late response text to revive the queue.
            self._text_queue.put_nowait(_STOP)
            return

        assert isinstance(dropped, _SpeechUnit)
        LOGGER.warning(
            "TTS text queue full; dropping oldest pending speech unit: %r",
            dropped.text[:80],
        )
        try:
            self._text_queue.put_nowait(unit)
        except queue.Full:
            # A producer/refill raced us after the drop. Stay non-blocking and
            # preserve the bounded queue; importantly, do not cancel the turn.
            LOGGER.warning(
                "TTS text queue refilled concurrently; dropping newest speech unit: %r",
                spoken[:80],
            )

    def _coalesce_available_units(self, first: _SpeechUnit) -> _SpeechUnit:
        """Merge a few already-queued same-response sentences without waiting."""
        parts = [first.text]
        for _ in range(self.max_coalesced_units - 1):
            try:
                next_item = self._text_queue.get_nowait()
            except queue.Empty:
                break

            if next_item is _STOP:
                self._text_queue.put_nowait(_STOP)
                break
            assert isinstance(next_item, _SpeechUnit)
            if next_item.response_id != first.response_id:
                # begin_response()/interrupt() normally drains the old response
                # before a new one can enqueue. Preserve the unexpected item if
                # that invariant is ever relaxed.
                self._text_queue.put_nowait(next_item)
                break
            parts.append(next_item.text)

        if len(parts) == 1:
            return first
        return _SpeechUnit(first.response_id, " ".join(parts))

    def _synthesis_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._text_queue.get()
                if item is _STOP:
                    return
                assert isinstance(item, _SpeechUnit)

                with self._lock:
                    current_id = self._response_id
                    cancel_event = self._cancel_event
                if item.response_id != current_id or cancel_event.is_set():
                    continue

                item = self._coalesce_available_units(item)

                with self._lock:
                    self._state = VoiceOutputState.SPEAKING

                for chunk in self.backend.synthesize(item.text, cancel_event):
                    if cancel_event.is_set() or item.response_id != self._response_id:
                        break
                    self._audio_queue.put(
                        _QueuedAudio(item.response_id, chunk, cancel_event)
                    )
        except Exception as exc:
            self._record_worker_error(exc)

    def _playback_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._audio_queue.get()
                if item is _STOP:
                    return
                assert isinstance(item, _QueuedAudio)

                if item.cancel_event.is_set() or item.response_id != self._response_id:
                    continue
                self.sink.play(item.chunk, item.cancel_event)

                if self._text_queue.empty() and self._audio_queue.empty():
                    with self._lock:
                        if not item.cancel_event.is_set():
                            self._state = VoiceOutputState.IDLE
        except Exception as exc:
            self._record_worker_error(exc)

    def _record_worker_error(self, exc: BaseException) -> None:
        LOGGER.exception("TTS worker failed", exc_info=exc)
        with self._lock:
            self._error = exc
            self._state = VoiceOutputState.ERROR
            self._cancel_event.set()
        self.sink.interrupt()

    @staticmethod
    def _drain_queue(target: queue.Queue[object]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _put_stop(target: queue.Queue[object]) -> None:
        try:
            target.put_nowait(_STOP)
        except queue.Full:
            VoiceOutput._drain_queue(target)
            target.put_nowait(_STOP)

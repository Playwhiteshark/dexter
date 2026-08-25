"""Opt-in end-to-end timing trace for Dexter's live TTS path.

Set ``DEXTER_TTS_TRACE=1`` before launching the Host. The wrappers in this
module do not change buffering, model settings, or synthesis scheduling; they
only timestamp the existing boundaries so startup delay and audible refill
pauses can be diagnosed in the real application.

When enabled, the trace is also written to
``%LOCALAPPDATA%\\Dexter\\Diagnostics\\tts-trace.log`` (or the path supplied by
``DEXTER_TTS_TRACE_FILE``). This keeps diagnostics usable when Dexter is
launched through pythonw/wscript with no console attached.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from Runtime.Voice.buffered_audio_sink import BufferedSoundDeviceAudioSink
from Runtime.Voice.tts import AudioChunk, SpeechBackend, VoiceOutput

_TRACE_ENABLED = os.environ.get("DEXTER_TTS_TRACE", "0").strip() == "1"
_TRACE_ORIGIN = time.perf_counter()
_TRACE_LOCK = threading.Lock()
_TRACE_STREAM = None

if _TRACE_ENABLED:
    configured_path = os.environ.get("DEXTER_TTS_TRACE_FILE", "").strip()
    if configured_path:
        trace_path = Path(configured_path).expanduser()
    else:
        runtime_root = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("TEMP")
            or str(Path.cwd())
        )
        trace_path = runtime_root / "Dexter" / "Diagnostics" / "tts-trace.log"
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        _TRACE_STREAM = trace_path.open("w", encoding="utf-8", buffering=1)
    except Exception:
        _TRACE_STREAM = None


def enabled() -> bool:
    return _TRACE_ENABLED


def event(name: str, **fields: object) -> None:
    if not _TRACE_ENABLED:
        return
    elapsed = time.perf_counter() - _TRACE_ORIGIN
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    line = f"[TTS TRACE] +{elapsed:8.3f}s {name}"
    if details:
        line += " " + details
    with _TRACE_LOCK:
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:
            pass
        if _TRACE_STREAM is not None:
            try:
                print(line, file=_TRACE_STREAM, flush=True)
            except Exception:
                pass


class TracingSpeechBackend:
    """Transparent backend wrapper that timestamps one synthesis request."""

    def __init__(self, inner: SpeechBackend) -> None:
        self.inner = inner
        self._request_number = 0

    @property
    def ready_info(self) -> Any:
        return getattr(self.inner, "ready_info", None)

    def start(self) -> None:
        self.inner.start()

    def synthesize(
        self,
        text: str,
        cancel_event: threading.Event,
    ) -> Iterable[AudioChunk]:
        self._request_number += 1
        request = self._request_number
        event(
            "synthesis_request",
            request=request,
            chars=len(text),
            preview=text[:80],
        )
        first = True
        chunks = 0
        started = time.perf_counter()
        previous_pcm_at: float | None = None
        try:
            for chunk in self.inner.synthesize(text, cancel_event):
                now = time.perf_counter()
                chunks += 1
                audio_seconds = len(chunk.samples) / chunk.sample_rate
                gap_seconds = (
                    None if previous_pcm_at is None else now - previous_pcm_at
                )
                previous_pcm_at = now
                if first:
                    first = False
                    event(
                        "first_pcm",
                        request=request,
                        after_s=round(now - started, 3),
                        audio_s=round(audio_seconds, 3),
                    )
                event(
                    "pcm_chunk",
                    request=request,
                    chunk=chunks,
                    after_s=round(now - started, 3),
                    gap_s=(
                        None if gap_seconds is None else round(gap_seconds, 3)
                    ),
                    audio_s=round(audio_seconds, 3),
                )
                yield chunk
        finally:
            event(
                "synthesis_done",
                request=request,
                after_s=round(time.perf_counter() - started, 3),
                chunks=chunks,
                cancelled=cancel_event.is_set(),
            )

    def stop(self) -> None:
        self.inner.stop()


class TracingBufferedSoundDeviceAudioSink(BufferedSoundDeviceAudioSink):
    """Timestamp PortAudio starts, PCM reserve, and true underflow/refill gaps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._underflow_pending = False
        self._underflow_started_at: float | None = None
        self._underflow_count = 0
        self._callback_status_pending = False
        self._playback_starts = 0

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        before = self.buffered_seconds
        had_underflow = self._underflow_pending
        underflow_started_at = self._underflow_started_at
        callback_status = self._callback_status_pending
        super().play(chunk, cancel_event)
        after = self.buffered_seconds

        if not cancel_event.is_set():
            event(
                "audio_chunk_buffered",
                audio_s=round(len(chunk.samples) / chunk.sample_rate, 3),
                buffered_before_s=round(before, 3),
                buffered_after_s=round(after, 3),
            )

        if had_underflow and not cancel_event.is_set():
            self._underflow_pending = False
            self._underflow_started_at = None
            self._callback_status_pending = False
            event(
                "audio_refill_after_underflow",
                underflow=self._underflow_count,
                gap_s=(
                    None
                    if underflow_started_at is None
                    else round(time.perf_counter() - underflow_started_at, 3)
                ),
                buffered_s=round(after, 3),
                callback_status=callback_status,
            )

    def _start_claimed_stream(self, stream: Any, *, propagate: bool) -> None:
        super()._start_claimed_stream(stream, propagate=propagate)
        with self._condition:
            started = self._stream is stream and self._stream_started
            buffered = (
                self._buffered_frames / self._sample_rate
                if self._sample_rate
                else 0.0
            )
        if started:
            self._playback_starts += 1
            event(
                "playback_start",
                start=self._playback_starts,
                buffered_s=round(buffered, 3),
            )

    def _audio_callback(
        self,
        outdata,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        was_rebuffering = self._rebuffering
        super()._audio_callback(outdata, frames, time_info, status)
        # Never print or perform file I/O from PortAudio's realtime callback.
        # Record only cheap state/timestamps here; the next normal playback
        # thread ``play`` call emits the diagnostic event.
        if status:
            self._callback_status_pending = True
        if not was_rebuffering and self._rebuffering:
            self._underflow_count += 1
            self._underflow_pending = True
            self._underflow_started_at = time.perf_counter()

    def interrupt(self) -> None:
        self._underflow_pending = False
        self._underflow_started_at = None
        self._callback_status_pending = False
        super().interrupt()


class TracingVoiceOutput(VoiceOutput):
    """Timestamp streamed text arrival and the first sentence dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trace_seen_text = False

    def begin_response(self) -> int:
        response_id = super().begin_response()
        self._trace_seen_text = False
        event("response_begin", response=response_id)
        return response_id

    def add_text(self, text: str) -> None:
        if text and not self._trace_seen_text:
            self._trace_seen_text = True
            event(
                "first_visible_text",
                chars=len(text),
                preview=text[:80],
            )
        super().add_text(text)

    def _enqueue_unit(self, response_id: int, raw_text: str) -> None:
        event(
            "speech_unit_queued",
            response=response_id,
            chars=len(raw_text),
            preview=raw_text[:80],
        )
        super()._enqueue_unit(response_id, raw_text)

    def finish_response(self) -> None:
        event("response_finish")
        super().finish_response()

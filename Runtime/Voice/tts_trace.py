"""Opt-in end-to-end timing trace for Dexter's live TTS path.

Set ``DEXTER_TTS_TRACE=1`` before launching the Host.  The wrappers in this
module do not change buffering, model settings, or synthesis scheduling; they
only timestamp the existing boundaries so startup delay and audible refill
pauses can be diagnosed in the real application.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterable
from typing import Any

from Runtime.Voice.buffered_audio_sink import BufferedSoundDeviceAudioSink
from Runtime.Voice.tts import AudioChunk, SpeechBackend, VoiceOutput

_TRACE_ENABLED = os.environ.get("DEXTER_TTS_TRACE", "0").strip() == "1"
_TRACE_ORIGIN = time.perf_counter()
_TRACE_LOCK = threading.Lock()


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
        print(line, file=sys.stderr, flush=True)


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
        try:
            for chunk in self.inner.synthesize(text, cancel_event):
                chunks += 1
                if first:
                    first = False
                    event(
                        "first_pcm",
                        request=request,
                        after_s=round(time.perf_counter() - started, 3),
                        audio_s=round(len(chunk.samples) / chunk.sample_rate, 3),
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
    """Timestamp real PortAudio start and refills after a true underflow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._underflow_pending = False
        self._playback_starts = 0

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        had_underflow = self._underflow_pending
        super().play(chunk, cancel_event)
        if had_underflow and not cancel_event.is_set():
            self._underflow_pending = False
            event(
                "audio_refill_after_underflow",
                buffered_s=round(self.buffered_seconds, 3),
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
        # Never print from PortAudio's realtime callback.  Mark the transition;
        # the next normal playback-thread ``play`` call reports it safely.
        if not was_rebuffering and self._rebuffering:
            self._underflow_pending = True

    def interrupt(self) -> None:
        self._underflow_pending = False
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

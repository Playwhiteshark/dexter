"""Continuous jitter-buffered local playback for Dexter TTS.

NeuTTS can produce valid streamed PCM more slowly and less regularly than the
sound device consumes it.  Writing each model chunk directly to a blocking
PortAudio stream makes those scheduling gaps audible.  This sink decouples
model-chunk arrival from the device callback with a small bounded PCM buffer.

The buffer is deliberately measured in audio seconds, not model chunks: chunk
sizes are an implementation detail of the TTS backend.  On underflow the sink
outputs silence and waits for a useful reserve before resuming, avoiding rapid
pause/unpause chatter.  PTT interruption still aborts the PortAudio stream and
clears all queued PCM immediately.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from Runtime.Voice.tts import AudioChunk

LOGGER = logging.getLogger(__name__)

StreamFactory = Callable[..., Any]


class BufferedSoundDeviceAudioSink:
    """Callback-driven PortAudio sink with bounded startup/rebuffer reserves."""

    def __init__(
        self,
        *,
        device: int | str | None = None,
        prebuffer_seconds: float = 1.0,
        rebuffer_seconds: float = 0.75,
        max_buffer_seconds: float = 5.0,
        startup_max_wait_seconds: float = 1.0,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        if prebuffer_seconds <= 0:
            raise ValueError("prebuffer_seconds must be positive")
        if rebuffer_seconds <= 0:
            raise ValueError("rebuffer_seconds must be positive")
        if max_buffer_seconds <= max(prebuffer_seconds, rebuffer_seconds):
            raise ValueError("max_buffer_seconds must exceed the reserve sizes")
        if startup_max_wait_seconds <= 0:
            raise ValueError("startup_max_wait_seconds must be positive")

        self.device = device
        self.prebuffer_seconds = float(prebuffer_seconds)
        self.rebuffer_seconds = float(rebuffer_seconds)
        self.max_buffer_seconds = float(max_buffer_seconds)
        self.startup_max_wait_seconds = float(startup_max_wait_seconds)
        self._stream_factory = stream_factory

        self._condition = threading.Condition(threading.RLock())
        self._chunks: deque[np.ndarray] = deque()
        self._head_offset = 0
        self._buffered_frames = 0
        self._sample_rate: int | None = None
        self._channels: int | None = None
        self._stream: Any | None = None
        self._stream_started = False
        self._rebuffering = False
        self._first_enqueue_time: float | None = None
        self._generation = 0
        self._closed = False

    def start(self) -> None:
        # PortAudio is opened lazily when the first TTS chunk reveals format.
        with self._condition:
            self._closed = False

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[0] == 0:
            return

        channels = int(samples.shape[1])
        sample_rate = int(chunk.sample_rate)
        if sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")

        # Own the PCM independently of the backend's temporary buffers.
        samples = np.ascontiguousarray(samples, dtype=np.float32).copy()

        with self._condition:
            if self._closed or cancel_event.is_set():
                return

            if (
                self._sample_rate is not None
                and (self._sample_rate != sample_rate or self._channels != channels)
            ):
                raise RuntimeError(
                    "TTS audio format changed mid-stream: "
                    f"{self._sample_rate}Hz/{self._channels}ch -> "
                    f"{sample_rate}Hz/{channels}ch"
                )

            if self._sample_rate is None:
                self._sample_rate = sample_rate
                self._channels = channels
                self._open_stream_locked()

            max_frames = max(1, int(self.max_buffer_seconds * sample_rate))
            while (
                not self._closed
                and not cancel_event.is_set()
                and self._buffered_frames >= max_frames
            ):
                # Backpressure only the playback worker, never Qt or Goose.
                self._condition.wait(timeout=0.05)

            if self._closed or cancel_event.is_set():
                return

            self._chunks.append(samples)
            self._buffered_frames += int(samples.shape[0])
            if self._first_enqueue_time is None:
                self._first_enqueue_time = time.monotonic()
                self._schedule_start_deadline_locked()

            self._maybe_start_stream_locked(force=False)
            self._condition.notify_all()

    def interrupt(self) -> None:
        with self._condition:
            self._generation += 1
            stream = self._stream
            self._stream = None
            self._stream_started = False
            self._reset_buffer_locked(clear_format=True)
            self._condition.notify_all()

        self._abort_and_close(stream)

    def stop(self) -> None:
        with self._condition:
            self._closed = True
            self._generation += 1
            stream = self._stream
            self._stream = None
            self._stream_started = False
            self._reset_buffer_locked(clear_format=True)
            self._condition.notify_all()

        self._abort_and_close(stream)

    @property
    def buffered_seconds(self) -> float:
        with self._condition:
            if not self._sample_rate:
                return 0.0
            return self._buffered_frames / self._sample_rate

    def _open_stream_locked(self) -> None:
        assert self._sample_rate is not None
        assert self._channels is not None
        if self._stream is not None:
            return

        factory = self._stream_factory
        if factory is None:
            import sounddevice as sd

            factory = sd.OutputStream

        self._stream = factory(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            device=self.device,
            latency="low",
            callback=self._audio_callback,
        )
        self._stream_started = False

    def _schedule_start_deadline_locked(self) -> None:
        generation = self._generation

        def deadline() -> None:
            time.sleep(self.startup_max_wait_seconds)
            with self._condition:
                if generation != self._generation or self._closed:
                    return
                self._maybe_start_stream_locked(force=True)

        threading.Thread(
            target=deadline,
            name="DexterTTSAudioPrebuffer",
            daemon=True,
        ).start()

    def _maybe_start_stream_locked(self, *, force: bool) -> None:
        if self._stream is None or self._stream_started or self._closed:
            return
        assert self._sample_rate is not None

        target_frames = max(1, int(self.prebuffer_seconds * self._sample_rate))
        if not force and self._buffered_frames < target_frames:
            return
        if self._buffered_frames <= 0:
            return

        # start() is quick and happens on Dexter's playback/deadline worker,
        # never on Qt or the PortAudio callback itself.
        self._stream.start()
        self._stream_started = True
        self._rebuffering = False

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        # PortAudio requires every output frame to be initialized.
        outdata.fill(0)
        if status:
            # Avoid logging from PortAudio's realtime thread.  Any starvation is
            # already represented by the rebuffering state below.
            pass

        with self._condition:
            if self._closed or self._sample_rate is None:
                return

            if self._rebuffering:
                reserve = max(1, int(self.rebuffer_seconds * self._sample_rate))
                if self._buffered_frames < reserve:
                    return
                self._rebuffering = False

            written = 0
            while written < frames and self._chunks:
                head = self._chunks[0]
                available = int(head.shape[0]) - self._head_offset
                take = min(frames - written, available)
                outdata[written : written + take, :] = head[
                    self._head_offset : self._head_offset + take, :
                ]
                written += take
                self._head_offset += take
                self._buffered_frames -= take

                if self._head_offset >= int(head.shape[0]):
                    self._chunks.popleft()
                    self._head_offset = 0

            # If PortAudio asked for more data than was available, the remainder
            # is already silence.  Do not resume on the very next tiny model
            # chunk: accumulate a useful reserve first, which turns rapid
            # pause/unpause chatter into at most one deliberate refill pause.
            if written < frames:
                self._rebuffering = True

            self._condition.notify_all()

    def _reset_buffer_locked(self, *, clear_format: bool) -> None:
        self._chunks.clear()
        self._head_offset = 0
        self._buffered_frames = 0
        self._rebuffering = False
        self._first_enqueue_time = None
        if clear_format:
            self._sample_rate = None
            self._channels = None

    @staticmethod
    def _abort_and_close(stream: Any | None) -> None:
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            LOGGER.debug("Audio stream abort failed", exc_info=True)
        try:
            stream.close()
        except Exception:
            LOGGER.debug("Audio stream close failed", exc_info=True)

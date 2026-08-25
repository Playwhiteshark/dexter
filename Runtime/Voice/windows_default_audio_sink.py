"""Windows-default TTS playback using Core Audio/WASAPI.

PortAudio/sounddevice caches the default device at initialization on Windows, so
an already-running Dexter can keep speaking through the old endpoint after the
user plugs or unplugs headphones. SoundCard queries Core Audio's current default
render endpoint directly. This sink keeps Dexter's existing jitter-buffer
semantics while reopening only TTS playback when that endpoint changes; STT's
always-open PortAudio microphone is untouched.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from Runtime.Voice.tts import AudioChunk

LOGGER = logging.getLogger(__name__)
TraceCallback = Callable[..., None]


def soundcard_available() -> bool:
    return importlib.util.find_spec("soundcard") is not None


class WindowsDefaultAudioSink:
    """Buffered WASAPI playback that follows the live Windows default speaker."""

    def __init__(
        self,
        *,
        prebuffer_seconds: float = 0.5,
        rebuffer_seconds: float = 0.5,
        max_buffer_seconds: float = 5.0,
        startup_max_wait_seconds: float = 0.75,
        expected_sample_rate: int = 24_000,
        expected_channels: int = 1,
        playback_block_seconds: float = 0.02,
        device_poll_seconds: float = 0.10,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        if expected_sample_rate <= 0 or expected_channels <= 0:
            raise ValueError("expected audio format must be positive")
        if expected_channels != 1:
            raise ValueError("WindowsDefaultAudioSink currently expects mono TTS")
        if prebuffer_seconds <= 0 or rebuffer_seconds <= 0:
            raise ValueError("buffer reserves must be positive")
        if max_buffer_seconds <= max(prebuffer_seconds, rebuffer_seconds):
            raise ValueError("max_buffer_seconds must exceed reserve sizes")
        if startup_max_wait_seconds <= 0:
            raise ValueError("startup_max_wait_seconds must be positive")
        if playback_block_seconds <= 0 or device_poll_seconds <= 0:
            raise ValueError("playback/poll intervals must be positive")

        self.prebuffer_seconds = float(prebuffer_seconds)
        self.rebuffer_seconds = float(rebuffer_seconds)
        self.max_buffer_seconds = float(max_buffer_seconds)
        self.startup_max_wait_seconds = float(startup_max_wait_seconds)
        self.expected_sample_rate = int(expected_sample_rate)
        self.expected_channels = int(expected_channels)
        self.playback_block_seconds = float(playback_block_seconds)
        self.device_poll_seconds = float(device_poll_seconds)
        self._trace_callback = trace_callback

        self._condition = threading.Condition(threading.RLock())
        self._chunks: deque[np.ndarray] = deque()
        self._head_offset = 0
        self._buffered_frames = 0
        self._generation = 0
        self._closed = True
        self._playing = False
        self._rebuffering = False
        self._first_buffered_at: float | None = None
        self._underflow_started_at: float | None = None
        self._underflow_count = 0
        self._force_reopen = False
        self._thread: threading.Thread | None = None

    @property
    def buffered_seconds(self) -> float:
        with self._condition:
            return self._buffered_frames / self.expected_sample_rate

    def start(self) -> None:
        if not soundcard_available():
            raise RuntimeError(
                "Windows default-device TTS playback requires SoundCard. "
                "Install with: uv pip install soundcard==0.4.6"
            )
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._generation += 1
            self._thread = threading.Thread(
                target=self._playback_loop,
                name="DexterTTSWASAPI",
                daemon=True,
            )
            self._thread.start()

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return
        if int(chunk.sample_rate) != self.expected_sample_rate:
            raise RuntimeError(
                f"TTS sample rate changed: {chunk.sample_rate} -> {self.expected_sample_rate}"
            )

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if samples.ndim == 2:
            if samples.shape[1] != 1:
                raise RuntimeError("Windows default TTS sink expects mono PCM")
            samples = samples[:, 0]
        if samples.ndim != 1 or samples.size == 0:
            return
        samples = np.ascontiguousarray(samples, dtype=np.float32).copy()

        with self._condition:
            if self._closed or cancel_event.is_set():
                return
            max_frames = max(1, int(self.max_buffer_seconds * self.expected_sample_rate))
            while (
                not self._closed
                and not cancel_event.is_set()
                and self._buffered_frames >= max_frames
            ):
                self._condition.wait(timeout=0.05)
            if self._closed or cancel_event.is_set():
                return

            was_empty = self._buffered_frames == 0
            self._chunks.append(samples)
            self._buffered_frames += int(samples.size)
            if was_empty and not self._playing:
                self._first_buffered_at = time.perf_counter()
            self._condition.notify_all()
            buffered = self._buffered_frames / self.expected_sample_rate

        self._trace(
            "audio_chunk_buffered",
            audio_s=round(samples.size / self.expected_sample_rate, 3),
            buffered_after_s=round(buffered, 3),
        )

    def interrupt(self) -> None:
        with self._condition:
            self._generation += 1
            self._reset_buffer_locked()
            self._playing = False
            self._rebuffering = False
            self._first_buffered_at = None
            self._underflow_started_at = None
            # Reopen to flush the small WASAPI queue already submitted to the old
            # response. With a 20 ms block this bounds audible interruption tail.
            self._force_reopen = True
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._reset_buffer_locked()
            self._condition.notify_all()
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _playback_loop(self) -> None:
        import soundcard as sc

        player: Any | None = None
        speaker_id: Any | None = None
        speaker_name: str | None = None
        next_device_check = 0.0
        block_frames = max(1, int(self.playback_block_seconds * self.expected_sample_rate))

        try:
            while True:
                now = time.perf_counter()
                if player is None or self._force_reopen or now >= next_device_check:
                    next_device_check = now + self.device_poll_seconds
                    try:
                        speaker = sc.default_speaker()
                        current_id = speaker.id
                        current_name = speaker.name
                    except Exception:
                        LOGGER.debug("Could not query Windows default speaker", exc_info=True)
                        speaker = None
                        current_id = None
                        current_name = None

                    with self._condition:
                        force_reopen = self._force_reopen
                        self._force_reopen = False
                        if self._closed:
                            break

                    if speaker is None:
                        if player is not None:
                            self._close_player(player)
                            player = None
                            speaker_id = None
                            speaker_name = None
                        time.sleep(min(self.device_poll_seconds, 0.1))
                        continue

                    if player is None or force_reopen or current_id != speaker_id:
                        old_name = speaker_name
                        if player is not None:
                            self._close_player(player)
                        try:
                            player = speaker.player(
                                samplerate=self.expected_sample_rate,
                                blocksize=block_frames,
                            )
                            player = player.__enter__()
                        except Exception:
                            LOGGER.exception("Could not open Windows default TTS speaker")
                            player = None
                            speaker_id = None
                            speaker_name = None
                            time.sleep(min(self.device_poll_seconds, 0.1))
                            continue
                        speaker_id = current_id
                        speaker_name = current_name
                        if old_name != speaker_name:
                            self._trace(
                                "audio_device_change",
                                previous=old_name,
                                current=speaker_name,
                            )

                data: np.ndarray | None = None
                generation: int | None = None
                with self._condition:
                    while True:
                        if self._closed:
                            break
                        if self._buffered_frames <= 0:
                            # Once playback has actually underflowed, preserve the
                            # rebuffering state until enough PCM arrives to satisfy
                            # the rebuffer reserve. Clearing _playing here used to
                            # route the refill through the startup path instead,
                            # hiding underflow telemetry and bypassing rebuffering.
                            if not self._rebuffering:
                                self._playing = False
                            self._condition.wait(timeout=min(self.device_poll_seconds, 0.05))
                            break

                        target = max(1, int(self.prebuffer_seconds * self.expected_sample_rate))
                        if not self._playing:
                            waited = (
                                0.0
                                if self._first_buffered_at is None
                                else time.perf_counter() - self._first_buffered_at
                            )
                            if (
                                self._buffered_frames < target
                                and waited < self.startup_max_wait_seconds
                            ):
                                self._condition.wait(timeout=0.02)
                                break
                            self._playing = True
                            self._rebuffering = False
                            self._trace(
                                "playback_start",
                                buffered_s=round(
                                    self._buffered_frames / self.expected_sample_rate, 3
                                ),
                            )

                        if self._rebuffering:
                            reserve = max(
                                1,
                                int(self.rebuffer_seconds * self.expected_sample_rate),
                            )
                            if self._buffered_frames < reserve:
                                self._condition.wait(timeout=0.02)
                                break
                            self._rebuffering = False
                            if self._underflow_started_at is not None:
                                gap = time.perf_counter() - self._underflow_started_at
                                self._trace(
                                    "audio_refill_after_underflow",
                                    underflow=self._underflow_count,
                                    gap_s=round(gap, 3),
                                    buffered_s=round(
                                        self._buffered_frames / self.expected_sample_rate,
                                        3,
                                    ),
                                )
                                self._underflow_started_at = None

                        generation = self._generation
                        data = self._peek_locked(block_frames)
                        break

                    if self._closed:
                        break

                # Waiting for initial prebuffer or rebuffer is a deliberate
                # no-play iteration. Never reuse data/generation from a previous
                # pass through the loop; doing so can replay stale PCM and consume
                # newly buffered samples before the reserve target is reached.
                if data is None or generation is None:
                    continue
                if not self._playing or self._buffered_frames <= 0:
                    continue
                if player is None:
                    continue

                try:
                    player.play(data)
                except Exception:
                    LOGGER.debug("Windows TTS playback failed; reopening default speaker", exc_info=True)
                    self._close_player(player)
                    player = None
                    speaker_id = None
                    speaker_name = None
                    next_device_check = 0.0
                    continue

                with self._condition:
                    if generation != self._generation:
                        continue
                    self._consume_locked(int(data.size))
                    if self._buffered_frames == 0:
                        self._rebuffering = True
                        self._underflow_count += 1
                        self._underflow_started_at = time.perf_counter()
                    self._condition.notify_all()
        finally:
            if player is not None:
                self._close_player(player)

    def _peek_locked(self, frames: int) -> np.ndarray:
        remaining = min(frames, self._buffered_frames)
        parts: list[np.ndarray] = []
        offset = self._head_offset
        for chunk in self._chunks:
            if remaining <= 0:
                break
            available = int(chunk.size) - offset
            take = min(remaining, available)
            parts.append(chunk[offset : offset + take])
            remaining -= take
            offset = 0
        if not parts:
            return np.empty(0, dtype=np.float32)
        if len(parts) == 1:
            return np.ascontiguousarray(parts[0], dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    def _consume_locked(self, frames: int) -> None:
        remaining = min(frames, self._buffered_frames)
        while remaining > 0 and self._chunks:
            head = self._chunks[0]
            available = int(head.size) - self._head_offset
            take = min(remaining, available)
            self._head_offset += take
            self._buffered_frames -= take
            remaining -= take
            if self._head_offset >= int(head.size):
                self._chunks.popleft()
                self._head_offset = 0

    def _reset_buffer_locked(self) -> None:
        self._chunks.clear()
        self._head_offset = 0
        self._buffered_frames = 0

    @staticmethod
    def _close_player(player: Any) -> None:
        try:
            player.__exit__(None, None, None)
        except Exception:
            LOGGER.debug("Could not close Windows TTS player", exc_info=True)

    def _trace(self, name: str, **fields: object) -> None:
        callback = self._trace_callback
        if callback is None:
            return
        try:
            callback(name, **fields)
        except Exception:
            pass

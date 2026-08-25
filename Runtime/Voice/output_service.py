"""Host-facing lifecycle wrapper for Dexter speech output.

The model backend is expensive to start, so the Host starts it in parallel with
Gateway/STT/Goose. A failed TTS startup degrades to text-only operation rather
than preventing Dexter from starting.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from Runtime.Voice.buffered_audio_sink import BufferedSoundDeviceAudioSink
from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend
from Runtime.Voice.tts import VoiceOutput, VoiceOutputState
from Runtime.Voice.tts_trace import (
    TracingBufferedSoundDeviceAudioSink,
    TracingSpeechBackend,
    TracingVoiceOutput,
    enabled as tts_trace_enabled,
)

ErrorCallback = Callable[[str], None]
ReadyCallback = Callable[[dict[str, object] | None], None]


class SpeechOutputService:
    """Own one VoiceOutput instance without exposing model lifecycle to Qt."""

    def __init__(self, output: VoiceOutput | None = None) -> None:
        if output is not None:
            self.output = output
        elif tts_trace_enabled():
            self.output = TracingVoiceOutput(
                TracingSpeechBackend(NeuTTS2EBackend()),
                sink=TracingBufferedSoundDeviceAudioSink(),
            )
        else:
            # Keep the normal production path byte-for-byte equivalent when
            # tracing is disabled; diagnostics must not perturb live behavior.
            self.output = VoiceOutput(
                NeuTTS2EBackend(),
                sink=BufferedSoundDeviceAudioSink(),
            )
        self._lock = threading.Lock()
        self._ready = False
        self._response_active = False
        self._stopping = False
        self._startup_thread: threading.Thread | None = None
        self._startup_error: str | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def startup_error(self) -> str | None:
        with self._lock:
            return self._startup_error

    def start_async(
        self,
        *,
        on_ready: ReadyCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        with self._lock:
            if self._startup_thread is not None:
                return
            self._stopping = False
            self._startup_error = None
            thread = threading.Thread(
                target=self._start_worker,
                args=(on_ready, on_error),
                name="DexterTTSStartup",
                daemon=True,
            )
            self._startup_thread = thread
        thread.start()

    def _start_worker(
        self,
        on_ready: ReadyCallback | None,
        on_error: ErrorCallback | None,
    ) -> None:
        try:
            self.output.start()
            with self._lock:
                stopping = self._stopping
                if not stopping:
                    self._ready = True
            if stopping:
                self.output.stop()
                return

            info = getattr(self.output.backend, "ready_info", None)
            if on_ready is not None:
                on_ready(info if isinstance(info, dict) else None)
        except Exception as exc:
            message = str(exc)
            with self._lock:
                self._ready = False
                self._response_active = False
                self._startup_error = message
                stopping = self._stopping
            if not stopping and on_error is not None:
                on_error(message)

    def begin_response(self) -> bool:
        with self._lock:
            if not self._ready or self._stopping:
                self._response_active = False
                return False
            self._response_active = True
        try:
            self.output.begin_response()
        except Exception:
            with self._lock:
                self._response_active = False
            raise
        return True

    def add_text(self, text: str) -> None:
        with self._lock:
            active = self._response_active and self._ready and not self._stopping
        if active:
            self.output.add_text(text)

    def finish_response(self) -> None:
        with self._lock:
            active = self._response_active and self._ready and not self._stopping
            self._response_active = False
        if active:
            self.output.finish_response()

    def interrupt(self) -> None:
        with self._lock:
            ready = self._ready
            self._response_active = False
        if ready:
            self.output.interrupt()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._ready = False
            self._response_active = False
            thread = self._startup_thread

        # VoiceOutput.stop() also stops a backend that is still STARTING. That
        # lets quit terminate a long model load instead of waiting ~45 seconds.
        if self.output.state is not VoiceOutputState.STOPPED:
            try:
                self.output.stop()
            except Exception:
                # Shutdown is best-effort; the Host is already exiting.
                pass

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)

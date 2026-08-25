"""Host-facing lifecycle wrapper for Dexter speech output.

TTS is optional at the Host boundary. It may be warmed at startup, completely
unloaded while Voice output is disabled, and warmed again later without
restarting Dexter. Disabling Voice stops the model-owning child process rather
than merely muting playback, so its Arc/CPU/RAM resources are released.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend
from Runtime.Voice.text_normalization import NormalizingSpeechBackend
from Runtime.Voice.tts import VoiceOutput, VoiceOutputState
from Runtime.Voice.windows_default_audio_sink import WindowsDefaultAudioSink
from Runtime.Voice.tts_trace import (
    TracingSpeechBackend,
    TracingVoiceOutput,
    enabled as tts_trace_enabled,
)

ErrorCallback = Callable[[str], None]
ReadyCallback = Callable[[dict[str, object] | None], None]
DisabledCallback = Callable[[], None]
OutputFactory = Callable[[], VoiceOutput]

_NEUTTS_SAMPLE_RATE = 24_000
_NEUTTS_CHANNELS = 1

_OFF = "off"
_STARTING = "starting"
_READY = "ready"
_STOPPING = "stopping"
_SHUTDOWN = "shutdown"


def _production_output() -> VoiceOutput:
    # TTS playback uses Core Audio/WASAPI directly so Dexter follows the live
    # Windows default render endpoint when headphones/speakers are plugged or
    # unplugged. STT remains on sounddevice/PortAudio and is not reinitialized.
    sink = WindowsDefaultAudioSink(
        expected_sample_rate=_NEUTTS_SAMPLE_RATE,
        expected_channels=_NEUTTS_CHANNELS,
    )
    if tts_trace_enabled():
        # Put normalization outside the tracing backend so synthesis traces show
        # the exact text NeuTTS receives rather than the visible source.
        return TracingVoiceOutput(
            NormalizingSpeechBackend(TracingSpeechBackend(NeuTTS2EBackend())),
            sink=sink,
        )
    return VoiceOutput(
        NormalizingSpeechBackend(NeuTTS2EBackend()),
        sink=sink,
    )


class SpeechOutputService:
    """Own the restartable, resource-releasing speech-output lifecycle."""

    def __init__(
        self,
        output: VoiceOutput | None = None,
        *,
        output_factory: OutputFactory | None = None,
    ) -> None:
        if output is not None and output_factory is not None:
            raise ValueError("pass output or output_factory, not both")

        if output_factory is not None:
            self._output_factory = output_factory
        elif output is not None:
            # Existing unit-test/injection path. A supplied instance is one-shot;
            # callers that need disable -> re-enable should supply a factory.
            supplied = output
            used = False

            def one_shot_factory() -> VoiceOutput:
                nonlocal used
                if used:
                    raise RuntimeError(
                        "Injected speech output cannot be restarted; use output_factory"
                    )
                used = True
                return supplied

            self._output_factory = one_shot_factory
        else:
            self._output_factory = _production_output

        self.output: VoiceOutput | None = None
        self._stopping_output: VoiceOutput | None = None
        self._lock = threading.Lock()
        self._state = _OFF
        self._ready = False
        self._response_active = False
        self._generation = 0
        self._startup_thread: threading.Thread | None = None
        self._shutdown_thread: threading.Thread | None = None
        self._startup_error: str | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._state in {_STARTING, _READY}

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def startup_error(self) -> str | None:
        with self._lock:
            return self._startup_error

    def start_async(
        self,
        *,
        on_ready: ReadyCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> bool:
        """Warm Voice output without blocking the Host UI.

        Returns ``True`` only when a new warm-up was started. Voice may be
        disabled again while warm-up is in progress; that cancels/unloads the
        model-owning worker rather than waiting for warm-up to finish.
        """
        with self._lock:
            if self._state in {_STARTING, _READY, _STOPPING, _SHUTDOWN}:
                return False
            try:
                output = self._output_factory()
            except Exception as exc:
                self._startup_error = str(exc)
                callback_error = self._startup_error if on_error is not None else None
                output = None
            else:
                callback_error = None
                self._generation += 1
                generation = self._generation
                self.output = output
                self._state = _STARTING
                self._ready = False
                self._response_active = False
                self._startup_error = None
                thread = threading.Thread(
                    target=self._start_worker,
                    args=(generation, output, on_ready, on_error),
                    name="DexterTTSStartup",
                    daemon=True,
                )
                self._startup_thread = thread

        if output is None:
            if callback_error is not None and on_error is not None:
                on_error(callback_error)
            return False

        thread.start()
        return True

    def _start_worker(
        self,
        generation: int,
        output: VoiceOutput,
        on_ready: ReadyCallback | None,
        on_error: ErrorCallback | None,
    ) -> None:
        try:
            output.start()
        except Exception as exc:
            message = str(exc)
            with self._lock:
                current = (
                    generation == self._generation
                    and self._state == _STARTING
                    and self.output is output
                )
                if current:
                    self._ready = False
                    self._response_active = False
                    self._startup_error = message
                    self._state = _OFF
                    self.output = None
                    self._startup_thread = None
            if current:
                self._safe_stop(output)
                if on_error is not None:
                    on_error(message)
            return

        with self._lock:
            current = (
                generation == self._generation
                and self._state == _STARTING
                and self.output is output
            )
            if current:
                self._ready = True
                self._state = _READY
                self._startup_thread = None

        # If disable invalidated this warm-up, its shutdown worker owns final
        # cleanup. Do not resurrect readiness or emit a stale ready callback.
        if not current:
            return

        info = getattr(output.backend, "ready_info", None)
        if on_ready is not None:
            on_ready(info if isinstance(info, dict) else None)

    def disable_async(
        self,
        *,
        on_disabled: DisabledCallback | None = None,
    ) -> bool:
        """Immediately make speech inactive, then unload TTS off the UI thread."""
        immediate_callback = False
        with self._lock:
            if self._state == _SHUTDOWN:
                return False
            if self._state == _OFF:
                immediate_callback = True
                output = None
                startup_thread = None
            elif self._state == _STOPPING:
                return False
            else:
                self._generation += 1
                generation = self._generation
                self._ready = False
                self._response_active = False
                self._state = _STOPPING
                output = self.output
                self.output = None
                self._stopping_output = output
                startup_thread = self._startup_thread
                self._startup_thread = None
                thread = threading.Thread(
                    target=self._disable_worker,
                    args=(generation, output, startup_thread, on_disabled),
                    name="DexterTTSUnload",
                    daemon=True,
                )
                self._shutdown_thread = thread

        if immediate_callback:
            if on_disabled is not None:
                on_disabled()
            return False

        thread.start()
        return True

    def _disable_worker(
        self,
        generation: int,
        output: VoiceOutput | None,
        startup_thread: threading.Thread | None,
        on_disabled: DisabledCallback | None,
    ) -> None:
        if output is not None:
            self._safe_stop(output)

        # Stopping the backend interrupts its blocking startup read. Because this
        # runs away from Qt, wait until that retired thread really exits before
        # exposing OFF/re-enable; a fresh worker must never overlap an old one.
        if (
            startup_thread is not None
            and startup_thread.is_alive()
            and startup_thread is not threading.current_thread()
        ):
            startup_thread.join()

        # A late start/stop race can leave VoiceOutput in ERROR after the first
        # stop killed the child. A second best-effort stop closes any sink/thread
        # that managed to appear before the startup thread exited.
        if output is not None:
            self._safe_stop(output)

        with self._lock:
            current = generation == self._generation and self._state == _STOPPING
            if current:
                self._state = _OFF
                self._ready = False
                self._response_active = False
                self._stopping_output = None
                self._shutdown_thread = None

        if current and on_disabled is not None:
            on_disabled()

    def begin_response(self) -> bool:
        with self._lock:
            output = self.output
            if self._state != _READY or not self._ready or output is None:
                self._response_active = False
                return False
            self._response_active = True
        try:
            output.begin_response()
        except Exception:
            with self._lock:
                self._response_active = False
                still_ready = self._state == _READY and self.output is output
            if still_ready:
                raise
            return False
        return True

    def add_text(self, text: str) -> None:
        with self._lock:
            output = self.output
            active = (
                self._response_active
                and self._ready
                and self._state == _READY
                and output is not None
            )
        if active and output is not None:
            output.add_text(text)

    def finish_response(self) -> None:
        with self._lock:
            output = self.output
            active = (
                self._response_active
                and self._ready
                and self._state == _READY
                and output is not None
            )
            self._response_active = False
        if active and output is not None:
            output.finish_response()

    def interrupt(self) -> None:
        with self._lock:
            output = self.output
            ready = self._ready and self._state == _READY and output is not None
            self._response_active = False
        if ready and output is not None:
            output.interrupt()

    def stop(self) -> None:
        """Permanent Host shutdown; unlike disable_async(), this is not restartable."""
        with self._lock:
            if self._state == _SHUTDOWN:
                return
            self._generation += 1
            self._state = _SHUTDOWN
            self._ready = False
            self._response_active = False
            active_output = self.output
            stopping_output = self._stopping_output
            self.output = None
            self._stopping_output = None
            startup_thread = self._startup_thread
            shutdown_thread = self._shutdown_thread
            self._startup_thread = None
            self._shutdown_thread = None

        outputs = []
        for candidate in (active_output, stopping_output):
            if candidate is not None and all(candidate is not item for item in outputs):
                outputs.append(candidate)
                self._safe_stop(candidate)

        for thread in (startup_thread, shutdown_thread):
            if (
                thread is not None
                and thread.is_alive()
                and thread is not threading.current_thread()
            ):
                thread.join(timeout=3.0)

        for output in outputs:
            self._safe_stop(output)

    @staticmethod
    def _safe_stop(output: VoiceOutput) -> None:
        try:
            if output.state is not VoiceOutputState.STOPPED:
                output.stop()
        except Exception:
            # Disable/quit must remain best-effort even if a native backend is
            # already failing. NeuTTS2EBackend.stop() terminates/kills its child
            # on the failure path, so there is no benefit in surfacing this to Qt.
            pass

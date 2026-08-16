"""Streaming push-to-talk speech input for Dexter.

This module deliberately knows nothing about Goose, Qt, or the Dexter UI.
It owns only:

* the global Ctrl+/ push-to-talk boundary on Windows;
* continuous microphone capture;
* streaming Moonshine transcription while the key is held;
* draining all audio accepted before key-up;
* a short digital-silence post-roll so Moonshine can observe the endpoint;
* one forced final transcription pass; and
* callbacks for provisional text, finalized prompts, state changes, and errors.

Callbacks are invoked from background threads. A GUI adapter must marshal them
onto its UI thread (Qt queued signals are a good fit for that).
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from moonshine_voice import (
    Error,
    LineCompleted,
    LineTextChanged,
    TranscriptEventListener,
    Transcriber,
    get_model_for_language,
)

LOGGER = logging.getLogger(__name__)

# Windows virtual-key codes for Ctrl + /
_VK_CONTROL = 0x11
_VK_OEM_2 = 0xBF

# Internal worker messages.
_SHUTDOWN = object()


class VoiceState(StrEnum):
    """Coarse user-facing state of the speech input subsystem."""

    STOPPED = "stopped"
    LOADING = "loading"
    IDLE = "idle"
    LISTENING = "listening"
    FINALIZING = "finalizing"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Operational settings for Dexter's local push-to-talk input."""

    language: str = "en"
    update_interval_seconds: float = 0.5
    audio_block_size: int = 1024
    post_roll_seconds: float = 0.60
    keyterm_boost: str = "2.0"
    microphone: int | str | None = None
    hotkey_poll_seconds: float = 0.005
    debug_wav_dir: Path | None = None
    keyterms: tuple[str, ...] = (
        "Dexter",
        "Jarvis",
        "Ollama",
        "Gemma",
        "Gemma 4",
        "Goose",
        "MCP",
        "Dexter model gateway",
    )


@dataclass(frozen=True, slots=True)
class PromptResult:
    """A finalized transcription for exactly one physical PTT press."""

    session_id: int
    text: str
    captured_audio_seconds: float
    release_to_prompt_ms: float


PartialCallback = Callable[[int, str], None]
PromptCallback = Callable[[PromptResult], None]
StateCallback = Callable[[VoiceState], None]
ErrorCallback = Callable[[str], None]


def _noop_partial(_session_id: int, _text: str) -> None:
    pass


def _noop_prompt(_result: PromptResult) -> None:
    pass


def _noop_state(_state: VoiceState) -> None:
    pass


def _noop_error(_message: str) -> None:
    pass


@dataclass(slots=True)
class _Segment:
    start_time: float
    text: str = ""


@dataclass(slots=True)
class _Session:
    session_id: int
    key_down_time: float
    key_up_time: float | None = None
    captured_samples: int = 0
    segments: dict[int, _Segment] = field(default_factory=dict)
    debug_chunks: list[np.ndarray] = field(default_factory=list)
    error: BaseException | None = None

    def provisional_text(self) -> str:
        return " ".join(
            segment.text
            for segment in sorted(
                self.segments.values(),
                key=lambda item: item.start_time,
            )
            if segment.text
        ).strip()


class _SessionListener(TranscriptEventListener):
    """Translate Moonshine line events into one PTT-session draft."""

    def __init__(self, owner: "VoiceInput", session: _Session):
        self._owner = owner
        self._session = session
        self._last_draft = ""

    def _remember(self, line: Any) -> None:
        line_id = int(line.line_id)
        text = (line.text or "").strip()
        segment = self._session.segments.get(line_id)

        if segment is None:
            segment = _Segment(start_time=float(line.start_time))
            self._session.segments[line_id] = segment

        if text:
            segment.text = text

    def _emit_draft(self) -> None:
        text = self._session.provisional_text()
        if text and text != self._last_draft:
            self._last_draft = text
            self._owner._emit_partial(self._session.session_id, text)

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        self._remember(event.line)
        self._emit_draft()

    def on_line_completed(self, event: LineCompleted) -> None:
        # Moonshine line completion is only an internal pause/segment boundary.
        # Dexter's prompt boundary remains the physical key-down -> key-up span.
        self._remember(event.line)
        self._emit_draft()

    def on_error(self, event: Error) -> None:
        self._session.error = event.error
        self._owner._emit_error(f"Moonshine error: {event.error}")


class VoiceInput:
    """Local streaming push-to-talk input.

    The class is intentionally self-contained so the future Dexter host can own
    it as one component. `start()` loads Moonshine and opens the microphone;
    `stop()` tears everything down.

    The model is loaded once. Each PTT press gets a fresh Moonshine Stream while
    sharing that loaded model. Audio capture never performs inference: the
    PortAudio callback only copies a block into a FIFO queue, and one dedicated
    worker performs all Moonshine calls in order.
    """

    def __init__(
        self,
        config: VoiceConfig | None = None,
        *,
        on_partial: PartialCallback = _noop_partial,
        on_prompt: PromptCallback = _noop_prompt,
        on_state: StateCallback = _noop_state,
        on_error: ErrorCallback = _noop_error,
    ) -> None:
        self.config = config or VoiceConfig()
        self._on_partial = on_partial
        self._on_prompt = on_prompt
        self._on_state = on_state
        self._on_error = on_error

        self._state = VoiceState.STOPPED
        self._state_lock = threading.Lock()

        self._capture_lock = threading.Lock()
        self._current_capture_session: _Session | None = None
        self._held_session: _Session | None = None
        self._next_session_id = 0
        self._unfinalized_sessions = 0

        self._work_queue: queue.Queue[object] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._hotkey_thread: threading.Thread | None = None

        self._microphone_stream: sd.InputStream | None = None
        self._transcriber: Transcriber | None = None
        self._sample_rate: int | None = None
        self._user32: Any | None = None

    @property
    def state(self) -> VoiceState:
        with self._state_lock:
            return self._state

    @property
    def sample_rate(self) -> int | None:
        return self._sample_rate

    def start(self) -> None:
        """Load Moonshine, open the microphone, and begin global PTT polling."""

        if self.state is not VoiceState.STOPPED:
            raise RuntimeError("VoiceInput is already running")
        if sys.platform != "win32":
            raise RuntimeError("This PTT implementation currently targets Windows")

        self._set_state(VoiceState.LOADING)

        try:
            self._user32 = ctypes.windll.user32

            model_path, model_arch = get_model_for_language(self.config.language)
            self._transcriber = Transcriber(
                model_path=model_path,
                model_arch=model_arch,
                update_interval=self.config.update_interval_seconds,
                options={
                    "keyterm_boost": self.config.keyterm_boost,
                    # One local speaker; speaker ID adds work we do not use.
                    "identify_speakers": "false",
                },
            )
            self._transcriber.set_keyterms(list(self.config.keyterms))

            device_info = sd.query_devices(self.config.microphone, kind="input")
            self._sample_rate = int(device_info["default_samplerate"])

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="DexterSTTWorker",
                daemon=True,
            )
            self._worker_thread.start()

            self._microphone_stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self.config.audio_block_size,
                channels=1,
                dtype="float32",
                device=self.config.microphone,
                callback=self._audio_callback,
            )
            self._microphone_stream.start()

            self._hotkey_thread = threading.Thread(
                target=self._hotkey_loop,
                name="DexterPTTHotkey",
                daemon=True,
            )
            self._hotkey_thread.start()

            LOGGER.info(
                "Voice input ready: microphone=%s sample_rate=%s",
                device_info["name"],
                self._sample_rate,
            )
            self._set_state(VoiceState.IDLE)

        except Exception:
            self._set_state(VoiceState.ERROR)
            self._shutdown_resources()
            raise

    def stop(self) -> None:
        """Stop accepting input and tear down the voice subsystem cleanly."""

        if self.state is VoiceState.STOPPED:
            return

        self._stop_event.set()

        # If the application is closed while PTT is held, preserve the same
        # boundary semantics as a physical key-up.
        with self._capture_lock:
            session = self._current_capture_session
            self._current_capture_session = None
            self._held_session = None
            if session is not None:
                now = time.perf_counter()
                session.key_up_time = now
                self._work_queue.put(("end", session.session_id, now))

        if self._hotkey_thread is not None:
            self._hotkey_thread.join(timeout=1.0)

        if self._microphone_stream is not None:
            try:
                self._microphone_stream.stop()
            finally:
                self._microphone_stream.close()
            self._microphone_stream = None

        # FIFO ordering ensures any already-queued session end is handled first.
        self._work_queue.put(_SHUTDOWN)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10.0)
            if self._worker_thread.is_alive():
                LOGGER.warning("STT worker did not stop within 10 seconds")

        self._shutdown_resources()
        self._set_state(VoiceState.STOPPED)

    def _shutdown_resources(self) -> None:
        if self._transcriber is not None:
            try:
                self._transcriber.close()
            except Exception:
                LOGGER.exception("Failed to close Moonshine transcriber")
            self._transcriber = None

    def _set_state(self, state: VoiceState) -> None:
        with self._state_lock:
            if self._state is state:
                return
            self._state = state
        try:
            self._on_state(state)
        except Exception:
            LOGGER.exception("Voice state callback failed")

    def _emit_partial(self, session_id: int, text: str) -> None:
        try:
            self._on_partial(session_id, text)
        except Exception:
            LOGGER.exception("Voice partial callback failed")

    def _emit_prompt(self, result: PromptResult) -> None:
        try:
            self._on_prompt(result)
        except Exception:
            LOGGER.exception("Voice prompt callback failed")

    def _emit_error(self, message: str) -> None:
        LOGGER.error(message)
        try:
            self._on_error(message)
        except Exception:
            LOGGER.exception("Voice error callback failed")

    def _hotkey_is_down(self) -> bool:
        assert self._user32 is not None
        return bool(self._user32.GetAsyncKeyState(_VK_CONTROL) & 0x8000) and bool(
            self._user32.GetAsyncKeyState(_VK_OEM_2) & 0x8000
        )

    def _hotkey_loop(self) -> None:
        was_pressed = False

        while not self._stop_event.is_set():
            pressed = self._hotkey_is_down()

            if pressed and not was_pressed:
                self._begin_capture()
            elif not pressed and was_pressed:
                self._end_capture()

            was_pressed = pressed
            self._stop_event.wait(self.config.hotkey_poll_seconds)

    def _begin_capture(self) -> None:
        now = time.perf_counter()

        with self._capture_lock:
            self._next_session_id += 1
            self._unfinalized_sessions += 1
            session = _Session(
                session_id=self._next_session_id,
                key_down_time=now,
            )

            # START enters the FIFO before any audio callback can associate
            # blocks with this session.
            self._work_queue.put(("start", session))
            self._current_capture_session = session
            self._held_session = session

        self._set_state(VoiceState.LISTENING)

    def _end_capture(self) -> None:
        now = time.perf_counter()

        with self._capture_lock:
            session = self._held_session
            self._current_capture_session = None
            self._held_session = None

            if session is None:
                return

            session.key_up_time = now
            # This END event is FIFO-ordered after every audio block whose
            # callback acquired the capture lock before key-up.
            self._work_queue.put(("end", session.session_id, now))

        self._set_state(VoiceState.FINALIZING)

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        with self._capture_lock:
            session = self._current_capture_session
            if session is None:
                # The microphone intentionally stays open between PTT presses.
                # PortAudio may report discarded *idle* input while Moonshine is
                # doing finalization work; that audio does not belong to a Dexter
                # prompt and is therefore not a user-facing capture error.
                return

            if status:
                # Do not do logging/UI work in PortAudio's time-critical callback.
                # A status during an active PTT press matters because it can mean
                # samples from the logical prompt were discarded.
                self._work_queue.put(
                    ("audio_status", session.session_id, str(status))
                )

            chunk = indata[:, 0].astype(np.float32).copy()
            assert self._sample_rate is not None
            self._work_queue.put_nowait(
                ("audio", session.session_id, chunk, self._sample_rate)
            )

    def _worker_loop(self) -> None:
        active_session: _Session | None = None
        active_stream: Any | None = None
        pending_item: object | None = None

        try:
            while True:
                if pending_item is not None:
                    item = pending_item
                    pending_item = None
                else:
                    item = self._work_queue.get()

                if item is _SHUTDOWN:
                    break

                if not isinstance(item, tuple):
                    raise RuntimeError(f"Unexpected STT work item: {item!r}")

                kind = item[0]

                if kind == "audio_status":
                    sid, status_text = item[1], item[2]
                    self._emit_error(
                        f"Microphone status during S{sid:03d}: {status_text}"
                    )
                    continue

                if kind == "start":
                    session = item[1]
                    if active_session is not None:
                        raise RuntimeError(
                            "PTT session reached worker before previous session finalized"
                        )
                    if self._transcriber is None:
                        raise RuntimeError("Moonshine transcriber is not loaded")

                    active_session = session
                    active_stream = self._transcriber.create_stream(
                        update_interval=self.config.update_interval_seconds,
                        transcribe_flags=0,
                    )
                    active_stream.add_listener(_SessionListener(self, session))
                    active_stream.start()
                    continue

                if kind == "audio":
                    sid, first_chunk, sample_rate = item[1], item[2], item[3]
                    if active_session is None or active_stream is None:
                        raise RuntimeError("Audio reached STT worker without a session")
                    if active_session.session_id != sid:
                        raise RuntimeError(
                            f"Audio for S{sid} reached S{active_session.session_id}"
                        )

                    # Mirror Moonshine's microphone worker strategy: coalesce all
                    # immediately available audio for this same session/rate into
                    # one add_audio() call so inference cannot starve capture.
                    chunks = [first_chunk]
                    while True:
                        try:
                            queued = self._work_queue.get_nowait()
                        except queue.Empty:
                            break

                        if (
                            queued is not _SHUTDOWN
                            and isinstance(queued, tuple)
                            and queued[0] == "audio"
                            and queued[1] == sid
                            and queued[3] == sample_rate
                        ):
                            chunks.append(queued[2])
                        else:
                            pending_item = queued
                            break

                    combined = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
                    active_session.captured_samples += len(combined)
                    if self.config.debug_wav_dir is not None:
                        active_session.debug_chunks.append(combined)

                    active_stream.add_audio(combined, sample_rate)
                    continue

                if kind == "end":
                    sid, release_time = item[1], item[2]
                    if active_session is None or active_stream is None:
                        raise RuntimeError("Session end reached worker without a session")
                    if active_session.session_id != sid:
                        raise RuntimeError(
                            f"End for S{sid} reached S{active_session.session_id}"
                        )
                    if self._sample_rate is None:
                        raise RuntimeError("Microphone sample rate is unavailable")

                    # Every real block accepted before key-up has now passed
                    # through add_audio(). Give Moonshine clean acoustic right
                    # context, but do NOT add this synthetic silence to the
                    # captured/debug WAV or to Dexter's logical prompt boundary.
                    if self.config.post_roll_seconds > 0:
                        post_roll = np.zeros(
                            int(self._sample_rate * self.config.post_roll_seconds),
                            dtype=np.float32,
                        )
                        active_stream.add_audio(post_roll, self._sample_rate)

                    # Ensure end-of-prompt finalization is a real transcription
                    # pass rather than Moonshine's <200 ms cached fast path.
                    active_stream.set_transcribe_flags(
                        Transcriber.MOONSHINE_FLAG_FORCE_UPDATE
                    )
                    final_transcript = active_stream.stop()
                    prompt_ready_time = time.perf_counter()

                    if active_session.error is not None or final_transcript is None:
                        self._emit_error(
                            f"Speech finalization failed for session {sid}"
                        )
                    else:
                        text = _transcript_to_prompt(final_transcript)
                        if text:
                            result = PromptResult(
                                session_id=sid,
                                text=text,
                                captured_audio_seconds=(
                                    active_session.captured_samples / self._sample_rate
                                ),
                                release_to_prompt_ms=(
                                    prompt_ready_time - release_time
                                )
                                * 1000.0,
                            )
                            self._emit_prompt(result)

                    if self.config.debug_wav_dir is not None:
                        _save_debug_wav(
                            active_session,
                            self._sample_rate,
                            self.config.debug_wav_dir,
                        )

                    active_stream.close()
                    active_stream = None
                    active_session = None

                    # Preserve truthful state if the user has already started
                    # or even released a newer prompt while this one finalized.
                    with self._capture_lock:
                        self._unfinalized_sessions = max(
                            0, self._unfinalized_sessions - 1
                        )
                        newer_capture_exists = self._current_capture_session is not None
                        unfinalized_remain = self._unfinalized_sessions > 0

                    if newer_capture_exists:
                        self._set_state(VoiceState.LISTENING)
                    elif unfinalized_remain:
                        self._set_state(VoiceState.FINALIZING)
                    else:
                        self._set_state(VoiceState.IDLE)
                    continue

                raise RuntimeError(f"Unknown STT work item kind: {kind!r}")

        except Exception as exc:
            LOGGER.exception("STT worker failed")
            self._emit_error(f"Speech worker failed: {exc}")
            self._set_state(VoiceState.ERROR)

        finally:
            if active_stream is not None:
                try:
                    active_stream.close()
                except Exception:
                    LOGGER.exception("Failed to close active Moonshine stream")


def _transcript_to_prompt(transcript: Any) -> str:
    """Join all pause-separated Moonshine lines from one PTT press."""

    lines = sorted(
        transcript.lines,
        key=lambda line: (float(line.start_time), int(line.line_id)),
    )
    return " ".join(
        (line.text or "").strip()
        for line in lines
        if (line.text or "").strip()
    ).strip()


def _save_debug_wav(session: _Session, sample_rate: int, directory: Path) -> None:
    if not session.debug_chunks:
        return

    directory.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(session.debug_chunks)
    if audio.size == 0:
        return

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    path = directory / f"session_{session.session_id:03d}.wav"

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())

"""Dexter local host shell: real PTT/STT + overlay + tray.

This is the Stage 3 boundary:
- VoiceInput owns Ctrl+/ and Moonshine.
- Qt owns presentation, tray, and user-facing state.
- No Goose/ACP integration yet.

Ctrl+/ while idle:
    VoiceInput -> LISTENING -> overlay is restored/shown immediately.
"""

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

# Running "python Runtime\\Host\\app.py" puts Runtime\\Host on sys.path but not
# the Jarvis repository root. Add the repo root explicitly for Runtime.Voice.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Host.overlay import DexterOverlay
from Runtime.Voice.stt import PromptResult, VoiceInput, VoiceState

ICON_PATH = Path(__file__).resolve().parent / "assets" / "dexter.ico"


class VoiceQtBridge(QObject):
    """Thread-safe bridge from VoiceInput callbacks into Qt's GUI thread."""

    partial = Signal(int, str)
    prompt = Signal(object)
    state = Signal(object)
    error = Signal(str)
    startup_failed = Signal(str)
    shutdown_complete = Signal()


class DexterHost:
    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.overlay = DexterOverlay()
        self.bridge = VoiceQtBridge()
        self.voice = VoiceInput(
            on_partial=self.bridge.partial.emit,
            on_prompt=self.bridge.prompt.emit,
            on_state=self.bridge.state.emit,
            on_error=self.bridge.error.emit,
        )

        self._quitting = False
        self._startup_thread: threading.Thread | None = None

        self._configure_icon()
        self._configure_tray()
        self._connect_voice()
        self._configure_ctrl_c()

    def _configure_icon(self) -> None:
        icon = QIcon(str(ICON_PATH)) if ICON_PATH.is_file() else QIcon()
        if icon.isNull():
            icon = self.app.style().standardIcon(
                QStyle.StandardPixmap.SP_ComputerIcon
            )

        self.icon = icon
        self.app.setWindowIcon(icon)
        self.overlay.setWindowIcon(icon)

    def _configure_tray(self) -> None:
        self.tray_available = QSystemTrayIcon.isSystemTrayAvailable()

        self.tray = QSystemTrayIcon(self.icon, self.app)
        self.tray.setToolTip("Dexter")

        menu = QMenu()
        show_action = QAction("Show Dexter", menu)
        hide_action = QAction("Hide Dexter", menu)
        quit_action = QAction("Quit Dexter", menu)

        menu.addAction(show_action)
        menu.addAction(hide_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)

        show_action.triggered.connect(self.show_overlay)
        hide_action.triggered.connect(self.hide_overlay)
        quit_action.triggered.connect(self.request_quit)
        self.overlay.minimize_requested.connect(self.hide_overlay)

        self.tray.activated.connect(self._tray_activated)
        if self.tray_available:
            self.tray.show()

    def _connect_voice(self) -> None:
        self.bridge.partial.connect(self._on_partial)
        self.bridge.prompt.connect(self._on_prompt)
        self.bridge.state.connect(self._on_voice_state)
        self.bridge.error.connect(self._on_voice_error)
        self.bridge.startup_failed.connect(self._on_startup_failed)
        self.bridge.shutdown_complete.connect(self._finish_quit)

    def _configure_ctrl_c(self) -> None:
        # Qt's native event loop needs an explicit SIGINT handler plus a tiny
        # timer that periodically returns control to Python.
        signal.signal(signal.SIGINT, lambda _sig, _frame: self.request_quit())
        self._signal_pump = QTimer(self.app)
        self._signal_pump.setInterval(200)
        self._signal_pump.timeout.connect(lambda: None)
        self._signal_pump.start()

    def start(self) -> None:
        # Start hidden-to-tray. Ctrl+/ is the normal summon gesture.
        self.overlay.set_status("Loading speech…")
        self.hide_overlay()

        self._startup_thread = threading.Thread(
            target=self._start_voice_worker,
            name="DexterVoiceStartup",
            daemon=True,
        )
        self._startup_thread.start()

    def _start_voice_worker(self) -> None:
        try:
            self.voice.start()
        except Exception as exc:
            self.bridge.startup_failed.emit(str(exc))

    @Slot()
    def show_overlay(self) -> None:
        """Restore Dexter because the user explicitly asked to see it."""
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.activateWindow()

    def summon_overlay(self) -> None:
        """Reveal Dexter for PTT without stealing focus from the working app."""
        self.overlay.show()
        self.overlay.raise_()

    @Slot()
    def hide_overlay(self) -> None:
        if self.tray_available:
            self.overlay.hide()
        else:
            # Never make an always-on-top application inaccessible.
            self.overlay.showMinimized()

    @Slot(object)
    def _tray_activated(self, reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_overlay()

    @Slot(int, str)
    def _on_partial(self, _session_id: int, text: str) -> None:
        # Safety: if a platform focus/show quirk delayed the LISTENING state
        # update, the first actual speech partial also ensures Dexter is visible.
        self.summon_overlay()
        self.overlay.set_stt_partial(text)

    @Slot(object)
    def _on_prompt(self, result: PromptResult) -> None:
        self.summon_overlay()
        self.overlay.finalize_user_prompt(result.text)
        # Until ACP is wired, a finalized transcript is the end of this stage.
        self.overlay.set_status("Idle")

    @Slot(object)
    def _on_voice_state(self, state: VoiceState) -> None:
        if state is VoiceState.LOADING:
            self.overlay.set_status("Loading speech…")
        elif state is VoiceState.IDLE:
            self.overlay.set_status("Idle")
        elif state is VoiceState.LISTENING:
            # This is the Ctrl+/ summon path. VoiceInput is the only global
            # hotkey owner; the UI does not install a second listener.
            self.summon_overlay()
            self.overlay.set_stt_partial("")
            self.overlay.set_status("Listening…")
        elif state is VoiceState.FINALIZING:
            self.overlay.set_status("Finishing speech…")
        elif state is VoiceState.ERROR:
            self.show_overlay()
            self.overlay.set_status("Speech error")
        elif state is VoiceState.STOPPED and not self._quitting:
            self.overlay.set_status("Speech stopped")

    @Slot(str)
    def _on_voice_error(self, message: str) -> None:
        self.show_overlay()
        self.overlay.set_status("Speech error")
        # Stage 3 keeps errors visible but deliberately avoids adding a second
        # modal/error UI system. Console detail remains useful while developing.
        print(f"[VOICE ERROR] {message}", file=sys.stderr)

    @Slot(str)
    def _on_startup_failed(self, message: str) -> None:
        self.show_overlay()
        self.overlay.set_status("Speech failed to start")
        print(f"[VOICE STARTUP ERROR] {message}", file=sys.stderr)

    @Slot()
    def request_quit(self) -> None:
        if self._quitting:
            return

        self._quitting = True
        self.overlay.set_status("Stopping…")
        self.tray.hide()

        thread = threading.Thread(
            target=self._stop_voice_worker,
            name="DexterVoiceShutdown",
            daemon=True,
        )
        thread.start()

    def _stop_voice_worker(self) -> None:
        try:
            # If Quit is requested during model/microphone startup, do not race
            # VoiceInput.start() with VoiceInput.stop(). Wait for startup to
            # finish (successfully or not), then perform one orderly stop.
            startup = self._startup_thread
            if (
                startup is not None
                and startup.is_alive()
                and startup is not threading.current_thread()
            ):
                startup.join()

            self.voice.stop()
        finally:
            self.bridge.shutdown_complete.emit()

    @Slot()
    def _finish_quit(self) -> None:
        self.app.quit()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dexter")
    app.setQuitOnLastWindowClosed(False)

    host = DexterHost(app)
    host.start()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

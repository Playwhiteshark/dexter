"""Dexter local host: PTT/STT + overlay + persistent Goose ACP session.

Stage 4A:
- VoiceInput owns Ctrl+/ and Moonshine.
- Qt owns presentation/tray.
- GooseACP owns one ``goose acp`` child process + persistent session.
- Finalized speech is sent to Goose.
- AgentMessageChunk text streams into the Dexter response card.
- Permission requests are cancelled until Stage 4B wires the approval UI.
"""

from __future__ import annotations

import signal
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Host.gateway_process import GatewayProcess, GatewayState
from Runtime.Host.goose_acp import (
    GooseACP,
    GooseState,
    PermissionRequest,
)
from Runtime.Host.overlay import ApprovalOption, DexterOverlay
from Runtime.Voice.stt import PromptResult, VoiceInput, VoiceState

ICON_PATH = Path(__file__).resolve().parent / "assets" / "dexter.ico"
GATEWAY_PATH = REPO_ROOT / "Runtime" / "Model Gateway" / "gateway.py"


class VoiceQtBridge(QObject):
    """Thread-safe bridge from VoiceInput callbacks into Qt's GUI thread."""

    partial = Signal(int, str)
    prompt = Signal(object)
    state = Signal(object)
    error = Signal(str)
    startup_failed = Signal(str)
    shutdown_complete = Signal()


class GooseQtBridge(QObject):
    """Thread-safe bridge from GooseACP's asyncio thread into Qt."""

    state = Signal(object)
    response_start = Signal()
    response_text = Signal(str)
    response_end = Signal(str)
    error = Signal(str)
    permission_request = Signal(object)
    permission_cleared = Signal()
    chat_cleared = Signal()


class RuntimeQtBridge(QObject):
    """Thread-safe bridge for owned runtime child processes."""

    gateway_state = Signal(object)
    gateway_error = Signal(str)


class DexterHost:
    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.overlay = DexterOverlay()

        self.voice_bridge = VoiceQtBridge()
        self.voice = VoiceInput(
            on_partial=self.voice_bridge.partial.emit,
            on_prompt=self.voice_bridge.prompt.emit,
            on_state=self.voice_bridge.state.emit,
            on_error=self.voice_bridge.error.emit,
        )
        # Do not accept PTT until Gateway + Goose are ready.
        self.voice.set_capture_enabled(False)

        self.runtime_bridge = RuntimeQtBridge()
        self.gateway = GatewayProcess(
            GATEWAY_PATH,
            on_state=self.runtime_bridge.gateway_state.emit,
            on_error=self.runtime_bridge.gateway_error.emit,
            on_log=lambda line: print(f"[GATEWAY] {line}"),
        )

        self.goose_bridge = GooseQtBridge()
        self.goose = GooseACP(
            cwd=REPO_ROOT,
            on_state=self.goose_bridge.state.emit,
            on_response_start=self.goose_bridge.response_start.emit,
            on_response_text=self.goose_bridge.response_text.emit,
            on_response_end=self.goose_bridge.response_end.emit,
            on_error=self.goose_bridge.error.emit,
            on_permission_request=self.goose_bridge.permission_request.emit,
            on_permission_cleared=self.goose_bridge.permission_cleared.emit,
            on_chat_cleared=self.goose_bridge.chat_cleared.emit,
        )

        self._quitting = False
        self._agent_turn_pending = False
        self._runtime_startup_thread: threading.Thread | None = None
        self._voice_startup_thread: threading.Thread | None = None

        self._configure_icon()
        self._configure_tray()
        self._connect_voice()
        self._connect_goose()
        self._connect_runtime()
        self.overlay.prompt_submitted.connect(self._on_typed_prompt)
        self.overlay.cancel_requested.connect(self._cancel_agent_turn)
        self.overlay.approval_selected.connect(self._on_approval_selected)
        self.overlay.clear_chat_requested.connect(self._clear_chat)
        self.overlay.set_prompt_input_enabled(False)
        self.overlay.set_agent_busy(False)
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
        self.voice_bridge.partial.connect(self._on_partial)
        self.voice_bridge.prompt.connect(self._on_prompt)
        self.voice_bridge.state.connect(self._on_voice_state)
        self.voice_bridge.error.connect(self._on_voice_error)
        self.voice_bridge.startup_failed.connect(self._on_voice_startup_failed)
        self.voice_bridge.shutdown_complete.connect(self._finish_quit)

    def _connect_goose(self) -> None:
        self.goose_bridge.state.connect(self._on_goose_state)
        self.goose_bridge.response_start.connect(self._on_response_start)
        self.goose_bridge.response_text.connect(self._on_response_text)
        self.goose_bridge.response_end.connect(self._on_response_end)
        self.goose_bridge.error.connect(self._on_goose_error)
        self.goose_bridge.permission_request.connect(
            self._on_permission_request
        )
        self.goose_bridge.permission_cleared.connect(
            self.overlay.clear_approval
        )
        self.goose_bridge.chat_cleared.connect(self._on_chat_cleared)

    def _connect_runtime(self) -> None:
        self.runtime_bridge.gateway_state.connect(self._on_gateway_state)
        self.runtime_bridge.gateway_error.connect(self._on_gateway_error)

    def _configure_ctrl_c(self) -> None:
        signal.signal(signal.SIGINT, lambda _sig, _frame: self.request_quit())
        self._signal_pump = QTimer(self.app)
        self._signal_pump.setInterval(200)
        self._signal_pump.timeout.connect(lambda: None)
        self._signal_pump.start()

    def start(self) -> None:
        self.overlay.set_status("Starting Dexter…")
        self.hide_overlay()

        # Dexter owns the Gateway process. Goose ACP starts only after our own
        # child has successfully bound and reported ready.
        self._runtime_startup_thread = threading.Thread(
            target=self._start_runtime_worker,
            name="DexterRuntimeStartup",
            daemon=True,
        )
        self._runtime_startup_thread.start()

        # Moonshine model/microphone startup remains independent so PTT can be
        # ready while the model runtime finishes booting.
        self._voice_startup_thread = threading.Thread(
            target=self._start_voice_worker,
            name="DexterVoiceStartup",
            daemon=True,
        )
        self._voice_startup_thread.start()

    def _start_runtime_worker(self) -> None:
        try:
            self.gateway.start()
            if self._quitting:
                return
            self.goose.start()
        except Exception as exc:
            self.runtime_bridge.gateway_error.emit(str(exc))

    def _start_voice_worker(self) -> None:
        try:
            self.voice.start()
        except Exception as exc:
            self.voice_bridge.startup_failed.emit(str(exc))

    @Slot()
    def show_overlay(self) -> None:
        """Restore Dexter because the user explicitly asked to see it."""
        self.overlay.show()
        self.overlay.raise_()
        self.overlay.activateWindow()
        self.overlay.focus_prompt_input()

    def summon_overlay(self) -> None:
        """Reveal Dexter for PTT without stealing focus from the working app."""
        self.overlay.show()
        self.overlay.raise_()

    @Slot()
    def hide_overlay(self) -> None:
        if self.tray_available:
            self.overlay.hide()
        else:
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
        self.summon_overlay()
        self.overlay.set_stt_partial(text)

    @Slot(object)
    def _on_prompt(self, result: PromptResult) -> None:
        self.summon_overlay()
        self.overlay.finalize_user_prompt(result.text)
        self._submit_to_agent(result.text)

    @Slot(str)
    def _on_typed_prompt(self, text: str) -> None:
        prompt = text.strip()
        if not prompt:
            return
        self.show_overlay()
        self.overlay.finalize_user_prompt(prompt)
        self._submit_to_agent(prompt)

    def _submit_to_agent(self, prompt: str) -> None:
        if not self.goose.submit_prompt(prompt):
            self._agent_turn_pending = False
            self.overlay.set_agent_busy(False)
            self._refresh_interaction_admission()
            if self.gateway.state is not GatewayState.READY:
                self.overlay.set_status("Gateway unavailable")
            elif self.goose.state is GooseState.THINKING:
                self.overlay.set_status("Dexter is already responding")
            else:
                self.overlay.set_status("Dexter unavailable")
            return

        self._agent_turn_pending = True
        self.overlay.set_agent_busy(True)
        self._refresh_interaction_admission()
        self.overlay.set_status("Thinking…")

    def _refresh_interaction_admission(self) -> None:
        """Keep typed and voice input governed by the same turn boundary."""
        can_prompt = (
            not self._quitting
            and not self._agent_turn_pending
            and self.gateway.state is GatewayState.READY
            and self.goose.state is GooseState.IDLE
        )
        self.overlay.set_prompt_input_enabled(can_prompt)
        self.overlay.set_clear_chat_enabled(can_prompt)
        self.voice.set_capture_enabled(can_prompt)

    @Slot()
    def _cancel_agent_turn(self) -> None:
        if not self._agent_turn_pending:
            return

        self.overlay.set_cancel_enabled(False)
        self.overlay.set_status("Cancelling…")

        if not self.goose.cancel_current():
            self.overlay.set_cancel_enabled(True)
            if self.goose.state is GooseState.IDLE:
                self._agent_turn_pending = False
                self.overlay.set_agent_busy(False)
                self._refresh_interaction_admission()
                self.overlay.set_status("Idle")

    @Slot(object)
    def _on_voice_state(self, state: VoiceState) -> None:
        self.overlay.set_recording(state is VoiceState.LISTENING)

        if state is VoiceState.LOADING:
            if self.goose.state is GooseState.STARTING:
                self.overlay.set_status("Starting Dexter…")
            else:
                self.overlay.set_status("Loading speech…")
        elif state is VoiceState.IDLE:
            # STT returns to IDLE immediately after emitting the final prompt.
            # Preserve Thinking while that prompt is queued or running.
            if (
                not self._agent_turn_pending
                and self.goose.state
                not in (GooseState.THINKING, GooseState.ERROR)
            ):
                self.overlay.set_status("Idle")
        elif state is VoiceState.LISTENING:
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

    @Slot(object)
    def _on_goose_state(self, state: GooseState) -> None:
        if state is GooseState.STARTING:
            self._refresh_interaction_admission()
            if self.voice.state is not VoiceState.LISTENING:
                self.overlay.set_status("Starting Dexter…")
        elif state is GooseState.IDLE:
            self._refresh_interaction_admission()
            if self.voice.state is VoiceState.IDLE and not self._agent_turn_pending:
                self.overlay.set_status("Idle")
        elif state is GooseState.RESETTING:
            self._refresh_interaction_admission()
            self.overlay.set_status("Clearing chat…")
        elif state is GooseState.THINKING:
            self._agent_turn_pending = True
            self.overlay.set_agent_busy(True)
            self._refresh_interaction_admission()
            self.overlay.set_status("Thinking…")
        elif state is GooseState.ERROR:
            self.overlay.set_agent_busy(False)
            self._refresh_interaction_admission()
            self.show_overlay()
            self.overlay.set_status("Agent error")
        elif state is GooseState.STOPPED and not self._quitting:
            self.overlay.set_agent_busy(False)
            self._refresh_interaction_admission()
            self.overlay.set_status("Agent stopped")

    @Slot()
    def _clear_chat(self) -> None:
        if self._agent_turn_pending:
            return

        if self.goose.new_chat():
            self.overlay.set_status("Clearing chat…")
            self._refresh_interaction_admission()
        else:
            self.overlay.set_status("Can't clear chat right now")

    @Slot()
    def _on_chat_cleared(self) -> None:
        self._agent_turn_pending = False
        self.overlay.clear_history()
        self.overlay.set_agent_busy(False)
        self._refresh_interaction_admission()
        self.overlay.set_status("Idle")
        self.overlay.focus_prompt_input()

    @Slot()
    def _on_response_start(self) -> None:
        self._agent_turn_pending = True
        self.overlay.set_agent_busy(True)
        self._refresh_interaction_admission()
        self.summon_overlay()
        self.overlay.begin_assistant_response()
        self.overlay.set_status("Thinking…")

    @Slot(str)
    def _on_response_text(self, text: str) -> None:
        self.summon_overlay()
        self.overlay.append_assistant_text(text)

    @Slot(str)
    def _on_response_end(self, stop_reason: str) -> None:
        self._agent_turn_pending = False
        self.overlay.clear_approval()
        self.overlay.finish_assistant_response()
        self.overlay.set_agent_busy(False)
        # Goose transitions to IDLE immediately after this callback; until that
        # state arrives, admission intentionally remains closed.
        self._refresh_interaction_admission()

        if stop_reason == "end_turn":
            self.overlay.set_status("Idle")
        elif stop_reason == "cancelled":
            self.overlay.set_status("Cancelled")
        elif stop_reason == "error":
            self.overlay.set_status("Agent error")
        else:
            self.overlay.set_status(f"Stopped: {stop_reason}")

    @Slot(object)
    def _on_gateway_state(self, state: GatewayState) -> None:
        if state is GatewayState.STARTING:
            if self.voice.state is not VoiceState.LISTENING:
                self.overlay.set_status("Starting gateway…")
        elif state is GatewayState.READY:
            self._refresh_interaction_admission()
            if self.goose.state is GooseState.STOPPED:
                self.overlay.set_status("Starting agent…")
        elif state is GatewayState.ERROR:
            self.overlay.set_agent_busy(False)
            self._refresh_interaction_admission()
            self.show_overlay()
            self.overlay.set_status("Gateway error")
        elif state is GatewayState.STOPPED and not self._quitting:
            self.overlay.set_agent_busy(False)
            self._refresh_interaction_admission()
            self.overlay.set_status("Gateway stopped")

    @Slot(str)
    def _on_gateway_error(self, message: str) -> None:
        self._agent_turn_pending = False
        self.overlay.set_agent_busy(False)
        self._refresh_interaction_admission()
        self.show_overlay()
        self.overlay.finish_assistant_response()
        self.overlay.set_status("Gateway error")
        print(f"[GATEWAY ERROR] {message}", file=sys.stderr)

        # If the owned Gateway dies after Goose is running, stop the ACP side
        # too rather than leaving a half-alive Dexter runtime.
        threading.Thread(
            target=self.goose.stop,
            name="DexterGooseStopAfterGatewayFailure",
            daemon=True,
        ).start()

    @Slot(object)
    def _on_permission_request(self, request: PermissionRequest) -> None:
        self.show_overlay()
        self.overlay.set_status("Approval required")

        options = [
            ApprovalOption(
                option_id=choice.option_id,
                label=choice.label,
                emphasized=False,
            )
            for choice in request.options
        ]
        self.overlay.show_approval(
            request.title,
            request.detail,
            options,
        )

    @Slot(str)
    def _on_approval_selected(self, option_id: str) -> None:
        # The card only contains ids supplied by the current ACP request.
        # GooseACP validates the id again before replying to the protocol.
        if self.goose.resolve_permission(option_id):
            self.overlay.set_status("Thinking…")
            self.overlay.resume_thinking_indicator()
        else:
            self.overlay.set_status("Approval expired")
            self.overlay.clear_approval()

    @Slot(str)
    def _on_voice_error(self, message: str) -> None:
        self.show_overlay()
        self.overlay.set_status("Speech error")
        print(f"[VOICE ERROR] {message}", file=sys.stderr)

    @Slot(str)
    def _on_voice_startup_failed(self, message: str) -> None:
        self.show_overlay()
        self.overlay.set_status("Speech failed to start")
        print(f"[VOICE STARTUP ERROR] {message}", file=sys.stderr)

    @Slot(str)
    def _on_goose_error(self, message: str) -> None:
        self._agent_turn_pending = False
        self.overlay.clear_approval()
        self.overlay.set_agent_busy(False)
        self._refresh_interaction_admission()
        self.show_overlay()
        self.overlay.finish_assistant_response()
        self.overlay.set_status("Agent error")
        print(f"[GOOSE ERROR] {message}", file=sys.stderr)

    @Slot()
    def request_quit(self) -> None:
        if self._quitting:
            return

        self._quitting = True
        self.overlay.clear_approval()
        self.overlay.set_agent_busy(False)
        self._refresh_interaction_admission()
        self.overlay.set_status("Stopping…")
        self.tray.hide()

        thread = threading.Thread(
            target=self._stop_runtime_worker,
            name="DexterRuntimeShutdown",
            daemon=True,
        )
        thread.start()

    def _stop_runtime_worker(self) -> None:
        try:
            voice_startup = self._voice_startup_thread
            if (
                voice_startup is not None
                and voice_startup.is_alive()
                and voice_startup is not threading.current_thread()
            ):
                voice_startup.join()

            self.voice.stop()

            # Stop consumers before the endpoint they depend on.
            self.goose.stop()

            # If Gateway startup is still in progress, stop() interrupts the
            # child; the startup worker will then unwind without adopting any
            # outside process.
            self.gateway.stop()

            runtime_startup = self._runtime_startup_thread
            if (
                runtime_startup is not None
                and runtime_startup.is_alive()
                and runtime_startup is not threading.current_thread()
            ):
                runtime_startup.join(timeout=2.0)
        finally:
            self.voice_bridge.shutdown_complete.emit()

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

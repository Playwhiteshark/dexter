"""Standalone Stage 2 visual/interaction demo for DexterOverlay.

Fake events only. There is no Moonshine, Goose, ACP, or tool execution.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QStyle,
    QSystemTrayIcon,
)

from overlay import ApprovalOption, DexterOverlay


def schedule(delay_ms: int, callback) -> None:
    QTimer.singleShot(delay_ms, callback)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dexter Overlay Demo")
    # Hiding the overlay to the tray must not terminate the process.
    app.setQuitOnLastWindowClosed(False)

    # Qt's native event loop can otherwise make Ctrl+C look like it was
    # received without actually terminating the application. Install an
    # explicit SIGINT handler and periodically return to Python so the signal
    # is processed promptly.
    signal.signal(signal.SIGINT, lambda _sig, _frame: app.quit())
    signal_pump = QTimer()
    signal_pump.setInterval(200)
    signal_pump.timeout.connect(lambda: None)
    signal_pump.start()

    overlay = DexterOverlay()

    tray_available = QSystemTrayIcon.isSystemTrayAvailable()
    tray = QSystemTrayIcon()
    tray.setToolTip("Dexter")

    icon_path = Path(__file__).resolve().parent / "assets" / "dexter.ico"
    icon = QIcon(str(icon_path)) if icon_path.is_file() else QIcon()
    if icon.isNull():
        icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)

    app.setWindowIcon(icon)
    overlay.setWindowIcon(icon)
    tray.setIcon(icon)

    tray_menu = QMenu()
    show_action = QAction("Show Dexter", tray_menu)
    hide_action = QAction("Hide Dexter", tray_menu)
    quit_action = QAction("Quit Dexter", tray_menu)

    tray_menu.addAction(show_action)
    tray_menu.addAction(hide_action)
    tray_menu.addSeparator()
    tray_menu.addAction(quit_action)
    tray.setContextMenu(tray_menu)

    def show_overlay() -> None:
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()

    def hide_overlay() -> None:
        if tray_available:
            overlay.hide()
        else:
            # Safety fallback: never make the demo inaccessible if a system
            # tray is unavailable for some reason.
            overlay.showMinimized()

    def quit_app() -> None:
        tray.hide()
        app.quit()

    show_action.triggered.connect(show_overlay)
    hide_action.triggered.connect(hide_overlay)
    quit_action.triggered.connect(quit_app)
    overlay.minimize_requested.connect(hide_overlay)

    def tray_activated(reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            show_overlay()

    tray.activated.connect(tray_activated)

    if tray_available:
        tray.show()

    show_overlay()

    partials = [
        "Check whether…",
        "Check whether Ollama is…",
        "Check whether Ollama is running through the Dexter…",
        "Check whether Ollama is running through the Dexter Model Gateway.",
    ]

    overlay.set_status("Listening…")
    for index, partial in enumerate(partials):
        schedule(500 + index * 420, lambda text=partial: overlay.set_stt_partial(text))

    def finalize_prompt() -> None:
        overlay.finalize_user_prompt(partials[-1])
        overlay.set_status("Thinking…")
        overlay.begin_assistant_response()

    schedule(2300, finalize_prompt)

    chunks = [
        "The Gateway is reachable on port 11435. ",
        "I would need permission before demonstrating an effectful action.",
    ]
    schedule(2850, lambda: overlay.append_assistant_text(chunks[0]))
    schedule(3400, lambda: overlay.append_assistant_text(chunks[1]))

    def request_approval() -> None:
        overlay.finish_assistant_response()
        overlay.set_status("Waiting for approval…")
        overlay.show_approval(
            "Dexter wants permission",
            "Move config.py into the Review folder.",
            [
                ApprovalOption("allow_once", "Allow once", emphasized=True),
                ApprovalOption("reject_once", "Reject"),
            ],
        )

    schedule(4100, request_approval)

    def approval_selected(option_id: str) -> None:
        if option_id == "allow_once":
            overlay.set_status("Approved")
        else:
            overlay.set_status("Rejected")
        schedule(900, lambda: overlay.set_status("Idle"))

    overlay.approval_selected.connect(approval_selected)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

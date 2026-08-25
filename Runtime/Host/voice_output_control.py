"""Secondary UI control for Dexter's expensive Voice-output resource mode.

This is intentionally not presented as a normal speaker mute. Turning Voice off
unloads the TTS worker/model and turning it back on requires a warm-up, so the
control lives behind the header overflow menu rather than beside Send/PTT.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QLabel, QMenu, QToolButton, QWidget

VoiceToggleCallback = Callable[[bool], None]


class VoiceOutputControl(QObject):
    """Install and maintain Dexter's secondary Voice-output toggle."""

    def __init__(
        self,
        overlay: QWidget,
        *,
        enabled: bool,
        on_toggle: VoiceToggleCallback,
    ) -> None:
        super().__init__(overlay)
        self._on_toggle = on_toggle

        header = overlay.findChild(QFrame, "header")
        if header is None or header.layout() is None:
            raise RuntimeError("Dexter overlay header is unavailable")
        layout = header.layout()

        self.menu = QMenu(overlay)
        self.menu.setToolTipsVisible(True)

        self.voice_action = QAction("Voice output", self.menu)
        self.voice_action.setCheckable(True)
        self.voice_action.setToolTip(
            "Turn Dexter speech generation on or off. Turning it off unloads "
            "the TTS model to free resources; turning it back on requires warm-up."
        )
        self.voice_action.toggled.connect(self._requested)
        self.menu.addAction(self.voice_action)

        self.voice_status = QLabel(header)
        self.voice_status.setObjectName("voiceStatus")
        self.voice_status.setAccessibleName("Dexter Voice status")
        self.voice_status.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.voice_status.setStyleSheet(
            """
            QLabel#voiceStatus {
                color: rgba(255, 255, 255, 115);
                font-size: 10px;
                padding: 0px 2px;
            }
            QLabel#voiceStatus[state="ready"] {
                color: rgba(190, 224, 194, 175);
            }
            QLabel#voiceStatus[state="starting"],
            QLabel#voiceStatus[state="stopping"] {
                color: rgba(255, 226, 165, 175);
            }
            QLabel#voiceStatus[state="error"] {
                color: rgba(255, 176, 176, 205);
            }
            """
        )

        self.more_button = QToolButton(header)
        self.more_button.setObjectName("moreButton")
        # Use one real centered arrow instead of drawing an ellipsis alongside
        # QToolButton's automatic menu indicator. The menu-indicator subcontrol
        # is suppressed below so Qt cannot add a second off-centre arrow.
        self.more_button.setArrowType(Qt.ArrowType.DownArrow)
        self.more_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.more_button.setToolTip("Dexter options")
        self.more_button.setAccessibleName("Dexter options")
        self.more_button.setAutoRaise(True)
        self.more_button.setFixedSize(24, 20)
        self.more_button.setMenu(self.menu)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setStyleSheet(
            """
            QToolButton#moreButton {
                background: transparent;
                border: none;
                border-radius: 5px;
                color: rgba(255, 255, 255, 155);
                padding: 0px;
            }
            QToolButton#moreButton:hover {
                background-color: rgba(255, 255, 255, 24);
                color: rgba(255, 255, 255, 235);
            }
            QToolButton#moreButton:pressed {
                background-color: rgba(255, 255, 255, 38);
            }
            QToolButton#moreButton::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            """
        )

        # Keep the expensive resource toggle secondary: status -> Voice state ->
        # overflow -> ordinary Clear/minimize commands. The label makes warm-up
        # visible without turning the resource toggle itself into a primary action.
        clear_button = overlay.findChild(QToolButton, "clearButton")
        insert_at = layout.indexOf(clear_button) if clear_button is not None else -1
        if insert_at >= 0:
            layout.insertWidget(insert_at, self.voice_status)
            layout.insertWidget(insert_at + 1, self.more_button)
        else:
            layout.addWidget(self.voice_status)
            layout.addWidget(self.more_button)

        self.set_state("starting" if enabled else "off")

    def _requested(self, enabled: bool) -> None:
        self._on_toggle(bool(enabled))

    def set_state(self, state: str) -> None:
        """Reflect lifecycle state without re-triggering the toggle callback."""
        if state not in {"ready", "starting", "off", "stopping", "error"}:
            raise ValueError(f"unknown Voice-output state: {state}")

        self.voice_status.setProperty("state", state)
        if state == "ready":
            self.voice_status.setText("Voice ready")
            self.voice_status.setToolTip("Dexter Voice output is ready")
        elif state == "starting":
            self.voice_status.setText("Voice loading…")
            self.voice_status.setToolTip("Dexter Voice output is warming up")
        elif state == "stopping":
            self.voice_status.setText("Voice unloading…")
            self.voice_status.setToolTip("Dexter Voice output is releasing resources")
        elif state == "error":
            self.voice_status.setText("Voice error")
            self.voice_status.setToolTip("Dexter Voice output failed to start")
        else:
            self.voice_status.setText("Voice off")
            self.voice_status.setToolTip("Dexter Voice output is disabled")
        self.voice_status.style().unpolish(self.voice_status)
        self.voice_status.style().polish(self.voice_status)

        previous = self.voice_action.blockSignals(True)
        try:
            if state == "ready":
                self.voice_action.setChecked(True)
                self.voice_action.setEnabled(True)
                self.voice_action.setText("Voice output")
                self.more_button.setToolTip("Dexter options · Voice ready")
            elif state == "starting":
                # Remain actionable so an accidental enable can be cancelled
                # during the expensive warm-up instead of forcing the user to wait.
                self.voice_action.setChecked(True)
                self.voice_action.setEnabled(True)
                self.voice_action.setText("Voice output · starting…")
                self.more_button.setToolTip(
                    "Dexter options · Voice starting (turn off to cancel)"
                )
            elif state == "stopping":
                self.voice_action.setChecked(False)
                self.voice_action.setEnabled(False)
                self.voice_action.setText("Voice output · stopping…")
                self.more_button.setToolTip("Dexter options · unloading Voice")
            elif state == "error":
                self.voice_action.setChecked(False)
                self.voice_action.setEnabled(True)
                self.voice_action.setText("Voice output · retry")
                self.more_button.setToolTip("Dexter options · Voice failed to start")
            else:
                self.voice_action.setChecked(False)
                self.voice_action.setEnabled(True)
                self.voice_action.setText("Voice output")
                self.more_button.setToolTip("Dexter options · Voice off")
        finally:
            self.voice_action.blockSignals(previous)

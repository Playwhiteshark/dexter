"""Compact staged runtime-readiness surface for Dexter's Host.

The Host already owns the real lifecycle state for Gateway, Goose, Moonshine and
Voice output. This module only presents those existing states; it deliberately
does not introduce a second runtime state machine or change interaction gating.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

LayoutChangedCallback = Callable[[], None]

_COMPONENT_LABELS = {
    "gateway": "Gateway",
    "agent": "Agent",
    "mic": "Mic",
    "voice": "Voice",
}
_STATE_LABELS = {
    "waiting": "Waiting",
    "starting": "Starting…",
    "ready": "Ready",
    "off": "Off",
    "stopping": "Stopping…",
    "error": "Failed",
    "stopped": "Stopped",
}
_ACTIVE_STATES = {"waiting", "starting", "stopping"}
_FAILURE_STATES = {"error", "stopped"}


def _compact_reason(message: str, *, limit: int = 320) -> str:
    """Keep the useful first error paragraph without dumping terminal logs."""
    raw = str(message or "").strip()
    if not raw:
        return ""

    paragraph = raw.split("\n\n", 1)[0]
    compact = " ".join(
        part.strip()
        for part in paragraph.splitlines()
        if part.strip()
    )
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


class _ReadinessRow(QFrame):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.setObjectName("readinessRow")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.name_label = QLabel(_COMPONENT_LABELS[component])
        self.name_label.setObjectName("readinessName")

        self.state_label = QLabel()
        self.state_label.setObjectName("readinessState")
        self.state_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self.reason_label = QLabel()
        self.reason_label.setObjectName("readinessReason")
        self.reason_label.setWordWrap(True)
        self.reason_label.setTextFormat(Qt.TextFormat.PlainText)
        self.reason_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.reason_label.hide()

        top.addWidget(self.name_label)
        top.addStretch(1)
        top.addWidget(self.state_label)
        layout.addLayout(top)
        layout.addWidget(self.reason_label)

    def set_state(
        self,
        state: str,
        reason: str = "",
        full_reason: str = "",
    ) -> None:
        self.state_label.setText(_STATE_LABELS[state])
        self.state_label.setProperty("state", state)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

        self.reason_label.setText(reason)
        tooltip = full_reason if full_reason and full_reason != reason else ""
        self.reason_label.setToolTip(tooltip)
        self.reason_label.setVisible(bool(reason))


class RuntimeReadinessView(QFrame):
    """Present Gateway / agent / mic / Voice startup and failure state."""

    def __init__(
        self,
        overlay: QWidget,
        *,
        on_layout_changed: LayoutChangedCallback | None = None,
    ) -> None:
        super().__init__(overlay)
        self.setObjectName("readinessPanel")
        self._on_layout_changed = on_layout_changed
        self._states = {
            component: "waiting" for component in _COMPONENT_LABELS
        }
        self._reasons = {component: "" for component in _COMPONENT_LABELS}
        self._full_reasons = {
            component: "" for component in _COMPONENT_LABELS
        }
        self._rows: dict[str, _ReadinessRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(5)

        self.title_label = QLabel("Starting Dexter")
        self.title_label.setObjectName("readinessTitle")
        layout.addWidget(self.title_label)

        for component in _COMPONENT_LABELS:
            row = _ReadinessRow(component)
            self._rows[component] = row
            layout.addWidget(row)

        self.setStyleSheet(
            """
            QFrame#readinessPanel {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 9px;
            }
            QLabel#readinessTitle {
                color: rgba(255, 255, 255, 185);
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#readinessName {
                color: rgba(255, 255, 255, 185);
                font-size: 10px;
            }
            QLabel#readinessState {
                color: rgba(255, 255, 255, 125);
                font-size: 10px;
            }
            QLabel#readinessState[state="ready"] {
                color: rgba(190, 224, 194, 180);
            }
            QLabel#readinessState[state="starting"],
            QLabel#readinessState[state="waiting"],
            QLabel#readinessState[state="stopping"] {
                color: rgba(255, 226, 165, 180);
            }
            QLabel#readinessState[state="error"],
            QLabel#readinessState[state="stopped"] {
                color: rgba(255, 176, 176, 215);
            }
            QLabel#readinessReason {
                color: rgba(255, 196, 196, 205);
                font-size: 10px;
                padding: 0px 0px 2px 8px;
            }
            """
        )

        panel = getattr(overlay, "panel", None)
        scroll = getattr(overlay, "scroll", None)
        panel_layout = panel.layout() if panel is not None else None
        if panel_layout is None or scroll is None:
            raise RuntimeError("Dexter overlay layout is unavailable")

        insert_at = panel_layout.indexOf(scroll)
        if insert_at < 0:
            raise RuntimeError("Dexter transcript is unavailable")
        panel_layout.insertWidget(insert_at, self)

        for component in _COMPONENT_LABELS:
            self._rows[component].set_state("waiting")
        self._refresh_visibility()

    def begin_startup(self, *, voice_enabled: bool) -> None:
        self.set_component("gateway", "starting")
        self.set_component("agent", "waiting")
        self.set_component("mic", "starting")
        self.set_component("voice", "starting" if voice_enabled else "off")

    def state(self, component: str) -> str:
        self._validate_component(component)
        return self._states[component]

    def reason(self, component: str) -> str:
        self._validate_component(component)
        return self._full_reasons[component]

    def set_component(
        self,
        component: str,
        state: str,
        *,
        reason: str | None = None,
    ) -> None:
        self._validate_component(component)
        if state not in _STATE_LABELS:
            raise ValueError(f"unknown readiness state: {state}")

        if state == "error":
            if reason is not None:
                full_reason = str(reason).strip()
                self._full_reasons[component] = full_reason
                self._reasons[component] = _compact_reason(full_reason)
        else:
            self._full_reasons[component] = ""
            self._reasons[component] = ""

        self._states[component] = state
        self._rows[component].set_state(
            state,
            self._reasons[component],
            self._full_reasons[component],
        )
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        states = set(self._states.values())
        has_failure = bool(states & _FAILURE_STATES)
        has_transition = bool(states & _ACTIVE_STATES)
        visible = has_failure or has_transition

        self.title_label.setText(
            "Dexter needs attention" if has_failure else "Starting Dexter"
        )
        # Dexter normally starts with its parent overlay hidden. isVisible()
        # therefore cannot tell us whether this child is explicitly hidden;
        # always set the requested child visibility so a settled panel does not
        # reappear the next time the tray window is opened.
        self.setVisible(visible)
        self.updateGeometry()
        if self._on_layout_changed is not None:
            self._on_layout_changed()

    @staticmethod
    def _validate_component(component: str) -> None:
        if component not in _COMPONENT_LABELS:
            raise ValueError(f"unknown readiness component: {component}")


__all__ = ["RuntimeReadinessView"]

"""Minimal always-on-top Dexter overlay for Stage 2.

Presentation-only: no Moonshine, Goose, ACP, or Dexter tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from PySide6.QtCore import QPoint, QPointF, QSize, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from Runtime.Host.thinking_messages import THINKING_MESSAGES
from Runtime.Host.activity_events import ActivityRecord, ActivityUpdate


@dataclass(frozen=True, slots=True)
class ApprovalOption:
    """One choice from a future ACP permission request."""

    option_id: str
    label: str
    emphasized: bool = False


class _DragHandle(QFrame):
    """Header region that delegates movement to the OS window manager."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window().windowHandle()
            if window is not None and window.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)


class _ThinkingWave(QWidget):
    """Small animated status line with a sine-wave character jiggle."""

    _FRAME_MS = 55
    _MESSAGE_MS = 3400
    _AMPLITUDE = 0.8

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("thinkingWave")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setMinimumHeight(24)

        self._messages = tuple(THINKING_MESSAGES) or ("Thinking…",)
        self._message = random.choice(self._messages)
        self._phase = 0.0

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(self._FRAME_MS)
        self._animation_timer.timeout.connect(self._advance_wave)

        self._message_timer = QTimer(self)
        self._message_timer.setInterval(self._MESSAGE_MS)
        self._message_timer.timeout.connect(self._rotate_message)

        self.hide()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(240, 24)

    def start(self) -> None:
        if not self.isVisible():
            self._rotate_message()
            self.show()
        if not self._animation_timer.isActive():
            self._animation_timer.start()
        if not self._message_timer.isActive():
            self._message_timer.start()

    def stop(self) -> None:
        self._animation_timer.stop()
        self._message_timer.stop()
        self.hide()

    def _advance_wave(self) -> None:
        self._phase += 0.34
        self.update()

    def _rotate_message(self) -> None:
        if len(self._messages) == 1:
            self._message = self._messages[0]
        else:
            choices = [m for m in self._messages if m != self._message]
            self._message = random.choice(choices)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setPen(QColor(255, 255, 255, 150))

        font = self.font()
        font.setPointSizeF(max(8.5, font.pointSizeF() - 0.5))
        painter.setFont(font)
        metrics = painter.fontMetrics()

        x = 2.0
        baseline = (
            (self.height() - metrics.height()) / 2.0
            + metrics.ascent()
        )

        available = max(0, self.width() - 4)
        for index, char in enumerate(self._message):
            width = metrics.horizontalAdvance(char)
            if x + width > available:
                break

            offset = math.sin(self._phase + index * 0.55) * self._AMPLITUDE
            painter.drawText(QPointF(x, baseline + offset), char)
            x += width

        painter.end()


class _ActivityEntryCard(QFrame):
    """One compact, selectable activity record inside an expanded response."""

    def __init__(self, record: ActivityRecord) -> None:
        super().__init__()
        self.setObjectName("activityEntry")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 7)
        layout.setSpacing(3)

        self.title_label = QLabel()
        self.title_label.setObjectName("activityEntryTitle")
        self.title_label.setWordWrap(True)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("activityEntryDetail")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextFormat(Qt.TextFormat.PlainText)
        self.detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.detail_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)
        self.update_record(record)

    def update_record(self, record: ActivityRecord) -> None:
        self.title_label.setText(record.heading())
        detail = record.details()
        self.detail_label.setText(detail)
        self.detail_label.setVisible(bool(detail))


class _MessageCard(QFrame):
    _MAX_ACTIVITY_RECORDS = 40

    def __init__(
        self,
        role: str,
        text: str = "",
        *,
        draft: bool = False,
        activity_enabled: bool = False,
        render_markdown: bool = False,
    ) -> None:
        super().__init__()
        self.setObjectName("messageCard")
        self.setProperty("draft", draft)

        self._activity_enabled = activity_enabled
        self._render_markdown = render_markdown
        self._raw_text = ""
        self._activity_records: dict[str, ActivityRecord] = {}
        self._activity_widgets: dict[str, _ActivityEntryCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(3)

        self.role_label = QLabel(role)
        self.role_label.setObjectName("roleLabel")

        role_row = QHBoxLayout()
        role_row.setContentsMargins(0, 0, 0, 0)
        role_row.setSpacing(6)
        role_row.addWidget(self.role_label)
        role_row.addStretch(1)

        self.web_badge = QLabel()
        self.web_badge.setObjectName("webBadge")
        self.web_badge.hide()
        role_row.addWidget(self.web_badge)

        self.copy_button = QToolButton()
        self.copy_button.setObjectName("copyButton")
        self.copy_button.setText("Copy")
        self.copy_button.setToolTip("Copy full message")
        self.copy_button.setAccessibleName("Copy full message")
        self.copy_button.setAutoRaise(True)
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_text)
        role_row.addWidget(self.copy_button)

        self.body_label = QLabel()
        self.body_label.setObjectName("messageBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(
            Qt.TextFormat.MarkdownText
            if self._render_markdown
            else Qt.TextFormat.PlainText
        )
        self.body_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.body_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.set_text(text)

        self.thinking_wave = _ThinkingWave()

        self.activity_button = QToolButton()
        self.activity_button.setObjectName("activityButton")
        self.activity_button.setCheckable(True)
        self.activity_button.setAutoRaise(True)
        self.activity_button.setToolTip("Show Dexter's reasoning and tool activity")
        self.activity_button.setAccessibleName("Show Dexter activity")
        self.activity_button.toggled.connect(self._set_activity_expanded)
        self.activity_button.hide()

        self.activity_panel = QFrame()
        self.activity_panel.setObjectName("activityPanel")
        self.activity_layout = QVBoxLayout(self.activity_panel)
        self.activity_layout.setContentsMargins(0, 2, 0, 0)
        self.activity_layout.setSpacing(5)
        self.activity_panel.hide()

        layout.addLayout(role_row)
        layout.addWidget(self.body_label)
        layout.addWidget(self.thinking_wave)
        if activity_enabled:
            layout.addWidget(self.activity_button)
            layout.addWidget(self.activity_panel)

    def set_text(self, text: str) -> None:
        self._raw_text = text
        self.body_label.setText(text)
        self.copy_button.setEnabled(bool(text))

    def append_text(self, text: str) -> None:
        if text:
            self._raw_text += text
            self.body_label.setText(self._raw_text)
            self.copy_button.setEnabled(True)

    @Slot()
    def _copy_text(self) -> None:
        if not self._raw_text:
            return
        QApplication.clipboard().setText(self._raw_text)

    def start_thinking(self) -> None:
        self.thinking_wave.start()

    def stop_thinking(self) -> None:
        self.thinking_wave.stop()

    def add_activity(self, update: ActivityUpdate) -> None:
        if not self._activity_enabled:
            return

        record = self._activity_records.get(update.activity_id)
        if record is None:
            if len(self._activity_records) >= self._MAX_ACTIVITY_RECORDS:
                return
            record = ActivityRecord.from_update(update)
            self._activity_records[update.activity_id] = record
            widget = _ActivityEntryCard(record)
            self._activity_widgets[update.activity_id] = widget
            self.activity_layout.addWidget(widget)
        else:
            record.apply(update)
            self._activity_widgets[update.activity_id].update_record(record)

        self._refresh_activity_summary()

    def _refresh_activity_summary(self) -> None:
        tool_count = sum(
            record.category == "tool"
            for record in self._activity_records.values()
        )
        has_thought = any(
            record.category == "thought"
            for record in self._activity_records.values()
        )
        has_plan = any(
            record.category == "plan"
            for record in self._activity_records.values()
        )

        parts: list[str] = []
        if tool_count:
            parts.append(f"{tool_count} tool" + ("s" if tool_count != 1 else ""))
        if has_thought:
            parts.append("reasoning")
        if has_plan:
            parts.append("plan")

        marker = "▼" if self.activity_button.isChecked() else "▶"
        suffix = f" · {' · '.join(parts)}" if parts else ""
        self.activity_button.setText(f"{marker} Activity{suffix}")
        self.activity_button.setVisible(bool(self._activity_records))

        web_records = [
            record
            for record in self._activity_records.values()
            if record.is_web_search
        ]
        if not web_records:
            self.web_badge.hide()
            return

        statuses = {record.status for record in web_records}
        if "completed" in statuses:
            self.web_badge.setText("Web used")
            self.web_badge.setProperty("state", "completed")
        elif statuses & {"failed", "error", "cancelled", "canceled"}:
            self.web_badge.setText("Web failed")
            self.web_badge.setProperty("state", "failed")
        else:
            self.web_badge.setText("Web search")
            self.web_badge.setProperty("state", "active")
        self.web_badge.style().unpolish(self.web_badge)
        self.web_badge.style().polish(self.web_badge)
        self.web_badge.show()

    def _set_activity_expanded(self, expanded: bool) -> None:
        self.activity_panel.setVisible(expanded)
        self._refresh_activity_summary()

    def finalize_user(self) -> None:
        self.setProperty("draft", False)
        self.role_label.setText("You")
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class _ApprovalCard(QFrame):
    option_selected = Signal(str)

    def __init__(
        self,
        title: str,
        detail: str,
        options: list[ApprovalOption],
    ) -> None:
        super().__init__()
        self.setObjectName("approvalCard")
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )
        self.setMinimumWidth(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 11)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setObjectName("approvalTitle")
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )

        detail_label = QLabel(detail)
        detail_label.setObjectName("approvalDetail")
        detail_label.setWordWrap(True)
        detail_label.setTextFormat(Qt.TextFormat.PlainText)
        detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        detail_label.setMinimumWidth(0)
        detail_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Minimum,
        )

        buttons = QHBoxLayout()
        buttons.setSpacing(7)
        buttons.addStretch(1)

        for option in options:
            button = QPushButton(option.label)
            button.setProperty("emphasized", option.emphasized)
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(
                lambda _checked=False, option_id=option.option_id: (
                    self.option_selected.emit(option_id)
                )
            )
            buttons.addWidget(button)

        layout.addWidget(title_label)
        if detail:
            layout.addWidget(detail_label)
        layout.addLayout(buttons)


class DexterOverlay(QWidget):
    """Frameless, translucent, resizable bottom-left overlay."""

    approval_selected = Signal(str)
    minimize_requested = Signal()
    prompt_submitted = Signal(str)
    cancel_requested = Signal()
    clear_chat_requested = Signal()

    # The entire band is a real painted (near-invisible) mouse surface.
    # 14 px is intentionally easier to acquire than the first demo's 8 px.
    _RESIZE_MARGIN = 14
    _DEFAULT_WIDTH = 520
    _DEFAULT_HEIGHT = 340
    _SCREEN_MARGIN = 18
    _MAX_PERMANENT_MESSAGES = 20

    def __init__(self) -> None:
        super().__init__()

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 180)
        self.resize(self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)

        self._draft_card: _MessageCard | None = None
        self._assistant_card: _MessageCard | None = None
        self._approval_card: _ApprovalCard | None = None
        self._prompt_input_available = True
        self._agent_busy = False
        self._cancel_enabled = True
        self._recording = False
        self._clear_chat_enabled = False
        self._permanent_cards: list[_MessageCard] = []

        self._thinking_resume_timer = QTimer(self)
        self._thinking_resume_timer.setSingleShot(True)
        self._thinking_resume_timer.setInterval(750)
        self._thinking_resume_timer.timeout.connect(
            self._resume_thinking_if_active
        )

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        # This margin is both visual breathing room and the resize hit-zone.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            self._RESIZE_MARGIN,
            self._RESIZE_MARGIN,
            self._RESIZE_MARGIN,
            self._RESIZE_MARGIN,
        )
        outer.setSpacing(0)

        self.panel = QFrame()
        self.panel.setObjectName("panel")
        self.panel.setCursor(Qt.CursorShape.ArrowCursor)
        outer.addWidget(self.panel)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(10, 8, 10, 10)
        panel_layout.setSpacing(8)

        header = _DragHandle()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 1, 2, 1)
        header_layout.setSpacing(8)

        title = QLabel("Dexter")
        title.setObjectName("headerTitle")
        # Let presses on header text reach the drag handle.
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.status_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

        self.clear_button = QToolButton()
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("Clear chat and start a new conversation")
        self.clear_button.setAccessibleName("Clear Dexter chat")
        self.clear_button.setAutoRaise(True)
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.clear_chat_requested.emit)

        self.minimize_button = QToolButton()
        self.minimize_button.setObjectName("minimizeButton")
        self.minimize_button.setText("—")
        self.minimize_button.setToolTip("Hide Dexter to the system tray")
        self.minimize_button.setAccessibleName("Minimize Dexter to tray")
        self.minimize_button.setAutoRaise(True)
        self.minimize_button.setFixedSize(24, 20)
        self.minimize_button.clicked.connect(self.minimize_requested.emit)

        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(self.clear_button)
        header_layout.addWidget(self.minimize_button)
        panel_layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("historyScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.viewport().setStyleSheet("background: transparent;")

        self.history_widget = QWidget()
        self.history_widget.setObjectName("historyWidget")
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(7)
        self.history_layout.addStretch(1)

        self.scroll.setWidget(self.history_widget)
        panel_layout.addWidget(self.scroll, 1)

        self.approval_host = QVBoxLayout()
        self.approval_host.setContentsMargins(0, 0, 0, 0)
        panel_layout.addLayout(self.approval_host)

        # A deliberately small typed-input path. Voice remains the fastest
        # interaction, but the host should never require speech for ordinary
        # prompts.
        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(9, 5, 5, 5)
        composer_layout.setSpacing(6)

        self.prompt_input = QLineEdit()
        self.prompt_input.setObjectName("promptInput")
        self.prompt_input.setPlaceholderText("Type a message…")
        self.prompt_input.setClearButtonEnabled(True)
        self.prompt_input.returnPressed.connect(self._submit_prompt_input)

        self.send_button = QToolButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setText("Send")
        self.send_button.setToolTip("Send message")
        self.send_button.setAccessibleName("Send message")
        self.send_button.setAutoRaise(True)
        self.send_button.setMinimumWidth(44)
        self.send_button.clicked.connect(self._submit_prompt_input)

        self.recording_dot = QLabel("●")
        self.recording_dot.setObjectName("recordingDot")
        self.recording_dot.setAccessibleName("Recording")
        self.recording_dot.hide()

        self.recording_label = QLabel("Recording…")
        self.recording_label.setObjectName("recordingLabel")
        self.recording_label.hide()

        self.cancel_button = QToolButton()
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setText("Cancel")
        self.cancel_button.setToolTip("Cancel current Dexter response")
        self.cancel_button.setAccessibleName("Cancel current Dexter response")
        self.cancel_button.setAutoRaise(True)
        self.cancel_button.setMinimumWidth(52)
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

        composer_layout.addWidget(self.prompt_input, 1)
        composer_layout.addWidget(self.recording_dot)
        composer_layout.addWidget(self.recording_label, 1)
        composer_layout.addWidget(self.send_button)
        composer_layout.addWidget(self.cancel_button)
        panel_layout.addWidget(composer)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                color: #f2f2f3;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QFrame#panel {
                background-color: rgba(22, 22, 26, 224);
                border: 1px solid rgba(255, 255, 255, 38);
                border-radius: 13px;
            }
            QFrame#header {
                background: transparent;
                min-height: 22px;
            }
            QLabel#headerTitle {
                font-size: 12px;
                font-weight: 600;
                color: rgba(255, 255, 255, 205);
            }
            QLabel#statusLabel {
                font-size: 11px;
                color: rgba(255, 255, 255, 145);
            }
            QToolButton#clearButton {
                background: transparent;
                border: none;
                border-radius: 5px;
                color: rgba(255, 255, 255, 135);
                font-size: 10px;
                padding: 2px 5px;
            }
            QToolButton#clearButton:hover {
                background-color: rgba(255, 255, 255, 24);
                color: rgba(255, 255, 255, 225);
            }
            QToolButton#clearButton:disabled {
                color: rgba(255, 255, 255, 55);
            }
            QToolButton#minimizeButton {
                background: transparent;
                border: none;
                border-radius: 5px;
                color: rgba(255, 255, 255, 155);
                font-size: 15px;
                font-weight: 600;
                padding: 0px;
            }
            QToolButton#minimizeButton:hover {
                background-color: rgba(255, 255, 255, 24);
                color: rgba(255, 255, 255, 235);
            }
            QToolButton#minimizeButton:pressed {
                background-color: rgba(255, 255, 255, 38);
            }
            QFrame#composer {
                background-color: rgba(255, 255, 255, 13);
                border: 1px solid rgba(255, 255, 255, 27);
                border-radius: 9px;
            }
            QLineEdit#promptInput {
                background: transparent;
                border: none;
                padding: 4px 3px;
                color: rgba(255, 255, 255, 238);
                selection-background-color: rgba(255, 255, 255, 55);
            }
            QLineEdit#promptInput:disabled {
                color: rgba(255, 255, 255, 105);
            }
            QLabel#recordingDot {
                color: rgb(235, 74, 74);
                font-size: 15px;
                padding-left: 3px;
            }
            QLabel#recordingLabel {
                color: rgba(255, 255, 255, 220);
                font-weight: 600;
                padding-left: 2px;
            }
            QToolButton#sendButton {
                background-color: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 6px;
                padding: 4px 7px;
                color: rgba(255, 255, 255, 190);
            }
            QToolButton#sendButton:hover {
                background-color: rgba(255, 255, 255, 31);
                color: rgba(255, 255, 255, 235);
            }
            QToolButton#sendButton:disabled {
                color: rgba(255, 255, 255, 75);
                border-color: rgba(255, 255, 255, 14);
            }
            QToolButton#cancelButton {
                background-color: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 6px;
                padding: 4px 7px;
                color: rgba(255, 255, 255, 200);
            }
            QToolButton#cancelButton:hover {
                background-color: rgba(255, 255, 255, 34);
                color: rgba(255, 255, 255, 245);
            }
            QToolButton#cancelButton:disabled {
                color: rgba(255, 255, 255, 75);
                border-color: rgba(255, 255, 255, 14);
            }
            QScrollArea#historyScroll, QWidget#historyWidget {
                background: transparent;
            }
            QFrame#messageCard {
                background-color: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 9px;
            }
            QFrame#messageCard[draft="true"] {
                background-color: rgba(255, 255, 255, 10);
                border: 1px dashed rgba(255, 255, 255, 45);
            }
            QLabel#roleLabel {
                font-size: 10px;
                font-weight: 600;
                color: rgba(255, 255, 255, 130);
            }
            QLabel#webBadge {
                background-color: rgba(86, 156, 214, 35);
                border: 1px solid rgba(100, 174, 235, 65);
                border-radius: 5px;
                color: rgba(180, 220, 252, 220);
                font-size: 9px;
                font-weight: 600;
                padding: 1px 5px;
            }
            QLabel#webBadge[state="failed"] {
                background-color: rgba(220, 88, 88, 28);
                border-color: rgba(235, 105, 105, 58);
                color: rgba(255, 190, 190, 215);
            }
            QToolButton#copyButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                color: rgba(255, 255, 255, 105);
                font-size: 9px;
                padding: 1px 4px;
            }
            QToolButton#copyButton:hover {
                background-color: rgba(255, 255, 255, 20);
                color: rgba(255, 255, 255, 220);
            }
            QToolButton#copyButton:disabled {
                color: rgba(255, 255, 255, 35);
            }
            QLabel#messageBody {
                font-size: 13px;
                color: rgba(255, 255, 255, 238);
            }
            QToolButton#activityButton {
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 145);
                font-size: 10px;
                padding: 3px 1px 1px 1px;
                text-align: left;
            }
            QToolButton#activityButton:hover {
                color: rgba(255, 255, 255, 225);
            }
            QFrame#activityPanel {
                background: transparent;
                border: none;
            }
            QFrame#activityEntry {
                background-color: rgba(0, 0, 0, 28);
                border: 1px solid rgba(255, 255, 255, 16);
                border-radius: 7px;
            }
            QLabel#activityEntryTitle {
                color: rgba(255, 255, 255, 185);
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#activityEntryDetail {
                color: rgba(255, 255, 255, 155);
                font-family: "Cascadia Mono", "Consolas";
                font-size: 10px;
            }
            QFrame#approvalCard {
                background-color: rgba(38, 38, 45, 242);
                border: 1px solid rgba(255, 255, 255, 46);
                border-radius: 10px;
            }
            QLabel#approvalTitle {
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#approvalDetail {
                font-size: 12px;
                color: rgba(255, 255, 255, 195);
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 38);
                border-radius: 7px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 31);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 42);
            }
            QPushButton[emphasized="true"] {
                background-color: rgba(255, 255, 255, 42);
                border: 1px solid rgba(255, 255, 255, 72);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 7px;
                margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 55);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override name
        # With a translucent top-level widget, completely unpainted pixels can
        # pass mouse input through to windows below. Paint an almost invisible
        # alpha-1 surface so the outer resize band remains a real hit target.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))
        painter.end()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override name
        super().showEvent(event)
        if not self.property("initialPositionApplied"):
            self._move_to_default_corner()
            self.setProperty("initialPositionApplied", True)

    def _move_to_default_corner(self) -> None:
        app = QApplication.instance()
        screen = None
        if app is not None:
            screen = app.screenAt(QCursor.pos()) or app.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        x = available.left() + self._SCREEN_MARGIN
        y = available.bottom() - self.height() - self._SCREEN_MARGIN + 1
        self.move(QPoint(x, y))

    @Slot(str)
    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    @Slot()
    def _submit_prompt_input(self) -> None:
        text = self.prompt_input.text().strip()
        if not text:
            return
        self.prompt_input.clear()
        self.prompt_submitted.emit(text)

    @Slot(bool)
    def set_prompt_input_enabled(self, enabled: bool) -> None:
        self._prompt_input_available = enabled
        self._refresh_composer_state()

    @Slot(bool)
    def set_clear_chat_enabled(self, enabled: bool) -> None:
        self._clear_chat_enabled = enabled
        self.clear_button.setEnabled(enabled)

    @Slot(bool)
    def set_agent_busy(self, busy: bool) -> None:
        self._agent_busy = busy
        if not busy:
            self._cancel_enabled = True
        self._refresh_composer_state()

    @Slot(bool)
    def set_recording(self, recording: bool) -> None:
        self._recording = recording
        self._refresh_composer_state()

    @Slot(bool)
    def set_cancel_enabled(self, enabled: bool) -> None:
        self._cancel_enabled = enabled
        self.cancel_button.setEnabled(enabled and self._agent_busy)

    def _refresh_composer_state(self) -> None:
        can_prompt = (
            self._prompt_input_available
            and not self._agent_busy
            and not self._recording
        )

        self.prompt_input.setEnabled(can_prompt)
        self.prompt_input.setVisible(not self._recording)

        self.recording_dot.setVisible(self._recording)
        self.recording_label.setVisible(self._recording)

        self.send_button.setEnabled(can_prompt)
        self.send_button.setVisible(
            not self._agent_busy and not self._recording
        )

        self.cancel_button.setVisible(
            self._agent_busy and not self._recording
        )
        self.cancel_button.setEnabled(
            self._agent_busy and self._cancel_enabled
        )

    @Slot()
    def focus_prompt_input(self) -> None:
        if self.prompt_input.isEnabled():
            self.prompt_input.setFocus(Qt.FocusReason.OtherFocusReason)

    @Slot(str)
    def set_stt_partial(self, text: str) -> None:
        text = text.strip()
        if not text:
            if self._draft_card is not None:
                self._remove_card(self._draft_card)
                self._draft_card = None
            return

        if self._draft_card is None:
            self._draft_card = _MessageCard("You · listening", text, draft=True)
            self._insert_card(self._draft_card)
        else:
            self._draft_card.set_text(text)

        self._scroll_to_bottom()

    @Slot(str)
    def finalize_user_prompt(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self._draft_card is None:
            card = _MessageCard("You", text)
            self._insert_card(card)
        else:
            card = self._draft_card
            self._draft_card = None
            card.set_text(text)
            card.finalize_user()

        self._permanent_cards.append(card)
        self._trim_history()
        self._scroll_to_bottom()

    @Slot()
    def begin_assistant_response(self) -> None:
        self.finish_assistant_response()
        self._assistant_card = _MessageCard(
            "Dexter",
            "",
            activity_enabled=True,
            render_markdown=True,
        )
        self._insert_card(self._assistant_card)
        self._assistant_card.start_thinking()
        self._scroll_to_bottom()

    @Slot(str)
    def append_assistant_text(self, text: str) -> None:
        if not text:
            return
        if self._assistant_card is None:
            self.begin_assistant_response()
        assert self._assistant_card is not None

        # Real output takes precedence. If output then goes quiet for a moment
        # while the turn remains active (model pause/tool work), the wave comes
        # back automatically.
        self._assistant_card.stop_thinking()
        self._thinking_resume_timer.stop()
        self._assistant_card.append_text(text)
        self._thinking_resume_timer.start()
        self._scroll_to_bottom()

    @Slot(object)
    def add_assistant_activity(self, update: ActivityUpdate) -> None:
        if self._assistant_card is None:
            self.begin_assistant_response()
        assert self._assistant_card is not None
        self._assistant_card.add_activity(update)
        self._scroll_to_bottom()

    @Slot()
    def resume_thinking_indicator(self) -> None:
        if self._assistant_card is not None and self._agent_busy:
            self._assistant_card.start_thinking()
            self._scroll_to_bottom()

    def _resume_thinking_if_active(self) -> None:
        if (
            self._assistant_card is not None
            and self._agent_busy
            and self._approval_card is None
        ):
            self._assistant_card.start_thinking()
            self._scroll_to_bottom()

    @Slot()
    def finish_assistant_response(self) -> None:
        self._thinking_resume_timer.stop()
        if self._assistant_card is None:
            return
        self._assistant_card.stop_thinking()
        self._permanent_cards.append(self._assistant_card)
        self._assistant_card = None
        self._trim_history()

    def show_approval(
        self,
        title: str,
        detail: str,
        options: list[ApprovalOption],
    ) -> None:
        self.clear_approval()
        self._thinking_resume_timer.stop()
        if self._assistant_card is not None:
            self._assistant_card.stop_thinking()

        if not options:
            return

        card = _ApprovalCard(title, detail, options)
        card.option_selected.connect(self._approval_option_selected)
        self._approval_card = card
        self.approval_host.addWidget(card)

    @Slot()
    def clear_approval(self) -> None:
        if self._approval_card is None:
            return
        card = self._approval_card
        self._approval_card = None
        card.setParent(None)
        card.deleteLater()

    @Slot(str)
    def _approval_option_selected(self, option_id: str) -> None:
        self.approval_selected.emit(option_id)
        self.clear_approval()

    @Slot()
    def clear_history(self) -> None:
        self._thinking_resume_timer.stop()
        self.clear_approval()

        if self._draft_card is not None:
            self._remove_card(self._draft_card)
            self._draft_card = None

        if self._assistant_card is not None:
            self._assistant_card.stop_thinking()
            self._remove_card(self._assistant_card)
            self._assistant_card = None

        for card in list(self._permanent_cards):
            self._remove_card(card)
        self._permanent_cards.clear()

        self._scroll_to_bottom()

    def _insert_card(self, card: _MessageCard) -> None:
        # Keep the one expanding stretch as the final layout item so short
        # conversations remain top-aligned. Appending cards after that stretch
        # creates expandable blank space above the transcript.
        stretch_index = max(0, self.history_layout.count() - 1)
        self.history_layout.insertWidget(stretch_index, card)

    def _remove_card(self, card: _MessageCard) -> None:
        self.history_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()

    def _trim_history(self) -> None:
        while len(self._permanent_cards) > self._MAX_PERMANENT_MESSAGES:
            oldest = self._permanent_cards.pop(0)
            self._remove_card(oldest)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def _resize_edges_at(self, pos: QPoint):
        margin = self._RESIZE_MARGIN
        rect = self.rect()
        edges = []

        if pos.x() <= margin:
            edges.append(Qt.Edge.LeftEdge)
        elif pos.x() >= rect.width() - margin:
            edges.append(Qt.Edge.RightEdge)

        if pos.y() <= margin:
            edges.append(Qt.Edge.TopEdge)
        elif pos.y() >= rect.height() - margin:
            edges.append(Qt.Edge.BottomEdge)

        if not edges:
            return None

        combined = edges[0]
        for edge in edges[1:]:
            combined |= edge
        return combined

    def _cursor_for_edges(self, edges):
        if not edges:
            return Qt.CursorShape.ArrowCursor

        left = bool(edges & Qt.Edge.LeftEdge)
        right = bool(edges & Qt.Edge.RightEdge)
        top = bool(edges & Qt.Edge.TopEdge)
        bottom = bool(edges & Qt.Edge.BottomEdge)

        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (right and top) or (left and bottom):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        edges = self._resize_edges_at(event.position().toPoint())
        self.setCursor(self._cursor_for_edges(edges))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._resize_edges_at(event.position().toPoint())
            if edges:
                window = self.windowHandle()
                if window is not None and window.startSystemResize(edges):
                    event.accept()
                    return
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override name
        self.unsetCursor()
        super().leaveEvent(event)

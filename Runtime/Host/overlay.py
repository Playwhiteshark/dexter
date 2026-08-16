"""Minimal always-on-top Dexter overlay for Stage 2.

Presentation-only: no Moonshine, Goose, ACP, or Dexter tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


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


class _MessageCard(QFrame):
    def __init__(self, role: str, text: str = "", *, draft: bool = False) -> None:
        super().__init__()
        self.setObjectName("messageCard")
        self.setProperty("draft", draft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(3)

        self.role_label = QLabel(role)
        self.role_label.setObjectName("roleLabel")

        self.body_label = QLabel(text)
        self.body_label.setObjectName("messageBody")
        self.body_label.setWordWrap(True)
        self.body_label.setTextFormat(Qt.TextFormat.PlainText)
        self.body_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.body_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout.addWidget(self.role_label)
        layout.addWidget(self.body_label)

    def set_text(self, text: str) -> None:
        self.body_label.setText(text)

    def append_text(self, text: str) -> None:
        if text:
            self.body_label.setText(self.body_label.text() + text)

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
                QSizePolicy.Policy.Ignored,
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
        self._permanent_cards: list[_MessageCard] = []

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
            QLabel#messageBody {
                font-size: 13px;
                color: rgba(255, 255, 255, 238);
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
        self._assistant_card = _MessageCard("Dexter", "")
        self._insert_card(self._assistant_card)
        self._scroll_to_bottom()

    @Slot(str)
    def append_assistant_text(self, text: str) -> None:
        if not text:
            return
        if self._assistant_card is None:
            self.begin_assistant_response()
        assert self._assistant_card is not None
        self._assistant_card.append_text(text)
        self._scroll_to_bottom()

    @Slot()
    def finish_assistant_response(self) -> None:
        if self._assistant_card is None:
            return
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

    def _insert_card(self, card: _MessageCard) -> None:
        self.history_layout.addWidget(card)

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

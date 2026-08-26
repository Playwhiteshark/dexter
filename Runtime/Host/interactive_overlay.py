"""Adaptive interaction layer for Dexter's Host overlay.

This keeps the presentation primitives in ``overlay.py`` small while owning the
higher-level interaction behavior that depends on runtime state:

* short transcripts hug their content instead of reserving an empty viewport;
* the window grows upward until the user's current height cap, then scrolls;
* typed input remains available while Dexter is responding; and
* the send action becomes an explicit interrupting send while a turn is active.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import QSizePolicy

from Runtime.Host.overlay import ApprovalOption
from Runtime.Host.overlay import DexterOverlay as _BaseDexterOverlay


class DexterOverlay(_BaseDexterOverlay):
    """Dexter overlay with adaptive sizing and interruption-friendly composer."""

    def __init__(self) -> None:
        self._fitting_height = False
        self._fit_scheduled = False
        self._auto_height_cap = self._DEFAULT_HEIGHT
        super().__init__()

        # QScrollArea has a useful size hint for ordinary windows, but here the
        # transcript must be able to collapse all the way to its actual content.
        # Once the overlay reaches its cap the same widget expands and scrolls.
        self.scroll.setMinimumHeight(0)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        self.history_widget.installEventFilter(self)
        self._schedule_content_fit()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if (
            watched is self.history_widget
            and event.type() == QEvent.Type.LayoutRequest
        ):
            self._schedule_content_fit()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._schedule_content_fit()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if (
            not self._fitting_height
            and event.oldSize().isValid()
            and event.size().height() != event.oldSize().height()
        ):
            # A user-driven vertical resize establishes the new maximum height.
            # Short conversations still hug their contents below this cap.
            self._auto_height_cap = max(
                self.minimumHeight(),
                event.size().height(),
            )
            self._schedule_content_fit()

    def _refresh_composer_state(self) -> None:
        # Typing is deliberately independent from the active agent turn. The
        # Host decides whether a submitted prompt is a normal send or a barge-in.
        can_type = self._prompt_input_available and not self._recording

        self.prompt_input.setEnabled(can_type)
        self.prompt_input.setVisible(not self._recording)

        self.recording_dot.setVisible(self._recording)
        self.recording_label.setVisible(self._recording)

        self.send_button.setEnabled(can_type)
        self.send_button.setVisible(not self._recording)
        if self._agent_busy:
            self.send_button.setText("Send now")
            self.send_button.setToolTip(
                "Cancel Dexter's current response and send this message"
            )
            self.send_button.setAccessibleName(
                "Cancel current response and send message"
            )
        else:
            self.send_button.setText("Send")
            self.send_button.setToolTip("Send message")
            self.send_button.setAccessibleName("Send message")

        self.cancel_button.setVisible(
            self._agent_busy and not self._recording
        )
        self.cancel_button.setEnabled(
            self._agent_busy and self._cancel_enabled
        )
        self._schedule_content_fit()

    def finish_interrupted_response(self) -> None:
        """Freeze the current response exactly where the user interrupted it."""
        self._thinking_resume_timer.stop()
        card = self._assistant_card
        if card is None:
            return

        card.stop_thinking()
        self._assistant_card = None

        if not card._raw_text and not card._activity_records:
            # Do not preserve an empty thinking card when the user barges in
            # before Dexter produces any meaningful visible output.
            self._remove_card(card)
            return

        card.role_label.setText("Dexter · interrupted")
        self._permanent_cards.append(card)
        self._trim_history()
        self._schedule_content_fit()

    def _schedule_content_fit(self) -> None:
        if self._fit_scheduled:
            return
        self._fit_scheduled = True
        QTimer.singleShot(0, self._fit_height_to_content)

    def _history_content_height(self) -> int:
        self.history_layout.invalidate()
        self.history_layout.activate()
        return max(0, self.history_layout.sizeHint().height())

    def _fit_height_to_content(self) -> None:
        self._fit_scheduled = False
        if not hasattr(self, "scroll"):
            return

        # Everything outside the scroll area is fixed-height UI chrome. The
        # transcript itself receives only the height its contents need, until
        # the current user/default height cap is reached.
        chrome_height = max(0, self.height() - self.scroll.height())
        desired = chrome_height + self._history_content_height()
        target = max(
            self.minimumHeight(),
            min(self._auto_height_cap, desired),
        )

        if abs(target - self.height()) <= 1:
            return

        bottom = self.geometry().bottom()
        self._fitting_height = True
        try:
            self.resize(self.width(), target)
            # Dexter lives at the bottom of the screen by default, so grow and
            # shrink upward rather than walking the composer down the desktop.
            self.move(self.x(), bottom - target + 1)
        finally:
            self._fitting_height = False

    # These explicit hooks make content-fit deterministic even when Qt chooses
    # not to emit a LayoutRequest for a particular child-label update.
    def set_stt_partial(self, text: str) -> None:
        super().set_stt_partial(text)
        self._schedule_content_fit()

    def finalize_user_prompt(self, text: str) -> None:
        super().finalize_user_prompt(text)
        self._schedule_content_fit()

    def begin_assistant_response(self) -> None:
        super().begin_assistant_response()
        self._schedule_content_fit()

    def append_assistant_text(self, text: str) -> None:
        super().append_assistant_text(text)
        self._schedule_content_fit()

    def add_assistant_activity(self, update) -> None:
        super().add_assistant_activity(update)
        self._schedule_content_fit()

    def finish_assistant_response(self) -> None:
        super().finish_assistant_response()
        self._schedule_content_fit()

    def clear_history(self) -> None:
        super().clear_history()
        self._schedule_content_fit()

    def show_approval(self, title: str, detail: str, options) -> None:
        super().show_approval(title, detail, options)
        self._schedule_content_fit()

    def clear_approval(self) -> None:
        super().clear_approval()
        self._schedule_content_fit()

    def set_recording(self, recording: bool) -> None:
        super().set_recording(recording)
        self._schedule_content_fit()


__all__ = ["ApprovalOption", "DexterOverlay"]

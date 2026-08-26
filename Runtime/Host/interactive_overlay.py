"""Adaptive interaction layer for Dexter's Host overlay.

This keeps the presentation primitives in ``overlay.py`` small while owning the
higher-level interaction behavior that depends on runtime state:

* short transcripts start compact without reserving a dead viewport;
* manual window resizing is always respected once the user changes the height;
* typed input remains available while Dexter is responding; and
* one right-hand action slot swaps between Send, Cancel, and Send now.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QTimer
from PySide6.QtWidgets import QLayout, QSizePolicy

from Runtime.Host.overlay import ApprovalOption
from Runtime.Host.overlay import DexterOverlay as _BaseDexterOverlay


class DexterOverlay(_BaseDexterOverlay):
    """Dexter overlay with compact startup sizing and interruption-friendly input."""

    def __init__(self) -> None:
        self._fitting_height = False
        self._fit_scheduled = False
        self._auto_fit_enabled = True
        super().__init__()

        # Keep the minimum just large enough for the header/composer chrome.
        # The previous 180 px minimum could itself create a visible blank band
        # below a short transcript even when the content-fit math was correct.
        self.setMinimumHeight(120)

        # QSizePolicy::Ignored tells Qt to disregard the scroll area's size hint
        # and give it as much space as possible. That is exactly the opposite of
        # what a compact transcript needs. Preferred still permits normal manual
        # expansion while respecting the content-derived size hint.
        self.scroll.setMinimumHeight(0)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        # Qt recommends constraining layouts whose scroll-area contents change
        # dynamically so the surrounding geometry is invalidated promptly.
        self.history_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinAndMaxSize
        )

        # The base panel used a global 8 px spacing between every layout item.
        # That left a permanent visible strip between the final transcript card
        # and the composer. Keep the useful header separation explicitly, but no
        # spacing after the transcript itself.
        panel_layout = self.panel.layout()
        if panel_layout is not None:
            panel_layout.setSpacing(0)
            if hasattr(panel_layout, "insertSpacing"):
                panel_layout.insertSpacing(1, 8)

        # Send / Cancel / Send now share the same physical action slot. Matching
        # minimum widths prevents the composer from shifting as the mode swaps.
        self.send_button.setMinimumWidth(72)
        self.cancel_button.setMinimumWidth(72)
        self.prompt_input.textChanged.connect(self._refresh_composer_state)

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
            # Any real user resize becomes authoritative for the rest of this
            # Host session. Content changes may scroll, but they must never pull
            # the window back to the transcript's preferred height.
            self._auto_fit_enabled = False

    def _refresh_composer_state(self) -> None:
        # Typing is deliberately independent from the active agent turn. The
        # Host decides whether a submitted prompt is a normal send or a barge-in.
        can_type = self._prompt_input_available and not self._recording
        has_text = bool(self.prompt_input.text().strip())

        self.prompt_input.setEnabled(can_type)
        self.prompt_input.setVisible(not self._recording)

        self.recording_dot.setVisible(self._recording)
        self.recording_label.setVisible(self._recording)

        if self._recording:
            self.send_button.hide()
            self.cancel_button.hide()
            self._schedule_content_fit()
            return

        if self._agent_busy:
            # Empty composer: the useful action is Cancel. As soon as the user
            # drafts a replacement, that exact slot becomes Send now.
            show_send_now = can_type and has_text
            self.send_button.setVisible(show_send_now)
            self.send_button.setEnabled(show_send_now)
            self.send_button.setText("Send now")
            self.send_button.setToolTip(
                "Cancel Dexter's current response and send this message"
            )
            self.send_button.setAccessibleName(
                "Cancel current response and send message"
            )

            self.cancel_button.setVisible(not show_send_now)
            self.cancel_button.setEnabled(self._cancel_enabled)
        else:
            self.send_button.show()
            self.send_button.setEnabled(can_type)
            self.send_button.setText("Send")
            self.send_button.setToolTip("Send message")
            self.send_button.setAccessibleName("Send message")

            self.cancel_button.hide()

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
        if not self._auto_fit_enabled or self._fit_scheduled:
            return
        self._fit_scheduled = True
        QTimer.singleShot(0, self._fit_height_to_content)

    def _history_content_height(self) -> int:
        self.history_layout.invalidate()
        self.history_layout.activate()
        return max(0, self.history_layout.sizeHint().height())

    def _fit_height_to_content(self) -> None:
        self._fit_scheduled = False
        if not self._auto_fit_enabled or not hasattr(self, "scroll"):
            return

        # Auto-fit is only the compact initial/default behavior. It grows up to
        # Dexter's historical default height; any manual vertical resize disables
        # this routine and becomes authoritative instead.
        chrome_height = max(0, self.height() - self.scroll.height())
        desired = chrome_height + self._history_content_height()
        target = max(
            self.minimumHeight(),
            min(self._DEFAULT_HEIGHT, desired),
        )

        if abs(target - self.height()) <= 1:
            return

        bottom = self.geometry().bottom()
        self._fitting_height = True
        try:
            self.resize(self.width(), target)
            # Dexter lives at the bottom of the screen by default, so automatic
            # compacting/growth moves the top edge rather than the composer.
            self.move(self.x(), bottom - target + 1)
        finally:
            self._fitting_height = False

    # These explicit hooks make compact fitting deterministic even when Qt
    # chooses not to emit a LayoutRequest for a particular child-label update.
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

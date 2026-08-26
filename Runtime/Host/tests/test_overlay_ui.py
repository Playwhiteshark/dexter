from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSizePolicy

from Runtime.Host.interactive_overlay import DexterOverlay
from Runtime.Host.voice_output_control import VoiceOutputControl


class OverlayUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.overlay = DexterOverlay()
        self.overlay.show()
        self._app.processEvents()

    def tearDown(self) -> None:
        self.overlay.close()
        self.overlay.deleteLater()
        self._app.processEvents()

    def test_history_cards_append_without_expanding_spacer(self) -> None:
        self.overlay.finalize_user_prompt("first")
        self.overlay.finalize_user_prompt("second")

        layout = self.overlay.history_layout
        self.assertEqual(layout.count(), 2)
        self.assertTrue(layout.alignment() & Qt.AlignmentFlag.AlignTop)
        self.assertIs(
            layout.itemAt(0).widget(),
            self.overlay._permanent_cards[0],
        )
        self.assertIs(
            layout.itemAt(1).widget(),
            self.overlay._permanent_cards[1],
        )

    def test_transcript_uses_content_preferred_sizing_without_bottom_spacing(self) -> None:
        policy = self.overlay.scroll.sizePolicy()
        self.assertEqual(
            policy.verticalPolicy(),
            QSizePolicy.Policy.Preferred,
        )
        self.assertEqual(self.overlay.panel.layout().spacing(), 0)
        self.assertEqual(self.overlay.minimumHeight(), 120)

    def test_manual_vertical_resize_disables_auto_fit_for_session(self) -> None:
        self.overlay.finalize_user_prompt("short message")
        self.overlay._fit_height_to_content()
        compact_height = self.overlay.height()
        self.assertLessEqual(compact_height, self.overlay._DEFAULT_HEIGHT)

        manual_height = compact_height + 90
        self.overlay.resize(self.overlay.width(), manual_height)
        self._app.processEvents()

        self.assertFalse(self.overlay._auto_fit_enabled)
        self.assertEqual(self.overlay.height(), manual_height)

        # Later transcript changes must not snap the user's chosen height back.
        self.overlay.finalize_user_prompt("another short message")
        self._app.processEvents()
        self.overlay._fit_height_to_content()
        self.assertEqual(self.overlay.height(), manual_height)

    def test_busy_composer_swaps_cancel_and_send_now_in_one_slot(self) -> None:
        self.overlay.set_prompt_input_enabled(True)
        self.overlay.set_agent_busy(True)

        self.assertTrue(self.overlay.prompt_input.isEnabled())
        self.assertFalse(self.overlay.send_button.isVisible())
        self.assertTrue(self.overlay.cancel_button.isVisible())

        self.overlay.prompt_input.setText("replacement")
        self._app.processEvents()
        self.assertTrue(self.overlay.send_button.isVisible())
        self.assertEqual(self.overlay.send_button.text(), "Send now")
        self.assertFalse(self.overlay.cancel_button.isVisible())
        self.assertEqual(
            self.overlay.send_button.minimumWidth(),
            self.overlay.cancel_button.minimumWidth(),
        )

        self.overlay.prompt_input.clear()
        self._app.processEvents()
        self.assertFalse(self.overlay.send_button.isVisible())
        self.assertTrue(self.overlay.cancel_button.isVisible())

        self.overlay.set_agent_busy(False)
        self.assertTrue(self.overlay.send_button.isVisible())
        self.assertEqual(self.overlay.send_button.text(), "Send")
        self.assertFalse(self.overlay.cancel_button.isVisible())

    def test_interrupted_response_is_frozen_before_next_user_turn(self) -> None:
        self.overlay.begin_assistant_response()
        self.overlay.append_assistant_text("partial answer")
        self.overlay.finish_interrupted_response()
        self.overlay.finalize_user_prompt("replacement")

        self.assertIsNone(self.overlay._assistant_card)
        self.assertEqual(len(self.overlay._permanent_cards), 2)
        self.assertEqual(
            self.overlay._permanent_cards[0].role_label.text(),
            "Dexter · interrupted",
        )
        self.assertEqual(
            self.overlay._permanent_cards[1].role_label.text(),
            "You",
        )

    def test_empty_interrupted_response_is_not_preserved(self) -> None:
        self.overlay.begin_assistant_response()
        self.overlay.finish_interrupted_response()

        self.assertIsNone(self.overlay._assistant_card)
        self.assertEqual(self.overlay.history_layout.count(), 0)
        self.assertEqual(self.overlay._permanent_cards, [])

    def test_assistant_markdown_preserves_raw_text_for_copy(self) -> None:
        self.overlay.begin_assistant_response()
        card = self.overlay._assistant_card
        self.assertIsNotNone(card)
        assert card is not None

        self.assertEqual(card.body_label.textFormat(), Qt.TextFormat.MarkdownText)

        card.append_text("**bold**")
        card.append_text(" and `code`")
        expected = "**bold** and `code`"
        self.assertEqual(card._raw_text, expected)

        card.copy_button.click()
        self.assertEqual(QApplication.clipboard().text(), expected)

    def test_user_prompt_remains_plain_text_and_copyable(self) -> None:
        expected = "**literal user input**"
        self.overlay.finalize_user_prompt(expected)
        card = self.overlay._permanent_cards[-1]

        self.assertEqual(card.body_label.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(card._raw_text, expected)

        card.copy_button.click()
        self.assertEqual(QApplication.clipboard().text(), expected)

    def test_voice_status_tracks_resource_lifecycle(self) -> None:
        requested: list[bool] = []
        control = VoiceOutputControl(
            self.overlay,
            enabled=True,
            on_toggle=requested.append,
        )

        self.assertEqual(control.voice_status.text(), "Voice loading…")

        control.set_state("ready")
        self.assertEqual(control.voice_status.text(), "Voice ready")

        control.set_state("stopping")
        self.assertEqual(control.voice_status.text(), "Voice unloading…")

        control.set_state("off")
        self.assertEqual(control.voice_status.text(), "Voice off")

        control.set_state("error")
        self.assertEqual(control.voice_status.text(), "Voice error")

        self.assertEqual(requested, [])


if __name__ == "__main__":
    unittest.main()

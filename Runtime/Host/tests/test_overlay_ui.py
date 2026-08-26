from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

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

    def test_short_history_hugs_content_below_height_cap(self) -> None:
        self.overlay.resize(
            self.overlay._DEFAULT_WIDTH,
            self.overlay._DEFAULT_HEIGHT,
        )
        self.overlay.finalize_user_prompt("short message")
        self.overlay._fit_height_to_content()

        self.assertLess(
            self.overlay.height(),
            self.overlay._DEFAULT_HEIGHT,
        )
        self.assertGreaterEqual(
            self.overlay.height(),
            self.overlay.minimumHeight(),
        )

    def test_busy_composer_keeps_typing_and_interrupt_send_available(self) -> None:
        self.overlay.set_prompt_input_enabled(True)
        self.overlay.set_agent_busy(True)

        self.assertTrue(self.overlay.prompt_input.isEnabled())
        self.assertTrue(self.overlay.send_button.isVisible())
        self.assertEqual(self.overlay.send_button.text(), "Send now")
        self.assertTrue(self.overlay.cancel_button.isVisible())

        self.overlay.set_agent_busy(False)
        self.assertEqual(self.overlay.send_button.text(), "Send")

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

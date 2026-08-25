from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from Runtime.Host.overlay import DexterOverlay


class OverlayUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.overlay = DexterOverlay()

    def tearDown(self) -> None:
        self.overlay.close()
        self.overlay.deleteLater()
        self._app.processEvents()

    def test_history_cards_are_inserted_before_terminal_stretch(self) -> None:
        self.overlay.finalize_user_prompt("first")
        self.overlay.finalize_user_prompt("second")

        layout = self.overlay.history_layout
        self.assertEqual(layout.count(), 3)
        self.assertIs(
            layout.itemAt(0).widget(),
            self.overlay._permanent_cards[0],
        )
        self.assertIs(
            layout.itemAt(1).widget(),
            self.overlay._permanent_cards[1],
        )
        self.assertIsNotNone(layout.itemAt(2).spacerItem())

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


if __name__ == "__main__":
    unittest.main()

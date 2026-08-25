from __future__ import annotations

import threading
import unittest

from Runtime.Voice.text_normalization import (
    NormalizingSpeechBackend,
    SpeechTextNormalizer,
)


class SpeechTextNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = SpeechTextNormalizer()

    def test_ai_is_spaced_and_json_uses_preferred_pronunciation(self) -> None:
        self.assertEqual(
            self.normalizer.normalize("AI and JSON are ready."),
            "A I and jay son are ready.",
        )

    def test_generic_acronyms_remain_utterwise_responsibility(self) -> None:
        spoken = self.normalizer.normalize("SQL, CPU, and GPU are ready.")
        self.assertEqual(spoken, "S Q L, C P U, and G P U are ready.")

    def test_underscore_and_word_hyphen_cleanup_is_conservative(self) -> None:
        spoken = self.normalizer.normalize(
            "_important_ snake_case is real-time; keep x_1 and -3°C meaningful."
        )
        self.assertIn("important snake case is real time;", spoken)
        self.assertIn("x sub one", spoken)
        self.assertIn("minus three degrees Celsius", spoken)
        self.assertNotIn("_important_", spoken)
        self.assertNotIn("snake_case", spoken)
        self.assertNotIn("real-time", spoken)

    def test_basic_punctuation_is_preserved_for_neutts_prosody(self) -> None:
        spoken = self.normalizer.normalize("Wait; AI is ready, right? Yes: it is!")
        self.assertEqual(spoken, "Wait; A I is ready, right? Yes: it is!")

    def test_semantic_numbers_dates_and_percentages_still_normalize(self) -> None:
        spoken = self.normalizer.normalize(
            "On 2026-04-03, GPU usage reached 81%."
        )
        self.assertIn("third of April twenty twenty six", spoken)
        self.assertIn("G P U usage reached eighty one percent", spoken)
        self.assertTrue(spoken.endswith("."))


class _FakeBackend:
    ready_info = {"speaker": "paul"}

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.requests: list[str] = []

    def start(self) -> None:
        self.started = True

    def synthesize(self, text: str, cancel_event: threading.Event):
        self.requests.append(text)
        return ()

    def stop(self) -> None:
        self.stopped = True


class NormalizingSpeechBackendTests(unittest.TestCase):
    def test_wrapper_is_transparent_except_for_text(self) -> None:
        inner = _FakeBackend()
        backend = NormalizingSpeechBackend(inner)
        self.assertEqual(backend.ready_info, {"speaker": "paul"})

        backend.start()
        list(backend.synthesize("AI uses JSON.", threading.Event()))
        backend.stop()

        self.assertTrue(inner.started)
        self.assertTrue(inner.stopped)
        self.assertEqual(inner.requests, ["A I uses jay son."])


if __name__ == "__main__":
    unittest.main()

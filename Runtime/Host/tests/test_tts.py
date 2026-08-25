from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from Runtime.Voice.tts import (
    AudioChunk,
    SpeechFormatter,
    SpeechTextBuffer,
    VoiceOutput,
)


class _FakeBackend:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.requests: list[str] = []
        self.generated = threading.Event()

    def start(self) -> None:
        self.started = True

    def synthesize(self, text: str, cancel_event: threading.Event):
        self.requests.append(text)
        self.generated.set()
        if not cancel_event.is_set():
            yield AudioChunk(np.zeros(80, dtype=np.float32), 16_000)

    def stop(self) -> None:
        self.stopped = True


class _FakeSink:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.interrupts = 0
        self.played = 0
        self.played_event = threading.Event()

    def start(self) -> None:
        self.started = True

    def play(self, chunk: AudioChunk, cancel_event: threading.Event) -> None:
        if not cancel_event.is_set():
            self.played += len(chunk.samples)
            self.played_event.set()

    def interrupt(self) -> None:
        self.interrupts += 1

    def stop(self) -> None:
        self.stopped = True


class SpeechTextBufferTests(unittest.TestCase):
    def test_streamed_fragments_emit_complete_sentence(self) -> None:
        buffer = SpeechTextBuffer()
        self.assertEqual(buffer.feed("Right, I've found"), [])
        self.assertEqual(
            buffer.feed(" the problem. The next"),
            ["Right, I've found the problem."],
        )
        self.assertEqual(buffer.flush(), ["The next"])

    def test_clause_punctuation_does_not_split_an_utterance(self) -> None:
        buffer = SpeechTextBuffer()
        self.assertEqual(
            buffer.feed("Dexter should keep this clause together, while the sentence"),
            [],
        )
        self.assertEqual(
            buffer.feed(" is still arriving. The next"),
            ["Dexter should keep this clause together, while the sentence is still arriving."],
        )
        self.assertEqual(buffer.flush(), ["The next"])

    def test_long_unpunctuated_text_is_bounded(self) -> None:
        buffer = SpeechTextBuffer(max_chars=80)
        text = ("word " * 30).strip()
        ready = buffer.feed(text)
        self.assertTrue(ready)
        self.assertLessEqual(len(ready[0]), 80)


class SpeechFormatterTests(unittest.TestCase):
    def test_markdown_urls_and_code_are_cleaned_without_rewriting(self) -> None:
        formatter = SpeechFormatter()
        spoken = formatter.format(
            "### Result\nSee [the report](https://example.com/report) and "
            "`app.py`. Visit https://example.com/x."
        )
        self.assertEqual(spoken, "Result See the report and app.py. Visit link")

    def test_fenced_code_is_omitted_across_chunks(self) -> None:
        formatter = SpeechFormatter()
        self.assertEqual(
            formatter.format("Before. ```python\nprint('x')"),
            "Before.",
        )
        self.assertEqual(formatter.format("\nprint('y')"), "")
        self.assertEqual(formatter.format("``` After."), "After.")


class VoiceOutputTests(unittest.TestCase):
    def test_sentence_can_synthesize_before_response_finishes(self) -> None:
        backend = _FakeBackend()
        sink = _FakeSink()
        output = VoiceOutput(backend, sink=sink)
        output.start()
        try:
            output.begin_response()
            output.add_text("First sentence. Still generating")
            self.assertTrue(backend.generated.wait(1.0))
            self.assertEqual(backend.requests, ["First sentence."])
            output.finish_response()
            deadline = time.monotonic() + 1.0
            while len(backend.requests) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(
                backend.requests,
                ["First sentence.", "Still generating"],
            )
        finally:
            output.stop()

    def test_interrupt_invalidates_current_response(self) -> None:
        backend = _FakeBackend()
        sink = _FakeSink()
        output = VoiceOutput(backend, sink=sink)
        output.start()
        try:
            output.begin_response()
            output.add_text("Keep this partial")
            output.interrupt()
            output.add_text("This should be ignored.")
            output.finish_response()
            time.sleep(0.05)
            self.assertEqual(backend.requests, [])
            self.assertGreaterEqual(sink.interrupts, 1)
        finally:
            output.stop()


if __name__ == "__main__":
    unittest.main()

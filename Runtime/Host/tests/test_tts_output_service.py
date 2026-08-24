from __future__ import annotations

import threading
import time
import unittest

from Runtime.Voice.output_service import SpeechOutputService
from Runtime.Voice.tts import VoiceOutputState


class _FakeBackend:
    ready_info = {"speaker": "paul"}


class _FakeOutput:
    def __init__(self, *, fail_start: bool = False, block_start: bool = False) -> None:
        self.backend = _FakeBackend()
        self.fail_start = fail_start
        self.block_start = block_start
        self.start_gate = threading.Event()
        self.state = VoiceOutputState.STOPPED
        self.calls: list[object] = []

    def start(self) -> None:
        self.state = VoiceOutputState.STARTING
        self.calls.append("start")
        if self.block_start:
            self.start_gate.wait(2.0)
        if self.fail_start:
            self.state = VoiceOutputState.ERROR
            raise RuntimeError("boom")
        self.state = VoiceOutputState.IDLE

    def begin_response(self) -> int:
        self.calls.append("begin")
        return 1

    def add_text(self, text: str) -> None:
        self.calls.append(("text", text))

    def finish_response(self) -> None:
        self.calls.append("finish")

    def interrupt(self) -> None:
        self.calls.append("interrupt")

    def stop(self) -> None:
        self.calls.append("stop")
        self.start_gate.set()
        self.state = VoiceOutputState.STOPPED


class SpeechOutputServiceTests(unittest.TestCase):
    def _wait_ready(self, service: SpeechOutputService) -> None:
        deadline = time.monotonic() + 1.0
        while not service.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.ready)

    def test_response_is_ignored_until_backend_is_ready(self) -> None:
        output = _FakeOutput(block_start=True)
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async()
        self.assertFalse(service.begin_response())
        service.add_text("ignored")
        self.assertNotIn(("text", "ignored"), output.calls)
        output.start_gate.set()
        self._wait_ready(service)
        service.stop()

    def test_ready_response_streams_and_finishes(self) -> None:
        output = _FakeOutput()
        ready = threading.Event()
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async(on_ready=lambda _info: ready.set())
        self.assertTrue(ready.wait(1.0))
        self.assertTrue(service.begin_response())
        service.add_text("Hello.")
        service.finish_response()
        self.assertIn("begin", output.calls)
        self.assertIn(("text", "Hello."), output.calls)
        self.assertIn("finish", output.calls)
        service.stop()

    def test_interrupt_ends_current_response(self) -> None:
        output = _FakeOutput()
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async()
        self._wait_ready(service)
        self.assertTrue(service.begin_response())
        service.interrupt()
        service.add_text("must not be spoken")
        self.assertIn("interrupt", output.calls)
        self.assertNotIn(("text", "must not be spoken"), output.calls)
        service.stop()

    def test_startup_failure_degrades_to_not_ready(self) -> None:
        output = _FakeOutput(fail_start=True)
        failed = threading.Event()
        errors: list[str] = []
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async(
            on_error=lambda message: (errors.append(message), failed.set())
        )
        self.assertTrue(failed.wait(1.0))
        self.assertFalse(service.ready)
        self.assertEqual(service.startup_error, "boom")
        self.assertEqual(errors, ["boom"])

    def test_stop_can_interrupt_startup(self) -> None:
        output = _FakeOutput(block_start=True)
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async()
        deadline = time.monotonic() + 1.0
        while output.state is not VoiceOutputState.STARTING and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIs(output.state, VoiceOutputState.STARTING)
        service.stop()
        self.assertFalse(service.ready)
        self.assertIn("stop", output.calls)


if __name__ == "__main__":
    unittest.main()

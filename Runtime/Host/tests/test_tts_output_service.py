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


class _OutputFactory:
    def __init__(self, *, first_blocks: bool = False) -> None:
        self.first_blocks = first_blocks
        self.outputs: list[_FakeOutput] = []

    def __call__(self):
        output = _FakeOutput(
            block_start=self.first_blocks and not self.outputs,
        )
        self.outputs.append(output)
        return output


class SpeechOutputServiceTests(unittest.TestCase):
    def _wait_ready(self, service: SpeechOutputService) -> None:
        deadline = time.monotonic() + 1.0
        while not service.ready and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.ready)

    def _wait_state(self, service: SpeechOutputService, expected: str) -> None:
        deadline = time.monotonic() + 1.0
        while service.state != expected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(service.state, expected)

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

    def test_startup_failure_degrades_to_off(self) -> None:
        output = _FakeOutput(fail_start=True)
        failed = threading.Event()
        errors: list[str] = []
        service = SpeechOutputService(output=output)  # type: ignore[arg-type]
        service.start_async(
            on_error=lambda message: (errors.append(message), failed.set())
        )
        self.assertTrue(failed.wait(1.0))
        self.assertFalse(service.ready)
        self.assertEqual(service.state, "off")
        self.assertEqual(service.startup_error, "boom")
        self.assertEqual(errors, ["boom"])
        self.assertIn("stop", output.calls)

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
        self.assertEqual(service.state, "shutdown")
        self.assertIn("stop", output.calls)

    def test_disable_releases_output_and_blocks_speech(self) -> None:
        factory = _OutputFactory()
        service = SpeechOutputService(output_factory=factory)  # type: ignore[arg-type]
        service.start_async()
        self._wait_ready(service)
        first = factory.outputs[0]

        disabled = threading.Event()
        self.assertTrue(service.disable_async(on_disabled=disabled.set))
        self.assertTrue(disabled.wait(1.0))
        self.assertEqual(service.state, "off")
        self.assertFalse(service.ready)
        self.assertIn("stop", first.calls)

        self.assertFalse(service.begin_response())
        service.add_text("must stay text-only")
        self.assertNotIn(("text", "must stay text-only"), first.calls)
        service.stop()

    def test_disable_then_reenable_uses_fresh_output(self) -> None:
        factory = _OutputFactory()
        service = SpeechOutputService(output_factory=factory)  # type: ignore[arg-type]
        service.start_async()
        self._wait_ready(service)
        first = factory.outputs[0]

        disabled = threading.Event()
        service.disable_async(on_disabled=disabled.set)
        self.assertTrue(disabled.wait(1.0))
        self.assertIn("stop", first.calls)

        ready_again = threading.Event()
        self.assertTrue(
            service.start_async(on_ready=lambda _info: ready_again.set())
        )
        self.assertTrue(ready_again.wait(1.0))
        self.assertTrue(service.ready)
        self.assertEqual(len(factory.outputs), 2)
        self.assertIsNot(factory.outputs[0], factory.outputs[1])
        service.stop()

    def test_disable_during_warmup_cancels_without_stale_ready(self) -> None:
        factory = _OutputFactory(first_blocks=True)
        service = SpeechOutputService(output_factory=factory)  # type: ignore[arg-type]
        ready = threading.Event()
        disabled = threading.Event()
        service.start_async(on_ready=lambda _info: ready.set())

        first = factory.outputs[0]
        deadline = time.monotonic() + 1.0
        while first.state is not VoiceOutputState.STARTING and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIs(first.state, VoiceOutputState.STARTING)

        self.assertTrue(service.disable_async(on_disabled=disabled.set))
        self.assertTrue(disabled.wait(1.0))
        self.assertFalse(ready.is_set())
        self.assertFalse(service.ready)
        self.assertEqual(service.state, "off")
        self.assertIn("stop", first.calls)
        service.stop()


if __name__ == "__main__":
    unittest.main()

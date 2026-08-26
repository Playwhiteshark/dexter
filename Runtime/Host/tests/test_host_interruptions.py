from __future__ import annotations

import unittest

from Runtime.Host.app import DexterHost
from Runtime.Host.gateway_process import GatewayState
from Runtime.Host.goose_acp import GooseState


class _FakeSpeech:
    def __init__(self) -> None:
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1


class _FakeGoose:
    def __init__(self) -> None:
        self.state = GooseState.THINKING
        self.cancel_calls = 0
        self.submitted: list[str] = []

    def cancel_current(self) -> bool:
        self.cancel_calls += 1
        return self.state is GooseState.THINKING

    def submit_prompt(self, prompt: str) -> bool:
        if self.state is not GooseState.IDLE:
            return False
        self.submitted.append(prompt)
        self.state = GooseState.THINKING
        return True


class _FakeGateway:
    state = GatewayState.READY


class _FakeVoice:
    def __init__(self) -> None:
        self.capture_enabled: bool | None = None

    def set_capture_enabled(self, enabled: bool) -> None:
        self.capture_enabled = enabled


class _FakeOverlay:
    def __init__(self) -> None:
        self.status = ""
        self.cancel_enabled = True
        self.prompt_enabled: bool | None = None
        self.clear_enabled: bool | None = None
        self.agent_busy = True
        self.finish_interrupted_calls = 0
        self.clear_approval_calls = 0

    def set_status(self, status: str) -> None:
        self.status = status

    def set_cancel_enabled(self, enabled: bool) -> None:
        self.cancel_enabled = enabled

    def set_prompt_input_enabled(self, enabled: bool) -> None:
        self.prompt_enabled = enabled

    def set_clear_chat_enabled(self, enabled: bool) -> None:
        self.clear_enabled = enabled

    def set_agent_busy(self, busy: bool) -> None:
        self.agent_busy = busy

    def clear_approval(self) -> None:
        self.clear_approval_calls += 1

    def finish_interrupted_response(self) -> None:
        self.finish_interrupted_calls += 1


class HostInterruptionTests(unittest.TestCase):
    def _host(self) -> DexterHost:
        host = DexterHost.__new__(DexterHost)
        host._quitting = False
        host._agent_turn_pending = True
        host._pending_replacement_prompt = None
        host._discard_interrupted_response = False
        host.speech = _FakeSpeech()
        host.goose = _FakeGoose()
        host.gateway = _FakeGateway()
        host.voice = _FakeVoice()
        host.overlay = _FakeOverlay()
        return host

    def test_replacement_waits_for_cancelled_turn_to_reach_idle(self) -> None:
        host = self._host()

        host._accept_user_prompt("replacement")

        self.assertEqual(host._pending_replacement_prompt, "replacement")
        self.assertEqual(host.goose.cancel_calls, 1)
        self.assertEqual(host.goose.submitted, [])
        self.assertFalse(host.overlay.prompt_enabled)
        self.assertFalse(host.voice.capture_enabled)

        host.goose.state = GooseState.IDLE
        host._agent_turn_pending = False

        self.assertTrue(host._submit_pending_replacement_if_ready())
        self.assertIsNone(host._pending_replacement_prompt)
        self.assertEqual(host.goose.submitted, ["replacement"])
        self.assertTrue(host.overlay.agent_busy)

    def test_barge_in_freezes_visible_turn_and_drops_late_chunks(self) -> None:
        host = self._host()

        host._interrupt_active_turn_for_replacement()

        self.assertEqual(host.speech.interrupts, 1)
        self.assertEqual(host.overlay.clear_approval_calls, 1)
        self.assertEqual(host.overlay.finish_interrupted_calls, 1)
        self.assertTrue(host._discard_interrupted_response)
        self.assertFalse(host.overlay.cancel_enabled)
        self.assertEqual(host.goose.cancel_calls, 1)


if __name__ == "__main__":
    unittest.main()

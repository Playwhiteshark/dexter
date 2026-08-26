from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from Runtime.Host.app import DexterHost
from Runtime.Host.gateway_process import GatewayState
from Runtime.Host.goose_acp import GooseState
from Runtime.Host.interactive_overlay import DexterOverlay
from Runtime.Host.readiness_view import RuntimeReadinessView
from Runtime.Voice.stt import VoiceState


class RuntimeReadinessViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.overlay = DexterOverlay()
        self.overlay.show()
        self._app.processEvents()
        self.layout_changes = 0
        self.view = RuntimeReadinessView(
            self.overlay,
            on_layout_changed=self._layout_changed,
        )

    def tearDown(self) -> None:
        self.overlay.close()
        self.overlay.deleteLater()
        self._app.processEvents()

    def _layout_changed(self) -> None:
        self.layout_changes += 1

    def _settle_core(self) -> None:
        self.view.set_component("gateway", "ready")
        self.view.set_component("agent", "ready")
        self.view.set_component("mic", "ready")

    def test_startup_stages_hide_only_after_every_component_settles(self) -> None:
        self.view.begin_startup(voice_enabled=True)
        self.assertTrue(self.view.isVisible())
        self.assertEqual(self.view.state("gateway"), "starting")
        self.assertEqual(self.view.state("agent"), "waiting")
        self.assertEqual(self.view.state("mic"), "starting")
        self.assertEqual(self.view.state("voice"), "starting")
        self.assertEqual(self.view.title_label.text(), "Starting Dexter")

        self._settle_core()
        self.assertTrue(self.view.isVisible())

        self.view.set_component("voice", "ready")
        self.assertFalse(self.view.isVisible())
        self.assertGreater(self.layout_changes, 0)

    def test_disabled_voice_is_a_settled_state(self) -> None:
        self.view.begin_startup(voice_enabled=False)
        self._settle_core()

        self.assertEqual(self.view.state("voice"), "off")
        self.assertFalse(self.view.isVisible())

    def test_settled_panel_stays_hidden_when_parent_was_hidden_during_startup(self) -> None:
        self.overlay.hide()
        self.view.begin_startup(voice_enabled=False)
        self._settle_core()

        self.assertTrue(self.view.isHidden())

        self.overlay.show()
        self._app.processEvents()
        self.assertFalse(self.view.isVisible())

    def test_failure_keeps_component_and_reason_visible_without_log_dump(self) -> None:
        full_reason = (
            "Dexter could not start its Model Gateway. Port 11435 appears to "
            "already be in use.\n\nGateway output:\nTraceback noise"
        )
        self.view.set_component("gateway", "error", reason=full_reason)

        self.assertTrue(self.view.isVisible())
        self.assertEqual(self.view.title_label.text(), "Dexter needs attention")
        self.assertEqual(self.view.state("gateway"), "error")
        self.assertEqual(self.view.reason("gateway"), full_reason)

        row = self.view._rows["gateway"]
        self.assertIn("Port 11435", row.reason_label.text())
        self.assertNotIn("Gateway output", row.reason_label.text())
        self.assertEqual(row.reason_label.toolTip(), full_reason)


class _FakeReadiness:
    def __init__(self) -> None:
        self.states: dict[str, tuple[str, str | None]] = {}

    def set_component(
        self,
        component: str,
        state: str,
        *,
        reason: str | None = None,
    ) -> None:
        self.states[component] = (state, reason)


class _FakeOverlay:
    def __init__(self) -> None:
        self.status = "Starting Dexter…"
        self.recording = False
        self.prompt_enabled: bool | None = None
        self.clear_enabled: bool | None = None

    def set_status(self, status: str) -> None:
        self.status = status

    def set_recording(self, recording: bool) -> None:
        self.recording = recording

    def set_prompt_input_enabled(self, enabled: bool) -> None:
        self.prompt_enabled = enabled

    def set_clear_chat_enabled(self, enabled: bool) -> None:
        self.clear_enabled = enabled


class _FakeVoice:
    def __init__(self, state: VoiceState = VoiceState.IDLE) -> None:
        self.state = state
        self.capture_enabled: bool | None = None

    def set_capture_enabled(self, enabled: bool) -> None:
        self.capture_enabled = enabled


class _FakeGateway:
    def __init__(self, state: GatewayState) -> None:
        self.state = state


class _FakeGoose:
    def __init__(self, state: GooseState) -> None:
        self.state = state


class HostReadinessWiringTests(unittest.TestCase):
    def _host(
        self,
        *,
        gateway: GatewayState,
        goose: GooseState,
        voice: VoiceState = VoiceState.IDLE,
    ) -> DexterHost:
        host = DexterHost.__new__(DexterHost)
        host._quitting = False
        host._agent_turn_pending = False
        host._pending_replacement_prompt = None
        host.gateway = _FakeGateway(gateway)
        host.goose = _FakeGoose(goose)
        host.voice = _FakeVoice(voice)
        host.overlay = _FakeOverlay()
        host.readiness = _FakeReadiness()
        return host

    def test_mic_ready_cannot_claim_idle_before_core_runtime_is_ready(self) -> None:
        host = self._host(
            gateway=GatewayState.STARTING,
            goose=GooseState.STARTING,
        )

        host._on_voice_state(VoiceState.IDLE)

        self.assertEqual(host.readiness.states["mic"], ("ready", None))
        self.assertEqual(host.overlay.status, "Starting Dexter…")

        host.gateway.state = GatewayState.READY
        host.goose.state = GooseState.IDLE
        host._on_voice_state(VoiceState.IDLE)
        self.assertEqual(host.overlay.status, "Idle")

    def test_text_admission_does_not_depend_on_mic_or_voice_output(self) -> None:
        host = self._host(
            gateway=GatewayState.READY,
            goose=GooseState.IDLE,
            voice=VoiceState.ERROR,
        )

        host._refresh_interaction_admission()

        self.assertTrue(host.overlay.prompt_enabled)
        self.assertTrue(host.overlay.clear_enabled)
        self.assertTrue(host.voice.capture_enabled)


if __name__ == "__main__":
    unittest.main()

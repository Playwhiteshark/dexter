from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

import Runtime.Voice.windows_default_audio_sink as windows_sink
from Runtime.Voice.tts import AudioChunk


class _FakePlayer:
    def __init__(self) -> None:
        self.play_calls: list[np.ndarray] = []
        self.played = threading.Event()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def play(self, data: np.ndarray) -> None:
        self.play_calls.append(np.asarray(data, dtype=np.float32).copy())
        self.played.set()


class _FakeSpeaker:
    def __init__(self, player: _FakePlayer) -> None:
        self.id = "fake-speaker"
        self.name = "Fake Speaker"
        self._player = player

    def player(self, **_kwargs):
        return self._player


class WindowsDefaultAudioSinkTests(unittest.TestCase):
    def test_underflow_waits_for_rebuffer_reserve_and_reports_refill(self) -> None:
        player = _FakePlayer()
        speaker = _FakeSpeaker(player)
        fake_soundcard = types.ModuleType("soundcard")
        fake_soundcard.default_speaker = lambda: speaker

        traces: list[tuple[str, dict[str, object]]] = []
        refill = threading.Event()

        def trace(name: str, **fields: object) -> None:
            traces.append((name, fields))
            if name == "audio_refill_after_underflow":
                refill.set()

        sink = windows_sink.WindowsDefaultAudioSink(
            prebuffer_seconds=0.1,
            rebuffer_seconds=0.2,
            max_buffer_seconds=1.0,
            startup_max_wait_seconds=1.0,
            expected_sample_rate=10,
            expected_channels=1,
            playback_block_seconds=0.1,
            device_poll_seconds=0.01,
            trace_callback=trace,
        )
        cancel = threading.Event()

        with (
            patch.object(windows_sink, "soundcard_available", return_value=True),
            patch.dict(sys.modules, {"soundcard": fake_soundcard}),
        ):
            sink.start()
            try:
                sink.play(AudioChunk(np.array([1.0], dtype=np.float32), 10), cancel)
                self.assertTrue(player.played.wait(1.0))

                deadline = time.monotonic() + 1.0
                while sink._underflow_count < 1 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertGreaterEqual(sink._underflow_count, 1)

                # One frame is below the two-frame rebuffer reserve. The sink
                # must remain paused instead of treating this as a fresh startup.
                initial_play_calls = len(player.play_calls)
                sink.play(AudioChunk(np.array([2.0], dtype=np.float32), 10), cancel)
                time.sleep(0.05)
                self.assertEqual(len(player.play_calls), initial_play_calls)

                sink.play(AudioChunk(np.array([3.0], dtype=np.float32), 10), cancel)
                self.assertTrue(refill.wait(1.0))
                self.assertTrue(
                    any(name == "audio_refill_after_underflow" for name, _ in traces)
                )
                self.assertEqual(
                    sum(1 for name, _ in traces if name == "playback_start"),
                    1,
                )
            finally:
                sink.stop()

        self.assertTrue(player.closed)


if __name__ == "__main__":
    unittest.main()

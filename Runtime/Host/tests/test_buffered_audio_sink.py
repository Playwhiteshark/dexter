from __future__ import annotations

import threading
import unittest

import numpy as np

from Runtime.Voice.buffered_audio_sink import BufferedSoundDeviceAudioSink
from Runtime.Voice.tts import AudioChunk


class _FakeOutputStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.start_calls = 0
        self.abort_calls = 0
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.start_calls += 1

    def abort(self) -> None:
        self.started = False
        self.abort_calls += 1

    def close(self) -> None:
        self.closed = True


class _StreamFactory:
    def __init__(self) -> None:
        self.streams: list[_FakeOutputStream] = []

    def __call__(self, **kwargs) -> _FakeOutputStream:
        stream = _FakeOutputStream(**kwargs)
        self.streams.append(stream)
        return stream


class BufferedSoundDeviceAudioSinkTests(unittest.TestCase):
    def make_sink(self):
        factory = _StreamFactory()
        sink = BufferedSoundDeviceAudioSink(
            prebuffer_seconds=0.2,
            rebuffer_seconds=0.2,
            max_buffer_seconds=1.0,
            startup_max_wait_seconds=30.0,
            stream_factory=factory,
        )
        sink.start()
        return sink, factory

    def test_stream_starts_only_after_prebuffer_target(self) -> None:
        sink, factory = self.make_sink()
        cancel = threading.Event()
        try:
            sink.play(AudioChunk(np.array([1.0], dtype=np.float32), 10), cancel)
            self.assertEqual(len(factory.streams), 1)
            self.assertFalse(factory.streams[0].started)

            sink.play(AudioChunk(np.array([2.0], dtype=np.float32), 10), cancel)
            self.assertTrue(factory.streams[0].started)
            self.assertAlmostEqual(sink.buffered_seconds, 0.2)
        finally:
            sink.stop()

    def test_fixed_format_preopens_and_reuses_stream_after_interrupt(self) -> None:
        factory = _StreamFactory()
        sink = BufferedSoundDeviceAudioSink(
            prebuffer_seconds=0.2,
            rebuffer_seconds=0.2,
            max_buffer_seconds=1.0,
            startup_max_wait_seconds=30.0,
            expected_sample_rate=10,
            expected_channels=1,
            stream_factory=factory,
        )
        sink.start()
        self.assertEqual(len(factory.streams), 1)
        stream = factory.streams[0]
        self.assertFalse(stream.started)

        cancel = threading.Event()
        try:
            sink.play(AudioChunk(np.array([1.0, 2.0], dtype=np.float32), 10), cancel)
            self.assertEqual(stream.start_calls, 1)

            sink.interrupt()
            self.assertEqual(stream.abort_calls, 1)
            self.assertFalse(stream.closed)
            self.assertEqual(sink.buffered_seconds, 0.0)

            sink.play(AudioChunk(np.array([3.0, 4.0], dtype=np.float32), 10), cancel)
            self.assertEqual(len(factory.streams), 1)
            self.assertEqual(stream.start_calls, 2)
        finally:
            sink.stop()

        self.assertTrue(stream.closed)

    def test_callback_rebuffers_after_underflow_instead_of_chattering(self) -> None:
        sink, factory = self.make_sink()
        cancel = threading.Event()
        try:
            sink.play(AudioChunk(np.array([1.0, 2.0], dtype=np.float32), 10), cancel)
            stream = factory.streams[0]
            self.assertTrue(stream.started)

            first = np.empty((4, 1), dtype=np.float32)
            stream.callback(first, 4, None, None)
            np.testing.assert_array_equal(
                first[:, 0],
                np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
            )

            # One new frame is below the two-frame rebuffer reserve, so the
            # next callback stays silent instead of playing a tiny fragment.
            sink.play(AudioChunk(np.array([3.0], dtype=np.float32), 10), cancel)
            while_low = np.empty((1, 1), dtype=np.float32)
            stream.callback(while_low, 1, None, None)
            np.testing.assert_array_equal(
                while_low[:, 0], np.array([0.0], dtype=np.float32)
            )

            sink.play(AudioChunk(np.array([4.0], dtype=np.float32), 10), cancel)
            resumed = np.empty((2, 1), dtype=np.float32)
            stream.callback(resumed, 2, None, None)
            np.testing.assert_array_equal(
                resumed[:, 0], np.array([3.0, 4.0], dtype=np.float32)
            )
        finally:
            sink.stop()

    def test_interrupt_aborts_and_closes_lazy_stream(self) -> None:
        sink, factory = self.make_sink()
        cancel = threading.Event()
        sink.play(AudioChunk(np.array([1.0, 2.0], dtype=np.float32), 10), cancel)
        stream = factory.streams[0]

        sink.interrupt()

        self.assertEqual(stream.abort_calls, 1)
        self.assertTrue(stream.closed)
        self.assertEqual(sink.buffered_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()

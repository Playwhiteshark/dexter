from __future__ import annotations

import unittest

import numpy as np
import neutts.neutts as neutts_core

from Runtime.Voice.neutts2e_pipelined_stream import _incremental_overlap_segment


class IncrementalOverlapAddTests(unittest.TestCase):
    def test_matches_upstream_history_recompute_for_streamed_segments(self) -> None:
        rng = np.random.default_rng(42)
        stride = 25
        frames = [
            rng.normal(size=27).astype(np.float32),
            rng.normal(size=27).astype(np.float32),
            rng.normal(size=27).astype(np.float32),
            rng.normal(size=19).astype(np.float32),
        ]

        history: list[np.ndarray] = []
        previous: np.ndarray | None = None
        for index, frame in enumerate(frames):
            final = index == len(frames) - 1
            history.append(frame)
            upstream = neutts_core._linear_overlap_add(history, stride=stride)
            start = index * stride
            expected = upstream[start:] if final else upstream[start : start + stride]

            actual = _incremental_overlap_segment(
                previous,
                frame,
                stride=stride,
                final=final,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            previous = frame

    def test_first_frame_matches_upstream(self) -> None:
        frame = np.linspace(-1.0, 1.0, 27, dtype=np.float32)
        expected = neutts_core._linear_overlap_add([frame], stride=25)[:25]
        actual = _incremental_overlap_segment(None, frame, stride=25, final=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

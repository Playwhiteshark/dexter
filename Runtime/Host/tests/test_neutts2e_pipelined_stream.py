from __future__ import annotations

import unittest

import numpy as np

from Runtime.Voice.neutts2e_pipelined_stream import (
    _frame_weight,
    _incremental_overlap_segment,
)


def _full_variable_overlap(frames: list[np.ndarray], strides: list[int]) -> np.ndarray:
    offsets = [0]
    for stride in strides[:-1]:
        offsets.append(offsets[-1] + stride)

    total = max(offset + len(frame) for offset, frame in zip(offsets, frames))
    out = np.zeros(total, dtype=np.float32)
    weights = np.zeros(total, dtype=np.float32)
    for offset, frame in zip(offsets, frames):
        weight = _frame_weight(frame)
        end = offset + len(frame)
        out[offset:end] += frame * weight
        weights[offset:end] += weight

    if not np.all(weights > 0):
        raise AssertionError("reference overlap has uncovered samples")
    return out / weights


class IncrementalOverlapTests(unittest.TestCase):
    def test_variable_stride_matches_full_history_overlap(self) -> None:
        # Toy equivalent of a short first window followed by larger steady
        # windows. Each frame retains a two-sample tail beyond its emit stride,
        # mirroring NeuTTS's overlap geometry at a much smaller scale.
        strides = [5, 8, 8]
        frames = [
            np.linspace(0.1, 0.7, 7, dtype=np.float32),
            np.linspace(1.0, 2.0, 10, dtype=np.float32),
            np.linspace(2.0, 2.6, 6, dtype=np.float32),
        ]
        reference = _full_variable_overlap(frames, strides)

        first = _incremental_overlap_segment(
            None,
            frames[0],
            previous_stride=None,
            emit_samples=strides[0],
            final=False,
        )
        second = _incremental_overlap_segment(
            frames[0],
            frames[1],
            previous_stride=strides[0],
            emit_samples=strides[1],
            final=False,
        )
        final = _incremental_overlap_segment(
            frames[1],
            frames[2],
            previous_stride=strides[1],
            emit_samples=strides[2],
            final=True,
        )

        actual = np.concatenate([first, second, final])
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

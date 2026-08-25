"""Diagnostic worker for measuring uniform NeuTTS streaming window sizes.

This wrapper changes NeuTTS's first streaming frame count and forces Dexter's
steady-state window override to the same value. Production uses a smaller first
window followed by a larger steady-state window; these diagnostics deliberately
remain uniform so 25 / 40 / 50 comparisons still mean exactly what they say.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

frames_raw = os.environ.get("DEXTER_TTS_DIAGNOSTIC_STREAM_FRAMES", "25").strip()
try:
    STREAM_FRAMES = int(frames_raw)
except ValueError as exc:
    raise SystemExit("DEXTER_TTS_DIAGNOSTIC_STREAM_FRAMES must be an integer") from exc
if STREAM_FRAMES < 10 or STREAM_FRAMES > 100:
    raise SystemExit("Diagnostic stream frames must be between 10 and 100")

# The production pipeline defaults to a 25-frame first decode and 50-frame
# steady-state decodes. Uniform-window benchmarks explicitly pin both values.
os.environ["DEXTER_TTS_STEADY_STREAM_FRAMES"] = str(STREAM_FRAMES)

import neutts.neutts as neutts_core

_original_init = neutts_core.NeuTTS.__init__


def _latency_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.streaming_frames_per_chunk = STREAM_FRAMES
    self.streaming_stride_samples = STREAM_FRAMES * self.hop_length


neutts_core.NeuTTS.__init__ = _latency_init

from Runtime.Voice import neutts2e_worker


if __name__ == "__main__":
    raise SystemExit(neutts2e_worker.main())

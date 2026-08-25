"""Diagnostic worker for measuring NeuTTS streaming frame sizes.

This wrapper changes only NeuTTS's streaming frame count and matching overlap-add
stride before warmup. Production remains at the upstream 25-frame default.
Arc offload, the production pipelined Perth path, voice, codec, seed and text all
remain unchanged.
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
if STREAM_FRAMES < 10 or STREAM_FRAMES > 60:
    raise SystemExit("Diagnostic stream frames must be between 10 and 60")

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

"""Minimal librosa stub for Dexter's direct NeuTTS worker process.

NeuTTS imports ``librosa`` eagerly even though Dexter's production path never
encodes a new reference audio file: Paul ships with pre-encoded reference
codes. Importing the real librosa package pulls SciPy native extensions into
the worker at startup, and Windows Application Control may block unsigned
SciPy ``.pyd`` files. Because the worker script directory is first on
``sys.path``, this module satisfies NeuTTS's unused top-level import without
loading SciPy.

If Dexter later adds runtime speaker-reference encoding to this worker, replace
this stub with a lean supported decoder/resampler path rather than silently
using it for encoding.
"""

from __future__ import annotations


def load(*args, **kwargs):
    raise RuntimeError(
        "Dexter's production NeuTTS worker does not support runtime reference "
        "audio encoding; it uses pre-encoded speaker references."
    )

"""Minimal librosa compatibility shim for Dexter's direct NeuTTS worker.

NeuTTS imports ``librosa`` eagerly even though Dexter's production path never
encodes a new reference audio file: Paul ships with pre-encoded reference
codes. Importing the real librosa package pulls SciPy native extensions into
the worker at startup, and Windows Application Control may block unsigned
SciPy ``.pyd`` files. Because the worker script directory is first on
``sys.path``, this module satisfies only the small API surface imported by
NeuTTS/Perth without loading SciPy.

Perth imports ``librosa.resample`` at module import time, but Dexter and Perth
both operate at 24 kHz in the production speech path, so Perth does not call
its resampler. The identity-only implementation below deliberately fails if a
future path actually requests a sample-rate conversion rather than silently
performing a low-quality substitute.

If Dexter later adds runtime speaker-reference encoding or a different Perth
sample rate, replace this shim with a lean supported decoder/resampler path.
"""

from __future__ import annotations


def load(*args, **kwargs):
    raise RuntimeError(
        "Dexter's production NeuTTS worker does not support runtime reference "
        "audio encoding; it uses pre-encoded speaker references."
    )


def resample(y, *, orig_sr, target_sr, **kwargs):
    if float(orig_sr) == float(target_sr):
        return y
    raise RuntimeError(
        "Dexter's lean TTS librosa shim cannot resample audio. Production "
        "NeuTTS/Perth should remain at the shared 24 kHz sample rate."
    )

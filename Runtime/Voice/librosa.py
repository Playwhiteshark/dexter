"""Minimal librosa compatibility shim for Dexter's direct NeuTTS worker.

NeuTTS imports ``librosa`` eagerly even though Dexter's production path never
encodes a new reference audio file: Paul ships with pre-encoded reference
codes. Importing the real librosa package pulls SciPy native extensions into
the worker at startup, and Windows Application Control may block unsigned
SciPy ``.pyd`` files. Because the worker script directory is first on
``sys.path``, this module satisfies only the small API surface imported by
NeuTTS/Perth without loading SciPy.

Perth imports ``librosa.resample`` and genuinely uses it to convert between
Dexter's 24 kHz PCM and PerthNet's internal sample rate. Use torchaudio's sinc
resampler for that narrow compatibility surface; Torch/Torchaudio are already
loaded by the current NeuTTS/NeuCodec dependency stack, so this avoids adding a
second SciPy-backed audio stack to the persistent worker.

If Dexter later adds runtime speaker-reference encoding, replace ``load`` with
a lean supported decoder rather than broadening this compatibility shim.
"""

from __future__ import annotations

import numpy as np


def load(*args, **kwargs):
    raise RuntimeError(
        "Dexter's production NeuTTS worker does not support runtime reference "
        "audio encoding; it uses pre-encoded speaker references."
    )


def resample(y, *, orig_sr, target_sr, **kwargs):
    """SciPy-free subset of ``librosa.resample`` used by Perth.

    Perth passes 1-D numpy waveforms and the ``orig_sr`` / ``target_sr`` keyword
    arguments. ``res_type`` may also be supplied when detecting a watermark;
    torchaudio's sinc resampler remains suitable for that path and deliberately
    ignores librosa-specific tuning keywords.
    """
    if float(orig_sr) == float(target_sr):
        return y

    import torch
    import torchaudio.functional as audio_f

    array = np.asarray(y)
    original_dtype = array.dtype
    tensor = torch.as_tensor(array, dtype=torch.float32)
    converted = audio_f.resample(
        tensor,
        orig_freq=int(round(float(orig_sr))),
        new_freq=int(round(float(target_sr))),
    )
    result = converted.detach().cpu().numpy()
    if np.issubdtype(original_dtype, np.floating):
        return result.astype(original_dtype, copy=False)
    return result

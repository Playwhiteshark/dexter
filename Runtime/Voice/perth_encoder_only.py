"""Encoder-only Perth compatibility path for Dexter production TTS.

Perth's ``apply_watermark`` path uses only the watermark encoder and audio
processor, but ``PerthNet`` also constructs a three-branch decoder used for
detection/training. Dexter never calls watermark detection. This module loads
the exact encoder weights from Perth's packaged checkpoint and mirrors
``PerthImplicitWatermarker.apply_watermark`` while deliberately not exposing a
working detector.

The installed Perth package also imports ``scipy.signal.butter`` and
``scipy.signal.filtfilt`` eagerly from a utility module even though watermark
insertion never calls those filtering helpers. Windows Application Control can
block SciPy's unsigned native extensions. During Perth's import only, Dexter
provides fail-loud placeholders for those two unused names, then removes the
placeholder modules immediately so they cannot shadow a later legitimate SciPy
import elsewhere in the process.

This class is inference-only, so watermark application runs under
``torch.inference_mode()``. Perth's upstream inference path leaves autograd
enabled even after ``eval()``, causing Torch to retain roughly 30--40 MB of
native memory per spoken response in Dexter's streaming workload. Target-machine
A/B testing over 64 identical watermark calls reduced growth from about 245 MB
working set / 258 MB private commit to about 0.3 MB / 0.3 MB, while producing
byte-identical output.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from librosa import resample


def _import_perth_encoder_dependencies():
    """Import the narrow Perth encoder path without loading SciPy native code.

    Perth's utility module imports ``butter``/``filtfilt`` at module import time,
    but neither function participates in the STFT -> encoder -> inverse-STFT
    watermark insertion path used by Dexter. Temporarily satisfy those imports
    with functions that raise if unexpectedly called. Remove the temporary
    modules after Perth has captured the names so this compatibility layer cannot
    affect unrelated libraries later in the worker lifetime.
    """

    scipy_existing = sys.modules.get("scipy")
    signal_existing = sys.modules.get("scipy.signal")
    installed_stub = scipy_existing is None and signal_existing is None

    scipy_stub = None
    signal_stub = None
    if installed_stub:
        scipy_stub = types.ModuleType("scipy")
        scipy_stub.__path__ = []  # mark as package for normal import semantics
        signal_stub = types.ModuleType("scipy.signal")

        def _unused_filter(*args: Any, **kwargs: Any):
            raise RuntimeError(
                "Dexter's encoder-only Perth path unexpectedly called a SciPy "
                "Butterworth filtering helper; use upstream Perth/SciPy for that path"
            )

        signal_stub.butter = _unused_filter
        signal_stub.filtfilt = _unused_filter
        scipy_stub.signal = signal_stub
        sys.modules["scipy"] = scipy_stub
        sys.modules["scipy.signal"] = signal_stub

    try:
        from perth.perth_net import PREPACKAGED_MODELS_DIR
        from perth.perth_net.perth_net_implicit.audio_processor import AudioProcessor
        from perth.perth_net.perth_net_implicit.checkpoint_manager import CheckpointManager
        from perth.perth_net.perth_net_implicit.model import compute_subband_freq
        from perth.perth_net.perth_net_implicit.model.encoder import Encoder

        return (
            PREPACKAGED_MODELS_DIR,
            AudioProcessor,
            CheckpointManager,
            compute_subband_freq,
            Encoder,
        )
    finally:
        if installed_stub:
            if sys.modules.get("scipy.signal") is signal_stub:
                sys.modules.pop("scipy.signal", None)
            if sys.modules.get("scipy") is scipy_stub:
                sys.modules.pop("scipy", None)


(
    PREPACKAGED_MODELS_DIR,
    AudioProcessor,
    CheckpointManager,
    compute_subband_freq,
    Encoder,
) = _import_perth_encoder_dependencies()


class PerthEncoderOnlyWatermarker:
    """Apply Perth's implicit watermark without constructing its detector."""

    def __init__(
        self,
        run_name: str = "implicit",
        models_dir: str | Path = PREPACKAGED_MODELS_DIR,
        device: str = "cpu",
    ) -> None:
        cm = CheckpointManager(models_dir, run_name)
        checkpoint = cm.load_latest()
        if checkpoint is None:
            raise RuntimeError(f"No Perth checkpoint found for run {run_name!r}")

        hp = cm.hp
        subband = compute_subband_freq(hp)
        self.hp = hp
        self.encoder = Encoder(hp.hidden_size, subband).to(device)
        self.ap = AudioProcessor(hp).to(device)

        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            raise RuntimeError("Unexpected Perth checkpoint: missing model state")
        prefix = "encoder."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError("Perth checkpoint contains no encoder weights")
        self.encoder.load_state_dict(encoder_state, strict=True)

        # Match stock inference behavior. The current encoder has no dropout or
        # batch-norm, but eval() makes layer behavior explicit. inference_mode()
        # on apply_watermark additionally disables autograd/native graph state.
        self.encoder.eval()
        self.ap.eval()

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def parameter_bytes(self) -> int:
        return sum(int(p.numel()) * int(p.element_size()) for p in self.encoder.parameters())

    @staticmethod
    def _to_tensor(value: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value.copy())
        return value.to(dtype=torch.float, device=device)

    @torch.inference_mode()
    def apply_watermark(self, signal: np.ndarray | torch.Tensor, sample_rate: int, **_: Any):
        change_rate = sample_rate != self.hp.sample_rate
        if change_rate:
            signal = resample(
                signal,
                orig_sr=sample_rate,
                target_sr=self.hp.sample_rate,
            )

        tensor = self._to_tensor(signal, self.device)
        magspec, phase = self.ap.signal_to_magphase(tensor)
        magspec = magspec[None].to(self.device)
        wm_magspec, _mask = self.encoder(magspec)
        wm_signal = self.ap.magphase_to_signal(wm_magspec[0], phase)
        output = wm_signal.detach().cpu().numpy()

        if change_rate:
            output = resample(
                output,
                orig_sr=self.hp.sample_rate,
                target_sr=sample_rate,
            )
        return output

    def get_watermark(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "Dexter's encoder-only Perth path cannot detect watermarks; "
            "use upstream Perth for detection"
        )

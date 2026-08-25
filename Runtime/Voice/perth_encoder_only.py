"""Encoder-only Perth compatibility path for Dexter experiments.

Perth's production ``apply_watermark`` path uses only the watermark encoder and
audio processor, but ``PerthNet`` also constructs a three-branch decoder used
for detection/training. Dexter never calls watermark detection. This module
loads the exact encoder weights from Perth's packaged checkpoint and mirrors
``PerthImplicitWatermarker.apply_watermark`` while deliberately not exposing a
working detector.

This module is experimental until full output/performance A/B validation passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from librosa import resample
from perth.perth_net import PREPACKAGED_MODELS_DIR
from perth.perth_net.perth_net_implicit.audio_processor import AudioProcessor
from perth.perth_net.perth_net_implicit.checkpoint_manager import CheckpointManager
from perth.perth_net.perth_net_implicit.model import compute_subband_freq
from perth.perth_net.perth_net_implicit.model.encoder import Encoder


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
        # batch-norm, but eval() makes the intent explicit and future-safe.
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

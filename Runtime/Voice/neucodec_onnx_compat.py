"""Minimal NeuCodec compatibility surface for Dexter's ONNX-only TTS path.

NeuTTS imports ``NeuCodec`` and ``DistillNeuCodec`` at module import time even
when the configured codec is the precompiled ONNX decoder. Importing the real
``neucodec`` package eagerly loads its PyTorch encoder/model implementation and
a sizeable dependency/module graph that Dexter never uses at runtime.

This module can be installed as ``sys.modules['neucodec']`` *before* NeuTTS is
imported. It preserves the tiny ``NeuCodecOnnxDecoder`` API used by NeuTTS while
making unsupported PyTorch codec paths fail loudly rather than silently doing
something different.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


class _UnsupportedTorchCodec:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "Dexter's lean NeuCodec compatibility module supports only the "
            "ONNX decoder production path"
        )


class NeuCodec(_UnsupportedTorchCodec):
    pass


class DistillNeuCodec(_UnsupportedTorchCodec):
    pass


class NeuCodecOnnxDecoder:
    """Small API-compatible wrapper around the compiled NeuCodec decoder graph."""

    def __init__(self, onnx_path: str | Path):
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(
                "Dexter NeuCodec ONNX decoding requires onnxruntime"
            ) from exc

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            str(onnx_path),
            sess_options=options,
        )
        self.sample_rate = 24_000

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies: dict[str, str] | None = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        map_location: str = "cpu",
        **_: Any,
    ) -> "NeuCodecOnnxDecoder":
        if map_location != "cpu":
            raise ValueError("The NeuCodec ONNX decoder currently supports CPU only")

        from huggingface_hub import hf_hub_download

        onnx_path = hf_hub_download(
            repo_id=model_id,
            filename="model.onnx",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        # Preserve upstream's meta.yaml fetch/download-count behavior. It is not
        # needed for inference, so failure here should match hf_hub_download's
        # normal behavior rather than being silently ignored.
        hf_hub_download(
            repo_id=model_id,
            filename="meta.yaml",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        return cls(onnx_path)

    def encode_code(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "The ONNX NeuCodec decoder contains no reference-audio encoder"
        )

    def decode_code(self, codes: np.ndarray) -> np.ndarray:
        if not isinstance(codes, np.ndarray):
            raise ValueError("`Codes` should be an np.array.")
        if len(codes.shape) != 3 or codes.shape[1] != 1:
            raise ValueError("`Codes` should be of shape [B, 1, F].")
        return self.session.run(None, {"codes": codes})[0].astype(np.float32)


def install_as_neucodec() -> None:
    """Expose this lean surface under the module name NeuTTS imports."""
    if "neucodec" in sys.modules:
        raise RuntimeError(
            "Cannot install Dexter's lean NeuCodec surface after neucodec was imported"
        )

    module = types.ModuleType("neucodec")
    module.__file__ = __file__
    module.__package__ = ""
    module.NeuCodec = NeuCodec
    module.DistillNeuCodec = DistillNeuCodec
    module.NeuCodecOnnxDecoder = NeuCodecOnnxDecoder
    module.__all__ = ["NeuCodec", "DistillNeuCodec", "NeuCodecOnnxDecoder"]
    sys.modules["neucodec"] = module

"""Minimal Transformers compatibility surface for Dexter's quantized NeuTTS path.

NeuTTS imports ``AutoTokenizer`` and ``AutoModelForCausalLM`` at module import
time even when the configured backbone is GGUF/llama.cpp. Dexter's production
voice path is intentionally quantized-only, so importing the complete
Transformers package is unused overhead.

Install this module as ``sys.modules['transformers']`` before importing NeuTTS.
Any accidental attempt to use NeuTTS's PyTorch backbone fails loudly rather
than silently running through an incomplete compatibility layer.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _UnsupportedTransformersBackbone:
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "Dexter's lean Transformers compatibility module supports only the "
            "quantized GGUF/llama.cpp NeuTTS backbone"
        )


class AutoTokenizer(_UnsupportedTransformersBackbone):
    pass


class AutoModelForCausalLM(_UnsupportedTransformersBackbone):
    pass


def install_as_transformers() -> None:
    """Expose only the symbols NeuTTS imports on Dexter's GGUF production path."""
    if "transformers" in sys.modules:
        raise RuntimeError(
            "Cannot install Dexter's lean Transformers surface after transformers was imported"
        )

    module = types.ModuleType("transformers")
    module.__file__ = __file__
    module.__package__ = ""
    module.AutoTokenizer = AutoTokenizer
    module.AutoModelForCausalLM = AutoModelForCausalLM
    module.__all__ = ["AutoTokenizer", "AutoModelForCausalLM"]
    sys.modules["transformers"] = module

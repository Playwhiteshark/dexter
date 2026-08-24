"""Diagnostic-only NeuTTS worker that makes llama.cpp model loading verbose.

NeuTTS 1.4.1 hardcodes ``verbose=False`` when it constructs its GGUF
``llama_cpp.Llama`` backbone. That is sensible in production, but it hides the
only version-compatible proof that Vulkan really selected a device and
actually offloaded model layers. This wrapper forces ``verbose=True`` only for
accelerator diagnostics, then delegates to the unchanged production worker.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

import llama_cpp

_original_llama_init = llama_cpp.Llama.__init__


def _verbose_llama_init(self, *args, **kwargs):
    kwargs["verbose"] = True
    return _original_llama_init(self, *args, **kwargs)


llama_cpp.Llama.__init__ = _verbose_llama_init

from Runtime.Voice import neutts2e_worker


if __name__ == "__main__":
    raise SystemExit(neutts2e_worker.main())

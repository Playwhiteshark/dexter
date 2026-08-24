"""Diagnostic-only NeuTTS worker with verbose llama.cpp model loading.

NeuTTS 1.4.1 hardcodes ``verbose=False`` when it constructs its GGUF
``llama_cpp.Llama`` backbone. Accelerator diagnostics need the native model-load
log to prove device selection and layer offload, so this wrapper forces
``verbose=True`` before delegating to the unchanged production worker.

For controlled A/B measurements, ``DEXTER_TTS_DISABLE_WATERMARK=1`` also
prevents NeuTTS from loading Perth at all. Production does not set that flag;
this exists only to measure the watermarking stage without keeping its neural
model resident in memory.
"""

from __future__ import annotations

import os
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

if os.environ.get("DEXTER_TTS_DISABLE_WATERMARK", "0").strip() == "1":
    import neutts.neutts as neutts_core

    neutts_core._load_watermarker = lambda: None

from Runtime.Voice import neutts2e_worker


if __name__ == "__main__":
    raise SystemExit(neutts2e_worker.main())

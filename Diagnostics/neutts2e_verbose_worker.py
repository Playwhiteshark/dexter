"""Diagnostic-only NeuTTS worker with verbose llama.cpp model loading.

NeuTTS 1.4.1 hardcodes ``verbose=False`` when it constructs its GGUF
``llama_cpp.Llama`` backbone. Accelerator diagnostics need the native model-load
log to prove device selection and layer offload, so this wrapper forces
``verbose=True`` before delegating to the production worker.

``DEXTER_TTS_PIPELINE_WATERMARK=1`` installs the same production
Arc-generation / CPU-watermark pipeline used by Dexter. Perth remains loaded
and applied to all streamed audio; only scheduling changes.
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

if os.environ.get("DEXTER_TTS_PIPELINE_WATERMARK", "0").strip() == "1":
    from Runtime.Voice.neutts2e_pipelined_stream import install

    install()

from Runtime.Voice import neutts2e_worker


if __name__ == "__main__":
    raise SystemExit(neutts2e_worker.main())

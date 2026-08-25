"""Launch Dexter's NeuTTS worker with validated llama.cpp backbone tuning.

This wrapper exists only around the isolated TTS child process. It patches the
llama-cpp-python ``Llama`` constructor before the normal worker imports NeuTTS,
then executes the existing worker unchanged. No global/site-wide monkey patch is
installed and Ollama is unaffected.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

WORKER_PATH = Path(__file__).resolve().parent / "neutts2e_worker.py"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if value < 1:
        raise SystemExit(f"{name} must be positive")
    return value


def main() -> int:
    # Target-machine A/B (2026-08-25): 128/128 reduced the Q4 backbone by about
    # 325 MB working set / 644 MB private versus NeuTTS's upstream 512/512,
    # while sustained concurrent Gemma + TTS remained faster than realtime
    # (RTF 0.794 vs 0.807 at 512) with only ~78 ms extra first-audio latency.
    batch = _positive_int("DEXTER_TTS_BACKBONE_BATCH", 128)
    ubatch = _positive_int("DEXTER_TTS_BACKBONE_UBATCH", batch)

    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args, **kwargs):
        kwargs["n_batch"] = batch
        kwargs["n_ubatch"] = ubatch
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init
    print(
        f"[TTS] llama.cpp backbone batch={batch}, ubatch={ubatch}, mlock=upstream-default",
        file=sys.stderr,
        flush=True,
    )

    # The launcher itself is executed from Runtime/Voice, so that directory is
    # already sys.path[0]. This preserves Dexter's local librosa compatibility
    # shim exactly as the previous direct-worker launch did.
    runpy.run_path(str(WORKER_PATH), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

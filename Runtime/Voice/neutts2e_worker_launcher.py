"""Launch Dexter's NeuTTS worker with validated production tuning.

This wrapper exists only around the isolated TTS child process. It applies the
validated llama-cpp-python batch settings and, by default, installs Dexter's
minimal ONNX-only NeuCodec compatibility surface before NeuTTS is imported.
No global/site-wide monkey patches are installed and Ollama is unaffected.
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


def _binary_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip()
    if raw not in {"0", "1"}:
        raise SystemExit(f"{name} must be 0 or 1")
    return raw == "1"


def main() -> int:
    # Target-machine A/B (2026-08-25): 128/128 reduced the Q4 backbone by about
    # 325 MB working set / 644 MB private versus NeuTTS's upstream 512/512,
    # while sustained concurrent Gemma + TTS remained faster than realtime
    # (RTF 0.794 vs 0.807 at 512) with only ~78 ms extra first-audio latency.
    batch = _positive_int("DEXTER_TTS_BACKBONE_BATCH", 128)
    ubatch = _positive_int("DEXTER_TTS_BACKBONE_UBATCH", batch)

    # NeuTTS imports the full neucodec PyTorch/Transformers implementation even
    # when only the precompiled ONNX decoder is selected. Dexter's compatibility
    # module exposes the exact ONNX decoder API NeuTTS needs and deliberately
    # rejects the unused PyTorch codec paths. Repeated sustained-Gemma A/B tests
    # found effectively unchanged TTS performance (median RTF 0.777 vs 0.772,
    # TTFA 1.089 s vs 1.114 s) while saving ~52.5 MB working set / 48.1 MB
    # private memory. Keep an explicit fallback for diagnostics.
    lean_neucodec = _binary_flag("DEXTER_TTS_LEAN_NEUCODEC", True)
    if lean_neucodec:
        from neucodec_onnx_compat import install_as_neucodec

        install_as_neucodec()
        print(
            "[TTS] NeuCodec ONNX path=lean-compat",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[TTS] NeuCodec ONNX path=upstream",
            file=sys.stderr,
            flush=True,
        )

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

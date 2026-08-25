"""Measure Dexter's lean ONNX-only NeuCodec compatibility path.

Each variant runs in a fresh process. Both preload the same common production
stack (NumPy, llama.cpp, ONNX Runtime, Torch, Torchaudio, Transformers, Perth)
so the difference isolates the real ``neucodec`` package import graph versus
Dexter's minimal ONNX compatibility surface.

Nothing in this probe changes production configuration.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_lean_neucodec_ab.py
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"


def _configure_environment() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or REPO_ROOT)
    cache = local / "Dexter" / "Voice" / "neutts2e" / "cache"
    hf_home = cache / "huggingface"
    hub = hf_home / "hub"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")

    # Preserve Dexter's local librosa compatibility module for Perth.
    for path in (str(REPO_ROOT), str(VOICE_DIR)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(VOICE_DIR))
    return hub


def _find_model(hub: Path) -> Path:
    candidates = sorted(
        hub.glob("models--neuphonic--neucodec-onnx-decoder-int8/snapshots/*/model.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Dexter's cached NeuCodec INT8 model.onnx was not found")
    return candidates[0]


def _memory() -> dict[str, float]:
    from ctypes import wintypes

    class PMCEX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMCEX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PMCEX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    mb = 1024.0 * 1024.0
    return {
        "ws": round(counters.WorkingSetSize / mb, 1),
        "private": round(counters.PrivateUsage / mb, 1),
    }


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return round(after[key] - before[key], 1)


def _child(mode: str) -> int:
    hub = _configure_environment()
    model_path = _find_model(hub)

    # Fixed common stack: do not let the stock variant's eager dependencies get
    # counted merely because the lean variant already needs some of them for
    # NeuTTS/Perth.
    import numpy as np
    import llama_cpp  # noqa: F401
    import onnxruntime  # noqa: F401
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    import transformers  # noqa: F401
    import perth  # noqa: F401

    common = _memory()

    if mode == "lean":
        from Runtime.Voice.neucodec_onnx_compat import install_as_neucodec

        install_as_neucodec()

    import_started = time.perf_counter()
    import neutts.neutts  # noqa: F401
    import_seconds = time.perf_counter() - import_started
    after_neutts = _memory()

    from neucodec import NeuCodecOnnxDecoder

    session_started = time.perf_counter()
    decoder = NeuCodecOnnxDecoder(str(model_path))
    session_seconds = time.perf_counter() - session_started
    time.sleep(0.2)
    after_session = _memory()

    input_meta = decoder.session.get_inputs()[0]
    dtype = np.int64 if input_meta.type == "tensor(int64)" else np.int32
    codes = np.arange(50, dtype=dtype).reshape(1, 1, 50) % 100

    timings = []
    output = None
    for _ in range(6):
        started = time.perf_counter()
        output = decoder.decode_code(codes)
        timings.append(time.perf_counter() - started)
    assert output is not None
    after_decode = _memory()

    digest = hashlib.sha256(np.asarray(output, dtype=np.float32).tobytes()).hexdigest()
    result = {
        "mode": mode,
        "neutts_import_seconds": round(import_seconds, 3),
        "neutts_import_ws_mb": _delta(after_neutts, common, "ws"),
        "neutts_import_private_mb": _delta(after_neutts, common, "private"),
        "session_seconds": round(session_seconds, 3),
        "session_ws_mb": _delta(after_session, after_neutts, "ws"),
        "session_private_mb": _delta(after_session, after_neutts, "private"),
        "decode_ws_mb": _delta(after_decode, after_session, "ws"),
        "decode_private_mb": _delta(after_decode, after_session, "private"),
        "total_ws_mb": _delta(after_decode, common, "ws"),
        "total_private_mb": _delta(after_decode, common, "private"),
        "steady_decode_ms": round(statistics.median(timings[1:]) * 1000.0, 2),
        "output_shape": list(output.shape),
        "output_sha256": digest,
    }
    print("DEXTER_LEAN_CODEC_RESULT=" + json.dumps(result), flush=True)
    return 0


def _run(mode: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} failed with exit code {completed.returncode}")
    marker = "DEXTER_LEAN_CODEC_RESULT="
    line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} returned no result marker")
    return json.loads(line[len(marker):])


def _parent() -> int:
    print("Dexter stock-vs-lean NeuCodec import A/B")
    print("Quit Dexter before running this probe. Each row is a fresh Python process.\n")

    results = []
    for mode in ("stock", "lean"):
        print(f"Running {mode} ...", flush=True)
        result = _run(mode)
        results.append(result)
        print(
            f"  neuttsImportWS={result['neutts_import_ws_mb']:+7.1f} MB  "
            f"sessionWS={result['session_ws_mb']:+7.1f} MB  "
            f"totalWS={result['total_ws_mb']:+7.1f} MB  "
            f"totalPriv={result['total_private_mb']:+7.1f} MB  "
            f"decode={result['steady_decode_ms']:6.2f} ms"
        )

    stock, lean = results
    print("\nLean savings relative to stock:")
    print(
        f"  working set: {float(stock['total_ws_mb']) - float(lean['total_ws_mb']):.1f} MB\n"
        f"  private:     {float(stock['total_private_mb']) - float(lean['total_private_mb']):.1f} MB\n"
        f"  decode delta:{float(lean['steady_decode_ms']) - float(stock['steady_decode_ms']):+.2f} ms"
    )
    same = (
        stock["output_shape"] == lean["output_shape"]
        and stock["output_sha256"] == lean["output_sha256"]
    )
    print(f"  exact output bytes match: {same}")
    if not same:
        raise SystemExit("Lean decoder output did not exactly match stock NeuCodec")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("stock", "lean"), default="stock")
    args = parser.parse_args()
    if args.child:
        return _child(args.mode)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())

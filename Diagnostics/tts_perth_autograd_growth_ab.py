"""A/B Perth watermark memory growth with and without Torch autograd.

Dexter only performs inference with Perth, but upstream Perth's apply_watermark
path does not enter torch.no_grad()/inference_mode().  ``Module.eval()`` does not
disable autograd, so every watermark chunk currently builds a backward graph
that Dexter will never use.  This diagnostic tests whether that unnecessary
native Torch work is responsible for the repeated-request memory staircase.

Each fresh child loads Dexter's production encoder-only Perth watermarker, uses
one fixed 48 kHz? No: production is 24 kHz, 48,000-sample waveform (~2 seconds),
warms it, then executes eight identical watermark calls per pseudo-request. The
only A/B difference is an outer ``torch.inference_mode()`` around each call.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_perth_autograd_growth_ab.py
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"


def _configure() -> None:
    for value in (str(REPO_ROOT), str(VOICE_DIR)):
        if value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(VOICE_DIR))


def _mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _memory() -> dict[str, float]:
    if os.name != "nt":
        raise RuntimeError("This probe currently targets Windows")
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
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PMCEX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PMCEX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "ws": _mb(counters.WorkingSetSize),
        "private": _mb(counters.PrivateUsage),
    }


def _child(mode: str) -> int:
    _configure()
    import numpy as np
    import torch
    from Runtime.Voice.perth_encoder_only import PerthEncoderOnlyWatermarker

    # A deterministic non-silent waveform exercises the exact STFT -> encoder ->
    # inverse-STFT watermark path without involving NeuTTS or ONNX.
    sample_rate = 24_000
    samples = 48_000
    t = np.arange(samples, dtype=np.float32) / sample_rate
    waveform = (
        0.16 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 617.0 * t)
    ).astype(np.float32)

    wm = PerthEncoderOnlyWatermarker(device="cpu")

    def apply() -> np.ndarray:
        if mode == "inference":
            with torch.inference_mode():
                return wm.apply_watermark(waveform, sample_rate=sample_rate)
        return wm.apply_watermark(waveform, sample_rate=sample_rate)

    # Pay one-time FFT/convolution initialization before the measured baseline.
    for _ in range(3):
        out = apply()
    expected = hashlib.sha256(np.asarray(out, dtype=np.float32).tobytes()).hexdigest()
    del out
    gc.collect()
    time.sleep(0.2)

    baseline = _memory()
    previous = baseline
    rows: list[dict[str, Any]] = []

    # Eight Perth invocations approximates the number of watermarked decode
    # windows in the identical full-pipeline request that leaked ~30-40 MB.
    for request in range(1, 9):
        started = time.perf_counter()
        digest = None
        for _ in range(8):
            out = apply()
            digest = hashlib.sha256(
                np.asarray(out, dtype=np.float32).tobytes()
            ).hexdigest()
        elapsed = time.perf_counter() - started
        del out
        time.sleep(0.15)
        current = _memory()
        row = {
            "request": request,
            "ws": current["ws"],
            "private": current["private"],
            "dws": current["ws"] - previous["ws"],
            "dprivate": current["private"] - previous["private"],
            "total_ws": current["ws"] - baseline["ws"],
            "total_private": current["private"] - baseline["private"],
            "seconds": elapsed,
            "hash_ok": digest == expected,
        }
        rows.append(row)
        previous = current

    result = {
        "mode": mode,
        "baseline": baseline,
        "rows": rows,
        "hash": expected,
        "final_ws_growth": rows[-1]["total_ws"],
        "final_private_growth": rows[-1]["total_private"],
    }
    print("DEXTER_PERTH_AUTOGRAD_RESULT=" + json.dumps(result), flush=True)
    return 0


def _run(mode: str) -> dict[str, Any]:
    flags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
        creationflags=flags,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} child failed with exit code {completed.returncode}")
    marker = "DEXTER_PERTH_AUTOGRAD_RESULT="
    line = next((ln for ln in completed.stdout.splitlines() if ln.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} child returned no result")
    return json.loads(line[len(marker):])


def _print_result(result: dict[str, Any]) -> None:
    print(f"\n{result['mode']}")
    print("  pseudo-request   WS MB   Private MB   dWS req   dPriv req   dWS total   dPriv total   8 calls")
    print("  --------------  ------  ----------  --------  ----------  ---------  -----------  --------")
    base = result["baseline"]
    print(
        f"  baseline        {base['ws']:6.1f}  {base['private']:10.1f}  "
        f"{'-':>8}  {'-':>10}  {'+0.0':>9}  {'+0.0':>11}  {'-':>8}"
    )
    for row in result["rows"]:
        print(
            f"  {row['request']:<14}  {row['ws']:6.1f}  {row['private']:10.1f}  "
            f"{row['dws']:+8.1f}  {row['dprivate']:+10.1f}  "
            f"{row['total_ws']:+9.1f}  {row['total_private']:+11.1f}  "
            f"{row['seconds']:8.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("current", "inference"), default="current")
    args = parser.parse_args()
    if args.child:
        return _child(args.mode)

    print("Dexter Perth autograd native-memory A/B")
    print(
        "Fresh processes repeatedly watermark the exact same waveform. The only "
        "difference is current autograd-enabled execution versus torch.inference_mode()."
    )
    current = _run("current")
    inference = _run("inference")
    _print_result(current)
    _print_result(inference)

    print("\nComparison after eight pseudo-requests (64 identical watermark calls):")
    print(
        f"  current:   dWS={current['final_ws_growth']:+.1f} MB, "
        f"dPrivate={current['final_private_growth']:+.1f} MB"
    )
    print(
        f"  inference: dWS={inference['final_ws_growth']:+.1f} MB, "
        f"dPrivate={inference['final_private_growth']:+.1f} MB"
    )
    hashes_match = current["hash"] == inference["hash"] and all(
        row["hash_ok"] for result in (current, inference) for row in result["rows"]
    )
    print(f"  exact watermark output match: {hashes_match}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

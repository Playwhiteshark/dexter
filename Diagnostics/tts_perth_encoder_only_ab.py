"""Fresh-process stock-vs-encoder-only Perth A/B for Dexter TTS.

Both variants import the same Perth/Torch/Torchaudio module graph. The only
measured difference is the long-lived watermarker object:

- stock: PerthImplicitWatermarker -> full PerthNet encoder + 3-branch decoder
- lean:  PerthEncoderOnlyWatermarker -> encoder + audio processor only

The probe measures steady object memory, first-call memory, watermark latency,
parameter bytes, and exact float32 output parity on a deterministic signal.
Nothing here changes Dexter production settings.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_perth_encoder_only_ab.py
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
SAMPLE_RATE = 24_000


def _configure_environment() -> None:
    for path in (str(REPO_ROOT), str(VOICE_DIR)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(VOICE_DIR))


def _mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


def _memory() -> dict[str, float]:
    if os.name != "nt":
        raise RuntimeError("This probe currently targets Dexter's Windows runtime")

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
        "working_set_mb": _mb(counters.WorkingSetSize),
        "private_mb": _mb(counters.PrivateUsage),
    }


def _settled_memory() -> dict[str, float]:
    gc.collect()
    time.sleep(0.15)
    return _memory()


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return round(after[key] - before[key], 1)


def _signal(np: Any):
    # Deterministic speech-like multitone/noise signal. Keep amplitude modest so
    # the watermark operates in the same numeric range as NeuCodec output.
    count = SAMPLE_RATE * 2
    t = np.arange(count, dtype=np.float32) / np.float32(SAMPLE_RATE)
    rng = np.random.default_rng(20260825)
    signal = (
        0.14 * np.sin(2 * np.pi * 173.0 * t)
        + 0.08 * np.sin(2 * np.pi * 421.0 * t)
        + 0.025 * rng.standard_normal(count).astype(np.float32)
    )
    return signal.astype(np.float32)


def _parameter_bytes(module: Any) -> int:
    return sum(int(p.numel()) * int(p.element_size()) for p in module.parameters())


def _child(mode: str) -> int:
    _configure_environment()

    # Import the exact same dependency/module graph before the memory baseline.
    # This isolates the cost of constructing/retaining Perth's unused decoder.
    import numpy as np
    import perth
    import torch  # noqa: F401
    from Runtime.Voice.perth_encoder_only import PerthEncoderOnlyWatermarker

    if perth.PerthImplicitWatermarker is None:
        raise RuntimeError("Upstream PerthImplicitWatermarker is unavailable")

    baseline = _settled_memory()
    started = time.perf_counter()
    if mode == "stock":
        watermarker = perth.PerthImplicitWatermarker(device="cpu")
        parameter_bytes = _parameter_bytes(watermarker.perth_net)
        encoder_bytes = _parameter_bytes(watermarker.perth_net.encoder)
        decoder_bytes = _parameter_bytes(watermarker.perth_net.decoder)
    else:
        watermarker = PerthEncoderOnlyWatermarker(device="cpu")
        parameter_bytes = watermarker.parameter_bytes()
        encoder_bytes = parameter_bytes
        decoder_bytes = 0
    construct_seconds = time.perf_counter() - started
    after_construct = _settled_memory()

    signal = _signal(np)
    first_started = time.perf_counter()
    first = np.asarray(
        watermarker.apply_watermark(signal.copy(), sample_rate=SAMPLE_RATE),
        dtype=np.float32,
    ).reshape(-1)
    first_seconds = time.perf_counter() - first_started
    after_first = _settled_memory()

    timings = []
    last = first
    for _ in range(5):
        started = time.perf_counter()
        last = np.asarray(
            watermarker.apply_watermark(signal.copy(), sample_rate=SAMPLE_RATE),
            dtype=np.float32,
        ).reshape(-1)
        timings.append(time.perf_counter() - started)

    final = _settled_memory()
    digest = hashlib.sha256(last.tobytes()).hexdigest()
    result = {
        "mode": mode,
        "baseline": baseline,
        "after_construct": after_construct,
        "after_first": after_first,
        "final": final,
        "construct_seconds": round(construct_seconds, 4),
        "construct_ws_mb": _delta(after_construct, baseline, "working_set_mb"),
        "construct_private_mb": _delta(after_construct, baseline, "private_mb"),
        "first_call_ws_mb": _delta(after_first, after_construct, "working_set_mb"),
        "first_call_private_mb": _delta(after_first, after_construct, "private_mb"),
        "final_ws_mb": _delta(final, baseline, "working_set_mb"),
        "final_private_mb": _delta(final, baseline, "private_mb"),
        "parameter_mb": _mb(parameter_bytes),
        "encoder_parameter_mb": _mb(encoder_bytes),
        "decoder_parameter_mb": _mb(decoder_bytes),
        "first_call_ms": round(first_seconds * 1000.0, 3),
        "steady_median_ms": round(statistics.median(timings) * 1000.0, 3),
        "output_samples": int(last.size),
        "sha256": digest,
    }
    print("DEXTER_PERTH_ENCODER_AB_RESULT=" + json.dumps(result), flush=True)
    return 0


def _run(mode: str) -> dict[str, Any]:
    creationflags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
        creationflags=creationflags,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} failed with exit code {completed.returncode}")
    marker = "DEXTER_PERTH_ENCODER_AB_RESULT="
    line = next((ln for ln in completed.stdout.splitlines() if ln.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} returned no result marker")
    return json.loads(line[len(marker) :])


def _parent() -> int:
    print("Dexter stock-vs-encoder-only Perth A/B")
    print(
        "Quit Dexter before running this probe. Both rows import the same Perth/Torch "
        "module graph; only the retained watermarker object differs.\n"
    )

    results = []
    for mode in ("stock", "lean"):
        print(f"Running {mode} ...", flush=True)
        result = _run(mode)
        results.append(result)
        print(
            f"  params={result['parameter_mb']:>6.1f} MB "
            f"(enc={result['encoder_parameter_mb']:.1f}, dec={result['decoder_parameter_mb']:.1f})  "
            f"constructWS={result['construct_ws_mb']:+7.1f} MB  "
            f"finalWS={result['final_ws_mb']:+7.1f} MB  "
            f"finalPriv={result['final_private_mb']:+7.1f} MB"
        )
        print(
            f"  construct={result['construct_seconds']:.3f}s  "
            f"first={result['first_call_ms']:.2f} ms  "
            f"steady={result['steady_median_ms']:.2f} ms"
        )

    stock, lean = results
    print("\nEncoder-only relative to stock:")
    print(f"  working set saved: {stock['final_ws_mb'] - lean['final_ws_mb']:.1f} MB")
    print(f"  private saved:     {stock['final_private_mb'] - lean['final_private_mb']:.1f} MB")
    print(f"  parameter bytes saved: {stock['parameter_mb'] - lean['parameter_mb']:.1f} MB")
    print(f"  steady latency delta: {lean['steady_median_ms'] - stock['steady_median_ms']:+.2f} ms")
    print(f"  exact output bytes match: {lean['sha256'] == stock['sha256']}")
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

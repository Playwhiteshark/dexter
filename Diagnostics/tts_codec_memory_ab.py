"""Isolated NeuCodec/ONNX Runtime memory A/B for Dexter TTS.

Each configuration runs in a fresh Python process. The probe uses Dexter's
cached INT8 ONNX decoder and production ORT 2/1 environment, then separates:

- base NumPy + ONNX Runtime import cost
- NeuCodec package import cost
- ONNX InferenceSession construction cost
- first-decode persistent allocation
- steady decode latency

It also tests whether disabling ORT's CPU memory arena and memory pattern can
reduce the decoder footprint. Nothing here changes Dexter production settings.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_codec_memory_ab.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "neuphonic/neucodec-onnx-decoder-int8"


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
    return hub


def _find_model(hub: Path) -> Path:
    candidates = sorted(
        hub.glob("models--neuphonic--neucodec-onnx-decoder-int8/snapshots/*/model.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find Dexter's cached NeuCodec INT8 model.onnx. "
            "Launch working Voice once first."
        )
    return candidates[0]


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

    class MEMSTAT(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
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
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMSTAT)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL

    process = kernel32.GetCurrentProcess()
    counters = PMCEX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())

    status = MEMSTAT()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())

    return {
        "working_set_mb": _mb(counters.WorkingSetSize),
        "private_mb": _mb(counters.PrivateUsage),
        "available_ram_mb": _mb(status.ullAvailPhys),
    }


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return round(after[key] - before[key], 1)


def _sample_codes(session: Any, np: Any):
    input_meta = session.get_inputs()[0]
    dtype_by_type = {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    dtype = dtype_by_type.get(input_meta.type)
    if dtype is None:
        raise RuntimeError(f"Unexpected NeuCodec input type: {input_meta.type}")
    return input_meta.name, np.zeros((1, 1, 50), dtype=dtype)


def _session_options(ort: Any, *, arena: bool, mem_pattern: bool):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = arena
    so.enable_mem_pattern = mem_pattern
    return so


def _session_option_snapshot(session: Any) -> dict[str, object]:
    try:
        so = session.get_session_options()
        return {
            "intra_op_num_threads": int(so.intra_op_num_threads),
            "inter_op_num_threads": int(so.inter_op_num_threads),
            "enable_cpu_mem_arena": bool(so.enable_cpu_mem_arena),
            "enable_mem_pattern": bool(so.enable_mem_pattern),
            "execution_mode": str(so.execution_mode),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _child(mode: str) -> int:
    hub = _configure_environment()
    model_path = _find_model(hub)

    baseline = _memory()
    import numpy as np
    import onnxruntime as ort
    post_base_imports = _memory()

    wrapper_import = None
    post_wrapper_import = post_base_imports
    if mode == "neucodec":
        before_wrapper = _memory()
        from neucodec import NeuCodecOnnxDecoder
        post_wrapper_import = _memory()
        wrapper_import = {
            "delta_working_set_mb": _delta(post_wrapper_import, before_wrapper, "working_set_mb"),
            "delta_private_mb": _delta(post_wrapper_import, before_wrapper, "private_mb"),
        }

    session_before = _memory()
    started = time.perf_counter()
    if mode == "neucodec":
        decoder = NeuCodecOnnxDecoder(str(model_path))
        session = decoder.session
        arena = True
        mem_pattern = True
    else:
        arena = mode != "raw_no_arena" and mode != "raw_no_arena_no_pattern"
        mem_pattern = mode != "raw_no_arena_no_pattern"
        so = _session_options(ort, arena=arena, mem_pattern=mem_pattern)
        session = ort.InferenceSession(str(model_path), sess_options=so)
        decoder = None
    create_seconds = time.perf_counter() - started
    time.sleep(0.20)
    session_after = _memory()

    input_name, codes = _sample_codes(session, np)
    decode_started = time.perf_counter()
    if decoder is not None:
        decoded = decoder.decode_code(codes)
    else:
        decoded = session.run(None, {input_name: codes})[0]
    first_decode_seconds = time.perf_counter() - decode_started
    assert decoded is not None
    time.sleep(0.10)
    first_decode_after = _memory()

    steady_times: list[float] = []
    for _ in range(8):
        call_started = time.perf_counter()
        if decoder is not None:
            decoded = decoder.decode_code(codes)
        else:
            decoded = session.run(None, {input_name: codes})[0]
        steady_times.append(time.perf_counter() - call_started)
    assert decoded is not None
    steady_after = _memory()

    result = {
        "mode": mode,
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 1),
        "baseline": baseline,
        "base_import_delta_ws_mb": _delta(post_base_imports, baseline, "working_set_mb"),
        "base_import_delta_private_mb": _delta(post_base_imports, baseline, "private_mb"),
        "wrapper_import": wrapper_import,
        "session_create_seconds": round(create_seconds, 4),
        "session_delta_ws_mb": _delta(session_after, session_before, "working_set_mb"),
        "session_delta_private_mb": _delta(session_after, session_before, "private_mb"),
        "session_delta_available_mb": _delta(session_after, session_before, "available_ram_mb"),
        "first_decode_seconds": round(first_decode_seconds, 4),
        "first_decode_delta_ws_mb": _delta(first_decode_after, session_after, "working_set_mb"),
        "first_decode_delta_private_mb": _delta(first_decode_after, session_after, "private_mb"),
        "steady_decode_median_ms": round(statistics.median(steady_times) * 1000.0, 2),
        "steady_decode_min_ms": round(min(steady_times) * 1000.0, 2),
        "steady_extra_ws_mb": _delta(steady_after, first_decode_after, "working_set_mb"),
        "steady_extra_private_mb": _delta(steady_after, first_decode_after, "private_mb"),
        "session_options": _session_option_snapshot(session),
        "providers": session.get_providers(),
        "input_name": input_name,
        "input_type": session.get_inputs()[0].type,
        "output_shape": list(decoded.shape),
        "final_ws_mb": steady_after["working_set_mb"],
        "final_private_mb": steady_after["private_mb"],
    }
    print("DEXTER_CODEC_RESULT=" + json.dumps(result), flush=True)
    return 0


def _parent() -> int:
    print("Dexter isolated NeuCodec / ONNX Runtime memory A/B")
    print("Quit Dexter before running this probe. Each row is a fresh Python process.\n")

    variants = [
        "neucodec",
        "raw_default",
        "raw_no_arena",
        "raw_no_arena_no_pattern",
    ]
    results: list[dict[str, Any]] = []
    marker = "DEXTER_CODEC_RESULT="

    for mode in variants:
        print(f"Running {mode} ...", flush=True)
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
        line = next((ln for ln in completed.stdout.splitlines() if ln.startswith(marker)), None)
        if line is None:
            print(completed.stdout, end="")
            raise SystemExit(f"{mode} returned no result marker")
        result = json.loads(line[len(marker):])
        results.append(result)
        wrapper = result.get("wrapper_import")
        wrapper_text = (
            f" wrapperWS=+{wrapper['delta_working_set_mb']:.1f} MB"
            if wrapper is not None else ""
        )
        print(
            f"  sessionWS=+{result['session_delta_ws_mb']:>6.1f} MB  "
            f"sessionPriv=+{result['session_delta_private_mb']:>6.1f} MB  "
            f"firstDecodeWS=+{result['first_decode_delta_ws_mb']:>5.1f} MB  "
            f"steady={result['steady_decode_median_ms']:>7.2f} ms"
            f"{wrapper_text}"
        )
        print(f"  ORT options: {result['session_options']}")

    raw_default = next(r for r in results if r["mode"] == "raw_default")
    print("\nRelative to raw ORT default session:")
    for result in results:
        if result["mode"] == "raw_default":
            continue
        ws_saved = raw_default["session_delta_ws_mb"] - result["session_delta_ws_mb"]
        priv_saved = raw_default["session_delta_private_mb"] - result["session_delta_private_mb"]
        latency_delta = result["steady_decode_median_ms"] - raw_default["steady_decode_median_ms"]
        print(
            f"  {result['mode']:<24} session WS saved={ws_saved:>7.1f} MB  "
            f"private saved={priv_saved:>7.1f} MB  "
            f"steady decode delta={latency_delta:+7.2f} ms"
        )

    print(f"\nCached INT8 ONNX model size: {raw_default['model_size_mb']} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("neucodec", "raw_default", "raw_no_arena", "raw_no_arena_no_pattern"),
        default="raw_default",
    )
    args = parser.parse_args()
    if args.child:
        return _child(args.mode)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())

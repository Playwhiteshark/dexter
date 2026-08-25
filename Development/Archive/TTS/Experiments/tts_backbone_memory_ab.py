"""Isolated llama.cpp/Vulkan memory A/B for Dexter's NeuTTS Q4 backbone.

Each configuration is launched in a fresh child process so llama.cpp/Vulkan
allocations from one run cannot contaminate the next. This probe loads only the
Q4 backbone; it does not load NeuCodec, Perth, or synthesize audio.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_backbone_memory_ab.py

The configurations intentionally keep n_ctx=2048, full Arc/Vulkan offload,
Flash Attention, and the validated 8 generation threads unchanged. We vary only
`use_mlock` and batch/ubatch size.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"


def _configure_environment() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or REPO_ROOT)
    cache = local / "Dexter" / "Voice" / "neutts2e" / "cache"
    hf_home = cache / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
    return hf_home / "hub"


def _find_model(hub: Path) -> Path:
    candidates = sorted(
        hub.glob("models--neuphonic--neutts-2e-q4-gguf/snapshots/*/*.gguf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find Dexter's cached NeuTTS Q4 GGUF. Launch working Voice once first."
        )
    return candidates[0]


def _mb(value: int) -> float:
    return round(value / (1024 * 1024), 1)


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


def _child(label: str, mlock: bool, batch: int, ubatch: int) -> int:
    hub = _configure_environment()
    model_path = _find_model(hub)

    # Match current production backbone settings except the explicit A/B knobs.
    from llama_cpp import Llama

    before = _memory()
    started = time.perf_counter()
    model = Llama(
        model_path=str(model_path),
        verbose=False,
        n_gpu_layers=-1,
        n_ctx=2048,
        n_batch=batch,
        n_ubatch=ubatch,
        n_threads=8,
        n_threads_batch=os.cpu_count() or 16,
        use_mlock=mlock,
        flash_attn=True,
        seed=42,
    )
    load_seconds = time.perf_counter() - started
    assert model is not None
    time.sleep(0.25)
    after = _memory()

    result = {
        "label": label,
        "mlock": mlock,
        "batch": batch,
        "ubatch": ubatch,
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 1),
        "load_seconds": round(load_seconds, 3),
        "before": before,
        "after": after,
        "delta_working_set_mb": round(after["working_set_mb"] - before["working_set_mb"], 1),
        "delta_private_mb": round(after["private_mb"] - before["private_mb"], 1),
        "delta_available_ram_mb": round(after["available_ram_mb"] - before["available_ram_mb"], 1),
    }
    print("DEXTER_BACKBONE_RESULT=" + json.dumps(result), flush=True)
    return 0


def _parent() -> int:
    print("Dexter isolated Q4 backbone memory A/B")
    print("Quit Dexter before running this probe. Each row is a fresh Python process.\n")

    variants = [
        ("production", True, 512, 512),
        ("no_mlock", False, 512, 512),
        ("batch256_mlock", True, 256, 256),
        ("batch256_no_mlock", False, 256, 256),
        ("batch128_mlock", True, 128, 128),
    ]
    results: list[dict[str, object]] = []

    for label, mlock, batch, ubatch in variants:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--label", label,
            "--mlock", "1" if mlock else "0",
            "--batch", str(batch),
            "--ubatch", str(ubatch),
        ]
        print(f"Running {label} ...", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            raise SystemExit(f"{label} failed with exit code {completed.returncode}")
        marker = "DEXTER_BACKBONE_RESULT="
        line = next((ln for ln in completed.stdout.splitlines() if ln.startswith(marker)), None)
        if line is None:
            print(completed.stdout, end="")
            raise SystemExit(f"{label} returned no result marker")
        result = json.loads(line[len(marker):])
        results.append(result)
        print(
            f"  load={result['load_seconds']:>6.2f}s  "
            f"dWS={result['delta_working_set_mb']:>8.1f} MB  "
            f"dPriv={result['delta_private_mb']:>8.1f} MB  "
            f"dAvail={result['delta_available_ram_mb']:>8.1f} MB"
        )

    base = results[0]
    print("\nRelative to production:")
    for result in results[1:]:
        ws_saved = float(base["delta_working_set_mb"]) - float(result["delta_working_set_mb"])
        priv_saved = float(base["delta_private_mb"]) - float(result["delta_private_mb"])
        print(
            f"  {result['label']:<22} WS saved={ws_saved:>7.1f} MB  "
            f"private saved={priv_saved:>7.1f} MB"
        )

    print(f"\nCached GGUF size: {results[0]['model_size_mb']} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--mlock", choices=("0", "1"), default="1")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    args = parser.parse_args()
    if args.child:
        return _child(args.label, args.mlock == "1", args.batch, args.ubatch)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())

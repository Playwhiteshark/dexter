"""Stage-by-stage memory attribution for Dexter's production TTS worker.

Run this with Dexter fully quit so a second NeuTTS/Arc allocation does not skew
results:

    uv run python Diagnostics/tts_memory_probe.py

The probe intentionally mirrors the production worker's import environment and
Q4 Arc/Vulkan configuration, but it is not imported by Dexter at runtime. It
prints every stage as it completes and also writes a JSON report under
Diagnostics/. Use ``--imports-only`` to stop before model construction.

On Windows, ``private_mb`` comes from PROCESS_MEMORY_COUNTERS_EX.PrivateUsage
(the process's private committed memory), while ``working_set_mb`` is resident
RAM currently charged to the process. ``available_ram_mb`` is system-wide and
is useful on Intel Arc UMA because some GPU/shared allocations are not obvious
from a single process's working set alone.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"

# Running a script by path makes its own directory sys.path[0]. Add the repo
# root explicitly so imports such as Runtime.Voice.* behave like the production
# module launch. Keep Runtime/Voice before it so NeuTTS/Perth still resolve
# Dexter's small SciPy-free librosa compatibility module first.
_repo_text = str(REPO_ROOT)
if _repo_text in sys.path:
    sys.path.remove(_repo_text)
sys.path.insert(0, _repo_text)
_voice_text = str(VOICE_DIR)
if _voice_text in sys.path:
    sys.path.remove(_voice_text)
sys.path.insert(0, _voice_text)

# Match the validated production worker configuration before native libraries
# are imported/initialized.
os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")
os.environ.setdefault("DEXTER_TTS_FIRST_STREAM_FRAMES", "40")
os.environ.setdefault("DEXTER_TTS_STEADY_STREAM_FRAMES", "50")

MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"


def _mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


if os.name == "nt":
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
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

    class MEMORYSTATUSEX(ctypes.Structure):
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

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    _kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    _kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL


def _memory_snapshot() -> dict[str, float | int | None]:
    if os.name != "nt":
        return {
            "working_set_mb": None,
            "peak_working_set_mb": None,
            "private_mb": None,
            "pagefile_usage_mb": None,
            "available_ram_mb": None,
            "system_memory_load_percent": None,
        }

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    ok = _psapi.GetProcessMemoryInfo(
        _kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())

    return {
        "working_set_mb": _mb(counters.WorkingSetSize),
        "peak_working_set_mb": _mb(counters.PeakWorkingSetSize),
        "private_mb": _mb(counters.PrivateUsage),
        "pagefile_usage_mb": _mb(counters.PagefileUsage),
        "available_ram_mb": _mb(status.ullAvailPhys),
        "system_memory_load_percent": int(status.dwMemoryLoad),
    }


def _interesting_modules() -> list[str]:
    prefixes = (
        "numpy",
        "llama_cpp",
        "onnxruntime",
        "torch",
        "torchaudio",
        "transformers",
        "torchtune",
        "neucodec",
        "perth",
        "librosa",
        "neutts",
    )
    roots = {name.split(".", 1)[0] for name in sys.modules if name.startswith(prefixes)}
    return sorted(roots)


def _format(value: Any, width: int = 9) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.1f}".rjust(width)
    return str(value).rjust(width)


rows: list[dict[str, Any]] = []
_previous: dict[str, Any] | None = None


def record(stage: str, elapsed_seconds: float = 0.0, *, note: str = "") -> None:
    global _previous
    gc.collect()
    # Give Windows a brief chance to update process accounting after large DLL
    # imports/allocations without deliberately trimming the working set.
    time.sleep(0.10)
    current = _memory_snapshot()
    row: dict[str, Any] = {
        "stage": stage,
        "elapsed_seconds": round(elapsed_seconds, 3),
        **current,
        "loaded_roots": _interesting_modules(),
        "module_count": len(sys.modules),
        "note": note,
    }
    if _previous is None:
        row["delta_working_set_mb"] = 0.0
        row["delta_private_mb"] = 0.0
        row["delta_available_ram_mb"] = 0.0
    else:
        for source, target in (
            ("working_set_mb", "delta_working_set_mb"),
            ("private_mb", "delta_private_mb"),
            ("available_ram_mb", "delta_available_ram_mb"),
        ):
            now = row.get(source)
            before = _previous.get(source)
            row[target] = (
                None if now is None or before is None else round(float(now) - float(before), 1)
            )
    rows.append(row)
    _previous = row

    print(
        f"{stage:<28} "
        f"WS={_format(row['working_set_mb'])} MB "
        f"dWS={_format(row['delta_working_set_mb'])}  "
        f"Private={_format(row['private_mb'])} MB "
        f"dPriv={_format(row['delta_private_mb'])}  "
        f"Avail={_format(row['available_ram_mb'])} MB "
        f"time={row['elapsed_seconds']:>7.3f}s"
        + (f"  {note}" if note else ""),
        flush=True,
    )


def run_stage(name: str, action: Callable[[], Any], *, note: str = "") -> Any:
    started = time.perf_counter()
    try:
        result = action()
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        record(f"{name} [FAILED]", elapsed, note=f"{type(exc).__name__}: {exc}")
        raise
    record(name, time.perf_counter() - started, note=note)
    return result


def import_stage(module_name: str, label: str | None = None) -> Any:
    return run_stage(label or f"import {module_name}", lambda: importlib.import_module(module_name))


def _consume_stream(tts: Any) -> int:
    chunks = 0
    for wav in tts.infer_stream(
        "Dexter memory probe. The sky is blue and the system is running normally.",
        speaker="paul",
        emotion="neutral",
        temperature=1.0,
        top_k=50,
    ):
        if wav is not None:
            chunks += 1
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="stop after package-import attribution; do not construct/load NeuTTS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path",
    )
    args = parser.parse_args()

    if os.name != "nt":
        raise SystemExit("This memory probe currently targets Dexter's Windows runtime")

    print("Dexter TTS staged memory probe")
    print(f"PID: {os.getpid()}  Python: {sys.version.split()[0]}  Platform: {platform.platform()}")
    print("Quit Dexter before running this probe so its normal TTS worker is not also resident.\n")

    record("python baseline")
    import_stage("numpy", "import numpy")
    import_stage("llama_cpp", "import llama_cpp")
    import_stage("onnxruntime", "import onnxruntime")
    import_stage("torch", "import torch")
    import_stage("torchaudio", "import torchaudio")
    import_stage("transformers", "import transformers")

    librosa_module = import_stage("librosa", "import librosa shim")
    librosa_path = Path(getattr(librosa_module, "__file__", "")).resolve()
    expected_librosa = (VOICE_DIR / "librosa.py").resolve()
    if librosa_path != expected_librosa:
        raise RuntimeError(
            f"Probe imported unexpected librosa at {librosa_path}; expected {expected_librosa}"
        )

    import_stage("perth", "import perth")
    import_stage("neucodec", "import neucodec")
    import_stage("neutts.neutts", "import neutts core")

    tts = None
    if not args.imports_only:
        def install_pipeline() -> None:
            from Runtime.Voice.neutts2e_pipelined_stream import install

            install()

        run_stage("install Dexter pipeline", install_pipeline)

        def construct_tts() -> Any:
            from neutts import NeuTTS2E

            return NeuTTS2E(
                backbone_repo=MODEL_REPO,
                backbone_device="gpu",
                codec_repo=CODEC_REPO,
                codec_device="cpu",
                seed=42,
            )

        tts = run_stage("construct Q4 NeuTTS", construct_tts)
        run_stage("NeuTTS warmup", tts.warmup)
        chunks = run_stage("one synthesis", lambda: _consume_stream(tts))
        rows[-1]["synthesis_chunks"] = chunks

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or (REPO_ROOT / "Diagnostics" / f"tts-memory-probe_{stamp}.json")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_repo": None if args.imports_only else MODEL_REPO,
        "codec_repo": None if args.imports_only else CODEC_REPO,
        "environment": {
            "GGML_VK_VISIBLE_DEVICES": os.environ.get("GGML_VK_VISIBLE_DEVICES"),
            "ORT_INTRA_OP_NUM_THREADS": os.environ.get("ORT_INTRA_OP_NUM_THREADS"),
            "ORT_INTER_OP_NUM_THREADS": os.environ.get("ORT_INTER_OP_NUM_THREADS"),
            "DEXTER_TTS_FIRST_STREAM_FRAMES": os.environ.get("DEXTER_TTS_FIRST_STREAM_FRAMES"),
            "DEXTER_TTS_STEADY_STREAM_FRAMES": os.environ.get("DEXTER_TTS_STEADY_STREAM_FRAMES"),
        },
        "rows": rows,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    if rows:
        baseline = rows[0]
        final = rows[-1]
        if baseline.get("private_mb") is not None and final.get("private_mb") is not None:
            print(
                "Total private-memory growth: "
                f"{float(final['private_mb']) - float(baseline['private_mb']):.1f} MB"
            )
        if baseline.get("working_set_mb") is not None and final.get("working_set_mb") is not None:
            print(
                "Total working-set growth: "
                f"{float(final['working_set_mb']) - float(baseline['working_set_mb']):.1f} MB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

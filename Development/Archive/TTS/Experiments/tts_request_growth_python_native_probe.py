"""Separate Python-heap growth from native TTS request-lifetime growth.

The production worker was observed to gain ~30-40 MB working set/private memory
per identical deterministic request. This diagnostic mirrors the production
Q4/Arc + lean NeuCodec + encoder-only Perth + 128/128 + 40/50 path in one heavy
Python process, starts tracemalloc only after model warmup, then synthesizes the
same utterance six times.

After every request it records Windows WS/private memory, Python traced heap,
thread count, and the effect of an explicit gc.collect(). The GC is diagnostic
only; production is not changed by this file.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_request_growth_python_native_probe.py
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
for value in (str(REPO_ROOT), str(VOICE_DIR)):
    if value in sys.path:
        sys.path.remove(value)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VOICE_DIR))

MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
TEST_TEXT = (
    "Dexter is repeating one deterministic speech request so we can separate "
    "Python object retention from native runtime memory growth."
)


def _configure() -> None:
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
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")
    os.environ.setdefault("DEXTER_TTS_FIRST_STREAM_FRAMES", "40")
    os.environ.setdefault("DEXTER_TTS_STEADY_STREAM_FRAMES", "50")


def _mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _memory() -> tuple[float, float]:
    if os.name != "nt":
        raise RuntimeError("This diagnostic currently targets Windows")
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
        wintypes.HANDLE, ctypes.POINTER(PMCEX), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = PMCEX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return _mb(counters.WorkingSetSize), _mb(counters.PrivateUsage)


def _install_production_overrides() -> None:
    from Runtime.Voice.neucodec_onnx_compat import install_as_neucodec

    install_as_neucodec()

    # Install the same encoder-only Perth class used by the production launcher.
    import perth
    from Runtime.Voice.perth_encoder_only import PerthEncoderOnlyWatermarker

    perth.PerthImplicitWatermarker = PerthEncoderOnlyWatermarker

    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args: Any, **kwargs: Any):
        kwargs["n_batch"] = 128
        kwargs["n_ubatch"] = 128
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init


def _synthesize(tts: Any) -> tuple[int, int, str, float]:
    digest = hashlib.sha256()
    samples = 0
    chunks = 0
    started = time.perf_counter()
    for wav in tts.infer_stream(
        TEST_TEXT,
        speaker="paul",
        emotion="neutral",
        temperature=1.0,
        top_k=50,
    ):
        array = np.asarray(wav, dtype=np.float32).reshape(-1)
        digest.update(array.tobytes())
        samples += int(array.size)
        chunks += 1
    return samples, chunks, digest.hexdigest(), time.perf_counter() - started


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This diagnostic currently targets Windows")

    _configure()
    _install_production_overrides()

    from Runtime.Voice.neutts2e_pipelined_stream import install

    install()
    from neutts import NeuTTS2E

    print("Dexter TTS request-growth Python-vs-native probe")
    print(
        "Quit Dexter before running this. Tracemalloc starts only after the production "
        "model has loaded and warmed, so model/import allocations do not dominate the trace.\n"
    )

    load_started = time.perf_counter()
    tts = NeuTTS2E(
        backbone_repo=MODEL_REPO,
        backbone_device="gpu",
        codec_repo=CODEC_REPO,
        codec_device="cpu",
        seed=42,
    )
    tts.warmup()
    print(f"Startup + warmup: {time.perf_counter() - load_started:.2f} s")

    gc.collect()
    time.sleep(0.3)
    base_ws, base_private = _memory()
    base_threads = threading.active_count()

    tracemalloc.start(25)
    base_snapshot = tracemalloc.take_snapshot()

    print()
    print(
        "Stage              WS MB   Private MB  PyHeap MB  PyPeak MB  threads  GC objs  "
        "dWS total  dPriv total  hash"
    )
    print(
        "-----------------  ------  ----------  ---------  ---------  -------  -------  "
        "---------  -----------  ------------"
    )
    print(
        f"ready / warmed     {base_ws:6.1f}  {base_private:10.1f}  {0.0:9.1f}  {0.0:9.1f}  "
        f"{base_threads:7d}  {'-':>7}  {0.0:+9.1f}  {0.0:+11.1f}  {'-':>12}"
    )

    reference_hash: str | None = None
    previous_ws = base_ws
    previous_private = base_private

    for index in range(1, 7):
        samples, chunks, digest, elapsed = _synthesize(tts)
        if reference_hash is None:
            reference_hash = digest
        elif digest != reference_hash:
            raise RuntimeError("Deterministic PCM changed between identical requests")

        time.sleep(0.2)
        pre_gc_ws, pre_gc_private = _memory()
        current_py, peak_py = tracemalloc.get_traced_memory()
        threads = threading.active_count()
        collected = gc.collect()
        time.sleep(0.2)
        ws, private = _memory()
        current_py_after, peak_py_after = tracemalloc.get_traced_memory()

        print(
            f"after request {index:<2}   {ws:6.1f}  {private:10.1f}  "
            f"{_mb(current_py_after):9.1f}  {_mb(peak_py_after):9.1f}  {threads:7d}  "
            f"{collected:7d}  {ws - base_ws:+9.1f}  {private - base_private:+11.1f}  "
            f"{digest[:12]}"
        )
        print(
            f"  request {index}: samples={samples}, chunks={chunks}, elapsed={elapsed:.3f}s; "
            f"pre-GC -> post-GC dWS={ws - pre_gc_ws:+.1f} MB, "
            f"dPrivate={private - pre_gc_private:+.1f} MB, "
            f"Python heap dCurrent={_mb(current_py_after - current_py):+.2f} MB"
        )
        previous_ws, previous_private = ws, private

    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(base_snapshot, "lineno")
    print("\nLargest net Python allocations since warmup (tracemalloc):")
    for stat in stats[:12]:
        print(f"  {stat}")

    current_py, peak_py = tracemalloc.get_traced_memory()
    final_ws, final_private = _memory()
    print("\nSummary")
    print(f"  OS growth: dWS={final_ws - base_ws:+.1f} MB, dPrivate={final_private - base_private:+.1f} MB")
    print(f"  Python traced heap now={_mb(current_py):.1f} MB, peak={_mb(peak_py):.1f} MB")
    print(f"  Active threads: baseline={base_threads}, final={threading.active_count()}")
    print(
        "Interpretation: large OS growth with a nearly flat traced Python heap and stable "
        "thread count points to native runtime/allocator retention. Comparable Python-heap "
        "growth should identify concrete Python lines above. GC reducing the OS footprint "
        "would instead implicate unreachable Python object graphs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

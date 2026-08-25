"""Measure the complete current production NeuTTS worker on Windows.

Unlike the component-level A/B probes, this diagnostic launches Dexter's actual
``NeuTTS2EBackend``. That means it uses the same worker launcher, runtime
environment, Q4 Arc/Vulkan backbone, 128/128 llama.cpp batch, lean NeuCodec,
production Perth path, ORT threading and streaming settings as the Host.

It reports the TTS child process working set/private commit after startup+warmup
and again after one complete synthesis. This is intended as a cumulative
checkpoint after individual memory optimizations have been validated.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_production_worker_memory.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_TEXT = (
    "Dexter is measuring the complete production speech worker after the recent "
    "memory optimizations, using the same runtime path as normal voice output."
)


def _mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


def _process_memory(pid: int) -> dict[str, float]:
    if os.name != "nt":
        raise RuntimeError("This diagnostic currently targets Dexter's Windows runtime")

    from ctypes import wintypes

    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

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

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_set_mb": _mb(counters.WorkingSetSize),
            "peak_working_set_mb": _mb(counters.PeakWorkingSetSize),
            "private_mb": _mb(counters.PrivateUsage),
            "peak_private_mb": _mb(counters.PeakPagefileUsage),
        }
    finally:
        kernel32.CloseHandle(handle)


def _print_snapshot(label: str, values: dict[str, float]) -> None:
    print(
        f"{label:<20} "
        f"WS={values['working_set_mb']:>8.1f} MB  "
        f"Private={values['private_mb']:>8.1f} MB  "
        f"PeakWS={values['peak_working_set_mb']:>8.1f} MB  "
        f"PeakPrivate={values['peak_private_mb']:>8.1f} MB"
    )


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This diagnostic currently targets Windows")

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    print("Dexter complete production TTS worker memory checkpoint")
    print(
        "Quit Dexter before running this. The diagnostic starts the same persistent "
        "NeuTTS child used by normal Voice output, waits through warmup, measures it, "
        "runs one utterance, then measures it again.\n"
    )

    backend = NeuTTS2EBackend()
    try:
        started = time.perf_counter()
        backend.start()
        startup_seconds = time.perf_counter() - started

        process = backend._process  # diagnostic access to the backend-owned child
        if process is None or process.poll() is not None:
            raise RuntimeError("Production TTS worker is not alive after backend.start()")
        pid = int(process.pid)

        # Give Windows process accounting a brief moment after the ready message.
        time.sleep(0.25)
        ready_memory = _process_memory(pid)

        cancel_event = threading.Event()
        samples = 0
        synthesis_started = time.perf_counter()
        for chunk in backend.synthesize(TEST_TEXT, cancel_event):
            samples += int(chunk.samples.size)
        synthesis_seconds = time.perf_counter() - synthesis_started
        time.sleep(0.25)
        post_memory = _process_memory(pid)

        ready = backend.ready_info or {}
        stats = backend.last_synthesis_stats or {}
        audio_seconds = samples / 24_000.0
        rtf = synthesis_seconds / audio_seconds if audio_seconds else 0.0

        print(f"Worker PID: {pid}")
        print(f"Startup + warmup: {startup_seconds:.2f} s")
        print(
            "Production settings: "
            f"variant={ready.get('variant')} "
            f"backbone={ready.get('backbone_device')} "
            f"batch={ready.get('backbone_batch')}/{ready.get('backbone_ubatch')} "
            f"ORT={ready.get('codec_ort_intra_threads')}/{ready.get('codec_ort_inter_threads')} "
            f"stream={ready.get('first_stream_frames')}/{ready.get('steady_stream_frames')}"
        )
        print()
        _print_snapshot("ready / warmed", ready_memory)
        _print_snapshot("after synthesis", post_memory)
        print()
        print(
            f"Synthesis: {synthesis_seconds:.3f} s for {audio_seconds:.3f} s audio "
            f"(RTF={rtf:.3f})"
        )
        if stats:
            print(
                "Worker timing: "
                f"first_audio={stats.get('first_audio_seconds')} s, "
                f"generation={stats.get('generation_seconds')} s, "
                f"chunks={stats.get('chunks')}"
            )
        print(
            "Post-synthesis growth: "
            f"dWS={post_memory['working_set_mb'] - ready_memory['working_set_mb']:+.1f} MB, "
            f"dPrivate={post_memory['private_mb'] - ready_memory['private_mb']:+.1f} MB"
        )
        return 0
    finally:
        backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())

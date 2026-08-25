"""Measure the complete production TTS process tree on Windows.

The earlier whole-worker checkpoint measured only ``Popen.pid``. On Windows,
Dexter's uv/venv Python executable can be a small redirector/launcher whose
heavy interpreter/model process is a descendant. This diagnostic enumerates the
entire descendant tree rooted at the backend-owned PID, reports each process,
and sums working set/private commit both after startup+warmup and after one full
synthesis.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_production_worker_tree_memory.py
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
    "Dexter is measuring the complete production speech process tree after the recent "
    "memory optimizations, using the same runtime path as normal voice output."
)


def _mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


def _windows_process_table() -> dict[int, dict[str, object]]:
    if os.name != "nt":
        raise RuntimeError("This diagnostic currently targets Dexter's Windows runtime")

    from ctypes import wintypes

    TH32CS_SNAPPROCESS = 0x00000002
    MAX_PATH = 260
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * MAX_PATH),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())

    table: dict[int, dict[str, object]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise ctypes.WinError(ctypes.get_last_error())
        while True:
            pid = int(entry.th32ProcessID)
            table[pid] = {
                "pid": pid,
                "ppid": int(entry.th32ParentProcessID),
                "exe": str(entry.szExeFile),
            }
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return table


def _descendant_pids(root_pid: int, table: dict[int, dict[str, object]]) -> list[int]:
    children: dict[int, list[int]] = {}
    for pid, row in table.items():
        children.setdefault(int(row["ppid"]), []).append(pid)

    ordered: list[int] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        if pid in table:
            ordered.append(pid)
        stack.extend(children.get(pid, ()))
    return ordered


def _process_memory(pid: int) -> dict[str, float]:
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
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

    rights = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    handle = kernel32.OpenProcess(rights, False, pid)
    if not handle:
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_set_mb": _mb(counters.WorkingSetSize),
            "private_mb": _mb(counters.PrivateUsage),
            "peak_working_set_mb": _mb(counters.PeakWorkingSetSize),
            "peak_private_mb": _mb(counters.PeakPagefileUsage),
        }
    finally:
        kernel32.CloseHandle(handle)


def _tree_snapshot(root_pid: int) -> dict[str, object]:
    table = _windows_process_table()
    pids = _descendant_pids(root_pid, table)
    rows: list[dict[str, object]] = []
    total_ws = 0.0
    total_private = 0.0

    for pid in pids:
        row = table.get(pid, {"pid": pid, "ppid": None, "exe": "?"})
        try:
            memory = _process_memory(pid)
        except OSError as exc:
            rows.append({**row, "error": str(exc)})
            continue
        merged = {**row, **memory}
        rows.append(merged)
        total_ws += float(memory["working_set_mb"])
        total_private += float(memory["private_mb"])

    return {
        "root_pid": root_pid,
        "rows": rows,
        "working_set_mb": round(total_ws, 1),
        "private_mb": round(total_private, 1),
    }


def _print_tree(label: str, snap: dict[str, object]) -> None:
    print(f"\n{label}")
    print("  PID      PPID     EXE                         WS MB   Private MB")
    print("  -------  -------  --------------------------  ------  ----------")
    for row in snap["rows"]:
        if "error" in row:
            print(
                f"  {int(row['pid']):<7}  {str(row.get('ppid')):<7}  "
                f"{str(row.get('exe', '?')):<26}  ERROR: {row['error']}"
            )
            continue
        print(
            f"  {int(row['pid']):<7}  {int(row['ppid']):<7}  {str(row['exe']):<26}  "
            f"{float(row['working_set_mb']):>6.1f}  {float(row['private_mb']):>10.1f}"
        )
    print(
        f"  TREE TOTAL                                      "
        f"{float(snap['working_set_mb']):>6.1f}  {float(snap['private_mb']):>10.1f}"
    )


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This diagnostic currently targets Windows")

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    print("Dexter production TTS process-tree memory checkpoint")
    print(
        "Quit Dexter before running this. This version measures the complete process tree "
        "under the backend-owned PID, not just the launcher PID.\n"
    )

    backend = NeuTTS2EBackend()
    try:
        startup_started = time.perf_counter()
        backend.start()
        startup_seconds = time.perf_counter() - startup_started

        process = backend._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Production TTS worker is not alive after backend.start()")
        root_pid = int(process.pid)

        time.sleep(0.5)
        ready = _tree_snapshot(root_pid)

        cancel_event = threading.Event()
        samples = 0
        synthesis_started = time.perf_counter()
        for chunk in backend.synthesize(TEST_TEXT, cancel_event):
            samples += int(chunk.samples.size)
        synthesis_seconds = time.perf_counter() - synthesis_started

        time.sleep(0.5)
        after = _tree_snapshot(root_pid)

        info = backend.ready_info or {}
        stats = backend.last_synthesis_stats or {}
        audio_seconds = samples / 24_000.0
        rtf = synthesis_seconds / audio_seconds if audio_seconds else 0.0

        print(f"Backend root PID: {root_pid}")
        print(f"Startup + warmup: {startup_seconds:.2f} s")
        print(
            "Production settings: "
            f"variant={info.get('variant')} backbone={info.get('backbone_device')} "
            f"batch={info.get('backbone_batch')}/{info.get('backbone_ubatch')} "
            f"ORT={info.get('codec_ort_intra_threads')}/{info.get('codec_ort_inter_threads')} "
            f"stream={info.get('first_stream_frames')}/{info.get('steady_stream_frames')}"
        )

        _print_tree("ready / warmed", ready)
        _print_tree("after synthesis", after)

        print(
            f"\nSynthesis: {synthesis_seconds:.3f} s for {audio_seconds:.3f} s audio "
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
            "Process-tree post-synthesis growth: "
            f"dWS={float(after['working_set_mb']) - float(ready['working_set_mb']):+.1f} MB, "
            f"dPrivate={float(after['private_mb']) - float(ready['private_mb']):+.1f} MB"
        )
        return 0
    finally:
        backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())

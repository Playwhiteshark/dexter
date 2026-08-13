from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

import psutil


if os.name == "nt":
    user32 = ctypes.windll.user32

    # Prefer physical pixel coordinates so window bounds line up with desktop
    # automation coordinates on scaled displays. Failure is harmless.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
else:
    user32 = None


SW_RESTORE = 9
WM_CLOSE = 0x0010


def _require_windows() -> None:
    if os.name != "nt" or user32 is None:
        raise RuntimeError("Window tools are currently implemented for Windows only.")


def _process_name(pid: int) -> str | None:
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _window_bounds(hwnd: int) -> dict:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError(f"Could not read bounds for window handle {hwnd}.")

    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "right": int(rect.right),
        "bottom": int(rect.bottom),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def _window_record(hwnd: int) -> dict:
    if not user32.IsWindow(hwnd):
        raise ValueError(f"Window handle no longer exists: {hwnd}")

    pid = _window_pid(hwnd)
    return {
        "hwnd": int(hwnd),
        "title": _window_title(hwnd),
        "pid": pid,
        "process_name": _process_name(pid),
        "visible": bool(user32.IsWindowVisible(hwnd)),
        "minimized": bool(user32.IsIconic(hwnd)),
        "bounds": _window_bounds(hwnd),
    }


def _window_handles_for_pid(pid: int) -> list[int]:
    handles: list[int] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if _window_pid(hwnd) == pid:
            handles.append(int(hwnd))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return handles


def get_foreground_window_info() -> dict:
    _require_windows()
    hwnd = int(user32.GetForegroundWindow())
    if hwnd == 0:
        return {
            "status": "ok",
            "operation": "get_foreground_window",
            "window": None,
        }

    return {
        "status": "ok",
        "operation": "get_foreground_window",
        "window": _window_record(hwnd),
    }


def list_visible_windows(
    query: str | None = None,
    max_results: int = 50,
    visible_only: bool = True,
) -> dict:
    _require_windows()

    if max_results < 1 or max_results > 200:
        raise ValueError("max_results must be between 1 and 200.")

    query_lower = query.lower().strip() if query else None
    records: list[dict] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if len(records) >= max_results:
            return False

        if visible_only and not user32.IsWindowVisible(hwnd):
            return True

        title = _window_title(hwnd)
        pid = _window_pid(hwnd)
        process_name = _process_name(pid)

        if query_lower:
            haystack = f"{title} {process_name or ''}".lower()
            if query_lower not in haystack:
                return True

        # Skip anonymous/zero-size windows unless the caller explicitly asks
        # for invisible windows.
        try:
            bounds = _window_bounds(hwnd)
        except RuntimeError:
            return True

        if visible_only and not title.strip():
            return True

        records.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "pid": pid,
                "process_name": process_name,
                "visible": bool(user32.IsWindowVisible(hwnd)),
                "minimized": bool(user32.IsIconic(hwnd)),
                "bounds": bounds,
            }
        )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)

    return {
        "status": "ok",
        "operation": "list_windows",
        "query": query,
        "windows": records,
        "count": len(records),
        "truncated": len(records) >= max_results,
    }


def get_window_bounds_info(hwnd: int) -> dict:
    _require_windows()
    if hwnd <= 0 or not user32.IsWindow(hwnd):
        raise ValueError(f"Invalid or stale window handle: {hwnd}")

    return {
        "status": "ok",
        "operation": "get_window_bounds",
        "hwnd": hwnd,
        "bounds": _window_bounds(hwnd),
    }


def focus_window_verified(hwnd: int) -> dict:
    _require_windows()
    if hwnd <= 0 or not user32.IsWindow(hwnd):
        raise ValueError(f"Invalid or stale window handle: {hwnd}")

    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    requested = bool(user32.SetForegroundWindow(hwnd))
    time.sleep(0.10)
    observed = int(user32.GetForegroundWindow())
    verified = observed == hwnd

    if not verified:
        raise RuntimeError(
            "Windows refused to make the requested window foreground. "
            "This can happen because Windows restricts focus stealing. "
            f"Requested hwnd={hwnd}, foreground hwnd={observed}."
        )

    return {
        "status": "ok",
        "operation": "focus_window",
        "hwnd": hwnd,
        "set_foreground_returned": requested,
        "foreground_hwnd": observed,
        "verified": True,
    }


def close_window_gracefully(hwnd: int, timeout_seconds: float = 5.0) -> dict:
    _require_windows()
    if hwnd <= 0 or not user32.IsWindow(hwnd):
        raise ValueError(f"Invalid or stale window handle: {hwnd}")
    if timeout_seconds < 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be between 0 and 60.")

    title = _window_title(hwnd)
    pid = _window_pid(hwnd)

    if not user32.PostMessageW(hwnd, WM_CLOSE, 0, 0):
        raise RuntimeError(f"Failed to send WM_CLOSE to window {hwnd}.")

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not user32.IsWindow(hwnd):
            break
        time.sleep(0.10)

    closed = not bool(user32.IsWindow(hwnd))

    return {
        "status": "ok",
        "operation": "close_window",
        "hwnd": hwnd,
        "title": title,
        "pid": pid,
        "closed": closed,
        "verified": closed,
        "note": (
            None
            if closed
            else "Window is still open; the app may be showing a save/confirmation prompt."
        ),
    }


def close_app_gracefully(pid: int, timeout_seconds: float = 5.0) -> dict:
    _require_windows()
    if pid <= 0:
        raise ValueError("pid must be a positive integer.")
    if not psutil.pid_exists(pid):
        raise ProcessLookupError(f"Process does not exist: {pid}")
    if timeout_seconds < 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be between 0 and 60.")

    targets = _window_handles_for_pid(pid)

    if not targets:
        raise RuntimeError(
            "No top-level windows were found for this process. "
            "Dexter refuses to hard-kill it through close_app."
        )

    requested_hwnds: list[int] = []
    for hwnd in targets:
        if user32.IsWindow(hwnd):
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            requested_hwnds.append(hwnd)

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and psutil.pid_exists(pid):
        time.sleep(0.10)

    process_exited = not psutil.pid_exists(pid)
    remaining_windows = [
        hwnd for hwnd in requested_hwnds
        if user32.IsWindow(hwnd)
    ]

    return {
        "status": "ok",
        "operation": "close_app",
        "pid": pid,
        "requested_windows": requested_hwnds,
        "process_exited": process_exited,
        "remaining_windows": remaining_windows,
        "verified": process_exited or not remaining_windows,
        "note": (
            None
            if process_exited or not remaining_windows
            else "App did not close; it may be showing a save/confirmation prompt."
        ),
    }

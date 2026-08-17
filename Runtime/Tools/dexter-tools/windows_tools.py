from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

import psutil


if os.name == "nt":
    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi

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
    dwmapi = None


SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_MINIMIZE = 6
SW_RESTORE = 9
WM_CLOSE = 0x0010
DWMWA_CLOAKED = 14
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]


def _require_windows() -> None:
    if os.name != "nt" or user32 is None:
        raise RuntimeError("Window tools are currently implemented for Windows only.")


def _process_name(pid: int) -> str | None:
    try:
        return psutil.Process(pid).name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _process_exe(pid: int) -> str | None:
    try:
        return psutil.Process(pid).exe() or None
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
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


def _is_window_cloaked(hwnd: int) -> bool:
    if dwmapi is None:
        return False
    cloaked = wintypes.DWORD()
    result = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        DWMWA_CLOAKED,
        ctypes.byref(cloaked),
        ctypes.sizeof(cloaked),
    )
    return result == 0 and bool(cloaked.value)


def _window_placement(hwnd: int) -> dict:
    placement = WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(WINDOWPLACEMENT)
    if not user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        raise RuntimeError(f"Could not read window placement for handle {hwnd}.")

    rect = placement.rcNormalPosition
    normal_bounds = {
        "left": int(rect.left),
        "top": int(rect.top),
        "right": int(rect.right),
        "bottom": int(rect.bottom),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }
    show_cmd = int(placement.showCmd)
    if show_cmd == SW_SHOWMINIMIZED:
        state = "minimized"
    elif show_cmd == SW_SHOWMAXIMIZED:
        state = "maximized"
    else:
        state = "normal"

    return {
        "show_state": state,
        "show_cmd": show_cmd,
        "flags": int(placement.flags),
        "min_position": {
            "x": int(placement.ptMinPosition.x),
            "y": int(placement.ptMinPosition.y),
        },
        "max_position": {
            "x": int(placement.ptMaxPosition.x),
            "y": int(placement.ptMaxPosition.y),
        },
        "normal_bounds": normal_bounds,
    }


def _saved_windowplacement_struct(placement: dict, show_cmd: int) -> WINDOWPLACEMENT:
    normal = placement.get("normal_bounds")
    if not isinstance(normal, dict):
        raise ValueError("placement is missing normal restore bounds.")
    try:
        left = int(normal["left"])
        top = int(normal["top"])
        right = int(normal["right"])
        bottom = int(normal["bottom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("placement contains invalid normal restore bounds.") from exc
    if right <= left or bottom <= top:
        raise ValueError("placement normal restore bounds must have positive size.")

    result = WINDOWPLACEMENT()
    result.length = ctypes.sizeof(WINDOWPLACEMENT)
    result.flags = int(placement.get("flags") or 0)
    result.showCmd = show_cmd

    min_position = placement.get("min_position") or {}
    max_position = placement.get("max_position") or {}
    result.ptMinPosition.x = int(min_position.get("x", 0))
    result.ptMinPosition.y = int(min_position.get("y", 0))
    result.ptMaxPosition.x = int(max_position.get("x", 0))
    result.ptMaxPosition.y = int(max_position.get("y", 0))
    result.rcNormalPosition.left = left
    result.rcNormalPosition.top = top
    result.rcNormalPosition.right = right
    result.rcNormalPosition.bottom = bottom
    return result


def restore_window_placement(hwnd: int, placement: dict) -> dict:
    """Restore deterministic size/position/show state for one current window."""
    _require_windows()
    if hwnd <= 0 or not user32.IsWindow(hwnd):
        raise ValueError(f"Invalid or stale window handle: {hwnd}")
    if not isinstance(placement, dict):
        raise ValueError("placement must be an object.")

    state = str(placement.get("show_state") or "normal")
    if state not in {"normal", "minimized", "maximized"}:
        raise ValueError(f"Unsupported saved window show state: {state!r}")

    current = placement.get("current_bounds")

    # GetWindowPlacement's rcNormalPosition is expressed in workspace
    # coordinates for ordinary top-level windows. Microsoft explicitly warns
    # not to feed those coordinates to SetWindowPos, which expects screen
    # coordinates. For minimized/maximized snapshots we therefore restore via
    # SetWindowPlacement. For a currently normal snapshot we saved GetWindowRect
    # screen coordinates too, so SetWindowPos can reproduce the exact aligned
    # geometry without coordinate conversion.
    if state == "normal" and isinstance(current, dict):
        try:
            left = int(current["left"])
            top = int(current["top"])
            width = int(current["width"])
            height = int(current["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("placement contains invalid current bounds.") from exc
        if width <= 0 or height <= 0:
            raise ValueError("placement current bounds must have positive size.")

        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, SW_RESTORE)
        ok = user32.SetWindowPos(
            hwnd,
            0,
            left,
            top,
            width,
            height,
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
        if not ok:
            raise RuntimeError(f"Windows refused to move/resize window {hwnd}.")
        user32.ShowWindow(hwnd, SW_SHOWNORMAL)
        verification_bounds = current
    else:
        show_cmd = {
            "normal": SW_SHOWNORMAL,
            "minimized": SW_SHOWMINIMIZED,
            "maximized": SW_SHOWMAXIMIZED,
        }[state]
        saved = _saved_windowplacement_struct(placement, show_cmd)
        if not user32.SetWindowPlacement(hwnd, ctypes.byref(saved)):
            raise RuntimeError(f"Windows refused to restore placement for window {hwnd}.")
        verification_bounds = None

    time.sleep(0.08)
    observed_placement = _window_placement(hwnd)
    observed_bounds = _window_bounds(hwnd)

    if state == "normal" and verification_bounds is not None:
        tolerance = 3
        mismatches = [
            abs(observed_bounds[key] - verification_bounds[key]) > tolerance
            for key in ("left", "top", "width", "height")
        ]
        if any(mismatches):
            raise RuntimeError(
                f"Window {hwnd} did not settle at the requested bounds. "
                f"Requested={verification_bounds}, observed={observed_bounds}."
            )
    elif observed_placement["show_state"] != state:
        raise RuntimeError(
            f"Window {hwnd} did not settle in the requested {state} state; "
            f"observed {observed_placement['show_state']}."
        )

    return {
        "status": "ok",
        "operation": "restore_window_placement",
        "hwnd": hwnd,
        "placement": observed_placement,
        "bounds": observed_bounds,
        "verified": True,
    }


def capture_workspace_windows() -> dict:
    """Capture user-facing top-level windows and their restorable placement.

    Covered/overlapping and minimized windows are included. DWM-cloaked windows
    are omitted so windows hidden by the shell (including typical inactive
    virtual-desktop windows) do not leak into the current layout snapshot.
    """
    _require_windows()
    records: list[dict] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if _is_window_cloaked(hwnd):
            return True

        title = _window_title(hwnd).strip()
        if not title:
            return True

        try:
            bounds = _window_bounds(hwnd)
            placement = _window_placement(hwnd)
            placement["current_bounds"] = dict(bounds)
        except RuntimeError:
            return True
        if bounds["width"] <= 0 or bounds["height"] <= 0:
            return True

        pid = _window_pid(hwnd)
        records.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "pid": pid,
                "process_name": _process_name(pid),
                "exe": _process_exe(pid),
                "visible": True,
                "minimized": bool(user32.IsIconic(hwnd)),
                "bounds": bounds,
                "placement": placement,
                "z_index": len(records),
            }
        )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return {
        "status": "ok",
        "operation": "capture_workspace_windows",
        "windows": records,
        "count": len(records),
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
        "exe": _process_exe(pid),
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
                "exe": _process_exe(pid),
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


def set_window_state_verified(hwnd: int, state: str) -> dict:
    """Minimize, maximize, or restore one current top-level window and verify it."""
    _require_windows()
    if hwnd <= 0 or not user32.IsWindow(hwnd):
        raise ValueError(f"Invalid or stale window handle: {hwnd}")

    requested = state.strip().lower()
    if requested not in {"normal", "minimized", "maximized"}:
        raise ValueError("state must be one of: normal, minimized, maximized.")

    command = {
        "normal": SW_RESTORE,
        "minimized": SW_MINIMIZE,
        "maximized": SW_SHOWMAXIMIZED,
    }[requested]

    # ShowWindow's return value reports the previous visibility state, not
    # whether the requested transition succeeded, so verify the resulting
    # state explicitly instead of treating that return value as success.
    user32.ShowWindow(hwnd, command)
    time.sleep(0.08)

    if user32.IsIconic(hwnd):
        observed = "minimized"
    elif user32.IsZoomed(hwnd):
        observed = "maximized"
    else:
        observed = "normal"

    if observed != requested:
        raise RuntimeError(
            f"Window {hwnd} did not settle in the requested {requested} state; "
            f"observed {observed}."
        )

    return {
        "status": "ok",
        "operation": "set_window_state",
        "hwnd": hwnd,
        "state": observed,
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

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

import psutil

from virtual_desktop import VirtualDesktopManager


_WIN_CALLBACK = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)


if os.name == "nt":
    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi

    # Prefer physical pixel coordinates so Win32 and DWM geometry line up on
    # scaled displays. Failure is harmless; the calls below still work in the
    # process' existing DPI-awareness mode.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

    # HDWP is pointer-sized. ctypes otherwise assumes c_int for an untyped
    # foreign function return, which is unsafe on 64-bit Windows.
    try:
        user32.BeginDeferWindowPos.argtypes = [ctypes.c_int]
        user32.BeginDeferWindowPos.restype = wintypes.HANDLE
        user32.DeferWindowPos.argtypes = [
            wintypes.HANDLE,
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.DeferWindowPos.restype = wintypes.HANDLE
        user32.EndDeferWindowPos.argtypes = [wintypes.HANDLE]
        user32.EndDeferWindowPos.restype = wintypes.BOOL
        user32.GetShellWindow.restype = wintypes.HWND
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        user32.GetClassNameW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        user32.GetClassNameW.restype = ctypes.c_int
    except Exception:
        pass

    try:
        _is_window_arranged_fn = user32.IsWindowArranged
        _is_window_arranged_fn.argtypes = [wintypes.HWND]
        _is_window_arranged_fn.restype = wintypes.BOOL
    except (AttributeError, OSError):
        _is_window_arranged_fn = None
else:
    user32 = None
    dwmapi = None
    _is_window_arranged_fn = None


SW_SHOWNORMAL = 1
SW_SHOWMINIMIZED = 2
SW_SHOWMAXIMIZED = 3
SW_MINIMIZE = 6
SW_RESTORE = 9
WM_CLOSE = 0x0010
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_CLOAKED = 14
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
GWL_STYLE = -16
WS_THICKFRAME = 0x00040000
WINDOW_SETTLE_SECONDS = 0.08
GEOMETRY_ATTEMPTS = 3
SHELL_WINDOW_CLASSES = frozenset(
    {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
)


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]


if os.name == "nt":
    try:
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
        user32.GetWindowPlacement.restype = wintypes.BOOL
        user32.SetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
        user32.SetWindowPlacement.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL
        user32.IsZoomed.argtypes = [wintypes.HWND]
        user32.IsZoomed.restype = wintypes.BOOL
        try:
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        except AttributeError:
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
    except Exception:
        pass


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


def _window_class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    copied = user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value if copied else ""


def _rect_dict(rect: wintypes.RECT) -> dict:
    return {
        "left": int(rect.left),
        "top": int(rect.top),
        "right": int(rect.right),
        "bottom": int(rect.bottom),
        "width": int(rect.right - rect.left),
        "height": int(rect.bottom - rect.top),
    }


def _window_bounds(hwnd: int) -> dict:
    """Return the Win32 outer window rectangle (includes invisible resize borders)."""
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError(f"Could not read bounds for window handle {hwnd}.")
    return _rect_dict(rect)


def _visible_window_bounds(hwnd: int) -> dict:
    """Return the DWM visible frame rectangle when available.

    GetWindowRect includes invisible resize borders on modern Windows. Workspace
    geometry is about what the user visually arranged, so DWM extended frame
    bounds are the canonical geometry. Falling back to GetWindowRect keeps the
    primitive usable if DWM refuses the attribute for an unusual window.
    """
    if dwmapi is not None:
        rect = wintypes.RECT()
        result = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return _rect_dict(rect)
    return _window_bounds(hwnd)


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


def _is_window_arranged(hwnd: int) -> bool:
    if _is_window_arranged_fn is None:
        return False
    try:
        return bool(_is_window_arranged_fn(wintypes.HWND(hwnd)))
    except Exception:
        return False


def _window_style(hwnd: int) -> int:
    """Return the top-level window style bits.

    WS_THICKFRAME/WS_SIZEBOX is the documented indication that the window has
    a sizing border. It is a useful readiness signal for applications that show
    a fixed-size splash card before their real editor/browser window appears.
    """
    try:
        fn = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
        return int(fn(wintypes.HWND(hwnd), GWL_STYLE)) & 0xFFFFFFFF
    except Exception:
        return 0


def _window_resizable(hwnd: int) -> bool:
    return bool(_window_style(hwnd) & WS_THICKFRAME)


def _current_show_state(hwnd: int) -> str:
    if user32.IsIconic(hwnd):
        return "minimized"
    if user32.IsZoomed(hwnd):
        return "maximized"
    if _is_window_arranged(hwnd):
        return "arranged"
    return "normal"


def _window_placement(hwnd: int) -> dict:
    placement = WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(WINDOWPLACEMENT)
    if not user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        raise RuntimeError(f"Could not read window placement for handle {hwnd}.")

    show_cmd = int(placement.showCmd)
    state = _current_show_state(hwnd)
    return {
        "show_state": state,
        "show_cmd": show_cmd,
        "arranged": state == "arranged",
        "flags": int(placement.flags),
        "min_position": {
            "x": int(placement.ptMinPosition.x),
            "y": int(placement.ptMinPosition.y),
        },
        "max_position": {
            "x": int(placement.ptMaxPosition.x),
            "y": int(placement.ptMaxPosition.y),
        },
        "normal_bounds": _rect_dict(placement.rcNormalPosition),
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


def _validated_bounds(bounds: dict, label: str) -> dict:
    if not isinstance(bounds, dict):
        raise ValueError(f"{label} must be an object.")
    try:
        left = int(bounds["left"])
        top = int(bounds["top"])
        right = int(bounds["right"])
        bottom = int(bounds["bottom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid bounds.") from exc
    if right <= left or bottom <= top:
        raise ValueError(f"{label} must have positive size.")
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _bounds_equal(first: dict, second: dict) -> bool:
    return all(int(first[key]) == int(second[key]) for key in ("left", "top", "right", "bottom"))


def _outer_bounds_for_visible_target(
    target_visible: dict,
    current_outer: dict,
    current_visible: dict,
) -> dict:
    """Translate a desired visible frame rectangle into SetWindowPos coordinates.

    SetWindowPos positions the Win32 outer rectangle, while the workspace stores
    DWM-visible frame bounds. Measure the current invisible margins and subtract
    them from the visible target. Re-measuring each correction round handles
    apps whose non-client margins change after a move or resize.
    """
    target = _validated_bounds(target_visible, "target visible bounds")
    outer = _validated_bounds(current_outer, "current outer bounds")
    visible = _validated_bounds(current_visible, "current visible bounds")

    left_margin = visible["left"] - outer["left"]
    top_margin = visible["top"] - outer["top"]
    right_margin = outer["right"] - visible["right"]
    bottom_margin = outer["bottom"] - visible["bottom"]

    left = target["left"] - left_margin
    top = target["top"] - top_margin
    right = target["right"] + right_margin
    bottom = target["bottom"] + bottom_margin
    if right <= left or bottom <= top:
        raise RuntimeError("Calculated outer window bounds have non-positive size.")
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _geometry_spec(placement: dict) -> tuple[str, dict] | None:
    state = str(placement.get("show_state") or "normal")
    if state not in {"normal", "arranged"}:
        return None
    visible = placement.get("visible_bounds")
    if isinstance(visible, dict):
        return "visible", _validated_bounds(visible, "placement visible bounds")
    # Backward compatibility for v1 workspace snapshots. Those snapshots used
    # GetWindowRect and therefore describe the outer rectangle, not the DWM
    # visible frame. Re-save the workspace to get pixel-visible v2 geometry.
    current = placement.get("current_bounds")
    if isinstance(current, dict):
        return "outer", _validated_bounds(current, "placement current bounds")
    return None


def _apply_position_targets(targets: list[dict]) -> None:
    """Apply multiple SetWindowPos targets in one deferred positioning transaction."""
    if not targets:
        return

    hdwp = user32.BeginDeferWindowPos(len(targets))
    if hdwp:
        current_hdwp = hdwp
        for item in targets:
            bounds = item["outer_bounds"]
            current_hdwp = user32.DeferWindowPos(
                current_hdwp,
                item["hwnd"],
                0,
                bounds["left"],
                bounds["top"],
                bounds["width"],
                bounds["height"],
                SWP_NOZORDER | SWP_NOACTIVATE,
            )
            if not current_hdwp:
                break
        if current_hdwp and user32.EndDeferWindowPos(current_hdwp):
            return

    # A failed HDWP transaction must not prevent the workspace from being
    # attempted. Fall back to ordinary SetWindowPos per window; workspace-level
    # code still verifies every result and reports any stubborn window exactly.
    for item in targets:
        bounds = item["outer_bounds"]
        ok = user32.SetWindowPos(
            item["hwnd"],
            0,
            bounds["left"],
            bounds["top"],
            bounds["width"],
            bounds["height"],
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
        if not ok:
            raise RuntimeError(f"Windows refused to move/resize window {item['hwnd']}.")


def _restore_geometry_batch(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    pending: list[dict] = []
    failures: list[dict] = []

    for item in entries:
        hwnd = int(item["hwnd"])
        placement = item["placement"]
        try:
            spec = _geometry_spec(placement)
            if spec is None:
                raise ValueError("placement has no restorable normal/arranged geometry.")
            if user32.IsIconic(hwnd) or user32.IsZoomed(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
            pending.append({"hwnd": hwnd, "placement": placement, "space": spec[0], "target": spec[1]})
        except Exception as exc:
            failures.append({"hwnd": hwnd, "error": f"{type(exc).__name__}: {exc}"})

    if pending:
        time.sleep(WINDOW_SETTLE_SECONDS)

    for _attempt in range(GEOMETRY_ATTEMPTS):
        if not pending:
            break
        targets: list[dict] = []
        still_pending: list[dict] = []
        for item in pending:
            hwnd = item["hwnd"]
            try:
                observed = _visible_window_bounds(hwnd) if item["space"] == "visible" else _window_bounds(hwnd)
                if _bounds_equal(observed, item["target"]):
                    continue
                if item["space"] == "visible":
                    outer_target = _outer_bounds_for_visible_target(
                        item["target"],
                        _window_bounds(hwnd),
                        _visible_window_bounds(hwnd),
                    )
                else:
                    outer_target = item["target"]
                targets.append({"hwnd": hwnd, "outer_bounds": outer_target})
                still_pending.append(item)
            except Exception as exc:
                failures.append({"hwnd": hwnd, "error": f"{type(exc).__name__}: {exc}"})
        pending = still_pending
        if not targets:
            break
        try:
            _apply_position_targets(targets)
        except Exception as exc:
            for item in pending:
                failures.append({"hwnd": item["hwnd"], "error": f"{type(exc).__name__}: {exc}"})
            pending = []
            break
        time.sleep(WINDOW_SETTLE_SECONDS)

    restored: list[dict] = []
    failed_hwnds = {item["hwnd"] for item in failures}
    for item in entries:
        hwnd = int(item["hwnd"])
        if hwnd in failed_hwnds:
            continue
        placement = item["placement"]
        spec = _geometry_spec(placement)
        if spec is None:
            continue
        space, requested = spec
        try:
            observed = _visible_window_bounds(hwnd) if space == "visible" else _window_bounds(hwnd)
            if not _bounds_equal(observed, requested):
                failures.append(
                    {
                        "hwnd": hwnd,
                        "error": "window did not settle at the requested bounds",
                        "coordinate_space": space,
                        "requested": requested,
                        "observed": observed,
                    }
                )
                continue
            desired_state = str(placement.get("show_state") or "normal")
            observed_state = _current_show_state(hwnd)
            warning = None
            if desired_state == "arranged" and observed_state != "arranged":
                warning = (
                    "geometry restored exactly, but Windows did not report the window as arranged/snapped"
                )
            restored.append(
                {
                    "hwnd": hwnd,
                    "state": observed_state,
                    "coordinate_space": space,
                    "bounds": observed,
                    "verified": True,
                    "warning": warning,
                }
            )
        except Exception as exc:
            failures.append({"hwnd": hwnd, "error": f"{type(exc).__name__}: {exc}"})

    return restored, failures


def restore_window_placements_batch(items: list[dict]) -> dict:
    """Restore a workspace worth of windows without one failure aborting the rest."""
    _require_windows()
    if not isinstance(items, list):
        raise ValueError("items must be a list.")

    geometry: list[dict] = []
    stateful: list[dict] = []
    failures: list[dict] = []
    for item in items:
        try:
            hwnd = int(item["hwnd"])
            placement = item["placement"]
            if hwnd <= 0 or not user32.IsWindow(hwnd):
                raise ValueError(f"Invalid or stale window handle: {hwnd}")
            if not isinstance(placement, dict):
                raise ValueError("placement must be an object.")
            state = str(placement.get("show_state") or "normal")
            if state in {"normal", "arranged"}:
                geometry.append({"hwnd": hwnd, "placement": placement})
            elif state in {"minimized", "maximized"}:
                stateful.append({"hwnd": hwnd, "placement": placement})
            else:
                raise ValueError(f"Unsupported saved window show state: {state!r}")
        except Exception as exc:
            hwnd = item.get("hwnd") if isinstance(item, dict) else None
            failures.append({"hwnd": hwnd, "error": f"{type(exc).__name__}: {exc}"})

    restored, geometry_failures = _restore_geometry_batch(geometry)
    failures.extend(geometry_failures)

    for item in stateful:
        hwnd = item["hwnd"]
        placement = item["placement"]
        desired_state = str(placement.get("show_state"))
        try:
            show_cmd = SW_SHOWMINIMIZED if desired_state == "minimized" else SW_SHOWMAXIMIZED
            saved = _saved_windowplacement_struct(placement, show_cmd)
            if not user32.SetWindowPlacement(hwnd, ctypes.byref(saved)):
                raise RuntimeError(f"Windows refused to restore placement for window {hwnd}.")
            time.sleep(WINDOW_SETTLE_SECONDS)
            observed_placement = _window_placement(hwnd)
            if observed_placement["show_state"] != desired_state:
                raise RuntimeError(
                    f"window settled in {observed_placement['show_state']} instead of {desired_state}"
                )
            requested_normal = _validated_bounds(placement["normal_bounds"], "placement normal bounds")
            observed_normal = _validated_bounds(observed_placement["normal_bounds"], "observed normal bounds")
            if not _bounds_equal(requested_normal, observed_normal):
                raise RuntimeError(
                    f"normal restore bounds changed: requested={requested_normal}, observed={observed_normal}"
                )
            restored.append(
                {
                    "hwnd": hwnd,
                    "state": observed_placement["show_state"],
                    "normal_bounds": observed_normal,
                    "verified": True,
                    "warning": None,
                }
            )
        except Exception as exc:
            failures.append({"hwnd": hwnd, "error": f"{type(exc).__name__}: {exc}"})

    restored_hwnds = {item["hwnd"] for item in restored}
    failed_hwnds = {item.get("hwnd") for item in failures}
    return {
        "status": "ok",
        "operation": "restore_window_placements_batch",
        "restored": restored,
        "failed": failures,
        "restored_count": len(restored_hwnds),
        "failed_count": len([hwnd for hwnd in failed_hwnds if hwnd is not None]),
        "complete": not failures,
    }


def restore_window_placement(hwnd: int, placement: dict) -> dict:
    """Restore one window through the same verified path used by workspaces."""
    result = restore_window_placements_batch([{"hwnd": hwnd, "placement": placement}])
    if result["failed"]:
        failure = result["failed"][0]
        raise RuntimeError(f"Window {hwnd} restore failed: {failure.get('error')}")
    restored = result["restored"][0]
    return {
        "status": "ok",
        "operation": "restore_window_placement",
        "hwnd": hwnd,
        "placement": _window_placement(hwnd),
        "bounds": _window_bounds(hwnd),
        "visible_bounds": _visible_window_bounds(hwnd),
        "verified": bool(restored.get("verified")),
        "warning": restored.get("warning"),
    }


def capture_workspace_windows() -> dict:
    """Capture current user-facing top-level windows and restorable placement.

    Covered/overlapping and minimized windows are included. DWM-cloaked windows
    are omitted, and the Shell desktop window itself is excluded explicitly.
    Normal/arranged windows store both Win32 outer bounds and DWM-visible frame
    bounds; the latter is the canonical workspace geometry.
    """
    _require_windows()
    records: list[dict] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    shell_hwnd = int(user32.GetShellWindow() or 0)

    def callback(hwnd, _lparam):
        if shell_hwnd and int(hwnd) == shell_hwnd:
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        if _is_window_cloaked(hwnd):
            return True

        title = _window_title(hwnd).strip()
        if not title:
            return True

        try:
            bounds = _window_bounds(hwnd)
            visible_bounds = _visible_window_bounds(hwnd)
            placement = _window_placement(hwnd)
            placement["current_bounds"] = dict(bounds)
            placement["visible_bounds"] = dict(visible_bounds)
            placement["geometry_space"] = "dwm_extended_frame"
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
                "arranged": bool(placement.get("arranged")),
                "resizable": _window_resizable(hwnd),
                "bounds": bounds,
                "visible_bounds": visible_bounds,
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
        "resizable": _window_resizable(hwnd),
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
    time.sleep(WINDOW_SETTLE_SECONDS)
    observed = _current_show_state(hwnd)

    # A snapped/arranged window is still a successful "restore/unminimize"
    # result. Windows treats arranged as a distinct state from ordinary normal,
    # but the model-facing primitive intentionally asks only for restored vs.
    # minimized/maximized.
    acceptable = observed == requested or (requested == "normal" and observed == "arranged")
    if not acceptable:
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


def close_all_apps_on_active_desktop(timeout_seconds: float = 5.0) -> dict:
    """Gracefully close user-facing windows on the active virtual desktop.

    Targets come from a top-level window snapshot, never from the process list.
    Windows shell surfaces, Dexter itself, untitled/invisible windows, and DWM-
    cloaked windows are excluded. The documented virtual-desktop membership
    API provides the definitive active-desktop check; failure aborts before any
    window receives WM_CLOSE.
    """
    _require_windows()
    if timeout_seconds < 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be between 0 and 60.")

    excluded_pids = {os.getpid()}
    host_pid = str(os.environ.get("DEXTER_HOST_PID") or "").strip()
    if host_pid.isdecimal() and int(host_pid) > 0:
        excluded_pids.add(int(host_pid))

    desktop_filter = "ivirtualdesktopmanager"
    desktop_filter_warning = None
    try:
        desktop_manager = VirtualDesktopManager()
    except Exception as exc:
        # Bulk closing must fail closed if active-desktop membership cannot be
        # verified. DWM cloaking remains a useful extra filter but is not used
        # as the sole authorization to touch other applications.
        raise RuntimeError(
            "Could not verify the active Windows virtual desktop; no windows "
            f"were closed ({type(exc).__name__}: {exc})."
        ) from exc

    candidates: list[dict] = []
    skipped_desktop_check: list[int] = []
    shell_hwnd = int(user32.GetShellWindow() or 0)
    EnumWindowsProc = _WIN_CALLBACK(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    def callback(hwnd, _lparam):
        handle = int(hwnd)
        if shell_hwnd and handle == shell_hwnd:
            return True
        if not user32.IsWindowVisible(hwnd) or _is_window_cloaked(handle):
            return True
        if _window_class_name(handle) in SHELL_WINDOW_CLASSES:
            return True

        title = _window_title(handle).strip()
        if not title:
            return True
        pid = _window_pid(handle)
        if pid in excluded_pids:
            return True

        if desktop_manager is not None:
            try:
                if not desktop_manager.is_window_on_current_desktop(handle):
                    return True
            except Exception:
                # A membership failure for one unusual window is a reason to
                # leave that window alone, not to risk another desktop.
                skipped_desktop_check.append(handle)
                return True

        try:
            bounds = _window_bounds(handle)
        except RuntimeError:
            return True
        if bounds["width"] <= 0 or bounds["height"] <= 0:
            return True

        candidates.append(
            {
                "hwnd": handle,
                "title": title,
                "pid": pid,
                "process_name": _process_name(pid),
            }
        )
        return True

    try:
        user32.EnumWindows(EnumWindowsProc(callback), 0)
    finally:
        if desktop_manager is not None:
            desktop_manager.close()

    requested: list[dict] = []
    request_failures: list[dict] = []
    for item in candidates:
        hwnd = item["hwnd"]
        if not user32.IsWindow(hwnd):
            continue
        if user32.PostMessageW(hwnd, WM_CLOSE, 0, 0):
            requested.append(item)
        else:
            request_failures.append({**item, "error": "WM_CLOSE was rejected"})

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(user32.IsWindow(item["hwnd"]) for item in requested):
            break
        time.sleep(0.10)

    remaining = [
        item for item in requested if user32.IsWindow(item["hwnd"])
    ]
    closed = [
        item for item in requested if not user32.IsWindow(item["hwnd"])
    ]
    verified = not remaining and not request_failures

    return {
        "status": "ok",
        "operation": "close_all_apps",
        "scope": "active_virtual_desktop",
        "desktop_filter": desktop_filter,
        "desktop_filter_warning": desktop_filter_warning,
        "requested_windows": requested,
        "closed_windows": closed,
        "closed_count": len(closed),
        "remaining_windows": remaining,
        "request_failures": request_failures,
        "skipped_desktop_check": skipped_desktop_check,
        "verified": verified,
        "note": (
            None
            if verified
            else "Some windows remain; an app may be showing a save/confirmation prompt."
        ),
    }

"""Internal Windows native-Snap restoration for saved Dexter workspaces.

This module is deliberately not an MCP tool. Workspace orchestration calls it
when a saved workspace contains a supported group of Windows-arranged windows.
It recreates the native Snap topology first, then restores the saved divider
coordinates through DPI-correct physical mouse input. Normal/floating windows
continue to use windows_tools' deterministic geometry path.
"""

from __future__ import annotations

import ctypes
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from ctypes import wintypes

import windows_tools


SNAP_SETTLE_SECONDS = 0.45
SNAP_ASSIST_SETTLE_SECONDS = 0.18
# Fast divider timings are the normal path. The slower profile is the exact
# conservative sequence already proven on the real machine and is kept as an
# automatic fallback when post-drag verification fails.
SNAP_DIVIDER_FAST_APPROACH_PIXELS = 12
SNAP_DIVIDER_FAST_APPROACH_STEPS = 4
SNAP_DIVIDER_FAST_HOVER_SECONDS = 0.12
SNAP_DIVIDER_FAST_MOUSE_DOWN_SECONDS = 0.05
SNAP_DIVIDER_FAST_DRAG_MIN_STEPS = 4
SNAP_DIVIDER_FAST_DRAG_MAX_STEPS = 8
SNAP_DIVIDER_FAST_PRE_RELEASE_SECONDS = 0.04
SNAP_DIVIDER_FAST_SETTLE_SECONDS = 0.18
SNAP_DIVIDER_FAST_CURSOR_SETTLE_SECONDS = 0.008
SNAP_DIVIDER_FAST_STEP_PAUSE_SECONDS = 0.004

SNAP_DIVIDER_SLOW_HOVER_SECONDS = 0.65
SNAP_DIVIDER_SLOW_SETTLE_SECONDS = 0.55
SNAP_DIVIDER_SLOW_ATTEMPTS = 2
SNAP_DIVIDER_ATTEMPTS = 1 + SNAP_DIVIDER_SLOW_ATTEMPTS
SNAP_COORDINATE_TOLERANCE = 1

# Input constants.
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000

VK_LWIN = 0x5B
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_ESCAPE = 0x1B
VK_RETURN = 0x0D
VK_MENU = 0x12

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4.
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(ctypes.c_size_t(-4).value)

WM_SYSCOMMAND = 0x0112
SC_SIZE = 0xF000
WMSZ_RIGHT = 2

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

SW_RESTORE = 9
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010

SUPPORTED_ZONE_SETS = {
    frozenset({"left", "right"}),
    frozenset({"left", "top-right", "bottom-right"}),
    frozenset({"right", "top-left", "bottom-left"}),
    frozenset({"top-left", "bottom-left", "top-right", "bottom-right"}),
}

ZONE_ARROWS = {
    "left": (VK_LEFT,),
    "right": (VK_RIGHT,),
    "top-left": (VK_LEFT, VK_UP),
    "bottom-left": (VK_LEFT, VK_DOWN),
    "top-right": (VK_RIGHT, VK_UP),
    "bottom-right": (VK_RIGHT, VK_DOWN),
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


if os.name == "nt":
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        user32.SendInput.restype = wintypes.UINT
        user32.GetPhysicalCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        user32.GetPhysicalCursorPos.restype = wintypes.BOOL
        user32.SetPhysicalCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        user32.SetPhysicalCursorPos.restype = wintypes.BOOL
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        user32.AttachThreadInput.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = [wintypes.HWND]
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.SetFocus.argtypes = [wintypes.HWND]
        user32.SetFocus.restype = wintypes.HWND
        user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        kernel32.GetCurrentThreadId.restype = wintypes.DWORD
    except Exception:
        pass
else:
    user32 = None
    kernel32 = None


@dataclass(frozen=True)
class SnapWindow:
    hwnd: int
    zone: str
    target: dict


@dataclass(frozen=True)
class SnapDivider:
    axis: str
    target: int
    before_zones: tuple[str, ...]
    after_zones: tuple[str, ...]
    segment_start: int
    segment_end: int


@dataclass(frozen=True)
class SnapPlan:
    area: dict
    windows: tuple[SnapWindow, ...]
    dividers: tuple[SnapDivider, ...]


def _validated_bounds(bounds: dict) -> dict:
    if not isinstance(bounds, dict):
        raise ValueError("Snap window placement is missing DWM visible bounds.")
    try:
        left = int(bounds["left"])
        top = int(bounds["top"])
        right = int(bounds["right"])
        bottom = int(bounds["bottom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Snap window placement contains invalid visible bounds.") from exc
    if right <= left or bottom <= top:
        raise ValueError("Snap window visible bounds must have positive size.")
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top,
    }


def _bounds_equal(a: dict, b: dict, tolerance: int = 0) -> bool:
    return all(abs(int(a[key]) - int(b[key])) <= tolerance for key in ("left", "top", "right", "bottom"))


def _union(bounds: list[dict]) -> dict:
    return {
        "left": min(item["left"] for item in bounds),
        "top": min(item["top"] for item in bounds),
        "right": max(item["right"] for item in bounds),
        "bottom": max(item["bottom"] for item in bounds),
        "width": max(item["right"] for item in bounds) - min(item["left"] for item in bounds),
        "height": max(item["bottom"] for item in bounds) - min(item["top"] for item in bounds),
    }


def _classify_zone(rect: dict, area: dict) -> str:
    left = rect["left"] == area["left"]
    right = rect["right"] == area["right"]
    top = rect["top"] == area["top"]
    bottom = rect["bottom"] == area["bottom"]

    if left and top and bottom and not right:
        return "left"
    if right and top and bottom and not left:
        return "right"
    if left and top and not right and not bottom:
        return "top-left"
    if left and bottom and not right and not top:
        return "bottom-left"
    if right and top and not left and not bottom:
        return "top-right"
    if right and bottom and not left and not top:
        return "bottom-right"
    raise ValueError(
        "Saved arranged layout is not one of Dexter's currently supported native Snap topologies."
    )


def infer_snap_plan(items: list[dict]) -> SnapPlan:
    """Infer a supported native Snap topology from saved visible rectangles.

    Supported today: left/right pairs, one full-height side plus two quarters on
    the other side, and four quarters. Arbitrary divider positions inside those
    topologies are supported and are restored after native snapping.
    """
    if len(items) < 2:
        raise ValueError("Native Snap-group restoration needs at least two arranged windows.")

    prepared = []
    for item in items:
        hwnd = int(item["hwnd"])
        placement = item.get("placement") or {}
        if str(placement.get("show_state") or "normal") != "arranged":
            raise ValueError("Native Snap plan received a non-arranged window.")
        target = _validated_bounds(placement.get("visible_bounds"))
        prepared.append((hwnd, target))

    area = _union([target for _hwnd, target in prepared])
    windows = []
    used_zones: set[str] = set()
    for hwnd, target in prepared:
        zone = _classify_zone(target, area)
        if zone in used_zones:
            raise ValueError(f"Saved arranged layout contains more than one {zone!r} window.")
        used_zones.add(zone)
        windows.append(SnapWindow(hwnd=hwnd, zone=zone, target=target))

    zone_set = frozenset(used_zones)
    if zone_set not in SUPPORTED_ZONE_SETS:
        raise ValueError(
            f"Saved arranged topology is not supported yet: {sorted(zone_set)}"
        )

    by_zone = {window.zone: window for window in windows}
    dividers: list[SnapDivider] = []

    left_zones = tuple(zone for zone in ("left", "top-left", "bottom-left") if zone in by_zone)
    right_zones = tuple(zone for zone in ("right", "top-right", "bottom-right") if zone in by_zone)
    if left_zones and right_zones:
        left_edges = {by_zone[zone].target["right"] for zone in left_zones}
        right_edges = {by_zone[zone].target["left"] for zone in right_zones}
        if len(left_edges) != 1 or len(right_edges) != 1 or left_edges != right_edges:
            raise ValueError("Saved arranged windows do not share one exact vertical divider.")
        target_x = next(iter(left_edges))
        dividers.append(
            SnapDivider(
                axis="vertical",
                target=target_x,
                before_zones=left_zones,
                after_zones=right_zones,
                segment_start=area["top"],
                segment_end=area["bottom"],
            )
        )

    if "top-left" in by_zone and "bottom-left" in by_zone:
        top = by_zone["top-left"].target
        bottom = by_zone["bottom-left"].target
        if top["bottom"] != bottom["top"]:
            raise ValueError("Saved left-side quarters do not share one exact horizontal divider.")
        if top["right"] != bottom["right"]:
            raise ValueError("Saved left-side quarters do not share one exact right edge.")
        dividers.append(
            SnapDivider(
                axis="horizontal",
                target=top["bottom"],
                before_zones=("top-left",),
                after_zones=("bottom-left",),
                segment_start=area["left"],
                segment_end=top["right"],
            )
        )

    if "top-right" in by_zone and "bottom-right" in by_zone:
        top = by_zone["top-right"].target
        bottom = by_zone["bottom-right"].target
        if top["bottom"] != bottom["top"]:
            raise ValueError("Saved right-side quarters do not share one exact horizontal divider.")
        if top["left"] != bottom["left"]:
            raise ValueError("Saved right-side quarters do not share one exact left edge.")
        dividers.append(
            SnapDivider(
                axis="horizontal",
                target=top["bottom"],
                before_zones=("top-right",),
                after_zones=("bottom-right",),
                segment_start=top["left"],
                segment_end=area["right"],
            )
        )

    return SnapPlan(area=area, windows=tuple(windows), dividers=tuple(dividers))


def _require_windows() -> None:
    if os.name != "nt" or user32 is None:
        raise RuntimeError("Native Snap restoration is implemented for Windows only.")


def _key(vk: int, up: bool = False) -> INPUT:
    event = INPUT()
    event.type = INPUT_KEYBOARD
    event.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    return event


def _mouse(flags: int, dx: int = 0, dy: int = 0) -> INPUT:
    event = INPUT()
    event.type = INPUT_MOUSE
    event.mi = MOUSEINPUT(dx, dy, 0, flags, 0, 0)
    return event


def _send_inputs(events: list[INPUT]) -> None:
    if not events:
        return
    array_type = INPUT * len(events)
    array = array_type(*events)
    sent = user32.SendInput(len(events), array, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise RuntimeError(
            f"Windows accepted only {sent}/{len(events)} synthetic input events; "
            "UIPI/integrity restrictions may be blocking workspace Snap restoration."
        )


def _press_key(vk: int) -> None:
    _send_inputs([_key(vk), _key(vk, True)])


def _physical_cursor_pos() -> tuple[int, int]:
    point = wintypes.POINT()
    if not user32.GetPhysicalCursorPos(ctypes.byref(point)):
        raise ctypes.WinError()
    return int(point.x), int(point.y)


def _virtual_screen_metrics() -> tuple[int, int, int, int]:
    values = (
        user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        user32.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    )
    if values[2] <= 1 or values[3] <= 1:
        raise RuntimeError(f"Invalid virtual-screen metrics: {values}")
    return values


def _physical_to_absolute(x: int, y: int) -> tuple[int, int]:
    vx, vy, vw, vh = _virtual_screen_metrics()
    nx = round((x - vx) * 65535 / (vw - 1))
    ny = round((y - vy) * 65535 / (vh - 1))
    return max(0, min(65535, nx)), max(0, min(65535, ny))


def _move_cursor_physical(
    x: int,
    y: int,
    *,
    settle_seconds: float = 0.025,
) -> tuple[int, int]:
    nx, ny = _physical_to_absolute(x, y)
    _send_inputs(
        [
            _mouse(
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                nx,
                ny,
            )
        ]
    )
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    observed = _physical_cursor_pos()
    if abs(observed[0] - x) > SNAP_COORDINATE_TOLERANCE or abs(observed[1] - y) > SNAP_COORDINATE_TOLERANCE:
        raise RuntimeError(
            "Physical cursor coordinates did not match the requested DWM pixel coordinate: "
            f"requested={(x, y)}, observed={observed}."
        )
    return observed


def _human_move(
    start: tuple[int, int],
    end: tuple[int, int],
    steps: int = 18,
    *,
    cursor_settle_seconds: float = 0.025,
    step_pause_seconds: float = 0.012,
) -> None:
    sx, sy = start
    ex, ey = end
    for index in range(1, steps + 1):
        x = round(sx + (ex - sx) * index / steps)
        y = round(sy + (ey - sy) * index / steps)
        _move_cursor_physical(x, y, settle_seconds=cursor_settle_seconds)
        if step_pause_seconds > 0:
            time.sleep(step_pause_seconds)


@contextmanager
def _physical_dpi_context():
    """Temporarily make the worker thread Per-Monitor-V2 DPI aware.

    The MCP tool server is a background process and may inherit a DPI context
    that virtualizes GetSystemMetrics coordinates. DWM extended-frame bounds
    and GetPhysicalCursorPos are physical pixels, so separator input must use
    the same coordinate space as the successful isolated v8 probe.
    """
    _require_windows()
    setter = getattr(user32, "SetThreadDpiAwarenessContext", None)
    if setter is None:
        yield
        return

    previous = setter(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    if not previous:
        raise RuntimeError("Windows refused to enable Per-Monitor-V2 DPI awareness for Snap restoration.")
    try:
        yield
    finally:
        setter(previous)


def _focus(hwnd: int) -> None:
    """Activate a target window from Dexter's background MCP process.

    SetForegroundWindow is intentionally restricted for background processes.
    The standalone probes were launched from the foreground console, while the
    production tool server runs behind Dexter/Goose. First try the ordinary
    verified path, then temporarily attach this worker thread's input queue to
    the current foreground and target UI threads and retry.
    """
    try:
        windows_tools.focus_window_verified(hwnd)
        return
    except Exception as first_error:
        pass

    current_tid = int(kernel32.GetCurrentThreadId())
    foreground_hwnd = int(user32.GetForegroundWindow() or 0)
    foreground_tid = int(user32.GetWindowThreadProcessId(foreground_hwnd, None)) if foreground_hwnd else 0
    target_tid = int(user32.GetWindowThreadProcessId(hwnd, None))

    attached: list[int] = []
    try:
        for tid in (foreground_tid, target_tid):
            if tid and tid != current_tid and tid not in attached:
                if user32.AttachThreadInput(current_tid, tid, True):
                    attached.append(tid)

        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
        time.sleep(0.12)
        if int(user32.GetForegroundWindow() or 0) == hwnd:
            return

        # Windows documents that an ALT-key action unlocks foreground changes.
        # Send a harmless ALT tap only as the final fallback, then retry while
        # the input queues are still attached.
        _press_key(VK_MENU)
        time.sleep(0.05)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
        time.sleep(0.12)
        if int(user32.GetForegroundWindow() or 0) == hwnd:
            return
    finally:
        for tid in reversed(attached):
            user32.AttachThreadInput(current_tid, tid, False)

    observed = int(user32.GetForegroundWindow() or 0)
    raise RuntimeError(
        "Windows refused foreground activation even after attached-input fallback; "
        f"requested hwnd={hwnd}, observed foreground hwnd={observed}. "
        f"Initial focus error: {first_error}"
    )


def _break_arranged_if_needed(hwnd: int) -> None:
    if not windows_tools._is_window_arranged(hwnd):
        return
    _focus(hwnd)
    # Empirically verified isolation path: entering Windows' own modal sizing
    # mode and nudging one edge takes one window out of arranged state without
    # touching the mouse. We immediately stage the window at a harmless normal
    # rectangle before constructing the new Snap topology.
    if not user32.PostMessageW(hwnd, WM_SYSCOMMAND, SC_SIZE | WMSZ_RIGHT, 0):
        raise RuntimeError(f"Windows refused to enter sizing mode for hwnd={hwnd}.")
    time.sleep(0.18)
    _press_key(VK_LEFT)
    _press_key(VK_RETURN)
    time.sleep(0.22)
    if windows_tools._is_window_arranged(hwnd):
        raise RuntimeError(f"Could not take arranged window {hwnd} out of its previous Snap slot.")


def _stage_normal_window(window: SnapWindow, area: dict, index: int) -> None:
    hwnd = window.hwnd
    # SW_RESTORE is also the cleanest first attempt to take an already-snapped
    # window back to its normal placement. If an application remains arranged,
    # the modal sizing fallback below breaks that stale Snap state explicitly.
    windows_tools.user32.ShowWindow(hwnd, SW_RESTORE)
    time.sleep(0.12)
    _break_arranged_if_needed(hwnd)

    width = max(420, min(900, area["width"] // 2))
    height = max(320, min(700, area["height"] // 2))
    max_left = max(area["left"], area["right"] - width)
    max_top = max(area["top"], area["bottom"] - height)
    left = min(max_left, area["left"] + 80 + index * 36)
    top = min(max_top, area["top"] + 80 + index * 32)
    if not windows_tools.user32.SetWindowPos(
        hwnd,
        0,
        left,
        top,
        width,
        height,
        SWP_NOZORDER | SWP_NOACTIVATE,
    ):
        raise RuntimeError(f"Windows refused to stage window {hwnd} before native snapping.")
    time.sleep(0.10)


def _send_native_snap(hwnd: int, zone: str) -> None:
    """Place one HWND into a native Windows Snap slot with paced key input.

    Corner placement is a stateful Win+Arrow sequence. Sending every key event
    in one zero-delay SendInput batch was flaky on Ahmed's real three-window
    layout, especially for top-right. Keep Win held, but pause between the side
    and vertical arrows so Shell can commit the intermediate snap state.
    """
    arrows = ZONE_ARROWS[zone]
    _focus(hwnd)

    _send_inputs([_key(VK_LWIN)])
    try:
        for index, arrow in enumerate(arrows):
            _send_inputs([_key(arrow), _key(arrow, True)])
            time.sleep(0.20 if index + 1 < len(arrows) else 0.10)
    finally:
        _send_inputs([_key(VK_LWIN, True)])

    time.sleep(SNAP_SETTLE_SECONDS)
    # Close Snap Assist only after the full slot sequence is committed.
    _press_key(VK_ESCAPE)
    time.sleep(SNAP_ASSIST_SETTLE_SECONDS)
    if not windows_tools._is_window_arranged(hwnd):
        observed = windows_tools._visible_window_bounds(hwnd)
        raise RuntimeError(
            f"Windows did not report hwnd={hwnd} as arranged after native Snap to {zone}; "
            f"observed visible bounds={observed}."
        )


def _zone_outer_edges_match(zone: str, bounds: dict, area: dict, tolerance: int = 2) -> bool:
    def eq(a: int, b: int) -> bool:
        return abs(int(a) - int(b)) <= tolerance

    if zone == "left":
        return eq(bounds["left"], area["left"]) and eq(bounds["top"], area["top"]) and eq(bounds["bottom"], area["bottom"])
    if zone == "right":
        return eq(bounds["right"], area["right"]) and eq(bounds["top"], area["top"]) and eq(bounds["bottom"], area["bottom"])
    if zone == "top-left":
        return eq(bounds["left"], area["left"]) and eq(bounds["top"], area["top"])
    if zone == "bottom-left":
        return eq(bounds["left"], area["left"]) and eq(bounds["bottom"], area["bottom"])
    if zone == "top-right":
        return eq(bounds["right"], area["right"]) and eq(bounds["top"], area["top"])
    if zone == "bottom-right":
        return eq(bounds["right"], area["right"]) and eq(bounds["bottom"], area["bottom"])
    return False


def _snap_window_to_zone(window: SnapWindow, area: dict, index: int) -> None:
    last_observed = None
    for attempt in range(2):
        _stage_normal_window(window, area, index + attempt)
        _send_native_snap(window.hwnd, window.zone)
        observed = windows_tools._visible_window_bounds(window.hwnd)
        last_observed = observed
        if _zone_outer_edges_match(window.zone, observed, area):
            return
    raise RuntimeError(
        f"Window {window.hwnd} did not settle in native Snap zone {window.zone!r}; "
        f"observed={last_observed}, expected outer area={area}."
    )


def _window_by_zone(plan: SnapPlan) -> dict[str, SnapWindow]:
    return {window.zone: window for window in plan.windows}


def _active_divider_pair(
    plan: SnapPlan,
    divider: SnapDivider,
) -> tuple[SnapWindow, SnapWindow, int, int] | None:
    """Choose one still-unrestored adjacent pair for this saved divider.

    A four-quarter layout can have two physical vertical separator segments.
    Moving the top segment does not have to move the bottom segment, so each
    retry must select a pair that is still away from the saved coordinate
    instead of repeatedly grabbing the first already-correct pair.
    """
    by_zone = _window_by_zone(plan)
    for before_zone in divider.before_zones:
        for after_zone in divider.after_zones:
            before_window = by_zone[before_zone]
            after_window = by_zone[after_zone]
            before_target = before_window.target
            after_target = after_window.target

            if divider.axis == "vertical":
                if before_target["right"] != divider.target or after_target["left"] != divider.target:
                    continue
                segment_start = max(before_target["top"], after_target["top"])
                segment_end = min(before_target["bottom"], after_target["bottom"])
                if segment_end <= segment_start:
                    continue
                before_now = windows_tools._visible_window_bounds(before_window.hwnd)
                after_now = windows_tools._visible_window_bounds(after_window.hwnd)
                if before_now["right"] == divider.target and after_now["left"] == divider.target:
                    continue
                return before_window, after_window, segment_start, segment_end

            if before_target["bottom"] != divider.target or after_target["top"] != divider.target:
                continue
            segment_start = max(before_target["left"], after_target["left"])
            segment_end = min(before_target["right"], after_target["right"])
            if segment_end <= segment_start:
                continue
            before_now = windows_tools._visible_window_bounds(before_window.hwnd)
            after_now = windows_tools._visible_window_bounds(after_window.hwnd)
            if before_now["bottom"] == divider.target and after_now["top"] == divider.target:
                continue
            return before_window, after_window, segment_start, segment_end
    return None


def _current_pair_divider(
    divider: SnapDivider,
    before_window: SnapWindow,
    after_window: SnapWindow,
) -> int:
    before = windows_tools._visible_window_bounds(before_window.hwnd)
    after = windows_tools._visible_window_bounds(after_window.hwnd)
    if divider.axis == "vertical":
        if before["right"] != after["left"]:
            raise RuntimeError(
                "Native Snap pair no longer exposes one shared vertical separator: "
                f"{before['right']} != {after['left']}."
            )
        return before["right"]
    if before["bottom"] != after["top"]:
        raise RuntimeError(
            "Native Snap pair no longer exposes one shared horizontal separator: "
            f"{before['bottom']} != {after['top']}."
        )
    return before["bottom"]


def _divider_target_verified(plan: SnapPlan, divider: SnapDivider) -> bool:
    by_zone = _window_by_zone(plan)
    if divider.axis == "vertical":
        return all(
            windows_tools._visible_window_bounds(by_zone[zone].hwnd)["right"] == divider.target
            for zone in divider.before_zones
        ) and all(
            windows_tools._visible_window_bounds(by_zone[zone].hwnd)["left"] == divider.target
            for zone in divider.after_zones
        )
    return all(
        windows_tools._visible_window_bounds(by_zone[zone].hwnd)["bottom"] == divider.target
        for zone in divider.before_zones
    ) and all(
        windows_tools._visible_window_bounds(by_zone[zone].hwnd)["top"] == divider.target
        for zone in divider.after_zones
    )


def _drag_divider_once(
    plan: SnapPlan,
    divider: SnapDivider,
    *,
    conservative: bool = False,
) -> None:
    active = _active_divider_pair(plan, divider)
    if active is None:
        return
    before_window, after_window, segment_start, segment_end = active
    current = _current_pair_divider(divider, before_window, after_window)
    if current == divider.target:
        return

    if conservative:
        approach_distance = min(90, max(30, abs(current - (plan.area["left"] if divider.axis == "vertical" else plan.area["top"])) // 4))
        approach_steps = 16
        hover_seconds = SNAP_DIVIDER_SLOW_HOVER_SECONDS
        mouse_down_seconds = 0.14
        drag_steps = 24
        pre_release_seconds = 0.12
        settle_seconds = SNAP_DIVIDER_SLOW_SETTLE_SECONDS
        cursor_settle_seconds = 0.025
        step_pause_seconds = 0.012
    else:
        # The Shell only needs real pointer motion across the last few pixels of
        # the separator. Jump close to it, use a handful of genuine SendInput
        # moves, and verify immediately. If Windows misses the gesture, the
        # conservative proven sequence is retried automatically.
        approach_distance = SNAP_DIVIDER_FAST_APPROACH_PIXELS
        approach_steps = SNAP_DIVIDER_FAST_APPROACH_STEPS
        hover_seconds = SNAP_DIVIDER_FAST_HOVER_SECONDS
        mouse_down_seconds = SNAP_DIVIDER_FAST_MOUSE_DOWN_SECONDS
        delta = abs(divider.target - current)
        drag_steps = max(
            SNAP_DIVIDER_FAST_DRAG_MIN_STEPS,
            min(SNAP_DIVIDER_FAST_DRAG_MAX_STEPS, max(1, delta // 24)),
        )
        pre_release_seconds = SNAP_DIVIDER_FAST_PRE_RELEASE_SECONDS
        settle_seconds = SNAP_DIVIDER_FAST_SETTLE_SECONDS
        cursor_settle_seconds = SNAP_DIVIDER_FAST_CURSOR_SETTLE_SECONDS
        step_pause_seconds = SNAP_DIVIDER_FAST_STEP_PAUSE_SECONDS

    if divider.axis == "vertical":
        span = max(120, segment_end - segment_start)
        cross = segment_start + span // 2
        cross = min(segment_end - 40, max(segment_start + 40, cross))
        approach = (current - approach_distance, cross)
        separator = (current, cross)
        destination = (divider.target, cross)
    else:
        span = max(120, segment_end - segment_start)
        cross = segment_start + span // 2
        cross = min(segment_end - 40, max(segment_start + 40, cross))
        approach = (cross, current - approach_distance)
        separator = (cross, current)
        destination = (cross, divider.target)

    _move_cursor_physical(
        *approach,
        settle_seconds=cursor_settle_seconds,
    )
    _human_move(
        approach,
        separator,
        steps=approach_steps,
        cursor_settle_seconds=cursor_settle_seconds,
        step_pause_seconds=step_pause_seconds,
    )
    time.sleep(hover_seconds)
    _send_inputs([_mouse(MOUSEEVENTF_LEFTDOWN)])
    time.sleep(mouse_down_seconds)
    _human_move(
        separator,
        destination,
        steps=drag_steps,
        cursor_settle_seconds=cursor_settle_seconds,
        step_pause_seconds=step_pause_seconds,
    )
    time.sleep(pre_release_seconds)
    _send_inputs([_mouse(MOUSEEVENTF_LEFTUP)])
    time.sleep(settle_seconds)


def _drag_plan_divider(plan: SnapPlan, divider: SnapDivider) -> None:
    if _divider_target_verified(plan, divider):
        return

    # Normal path: one fast drag. Most successful divider restores should finish
    # here in well under a second instead of spending ~3 seconds animating.
    _drag_divider_once(plan, divider, conservative=False)
    if _divider_target_verified(plan, divider):
        return

    # Reliability fallback: preserve the already-proven slow gesture. Two tries
    # retain the previous total of three attempts (1 fast + 2 conservative).
    for _attempt in range(SNAP_DIVIDER_SLOW_ATTEMPTS):
        _drag_divider_once(plan, divider, conservative=True)
        if _divider_target_verified(plan, divider):
            return

    raise RuntimeError(
        f"Windows Snap divider did not settle at requested {divider.axis} coordinate {divider.target}."
    )


def _verify_plan(plan: SnapPlan) -> list[dict]:
    failures = []
    for window in plan.windows:
        observed = windows_tools._visible_window_bounds(window.hwnd)
        arranged = windows_tools._is_window_arranged(window.hwnd)
        if not arranged or not _bounds_equal(observed, window.target):
            failures.append(
                {
                    "hwnd": window.hwnd,
                    "zone": window.zone,
                    "requested": window.target,
                    "observed": observed,
                    "arranged": arranged,
                }
            )
    return failures


def restore_native_snap_layout(items: list[dict]) -> dict:
    """Rebuild a supported saved Snap group and restore its arbitrary dividers.

    The caller should fall back to ordinary exact geometry if this returns
    complete=False. The original cursor position is restored even on failure.
    """
    try:
        plan = infer_snap_plan(items)
    except Exception as exc:
        return {
            "status": "ok",
            "operation": "restore_native_snap_layout",
            "attempted": False,
            "supported": False,
            "complete": False,
            "restored": [],
            "failed": [{"reason": f"{type(exc).__name__}: {exc}"}],
        }

    _require_windows()
    original_cursor = _physical_cursor_pos()
    try:
        with _physical_dpi_context():
            # Build the topology from clean normal windows. Ordering keeps the
            # full-height side first, then quarter slots; Snap Assist is dismissed
            # after every placement so it cannot consume subsequent input.
            order = {
                "left": 0,
                "right": 0,
                "top-left": 1,
                "top-right": 1,
                "bottom-left": 2,
                "bottom-right": 2,
            }
            for index, window in enumerate(sorted(plan.windows, key=lambda item: (order[item.zone], item.zone))):
                _snap_window_to_zone(window, plan.area, index)

            for divider in plan.dividers:
                _drag_plan_divider(plan, divider)

            failures = _verify_plan(plan)
        if failures:
            return {
                "status": "ok",
                "operation": "restore_native_snap_layout",
                "attempted": True,
                "supported": True,
                "complete": False,
                "restored": [],
                "failed": failures,
            }
        return {
            "status": "ok",
            "operation": "restore_native_snap_layout",
            "attempted": True,
            "supported": True,
            "complete": True,
            "restored": [
                {
                    "hwnd": window.hwnd,
                    "zone": window.zone,
                    "state": "arranged",
                    "bounds": dict(window.target),
                    "verified": True,
                }
                for window in plan.windows
            ],
            "failed": [],
        }
    except Exception as exc:
        return {
            "status": "ok",
            "operation": "restore_native_snap_layout",
            "attempted": True,
            "supported": True,
            "complete": False,
            "restored": [],
            "failed": [{"reason": f"{type(exc).__name__}: {exc}"}],
        }
    finally:
        try:
            user32.SetPhysicalCursorPos(original_cursor[0], original_cursor[1])
        except Exception:
            pass

from __future__ import annotations

import time
from typing import Literal

import pyautogui

from config import MAX_TYPE_TEXT_CHARS, MAX_WAIT_SECONDS


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

MouseButton = Literal["left", "right", "middle"]
ComputerAction = Literal[
    "screen_size",
    "cursor_position",
    "mouse_move",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "drag_to",
    "scroll",
    "key",
    "hotkey",
    "type",
    "wait",
]


def _screen_size() -> tuple[int, int]:
    size = pyautogui.size()
    return int(size.width), int(size.height)


def _validate_coordinate(x: int, y: int) -> None:
    width, height = _screen_size()
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"Coordinate ({x}, {y}) is outside the primary screen "
            f"({width}x{height})."
        )


def _validate_optional_pair(x: int | None, y: int | None) -> None:
    if (x is None) != (y is None):
        raise ValueError("x and y must be supplied together or both omitted.")
    if x is not None and y is not None:
        _validate_coordinate(x, y)


def _validate_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Key must be a non-empty string.")
    normalized = key.strip().lower()
    if normalized not in pyautogui.KEYBOARD_KEYS:
        raise ValueError(
            f"Unsupported PyAutoGUI key name: {key!r}. "
            "Use a standard key such as enter, tab, ctrl, shift, a, f5, etc."
        )
    return normalized


def click_mouse(
    button: MouseButton = "left",
    x: int | None = None,
    y: int | None = None,
    clicks: int = 1,
    interval: float = 0.0,
) -> dict:
    _validate_optional_pair(x, y)

    if button not in {"left", "right", "middle"}:
        raise ValueError("button must be left, right, or middle.")
    if clicks < 1 or clicks > 10:
        raise ValueError("clicks must be between 1 and 10.")
    if interval < 0 or interval > 10:
        raise ValueError("interval must be between 0 and 10 seconds.")

    before = pyautogui.position()

    if x is None:
        pyautogui.click(button=button, clicks=clicks, interval=interval)
        target = {"x": int(before.x), "y": int(before.y)}
        cursor_was_moved = False
    else:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
        target = {"x": x, "y": y}
        cursor_was_moved = (int(before.x), int(before.y)) != (x, y)

    after = pyautogui.position()

    return {
        "status": "ok",
        "operation": "click",
        "button": button,
        "clicks": clicks,
        "target": target,
        "cursor_was_moved": cursor_was_moved,
        "cursor_after": {"x": int(after.x), "y": int(after.y)},
    }


def perform_computer_action(
    action: ComputerAction,
    x: int | None = None,
    y: int | None = None,
    text: str | None = None,
    key: str | None = None,
    keys: list[str] | None = None,
    amount: int | None = None,
    duration: float = 0.2,
) -> dict:
    if action == "screen_size":
        width, height = _screen_size()
        return {
            "status": "ok",
            "operation": "screen_size",
            "width": width,
            "height": height,
        }

    if action == "cursor_position":
        pos = pyautogui.position()
        return {
            "status": "ok",
            "operation": "cursor_position",
            "x": int(pos.x),
            "y": int(pos.y),
        }

    if action == "mouse_move":
        if x is None or y is None:
            raise ValueError("mouse_move requires x and y.")
        _validate_coordinate(x, y)
        if duration < 0 or duration > 10:
            raise ValueError("duration must be between 0 and 10 seconds.")
        pyautogui.moveTo(x, y, duration=duration)
        pos = pyautogui.position()
        verified = int(pos.x) == x and int(pos.y) == y
        if not verified:
            raise RuntimeError("Mouse move returned but coordinate verification failed.")
        return {
            "status": "ok",
            "operation": "mouse_move",
            "x": x,
            "y": y,
            "verified": True,
        }

    if action in {"left_click", "right_click", "middle_click", "double_click"}:
        button: MouseButton = (
            "left"
            if action in {"left_click", "double_click"}
            else "right"
            if action == "right_click"
            else "middle"
        )
        return click_mouse(
            button=button,
            x=x,
            y=y,
            clicks=2 if action == "double_click" else 1,
            interval=0.10 if action == "double_click" else 0.0,
        )

    if action == "drag_to":
        if x is None or y is None:
            raise ValueError("drag_to requires x and y.")
        _validate_coordinate(x, y)
        if duration < 0 or duration > 10:
            raise ValueError("duration must be between 0 and 10 seconds.")
        pyautogui.dragTo(x, y, duration=duration, button="left")
        pos = pyautogui.position()
        verified = int(pos.x) == x and int(pos.y) == y
        if not verified:
            raise RuntimeError("Drag returned but final-coordinate verification failed.")
        return {
            "status": "ok",
            "operation": "drag_to",
            "x": x,
            "y": y,
            "verified": True,
        }

    if action == "scroll":
        if amount is None or amount == 0:
            raise ValueError("scroll requires a non-zero integer amount.")
        if abs(amount) > 10_000:
            raise ValueError("scroll amount is unreasonably large.")
        _validate_optional_pair(x, y)
        pyautogui.scroll(amount, x=x, y=y)
        return {
            "status": "ok",
            "operation": "scroll",
            "amount": amount,
            "x": x,
            "y": y,
        }

    if action == "key":
        if key is None:
            raise ValueError("key action requires key.")
        normalized = _validate_key(key)
        pyautogui.press(normalized)
        return {
            "status": "ok",
            "operation": "key",
            "key": normalized,
        }

    if action == "hotkey":
        if not keys or len(keys) < 2 or len(keys) > 8:
            raise ValueError("hotkey requires between 2 and 8 keys.")
        normalized = [_validate_key(item) for item in keys]
        pyautogui.hotkey(*normalized)
        return {
            "status": "ok",
            "operation": "hotkey",
            "keys": normalized,
        }

    if action == "type":
        if text is None:
            raise ValueError("type action requires text.")
        if len(text) > MAX_TYPE_TEXT_CHARS:
            raise ValueError(
                f"Text exceeds the {MAX_TYPE_TEXT_CHARS}-character typing limit."
            )
        try:
            text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "Direct keyboard typing currently supports ASCII text only. "
                "For Unicode text, put it on the clipboard and paste it."
            ) from exc

        interval = max(0.0, min(duration, 1.0))
        pyautogui.write(text, interval=interval)
        return {
            "status": "ok",
            "operation": "type",
            "characters": len(text),
        }

    if action == "wait":
        if duration < 0 or duration > MAX_WAIT_SECONDS:
            raise ValueError(
                f"duration must be between 0 and {MAX_WAIT_SECONDS} seconds."
            )
        time.sleep(duration)
        return {
            "status": "ok",
            "operation": "wait",
            "seconds": duration,
        }

    raise ValueError(f"Unsupported computer action: {action}")

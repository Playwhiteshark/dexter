from __future__ import annotations

import time
from pathlib import Path

import pyautogui

from filesystem import resolve_allowed_path


def _region_tuple(
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int] | None:
    values = (x, y, width, height)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Region requires x, y, width, and height together.")

    assert x is not None and y is not None and width is not None and height is not None

    screen = pyautogui.size()
    if width <= 0 or height <= 0:
        raise ValueError("Region width and height must be positive.")
    if x < 0 or y < 0 or x + width > screen.width or y + height > screen.height:
        raise ValueError(
            f"Region ({x}, {y}, {width}, {height}) is outside the "
            f"primary screen ({screen.width}x{screen.height})."
        )

    return int(x), int(y), int(width), int(height)


def capture_screenshot(
    output_path: str,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
) -> dict:
    destination = resolve_allowed_path(output_path)

    if destination.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Screenshot output must end in .png, .jpg, or .jpeg.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Screenshot destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Screenshot destination directory does not exist: {destination.parent}"
        )

    region = _region_tuple(x, y, width, height)
    image = pyautogui.screenshot(region=region)
    image.save(destination)

    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("Screenshot save returned but file verification failed.")

    return {
        "status": "ok",
        "operation": "screenshot",
        "path": str(destination),
        "width": int(image.width),
        "height": int(image.height),
        "size_bytes": destination.stat().st_size,
        "region": region,
        "verified": True,
    }


def _locate(
    template_path: Path,
    confidence: float,
    region: tuple[int, int, int, int] | None,
    grayscale: bool,
):
    try:
        return pyautogui.locateOnScreen(
            str(template_path),
            confidence=confidence,
            region=region,
            grayscale=grayscale,
        )
    except getattr(pyautogui, "ImageNotFoundException", Exception) as exc:
        # Newer PyAutoGUI/PyScreeze versions raise ImageNotFoundException
        # instead of returning None. Treat "not found" as a normal condition.
        if exc.__class__.__name__ == "ImageNotFoundException":
            return None
        raise


def find_image_on_screen(
    template_path: str,
    confidence: float = 0.90,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    grayscale: bool = False,
) -> dict:
    template = resolve_allowed_path(template_path)
    if not template.is_file():
        raise FileNotFoundError(f"Template image does not exist: {template}")
    if template.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        raise ValueError("Template must be a supported image file.")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be greater than 0 and at most 1.")

    region = _region_tuple(x, y, width, height)
    box = _locate(template, confidence, region, grayscale)

    if box is None:
        return {
            "status": "ok",
            "operation": "find_image",
            "template": str(template),
            "found": False,
            "match": None,
        }

    match = {
        "left": int(box.left),
        "top": int(box.top),
        "width": int(box.width),
        "height": int(box.height),
        "center_x": int(box.left + box.width // 2),
        "center_y": int(box.top + box.height // 2),
    }

    return {
        "status": "ok",
        "operation": "find_image",
        "template": str(template),
        "found": True,
        "match": match,
        "confidence_threshold": confidence,
    }


def wait_for_image_on_screen(
    template_path: str,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.50,
    confidence: float = 0.90,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    grayscale: bool = False,
) -> dict:
    if timeout_seconds < 0 or timeout_seconds > 3600:
        raise ValueError("timeout_seconds must be between 0 and 3600.")
    if poll_interval_seconds < 0.05 or poll_interval_seconds > 60:
        raise ValueError("poll_interval_seconds must be between 0.05 and 60.")

    deadline = time.monotonic() + timeout_seconds
    attempts = 0

    while True:
        attempts += 1
        result = find_image_on_screen(
            template_path=template_path,
            confidence=confidence,
            x=x,
            y=y,
            width=width,
            height=height,
            grayscale=grayscale,
        )

        if result["found"]:
            result.update(
                {
                    "operation": "wait_for_image",
                    "timed_out": False,
                    "attempts": attempts,
                }
            )
            return result

        if time.monotonic() >= deadline:
            return {
                "status": "ok",
                "operation": "wait_for_image",
                "template": str(resolve_allowed_path(template_path)),
                "found": False,
                "match": None,
                "timed_out": True,
                "attempts": attempts,
            }

        time.sleep(poll_interval_seconds)

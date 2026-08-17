"""Windows Explorer folder capture/restore using Shell.Application COM."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def _shell_windows():
    if os.name != "nt":
        raise RuntimeError("Explorer state capture is Windows-only.")
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("pywin32 is required for Explorer state capture.") from exc
    return win32com.client.Dispatch("Shell.Application").Windows()


def _path_from_location_url(url: str) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme.casefold() != "file":
        return None
    path = unquote(parsed.path or "")
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    elif len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return str(Path(path)) if path else None


def _find_shell_window(hwnd: int):
    windows = _shell_windows()
    for item in windows:
        try:
            if int(item.HWND) == int(hwnd):
                return item
        except Exception:
            continue
    return None


def capture_explorer_state(window: dict) -> dict:
    item = _find_shell_window(int(window["hwnd"]))
    if item is None:
        raise LookupError("Explorer COM window was not found for the captured HWND.")
    try:
        location_url = str(item.LocationURL or "")
    except Exception as exc:
        raise RuntimeError(f"Explorer did not expose LocationURL: {exc}") from exc
    folder = _path_from_location_url(location_url)
    if not folder:
        raise RuntimeError("Explorer window is not showing a normal filesystem folder.")
    return {
        "kind": "explorer",
        "folder": folder,
    }


def restore_explorer_state(saved: dict, current_window: dict) -> dict:
    state = saved.get("app_state") or {}
    folder = state.get("folder")
    if not isinstance(folder, str) or not folder:
        raise LookupError("Saved Explorer state has no folder path.")
    item = _find_shell_window(int(current_window["hwnd"]))
    if item is None:
        raise LookupError("Restored Explorer HWND is not available through Shell.Application.")
    item.Navigate(Path(folder).as_uri())
    return {"status": "ok", "kind": "explorer", "folder": folder}

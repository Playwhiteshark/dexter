"""Local bridge client for Chromium workspace tab capture/restore.

The browser extension owns browser-internal APIs. Dexter talks to it indirectly
through request/response files serviced by the native-messaging host. This module
contains no MCP tools and exposes no browser internals to the model.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


BRIDGE_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Dexter" / "browser_bridge"
SESSIONS_DIR = BRIDGE_ROOT / "sessions"
REQUESTS_DIR = BRIDGE_ROOT / "requests"
RESPONSES_DIR = BRIDGE_ROOT / "responses"
SESSION_FRESH_SECONDS = 20.0
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 2.5

_BROWSER_KIND_BY_PROCESS = {
    "brave.exe": "brave",
    "chrome.exe": "chrome",
    "msedge.exe": "edge",
}


def browser_kind_for_process(process_name: str | None) -> str | None:
    if not process_name:
        return None
    return _BROWSER_KIND_BY_PROCESS.get(process_name.strip().casefold())


def _ensure_dirs() -> None:
    for path in (SESSIONS_DIR, REQUESTS_DIR, RESPONSES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _session_records(browser_kind: str) -> list[dict]:
    _ensure_dirs()
    now = time.time()
    result: list[dict] = []
    for path in SESSIONS_DIR.glob("*.json"):
        record = _read_json(path)
        if not record or record.get("browser_kind") != browser_kind:
            continue
        try:
            updated = float(record.get("updated_unix", 0))
        except (TypeError, ValueError):
            continue
        if now - updated > SESSION_FRESH_SECONDS:
            continue
        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        result.append(record)
    result.sort(key=lambda item: float(item.get("updated_unix", 0)), reverse=True)
    return result


def bridge_available(browser_kind: str) -> bool:
    return bool(_session_records(browser_kind))


def _request_session(
    session_id: str,
    browser_kind: str,
    op: str,
    payload: dict | None = None,
    *,
    timeout_seconds: float = DEFAULT_RESPONSE_TIMEOUT_SECONDS,
) -> dict:
    _ensure_dirs()
    request_id = f"req_{uuid.uuid4().hex}"
    session_request_dir = REQUESTS_DIR / session_id
    session_request_dir.mkdir(parents=True, exist_ok=True)
    request_path = session_request_dir / f"{request_id}.json"
    response_path = RESPONSES_DIR / f"{request_id}.json"

    body = {
        "request_id": request_id,
        "browser_kind": browser_kind,
        "op": op,
        "payload": payload or {},
        "created_unix": time.time(),
    }
    temp = request_path.with_suffix(".tmp")
    temp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    temp.replace(request_path)

    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            if response_path.is_file():
                response = _read_json(response_path)
                if response is None:
                    raise RuntimeError("Browser bridge returned an unreadable response.")
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error") or "Browser bridge request failed."))
                result = response.get("result")
                return result if isinstance(result, dict) else {"value": result}
            time.sleep(0.04)
    finally:
        try:
            request_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            response_path.unlink(missing_ok=True)
        except OSError:
            pass

    raise TimeoutError(
        f"Browser bridge session {session_id!r} did not answer {op!r} within {timeout_seconds:.1f}s."
    )


def capture_browser_windows(browser_kind: str) -> list[dict]:
    windows: list[dict] = []
    errors: list[str] = []
    for session in _session_records(browser_kind):
        session_id = session["session_id"]
        try:
            result = _request_session(session_id, browser_kind, "capture")
        except Exception as exc:
            errors.append(f"{session_id}: {type(exc).__name__}: {exc}")
            continue
        for window in result.get("windows", []):
            if not isinstance(window, dict) or window.get("incognito"):
                continue
            item = dict(window)
            item["_bridge_session"] = session_id
            windows.append(item)
    if not windows and errors:
        raise RuntimeError("; ".join(errors[:3]))
    return windows


def restore_browser_window(
    browser_kind: str,
    session_id: str,
    browser_window_id: int,
    state: dict,
) -> dict:
    payload = {
        "window_id": int(browser_window_id),
        "tabs": state.get("tabs", []),
        "groups": state.get("groups", []),
        "active_index": state.get("active_index", 0),
    }
    return _request_session(session_id, browser_kind, "restore_window", payload, timeout_seconds=5.0)

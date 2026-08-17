"""VS Code workspace identity captured by a tiny local extension bridge.

Dexter intentionally does not recreate individual VS Code editor tabs. VS Code
already persists folders, layout, opened files, dirty files, and untitled
buffers. Reopening them ourselves is both redundant and disruptive.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Dexter" / "vscode_bridge"
SESSIONS = ROOT / "sessions"
GENERATED = ROOT / "generated_workspaces"
SESSION_FRESH_SECONDS = 20.0


def _norm(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _sessions() -> list[dict]:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    now = time.time()
    result = []
    for path in SESSIONS.glob("*.json"):
        item = _read_json(path)
        if not item:
            continue
        try:
            updated = float(item.get("updated_unix", 0))
        except (TypeError, ValueError):
            continue
        if now - updated <= SESSION_FRESH_SECONDS and item.get("session_id"):
            result.append(item)
    result.sort(key=lambda item: float(item.get("updated_unix", 0)), reverse=True)
    return result


def _window_match_score(os_window: dict, session: dict) -> int:
    title = _norm(os_window.get("title"))
    score = 0
    workspace_name = _norm(session.get("workspace_name"))
    active_file = Path(session.get("active_file") or "").name.casefold()
    if workspace_name and workspace_name in title:
        score += 100
    if active_file and active_file in title:
        score += 60
    for folder in session.get("workspace_folders") or []:
        name = Path(folder).name.casefold()
        if name and name in title:
            score += 35

    # Focus is useful only as a tie-breaker after the OS window title has
    # provided an actual identity signal. A focused bridge session by itself
    # must never claim an unrelated Code HWND; doing so can make deterministic
    # tests depend on a real live VS Code session and, more importantly, can
    # make production matching reject the correct window as the wrong project.
    if score > 0 and session.get("focused"):
        score += 5
    return score


def match_vscode_session(os_window: dict) -> dict | None:
    ranked = sorted(
        ((_window_match_score(os_window, item), item) for item in _sessions()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not ranked or ranked[0][0] <= 0:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def capture_vscode_state(os_window: dict) -> dict:
    session = match_vscode_session(os_window)
    if session is None:
        raise LookupError(
            "Dexter's VS Code workspace extension is not active for this Code window, or the window could not be matched."
        )
    return {
        "kind": "vscode",
        "workspace_name": session.get("workspace_name"),
        "workspace_file": session.get("workspace_file"),
        "workspace_folders": session.get("workspace_folders") or [],
        "active_file": session.get("active_file"),
        "editor_restore_strategy": "vscode_native_session",
    }


def vscode_open_target(state: dict) -> str | None:
    workspace_file = state.get("workspace_file")
    if isinstance(workspace_file, str) and workspace_file:
        if workspace_file.casefold().startswith("file://"):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(workspace_file)
            path = unquote(parsed.path)
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return path
        if not workspace_file.casefold().startswith("untitled:"):
            return workspace_file

    folders = [folder for folder in state.get("workspace_folders") or [] if isinstance(folder, str) and folder]
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        GENERATED.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256("\0".join(folders).encode("utf-8")).hexdigest()[:12]
        path = GENERATED / f"dexter-{digest}.code-workspace"
        payload = {"folders": [{"path": folder} for folder in folders]}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
    return None


def _path_from_uri_or_path(value: str | None) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.casefold().startswith("file://"):
        from urllib.parse import unquote, urlparse
        parsed = urlparse(value)
        path = unquote(parsed.path)
        if parsed.netloc:
            path = f"//{parsed.netloc}{path}"
        elif len(path) >= 3 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path
    if "://" in value:
        return value
    return value


def _path_key(value: str | None) -> str:
    path = _path_from_uri_or_path(value)
    if not path:
        return ""
    return os.path.normcase(os.path.normpath(path))


def _session_matches_saved_workspace(session: dict, state: dict) -> bool:
    saved_file = state.get("workspace_file")
    session_file = session.get("workspace_file")
    if saved_file and not str(saved_file).casefold().startswith("untitled:"):
        if _path_key(str(saved_file)) == _path_key(str(session_file or "")):
            return True

    saved_folders = {
        _path_key(folder)
        for folder in (state.get("workspace_folders") or [])
        if isinstance(folder, str) and folder
    }
    session_folders = {
        _path_key(folder)
        for folder in (session.get("workspace_folders") or [])
        if isinstance(folder, str) and folder
    }
    if saved_folders:
        return saved_folders == session_folders

    return not saved_file and not saved_folders and not session_file and not session_folders


def vscode_window_matches_saved_state(os_window: dict, state: dict) -> bool | None:
    """Return True/False when the bridge can identify this Code window.

    None means the bridge cannot currently identify the window, in which case
    generic HWND matching may still be used as a compatibility fallback.
    """
    session = match_vscode_session(os_window)
    if session is None:
        return None
    return _session_matches_saved_workspace(session, state)


def _wait_for_saved_workspace_session(
    state: dict,
    *,
    timeout_seconds: float = 6.0,
) -> dict | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for session in _sessions():
            if _session_matches_saved_workspace(session, state):
                return session
        time.sleep(0.10)
    return None


def restore_vscode_state(saved: dict, current_window: dict) -> dict:
    state = saved.get("app_state") or {}

    # A freshly launched Code HWND can appear before the extension host has
    # written its first heartbeat. Wait for the saved project identity, but do
    # not open/close/replay any editor tabs. VS Code owns that state itself.
    deadline = time.monotonic() + 4.0
    session = None
    while time.monotonic() < deadline:
        candidate = match_vscode_session(current_window)
        if candidate is not None and _session_matches_saved_workspace(candidate, state):
            session = candidate
            break
        time.sleep(0.10)

    if session is None:
        session = _wait_for_saved_workspace_session(state, timeout_seconds=2.0)
    if session is None:
        raise LookupError(
            "Restored VS Code window is not connected to Dexter's bridge for the saved project. "
            "Dexter will not repurpose another Code window because that could disturb dirty editors."
        )

    return {
        "status": "ok",
        "kind": "vscode",
        "workspace_switched": False,
        "editor_restore": "vscode_native_session",
        "active_file": state.get("active_file"),
    }

"""Capture and restore local Windows application layouts."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from config import DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS, WORKSPACE_EXCLUDED_WINDOW_TITLES
from launcher import configured_app_alias_for_executable, launch_app
from resources import open_saved_resource, resource_launch_candidates_for_window
from state_store import load_state, mutate_state
from windows_tools import capture_workspace_windows, restore_window_placement


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _validate_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Workspace name must be a non-empty string.")
    clean = " ".join(value.strip().split())
    if len(clean) > 160:
        raise ValueError("Workspace name is too long; limit is 160 characters.")
    return clean


def _resolve_workspace(query: str, state: dict | None = None) -> dict:
    clean = _validate_name(query)
    records = list((state or load_state())["workspaces"].values())
    exact = [r for r in records if _normalize(r["name"]) == _normalize(clean)]
    if len(exact) == 1:
        return exact[0]
    partial = [r for r in records if _normalize(clean) in _normalize(r["name"])]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise LookupError(f"No saved workspace matches {clean!r}.")
    names = ", ".join(sorted(r["name"] for r in partial))
    raise LookupError(f"Workspace name is ambiguous: {clean!r}. Matches: {names}")


def _window_persistent_identity(window: dict) -> dict:
    item = {
        "title": window.get("title") or "",
        "process_name": window.get("process_name"),
        "exe": window.get("exe"),
        "placement": window["placement"],
        "z_index": window.get("z_index"),
    }

    alias = configured_app_alias_for_executable(window.get("exe"))
    if alias:
        item["app_alias"] = alias

    resource_candidates = resource_launch_candidates_for_window(window)
    if len(resource_candidates) == 1:
        item["resource_id"] = resource_candidates[0]["id"]
        item["resource_name"] = resource_candidates[0]["name"]

    return item


def save_workspace(name: str, overwrite: bool = False) -> dict:
    clean_name = _validate_name(name)
    captured = capture_workspace_windows()
    excluded_titles = {value.casefold() for value in WORKSPACE_EXCLUDED_WINDOW_TITLES}
    candidates = [
        window for window in captured["windows"]
        if (window.get("title") or "").strip().casefold() not in excluded_titles
    ]
    windows = [_window_persistent_identity(window) for window in candidates]
    if not windows:
        raise RuntimeError("No user-facing windows were available to save in this workspace.")

    now = _utc_now()

    def mutate(state: dict) -> dict:
        existing_id = None
        for workspace_id, record in state["workspaces"].items():
            if _normalize(record["name"]) == _normalize(clean_name):
                existing_id = workspace_id
                break
        if existing_id and not overwrite:
            raise FileExistsError(
                f"Saved workspace already exists: {clean_name!r}. Use overwrite=true to replace it."
            )

        workspace_id = existing_id or f"ws_{uuid.uuid4().hex[:12]}"
        created_utc = state["workspaces"].get(workspace_id, {}).get("created_utc", now)
        state["workspaces"][workspace_id] = {
            "id": workspace_id,
            "name": clean_name,
            "scope": "non-cloaked visible top-level windows",
            "windows": windows,
            "created_utc": created_utc,
            "updated_utc": now,
        }
        return {
            "status": "ok",
            "operation": "save_workspace",
            "id": workspace_id,
            "name": clean_name,
            "window_count": len(windows),
            "created": existing_id is None,
            "updated": existing_id is not None,
        }

    return mutate_state(mutate)


def search_workspaces(query: str | None = None, max_results: int = 10) -> dict:
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50.")
    state = load_state()
    records = list(state["workspaces"].values())
    if query:
        q = _normalize(query)
        records = [record for record in records if q in _normalize(record["name"])]
    records.sort(key=lambda record: _normalize(record["name"]))
    selected = records[:max_results]
    return {
        "status": "ok",
        "operation": "search_workspaces",
        "workspaces": [
            {
                "id": record["id"],
                "name": record["name"],
                "window_count": len(record.get("windows", [])),
                "updated_utc": record.get("updated_utc"),
            }
            for record in selected
        ],
        "truncated": len(records) > max_results,
    }


def _identity_matches(saved: dict, current: dict) -> bool:
    saved_exe = (saved.get("exe") or "").casefold()
    current_exe = (current.get("exe") or "").casefold()
    if saved_exe and current_exe:
        return saved_exe == current_exe
    return (saved.get("process_name") or "").casefold() == (current.get("process_name") or "").casefold()


def _match_window(saved: dict, current_windows: list[dict], used: set[int]) -> tuple[dict | None, str | None]:
    candidates = [
        window for window in current_windows
        if window["hwnd"] not in used and _identity_matches(saved, window)
    ]
    if not candidates:
        return None, None

    saved_title = (saved.get("title") or "").casefold()
    exact_title = [w for w in candidates if (w.get("title") or "").casefold() == saved_title]
    if len(exact_title) == 1:
        return exact_title[0], None
    if len(candidates) == 1:
        return candidates[0], None
    return None, "multiple current windows match the saved process but not the saved title"


def _resource_name_by_id(resource_id: str) -> str | None:
    record = load_state()["resources"].get(resource_id)
    return record.get("name") if isinstance(record, dict) else None


def _launch_missing_window(saved: dict) -> str | None:
    resource_id = saved.get("resource_id")
    if resource_id:
        name = _resource_name_by_id(resource_id)
        if name:
            open_saved_resource(name)
            return f"resource:{name}"

    alias = saved.get("app_alias")
    if alias:
        launch_app(alias)
        return f"app:{alias}"
    return None


def restore_workspace(
    name: str,
    launch_missing: bool = True,
    timeout_seconds: float = DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS,
) -> dict:
    if timeout_seconds < 0 or timeout_seconds > 30:
        raise ValueError("timeout_seconds must be between 0 and 30.")

    workspace = _resolve_workspace(name)
    saved_windows = workspace.get("windows", [])
    current = capture_workspace_windows()["windows"]
    used: set[int] = set()
    restored: list[dict] = []
    missing: list[dict] = []
    ambiguous: list[dict] = []
    launched: list[str] = []

    for saved in saved_windows:
        match, problem = _match_window(saved, current, used)

        if match is None and problem is None and launch_missing:
            launch_label = _launch_missing_window(saved)
            if launch_label:
                launched.append(launch_label)
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    time.sleep(0.20)
                    current = capture_workspace_windows()["windows"]
                    match, problem = _match_window(saved, current, used)
                    if match is not None or problem is not None:
                        break

        if match is None:
            info = {
                "title": saved.get("title"),
                "process_name": saved.get("process_name"),
            }
            if problem:
                info["reason"] = problem
                ambiguous.append(info)
            else:
                info["reason"] = "window is not open and has no usable saved launch hint"
                missing.append(info)
            continue

        restore_window_placement(match["hwnd"], saved["placement"])
        used.add(match["hwnd"])
        restored.append(
            {
                "hwnd": match["hwnd"],
                "title": match.get("title"),
                "process_name": match.get("process_name"),
            }
        )

    return {
        "status": "ok",
        "operation": "restore_workspace",
        "name": workspace["name"],
        "restored": restored,
        "restored_count": len(restored),
        "launched": launched,
        "missing": missing,
        "ambiguous": ambiguous,
        "complete": not missing and not ambiguous,
    }


def forget_workspace(query: str) -> dict:
    def mutate(state: dict) -> dict:
        record = _resolve_workspace(query, state)
        del state["workspaces"][record["id"]]
        return {
            "status": "ok",
            "operation": "forget_workspace",
            "id": record["id"],
            "name": record["name"],
            "deleted": True,
        }

    return mutate_state(mutate)

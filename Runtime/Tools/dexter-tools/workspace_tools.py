"""Capture, persist, reopen, and restore local Windows application layouts."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from config import DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS, WORKSPACE_EXCLUDED_WINDOW_TITLES
from launcher import configured_app_alias_for_executable, launch_app
from resources import open_saved_resource, resource_launch_candidates_for_window
from saved_name_resolution import rank_named_records, resolve_named_record
from state_store import load_state, mutate_state
from workspace_app_state import capture_window_app_state, restore_window_app_state
from snap_layout import restore_native_snap_layout
from windows_tools import capture_workspace_windows, restore_window_placements_batch
from vscode_state import vscode_window_matches_saved_state
from workspace_providers import (
    launch_workspace_provider,
    provider_for_process_name,
    provider_for_saved_window,
    provider_label,
    provider_matches_window,
    provider_requires_resizable_main,
)


LAUNCH_POLL_SECONDS = 0.20
LAUNCH_STABLE_POLLS = 3
MAX_WORKSPACE_LAUNCH_TIMEOUT_SECONDS = 120.0


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
    return resolve_named_record(clean, records, record_label="workspace")


def _window_persistent_identity(window: dict) -> dict:
    item = {
        "title": window.get("title") or "",
        "process_name": window.get("process_name"),
        "exe": window.get("exe"),
        "placement": window["placement"],
        "z_index": window.get("z_index"),
    }

    if isinstance(window.get("resizable"), bool):
        item["resizable"] = window["resizable"]

    provider = provider_for_process_name(window.get("process_name"))
    if provider:
        item["workspace_provider"] = provider

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
    windows: list[dict] = []
    state_warnings: list[dict] = []
    for window in candidates:
        item = _window_persistent_identity(window)
        app_state, warning = capture_window_app_state(window)
        if app_state:
            item["app_state"] = app_state
        if warning:
            state_warnings.append(
                {
                    "title": window.get("title"),
                    "process_name": window.get("process_name"),
                    "warning": warning,
                }
            )
        windows.append(item)
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
            "scope": "non-cloaked visible top-level windows excluding shell desktop",
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
            "captured_windows": [
                {
                    "title": item.get("title"),
                    "process_name": item.get("process_name"),
                    "provider": item.get("workspace_provider"),
                }
                for item in windows
            ],
            "created": existing_id is None,
            "updated": existing_id is not None,
            "state_warnings": state_warnings,
            "app_state_window_count": sum(1 for item in windows if item.get("app_state")),
        }

    return mutate_state(mutate)


def search_workspaces(query: str | None = None, max_results: int = 10) -> dict:
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50.")
    state = load_state()
    records = list(state["workspaces"].values())
    if query:
        ranked = rank_named_records(query, records)
        ordered = [record for _score, record in ranked]
    else:
        ordered = sorted(records, key=lambda record: _normalize(record["name"]))
    selected = ordered[:max_results]
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
        "truncated": len(ordered) > max_results,
    }


def _identity_matches(saved: dict, current: dict) -> bool:
    saved_exe = (saved.get("exe") or "").casefold()
    current_exe = (current.get("exe") or "").casefold()
    if saved_exe and current_exe and saved_exe == current_exe:
        return True

    saved_process = (saved.get("process_name") or "").casefold()
    current_process = (current.get("process_name") or "").casefold()
    if saved_process and current_process and saved_process == current_process:
        return True

    # Only providers whose visible window can legitimately move between helper
    # process names should broaden process identity. In particular, do not let
    # the Chromium provider turn Brave into Chrome/Edge just because they share
    # a provider implementation.
    provider = provider_for_saved_window(saved)
    if provider == "steam" and provider_matches_window(provider, current):
        return True
    return False


def _is_main_window_candidate(saved: dict, window: dict) -> bool:
    provider = provider_for_saved_window(saved)
    if not provider_requires_resizable_main(provider):
        return True

    # If a future snapshot explicitly says the saved window itself was fixed
    # size, do not incorrectly require a sizing border. Old snapshots lack this
    # field and therefore use the safer provider default.
    if saved.get("resizable") is False:
        return True
    return bool(window.get("resizable"))


def _candidate_windows(
    saved: dict,
    current_windows: list[dict],
    used: set[int],
    *,
    only_hwnds: set[int] | None = None,
) -> list[dict]:
    result = []
    for window in current_windows:
        hwnd = int(window["hwnd"])
        if hwnd in used:
            continue
        if only_hwnds is not None and hwnd not in only_hwnds:
            continue
        if not _identity_matches(saved, window):
            continue

        # VS Code workspaces are stateful enough that process-name equality is
        # not sufficient. If the local bridge can identify a live Code window
        # as a different project, do not hijack it; let the provider launch the
        # saved project in its own window. This also avoids triggering dirty-file
        # prompts by force-switching an unrelated editor window.
        if provider_for_saved_window(saved) == "vscode":
            state = saved.get("app_state")
            if isinstance(state, dict) and state.get("kind") == "vscode":
                verdict = vscode_window_matches_saved_state(window, state)
                if verdict is False:
                    continue

        result.append(window)
    return result


def _select_match(
    saved: dict,
    current_windows: list[dict],
    used: set[int],
    *,
    only_hwnds: set[int] | None = None,
) -> tuple[dict | None, str | None, list[dict]]:
    candidates = _candidate_windows(saved, current_windows, used, only_hwnds=only_hwnds)
    if not candidates:
        return None, None, []

    ready = [window for window in candidates if _is_main_window_candidate(saved, window)]
    if not ready:
        return None, "startup_pending", candidates

    saved_title = (saved.get("title") or "").casefold()
    exact_title = [w for w in ready if (w.get("title") or "").casefold() == saved_title]
    if len(exact_title) == 1:
        return exact_title[0], None, ready
    if len(ready) == 1:
        return ready[0], None, ready
    return None, "ambiguous", ready


def _window_stability_signature(window: dict) -> tuple:
    bounds = window.get("visible_bounds") or window.get("bounds") or {}
    return (
        (window.get("title") or "").strip(),
        int(bounds.get("left", 0)),
        int(bounds.get("top", 0)),
        int(bounds.get("right", 0)),
        int(bounds.get("bottom", 0)),
        bool(window.get("resizable")),
    )


def _wait_for_ready_window(
    saved: dict,
    used: set[int],
    timeout_seconds: float,
    *,
    baseline_hwnds: set[int] | None,
) -> tuple[dict | None, str | None, dict]:
    """Wait for a stable main HWND, deliberately ignoring startup splash cards."""
    deadline = time.monotonic() + timeout_seconds
    stable: dict[int, tuple[tuple, int]] = {}
    transient_titles: set[str] = set()
    last_ambiguous: list[str] = []

    while True:
        current = capture_workspace_windows()["windows"]
        only_hwnds = None
        if baseline_hwnds is not None:
            only_hwnds = {int(window["hwnd"]) for window in current if int(window["hwnd"]) not in baseline_hwnds}

        match, problem, considered = _select_match(
            saved,
            current,
            used,
            only_hwnds=only_hwnds,
        )

        if problem == "startup_pending":
            transient_titles.update((w.get("title") or "<untitled>") for w in considered)
        elif problem == "ambiguous":
            last_ambiguous = [(w.get("title") or "<untitled>") for w in considered]
        elif match is not None:
            hwnd = int(match["hwnd"])
            signature = _window_stability_signature(match)
            previous = stable.get(hwnd)
            count = previous[1] + 1 if previous and previous[0] == signature else 1
            stable[hwnd] = (signature, count)
            if count >= LAUNCH_STABLE_POLLS:
                return match, None, {
                    "transient_titles": sorted(transient_titles),
                    "stable_polls": count,
                }

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(LAUNCH_POLL_SECONDS, remaining))

    if last_ambiguous:
        return None, "ambiguous", {
            "candidate_titles": last_ambiguous,
            "transient_titles": sorted(transient_titles),
        }
    return None, "pending", {
        "transient_titles": sorted(transient_titles),
    }


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

    launched = launch_workspace_provider(saved)
    if launched:
        return f"provider:{launched.get('provider') or provider_label(provider_for_saved_window(saved))}"
    return None


def _window_identity_summary(saved: dict) -> dict:
    result = {
        "title": saved.get("title"),
        "process_name": saved.get("process_name"),
    }
    provider = provider_for_saved_window(saved)
    if provider:
        result["provider"] = provider
    return result


def restore_workspace(
    name: str,
    launch_missing: bool = True,
    timeout_seconds: float = DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS,
) -> dict:
    if timeout_seconds < 0 or timeout_seconds > MAX_WORKSPACE_LAUNCH_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be between 0 and {int(MAX_WORKSPACE_LAUNCH_TIMEOUT_SECONDS)}."
        )

    workspace = _resolve_workspace(name)
    saved_windows = workspace.get("windows", [])
    current = capture_workspace_windows()["windows"]
    used: set[int] = set()
    matched: list[dict] = []
    missing: list[dict] = []
    ambiguous: list[dict] = []
    pending: list[dict] = []
    launched: list[str] = []
    pre_failures: list[dict] = []

    for saved in saved_windows:
        match, problem, candidates = _select_match(saved, current, used)

        # An already-running app may currently expose only a splash/loading
        # window. Do not launch a duplicate and do not resize that transient
        # card; wait for its real resizable main window first.
        if match is None and problem == "startup_pending":
            match, wait_problem, details = _wait_for_ready_window(
                saved,
                used,
                timeout_seconds,
                baseline_hwnds=None,
            )
            if match is None:
                info = _window_identity_summary(saved)
                info["reason"] = (
                    "application is still starting; no stable resizable main window appeared before timeout"
                )
                if details.get("transient_titles"):
                    info["transient_titles"] = details["transient_titles"]
                if wait_problem == "ambiguous":
                    info["reason"] = "multiple ready windows match after startup"
                    info["candidate_titles"] = details.get("candidate_titles", [])
                    ambiguous.append(info)
                else:
                    pending.append(info)
                continue
            current = capture_workspace_windows()["windows"]
            problem = None

        if match is None and problem == "ambiguous":
            info = _window_identity_summary(saved)
            info["reason"] = "multiple current windows match the saved application but not the saved title"
            info["candidate_titles"] = [(w.get("title") or "<untitled>") for w in candidates]
            ambiguous.append(info)
            continue

        if match is None and problem is None and launch_missing:
            baseline_hwnds = {int(window["hwnd"]) for window in current}
            try:
                launch_label = _launch_missing_window(saved)
            except Exception as exc:
                info = _window_identity_summary(saved)
                info["reason"] = f"launch failed: {type(exc).__name__}: {exc}"
                pre_failures.append(info)
                continue

            if launch_label:
                launched.append(launch_label)
                match, wait_problem, details = _wait_for_ready_window(
                    saved,
                    used,
                    timeout_seconds,
                    baseline_hwnds=baseline_hwnds,
                )
                current = capture_workspace_windows()["windows"]
                if match is None:
                    info = _window_identity_summary(saved)
                    if wait_problem == "ambiguous":
                        info["reason"] = "reopened application produced multiple matching main windows"
                        info["candidate_titles"] = details.get("candidate_titles", [])
                        ambiguous.append(info)
                    else:
                        info["reason"] = (
                            "launch was dispatched, but no stable resizable main window appeared before timeout"
                            if provider_requires_resizable_main(provider_for_saved_window(saved))
                            else "launch was dispatched, but no stable matching window appeared before timeout"
                        )
                        if details.get("transient_titles"):
                            info["transient_titles"] = details["transient_titles"]
                        pending.append(info)
                    continue

        if match is None:
            info = _window_identity_summary(saved)
            provider = provider_for_saved_window(saved)
            if provider:
                info["reason"] = (
                    f"window is not open; provider {provider!r} has no usable launch path for this saved entry"
                )
            else:
                info["reason"] = "window is not open and has no usable saved resource, app alias, or provider"
            missing.append(info)
            continue

        used.add(int(match["hwnd"]))
        matched.append(
            {
                "hwnd": int(match["hwnd"]),
                "title": match.get("title"),
                "process_name": match.get("process_name"),
                "placement": saved["placement"],
                "saved": saved,
                "current_window": match,
            }
        )

    # Restore app-internal state before final geometry. Browser tab replacement,
    # VS Code editor reopening, and Explorer navigation stay internal to the
    # workspace provider layer and never become extra MCP tools.
    app_state_restored: list[dict] = []
    app_state_warnings: list[dict] = []
    for item in matched:
        restored_state, warning = restore_window_app_state(item["saved"], item["current_window"])
        if restored_state:
            app_state_restored.append(
                {
                    "title": item.get("title"),
                    "process_name": item.get("process_name"),
                    **restored_state,
                }
            )
        if warning:
            app_state_warnings.append(
                {
                    "title": item.get("title"),
                    "process_name": item.get("process_name"),
                    "warning": warning,
                }
            )

    # Floating/minimized/maximized windows keep using the verified Win32 geometry
    # path. A supported group of saved arranged windows is different: recreate
    # Windows' native Snap topology first, then move the real shared Snap
    # dividers to the saved DWM-visible pixel coordinates. This preserves the
    # coupled divider behavior instead of merely making windows look adjacent.
    ordinary_items = [
        {"hwnd": item["hwnd"], "placement": item["placement"]}
        for item in matched
        if str(item["placement"].get("show_state") or "normal") != "arranged"
    ]
    arranged_items = [
        {"hwnd": item["hwnd"], "placement": item["placement"]}
        for item in matched
        if str(item["placement"].get("show_state") or "normal") == "arranged"
    ]

    batches: list[dict] = []
    snap_warning: str | None = None

    if ordinary_items:
        batches.append(restore_window_placements_batch(ordinary_items))

    if len(arranged_items) >= 2:
        native_snap = restore_native_snap_layout(arranged_items)
        if native_snap.get("complete"):
            batches.append(
                {
                    "restored": native_snap.get("restored", []),
                    "failed": [],
                }
            )
        else:
            # Native Snap is best-effort for the currently supported topology.
            # If Windows/Shell input is unavailable or the layout is unsupported,
            # repair the windows through the old exact geometry path so workspace
            # restore still degrades gracefully rather than leaving a half-mutated
            # desktop.
            batches.append(restore_window_placements_batch(arranged_items))
            reasons = native_snap.get("failed") or []
            detail = "; ".join(
                str(item.get("reason") or item) if isinstance(item, dict) else str(item)
                for item in reasons[:3]
            ) or "native Snap restoration was unavailable"
            snap_warning = (
                "Exact workspace geometry was restored, but native coupled Snap "
                f"behavior could not be reconstructed: {detail}"
            )
    elif arranged_items:
        # A lone arranged window has no saved group relationship to reconstruct.
        batches.append(restore_window_placements_batch(arranged_items))

    batch_restored = [item for batch in batches for item in batch.get("restored", [])]
    batch_failed = [item for batch in batches for item in batch.get("failed", [])]

    by_hwnd = {item["hwnd"]: item for item in matched}
    restored: list[dict] = []
    failed: list[dict] = list(pre_failures)
    warnings: list[dict] = list(app_state_warnings)

    for result in batch_restored:
        identity = by_hwnd.get(result["hwnd"], {})
        restored.append(
            {
                "hwnd": result["hwnd"],
                "title": identity.get("title"),
                "process_name": identity.get("process_name"),
                "state": result.get("state"),
            }
        )
        if result.get("warning"):
            warnings.append(
                {
                    "title": identity.get("title"),
                    "process_name": identity.get("process_name"),
                    "warning": result["warning"],
                }
            )

    if snap_warning:
        warnings.append({"warning": snap_warning})

    for result in batch_failed:
        identity = by_hwnd.get(result.get("hwnd"), {})
        info = {
            "hwnd": result.get("hwnd"),
            "title": identity.get("title"),
            "process_name": identity.get("process_name"),
            "reason": result.get("error") or result.get("reason") or "window restore failed",
        }
        for key in ("coordinate_space", "requested", "observed"):
            if key in result:
                info[key] = result[key]
        failed.append(info)

    degraded = bool(snap_warning or app_state_warnings)
    return {
        "status": "ok",
        "operation": "restore_workspace",
        "name": workspace["name"],
        "restored": restored,
        "restored_count": len(restored),
        "launched": launched,
        "missing": missing,
        "pending": pending,
        "ambiguous": ambiguous,
        "failed": failed,
        "warnings": warnings,
        "app_state_restored": app_state_restored,
        "app_state_restored_count": len(app_state_restored),
        "degraded": degraded,
        # Once native Snap semantics are part of a saved arranged workspace, an
        # exact-geometry fallback is useful damage control but is not a complete
        # restore. Keep the contract honest so Dexter does not report "Done"
        # when the windows merely look adjacent.
        "complete": not missing and not pending and not ambiguous and not failed and not degraded,
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


def rename_workspace(query: str, new_name: str) -> dict:
    """Rename one saved workspace layout without recapturing it."""
    clean_new_name = _validate_name(new_name)
    now = _utc_now()

    def mutate(state: dict) -> dict:
        record = _resolve_workspace(query, state)
        old_name = record["name"]
        for workspace_id, other in state["workspaces"].items():
            if workspace_id == record["id"]:
                continue
            if _normalize(other["name"]) == _normalize(clean_new_name):
                raise FileExistsError(f"Saved workspace already exists: {clean_new_name!r}.")

        record["name"] = clean_new_name
        record["updated_utc"] = now
        return {
            "status": "ok",
            "operation": "rename_workspace",
            "id": record["id"],
            "old_name": old_name,
            "name": clean_new_name,
            "renamed": _normalize(old_name) != _normalize(clean_new_name),
        }

    return mutate_state(mutate)

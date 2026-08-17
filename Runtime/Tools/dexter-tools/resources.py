"""Named operational resources for Dexter's persistent computer map."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from config import RESOURCE_APP_ALIASES
from filesystem import resolve_allowed_path
from launcher import (
    launch_app_with_trusted_arguments,
    open_allowed_file,
    open_allowed_folder,
    open_http_url,
)
from saved_name_resolution import rank_named_records, resolve_named_record
from state_store import load_state, mutate_state
from vscode_state import GENERATED, current_vscode_session, vscode_open_target

ResourceKind = Literal["url", "folder", "file", "vscode", "visual_studio", "unity"]
RESOURCE_KINDS = {"url", "folder", "file", "vscode", "visual_studio", "unity"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _validate_label(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    clean = " ".join(value.strip().split())
    if len(clean) > 160:
        raise ValueError(f"{field} is too long; limit is 160 characters.")
    return clean


def _validate_aliases(aliases: list[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for alias in aliases or []:
        clean = _validate_label(alias, "alias")
        key = _normalize(clean)
        if key not in seen:
            result.append(clean)
            seen.add(key)
    if len(result) > 20:
        raise ValueError("A resource may have at most 20 aliases.")
    return result


def _validate_http_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")
    clean = url.strip()
    if len(clean) > 8192:
        raise ValueError("URL is unreasonably long.")
    parsed = urlparse(clean)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Saved URLs must be complete http:// or https:// URLs.")
    return clean


def _validate_target(kind: str, target: str) -> str:
    if kind == "url":
        return _validate_http_url(target)

    resolved = resolve_allowed_path(target)
    if not resolved.exists():
        raise FileNotFoundError(f"Saved resource target does not exist: {resolved}")

    if kind == "folder":
        if not resolved.is_dir():
            raise ValueError(f"Saved folder target is not a directory: {resolved}")
    elif kind == "file":
        if not resolved.is_file():
            raise ValueError(f"Saved file target is not a regular file: {resolved}")
    elif kind == "vscode":
        if not resolved.is_dir() and resolved.suffix.casefold() != ".code-workspace":
            raise ValueError("A VS Code resource must be a folder or .code-workspace file.")
        if resolved.is_file() and resolved.suffix.casefold() != ".code-workspace":
            raise ValueError("A VS Code workspace file must end in .code-workspace.")
    elif kind == "visual_studio":
        if not resolved.is_file() or resolved.suffix.casefold() not in {".sln", ".slnx"}:
            raise ValueError("A Visual Studio resource must be an existing .sln or .slnx file.")
    elif kind == "unity":
        if not resolved.is_dir():
            raise ValueError("A Unity resource must be a project directory.")
        marker = resolved / "ProjectSettings" / "ProjectVersion.txt"
        if not marker.is_file():
            raise ValueError(
                "Directory does not look like a Unity project; "
                "ProjectSettings\\ProjectVersion.txt was not found."
            )
    else:
        raise ValueError(f"Unsupported resource kind: {kind!r}")

    return str(resolved)


def _all_identity_keys(record: dict) -> set[str]:
    return {_normalize(record["name"]), *(_normalize(v) for v in record.get("aliases", []))}


def _resolve_resource_record(query: str, state: dict | None = None) -> dict:
    clean = _validate_label(query, "resource name")
    records = list((state or load_state())["resources"].values())
    return resolve_named_record(clean, records, record_label="resource")


def _remember_validated_resource(
    clean_name: str,
    clean_kind: str,
    clean_target: str,
    clean_aliases: list[str],
    overwrite: bool,
) -> dict:
    """Persist a resource after its type-specific target has been verified."""
    requested_keys = {_normalize(clean_name), *(_normalize(a) for a in clean_aliases)}
    if len(requested_keys) != 1 + len(clean_aliases):
        raise ValueError("Resource name and aliases must be distinct.")

    now = _utc_now()

    def mutate(state: dict) -> dict:
        resources = state["resources"]
        existing_id = None
        for resource_id, record in resources.items():
            if _normalize(record["name"]) == _normalize(clean_name):
                existing_id = resource_id
                break

        if existing_id is not None and not overwrite:
            raise FileExistsError(
                f"Saved resource already exists: {clean_name!r}. Use overwrite=true to replace it."
            )

        for resource_id, record in resources.items():
            if resource_id == existing_id:
                continue
            collision = requested_keys & _all_identity_keys(record)
            if collision:
                raise ValueError(
                    f"Resource name/alias conflicts with saved resource {record['name']!r}."
                )

        resource_id = existing_id or f"res_{uuid.uuid4().hex[:12]}"
        created_utc = resources.get(resource_id, {}).get("created_utc", now)
        resources[resource_id] = {
            "id": resource_id,
            "name": clean_name,
            "kind": clean_kind,
            "target": clean_target,
            "aliases": clean_aliases,
            "created_utc": created_utc,
            "updated_utc": now,
        }
        return {
            "status": "ok",
            "operation": "remember_resource",
            "resource": dict(resources[resource_id]),
            "created": existing_id is None,
            "updated": existing_id is not None,
        }

    return mutate_state(mutate)


def remember_resource(
    name: str,
    kind: ResourceKind,
    target: str,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    clean_name = _validate_label(name, "name")
    clean_kind = str(kind).strip().casefold()
    if clean_kind not in RESOURCE_KINDS:
        raise ValueError(f"Unsupported resource kind: {kind!r}")
    clean_aliases = _validate_aliases(aliases)
    clean_target = _validate_target(clean_kind, target)
    return _remember_validated_resource(
        clean_name,
        clean_kind,
        clean_target,
        clean_aliases,
        overwrite,
    )


def remember_current_vscode_project(
    name: str | None = None,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Remember the focused VS Code project using bridge-grounded identity."""
    session = current_vscode_session()
    folders = [
        str(folder).strip()
        for folder in session.get("workspace_folders") or []
        if isinstance(folder, str) and folder.strip()
    ]
    for folder in folders:
        resolved_folder = resolve_allowed_path(folder)
        if not resolved_folder.is_dir():
            raise ValueError(f"VS Code workspace folder is not a directory: {resolved_folder}")

    target = vscode_open_target(session)
    if not target:
        raise LookupError(
            "The active VS Code window has no folder or saved workspace file to remember."
        )
    candidate = Path(target).expanduser().resolve(strict=False)
    generated_root = GENERATED.expanduser().resolve(strict=False)
    if candidate.is_relative_to(generated_root):
        if not candidate.is_file():
            raise FileNotFoundError(f"Generated VS Code workspace was not created: {candidate}")
        clean_target = str(candidate)
    else:
        clean_target = _validate_target("vscode", str(candidate))

    workspace_name = str(session.get("workspace_name") or "").strip()
    inferred_names = [
        workspace_name,
        Path(clean_target).stem,
        *(Path(folder).name for folder in folders),
    ]
    if name is None:
        clean_name = next(
            (
                _validate_label(value, "name")
                for value in inferred_names
                if value and not value.casefold().startswith("dexter-")
            ),
            None,
        )
        if clean_name is None:
            raise ValueError(
                "Could not infer a stable project name; provide the name to remember."
            )
    else:
        clean_name = _validate_label(name, "name")

    alias_candidates = [*(aliases or []), *inferred_names]
    clean_aliases: list[str] = []
    seen = {_normalize(clean_name)}
    for value in alias_candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        clean = _validate_label(value, "alias")
        key = _normalize(clean)
        if key not in seen and not key.startswith("dexter-"):
            clean_aliases.append(clean)
            seen.add(key)
    clean_aliases = _validate_aliases(clean_aliases)

    result = _remember_validated_resource(
        clean_name,
        "vscode",
        clean_target,
        clean_aliases,
        overwrite,
    )
    result["source"] = {
        "session_id": session.get("session_id"),
        "workspace_name": workspace_name or None,
        "workspace_folders": folders,
        "active_file": session.get("active_file"),
    }
    return result


def search_resources(
    query: str | None = None,
    kind: ResourceKind | None = None,
    max_results: int = 10,
) -> dict:
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50.")
    if kind is not None and str(kind).casefold() not in RESOURCE_KINDS:
        raise ValueError(f"Unsupported resource kind: {kind!r}")

    state = load_state()
    records = list(state["resources"].values())
    records = [
        record
        for record in records
        if kind is None or record["kind"] == str(kind).casefold()
    ]
    if query:
        ranked = rank_named_records(query, records)
        ordered = [record for _score, record in ranked]
    else:
        ordered = sorted(records, key=lambda record: _normalize(record["name"]))
    selected = [dict(item) for item in ordered[:max_results]]
    return {
        "status": "ok",
        "operation": "search_resources",
        "resources": selected,
        "count": len(selected),
        "truncated": len(ordered) > max_results,
    }


def open_saved_resource(query: str) -> dict:
    record = _resolve_resource_record(query)
    kind = record["kind"]
    target = record["target"]

    if kind == "url":
        result = open_http_url(target)
        opened = bool(result.get("browser_accepted_request", False))
    elif kind == "folder":
        open_allowed_folder(target)
        opened = True
    elif kind == "file":
        open_allowed_file(target)
        opened = True
    elif kind == "vscode":
        alias = RESOURCE_APP_ALIASES.get("vscode", "vscode")
        launch_app_with_trusted_arguments(alias, ["--new-window", target])
        opened = True
    elif kind == "visual_studio":
        alias = RESOURCE_APP_ALIASES.get("visual_studio", "visual studio")
        launch_app_with_trusted_arguments(alias, [target])
        opened = True
    elif kind == "unity":
        alias = RESOURCE_APP_ALIASES.get("unity", "unity")
        launch_app_with_trusted_arguments(alias, ["-projectPath", target])
        opened = True
    else:
        raise RuntimeError(f"Unsupported saved resource kind: {kind!r}")

    return {
        "status": "ok",
        "operation": "open_resource",
        "id": record["id"],
        "name": record["name"],
        "kind": kind,
        "opened": opened,
    }


def forget_resource(query: str) -> dict:
    def mutate(state: dict) -> dict:
        record = _resolve_resource_record(query, state)
        del state["resources"][record["id"]]
        return {
            "status": "ok",
            "operation": "forget_resource",
            "id": record["id"],
            "name": record["name"],
            "deleted": True,
        }

    return mutate_state(mutate)


def resource_launch_candidates_for_window(window: dict) -> list[dict]:
    """Return saved project resources that can plausibly identify this window.

    Used only to add a safe launch hint to workspace snapshots. It deliberately
    avoids browser URLs because a window title is not a reliable URL identity.
    """
    process = (window.get("process_name") or "").casefold()
    title = (window.get("title") or "").casefold()
    if not process or not title:
        return []

    kind_processes = {
        "vscode": {"code.exe", "code"},
        "visual_studio": {"devenv.exe", "devenv"},
        "unity": {"unity.exe", "unity"},
    }

    results: list[dict] = []
    for record in load_state()["resources"].values():
        kind = record.get("kind")
        if kind not in kind_processes or process not in kind_processes[kind]:
            continue
        target_name = Path(record["target"]).stem.casefold()
        tokens = [record["name"], *record.get("aliases", []), target_name]
        if any(_normalize(token) and _normalize(token) in _normalize(title) for token in tokens):
            results.append(record)
    return results

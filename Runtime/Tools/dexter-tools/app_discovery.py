"""Internal Windows installed-app discovery for Dexter's open_app tool.

This module is deliberately not model-facing. The model supplies only a human app
name; trusted code discovers launchable Start entries, resolves a strong unique
match, and dispatches that OS-provided entry without constructing a shell command.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from config import (
    APP_DISCOVERY_CACHE_SECONDS,
    APP_DISCOVERY_POWERSHELL_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class AppCandidate:
    name: str
    source: str
    launch_target: str


_CACHE_AT = 0.0
_CACHE: tuple[AppCandidate, ...] = ()


class _NoAppMatch(LookupError):
    pass


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def _normalize_query(value: str) -> str:
    normalized = _normalize_name(value)
    tokens = normalized.split()
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    if len(tokens) > 1 and tokens[-1] in {"app", "application"}:
        tokens = tokens[:-1]
    return " ".join(tokens)


def _validate_requested_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("App name must be a non-empty string.")
    if "\x00" in name:
        raise ValueError("App name may not contain NUL characters.")
    if len(name) > 256:
        raise ValueError("App name is unreasonably long.")
    normalized = _normalize_query(name)
    if not normalized:
        raise ValueError("App name must contain letters or numbers.")
    return normalized


def _start_menu_roots() -> list[Path]:
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        roots.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    if programdata:
        roots.append(Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return roots


def _discover_start_menu_shortcuts() -> list[AppCandidate]:
    candidates: list[AppCandidate] = []
    seen_paths: set[str] = set()

    for root in _start_menu_roots():
        if not root.is_dir():
            continue
        try:
            paths = sorted(root.rglob("*.lnk"), key=lambda item: str(item).casefold())
        except OSError:
            continue

        for path in paths:
            try:
                if not path.is_file():
                    continue
            except OSError:
                continue

            key = os.path.normcase(os.path.abspath(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)

            name = " ".join(path.stem.split())
            if not _normalize_name(name):
                continue
            candidates.append(
                AppCandidate(
                    name=name,
                    source="start_menu_shortcut",
                    launch_target=str(path),
                )
            )
    return candidates


def _discover_start_apps() -> list[AppCandidate]:
    """Read Windows' current-user Start app catalog with one fixed PowerShell command."""
    command = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        "$apps=@(Get-StartApps | Select-Object Name,AppID); "
        "ConvertTo-Json -InputObject $apps -Compress -Depth 3"
    )
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=APP_DISCOVERY_POWERSHELL_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return []

    if completed.returncode != 0 or not completed.stdout.strip():
        return []

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return []

    candidates: list[AppCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("Name")
        app_id = row.get("AppID")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(app_id, str) or not app_id.strip():
            continue
        if "\x00" in app_id or len(app_id) > 4096:
            continue
        candidates.append(
            AppCandidate(
                name=" ".join(name.split()),
                source="start_apps",
                launch_target=app_id.strip(),
            )
        )
    return candidates


def _merge_candidates(
    shortcuts: list[AppCandidate],
    start_apps: list[AppCandidate],
) -> tuple[AppCandidate, ...]:
    """Prefer a real Start Menu shortcut when the display name is duplicated."""
    merged: dict[str, AppCandidate] = {}

    for candidate in shortcuts:
        key = _normalize_name(candidate.name)
        if key and key not in merged:
            merged[key] = candidate

    for candidate in start_apps:
        key = _normalize_name(candidate.name)
        if key and key not in merged:
            merged[key] = candidate

    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (_normalize_name(item.name), item.source, item.launch_target.casefold()),
        )
    )


def clear_app_discovery_cache() -> None:
    global _CACHE_AT, _CACHE
    _CACHE_AT = 0.0
    _CACHE = ()


def discover_installed_apps(force_refresh: bool = False) -> tuple[AppCandidate, ...]:
    if os.name != "nt":
        raise RuntimeError("Installed-app discovery is currently implemented for Windows only.")

    global _CACHE_AT, _CACHE
    now = time.monotonic()
    if (
        not force_refresh
        and _CACHE
        and now - _CACHE_AT < APP_DISCOVERY_CACHE_SECONDS
    ):
        return _CACHE

    candidates = _merge_candidates(
        _discover_start_menu_shortcuts(),
        _discover_start_apps(),
    )
    if not candidates:
        raise RuntimeError(
            "Windows app discovery returned no launchable Start entries. "
            "Configured APP_ALIASES remain available as explicit overrides."
        )

    _CACHE = candidates
    _CACHE_AT = now
    return candidates


def _candidate_score(query: str, candidate: AppCandidate) -> float:
    candidate_name = _normalize_name(candidate.name)
    if not candidate_name:
        return 0.0
    if query == candidate_name:
        return 100.0

    query_tokens = query.split()
    candidate_tokens = candidate_name.split()

    if candidate_name.startswith(query + " "):
        return 94.0 - min(6.0, 0.5 * (len(candidate_tokens) - len(query_tokens)))

    if f" {query} " in f" {candidate_name} ":
        return 91.0 - min(6.0, 0.5 * (len(candidate_tokens) - len(query_tokens)))

    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    if query_set and query_set.issubset(candidate_set):
        return 88.0 - min(8.0, 1.0 * (len(candidate_set) - len(query_set)))

    if candidate_set and candidate_set.issubset(query_set):
        return 84.0 - min(8.0, 1.0 * (len(query_set) - len(candidate_set)))

    ratio = SequenceMatcher(None, query, candidate_name).ratio()
    if ratio >= 0.78:
        return 60.0 + 30.0 * ratio

    if len(query_tokens) == 1 and len(query_tokens[0]) >= 4 and query_tokens[0] in candidate_set:
        return 82.0

    return 0.0


def _pick_candidate(query: str, candidates: tuple[AppCandidate, ...]) -> AppCandidate:
    scored = [
        (_candidate_score(query, candidate), candidate)
        for candidate in candidates
    ]
    scored = [item for item in scored if item[0] >= 78.0]
    scored.sort(
        key=lambda item: (
            -item[0],
            len(_normalize_name(item[1].name)),
            _normalize_name(item[1].name),
        )
    )

    if not scored:
        raise _NoAppMatch(query)

    best_score, best = scored[0]
    if best_score == 100.0:
        return best

    near_best = [
        candidate
        for score, candidate in scored
        if best_score - score < 4.0
    ]
    distinct_names = []
    seen_names: set[str] = set()
    for candidate in near_best:
        key = _normalize_name(candidate.name)
        if key not in seen_names:
            distinct_names.append(candidate.name)
            seen_names.add(key)

    if len(distinct_names) > 1:
        shown = ", ".join(distinct_names[:5])
        raise ValueError(
            f"App name is ambiguous. Strong matches for {query!r}: {shown}. "
            "Use a more specific app name or configure an APP_ALIASES override."
        )

    return best


def resolve_installed_app(name: str) -> AppCandidate:
    query = _validate_requested_name(name)

    candidates = discover_installed_apps(force_refresh=False)
    try:
        return _pick_candidate(query, candidates)
    except _NoAppMatch:
        refreshed = discover_installed_apps(force_refresh=True)
        try:
            return _pick_candidate(query, refreshed)
        except _NoAppMatch:
            ranked = sorted(
                (
                    (SequenceMatcher(None, query, _normalize_name(candidate.name)).ratio(), candidate.name)
                    for candidate in refreshed
                ),
                reverse=True,
            )
            suggestions = []
            seen: set[str] = set()
            for ratio, candidate_name in ranked:
                if ratio < 0.45:
                    break
                key = _normalize_name(candidate_name)
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(candidate_name)
                if len(suggestions) == 4:
                    break

            suffix = f" Closest installed names: {', '.join(suggestions)}." if suggestions else ""
            raise ValueError(
                f"No installed Windows app matched {name!r}.{suffix} "
                "If this is a custom/special launch target, configure an APP_ALIASES override."
            ) from None


def launch_installed_app(candidate: AppCandidate) -> dict:
    if os.name != "nt":
        raise RuntimeError("Installed-app launching is currently implemented for Windows only.")

    if candidate.source == "start_menu_shortcut":
        path = Path(candidate.launch_target)
        if not path.is_absolute() or not path.is_file() or path.suffix.casefold() != ".lnk":
            raise FileNotFoundError(f"Discovered Start Menu shortcut no longer exists: {path}")
        os.startfile(str(path))
        return {
            "status": "ok",
            "operation": "open_app",
            "app": candidate.name,
            "matched_name": candidate.name,
            "source": candidate.source,
            "pid": None,
            "process_alive_after_launch": None,
            "dispatched": True,
            "verified": False,
        }

    if candidate.source == "start_apps":
        windows_dir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        explorer = windows_dir / "explorer.exe"
        command = [
            str(explorer),
            rf"shell:AppsFolder\{candidate.launch_target}",
        ]
        proc = subprocess.Popen(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "status": "ok",
            "operation": "open_app",
            "app": candidate.name,
            "matched_name": candidate.name,
            "source": candidate.source,
            "pid": proc.pid,
            "process_alive_after_launch": None,
            "dispatched": True,
            "verified": False,
        }

    raise ValueError(f"Unsupported discovered-app source: {candidate.source!r}")

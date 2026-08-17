"""Dexter-owned, per-chat permission policy for Goose ACP tool calls.

The policy is deliberately independent of Qt and ACP generated classes.  It
uses the small duck-typed surface exposed by ACP so it can be regression tested
without starting Goose.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PermissionDecision(str, Enum):
    AUTO_ALLOW = "auto_allow"
    ASK_ONCE = "ask_once"
    ASK_FILE = "ask_file"


@dataclass(frozen=True, slots=True)
class PermissionEvaluation:
    decision: PermissionDecision
    tool_name: str | None
    file_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    path: Path
    trust_after_success: bool


# Every currently registered Dexter tool is classified explicitly.  Future
# tools are absent and therefore default to ASK_ONCE.
AUTO_APPROVED_TOOLS = frozenset(
    {
        "list_allowed_directories",
        "file_sha256",
        "file_info",
        "list_directory",
        "search_files",
        "search_file_contents",
        "read_file",
        "read_image",
        "create_directory",
        "copy_file",
        "move_file",
        "rename_file",
        "delete_file",
        "remember_resource",
        "remember_current_project",
        "search_saved",
        "open_resource",
        "save_workspace",
        "restore_workspace",
        "forget_saved",
        "list_app_aliases",
        "open_app",
        "open_file",
        "open_folder",
        "open_url",
        "get_clipboard",
        "set_clipboard",
        "list_processes",
        "is_process_running",
        "get_foreground_window",
        "list_windows",
        "get_window_bounds",
        "focus_window",
        "set_window_state",
        "close_window",
        "close_app",
        "close_all_windows_for_app",
        "close_every_app_on_active_desktop",
        "computer_action",
        "find_image",
        "wait_for_image",
        "start_click_job",
        "cancel_job",
        "job_status",
        "list_jobs",
        "web_search",
    }
)

FILE_WRITE_TARGETS = {
    "write_file": "path",
    "screenshot": "output_path",
}

_KNOWN_TOOLS = AUTO_APPROVED_TOOLS | FILE_WRITE_TARGETS.keys()
_TERMINAL_SUCCESS = frozenset({"completed", "complete", "succeeded", "success"})
_TERMINAL_FAILURE = frozenset({"failed", "error", "cancelled", "canceled"})


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalized_enum(value: Any) -> str:
    raw = _value(value, "value", value)
    text = str(raw or "").strip().casefold().replace("-", "_")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _raw_input(obj: Any) -> dict[str, Any]:
    value = _value(obj, "raw_input")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(decoded, dict):
            return decoded
    return {}


def tool_name_from_call(tool_call: Any) -> str | None:
    """Extract one exact registered tool name from an ACP tool-call title."""
    for attribute in ("name", "tool_name", "title"):
        value = str(_value(tool_call, attribute, "") or "").strip()
        if not value:
            continue
        stable = value.split(" · ", 1)[0].strip().casefold()
        normalized = re.sub(r"[^a-z0-9_]+", "_", stable).strip("_")
        candidates = [normalized]
        if "__" in normalized:
            candidates.append(normalized.rsplit("__", 1)[-1])
        # Goose 1.45 may render MCP calls as either
        # ``dexter_tools__close_all_windows_for_app`` or the humanized title
        # ``dexter tools: close all windows for app``. The latter normalizes to
        # a single-underscore server prefix, so handle that exact known prefix
        # without broadly suffix-matching arbitrary unknown tools.
        for prefix in ("dexter_tools__", "dexter_tools_"):
            if normalized.startswith(prefix):
                candidates.append(normalized[len(prefix):])
        for candidate in candidates:
            if candidate in _KNOWN_TOOLS:
                return candidate
    return None


class ToolPermissionPolicy:
    """Per-chat permission decisions and hash-verified file trust."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = Path(cwd).resolve()
        self._trusted_hashes: dict[str, str] = {}
        self._pending_writes: dict[str, _PendingWrite] = {}

    def reset(self) -> None:
        self._trusted_hashes.clear()
        self._pending_writes.clear()

    def evaluate(self, tool_call: Any) -> PermissionEvaluation:
        tool_name = tool_name_from_call(tool_call)
        if tool_name in AUTO_APPROVED_TOOLS:
            return PermissionEvaluation(PermissionDecision.AUTO_ALLOW, tool_name)
        if tool_name not in FILE_WRITE_TARGETS:
            return PermissionEvaluation(PermissionDecision.ASK_ONCE, tool_name)

        path = self._file_target(tool_call, tool_name)
        if path is None:
            return PermissionEvaluation(PermissionDecision.ASK_ONCE, tool_name)

        call_id = self._call_id(tool_call)
        if not path.exists():
            self._remember_pending(call_id, path, trust_after_success=True)
            return PermissionEvaluation(PermissionDecision.AUTO_ALLOW, tool_name, path)

        # Both current file writers refuse an existing target unless the
        # schema's explicit overwrite flag is true. A guaranteed refusal is
        # not a modification and does not need an approval card.
        if _raw_input(tool_call).get("overwrite") is not True:
            return PermissionEvaluation(PermissionDecision.AUTO_ALLOW, tool_name, path)

        key = self._path_key(path)
        expected = self._trusted_hashes.get(key)
        observed = self._try_sha256(path)
        if expected is not None and observed is not None and expected == observed:
            self._remember_pending(call_id, path, trust_after_success=True)
            return PermissionEvaluation(PermissionDecision.AUTO_ALLOW, tool_name, path)

        # Any out-of-band change revokes silent trust until the user approves
        # the file again.
        self._trusted_hashes.pop(key, None)
        return PermissionEvaluation(PermissionDecision.ASK_FILE, tool_name, path)

    def approve_file_for_chat(self, path: Path, tool_call_id: str | None) -> None:
        resolved = self._resolve_path(path)
        observed = self._try_sha256(resolved)
        if observed is not None:
            self._trusted_hashes[self._path_key(resolved)] = observed
        self._remember_pending(tool_call_id, resolved, trust_after_success=True)

    def observe_tool_update(self, update: Any) -> None:
        """Track successful Dexter-created/written files from ACP updates."""
        candidates = [
            update,
            _value(update, "tool_call"),
            _value(update, "content"),
        ]
        call_id = next((self._call_id(item) for item in candidates if item), None)
        if not call_id:
            return

        status = next(
            (
                _normalized_enum(_value(item, "status"))
                for item in candidates
                if item is not None and _value(item, "status") is not None
            ),
            "",
        )

        # Tool-call updates are observable even when Goose did not ask for
        # permission.  This is how a newly created file becomes trusted for
        # later edits in the same chat.
        if call_id not in self._pending_writes:
            for item in candidates:
                if item is None:
                    continue
                tool_name = tool_name_from_call(item)
                if tool_name not in FILE_WRITE_TARGETS:
                    continue
                path = self._file_target(item, tool_name)
                if path is not None and not path.exists():
                    self._remember_pending(call_id, path, trust_after_success=True)
                break

        if status in _TERMINAL_SUCCESS:
            pending = self._pending_writes.pop(call_id, None)
            if pending and pending.trust_after_success:
                observed = self._try_sha256(pending.path)
                if observed is not None:
                    self._trusted_hashes[self._path_key(pending.path)] = observed
        elif status in _TERMINAL_FAILURE:
            self._pending_writes.pop(call_id, None)

    @staticmethod
    def _call_id(tool_call: Any) -> str | None:
        if tool_call is None:
            return None
        for name in ("tool_call_id", "id"):
            value = str(_value(tool_call, name, "") or "").strip()
            if value:
                return value
        return None

    def _file_target(self, tool_call: Any, tool_name: str) -> Path | None:
        value = _raw_input(tool_call).get(FILE_WRITE_TARGETS[tool_name])
        if not isinstance(value, str) or not value.strip():
            return None
        return self._resolve_path(Path(value.strip()).expanduser())

    def _resolve_path(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else self._cwd / path
        return candidate.resolve(strict=False)

    @staticmethod
    def _path_key(path: Path) -> str:
        return os.path.normcase(str(path))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _try_sha256(cls, path: Path) -> str | None:
        if not path.is_file():
            return None
        try:
            return cls._sha256(path)
        except OSError:
            return None

    def _remember_pending(
        self,
        call_id: str | None,
        path: Path,
        *,
        trust_after_success: bool,
    ) -> None:
        if call_id:
            self._pending_writes[call_id] = _PendingWrite(
                path=path,
                trust_after_success=trust_after_success,
            )

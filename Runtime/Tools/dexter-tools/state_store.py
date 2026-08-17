"""Small persistent store for Dexter's operational/computer memory.

This is intentionally separate from conversational or biographical memory.
Records are JSON documents grouped into typed namespaces inside one local file.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from config import COMPUTER_MEMORY_PATH

_STORE_VERSION = 1
_LOCK = threading.RLock()


def _empty_state() -> dict[str, Any]:
    return {
        "version": _STORE_VERSION,
        "resources": {},
        "workspaces": {},
    }


def _validate_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Dexter computer memory is not a JSON object.")
    if value.get("version") != _STORE_VERSION:
        raise ValueError(
            f"Unsupported Dexter computer-memory version: {value.get('version')!r}. "
            f"Expected {_STORE_VERSION}."
        )
    for namespace in ("resources", "workspaces"):
        if not isinstance(value.get(namespace), dict):
            raise ValueError(f"Dexter computer memory has invalid {namespace!r} data.")
    return value


def load_state() -> dict[str, Any]:
    """Return a defensive copy of the current operational state."""
    path = Path(COMPUTER_MEMORY_PATH)
    with _LOCK:
        if not path.exists():
            return _empty_state()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read Dexter computer memory: {path}") from exc
        return copy.deepcopy(_validate_state(raw))


def _write_state_locked(state: dict[str, Any]) -> None:
    state = _validate_state(state)
    path = Path(COMPUTER_MEMORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")
    payload = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def mutate_state(mutator: Callable[[dict[str, Any]], Any]) -> Any:
    """Atomically read, mutate, and replace the state file under one process lock."""
    with _LOCK:
        current = load_state()
        result = mutator(current)
        _write_state_locked(current)
        return result

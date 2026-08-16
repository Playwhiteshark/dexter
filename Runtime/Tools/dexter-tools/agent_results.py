"""Pure adapters from proof-rich deterministic results to concise agent results."""

from __future__ import annotations

from typing import Any


_ENVELOPE_KEYS = {"status", "operation", "verified"}


def fields(result: dict[str, Any], *names: str) -> dict[str, Any]:
    """Return selected useful fields, omitting absent/None values."""
    return {name: result[name] for name in names if name in result and result[name] is not None}


def without_envelope(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in _ENVELOPE_KEYS}


def file_info(result: dict[str, Any]) -> dict[str, Any]:
    return fields(
        result,
        "path", "name", "type", "size_bytes", "modified_utc", "readable", "writable",
    )


def list_directory(result: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in result.get("entries", []):
        compact = fields(item, "name", "path", "type", "size_bytes", "modified_utc", "error")
        entries.append(compact)
    return {
        "directory": result.get("directory"),
        "entries": entries,
        "truncated": bool(result.get("truncated", False)),
    }


def search_files(result: dict[str, Any]) -> dict[str, Any]:
    matches = [fields(item, "path", "name", "type") for item in result.get("matches", [])]
    return {"matches": matches, "truncated": bool(result.get("truncated", False))}


def search_file_contents(result: dict[str, Any]) -> dict[str, Any]:
    matches = [
        fields(item, "path", "line_number", "line", "encoding")
        for item in result.get("matches", [])
    ]
    return {"matches": matches, "truncated": bool(result.get("truncated", False))}


def read_file(result: dict[str, Any]) -> dict[str, Any]:
    return fields(
        result,
        "path", "content", "encoding", "start_line", "end_line", "total_lines", "truncated",
    )



def write_file(result: dict[str, Any]) -> dict[str, Any]:
    """Return only the useful conclusion of a verified text-file write."""
    return fields(
        result,
        "path", "created", "overwritten", "size_bytes",
    )


def processes(result: dict[str, Any]) -> dict[str, Any]:
    items = [fields(item, "pid", "name", "exe", "username") for item in result.get("processes", [])]
    return {"processes": items, "truncated": bool(result.get("truncated", False))}


def foreground_window(result: dict[str, Any]) -> dict[str, Any]:
    window = result.get("window")
    if window is None:
        return {"window": None}
    if not isinstance(window, dict):
        return {"window": window}
    return fields(window, "hwnd", "title", "pid", "process_name", "bounds")


def windows(result: dict[str, Any]) -> dict[str, Any]:
    compact = [fields(item, "hwnd", "title", "pid", "process_name", "bounds") for item in result.get("windows", [])]
    return {"windows": compact, "truncated": bool(result.get("truncated", False))}


def job_snapshot(item: dict[str, Any]) -> dict[str, Any]:
    return fields(
        item,
        "job_id", "kind", "state", "runs", "stop_reason", "error",
    )


def job(result: dict[str, Any]) -> dict[str, Any]:
    item = result.get("job") if isinstance(result.get("job"), dict) else result
    return job_snapshot(item)


def jobs(result: dict[str, Any]) -> dict[str, Any]:
    return {"jobs": [job_snapshot(item) for item in result.get("jobs", [])]}

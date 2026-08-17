"""Safe MCP server for real-Goose Dexter benchmarks.

The benchmark must expose the same schemas as production without letting an
unexpected model choice click, close, launch, or mutate the real computer.  We
therefore reuse the already-registered production Tool objects and replace only
their callables inside this benchmark process:

- read-only tools execute their real implementation;
- effectful tools return small simulated success results;
- every call is appended to a JSONL trace.

This file is test infrastructure only.  Production ``server.py`` has no
benchmark branches or test-mode switches.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import config
import server


TRACE_ENV = "DEXTER_BENCHMARK_TRACE"
BENCHMARK_FIXTURE_ROOT_ENV = "DEXTER_BENCHMARK_FIXTURE_ROOT"
_SIMULATED_JOB_ID = "benchmark-job"
_SIMULATED_JOB_STATE = "running"

# These tools are read-only from the model's perspective but depend on state
# created by start_click_job, which is intentionally simulated in benchmarks.
_STATEFUL_SIMULATED_TOOLS = {"job_status", "list_jobs"}


def _trace_path() -> Path:
    value = os.environ.get(TRACE_ENV)
    if not value:
        raise RuntimeError(f"{TRACE_ENV} must point to the benchmark JSONL trace file.")
    path = Path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(value: Any) -> Any:
    data = getattr(value, "data", None)
    image_format = getattr(value, "format", None)
    if isinstance(data, bytes) and image_format is not None:
        return {
            "type": type(value).__name__,
            "format": str(image_format),
            "byte_length": len(data),
            "signature_hex": data[:8].hex(),
        }
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__}


def _append_event(event: str, **details: Any) -> None:
    record = {"time": time.time(), "event": event, **_json_safe(details)}
    with _trace_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_trace(tool: str, arguments: dict[str, Any], *, result: Any = None, error: Exception | None = None) -> None:
    record: dict[str, Any] = {
        "time": time.time(),
        "event": "tool_finished",
        "session_id": os.environ.get("AGENT_SESSION_ID"),
        "tool": tool,
        "arguments": _json_safe(arguments),
    }
    if error is None:
        record["result"] = _json_safe(result)
    else:
        record["error"] = f"{type(error).__name__}: {error}"

    with _trace_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _simulate(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return just enough state for the agent to continue after a safe simulation."""
    global _SIMULATED_JOB_STATE

    if tool == "create_directory":
        return {"path": arguments.get("path"), "created": True, "already_existed": False}
    if tool in {"copy_file", "move_file"}:
        return {"destination": arguments.get("destination")}
    if tool == "rename_file":
        path = Path(str(arguments.get("path", "")))
        return {"destination": str(path.with_name(str(arguments.get("new_name", path.name))))}
    if tool == "delete_file":
        return {"path": arguments.get("path"), "method": "simulated"}
    if tool == "open_app":
        return {"pid": 4242, "process_alive_after_launch": True}
    if tool in {"open_file", "open_folder", "open_url", "open_resource"}:
        return {
            "opened": True,
            "mode": arguments.get("mode", "default"),
        }
    if tool == "remember_current_project":
        return {
            "name": arguments.get("name") or "Current Project",
            "kind": "vscode",
            "target": r"C:\Work\CurrentProject",
            "aliases": arguments.get("aliases") or [],
            "created": True,
            "updated": False,
        }
    if tool == "rename_saved":
        return {
            "old_name": arguments.get("name"),
            "name": arguments.get("new_name"),
            "renamed": True,
        }
    if tool == "forget_saved":
        return {"name": arguments.get("name"), "deleted": True}
    if tool == "restore_workspace":
        return {
            "name": arguments.get("name"),
            "restored_count": 2,
            "complete": True,
        }
    if tool == "set_clipboard":
        return {"length": len(str(arguments.get("text", "")))}
    if tool == "focus_window":
        return {"focused": True}
    if tool == "close_window":
        return {"closed": True}
    if tool == "close_app":
        return {"process_exited": True, "remaining_windows": 0}
    if tool == "close_all_windows_for_app":
        return {
            "scope": "one_process_name_on_active_virtual_desktop",
            "process_name": arguments.get("process_name"),
            "closed_count": 3,
            "remaining_windows": [],
        }
    if tool == "close_every_app_on_active_desktop":
        return {
            "scope": "active_virtual_desktop",
            "closed_count": 3,
            "remaining_windows": [],
        }
    if tool == "computer_action":
        action = arguments.get("action")
        if action == "screen_size":
            return {"width": 1920, "height": 1080}
        if action == "cursor_position":
            return {"x": 960, "y": 540}
        return {"action": action, "completed": True}
    if tool == "screenshot":
        return {"path": arguments.get("output_path"), "width": 1920, "height": 1080}
    if tool == "start_click_job":
        _SIMULATED_JOB_STATE = "running"
        return {"job_id": _SIMULATED_JOB_ID, "state": _SIMULATED_JOB_STATE, "runs": 0}
    if tool == "cancel_job":
        _SIMULATED_JOB_STATE = "cancelled"
        return {
            "job_id": arguments.get("job_id", _SIMULATED_JOB_ID),
            "state": _SIMULATED_JOB_STATE,
            "runs": 0,
            "stop_reason": "cancelled_by_user",
        }
    if tool == "job_status":
        return {"job_id": arguments.get("job_id", _SIMULATED_JOB_ID), "state": _SIMULATED_JOB_STATE, "runs": 0}
    if tool == "list_jobs":
        return {"jobs": [{"job_id": _SIMULATED_JOB_ID, "state": _SIMULATED_JOB_STATE, "runs": 0}], "count": 1}

    # Fail closed for a newly added effectful tool until the benchmark explicitly
    # knows how to simulate a useful response for it.
    raise RuntimeError(f"Benchmark has no safe simulation for effectful tool: {tool}")


def _is_read_only(tool: Any) -> bool:
    annotations = getattr(tool, "annotations", None)
    return bool(annotations is not None and getattr(annotations, "read_only_hint", False))


def _wrap_sync(name: str, original: Callable[..., Any], execute_real: bool) -> Callable[..., Any]:
    def wrapped(**arguments: Any) -> Any:
        _append_event("tool_started", tool=name, arguments=arguments)
        try:
            result = original(**arguments) if execute_real else _simulate(name, arguments)
        except Exception as exc:
            _append_trace(name, arguments, error=exc)
            raise
        _append_trace(name, arguments, result=result)
        return result

    return wrapped


def _wrap_async(name: str, original: Callable[..., Any], execute_real: bool) -> Callable[..., Any]:
    async def wrapped(**arguments: Any) -> Any:
        _append_event("tool_started", tool=name, arguments=arguments)
        try:
            result = await original(**arguments) if execute_real else _simulate(name, arguments)
        except Exception as exc:
            _append_trace(name, arguments, error=exc)
            raise
        _append_trace(name, arguments, result=result)
        return result

    return wrapped


def _allow_fixture_root() -> None:
    value = os.environ.get(BENCHMARK_FIXTURE_ROOT_ENV)
    if not value:
        return
    root = Path(value).expanduser().resolve(strict=False)
    # Benchmark file tools must never inherit the user's normal allowed roots.
    # Keep the benchmark process restricted to generated fixtures only.
    config.ALLOWED_ROOTS[:] = [root]


def install_safe_wrappers() -> None:
    """Replace callables while preserving production schemas/metadata exactly."""
    manager = server.mcp._tool_manager  # pinned MCP 2.0 test harness API surface
    for tool in manager.list_tools():
        original = tool.fn
        execute_real = _is_read_only(tool) and tool.name not in _STATEFUL_SIMULATED_TOOLS
        if inspect.iscoroutinefunction(original):
            tool.fn = _wrap_async(tool.name, original, execute_real)
        else:
            tool.fn = _wrap_sync(tool.name, original, execute_real)


if __name__ == "__main__":
    _allow_fixture_root()
    install_safe_wrappers()
    _append_event(
        "server_started",
        tool_count=len(server.mcp._tool_manager.list_tools()),
        fixture_root=os.environ.get(BENCHMARK_FIXTURE_ROOT_ENV),
    )
    server.mcp.run()

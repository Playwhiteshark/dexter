from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Image as MCPImage
from mcp_types import ToolAnnotations

import agent_results as agent
from clipboard_tools import get_clipboard_text, set_clipboard_text
from computer_tools import ComputerAction, perform_computer_action
from config import (
    DEFAULT_CLIPBOARD_CHARS,
    DEFAULT_CONTENT_MATCHES,
    DEFAULT_CONTENT_MATCHES_PER_FILE,
    DEFAULT_LIST_ENTRIES,
    DEFAULT_PROCESS_RESULTS,
    DEFAULT_READ_CHARS,
    DEFAULT_READ_LINES,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_WINDOW_RESULTS,
    DEFAULT_SAVED_SEARCH_RESULTS,
    DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS,
)
from filesystem import (
    copy_file_verified,
    create_directory_verified,
    delete_file_verified,
    get_file_sha256,
    get_path_info,
    list_allowed_roots,
    list_directory_entries,
    move_file_verified,
    read_text_file,
    rename_file_verified,
    search_paths,
    search_text_contents,
    write_file_verified,
)
from image_tools import load_image_for_model
from jobs import (
    cancel_background_job,
    get_job_status,
    list_background_jobs,
    start_click_interval_job,
)
from launcher import (
    launch_app,
    list_app_aliases_configured,
    open_allowed_file,
    open_allowed_folder,
    open_http_url,
)
from process_tools import list_processes_info, process_running
from resources import (
    forget_resource,
    open_saved_resource,
    remember_resource as remember_resource_record,
    search_resources,
)
from screen_tools import capture_screenshot, find_image_on_screen, wait_for_image_on_screen
from windows_tools import (
    close_app_gracefully,
    close_window_gracefully,
    focus_window_verified,
    get_foreground_window_info,
    get_window_bounds_info,
    list_visible_windows,
    set_window_state_verified,
)
from workspace_tools import (
    forget_workspace,
    restore_workspace as restore_saved_workspace,
    save_workspace as save_current_workspace,
    search_workspaces,
)


READ_ONLY_LOCAL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
WRITE_LOCAL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
WRITE_IDEMPOTENT_LOCAL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
DESTRUCTIVE_LOCAL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=False,
)
OPEN_WORLD = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)

mcp = MCPServer(
    "Dexter Tools",
    description="Deterministic local-computer tools with configured filesystem and app boundaries.",
)


# Filesystem -----------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_allowed_directories() -> dict:
    """List filesystem roots Dexter may access."""
    return agent.fields(list_allowed_roots(), "roots")


@mcp.tool(annotations=READ_ONLY_LOCAL)
def file_sha256(path: str) -> dict:
    """Compute SHA-256 for one allowed file."""
    return agent.fields(get_file_sha256(path), "sha256")


@mcp.tool(annotations=READ_ONLY_LOCAL)
def file_info(path: str) -> dict:
    """Get metadata for one allowed file or directory."""
    return agent.file_info(get_path_info(path))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_directory(
    path: str,
    max_entries: int = DEFAULT_LIST_ENTRIES,
    include_hidden: bool = False,
) -> dict:
    """List one allowed directory; hidden entries are omitted by default."""
    return agent.list_directory(list_directory_entries(path, max_entries, include_hidden))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def search_files(
    root: str,
    pattern: str,
    extensions: list[str] | None = None,
    include_directories: bool = False,
    include_hidden: bool = False,
    max_results: int = DEFAULT_SEARCH_RESULTS,
) -> dict:
    """Recursively find file/directory names under an allowed root."""
    return agent.search_files(
        search_paths(root, pattern, extensions, include_directories, include_hidden, max_results)
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def search_file_contents(
    root: str,
    query: str,
    extensions: list[str] | None = None,
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = DEFAULT_CONTENT_MATCHES,
    max_matches_per_file: int = DEFAULT_CONTENT_MATCHES_PER_FILE,
) -> dict:
    """Search text contents under an allowed root and return bounded matching lines."""
    return agent.search_file_contents(
        search_text_contents(
            root,
            query,
            extensions,
            case_sensitive,
            include_hidden,
            max_results,
            max_matches_per_file,
        )
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def read_file(
    path: str,
    start_line: int = 1,
    max_lines: int = DEFAULT_READ_LINES,
    max_chars: int = DEFAULT_READ_CHARS,
) -> dict:
    """Read a bounded section of an allowed text file."""
    return agent.read_file(read_text_file(path, start_line, max_lines, max_chars))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def read_image(path: str) -> MCPImage:
    """Read an allowed local image for vision; common formats are normalized to PNG."""
    image = load_image_for_model(path)
    return MCPImage(data=image.data, format="png")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
) -> dict:
    """Create a UTF-8 text file at an exact allowed path.

    path is the complete file path, including the filename and extension when
    one is desired. Extensionless files such as Dockerfile and .gitignore are
    valid. The parent directory must already exist. Existing files are refused
    unless overwrite=true.
    """
    return agent.write_file(
        write_file_verified(path, content, overwrite)
    )


@mcp.tool(annotations=WRITE_IDEMPOTENT_LOCAL)
def create_directory(path: str) -> dict:
    """Create an allowed directory, including missing parents."""
    return agent.fields(create_directory_verified(path), "path", "created", "already_existed")


@mcp.tool(annotations=WRITE_LOCAL)
def copy_file(source: str, destination: str) -> dict:
    """Copy one allowed file without overwriting; verifies the copy internally."""
    return agent.fields(copy_file_verified(source, destination), "destination")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def move_file(source: str, destination: str) -> dict:
    """Move one allowed file without overwriting; verifies the move internally."""
    return agent.fields(move_file_verified(source, destination), "destination")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def rename_file(path: str, new_name: str) -> dict:
    """Rename one allowed file in place without overwriting."""
    return agent.fields(rename_file_verified(path, new_name), "destination")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def delete_file(path: str) -> dict:
    """Send one allowed file to the Windows Recycle Bin."""
    return agent.fields(delete_file_verified(path, permanent=False), "path", "method")


# Persistent computer map / workspaces --------------------------------------

@mcp.tool(annotations=WRITE_LOCAL)
def remember_resource(
    name: str,
    kind: Literal["url", "folder", "file", "vscode", "visual_studio", "unity"],
    target: str,
    aliases: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    """Remember a named URL, path, or development project for later lookup/opening."""
    result = remember_resource_record(name, kind, target, aliases, overwrite)
    resource = result["resource"]
    return {
        "name": resource["name"],
        "kind": resource["kind"],
        "created": result["created"],
        "updated": result["updated"],
    }


@mcp.tool(annotations=READ_ONLY_LOCAL)
def search_saved(
    query: str | None = None,
    category: Literal["all", "resources", "workspaces"] = "all",
    max_results: int = DEFAULT_SAVED_SEARCH_RESULTS,
) -> dict:
    """Search Dexter's saved operational resources and workspace layouts by name."""
    if category not in {"all", "resources", "workspaces"}:
        raise ValueError("category must be all, resources, or workspaces.")

    resources = []
    workspaces = []
    if category in {"all", "resources"}:
        resource_result = search_resources(query=query, max_results=max_results)
        resources = [
            agent.fields(item, "id", "name", "kind", "target", "aliases")
            for item in resource_result["resources"]
        ]
    if category in {"all", "workspaces"}:
        workspace_result = search_workspaces(query=query, max_results=max_results)
        workspaces = workspace_result["workspaces"]

    return {
        "resources": resources,
        "workspaces": workspaces,
    }


@mcp.tool(annotations=OPEN_WORLD)
def open_resource(name: str) -> dict:
    """Open one previously saved resource by its name or alias."""
    return agent.fields(open_saved_resource(name), "name", "kind", "opened")


@mcp.tool(annotations=WRITE_LOCAL)
def save_workspace(name: str, overwrite: bool = False) -> dict:
    """Save eligible current-desktop windows plus supported app state such as browser tabs, VS Code editors, Explorer folders, and Terminal cwd."""
    return agent.fields(
        save_current_workspace(name, overwrite),
        "name",
        "window_count",
        "captured_windows",
        "created",
        "updated",
        "app_state_window_count",
        "state_warnings",
    )


@mcp.tool(annotations=WRITE_LOCAL)
def restore_workspace(
    name: str,
    launch_missing: bool = True,
    timeout_seconds: float = DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS,
) -> dict:
    """Restore a saved Windows layout and its supported per-app state; safely relaunch known resources/apps when possible."""
    return agent.fields(
        restore_saved_workspace(name, launch_missing, timeout_seconds),
        "name",
        "restored_count",
        "launched",
        "missing",
        "pending",
        "ambiguous",
        "failed",
        "warnings",
        "app_state_restored",
        "app_state_restored_count",
        "degraded",
        "complete",
    )


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def forget_saved(
    category: Literal["resource", "workspace"],
    name: str,
) -> dict:
    """Delete one saved operational resource or workspace layout from Dexter's persistent memory."""
    if category == "resource":
        result = forget_resource(name)
    elif category == "workspace":
        result = forget_workspace(name)
    else:
        raise ValueError("category must be resource or workspace.")
    return agent.fields(result, "name", "deleted")


# Launching ------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_app_aliases() -> dict:
    """List optional configured open_app aliases/overrides; ordinary installed apps need no alias."""
    return agent.fields(list_app_aliases_configured(), "aliases")


@mcp.tool(annotations=WRITE_LOCAL)
def open_app(name: str, arguments: list[str] | None = None) -> dict:
    """Launch an installed Windows app by name; configured aliases are optional overrides."""
    result = launch_app(name, arguments)
    return agent.fields(result, "app", "source", "pid", "process_alive_after_launch")


@mcp.tool(annotations=WRITE_LOCAL)
def open_file(path: str) -> dict:
    """Open an allowed non-executable file with its Windows default app."""
    open_allowed_file(path)
    return {"opened": True}


@mcp.tool(annotations=WRITE_LOCAL)
def open_folder(path: str) -> dict:
    """Open one allowed folder in Windows Explorer."""
    open_allowed_folder(path)
    return {"opened": True}


@mcp.tool(annotations=OPEN_WORLD)
def open_url(url: str) -> dict:
    """Open an http:// or https:// URL in the default browser."""
    result = open_http_url(url)
    return {"opened": bool(result.get("browser_accepted_request", False))}


# Clipboard / processes / windows -------------------------------------------

@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_clipboard(max_chars: int = DEFAULT_CLIPBOARD_CHARS) -> dict:
    """Read bounded text from the local clipboard."""
    return agent.fields(get_clipboard_text(max_chars), "text", "length", "truncated")


@mcp.tool(annotations=WRITE_LOCAL)
def set_clipboard(text: str) -> dict:
    """Set local clipboard text and verify it round-tripped."""
    return agent.fields(set_clipboard_text(text), "length")


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_processes(query: str | None = None, max_results: int = DEFAULT_PROCESS_RESULTS) -> dict:
    """List local processes, optionally filtering name/path text."""
    return agent.processes(list_processes_info(query, max_results))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def is_process_running(pid: int | None = None, name: str | None = None) -> dict:
    """Check for a process by PID, exact process name, or both. On Windows, the .exe suffix is optional."""
    return agent.fields(process_running(pid, name), "running", "matches", "truncated")


@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_foreground_window() -> dict:
    """Return the current foreground window, process, and title."""
    return agent.foreground_window(get_foreground_window_info())


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_windows(
    query: str | None = None,
    max_results: int = DEFAULT_WINDOW_RESULTS,
    visible_only: bool = True,
) -> dict:
    """List top-level windows, optionally filtering title/process text."""
    return agent.windows(list_visible_windows(query, max_results, visible_only))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_window_bounds(hwnd: int) -> dict:
    """Get pixel bounds for a current window handle."""
    return agent.fields(get_window_bounds_info(hwnd), "bounds")


@mcp.tool(annotations=WRITE_LOCAL)
def focus_window(hwnd: int) -> dict:
    """Focus a window and verify Windows made it foreground."""
    focus_window_verified(hwnd)
    return {"focused": True}


@mcp.tool(annotations=WRITE_IDEMPOTENT_LOCAL)
def set_window_state(
    hwnd: int,
    state: Literal["normal", "minimized", "maximized"],
) -> dict:
    """Minimize, maximize, or restore/unminimize a window without closing it."""
    result = set_window_state_verified(hwnd, state)
    return agent.fields(result, "state")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def close_window(hwnd: int, timeout_seconds: float = 5.0) -> dict:
    """Gracefully request a window to close; never hard-kills its process."""
    return agent.fields(close_window_gracefully(hwnd, timeout_seconds), "closed", "note")


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def close_app(pid: int, timeout_seconds: float = 5.0) -> dict:
    """Gracefully close a process through its top-level windows; never hard-kills."""
    return agent.fields(
        close_app_gracefully(pid, timeout_seconds),
        "process_exited",
        "remaining_windows",
        "note",
    )


# Desktop control ------------------------------------------------------------

@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def computer_action(
    action: ComputerAction,
    x: int | None = None,
    y: int | None = None,
    text: str | None = None,
    key: str | None = None,
    keys: list[str] | None = None,
    amount: int | None = None,
    duration: float = 0.2,
) -> dict:
    """Perform one mouse/keyboard/wait action; screen_size and cursor_position are read-only actions."""
    return agent.without_envelope(
        perform_computer_action(
            action,
            x=x,
            y=y,
            text=text,
            key=key,
            keys=keys,
            amount=amount,
            duration=duration,
        )
    )


@mcp.tool(annotations=WRITE_LOCAL)
def screenshot(
    output_path: str,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Save the whole primary screen or a rectangular region to an allowed image path."""
    return agent.fields(
        capture_screenshot(output_path, x, y, width, height, overwrite),
        "path",
        "width",
        "height",
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def find_image(
    template_path: str,
    confidence: float = 0.90,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    grayscale: bool = False,
) -> dict:
    """Find a template image on the primary screen; absence is a normal result."""
    return agent.fields(
        find_image_on_screen(template_path, confidence, x, y, width, height, grayscale),
        "found",
        "match",
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def wait_for_image(
    template_path: str,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.50,
    confidence: float = 0.90,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    grayscale: bool = False,
) -> dict:
    """Poll for a template image until found or the timeout expires."""
    return agent.fields(
        wait_for_image_on_screen(
            template_path,
            timeout_seconds,
            poll_interval_seconds,
            confidence,
            x,
            y,
            width,
            height,
            grayscale,
        ),
        "found",
        "match",
        "timed_out",
    )


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def start_click_job(
    interval_seconds: float,
    position_mode: Literal["current", "fixed"] = "current",
    x: int | None = None,
    y: int | None = None,
    button: Literal["left", "right", "middle"] = "left",
    max_runs: int | None = None,
    max_seconds: float | None = None,
    stop_image_path: str | None = None,
    stop_image_confidence: float = 0.90,
) -> dict:
    """Start a cancellable background click job with optional stop conditions."""
    return agent.job(
        start_click_interval_job(
            interval_seconds=interval_seconds,
            position_mode=position_mode,
            x=x,
            y=y,
            button=button,
            max_runs=max_runs,
            max_seconds=max_seconds,
            stop_image_path=stop_image_path,
            stop_image_confidence=stop_image_confidence,
        )
    )


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def cancel_job(job_id: str, wait_seconds: float = 2.0) -> dict:
    """Cancel a Dexter background job and return its observed state."""
    return agent.job(cancel_background_job(job_id, wait_seconds))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def job_status(job_id: str) -> dict:
    """Get state, run count, stop reason, and error for one background job."""
    return agent.job(get_job_status(job_id))


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_jobs(include_finished: bool = False) -> dict:
    """List active jobs, optionally including finished jobs."""
    return agent.jobs(list_background_jobs(include_finished))


if __name__ == "__main__":
    mcp.run()

from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer

try:
    from mcp_types import ToolAnnotations
except ImportError:
    ToolAnnotations = None  # type: ignore[assignment]

from clipboard_tools import get_clipboard_text, set_clipboard_text
from computer_tools import ComputerAction, perform_computer_action
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
)
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
from screen_tools import (
    capture_screenshot,
    find_image_on_screen,
    wait_for_image_on_screen,
)
from windows_tools import (
    close_app_gracefully,
    close_window_gracefully,
    focus_window_verified,
    get_foreground_window_info,
    get_window_bounds_info,
    list_visible_windows,
)


def _annotations(
    *,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
):
    """Create MCP annotations while staying tolerant of SDK field aliases."""
    if ToolAnnotations is None:
        return None

    values = {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": idempotent,
        "openWorldHint": open_world,
    }

    try:
        return ToolAnnotations(**values)
    except Exception:
        # Some generated/Pydantic variants expose snake_case constructor names.
        try:
            return ToolAnnotations(
                read_only_hint=read_only,
                destructive_hint=destructive,
                idempotent_hint=idempotent,
                open_world_hint=open_world,
            )
        except Exception:
            return None


READ_ONLY_LOCAL = _annotations(read_only=True, open_world=False)
WRITE_LOCAL = _annotations(read_only=False, destructive=False, open_world=False)
WRITE_IDEMPOTENT_LOCAL = _annotations(
    read_only=False, destructive=False, idempotent=True, open_world=False
)
DESTRUCTIVE_LOCAL = _annotations(read_only=False, destructive=True, open_world=False)
OPEN_WORLD = _annotations(read_only=False, destructive=False, open_world=True)

mcp = MCPServer(
    "Dexter Tools",
    description=(
        "Deterministic local-computer tools for Dexter. "
        "Filesystem operations are restricted to configured roots; "
        "app launching is restricted to configured aliases."
    ),
)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def dexter_ping() -> dict[str, str]:
    """Check that the Dexter MCP tool server is reachable."""
    return {
        "status": "ok",
        "server": "dexter-tools",
        "message": "Dexter tools are reachable.",
    }


# ---------------------------------------------------------------------------
# Batch 1: filesystem + launching
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_allowed_directories() -> dict:
    """List the filesystem roots Dexter is permitted to access."""
    return list_allowed_roots()


@mcp.tool(annotations=READ_ONLY_LOCAL)
def file_sha256(path: str) -> dict:
    """Compute SHA-256 for one allowed regular file."""
    return get_file_sha256(path)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def file_info(path: str) -> dict:
    """Get metadata for one allowed file or directory."""
    return get_path_info(path)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_directory(
    path: str,
    max_entries: int = 200,
    include_hidden: bool = False,
) -> dict:
    """
    List one allowed directory.

    Returns structured entries and truncation information. Hidden entries are
    omitted by default to keep agent context small.
    """
    return list_directory_entries(path, max_entries, include_hidden)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def search_files(
    root: str,
    pattern: str,
    extensions: list[str] | None = None,
    include_directories: bool = False,
    include_hidden: bool = False,
    max_results: int = 100,
) -> dict:
    """
    Recursively search names under one allowed root.

    pattern is case-insensitive. Glob patterns such as *.py are supported;
    otherwise it is treated as a substring. Common dependency/VCS directories
    are skipped by default.
    """
    return search_paths(
        root,
        pattern,
        extensions,
        include_directories,
        include_hidden,
        max_results,
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def search_file_contents(
    root: str,
    query: str,
    extensions: list[str] | None = None,
    case_sensitive: bool = False,
    include_hidden: bool = False,
    max_results: int = 100,
    max_matches_per_file: int = 5,
) -> dict:
    """
    Search text file contents under one allowed root.

    Returns path, line number, and a bounded matching line. Binary and very
    large files are skipped rather than flooding context.
    """
    return search_text_contents(
        root,
        query,
        extensions,
        case_sensitive,
        include_hidden,
        max_results,
        max_matches_per_file,
    )


@mcp.tool(annotations=READ_ONLY_LOCAL)
def read_file(
    path: str,
    start_line: int = 1,
    max_lines: int = 400,
    max_chars: int = 50_000,
) -> dict:
    """
    Read a bounded section of an allowed text file.

    Binary/oversized files are rejected. Returns line range, encoding, total
    lines, and whether the result was truncated.
    """
    return read_text_file(path, start_line, max_lines, max_chars)


@mcp.tool(annotations=WRITE_IDEMPOTENT_LOCAL)
def create_directory(path: str) -> dict:
    """
    Create an allowed directory and any missing parents.

    Idempotent: if the directory already exists, reports that without changing it.
    """
    return create_directory_verified(path)


@mcp.tool(annotations=WRITE_LOCAL)
def copy_file(source: str, destination: str) -> dict:
    """
    Copy one allowed file without overwriting.

    Verifies destination existence, size, and SHA-256 equality.
    """
    return copy_file_verified(source, destination)


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def move_file(source: str, destination: str) -> dict:
    """
    Move one allowed file without overwriting.

    Verifies source disappearance plus destination size and SHA-256.
    """
    return move_file_verified(source, destination)


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def rename_file(path: str, new_name: str) -> dict:
    """
    Rename one allowed file in place.

    new_name must be only a filename, not a path. Refuses overwrite and verifies
    source disappearance plus destination hash/size.
    """
    return rename_file_verified(path, new_name)


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def delete_file(path: str) -> dict:
    """
    Send one allowed file to the Windows Recycle Bin.

    Verifies that the original path disappeared. Permanent deletion is not
    exposed to the agent.
    """
    return delete_file_verified(path, permanent=False)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_app_aliases() -> dict:
    """List app aliases that open_app is explicitly permitted to launch."""
    return list_app_aliases_configured()


@mcp.tool(annotations=WRITE_LOCAL)
def open_app(name: str, arguments: list[str] | None = None) -> dict:
    """
    Launch a configured app alias without invoking a shell.

    Arbitrary executables/commands are rejected; add trusted apps to APP_ALIASES.
    """
    return launch_app(name, arguments)


@mcp.tool(annotations=WRITE_LOCAL)
def open_file(path: str) -> dict:
    """Open one allowed file with its Windows default application."""
    return open_allowed_file(path)


@mcp.tool(annotations=WRITE_LOCAL)
def open_folder(path: str) -> dict:
    """Open one allowed folder in the Windows shell."""
    return open_allowed_folder(path)


@mcp.tool(annotations=OPEN_WORLD)
def open_url(url: str) -> dict:
    """Open an http:// or https:// URL in the default browser."""
    return open_http_url(url)


# ---------------------------------------------------------------------------
# Batch 2: clipboard + processes + windows
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_clipboard(max_chars: int = 50_000) -> dict:
    """Read bounded text from the local clipboard."""
    return get_clipboard_text(max_chars)


@mcp.tool(annotations=WRITE_LOCAL)
def set_clipboard(text: str) -> dict:
    """Set local clipboard text and verify the exact value round-tripped."""
    return set_clipboard_text(text)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_processes(query: str | None = None, max_results: int = 100) -> dict:
    """List local processes, optionally filtering by process name/path substring."""
    return list_processes_info(query, max_results)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def is_process_running(
    pid: int | None = None,
    name: str | None = None,
) -> dict:
    """Check whether a local process exists by PID, exact process name, or both."""
    return process_running(pid, name)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_foreground_window() -> dict:
    """Return the current Windows foreground window, process, title, and bounds."""
    return get_foreground_window_info()


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_windows(
    query: str | None = None,
    max_results: int = 50,
    visible_only: bool = True,
) -> dict:
    """List top-level Windows windows, optionally filtering title/process text."""
    return list_visible_windows(query, max_results, visible_only)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def get_window_bounds(hwnd: int) -> dict:
    """Read pixel bounds for a current Windows window handle."""
    return get_window_bounds_info(hwnd)


@mcp.tool(annotations=WRITE_LOCAL)
def focus_window(hwnd: int) -> dict:
    """
    Ask Windows to focus a window and verify it actually became foreground.

    Windows may refuse focus stealing; that is reported as a failure rather
    than pretending the focus changed.
    """
    return focus_window_verified(hwnd)


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def close_window(hwnd: int, timeout_seconds: float = 5.0) -> dict:
    """
    Gracefully request a window to close with WM_CLOSE.

    Does not hard-kill the process. Reports if a save/confirmation prompt keeps
    the window open.
    """
    return close_window_gracefully(hwnd, timeout_seconds)


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def close_app(pid: int, timeout_seconds: float = 5.0) -> dict:
    """
    Gracefully close all top-level windows belonging to a process.

    Refuses to hard-kill a headless/no-window process.
    """
    return close_app_gracefully(pid, timeout_seconds)


# ---------------------------------------------------------------------------
# Batch 3: desktop control + screen matching + background jobs
# ---------------------------------------------------------------------------

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
    """
    Perform one deterministic mouse/keyboard/wait action.

    Actions: screen_size, cursor_position, mouse_move, left_click, right_click,
    middle_click, double_click, drag_to, scroll, key, hotkey, type, wait.
    PyAutoGUI's corner fail-safe remains enabled.
    """
    return perform_computer_action(
        action,
        x=x,
        y=y,
        text=text,
        key=key,
        keys=keys,
        amount=amount,
        duration=duration,
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
    """
    Save the whole primary screen or a rectangular region to an allowed image path.

    Refuses overwrite by default and verifies that a non-empty image was saved.
    """
    return capture_screenshot(output_path, x, y, width, height, overwrite)


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
    """
    Find a template image on the primary screen.

    Returns found=false as a normal result when absent. confidence matching uses
    OpenCV via PyAutoGUI/PyScreeze.
    """
    return find_image_on_screen(
        template_path,
        confidence,
        x,
        y,
        width,
        height,
        grayscale,
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
    """
    Poll for a template image until found or timeout.

    A timeout is returned as found=false/timed_out=true, not raised as an error.
    """
    return wait_for_image_on_screen(
        template_path,
        timeout_seconds,
        poll_interval_seconds,
        confidence,
        x,
        y,
        width,
        height,
        grayscale,
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
    """
    Start a cancellable background auto-click job.

    position_mode='current' clicks wherever the cursor currently is on every
    interval without moving it. position_mode='fixed' anchors to x/y, or to the
    cursor position at job creation if x/y are omitted. Optional max_runs,
    max_seconds, or stop_image_path provide deterministic stop conditions.
    """
    return start_click_interval_job(
        interval_seconds=interval_seconds,
        position_mode=position_mode,  # validated inside
        x=x,
        y=y,
        button=button,  # validated inside
        max_runs=max_runs,
        max_seconds=max_seconds,
        stop_image_path=stop_image_path,
        stop_image_confidence=stop_image_confidence,
    )


@mcp.tool(annotations=DESTRUCTIVE_LOCAL)
def cancel_job(job_id: str, wait_seconds: float = 2.0) -> dict:
    """Cancel a Dexter background job and return its observed state."""
    return cancel_background_job(job_id, wait_seconds)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def job_status(job_id: str) -> dict:
    """Get current state, run count, stop reason, and errors for one background job."""
    return get_job_status(job_id)


@mcp.tool(annotations=READ_ONLY_LOCAL)
def list_jobs(include_finished: bool = False) -> dict:
    """List active Dexter background jobs, optionally including finished jobs."""
    return list_background_jobs(include_finished)


if __name__ == "__main__":
    mcp.run()

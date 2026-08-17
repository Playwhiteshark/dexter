from __future__ import annotations

from pathlib import Path
import os

try:
    import local_config as _local
except ModuleNotFoundError:
    _local = None

# Machine policy is intentionally local and deny-by-default. A partial local
# config is tolerated so adding roots does not accidentally require app aliases
# (or vice versa).
_LOCAL_ALLOWED_ROOTS = getattr(_local, "ALLOWED_ROOTS", []) if _local else []
_LOCAL_APP_ALIASES = getattr(_local, "APP_ALIASES", {}) if _local else {}
_LOCAL_RESOURCE_APP_ALIASES = getattr(_local, "RESOURCE_APP_ALIASES", {}) if _local else {}

ALLOWED_ROOTS = [Path(path).expanduser().resolve(strict=False) for path in _LOCAL_ALLOWED_ROOTS]
APP_ALIASES = dict(_LOCAL_APP_ALIASES)
RESOURCE_APP_ALIASES = {
    "vscode": "vscode",
    "visual_studio": "visual studio",
    "unity": "unity",
}
RESOURCE_APP_ALIASES.update(_LOCAL_RESOURCE_APP_ALIASES)

# Operational/computer memory is application state, not an allowed filesystem
# root and not conversational/biographical memory. Keep it outside the repo by
# default so it survives code updates and cannot be committed accidentally.
_DEFAULT_STATE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Dexter" / "state"
STATE_DIR = Path(getattr(_local, "STATE_DIR", _DEFAULT_STATE_DIR)).expanduser().resolve(strict=False) if _local else _DEFAULT_STATE_DIR
COMPUTER_MEMORY_PATH = STATE_DIR / "computer_memory.json"

WORKSPACE_EXCLUDED_WINDOW_TITLES = {"Dexter"}

# Normal model-facing defaults. Callers may explicitly request more up to the
# safety ceilings below.
DEFAULT_LIST_ENTRIES = 50
DEFAULT_SEARCH_RESULTS = 25
DEFAULT_CONTENT_MATCHES = 20
DEFAULT_CONTENT_MATCHES_PER_FILE = 3
DEFAULT_READ_LINES = 160
DEFAULT_READ_CHARS = 12_000
DEFAULT_CLIPBOARD_CHARS = 10_000
DEFAULT_PROCESS_RESULTS = 30
DEFAULT_WINDOW_RESULTS = 30
DEFAULT_SAVED_SEARCH_RESULTS = 10
DEFAULT_WORKSPACE_LAUNCH_TIMEOUT_SECONDS = 20.0

# open_file uses Windows file associations, so executable/script/shortcut types
# are blocked to avoid bypassing open_app's explicit allowlist.
DANGEROUS_OPEN_FILE_EXTENSIONS = {
    ".bat", ".cmd", ".com", ".cpl", ".exe", ".hta", ".jar", ".js", ".jse",
    ".lnk", ".msi", ".msp", ".ps1", ".py", ".pyw", ".reg", ".scr", ".url",
    ".vbe", ".vbs", ".wsf", ".wsh",
}

DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".idea", ".vs", ".vscode", ".venv",
    "venv", "env", "node_modules", "__pycache__",
}

# Process/safety ceilings, not normal response targets.
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200
MAX_CONTENT_MATCHES = 200
MAX_CONTENT_SEARCH_FILE_BYTES = 5 * 1024 * 1024
MAX_READ_FILE_BYTES = 8 * 1024 * 1024
MAX_READ_CHARS = 100_000
MAX_CLIPBOARD_CHARS = 100_000
MAX_TYPE_TEXT_CHARS = 20_000

ALLOW_PERMANENT_DELETE = False

MAX_BACKGROUND_JOBS = 16
MIN_CLICK_INTERVAL_SECONDS = 0.10
MAX_WAIT_SECONDS = 3600.0

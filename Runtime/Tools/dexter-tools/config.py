from pathlib import Path

# Filesystem access is deny-by-default: every file/directory tool resolves the
# requested path and requires it to remain under one of these roots.
ALLOWED_ROOTS = [
    Path(r"C:\Users\ahmad\COMPUTER STUFF\Self-hosted AI\Jarvis Sandbox").resolve()
]

# open_app never accepts an arbitrary command line. The model can launch only
# aliases explicitly registered here. Add more apps as Dexter learns them.
APP_ALIASES = {
    "notepad": {
        "command": [r"C:\Windows\System32\notepad.exe"],
        "allow_arguments": False,
    },
    "file explorer": {
        "command": [r"C:\Windows\explorer.exe"],
        "allow_arguments": False,
    },
    "explorer": {
        "command": [r"C:\Windows\explorer.exe"],
        "allow_arguments": False,
    },
}

# open_file uses the Windows file association, so executable/script/shortcut
# types are blocked. Otherwise a file tool could accidentally become a generic
# code-execution tool and bypass open_app's allowlist.
DANGEROUS_OPEN_FILE_EXTENSIONS = {
    ".bat", ".cmd", ".com", ".cpl", ".exe", ".hta", ".jar", ".js", ".jse",
    ".lnk", ".msi", ".msp", ".ps1", ".py", ".pyw", ".reg", ".scr", ".url",
    ".vbe", ".vbs", ".wsf", ".wsh",
}

# Search defaults deliberately skip noisy dependency/VCS trees. The caller can
# opt into hidden directories when it really needs them.
DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vs",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
}

MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 200
MAX_CONTENT_MATCHES = 200
MAX_CONTENT_SEARCH_FILE_BYTES = 5 * 1024 * 1024
MAX_READ_FILE_BYTES = 8 * 1024 * 1024
MAX_READ_CHARS = 100_000
MAX_CLIPBOARD_CHARS = 100_000
MAX_TYPE_TEXT_CHARS = 20_000

# Destructive deletion is deliberately disabled. delete_file sends files to the
# Recycle Bin by default. Flip this only if a later Dexter policy explicitly
# supports permanent deletion.
ALLOW_PERMANENT_DELETE = False

MAX_BACKGROUND_JOBS = 16
MIN_CLICK_INTERVAL_SECONDS = 0.10
MAX_WAIT_SECONDS = 3600.0

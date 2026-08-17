# Copy this file to local_config.py, then add only machine-local policy you trust.
# The committed example is intentionally deny-by-default and contains no user-specific paths.
ALLOWED_ROOTS = [
    # r"C:\path\to\trusted\root",
]

# Optional open_app overrides/special launch targets. Ordinary installed Windows
# apps are discovered automatically by name; aliases are not required for them.
# Alias commands still never invoke a shell, and arguments remain denied by default.
APP_ALIASES = {
    # "example app": {
    #     "command": [r"C:\path\to\app.exe"],
    #     "allow_arguments": False,
    # },
}

# Optional aliases used by saved project resources. The values refer to keys
# in APP_ALIASES. Project paths themselves are still checked against ALLOWED_ROOTS.
RESOURCE_APP_ALIASES = {
    "vscode": "vscode",
    "visual_studio": "visual studio",
    "unity": "unity",
}

# Optional persistent-state location. If omitted, Dexter uses %LOCALAPPDATA%\Dexter\state on Windows.
# STATE_DIR = r"C:\path\to\DexterState"

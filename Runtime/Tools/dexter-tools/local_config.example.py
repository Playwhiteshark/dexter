# Copy this file to local_config.py and keep the local copy out of Git.
ALLOWED_ROOTS = [
    r"C:\Users\ahmad\COMPUTER STUFF\Self-hosted AI\Jarvis Sandbox",
]

# open_app accepts only aliases listed here and never an arbitrary command.
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

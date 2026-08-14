# Copy this file to local_config.py, then add only machine-local policy you trust.
# The committed example is intentionally deny-by-default and contains no user-specific paths.
ALLOWED_ROOTS = [
    # r"C:\path\to\trusted\root",
]

# open_app accepts only aliases listed here and never an arbitrary command.
APP_ALIASES = {
    # "example app": {
    #     "command": [r"C:\path\to\app.exe"],
    #     "allow_arguments": False,
    # },
}

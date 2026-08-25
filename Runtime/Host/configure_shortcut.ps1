[CmdletBinding()]
param(
    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Dexter.lnk")
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonw = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
$iconPath = Join-Path $PSScriptRoot "assets\dexter.ico"

if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw "Dexter's root virtual environment is missing. Run 'uv sync' from '$repoRoot' first."
}

$shortcutDirectory = Split-Path -Parent $ShortcutPath
if ($shortcutDirectory -and -not (Test-Path -LiteralPath $shortcutDirectory)) {
    New-Item -ItemType Directory -Path $shortcutDirectory -Force | Out-Null
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)

# The interactive/dev command is `uv run python -m Runtime.Host.app`, but a
# persistent Windows shortcut can call the already-synced root environment
# directly. This avoids an extra uv process on every normal Dexter launch while
# guaranteeing that Host, STT and TTS use the project's pinned Python/runtime.
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m Runtime.Host.app"
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Dexter local assistant"

if (Test-Path -LiteralPath $iconPath -PathType Leaf) {
    $shortcut.IconLocation = "$iconPath,0"
}

$shortcut.Save()
Write-Host "Dexter shortcut configured: $ShortcutPath"
Write-Host "Target: $pythonw -m Runtime.Host.app"
Write-Host "Start in: $repoRoot"

[CmdletBinding()]
param(
    [string]$ShortcutPath = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Dexter.lnk")
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pythonw = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
$launcher = Join-Path $PSScriptRoot "launch_hidden.vbs"
$wscript = Join-Path $env:WINDIR "System32\wscript.exe"
$iconPath = Join-Path $PSScriptRoot "assets\dexter.ico"

if (-not (Test-Path -LiteralPath $pythonw -PathType Leaf)) {
    throw "Dexter's root virtual environment is missing. Run 'uv sync' from '$repoRoot' first."
}
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Dexter's hidden launcher is missing: '$launcher'."
}
if (-not (Test-Path -LiteralPath $wscript -PathType Leaf)) {
    throw "Windows Script Host was not found at '$wscript'."
}

$shortcutDirectory = Split-Path -Parent $ShortcutPath
if ($shortcutDirectory -and -not (Test-Path -LiteralPath $shortcutDirectory)) {
    New-Item -ItemType Directory -Path $shortcutDirectory -Force | Out-Null
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($ShortcutPath)

# Use Windows Script Host as the GUI-subsystem bootstrap, then launch the root
# pythonw.exe invisibly from launch_hidden.vbs. This keeps normal Dexter startup
# console-free even if Windows would otherwise allocate a terminal for a console
# subsystem executable somewhere in the launch chain.
$shortcut.TargetPath = $wscript
$shortcut.Arguments = "//nologo `"$launcher`""
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = "Dexter local assistant"
$shortcut.WindowStyle = 7

if (Test-Path -LiteralPath $iconPath -PathType Leaf) {
    $shortcut.IconLocation = "$iconPath,0"
}

$shortcut.Save()
Write-Host "Dexter shortcut configured: $ShortcutPath"
Write-Host "Target: $wscript //nologo `"$launcher`""
Write-Host "Start in: $repoRoot"

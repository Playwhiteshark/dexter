[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$packagePath = Join-Path $PSScriptRoot "package.json"
$package = Get-Content -LiteralPath $packagePath -Raw | ConvertFrom-Json
$folderName = "$($package.publisher).$($package.name)-$($package.version)"
$extensionsRoot = if ($env:VSCODE_EXTENSIONS) {
    $env:VSCODE_EXTENSIONS
} else {
    Join-Path $env:USERPROFILE ".vscode\extensions"
}
$destination = Join-Path $extensionsRoot $folderName

New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath $packagePath -Destination (Join-Path $destination "package.json") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "extension.js") -Destination (Join-Path $destination "extension.js") -Force

Write-Host "Installed Dexter Workspace State $($package.version) to:"
Write-Host "  $destination"
Write-Host "Reload or restart each VS Code window to activate the update."

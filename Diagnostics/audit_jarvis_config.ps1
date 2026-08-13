# Jarvis Configuration Audit (read-only)
# Generated for the Local AI / Jarvis project.
# This script DOES NOT modify config, tasks, processes, registry, models, or files.
# It writes a Markdown report beside this script.

$ErrorActionPreference = "Continue"

$Base = "C:\Users\ahmad\COMPUTER STUFF\Self-hosted AI"
$Jarvis = Join-Path $Base "Jarvis"
$Sandbox = Join-Path $Base "Jarvis Sandbox"
$GooseCfg = Join-Path $env:APPDATA "Block\goose\config\config.yaml"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Out = Join-Path $PSScriptRoot "jarvis-config-audit_$Stamp.md"

function Add-Line([string]$s = "") {
    Add-Content -LiteralPath $Out -Value $s -Encoding UTF8
}

function Add-CodeBlock([string[]]$lines, [string]$lang = "text") {
    Add-Line "````$lang"
    foreach ($line in $lines) { Add-Line $line }
    Add-Line "````"
}

function Capture([scriptblock]$cmd) {
    try {
        @(& $cmd 2>&1 | Out-String -Width 260).TrimEnd()
    } catch {
        @("ERROR: $($_.Exception.Message)")
    }
}

function Redact-Secrets([string]$text) {
    if ($null -eq $text) { return "" }
    $patterns = @(
        '(?im)^(\s*[A-Za-z0-9_.-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL)[A-Za-z0-9_.-]*\s*[:=]\s*).*$',
        '(?im)^(\s*Authorization\s*[:=]\s*).*$'
    )
    foreach ($p in $patterns) {
        $text = [regex]::Replace($text, $p, '$1[REDACTED]')
    }
    return $text
}

"# Jarvis Configuration Audit" | Set-Content -LiteralPath $Out -Encoding UTF8
Add-Line ""
Add-Line "- Generated: `$(Get-Date -Format o)`"
Add-Line "- Computer: `$env:COMPUTERNAME`"
Add-Line "- User: `$env:USERNAME`"
Add-Line "- Base: `$Base`"
Add-Line "- This report is read-only and redacts obvious secret/token variables."
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 1. Core executable versions and locations"
Add-Line ""
$commands = @("goose", "ollama", "python", "pythonw", "git")
foreach ($name in $commands) {
    Add-Line "### $name"
    $result = Capture {
        $c = Get-Command $name -ErrorAction Stop
        $c | Select-Object Name, Source, Version, CommandType | Format-List
    }
    Add-CodeBlock $result
    Add-Line ""
}

Add-Line "### Version commands"
Add-CodeBlock (Capture {
    "goose:"
    goose --version
    ""
    "ollama:"
    ollama --version
    ""
    "python:"
    python --version
    ""
    "git:"
    git --version
})
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 2. Current Jarvis directory overview"
Add-Line ""
foreach ($root in @($Jarvis, $Sandbox)) {
    Add-Line "### $root"
    if (Test-Path -LiteralPath $root) {
        $tree = Capture {
            Get-ChildItem -LiteralPath $root -Force -ErrorAction SilentlyContinue |
                Sort-Object @{Expression = {$_.PSIsContainer}; Descending = $true}, Name |
                Select-Object Mode, Length, LastWriteTime, Name |
                Format-Table -AutoSize
        }
        Add-CodeBlock $tree
    } else {
        Add-Line "_Not found._"
    }
    Add-Line ""
}

Add-Line "### Jarvis files up to depth 4"
if (Test-Path -LiteralPath $Jarvis) {
    $rootResolved = (Resolve-Path -LiteralPath $Jarvis).Path
    $files = Get-ChildItem -LiteralPath $Jarvis -File -Recurse -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            $rel = $_.FullName.Substring($rootResolved.Length).TrimStart('\')
            $depth = ($rel -split '\\').Count
            if ($depth -le 4) {
                [PSCustomObject]@{
                    Path = $rel
                    SizeKB = [math]::Round($_.Length / 1KB, 2)
                    Modified = $_.LastWriteTime
                    SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
                }
            }
        } | Sort-Object Path
    Add-CodeBlock ($files | Format-Table -AutoSize | Out-String -Width 260)
} else {
    Add-Line "_Jarvis directory not found._"
}
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 3. Goose effective configuration"
Add-Line ""
Add-Line "### goose info -v"
Add-CodeBlock (Capture { goose info -v })
Add-Line ""

Add-Line "### config.yaml"
if (Test-Path -LiteralPath $GooseCfg) {
    Add-Line "- Path: `$GooseCfg`"
    Add-Line "- SHA256: `$((Get-FileHash -LiteralPath $GooseCfg -Algorithm SHA256).Hash)`"
    $cfgText = Get-Content -LiteralPath $GooseCfg -Raw
    Add-CodeBlock @((Redact-Secrets $cfgText)) "yaml"
} else {
    Add-Line "_Not found._"
}
Add-Line ""

Add-Line "### Nearby Goose config backups"
$cfgDir = Split-Path -Parent $GooseCfg
if (Test-Path -LiteralPath $cfgDir) {
    Add-CodeBlock (Capture {
        Get-ChildItem -LiteralPath $cfgDir -Force -File |
            Select-Object Name, Length, LastWriteTime,
                @{N='SHA256';E={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}} |
            Format-Table -AutoSize
    })
}
Add-Line ""

Add-Line "### Process environment variables relevant to Goose/Ollama"
$envRows = Get-ChildItem Env: |
    Where-Object { $_.Name -match '^(GOOSE_|OLLAMA_)' } |
    Sort-Object Name |
    ForEach-Object {
        $value = $_.Value
        if ($_.Name -match '(KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL)') { $value = "[REDACTED]" }
        [PSCustomObject]@{ Name=$_.Name; Value=$value }
    }
Add-CodeBlock ($envRows | Format-Table -AutoSize | Out-String -Width 260)
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 4. Ollama models and runtime"
Add-Line ""
Add-Line "### ollama list"
Add-CodeBlock (Capture { ollama list })
Add-Line ""
Add-Line "### ollama ps"
Add-CodeBlock (Capture { ollama ps })
Add-Line ""

$models = @("gemma4-jarvis:12b", "gemma4:12b", "qwen3.5:9b", "magenticbrain-fixed:latest")
foreach ($model in $models) {
    Add-Line "### Modelfile: $model"
    $show = Capture { ollama show --modelfile $model }
    Add-CodeBlock $show
    Add-Line ""
}

Add-Line "### Ollama-related processes"
Add-CodeBlock (Capture {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match '^(ollama|llama-server)\.exe$' } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine |
        Format-List
})
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 5. Local listeners / gateways"
Add-Line ""
Add-Line "### Ports 11434–11436"
Add-CodeBlock (Capture {
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in 11434,11435,11436 } |
        ForEach-Object {
            $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
            [PSCustomObject]@{
                Address = $_.LocalAddress
                Port = $_.LocalPort
                PID = $_.OwningProcess
                Process = $proc.ProcessName
            }
        } | Sort-Object Port | Format-Table -AutoSize
})
Add-Line ""

Add-Line "### Python/Pythonw/Goose process command lines"
Add-CodeBlock (Capture {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match '^(python|pythonw|goose)\.exe$' } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine |
        Format-List
})
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 6. Scheduled Tasks relevant to Jarvis / Ollama / Goose / AI"
Add-Line ""
$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue |
    Where-Object {
        $_.TaskName -match '(?i)(jarvis|ollama|goose|anything|ai|model|proxy)' -or
        $_.TaskPath -match '(?i)(jarvis|ollama|goose|anything|ai|model|proxy)'
    }

if ($tasks) {
    foreach ($task in $tasks) {
        Add-Line "### $($task.TaskPath)$($task.TaskName)"
        $info = $task | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue
        Add-CodeBlock @(
            ($task | Select-Object TaskName, TaskPath, State | Format-List | Out-String -Width 260).TrimEnd(),
            "Actions:",
            ($task.Actions | Format-List * | Out-String -Width 260).TrimEnd(),
            "Triggers:",
            ($task.Triggers | Format-List * | Out-String -Width 260).TrimEnd(),
            "Settings:",
            ($task.Settings | Format-List * | Out-String -Width 260).TrimEnd(),
            "Info:",
            ($info | Format-List * | Out-String -Width 260).TrimEnd()
        )
        Add-Line ""
    }
} else {
    Add-Line "_No matching scheduled tasks found._"
    Add-Line ""
}

# -------------------------------------------------------------------------
Add-Line "## 7. Other Windows startup entries"
Add-Line ""
Add-Line "### HKCU Run"
Add-CodeBlock (Capture {
    Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -ErrorAction SilentlyContinue |
        Select-Object * -ExcludeProperty PSPath,PSParentPath,PSChildName,PSDrive,PSProvider |
        Format-List
})
Add-Line ""

Add-Line "### HKLM Run"
Add-CodeBlock (Capture {
    Get-ItemProperty "HKLM:\Software\Microsoft\Windows\CurrentVersion\Run" -ErrorAction SilentlyContinue |
        Select-Object * -ExcludeProperty PSPath,PSParentPath,PSChildName,PSDrive,PSProvider |
        Format-List
})
Add-Line ""

Add-Line "### User Startup folder"
$userStartup = [Environment]::GetFolderPath("Startup")
Add-Line "- `$userStartup`"
if (Test-Path $userStartup) {
    Add-CodeBlock (Capture { Get-ChildItem -LiteralPath $userStartup -Force | Format-Table Mode,Length,LastWriteTime,Name -AutoSize })
}
Add-Line ""

Add-Line "### Common Startup folder"
$commonStartup = [Environment]::GetFolderPath("CommonStartup")
Add-Line "- `$commonStartup`"
if (Test-Path $commonStartup) {
    Add-CodeBlock (Capture { Get-ChildItem -LiteralPath $commonStartup -Force | Format-Table Mode,Length,LastWriteTime,Name -AutoSize })
}
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 8. Installed-app indicators"
Add-Line ""
$uninstallRoots = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
$apps = foreach ($r in $uninstallRoots) {
    Get-ItemProperty $r -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -match '(?i)(AnythingLLM|Ollama|Goose)' } |
        Select-Object DisplayName, DisplayVersion, InstallLocation, Publisher, UninstallString
}
if ($apps) {
    Add-CodeBlock ($apps | Sort-Object DisplayName -Unique | Format-List | Out-String -Width 260)
} else {
    Add-Line "_No matching uninstall entries found._"
}
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 9. Known Jarvis script/config candidates"
Add-Line ""
if (Test-Path $Jarvis) {
    $candidates = Get-ChildItem -LiteralPath $Jarvis -File -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension -in ".py",".ps1",".yaml",".yml",".json",".toml",".ini",".xml" -or
            $_.Name -match '(?i)(config|modelfile|gateway|proxy|bench|audit|launcher|startup)'
        } |
        Select-Object FullName, Length, LastWriteTime,
            @{N='SHA256';E={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}} |
        Sort-Object FullName
    Add-CodeBlock ($candidates | Format-Table -AutoSize | Out-String -Width 300)
} else {
    Add-Line "_Jarvis directory not found._"
}
Add-Line ""

# -------------------------------------------------------------------------
Add-Line "## 10. Quick interpretation checklist"
Add-Line ""
Add-Line "- [ ] Exactly one intended production model endpoint."
Add-Line "- [ ] Diagnostic proxy is not configured to auto-start."
Add-Line "- [ ] Benchmark scripts are not configured to auto-start."
Add-Line "- [ ] Goose production settings have one source of truth."
Add-Line "- [ ] No stale scheduled task points at an archived/obsolete proxy."
Add-Line "- [ ] AnythingLLM is not in the production path unless intentionally retained."
Add-Line "- [ ] Jarvis Sandbox remains separate from production runtime/config."
Add-Line "- [ ] Production gateway, if used, has one explicit startup mechanism."
Add-Line ""

Write-Host ""
Write-Host "Read-only audit complete:"
Write-Host $Out

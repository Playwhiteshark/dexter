param(
    [int]$DurationSeconds = 30,
    [int]$SampleIntervalSeconds = 1,
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

if ($DurationSeconds -lt 5 -or $DurationSeconds -gt 600) {
    throw "DurationSeconds must be between 5 and 600."
}
if ($SampleIntervalSeconds -lt 1 -or $SampleIntervalSeconds -gt 30) {
    throw "SampleIntervalSeconds must be between 1 and 30."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputPath) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = Join-Path $PSScriptRoot "runtime-performance-audit_$stamp.json"
}

function To-MB([double]$bytes) {
    return [math]::Round($bytes / 1MB, 1)
}

function Get-DexterRole([string]$name, [string]$commandLine) {
    $n = if ($name) { $name.ToLowerInvariant() } else { "" }
    $c = if ($commandLine) { $commandLine.ToLowerInvariant() } else { "" }

    if ($c -match "neutts2e_worker\.py") { return "TTS worker" }
    if ($c -match "runtime[\\/]host[\\/]app\.py" -or $c -match "runtime\.host\.app") { return "Host + STT" }
    if ($c -match "model gateway[\\/]gateway\.py" -or $c -match "gateway\.py") { return "Model gateway" }
    if ($c -match "runtime[\\/]tools[\\/]dexter-tools" -or ($c -match "server\.py" -and $c -match "dexter")) { return "MCP tools" }
    if ($n -eq "goose.exe" -or $c -match "(^|[\\/\s])goose(?:\.exe)?\s+acp") { return "Goose ACP" }
    if ($n -like "ollama*" -or $c -match "ollama") { return "Ollama" }
    if ($c -match "dexter-browser-native-host") { return "Browser native host" }
    if (($n -eq "python.exe" -or $n -eq "pythonw.exe") -and ($c -match "dexter" -or $c -match "jarvis")) { return "Other Dexter Python" }
    return $null
}

function Get-RelevantProcessRows {
    $cimByPid = @{}
    foreach ($item in (Get-CimInstance Win32_Process)) {
        $cimByPid[[int]$item.ProcessId] = $item
    }

    $rows = @()
    foreach ($proc in (Get-Process -ErrorAction SilentlyContinue)) {
        $processId = [int]$proc.Id
        $cim = $cimByPid[$processId]
        if ($null -eq $cim) { continue }

        $commandLine = [string]$cim.CommandLine
        $role = Get-DexterRole ([string]$cim.Name) $commandLine
        if (-not $role) { continue }

        try {
            $cpuSeconds = [double]$proc.CPU
            $privateBytes = [double]$proc.PrivateMemorySize64
            $workingSetBytes = [double]$proc.WorkingSet64
            $threadCount = [int]$proc.Threads.Count
            $handleCount = [int]$proc.HandleCount
        } catch {
            continue
        }

        $rows += [pscustomobject]@{
            pid = $processId
            ppid = [int]$cim.ParentProcessId
            role = $role
            name = [string]$cim.Name
            executable = [string]$cim.ExecutablePath
            command_line = $commandLine
            cpu_seconds = $cpuSeconds
            private_mb = To-MB $privateBytes
            working_set_mb = To-MB $workingSetBytes
            threads = $threadCount
            handles = $handleCount
        }
    }
    return @($rows)
}

function Get-GpuUsageByPid {
    $result = @{}
    try {
        $counter = Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop
        foreach ($sample in $counter.CounterSamples) {
            $instance = [string]$sample.InstanceName
            if ($instance -notmatch 'pid_(\d+)') { continue }
            $processId = [int]$Matches[1]
            $value = [double]$sample.CookedValue
            if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { continue }
            if (-not $result.ContainsKey($processId) -or $value -gt $result[$processId]) {
                # Use the busiest engine instead of summing engines, which can
                # exceed 100% and is misleading for a quick process comparison.
                $result[$processId] = [math]::Round($value, 1)
            }
        }
    } catch {
        # GPU Engine counters are not guaranteed to exist on every Windows build.
    }
    return $result
}

$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$video = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    [pscustomobject]@{
        name = [string]$_.Name
        driver_version = [string]$_.DriverVersion
        adapter_ram_mb = if ($_.AdapterRAM) { To-MB ([double]$_.AdapterRAM) } else { $null }
        status = [string]$_.Status
    }
})
$npu = @(Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match 'NPU|Neural|AI Boost'
} | ForEach-Object {
    [pscustomobject]@{
        name = [string]$_.Name
        status = [string]$_.Status
        pnp_class = [string]$_.PNPClass
        device_id = [string]$_.DeviceID
    }
})

$logicalProcessors = [int]$cpu.NumberOfLogicalProcessors
if ($logicalProcessors -lt 1) { $logicalProcessors = [Environment]::ProcessorCount }

$system = [ordered]@{
    os_caption = [string]$os.Caption
    os_version = [string]$os.Version
    os_build = [string]$os.BuildNumber
    total_ram_mb = To-MB ([double]$os.TotalVisibleMemorySize * 1KB)
    free_ram_mb_at_start = To-MB ([double]$os.FreePhysicalMemory * 1KB)
    cpu_name = [string]$cpu.Name
    physical_cores = [int]$cpu.NumberOfCores
    logical_processors = $logicalProcessors
    video_adapters = $video
    npu_devices = $npu
}

Write-Host "Dexter runtime performance audit"
Write-Host "CPU: $($system.cpu_name)"
Write-Host "RAM: $($system.total_ram_mb) MB total; $($system.free_ram_mb_at_start) MB free at start"
Write-Host "OS:  $($system.os_caption) $($system.os_version) (build $($system.os_build))"
foreach ($gpu in $video) {
    Write-Host "GPU: $($gpu.name)  driver=$($gpu.driver_version)"
}
if ($npu.Count -eq 0) {
    Write-Host "NPU: no matching PnP device found"
} else {
    foreach ($device in $npu) { Write-Host "NPU: $($device.name)  status=$($device.status)" }
}
Write-Host ""
Write-Host "Sampling for $DurationSeconds seconds. Use Dexter normally and send one representative prompt now."

$stats = @{}
$previousCpu = @{}
$previousAt = @{}
$samples = @()
$pythonExecutables = @{}
$startedAt = Get-Date
$deadline = $startedAt.AddSeconds($DurationSeconds)

while ((Get-Date) -lt $deadline) {
    $sampleAt = Get-Date
    $rows = @(Get-RelevantProcessRows)
    $gpuByPid = Get-GpuUsageByPid
    $sampleRows = @()

    foreach ($row in $rows) {
        $processId = [int]$row.pid
        $key = "$processId|$($row.role)"
        $nowCpu = [double]$row.cpu_seconds
        $cpuPct = 0.0

        if ($previousCpu.ContainsKey($processId) -and $previousAt.ContainsKey($processId)) {
            $elapsed = ($sampleAt - $previousAt[$processId]).TotalSeconds
            if ($elapsed -gt 0) {
                $delta = $nowCpu - [double]$previousCpu[$processId]
                if ($delta -ge 0) {
                    # Normalize to Task-Manager-like whole-machine percentage.
                    $cpuPct = [math]::Round(($delta / $elapsed / $logicalProcessors) * 100.0, 1)
                }
            }
        }
        $previousCpu[$processId] = $nowCpu
        $previousAt[$processId] = $sampleAt

        $gpuPct = 0.0
        if ($gpuByPid.ContainsKey($processId)) { $gpuPct = [double]$gpuByPid[$processId] }

        if (-not $stats.ContainsKey($key)) {
            $stats[$key] = [ordered]@{
                pid = $processId
                ppid = [int]$row.ppid
                role = [string]$row.role
                name = [string]$row.name
                executable = [string]$row.executable
                command_line = [string]$row.command_line
                first_private_mb = [double]$row.private_mb
                peak_private_mb = [double]$row.private_mb
                peak_working_set_mb = [double]$row.working_set_mb
                peak_cpu_percent = $cpuPct
                peak_gpu_engine_percent = $gpuPct
                samples = 0
            }
        }
        $s = $stats[$key]
        $s.samples++
        if ([double]$row.private_mb -gt $s.peak_private_mb) { $s.peak_private_mb = [double]$row.private_mb }
        if ([double]$row.working_set_mb -gt $s.peak_working_set_mb) { $s.peak_working_set_mb = [double]$row.working_set_mb }
        if ($cpuPct -gt $s.peak_cpu_percent) { $s.peak_cpu_percent = $cpuPct }
        if ($gpuPct -gt $s.peak_gpu_engine_percent) { $s.peak_gpu_engine_percent = $gpuPct }

        if (($row.name -eq "python.exe" -or $row.name -eq "pythonw.exe") -and $row.executable) {
            $pythonExecutables[[string]$row.executable] = $true
        }

        $sampleRows += [pscustomobject]@{
            pid = $processId
            role = [string]$row.role
            private_mb = [double]$row.private_mb
            working_set_mb = [double]$row.working_set_mb
            cpu_percent = $cpuPct
            gpu_engine_percent = $gpuPct
        }
    }

    $samples += [pscustomobject]@{
        at = $sampleAt.ToString("o")
        processes = $sampleRows
    }

    $remaining = ($deadline - (Get-Date)).TotalSeconds
    if ($remaining -le 0) { break }
    Start-Sleep -Seconds ([math]::Min($SampleIntervalSeconds, [math]::Ceiling($remaining)))
}

$osEnd = Get-CimInstance Win32_OperatingSystem
$summary = @($stats.Values | Sort-Object role, pid | ForEach-Object { [pscustomobject]$_ })
$totalPeakPrivate = [math]::Round((($summary | Measure-Object -Property peak_private_mb -Sum).Sum), 1)

$nvidia = $null
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    try {
        $nvidia = @(& $nvidiaSmi.Source --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null)
    } catch {
        $nvidia = $null
    }
}

$result = [ordered]@{
    started_at = $startedAt.ToString("o")
    duration_seconds = $DurationSeconds
    sample_interval_seconds = $SampleIntervalSeconds
    system = $system
    free_ram_mb_at_end = To-MB ([double]$osEnd.FreePhysicalMemory * 1KB)
    relevant_peak_private_mb_sum = $totalPeakPrivate
    python_executables_observed = @($pythonExecutables.Keys | Sort-Object)
    nvidia_smi = $nvidia
    process_summary = $summary
    samples = $samples
}

$result | ConvertTo-Json -Depth 8 | Set-Content -Path $OutputPath -Encoding UTF8

Write-Host ""
Write-Host "Process summary (peak during sample):"
$summary |
    Select-Object role, pid, name, peak_private_mb, peak_working_set_mb, peak_cpu_percent, peak_gpu_engine_percent |
    Format-Table -AutoSize

Write-Host "Relevant-process peak private-memory sum: $totalPeakPrivate MB"
Write-Host "Free RAM at end: $(To-MB ([double]$osEnd.FreePhysicalMemory * 1KB)) MB"
Write-Host ""
Write-Host "Python executables observed:"
if ($pythonExecutables.Count -eq 0) {
    Write-Host "  none"
} else {
    foreach ($path in ($pythonExecutables.Keys | Sort-Object)) { Write-Host "  $path" }
}
Write-Host ""
Write-Host "Report: $OutputPath"

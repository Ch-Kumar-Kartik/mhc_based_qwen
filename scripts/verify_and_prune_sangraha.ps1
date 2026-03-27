param(
    [string]$RawDir = "sangraha_verified",
    [string]$PackedDir = "data/sangraha_packed",
    [string]$PythonExe = "python",
    [switch]$Apply,
    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-DirSizeBytes {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return [int64]0 }
    $sum = (Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $sum) { return [int64]0 }
    return [int64]$sum
}

function Format-Size {
    param([double]$Bytes)
    if ($Bytes -ge 1TB) { return "{0:N2} TB" -f ($Bytes / 1TB) }
    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    return "{0:N2} KB" -f ($Bytes / 1KB)
}

if (-not (Test-Path $RawDir)) {
    throw "RawDir not found: $RawDir"
}
if (-not (Test-Path $PackedDir)) {
    throw "PackedDir not found: $PackedDir"
}

$packedShards = Get-ChildItem -Path $PackedDir -Directory | Where-Object { $_.Name -like 'shard_*' }
if (-not $packedShards) {
    Write-Host "No shard_* folders found in $PackedDir"
    exit 0
}

$pythonCheckCode = @"
import sys
from datasets import load_from_disk
p = sys.argv[1]
ds = load_from_disk(p)
print(len(ds))
"@

$report = @()
$totalReclaimBytes = [int64]0

foreach ($shard in $packedShards) {
    $lang = $shard.Name.Substring(6)
    $rawPath = Join-Path $RawDir $lang
    if (-not (Test-Path $rawPath)) {
        continue
    }

    $isValid = $false
    $rows = $null

    if ($SkipValidation) {
        $isValid = $true
    } else {
        try {
            $rowsOut = & $PythonExe -c $pythonCheckCode $shard.FullName
            if ($LASTEXITCODE -eq 0) {
                $rows = ($rowsOut | Select-Object -Last 1).ToString().Trim()
                $isValid = $true
            }
        } catch {
            $isValid = $false
        }
    }

    $rawBytes = Get-DirSizeBytes -Path $rawPath
    if ($isValid) {
        $totalReclaimBytes += $rawBytes
    }

    $report += [PSCustomObject]@{
        Lang = $lang
        Shard = $shard.FullName
        RawPath = $rawPath
        RawSizeGB = [math]::Round($rawBytes / 1GB, 2)
        Validated = $isValid
        Rows = $rows
    }
}

if (-not $report) {
    Write-Host "No matching raw-language folders found for existing shard_* outputs."
    exit 0
}

$report | Sort-Object RawSizeGB -Descending | Format-Table -AutoSize
Write-Host ""
Write-Host "Potential reclaim (validated only): $(Format-Size $totalReclaimBytes)"

$invalid = $report | Where-Object { -not $_.Validated }
if ($invalid) {
    Write-Warning "Some shards failed validation. Their raw folders will NOT be deleted."
}

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to delete validated raw folders."
    exit 0
}

$toDelete = $report | Where-Object { $_.Validated }
if (-not $toDelete) {
    Write-Host "Nothing to delete after validation."
    exit 0
}

foreach ($item in $toDelete) {
    try {
        Remove-Item -Path $item.RawPath -Recurse -Force
        Write-Host "Deleted raw folder: $($item.RawPath)"
    } catch {
        Write-Warning "Failed to delete $($item.RawPath): $_"
    }
}

Write-Host ""
Write-Host "Cleanup complete. Space reclaimed depends on successful deletions above."

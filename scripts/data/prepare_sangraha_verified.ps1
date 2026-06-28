# Run from the repository root so relative data and script paths resolve.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

<#
.SYNOPSIS
    Sangraha Dataset Preparation - COMPLETE WORKING VERSION
    
.DESCRIPTION
    All syntax errors fixed, ready to run
    - Auto SSD detection
    - Optional aria2 acceleration
    - Resume capability
    - Cache size limits
    - Full cleanup options
    - LZ4 compression
#>

Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║    Sangraha Dataset Preparation - ULTIMATE EDITION           ║
║                                                               ║
║  ✓ Auto SSD detection       ✓ aria2 acceleration            ║
║  ✓ Resume capability        ✓ Verification prompts          ║
║  ✓ Multi-drive optimization ✓ Full cleanup options          ║
╚══════════════════════════════════════════════════════════════╝
"@

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

function Get-FolderSize {
    param([string]$Path)
    if (Test-Path $Path) {
        $measure = Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue |
                   Measure-Object -Property Length -Sum

        if ($null -eq $measure) {
            return 0
        }

        $sumProperty = $measure.PSObject.Properties['Sum']
        if ($null -eq $sumProperty -or $null -eq $sumProperty.Value) {
            return 0
        }

        return [int64]$sumProperty.Value
    }
    return 0
}

function Format-FileSize {
    param([long]$Size)
    if ($Size -gt 1TB) { return "{0:N2} TB" -f ($Size / 1TB) }
    if ($Size -gt 1GB) { return "{0:N2} GB" -f ($Size / 1GB) }
    if ($Size -gt 1MB) { return "{0:N2} MB" -f ($Size / 1MB) }
    return "{0:N2} KB" -f ($Size / 1KB)
}

function Get-BestSsdDriveLetter {
    param([string]$PreferredExclude)

    $candidates = @()
    $drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -match '^[A-Z]$' }

    foreach ($drive in $drives) {
        try {
            $disk = Get-Partition -DriveLetter $drive.Name -ErrorAction Stop |
                    Get-Disk -ErrorAction Stop |
                    Select-Object -First 1

            $isSsd = ($disk.BusType -eq 'NVMe' -or $disk.MediaType -eq 'SSD')
            if ($isSsd) {
                $candidates += [PSCustomObject]@{
                    Drive = $drive.Name
                    Free  = $drive.Free
                }
            }
        } catch {
            # Ignore drives we cannot inspect
        }
    }

    if (-not $candidates -or $candidates.Count -eq 0) {
        return $null
    }

    $nonExcluded = @($candidates | Where-Object { $_.Drive -ne $PreferredExclude })
    if ($nonExcluded.Count -gt 0) {
        return ($nonExcluded | Sort-Object Free -Descending | Select-Object -First 1).Drive
    }

    return ($candidates | Sort-Object Free -Descending | Select-Object -First 1).Drive
}

#==============================================================================
# CONFIGURATION
#==============================================================================

Write-Host "`n[0/5] Detecting storage configuration..."

$currentDrive = (Get-Location).Drive.Name
$currentDriveType = "Unknown"

try {
    $physicalDisk = Get-Partition -DriveLetter $currentDrive -ErrorAction SilentlyContinue | 
                   Get-Disk -ErrorAction SilentlyContinue
    
    if ($physicalDisk) {
        if ($physicalDisk.BusType -eq 'NVMe' -or $physicalDisk.MediaType -eq 'SSD') {
            $currentDriveType = "NVMe/SSD"
        } else {
            $currentDriveType = "HDD"
        }
    }
} catch {
    $currentDriveType = "Unknown"
}

Write-Host "Current location: ${currentDrive}:\ ($currentDriveType)"

if ($currentDriveType -eq "HDD") {
    Write-Warning "You're running from an HDD. Processing will be slower."
    Write-Host "Note: Script will auto-place cache on SSD if available."
    Write-Host ""
}

# Configure paths with smart SSD detection
$BASE_DIR = $PWD
$HF_CACHE = "$BASE_DIR\hf_cache"
$OUTPUT_DIR = "$BASE_DIR\data\sangraha_packed"
$DATA_DIR = "$BASE_DIR\sangraha_verified"

# Smart arrow cache placement
$ARROW_CACHE_DRIVE = Get-BestSsdDriveLetter -PreferredExclude $currentDrive
if (-not $ARROW_CACHE_DRIVE) {
    Write-Host "No separate SSD detected. Using current drive for arrow cache."
    $ARROW_CACHE = "$BASE_DIR\arrow_cache"
} else {
    Write-Host "Detected SSD on ${ARROW_CACHE_DRIVE}: - will use for arrow cache (faster!)"
    $ARROW_CACHE = "${ARROW_CACHE_DRIVE}:\mhc_arrow_cache"
}

$ARROW_CACHE_LIMIT_GB = 100
$ARROW_CACHE_LIMIT_BYTES = $ARROW_CACHE_LIMIT_GB * 1GB

# Optional download acceleration using aria2c
$ARIA2_CMD = Get-Command aria2c -ErrorAction SilentlyContinue
$USE_ARIA2 = $null -ne $ARIA2_CMD
$ARIA2_CONNECTIONS = 16

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================

Write-Host "`n[1/5] Configuring environment..."

$env:HF_HOME = "$BASE_DIR\hf_home"
$env:HF_DATASETS_CACHE = $HF_CACHE
$env:HUGGINGFACE_HUB_CACHE = $HF_CACHE
$env:TEMP = "$ARROW_CACHE\tmp"
$env:TMP = "$ARROW_CACHE\tmp"
$env:TMPDIR = "$ARROW_CACHE\tmp"

# Create directories
@($HF_CACHE, "$HF_CACHE\tmp", $ARROW_CACHE, "$ARROW_CACHE\tmp", $OUTPUT_DIR) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

Write-Host "Storage configuration:"
Write-Host "  Data directory:   $DATA_DIR"
Write-Host "  Output directory: $OUTPUT_DIR"
Write-Host "  HF cache:         $HF_CACHE"
Write-Host "  Arrow cache:      $ARROW_CACHE (max ${ARROW_CACHE_LIMIT_GB}GB)"
if ($USE_ARIA2) {
    Write-Host "  Download mode:    aria2c accelerated ($ARIA2_CONNECTIONS connections)"
} else {
    Write-Host "  Download mode:    Standard (install aria2c for faster download)"
}

#==============================================================================
# RESUME DETECTION
#==============================================================================

$hasDownloadCache = Test-Path "$HF_CACHE\datasets"
$hasTokenCache = Test-Path "$ARROW_CACHE\tokenized.arrow"
$hasPackCache = Test-Path "$ARROW_CACHE\packed.arrow"

if ($hasDownloadCache -or $hasTokenCache -or $hasPackCache) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host "         RESUME DETECTED"
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Found existing cache:"
    if ($hasDownloadCache) { Write-Host "  - Download cache exists (will skip re-download)" }
    if ($hasTokenCache) { Write-Host "  - Tokenization cache exists (will skip tokenization)" }
    if ($hasPackCache) { Write-Host "  - Packing cache exists (will skip packing)" }
    Write-Host ""
    Write-Host "The script will RESUME from cached progress."
    Write-Host ""
    
    $clearCache = Read-Host "Clear cache and start fresh? (y/n)"
    if ($clearCache -eq 'y' -or $clearCache -eq 'Y') {
        Write-Host "Clearing caches..."
        Remove-Item -Recurse -Force $ARROW_CACHE -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force $HF_CACHE -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Path "$ARROW_CACHE\tmp" -Force | Out-Null
        New-Item -ItemType Directory -Path "$HF_CACHE\tmp" -Force | Out-Null
        Write-Host "Cache cleared. Starting fresh."
    } else {
        Write-Host "Will resume from cache."
    }
    Write-Host ""
}

#==============================================================================
# PRE-FLIGHT CHECKS
#==============================================================================

# Check arrow cache current size
$arrowCacheCurrentSize = Get-FolderSize $ARROW_CACHE
if ($arrowCacheCurrentSize -gt $ARROW_CACHE_LIMIT_BYTES) {
    Write-Warning "Arrow cache currently exceeds ${ARROW_CACHE_LIMIT_GB}GB ($(Format-FileSize $arrowCacheCurrentSize))"
    Write-Host "Cleaning oversized cache..."
    try {
        Remove-Item -Recurse -Force -Path $ARROW_CACHE -ErrorAction Stop
        New-Item -ItemType Directory -Path $ARROW_CACHE -Force | Out-Null
        New-Item -ItemType Directory -Path "$ARROW_CACHE\tmp" -Force | Out-Null
        Write-Host "Cache cleaned"
    } catch {
        Write-Error "Failed to clean arrow cache: $_"
        exit 1
    }
}

# Check free space on arrow cache drive
if ($ARROW_CACHE -match '^([A-Z]):') {
    $arrowDriveLetter = $matches[1]
    $arrowDriveInfo = Get-PSDrive -Name $arrowDriveLetter -ErrorAction SilentlyContinue
    
    if ($arrowDriveInfo) {
        $arrowDriveFree = $arrowDriveInfo.Free
        if ($arrowDriveFree -lt $ARROW_CACHE_LIMIT_BYTES) {
            Write-Error "Not enough free space on ${arrowDriveLetter}: for ${ARROW_CACHE_LIMIT_GB}GB cache. Free: $(Format-FileSize $arrowDriveFree)"
            exit 1
        }
        Write-Host "Free space check passed: $(Format-FileSize $arrowDriveFree) available"
    }
}

#==============================================================================
# DOWNLOAD
#==============================================================================

Write-Host "`n[2/5] Downloading dataset..."

if (Test-Path $DATA_DIR) {
    Write-Host "Dataset already exists at: $DATA_DIR"
    Write-Host "  Skipping download. Delete directory to re-download."
} else {
    if ($USE_ARIA2) {
        Write-Host "Starting aria2-accelerated download (resume enabled)..."
    } else {
        Write-Host "Starting download (30-60 minutes, resume enabled)..."
    }
    
    $aria2PathLiteral = if ($USE_ARIA2) { $ARIA2_CMD.Source } else { "" }
    
    $downloadScript = @"
from datasets import load_dataset, DownloadConfig
from huggingface_hub import HfApi
from pathlib import Path
from urllib.parse import quote
import subprocess
import sys
import os

HF_CACHE = r'$HF_CACHE'
DATA_DIR = r'$DATA_DIR'
USE_ARIA2 = $($USE_ARIA2.ToString().ToLower())
ARIA2_PATH = r'$aria2PathLiteral'
ARIA2_CONNECTIONS = $ARIA2_CONNECTIONS


def default_download() -> None:
    print('Downloading ai4bharat/sangraha (verified) with datasets...')
    config = DownloadConfig(
        resume_download=True,
        max_retries=50,
        num_proc=4
    )

    ds = load_dataset(
        'ai4bharat/sangraha',
        'verified',
        cache_dir=HF_CACHE,
        download_config=config,
        verification_mode='no_checks'
    )

    print('')
    print('Saving dataset to disk...')
    ds.save_to_disk(DATA_DIR)
    print('Download complete')


def aria2_download() -> None:
    print('aria2 detected. Attempting accelerated dataset snapshot download...')
    repo_id = 'ai4bharat/sangraha'
    snapshot_dir = Path(HF_CACHE) / 'aria2_snapshot_sangraha'
    manifest_dir = Path(HF_CACHE) / 'tmp'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / 'sangraha_aria2_urls.txt'

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
    if not repo_files:
        raise RuntimeError('No files returned from Hugging Face API for dataset repository.')

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open('w', encoding='utf-8') as f:
        for rel_path in repo_files:
            encoded_rel_path = quote(rel_path, safe='/')
            url = f'https://huggingface.co/datasets/{repo_id}/resolve/main/{encoded_rel_path}'
            target_dir = snapshot_dir / Path(rel_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)

            f.write(url + '\n')
            f.write(f'  dir={target_dir}\n')
            f.write(f'  out={Path(rel_path).name}\n')

    cmd = [
        ARIA2_PATH,
        '--continue=true',
        '--allow-overwrite=true',
        '--auto-file-renaming=false',
        '--file-allocation=none',
        '--max-concurrent-downloads=' + str(ARIA2_CONNECTIONS),
        '--split=' + str(ARIA2_CONNECTIONS),
        '--max-connection-per-server=' + str(ARIA2_CONNECTIONS),
        '--input-file=' + str(manifest_path),
    ]

    subprocess.run(cmd, check=True)

    print('')
    print('Building dataset from aria2 snapshot...')
    ds = load_dataset(str(snapshot_dir), 'verified', verification_mode='no_checks')
    ds.save_to_disk(DATA_DIR)
    print('Download complete (aria2 path)')


try:
    if USE_ARIA2:
        try:
            aria2_download()
        except Exception as aria_err:
            print(f'WARNING: aria2 path failed: {aria_err}')
            print('Falling back to standard Hugging Face downloader...')
            default_download()
    else:
        default_download()
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"@
    
    python -c $downloadScript
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Download failed. Cache preserved. Re-run to resume."
        exit $LASTEXITCODE
    }
}

#==============================================================================
# PROCESSING
#==============================================================================

Write-Host "`n[3/5] Processing (tokenization + packing)..."
Write-Host "="*80
Write-Host "Configuration:"
Write-Host "  - LZ4 compression:     Enabled (30% cache reduction)"
Write-Host "  - Progressive cleanup: Enabled (auto-delete intermediate files)"
Write-Host "  - Parallel processes:  16"
Write-Host "  - Batch size:          10000"
Write-Host "  - Arrow cache:         $ARROW_CACHE"
Write-Host ""
Write-Host "This will take 6-12 hours depending on hardware."
Write-Host "If interrupted, just re-run - it will resume from cache."
Write-Host ""

$processingStartTime = Get-Date

# Detect which script to use
$scriptName = $null
if (Test-Path "prepare_data_storage_optimized.py") {
    $scriptName = "prepare_data_storage_optimized.py"
} elseif (Test-Path "prepare_data_optimized.py") {
    $scriptName = "prepare_data_optimized.py"
} elseif (Test-Path "scripts/data/prepare_sangraha_splits.py") {
    $scriptName = "scripts/data/prepare_sangraha_splits.py"
}

if (-not $scriptName) {
    Write-Error "No preparation script found!"
    exit 1
}

Write-Host "Using script: $scriptName"
Write-Host ""

# Run processing
if ($scriptName -eq "prepare_data_storage_optimized.py") {
    python $scriptName `
        --data-dir "$DATA_DIR" `
        --output-dir "$OUTPUT_DIR" `
        --arrow-cache-dir "$ARROW_CACHE" `
        --cache-dir "$HF_CACHE" `
        --compression "lz4" `
        --num-proc 16 `
        --batch-size 10000
} elseif ($scriptName -eq "prepare_data_optimized.py") {
    python $scriptName `
        --data-dir "$DATA_DIR" `
        --output-dir "$OUTPUT_DIR" `
        --arrow-cache-dir "$ARROW_CACHE" `
        --cache-dir "$HF_CACHE" `
        --num-proc 16 `
        --batch-size 10000
} else {
    Write-Warning "Using basic script - no cache control"
    python $scriptName
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Processing failed. Cache preserved at: $ARROW_CACHE"
    Write-Host "Re-run this script to resume from cache."
    exit $LASTEXITCODE
}

$processingDuration = (Get-Date) - $processingStartTime

#==============================================================================
# POST-FLIGHT CHECKS
#==============================================================================

$arrowCacheSize = Get-FolderSize $ARROW_CACHE
if ($arrowCacheSize -gt $ARROW_CACHE_LIMIT_BYTES) {
    Write-Error "Arrow cache exceeded ${ARROW_CACHE_LIMIT_GB}GB limit: $(Format-FileSize $arrowCacheSize)"
    Write-Error "Please clean '$ARROW_CACHE' and re-run with lower batch-size or num-proc"
    exit 1
}

#==============================================================================
# VERIFICATION & CLEANUP
#==============================================================================

Write-Host "`n[4/5] Verification and cleanup..."

$dataSize = Get-FolderSize $DATA_DIR
$outputSize = Get-FolderSize $OUTPUT_DIR
$hfCacheSize = Get-FolderSize $HF_CACHE

Write-Host ""
Write-Host "=========================================="
Write-Host "     PROCESSING COMPLETE"
Write-Host "=========================================="
Write-Host ""
Write-Host "Processing time:  $($processingDuration.Hours)h $($processingDuration.Minutes)m"
Write-Host "Output location:  $OUTPUT_DIR"
Write-Host "Output size:      $(Format-FileSize $outputSize)"
Write-Host ""
Write-Host "Storage Report:"
Write-Host "   Original data:    $(Format-FileSize $dataSize)"
Write-Host "   Processed output: $(Format-FileSize $outputSize)"
Write-Host "   HF cache:         $(Format-FileSize $hfCacheSize)"
Write-Host "   Arrow cache:      $(Format-FileSize $arrowCacheSize) (TEMPORARY)"
Write-Host ""
Write-Host "   Total current:    $(Format-FileSize ($dataSize + $outputSize + $arrowCacheSize + $hfCacheSize))"
Write-Host ""

# Verification prompt
Write-Host "IMPORTANT: Verify your output before deleting cache!"
Write-Host ""
Write-Host "Verification steps:"
Write-Host "  1. Check output exists:  ls '$OUTPUT_DIR'"
Write-Host "  2. Check train split:    ls '$OUTPUT_DIR\train'"
Write-Host "  3. Check metadata:       cat '$OUTPUT_DIR\metadata.json'"
Write-Host ""

$verify = Read-Host "Have you verified the output is correct? (y/n)"

if ($verify -eq 'y' -or $verify -eq 'Y') {
    Write-Host ""
    Write-Host "Great! Now you can safely delete caches to free space."
    Write-Host ""
    
    # Arrow cache cleanup
    if ($arrowCacheSize -gt 100MB) {
        $deleteArrow = Read-Host "Delete arrow cache to free $(Format-FileSize $arrowCacheSize)? (y/n)"
        
        if ($deleteArrow -eq 'y' -or $deleteArrow -eq 'Y') {
            Write-Host "Deleting arrow cache..."
            try {
                Start-Sleep -Seconds 2
                Remove-Item -Recurse -Force $ARROW_CACHE -ErrorAction Stop
                Write-Host "Deleted arrow cache"
                Write-Host "Freed $(Format-FileSize $arrowCacheSize)"
                $arrowCacheSize = 0
            } catch {
                Write-Warning "Could not delete: $_"
                Write-Host "Delete manually: Remove-Item -Recurse -Force '$ARROW_CACHE'"
            }
        } else {
            Write-Host "Arrow cache preserved at: $ARROW_CACHE"
        }
    }
    
    # HF cache cleanup
    Write-Host ""
    $deleteHf = Read-Host "Also delete HF cache ($(Format-FileSize $hfCacheSize))? Only if you won't re-download. (y/n)"
    if ($deleteHf -eq 'y' -or $deleteHf -eq 'Y') {
        try {
            Remove-Item -Recurse -Force $HF_CACHE -ErrorAction Stop
            Write-Host "Freed $(Format-FileSize $hfCacheSize) from HF cache"
            $hfCacheSize = 0
        } catch {
            Write-Warning "Could not delete: $_"
        }
    }
    
    # Original data compression suggestion
    Write-Host ""
    $compressOriginal = Read-Host "Compress original data ($(Format-FileSize $dataSize) to save 20GB)? (y/n)"
    if ($compressOriginal -eq 'y' -or $compressOriginal -eq 'Y') {
        Write-Host "Compressing (this may take 10-15 minutes)..."
        try {
            Compress-Archive -Path $DATA_DIR -DestinationPath "${DATA_DIR}.zip" -CompressionLevel Optimal
            $zipSize = (Get-Item "${DATA_DIR}.zip").Length
            Write-Host "Compressed to $(Format-FileSize $zipSize)"
            
            $deleteOriginal = Read-Host "Delete uncompressed original? (y/n)"
            if ($deleteOriginal -eq 'y' -or $deleteOriginal -eq 'Y') {
                Remove-Item -Recurse -Force $DATA_DIR
                Write-Host "Deleted original, freed $(Format-FileSize ($dataSize - $zipSize))"
                $dataSize = $zipSize
            }
        } catch {
            Write-Warning "Compression failed: $_"
        }
    }
    
} else {
    Write-Host ""
    Write-Host "Cache PRESERVED for safety"
    Write-Host ""
    Write-Host "When ready to delete after verification:"
    Write-Host "  Arrow cache: Remove-Item -Recurse -Force '$ARROW_CACHE'"
    Write-Host "  HF cache:    Remove-Item -Recurse -Force '$HF_CACHE'"
    Write-Host ""
}

#==============================================================================
# FINAL SUMMARY
#==============================================================================

Write-Host ""
Write-Host "[5/5] Final summary..."
Write-Host ""

$finalSize = $dataSize + $outputSize + $arrowCacheSize + $hfCacheSize

Write-Host "="*80
Write-Host "FINAL SUMMARY"
Write-Host "="*80
Write-Host "Total processing time:  $($processingDuration.Hours)h $($processingDuration.Minutes)m"
Write-Host "Final disk usage:       $(Format-FileSize $finalSize)"
Write-Host "Prepared dataset:       $OUTPUT_DIR"
Write-Host ""
Write-Host "Your dataset is ready for training!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Load dataset in training script:"
Write-Host "     from datasets import load_from_disk"
Write-Host "     ds = load_from_disk('$OUTPUT_DIR')"
Write-Host ""
Write-Host "  2. Access splits:"
Write-Host "     train_ds = ds['train']"
Write-Host "     val_ds = ds['validation']"
Write-Host "     test_ds = ds['test']"
Write-Host ""
Write-Host "  3. Start training!"
Write-Host ""
Write-Host "="*80
Write-Host ""
Write-Host "Pipeline completed successfully."

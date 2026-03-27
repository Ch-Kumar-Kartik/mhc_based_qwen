# train_all_indiccorp.ps1
# PowerShell script to train mHC model on all IndicCorpV2 languages
# Optimized for RTX 4070 (12GB VRAM)

# Languages with their training steps
$Languages = @(
    @{ Name = "indiccorp_hi"; Steps = 100000; Desc = "Hindi" },
    @{ Name = "indiccorp_te"; Steps = 50000; Desc = "Telugu" },
    @{ Name = "indiccorp_ta"; Steps = 50000; Desc = "Tamil" },
    @{ Name = "indiccorp_mr"; Steps = 50000; Desc = "Marathi" },
    @{ Name = "indiccorp_bn"; Steps = 50000; Desc = "Bengali" },
    @{ Name = "indiccorp_gu"; Steps = 50000; Desc = "Gujarati" },
    @{ Name = "indiccorp_kn"; Steps = 50000; Desc = "Kannada" },
    @{ Name = "indiccorp_ml"; Steps = 50000; Desc = "Malayalam" },
    @{ Name = "indiccorp_pa"; Steps = 30000; Desc = "Punjabi" },
    @{ Name = "indiccorp_or"; Steps = 30000; Desc = "Odia" },
    @{ Name = "indiccorp_as"; Steps = 20000; Desc = "Assamese" },
    @{ Name = "indiccorp_ne"; Steps = 20000; Desc = "Nepali" },
    @{ Name = "indiccorp_sa"; Steps = 10000; Desc = "Sanskrit" },
    @{ Name = "indiccorp_ur"; Steps = 50000; Desc = "Urdu" }
)

# Training parameters for RTX 4070 (12GB VRAM)
$BatchSize = 4
$GradAccum = 8
$MaxLength = 1024
$SaveInterval = 1000

# Start fresh or provide path to pre-converted mHC model
$ModelPath = ""

# Output base directory
$OutputBase = "./output"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "IndicCorpV2 Multi-Language Training Script" -ForegroundColor Cyan
Write-Host "RTX 4070 Optimized Settings" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "Batch size: $BatchSize"
Write-Host "Gradient accumulation: $GradAccum"
Write-Host "Effective batch size: $($BatchSize * $GradAccum)"
Write-Host "Max sequence length: $MaxLength"
Write-Host ""

$TotalSteps = ($Languages | Measure-Object -Property Steps -Sum).Sum
Write-Host "Total training steps across all languages: $TotalSteps" -ForegroundColor Yellow
Write-Host ""

foreach ($Lang in $Languages) {
    $LangName = $Lang.Name
    $Steps = $Lang.Steps
    $Desc = $Lang.Desc
    $OutputDir = "$OutputBase/mhc_$LangName"
    
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "Training on $Desc ($LangName) for $Steps steps..." -ForegroundColor Green
    Write-Host "Output directory: $OutputDir" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    
    $Args = @(
        "scripts/train_amd_v2.py",
        "--dataset", $LangName,
        "--batch-size", $BatchSize,
        "--grad-accum", $GradAccum,
        "--max-length", $MaxLength,
        "--total-steps", $Steps,
        "--save-interval", $SaveInterval,
        "--output", $OutputDir
    )
    
    # Add mhc-model flag if we have a previous model
    if ($ModelPath -ne "") {
        $Args += @("--mhc-model", $ModelPath)
    }
    
    # Run training
    $StartTime = Get-Date
    python @Args
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    # Check exit code
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for $Desc!" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "Completed $Desc in $($Duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Cyan
    Write-Host ""
    
    # Update model path for next language (continual training)
    $ModelPath = "$OutputDir/final"
}

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Training complete on all languages!" -ForegroundColor Cyan
Write-Host "Final model saved to: $ModelPath" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
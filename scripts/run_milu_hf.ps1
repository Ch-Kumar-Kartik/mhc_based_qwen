param(
    [string]$BaseModel = "Qwen/Qwen3-0.6B",
    [string]$MhcModel = "E:\mhc_based_qwen-2\output\qwen3_mhc_v2_parity_mixed_30k_1_latest\checkpoint-30000",
    [string]$Tasks = "milu",
    [string]$OutputDir = "E:\mhc_based_qwen-2\output\benchmark_milu_hf",
    [string]$BatchSize = "auto",
    [int]$TensorParallelSize = 1,
    [string]$DType = "bfloat16",
    [double]$GpuMemoryUtilization = 0.90,
    [int]$NumFewshot = 0,
    [int]$MaxModelLen = 0,
    [switch]$ApplyChatTemplate
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseOutput = Join-Path $OutputDir "milu_hf_base_$timestamp.json"
$mhcOutput = Join-Path $OutputDir "milu_hf_mhc_$timestamp.json"

function Get-ModelArgs([string]$ModelPath) {
    $args = @(
        "pretrained=$ModelPath",
        "dtype=$DType"
    )

    if ($MaxModelLen -gt 0) {
        $args += "max_model_len=$MaxModelLen"
    }

    return $args -join ","
}

function Invoke-MiluEval([string]$ModelPath, [string]$OutputPath) {
    $modelArgs = Get-ModelArgs -ModelPath $ModelPath
    $commandArgs = @(
        "-m", "lm_eval",
        "--model", "hf",
        "--model_args", $modelArgs,
        "--gen_kwargs", "temperature=0.0,top_p=1.0",
        "--tasks", $Tasks,
        "--batch_size", $BatchSize,
        "--log_samples",
        "--num_fewshot", $NumFewshot,
        "--output_path", $OutputPath
    )

    if ($ApplyChatTemplate) {
        $commandArgs += "--apply_chat_template"
    }

    Push-Location "E:\mhc_based_qwen-2\external\MILU"
    try {
        & python @commandArgs
    }
    finally {
        Pop-Location
    }
}

Write-Host "Running MILU (HF lm_eval) for base model: $BaseModel"
Invoke-MiluEval -ModelPath $BaseModel -OutputPath $baseOutput

Write-Host "Running MILU (HF lm_eval) for mHC checkpoint: $MhcModel"
Invoke-MiluEval -ModelPath $MhcModel -OutputPath $mhcOutput

Write-Host "Done. Outputs:"
Write-Host "Base: $baseOutput"
Write-Host "mHC : $mhcOutput"

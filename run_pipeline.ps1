# Full Watermarking Pipeline Runner (Undetectable + PRC)
# Runs only watermark experiments and requires an existing baseline file.
# Use run_baseline.ps1 to generate baseline_results.jsonl first.
# - Undetectable: loops over multiple lambda values
# - PRC: runs once (lambda is ignored by PRC)
#
# Examples:
#   powershell -ExecutionPolicy Bypass -File run_baseline.ps1
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Schemes Undetectable -Lambdas 5.0,7.0
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Schemes PRC
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Schemes PRC -NumSamples 5 -MaxTokens 150
#
param(
    [ValidateSet("Undetectable", "PRC")]
    [string[]]$Schemes = @("Undetectable", "PRC"),

    [double[]]$Lambdas = @(5.0, 7.0),  # Only used for Undetectable

    [ValidateSet("c4")]
    [string]$Dataset = "c4",

    [int]$MaxTokens = 150,
    [int]$NumSamples = 5,
    [string]$OutputDir = "outputs",

    [switch]$RunRobustness
)

# Ensure logs and outputs exist
if (!(Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

# ------------------------------------------------------------------ #
#  1. Baseline File Check                                             #
# ------------------------------------------------------------------ #
$baseline_path = Join-Path $OutputDir "baseline_results.jsonl"
if (!(Test-Path $baseline_path)) {
    throw "Baseline file not found: $baseline_path. Run run_baseline.ps1 first to generate baseline_results.jsonl."
}
Write-Host "--- Using existing baseline: $baseline_path ---" -ForegroundColor Cyan

# ------------------------------------------------------------------ #
#  2. EXPERIMENT LOOPS: Watermarked Generation + Evaluation          #
# ------------------------------------------------------------------ #
foreach ($scheme in $Schemes) {
    if ($scheme -eq "Undetectable") {
        foreach ($lam in $Lambdas) {
            # Ensure $lam is formatted as 5.0, 8.0 for naming
            $lam_str = $lam.ToString("F1")
            $suffix = "_lam$lam_str"
            $wm_results = Join-Path $OutputDir "undetectable_results$suffix.jsonl"
            $robust_out = Join-Path $OutputDir "robustness_results_undetectable_lam$lam_str.csv"

            Write-Host "`n====================================================" -ForegroundColor Magenta
            Write-Host "  RUNNING EXPERIMENT: $scheme (LAMBDA = $lam_str)" -ForegroundColor Magenta
            Write-Host "====================================================`n" -ForegroundColor Magenta

            # A. Generation
            Write-Host "--- [A] Generating Watermarked Text ($scheme$suffix) ---" -ForegroundColor Cyan
            python -m llm_watermarking.main --watermark $scheme --dataset $Dataset --max-tokens $MaxTokens --num-samples $NumSamples --lambda $lam --suffix $suffix --output-dir $OutputDir --no-plots

            # B. Detection and Quality Evaluation
            Write-Host "`n--- [B] Running Detection and Quality Evaluation ---" -ForegroundColor Cyan
            # Note: Table 1 appends, Table 2 is unique per run
            python -m llm_watermarking.evaluate_experiment --baseline $baseline_path --watermarked $wm_results

            if ($RunRobustness) {
                Write-Host "`n--- [C] Running Robustness Evaluation ---" -ForegroundColor Cyan
                python evaluate_robustness.py --results $wm_results --output $robust_out
            }
        }
    }
    elseif ($scheme -eq "PRC") {
        $suffix = "_c4_s$NumSamples" + "_t$MaxTokens"
        $wm_results = Join-Path $OutputDir "prc_results$suffix.jsonl"
        $robust_out = Join-Path $OutputDir "robustness_results_prc$suffix.csv"

        Write-Host "`n====================================================" -ForegroundColor Magenta
        Write-Host "  RUNNING EXPERIMENT: $scheme" -ForegroundColor Magenta
        Write-Host "====================================================`n" -ForegroundColor Magenta

        # A. Generation
        Write-Host "--- [A] Generating Watermarked Text ($scheme$suffix) ---" -ForegroundColor Cyan
        python -m llm_watermarking.main --watermark $scheme --dataset $Dataset --max-tokens $MaxTokens --num-samples $NumSamples --suffix $suffix --output-dir $OutputDir --no-plots

        # B. Detection and Quality Evaluation
        Write-Host "`n--- [B] Running Detection and Quality Evaluation ---" -ForegroundColor Cyan
        python -m llm_watermarking.evaluate_experiment --baseline $baseline_path --watermarked $wm_results

        if ($RunRobustness) {
            Write-Host "`n--- [C] Running Robustness Evaluation ---" -ForegroundColor Cyan
            python evaluate_robustness.py --results $wm_results --output $robust_out
        }
    }
    else {
        Write-Host "Skipping unknown scheme: $scheme" -ForegroundColor Yellow
    }
}

Write-Host "`n--- Pipeline Execution Complete! ---" -ForegroundColor Green
Write-Host "Check 'outputs/table1_detectability.csv' for the combined results."
Write-Host "Check 'outputs/table2_quality_metrics_*.csv' for individual quality stats."

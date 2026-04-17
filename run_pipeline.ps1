# Full Watermarking Pipeline Runner (Multiple Lambda Optimized)
# This script runs the baseline generation ONCE and then loops through
# multiple lambda values to compare their performance.

$lambdas = @(2.0, 5.0, 8.0, 10.0)
$max_tokens = 150
$num_samples = 50

# Ensure logs and outputs exist
if (!(Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" }

# ------------------------------------------------------------------ #
#  1. Baseline Generation (No Watermark) - Run ONCE                  #
# ------------------------------------------------------------------ #
Write-Host "--- [1/3] Starting Baseline Generation (Shared) ---" -ForegroundColor Cyan
python -m llm_watermarking.main --dataset c4 --max-tokens $max_tokens --num-samples $num_samples --output-dir outputs

# ------------------------------------------------------------------ #
#  2. EXPERIMENT LOOP: Watermarked Generation + Evaluation           #
# ------------------------------------------------------------------ #
foreach ($lam in $lambdas) {
    # Ensure $lam is formatted as 5.0, 8.0 for naming
    $lam_str = $lam.ToString("F1")
    $suffix = "_lam$lam_str"
    $wm_results = "outputs/undetectable_results$suffix.jsonl"
    $robust_out = "outputs/robustness_results_lam$lam_str.csv"

    Write-Host "`n====================================================" -ForegroundColor Magenta
    Write-Host "  RUNNING EXPERIMENT: LAMBDA = $lam_str" -ForegroundColor Magenta
    Write-Host "====================================================`n" -ForegroundColor Magenta

    # A. Generation
    Write-Host "--- [A] Generating Watermarked Text ($suffix) ---" -ForegroundColor Cyan
    python -m llm_watermarking.main --watermark Undetectable --dataset c4 --max-tokens $max_tokens --num-samples $num_samples --lambda $lam --suffix $suffix --output-dir outputs

    # B. Detection and Quality Evaluation
    Write-Host "`n--- [B] Running Detection and Quality Evaluation ---" -ForegroundColor Cyan
    # Note: Table 1 appends, Table 2 is unique per lambda
    python -m llm_watermarking.evaluate_experiment --baseline outputs/baseline_results.jsonl --watermarked $wm_results

    # C. Robustness Evaluation
    Write-Host "`n--- [C] Running Robustness Evaluation ---" -ForegroundColor Cyan
    python evaluate_robustness.py --results $wm_results --output $robust_out
}

Write-Host "`n--- Pipeline Execution Complete! ---" -ForegroundColor Green
Write-Host "Check 'outputs/table1_detectability.csv' for the combined results."
Write-Host "Check 'outputs/table2_quality_metrics_lamX.csv' for individual quality stats."

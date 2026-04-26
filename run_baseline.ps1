param(
    [ValidateSet("c4")]
    [string]$Dataset = "c4",

    [int]$MaxTokens = 150,
    [int]$NumSamples = 10,
    [string]$OutputDir = "outputs"
)

if (!(Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$baseline_path = Join-Path $OutputDir "baseline_results.jsonl"

Write-Host "--- Generating baseline results to $baseline_path ---" -ForegroundColor Cyan
python -m llm_watermarking.main --dataset $Dataset --max-tokens $MaxTokens --num-samples $NumSamples --output-dir $OutputDir --no-plots

if (Test-Path $baseline_path) {
    Write-Host "--- Baseline generation complete: $baseline_path ---" -ForegroundColor Green
}
else {
    throw "Baseline generation failed: $baseline_path not found. Check the Python logs for errors."
}

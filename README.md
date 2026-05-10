# LLM Text Watermarking

Implementation project for cryptographic watermarking of LLM-generated text. The repository implements two generation-time watermarking schemes:

- **Undetectable watermarking** using an entropy-based anchor and keyed pseudorandom continuation.
- **PRC watermarking** using a pseudorandom error-correcting-code style signal over a binary token representation.

The experiments use `microsoft/phi-2`, C4 validation prompts, and evaluate detectability, quality, entropy behavior, and robustness under random substitution noise.

## Repository Structure

- `llm_watermarking/`: implementation code for generation, watermarking, detection, evaluation, and plotting.
- `llm_watermarking/watermarks/undetectable/`: Undetectable watermark generator and detector.
- `llm_watermarking/watermarks/prc/`: PRC generator, detector, and PRC primitive.
- `outputs/`: selected final experiment outputs used in the report.
- `project report/`: final LaTeX report and report figures.
- `run_baseline.ps1`: baseline generation runner.
- `run_pipeline.ps1`: watermark experiment runner.
- `run_commands.txt`: exact commands for reproducing the final experiments.

## Setup

```powershell
pip install -r llm_watermarking/requirements.txt
```

The model is loaded from Hugging Face, so the first run may take time and requires enough disk space/GPU memory for `microsoft/phi-2`.

## Main Reproduction Commands

Generate the baseline:

```powershell
powershell -ExecutionPolicy Bypass -File run_baseline.ps1 -Dataset c4 -NumSamples 100 -MaxTokens 150 -OutputDir outputs
```

Run the final Undetectable watermark experiments:

```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Schemes Undetectable -Lambdas 2.0,4.0,5.0,6.0 -NumSamples 100 -MaxTokens 150 -OutputDir outputs
```

Run the main PRC experiment:

```powershell
python -m llm_watermarking.main --watermark PRC --dataset c4 --max-tokens 150 --num-samples 100 --suffix _c4_s100_t150 --output-dir outputs --no-plots --prc-n 2048 --prc-g 128 --prc-t 1 --prc-r 2028 --prc-eta 0.005 --prc-zeta 0.045
```

Additional PRC settings used in the report and all evaluation commands are listed in `run_commands.txt`.

## Important Outputs

- `outputs/table1_detectability.csv`: detectability metrics.
- `outputs/table2_quality_*.csv`: quality metrics.
- `outputs/table3_robustness_prc_c4_s100_t150.csv`: PRC robustness results.
- `outputs/robustness_results_undetectable_lam5.0.csv`: Undetectable robustness results.
- `outputs/*png`: selected figures used in the report.


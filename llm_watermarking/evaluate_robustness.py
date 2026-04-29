import argparse
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from llm_watermarking.config import Config
from llm_watermarking.model_loader import only_load_tokenizer
from llm_watermarking.robustness import TextModifier
from llm_watermarking.watermarks.prc import LDPCPRC0Params, PRCWatermarkDetector
from llm_watermarking.watermarks.undetectable import WatermarkDetector


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_label_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_results", "", 1)


def infer_metadata(results, results_path):
    first = results[0] if results else {}
    prc_p = first.get("prc_params")
    run_label = run_label_from_path(results_path)

    if isinstance(prc_p, dict):
        return {
            "scheme": "PRC",
            "run_label": run_label,
            "parameter": (
                f"n={int(prc_p['n'])};g={int(prc_p['g'])};t={int(prc_p['t'])};"
                f"r={int(prc_p['r'])};eta={float(prc_p['eta'])};zeta={float(prc_p['zeta'])}"
            ),
        }

    lam = first.get("lambda_entropy", 5.0)
    return {
        "scheme": "Undetectable",
        "run_label": run_label,
        "parameter": f"lambda={lam}",
    }


def build_detector(original_results, tokenizer):
    key_hex = original_results[0].get("key_hex")
    lam = original_results[0].get("lambda_entropy", 16.0)
    prc_p = original_results[0].get("prc_params")

    if not key_hex:
        raise ValueError("No key found in results.")

    if isinstance(prc_p, dict):
        params = LDPCPRC0Params(
            n=int(prc_p["n"]),
            g=int(prc_p["g"]),
            t=int(prc_p["t"]),
            r=int(prc_p["r"]),
            eta=float(prc_p["eta"]),
            zeta=float(prc_p["zeta"]),
        )
        return PRCWatermarkDetector(bytes.fromhex(key_hex), params, tokenizer=tokenizer, robust_scan=True)

    return WatermarkDetector(bytes.fromhex(key_hex), lam, tokenizer=tokenizer)


def default_noise_levels():
    return [round(i * 0.05, 2) for i in range(21)]


def default_figure_path(output_path):
    root, _ = os.path.splitext(output_path)
    return f"{root}.png"


def plot_robustness(df, figure_path, title=None):
    os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["NoiseLevel"], df["TPR"], marker="o", linewidth=2, label="TPR")

    plt.xlabel("Noise level")
    plt.ylabel("True positive rate")
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title or "Watermark Robustness Across Noise Levels")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


def evaluate_robustness(results_path, output_path, noise_levels, figure_path=None):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading results from {results_path}...")
    original_results = load_jsonl(results_path)

    if not original_results:
        raise ValueError("No results found.")

    cfg = Config()
    tokenizer = only_load_tokenizer(cfg)
    modifier = TextModifier(seed=42)
    metadata = infer_metadata(original_results, results_path)
    detector = build_detector(original_results, tokenizer)

    summary_data = []

    for p in noise_levels:
        print(f"\n--- Testing Noise Level: {p * 100:.0f}% ---")
        detected_count = 0

        for res in tqdm(original_results, desc=f"watermarked noise={p:g}"):
            modified_text = modifier.apply_substitutions(res["generated_text"], p)
            test_res = {
                "generated_text": modified_text,
                "prompt": res["prompt"],
            }

            det = detector.detect(test_res)
            is_detected = det["detected"]

            if is_detected:
                detected_count += 1

        total = len(original_results)
        missed_count = total - detected_count
        tpr = detected_count / total if total else 0.0
        fnr = missed_count / total if total else 0.0

        row = {
            "Scheme": metadata["scheme"],
            "RunLabel": metadata["run_label"],
            "Parameter": metadata["parameter"],
            "NoiseLevel": p,
            "TPR": round(tpr, 4),
            "FNR": round(fnr, 4),
            "Detected": detected_count,
            "Missed": missed_count,
            "Total": total,
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nRobustness results saved to {output_path}")

    figure_path = figure_path or default_figure_path(output_path)
    plot_title = f"{metadata['scheme']} Watermark Robustness Across Noise Levels"
    plot_robustness(df, figure_path, title=plot_title)
    print(f"Robustness figure saved to {figure_path}")

    print("\nSummary Table:")
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Watermark Robustness")
    parser.add_argument("--results", type=str, default="outputs/undetectable_results.jsonl")
    parser.add_argument("--output", type=str, default="outputs/robustness_results.csv")
    parser.add_argument("--figure", type=str, default=None)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=default_noise_levels())
    args = parser.parse_args()

    try:
        evaluate_robustness(args.results, args.output, args.noise_levels, args.figure)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

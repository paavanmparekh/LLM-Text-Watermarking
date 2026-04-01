import argparse
import math
import os

import pandas as pd
import numpy as np

from llm_watermarking.pipeline import load_results
from llm_watermarking.config import Config
from llm_watermarking.model_loader import load_model_and_tokenizer
from llm_watermarking.watermarks.undetectable import WatermarkDetector


def check_detected(stat, n_rem, lam):
    if n_rem == 0:
        return False
    return stat > float(n_rem) + lam * math.sqrt(float(n_rem))


def evaluate_single_lambda(base_stats, base_bits, wm_stats, wm_bits, lam):
    tp = sum(check_detected(s, b, lam) for s, b in zip(wm_stats, wm_bits))
    fn = len(wm_stats) - tp
    
    fp = sum(check_detected(s, b, lam) for s, b in zip(base_stats, base_bits))
    tn = len(base_stats) - fp

    tpr = tp / len(wm_stats) if len(wm_stats) > 0 else 0.0
    fpr = fp / len(base_stats) if len(base_stats) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    row = {
        "Lambda": lam,
        "TPR": round(tpr, 4),
        "FNR": round(fnr, 4),
        "TNR": round(tnr, 4),
        "FPR": round(fpr, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Accuracy": round(acc, 4)
    }

    df = pd.DataFrame([row])
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/table1_detectability.csv"
    
    # Append if file exists so you can build a table of different lambdas
    if os.path.exists(out_path):
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
        
    print(f"Detectability for Lambda={lam} appended to -> {out_path}")
    print(df.to_string(index=False))


def generate_table_2_csv(base_results, wm_results, lam):
    def get_metric(results, key):
        return [r["eval"][key] for r in results if "eval" in r and key in r["eval"]]
        
    base_ppl = get_metric(base_results, "perplexity")
    wm_ppl = get_metric(wm_results, "perplexity")
    base_log = get_metric(base_results, "log_diversity")
    wm_log = get_metric(wm_results, "log_diversity")

    def mean_std(arr):
        if not arr: return 0.0, 0.0
        return np.mean(arr), np.std(arr)

    b_ppl_m, b_ppl_s = mean_std(base_ppl)
    w_ppl_m, w_ppl_s = mean_std(wm_ppl)
    b_log_m, _ = mean_std(base_log)
    w_log_m, _ = mean_std(wm_log)

    rows = [
        {
            "Lambda_Run": lam,
            "Metric": "Perplexity",
            "Watermarked_Mean": round(w_ppl_m, 2),
            "Watermarked_Std": round(w_ppl_s, 2),
            "Unwatermarked_Mean": round(b_ppl_m, 2),
            "Unwatermarked_Std": round(b_ppl_s, 2),
            "Delta": round(w_ppl_m - b_ppl_m, 3)
        },
        {
            "Lambda_Run": lam,
            "Metric": "Log_Diversity",
            "Watermarked_Mean": round(w_log_m, 3),
            "Watermarked_Std": 0.0,
            "Unwatermarked_Mean": round(b_log_m, 3),
            "Unwatermarked_Std": 0.0,
            "Delta": round(w_log_m - b_log_m, 3)
        }
    ]
    df = pd.DataFrame(rows)
    out_path = "outputs/table2_quality_metrics.csv"
    
    if os.path.exists(out_path):
        df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
        
    print(f"Quality metrics for Lambda={lam} appended to -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/baseline_results.jsonl")
    parser.add_argument("--watermarked", default="outputs/undetectable_results.jsonl")
    args = parser.parse_args()

    # Load results
    print(f"Loading baseline: {args.baseline}")
    base_results = load_results(args.baseline)
    print(f"Loading watermarked: {args.watermarked}")
    wm_results = load_results(args.watermarked)

    if not wm_results:
        print("No watermarked results found. Please generate them first.")
        return

    # Infer key and lambda
    key_hex = wm_results[0].get("key_hex")
    lam = wm_results[0].get("lambda_entropy", 5.0)

    if not key_hex:
        print("Could not find key_hex in watermarked results. Aborting.")
        return

    # Setup detector
    cfg = Config()
    _, tokenizer = load_model_and_tokenizer(cfg)
    detector = WatermarkDetector(bytes.fromhex(key_hex), lam, tokenizer=tokenizer)

    print("\n--- Running detection on Base Responses (unwatermarked) ---")
    base_stats, base_bits = [], []
    for res in base_results:
        det = detector.detect(res)
        base_stats.append(det["stat"])
        base_bits.append(det["num_bits"])

    print("--- Running detection on Watermarked Responses ---")
    wm_stats, wm_bits = [], []
    for res in wm_results:
        det = detector.detect(res)
        wm_stats.append(det["stat"])
        wm_bits.append(det["num_bits"])

    print("\n" + "="*80)
    evaluate_single_lambda(base_stats, base_bits, wm_stats, wm_bits, lam)
    generate_table_2_csv(base_results, wm_results, lam)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

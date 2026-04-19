import argparse
import math
import os
import json

import pandas as pd
import numpy as np

from llm_watermarking.pipeline import load_results
from llm_watermarking.config import Config
from llm_watermarking.model_loader import only_load_tokenizer
from llm_watermarking.watermarks.undetectable import WatermarkDetector


def evaluate_single_lambda(base_detects, wm_detects, lam):
    tp = sum(wm_detects)
    fn = len(wm_detects) - tp
    
    fp = sum(base_detects)
    tn = len(base_detects) - fp

    tpr = tp / len(wm_detects) if len(wm_detects) > 0 else 0.0
    fpr = fp / len(base_detects) if len(base_detects) > 0 else 0.0
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
    
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        if "Lambda" not in existing_df.columns:
            df.to_csv(out_path, index=False)
        else:
            df.to_csv(out_path, mode='a', header=False, index=False)
    else:
        df.to_csv(out_path, index=False)
        
    print(f"Detectability for Lambda={lam} appended to -> {out_path}")
    print(df.to_string(index=False))


def generate_table_2_csv(base_results, wm_results, out_path="outputs/table2_quality_metrics.csv"):
    rows = []
    
    base_ppls = []
    wm_ppls = []
    base_divs = []
    wm_divs = []

    for b_res, w_res in zip(base_results, wm_results):
        prompt = b_res.get("prompt", "")
        nw_text = b_res.get("generated_text", "")
        w_text = w_res.get("generated_text", "")
        
        w_eval = w_res.get("eval", {})
        b_eval = b_res.get("eval", {})
        
        b_ppl = b_eval.get("perplexity", 0.0)
        w_ppl = w_eval.get("perplexity", 0.0)
        b_div = b_eval.get("log_diversity", 0.0)
        w_div = w_eval.get("log_diversity", 0.0)
        
        if isinstance(b_ppl, float): base_ppls.append(b_ppl)
        if isinstance(w_ppl, float): wm_ppls.append(w_ppl)
        if isinstance(b_div, float): base_divs.append(b_div)
        if isinstance(w_div, float): wm_divs.append(w_div)
        
        rows.append({
            "Prompt": prompt,
            "No Watermarked Response": nw_text,
            "Watermarked Response": w_text,
            "PPL (No Watermarked)": round(b_ppl, 2) if b_ppl else "",
            "PPL (Watermarked)": round(w_ppl, 2) if w_ppl else "",
            "Log Div (No Watermarked)": round(b_div, 3) if b_div else "",
            "Log Div (Watermarked)": round(w_div, 3) if w_div else ""
        })

    # Deltas
    b_ppl_m = np.mean(base_ppls) if base_ppls else 0.0
    w_ppl_m = np.mean(wm_ppls) if wm_ppls else 0.0
    delta_ppl = w_ppl_m - b_ppl_m

    b_div_m = np.mean(base_divs) if base_divs else 0.0
    w_div_m = np.mean(wm_divs) if wm_divs else 0.0
    delta_div = w_div_m - b_div_m

    rows.append({
        "Prompt": "--- DELTAS ---",
        "No Watermarked Response": "",
        "Watermarked Response": "",
        "PPL (No Watermarked)": f"Delta PPL: {delta_ppl:.3f}",
        "PPL (Watermarked)": "",
        "Log Div (No Watermarked)": f"Delta Div: {delta_div:.3f}",
        "Log Div (Watermarked)": ""
    })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
        
    print(f"Table 2 (Quality metrics) saved to -> {out_path}")


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
    tokenizer = only_load_tokenizer(cfg)
    detector = WatermarkDetector(bytes.fromhex(key_hex), lam, tokenizer=tokenizer)

    print("\n--- Running detection on Base Responses (unwatermarked) ---")
    base_detects = []
    for res in base_results:
        det = detector.detect(res)
        base_detects.append(det["detected"])
        res["Watermark_Detected"] = det["detected"]

    print("--- Running detection on Watermarked Responses ---")
    wm_detects = []
    for res in wm_results:
        det = detector.detect(res)
        wm_detects.append(det["detected"])
        res["Watermark_Detected"] = det["detected"]
        
    def format_df(results_list, is_wm=False):
        formatted = []
        for r in results_list:
            row = r.copy()
            # Extract log_diversity and perplexity from eval
            if "eval" in row:
                ev = row.pop("eval")
                if isinstance(ev, dict):
                    row["log_diversity"] = ev.get("log_diversity", "")
                    row["perplexity"] = ev.get("perplexity", "")
            
            # Remove unwanted columns globally
            row.pop("mode", None)
            row.pop("Detection_Score", None)
            
            if is_wm:
                row.pop("bit_surprisals", None)
                row.pop("key_hex", None)
                row.pop("lambda_entropy", None)
            formatted.append(row)
        return pd.DataFrame(formatted)

    # Overwrite the CSV files with the appended detection features
    csv_wm_path = args.watermarked.replace('.jsonl', '.csv')
    df_wm = format_df(wm_results, is_wm=True)
    df_wm.to_csv(csv_wm_path, index=False)
    print(f"Updated {csv_wm_path} with pristine detection columns!")

    csv_base_path = args.baseline.replace('.jsonl', '.csv')
    df_base = format_df(base_results, is_wm=False)
    df_base.to_csv(csv_base_path, index=False)
    print(f"Updated {csv_base_path} with pristine detection columns!")

    print("\n" + "="*80)
    evaluate_single_lambda(base_detects, wm_detects, lam)
    
    # Generate a lambda-specific name for Table 2
    table2_path = f"outputs/table2_quality_metrics_lam{lam}.csv"
    generate_table_2_csv(base_results, wm_results, out_path=table2_path)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

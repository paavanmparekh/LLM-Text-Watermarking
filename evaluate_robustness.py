import json
import os
import argparse
import pandas as pd
from tqdm import tqdm

from llm_watermarking.config import Config
from llm_watermarking.model_loader import load_model_and_tokenizer
from llm_watermarking.watermarks.undetectable import WatermarkDetector
from llm_watermarking.watermarks.prc import PRCWatermarkDetector, LDPCPRC0Params
from llm_watermarking.robustness import TextModifier

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Watermark Robustness")
    parser.add_argument("--results", type=str, default="outputs/undetectable_results.jsonl")
    parser.add_argument("--output", type=str, default="outputs/robustness_results.csv")
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found.")
        return

    print(f"Loading results from {args.results}...")
    original_results = load_jsonl(args.results)
    
    if not original_results:
        print("No results found.")
        return

    # Initialize tokenizer for detector
    cfg = Config()
    from llm_watermarking.model_loader import only_load_tokenizer
    tokenizer = only_load_tokenizer(cfg)
    
    modifier = TextModifier(seed=42)
    
    # We'll use the key and scheme params from the first result (assuming consistency).
    key_hex = original_results[0].get("key_hex")
    lam = original_results[0].get("lambda_entropy", 16.0)
    prc_p = original_results[0].get("prc_params")
    metadata = infer_metadata(original_results, args.results)
    
    if not key_hex:
        print("Error: No key found in results.")
        return
    
    if isinstance(prc_p, dict):
        params = LDPCPRC0Params(
            n=int(prc_p["n"]),
            g=int(prc_p["g"]),
            t=int(prc_p["t"]),
            r=int(prc_p["r"]),
            eta=float(prc_p["eta"]),
            zeta=float(prc_p["zeta"]),
        )
        detector = PRCWatermarkDetector(bytes.fromhex(key_hex), params, tokenizer=tokenizer, robust_scan=True)
    else:
        detector = WatermarkDetector(bytes.fromhex(key_hex), lam, tokenizer=tokenizer)
    
    summary_data = []

    for p in args.noise_levels:
        print(f"\n--- Testing Noise Level: {p*100:.0f}% ---")
        
        detected_count = 0
        
        for i, res in enumerate(original_results):
            # 1. Apply modification (Substitution + Insertion mix)
            # For simplicity, we just do substitution here, or a mix.
            # Let's do substitutions as it's the most common attack.
            modified_text = modifier.apply_substitutions(res["generated_text"], p)
            
            # Prepare result dict for detector
            test_res = {
                "generated_text": modified_text,
                "prompt": res["prompt"]
            }
            
            # 2. Run Detection
            det = detector.detect(test_res)
            
            is_detected = det["detected"]
            score = det["detection_score"]
            
            if is_detected:
                detected_count += 1
            
            print(f"  Sample {i+1}: Score={score:.2f} | Detected={is_detected}")

        total = len(original_results)
        missed_count = total - detected_count
        tpr = detected_count / total if total else 0.0
        fnr = missed_count / total if total else 0.0

        summary_data.append({
            "Scheme": metadata["scheme"],
            "RunLabel": metadata["run_label"],
            "Parameter": metadata["parameter"],
            "NoiseLevel": p,
            "TPR": round(tpr, 4),
            "FNR": round(fnr, 4),
            "Detected": detected_count,
            "Missed": missed_count,
            "Total": total,
        })

    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(args.output, index=False)
    print(f"\nRobustness results saved to {args.output}")
    
    # Print high-level summary
    print("\nSummary Table:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

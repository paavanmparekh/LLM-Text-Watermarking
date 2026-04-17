import json
import os
import argparse
import pandas as pd
from tqdm import tqdm

from llm_watermarking.config import Config
from llm_watermarking.model_loader import load_model_and_tokenizer
from llm_watermarking.watermarks.undetectable import WatermarkDetector
from llm_watermarking.robustness import TextModifier

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

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

    # Initialize model/tokenizer for detector
    cfg = Config()
    _, tokenizer = load_model_and_tokenizer(cfg)
    
    modifier = TextModifier(seed=42)
    
    # We'll use the key and lambda from the first result (assuming they are consistent)
    key_hex = original_results[0].get("key_hex")
    lam = original_results[0].get("lambda_entropy", 16.0)
    bit_length = original_results[0].get("bit_length", 16)
    
    if not key_hex:
        print("Error: No key found in results.")
        return
    
    detector = WatermarkDetector(bytes.fromhex(key_hex), lam, tokenizer=tokenizer)
    
    summary_data = []

    for p in args.noise_levels:
        print(f"\n--- Testing Noise Level: {p*100:.0f}% ---")
        
        detected_count = 0
        total_score = 0.0
        
        for i, res in enumerate(original_results):
            # 1. Apply modification (Substitution + Insertion mix)
            # For simplicity, we just do substitution here, or a mix.
            # Let's do substitutions as it's the most common attack.
            modified_text = modifier.apply_substitutions(res["generated_text"], p)
            
            # Prepare result dict for detector
            test_res = {
                "generated_text": modified_text,
                "bit_length": bit_length,
                "prompt": res["prompt"]
            }
            
            # 2. Run Detection
            det = detector.detect(test_res)
            
            is_detected = det["detected"]
            score = det["detection_score"]
            
            if is_detected:
                detected_count += 1
            total_score += score
            
            print(f"  Sample {i+1}: Score={score:.2f} | Detected={is_detected}")
            
            summary_data.append({
                "noise_level": p,
                "sample_idx": i,
                "detected": is_detected,
            })

    # Save to CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(args.output, index=False)
    print(f"\nRobustness results saved to {args.output}")
    
    # Print high-level summary
    summary = df.groupby("noise_level").agg({
        "detected": "mean",
    }).rename(columns={"detected": "TPR"})
    print("\nSummary Table:")
    print(summary)

if __name__ == "__main__":
    main()

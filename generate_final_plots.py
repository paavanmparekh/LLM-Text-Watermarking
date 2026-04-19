import json
import os
from llm_watermarking.visualization import plot_evaluation_metrics, plot_total_entropy_vs_detection

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    results_5 = load_jsonl("outputs/undetectable_results_lam5.0.jsonl")
    results_8 = load_jsonl("outputs/undetectable_results_lam8.0.jsonl")
    
    # Combined plot
    all_results = results_5 + results_8
    plot_evaluation_metrics(all_results, output_dir="outputs/")
    print("Combined plot generated: outputs/entropy_vs_detection.png")
    
    # New plot: Total Empirical Entropy vs Detection
    csv_paths = [
        "outputs/undetectable_results_lam4.0.csv",
        "outputs/undetectable_results_lam5.0.csv",
        "outputs/undetectable_results_lam6.0.csv",
        "outputs/undetectable_results_lam7.0.csv",
        "outputs/undetectable_results_lam8.0.csv"
    ]
    plot_total_entropy_vs_detection(csv_paths, output_dir="outputs/")
    print("Total entropy vs detection plot generated: outputs/total_entropy_vs_detection.png")

if __name__ == "__main__":
    main()

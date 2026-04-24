"""
visualization.py — Plotting utilities for the LLM Watermarking project.

All functions accept ``results`` (a list of the dicts produced by
pipeline.run_pipeline) and an optional ``output_dir`` where figures
are saved.  If ``output_dir`` is None, figures are shown interactively.

Entropy Plots
-------------
plot_evaluation_metrics     : convenience wrapper (entropy + detection if present)
"""

import os
from typing import List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _save_or_show(fig: plt.Figure, filename: str, output_dir: Optional[str]) -> None:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {path}")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------ #
#  2. Entropy vs Detection                                            #
# ------------------------------------------------------------------ #

def plot_detection_vs_entropy(
    results: List[dict],
    output_dir: Optional[str] = None,
    filename: str = "entropy_vs_detection.png"
) -> None:
    """Box plot + Jitter: Average Empirical Entropy vs Detection Status."""
    sns.set_theme(style="whitegrid")
    
    data = []
    for r in results:
        ev = r.get("eval", {})
        entropy = ev.get("avg_empirical_entropy")
        if entropy is None:
            total_entropy = r.get("total_empirical_entropy")
            token_count = r.get("detector_token_count") or r.get("num_tokens")
            if total_entropy is not None and token_count:
                entropy = total_entropy / token_count
        det = r.get("detection", {})
        
        if det:
            status = "Yes" if det.get("detected") else "No"
        else:
            status = "N/A (Baseline)" if r.get("mode") == "baseline" else "N/A (No Detection)"
            
        if entropy is not None:
            data.append({"Entropy": entropy, "Detected": status})

    if not data:
        print("No data available for Detection vs Entropy plot.")
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ensure order if both exist
    order = ["No", "Yes"] if "Yes" in df["Detected"].values and "No" in df["Detected"].values else None
    if order is None and "N/A (Baseline)" in df["Detected"].values and "N/A (No Detection)" not in df["Detected"].values:
        order = ["N/A (Baseline)"]

    sns.boxplot(
        data=df, x="Detected", y="Entropy", ax=ax,
        palette="husl", showfliers=False, order=order
    )
    sns.stripplot(
        data=df, x="Detected", y="Entropy", ax=ax,
        color=".3", alpha=0.5, jitter=True, order=order
    )

    ax.set_title("Avg Empirical Entropy vs Watermark Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Watermark Detected")
    ax.set_ylabel("Average Empirical Entropy (bits/token)")

    plt.tight_layout()
    _save_or_show(fig, filename, output_dir)



# ------------------------------------------------------------------ #
#  3. Total Empirical Entropy vs Detection (from CSV)                #
# ------------------------------------------------------------------ #

def discover_watermarked_csvs(
    output_dir: str = "outputs",
    pattern: str = "undetectable_results_lam*.csv",
) -> List[str]:
    """Find watermarked result CSVs (typically written by evaluate_experiment.py)."""
    base = Path(output_dir)
    return sorted(str(p) for p in base.glob(pattern) if p.is_file())


def plot_entropy_vs_detection_from_csvs(
    csv_paths: List[str],
    output_dir: Optional[str] = None,
    filename: str = "entropy_vs_detection.png",
) -> None:
    """
    Box plot + Jitter: Avg empirical entropy (bits/token) vs Detection (from CSV data).

    Required columns:
      - total_empirical_entropy
      - Watermark_Detected
    Token count column (first available):
      - detector_token_count
      - num_tokens
    """
    sns.set_theme(style="whitegrid")

    data = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "total_empirical_entropy" not in df.columns or "Watermark_Detected" not in df.columns:
            print(f"Skipping {csv_path}: Missing required columns.")
            continue

        token_col = "detector_token_count" if "detector_token_count" in df.columns else ("num_tokens" if "num_tokens" in df.columns else None)

        for _, row in df.iterrows():
            total_entropy = row.get("total_empirical_entropy")
            token_count = row.get(token_col) if token_col else None
            if pd.isna(total_entropy) or token_count in (None, 0) or pd.isna(token_count):
                continue
            avg_entropy = float(total_entropy) / float(token_count)
            detected = "Yes" if bool(row.get("Watermark_Detected")) else "No"
            data.append({"Entropy": avg_entropy, "Detected": detected})

    if not data:
        print("No data available for Entropy vs Detection plot.")
        return

    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))

    order = ["No", "Yes"] if {"Yes", "No"}.issubset(set(df_plot["Detected"].values)) else None
    sns.boxplot(
        data=df_plot, x="Detected", y="Entropy", ax=ax,
        palette={"Yes": "green", "No": "red"}, showfliers=False, order=order
    )
    sns.stripplot(
        data=df_plot, x="Detected", y="Entropy", ax=ax,
        color=".3", alpha=0.5, jitter=True, order=order
    )

    ax.set_title("Avg Empirical Entropy vs Watermark Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Watermark Detected")
    ax.set_ylabel("Average Empirical Entropy (bits/token)")

    plt.tight_layout()
    _save_or_show(fig, filename, output_dir)

def plot_total_entropy_vs_detection(
    csv_paths: List[str],
    output_dir: Optional[str] = None,
    filename: str = "total_entropy_vs_detection.png"
) -> None:
    """Box plot + Jitter: Total Empirical Entropy vs Detection Status (from CSV data)."""
    sns.set_theme(style="whitegrid")
    
    data = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "total_empirical_entropy" not in df.columns or "Watermark_Detected" not in df.columns:
            print(f"Skipping {csv_path}: Missing required columns.")
            continue
        
        # Extract lambda from filename (e.g., '4.0' from 'undetectable_results_lam4.0.csv')
        lam = csv_path.split("lam")[-1].split(".csv")[0]
        
        for _, row in df.iterrows():
            entropy = row["total_empirical_entropy"]
            detected = "Yes" if row["Watermark_Detected"] else "No"
            data.append({"Entropy": entropy, "Detected": detected, "Lambda": lam})

    if not data:
        print("No data available for Total Entropy vs Detection plot.")
        return

    df_plot = pd.DataFrame(data)
    
    # Use catplot to facet by Lambda
    g = sns.catplot(
        data=df_plot, kind="box", col="Lambda", x="Detected", y="Entropy",
        palette={"Yes": "green", "No": "red"}, showfliers=False, height=4, aspect=0.7
    )
    
    # Add stripplot for jitter
    for ax, lam in zip(g.axes.flat, g.col_names):
        subset = df_plot[df_plot["Lambda"] == lam]
        sns.stripplot(
            data=subset, x="Detected", y="Entropy", ax=ax,
            color=".3", alpha=0.5, jitter=True
        )
    
    g.set_titles("Lambda {col_name}")
    g.set_axis_labels("Watermark Detected", "Total Empirical Entropy (bits)")
    g.fig.suptitle("Total Empirical Entropy vs Watermark Detection", fontsize=14, fontweight="bold")
    g.fig.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        g.fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved -> {path}")
        plt.close(g.fig)
    else:
        plt.show()



def plot_evaluation_metrics(
    results: List[dict],
    output_dir: Optional[str] = None,
    suffix: str = ""
) -> None:
    """
    Run the Entropy vs Detection plot.

    Parameters
    ----------
    results : list of dicts from pipeline.run_pipeline().
    output_dir : str, optional
    suffix : str, optional
        Suffix for the filename (e.g. '_lam5.0')
    """
    filename = f"entropy_vs_detection{suffix}.png"
    plot_detection_vs_entropy(results, output_dir, filename=filename)


def generate_final_plots_from_outputs(output_dir: str = "outputs") -> None:
    """Generate per-lambda plots by scanning `output_dir` for result CSVs."""
    csv_paths = discover_watermarked_csvs(output_dir=output_dir)
    if not csv_paths:
        print(f"No watermarked CSVs found in {output_dir}.")
        return

    for csv_path in csv_paths:
        lam = str(csv_path).split("lam")[-1].split(".csv")[0]
        plot_entropy_vs_detection_from_csvs(
            [csv_path],
            output_dir=output_dir,
            filename=f"entropy_vs_detection_lam{lam}.png",
        )
        plot_total_entropy_vs_detection(
            [csv_path],
            output_dir=output_dir,
            filename=f"total_entropy_vs_detection_lam{lam}.png",
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate final watermarking plots from outputs/*.csv")
    parser.add_argument("--output-dir", default="outputs", help="Directory containing result CSVs.")
    parser.add_argument("--combined", action="store_true", help="Also write combined plots across all lambdas.")
    args = parser.parse_args()

    generate_final_plots_from_outputs(output_dir=args.output_dir)
    if args.combined:
        csv_paths = discover_watermarked_csvs(output_dir=args.output_dir)
        plot_entropy_vs_detection_from_csvs(csv_paths, output_dir=args.output_dir, filename="entropy_vs_detection.png")
        plot_total_entropy_vs_detection(csv_paths, output_dir=args.output_dir, filename="total_entropy_vs_detection.png")


if __name__ == "__main__":
    main()

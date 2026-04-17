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
        det = r.get("detection", {})
        
        if not det:
            status = "N/A (Baseline)"
        else:
            status = "Yes" if det.get("detected") else "No"
            
        if entropy is not None:
            data.append({"Entropy": entropy, "Detected": status})

    if not data:
        print("No data available for Detection vs Entropy plot.")
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ensure order if both exist
    order = ["No", "Yes"] if "Yes" in df["Detected"].values and "No" in df["Detected"].values else None
    if order is None and "N/A (Baseline)" in df["Detected"].values:
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

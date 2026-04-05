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
import seaborn as sns


# ------------------------------------------------------------------ #
#  Helper                                                             #
# ------------------------------------------------------------------ #

def _save_or_show(fig: plt.Figure, filename: str, output_dir: Optional[str]) -> None:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved → {path}")
        plt.close(fig)
    else:
        plt.show()


def _prompt_label(result: dict, idx: int, max_chars: int = 30) -> str:
    raw = result["prompt"].replace("[INST]", "").replace("[/INST]", "").strip()
    return f"P{idx+1}" if len(raw) > max_chars else f"P{idx+1}"


# ------------------------------------------------------------------ #
#  2. Entropy Distributions (Box Plots)                               #
# ------------------------------------------------------------------ #

def plot_entropy_distributions(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """Side-by-side box plots of per-token surprisal for each prompt."""
    sns.set_theme(style="whitegrid")
    labels = [_prompt_label(r, i) for i, r in enumerate(results)]
    data   = [r["token_surprisals"] for r in results]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(results)), 5))
    ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="navy"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5),
    )
    ax.set_title("Distribution of Empirical Entropy per Prompt", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Empirical Entropy (bits/token)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "entropy_distributions.png", output_dir)


# ------------------------------------------------------------------ #
#  3. Aggregate Totals (Bar Chart)                                    #
# ------------------------------------------------------------------ #

def plot_entropy_aggregate(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart: total Shannon vs total empirical entropy per prompt."""
    sns.set_theme(style="whitegrid")
    labels     = [f"P{i+1}" for i in range(len(results))]
    total_emp  = [r["eval"]["total_empirical_entropy"] for r in results]
    total_shan = [r["eval"]["total_shannon_entropy"]   for r in results]
    x, w       = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(results)), 5))
    ax.bar(x - w / 2, total_emp,  w, label="Total Empirical Entropy", color="darkorange")
    ax.bar(x + w / 2, total_shan, w, label="Total Shannon Entropy",   color="steelblue")
    ax.set_title("Total Empirical vs Total Shannon Entropy per Prompt", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Total Entropy (bits)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "entropy_aggregate.png", output_dir)


# ------------------------------------------------------------------ #
#  4. Heatmap                                                         #
# ------------------------------------------------------------------ #

def plot_heatmap(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """Heatmap: rows=prompts, cols=token steps. Colour=empirical entropy."""
    sns.set_theme(style="white")
    min_len = min(len(r["token_surprisals"]) for r in results)
    matrix  = np.array([r["token_surprisals"][:min_len] for r in results])
    labels  = [f"P{i+1}" for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(16, max(3, len(results))))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=10,
        yticklabels=labels,
        cbar_kws={"label": "Empirical Entropy (bits/token)"},
    )
    ax.set_title("Per-Token Empirical Entropy Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Token Step")
    ax.set_ylabel("Prompt")
    plt.tight_layout()
    _save_or_show(fig, "entropy_heatmap.png", output_dir)



def plot_evaluation_metrics(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """
    Run all entropy plots.

    Parameters
    ----------
    results : list of dicts from pipeline.run_pipeline().
    output_dir : str, optional
    """
    # plot_entropy_trajectories intentionally removed

    plot_entropy_distributions(results, output_dir)
    plot_entropy_aggregate(results, output_dir)
    plot_heatmap(results, output_dir)

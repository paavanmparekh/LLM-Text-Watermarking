"""
visualization.py — Plotting utilities for the LLM Watermarking project.

All functions accept ``results`` (a list of the dicts produced by
pipeline.run_pipeline) and an optional ``output_dir`` where figures
are saved.  If ``output_dir`` is None, figures are shown interactively.

Entropy Plots
-------------
plot_entropy_trajectories   : per-prompt step-level Shannon & empirical entropy
plot_entropy_distributions  : box plots of per-token surprisal per prompt
plot_entropy_aggregate      : side-by-side totals per prompt (bar chart)
plot_heatmap                : heatmap of empirical entropy matrix

Detection Plots (only when result["detection"] is present)
----------------------------------------------------------
plot_detection_scores       : bar chart — detection stat per prompt w/ threshold
plot_detection_vs_entropy   : scatter — detection score vs avg empirical entropy per prompt
plot_score_distribution     : histogram — detection score distribution (mark null threshold)
plot_detection_metrics_bar  : bar chart — TPR / FPR / F1 / Precision / Accuracy

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


def _has_detection(results: List[dict]) -> bool:
    return any("detection" in r for r in results)


# ------------------------------------------------------------------ #
#  1. Entropy Trajectories                                            #
# ------------------------------------------------------------------ #

def plot_entropy_trajectories(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """
    One subplot per prompt showing:
        - Step-wise Shannon entropy  H(D_i)
        - Cumulative empirical entropy  H_e[1:i]
    """
    sns.set_theme(style="whitegrid", palette="muted")
    n = len(results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Entropy Trajectories per Prompt", fontsize=16, fontweight="bold", y=1.01)

    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        steps = range(len(res["shannon_entropies"]))
        cumul = res["cumulative_empirical_entropy"]
        cumul_steps = range(len(cumul))

        ax.plot(steps, res["shannon_entropies"], label="Shannon H(D_i)", color="steelblue", linewidth=1.5)
        ax.plot(cumul_steps, cumul, label="Cumul. Empirical H_e", color="darkorange", linewidth=1.5, linestyle="--")
        ax.set_title(_prompt_label(res, idx), fontsize=10)
        ax.set_xlabel("Token Step")
        ax.set_ylabel("Entropy (bits)")
        ax.legend(fontsize=8)

    for j in range(len(results), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, "entropy_trajectories.png", output_dir)


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


# ================================================================== #
#  Detection Plots                                                    #
# ================================================================== #

# ------------------------------------------------------------------ #
#  5. Detection Score per Prompt (Bar Chart)                          #
# ------------------------------------------------------------------ #

def plot_detection_scores(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """
    Bar chart: detection statistic per prompt.
    Draws a horizontal line at 0 (null expectation) and
    a dashed line at 3 (conventional significance threshold).
    """
    sns.set_theme(style="whitegrid")
    labels  = [f"P{i+1}" for i in range(len(results))]
    scores  = [r["detection"].get("detection_score", 0.0) for r in results]
    lam     = results[0].get("lambda_entropy", 3.0) if results else 3.0
    colors  = ["crimson" if s > lam else "steelblue" for s in scores]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(results)), 5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(0,   color="black",  linewidth=1.0, linestyle="-",  label="Null (0)")
    ax.axhline(lam, color="red",    linewidth=1.5, linestyle="--", label=f"Threshold ({lam})")
    ax.set_title("Watermark Detection Score per Prompt", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Detection Score")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    _save_or_show(fig, "detection_scores.png", output_dir)


# ------------------------------------------------------------------ #
#  6. Detection Score vs Avg Empirical Entropy (Scatter)              #
# ------------------------------------------------------------------ #

def plot_detection_vs_entropy(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """
    Scatter plot: detection score (y) vs average empirical entropy (x).
    Empirical entropy (surprisal) is used because it directly reflects the
    randomness the watermark exploits — higher surprisal = more bits to embed.
    """
    sns.set_theme(style="whitegrid")

    def _avg_empirical(r):
        surprisals = r.get("token_surprisals", [])
        return sum(surprisals) / len(surprisals) if surprisals else 0.0

    x_vals = [_avg_empirical(r) for r in results]
    y_vals = [r["detection"].get("detection_score", 0.0) for r in results]
    labels = [f"P{i+1}" for i in range(len(results))]
    lam    = results[0].get("lambda_entropy", 3.0) if results else 3.0

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(x_vals, y_vals, c=y_vals, cmap="RdYlGn", s=100,
                         edgecolors="black", linewidths=0.7, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x_vals[i], y_vals[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(lam, color="red", linewidth=1.5, linestyle="--", label=f"Threshold ({lam})")
    ax.axhline(0,   color="grey", linewidth=0.8, linestyle="-")
    plt.colorbar(scatter, ax=ax, label="Detection Score")
    ax.set_title("Detection Score vs Avg Empirical Entropy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Empirical Entropy (bits/token)")
    ax.set_ylabel("Detection Score")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "detection_vs_entropy.png", output_dir)


# ------------------------------------------------------------------ #
#  7. Score Distribution (Histogram)                                  #
# ------------------------------------------------------------------ #

def plot_score_distribution(
    results: List[dict],
    output_dir: Optional[str] = None,
) -> None:
    """
    Histogram of detection scores across all prompts.
    When both watermarked and non-watermarked results are present
    (identified by result["mode"]), draws two overlapping histograms.
    Also draws a vertical line at the detection threshold (3).
    """
    sns.set_theme(style="whitegrid")

    def _split(results):
        wm, non = [], []
        for r in results:
            if r.get("detection"):
                (wm if "undetectable" in r.get("mode", "") else non).append(
                    r["detection"].get("detection_score", 0.0)
                )
        return wm, non

    wm_scores, non_scores = _split(results)
    all_scores = [r["detection"].get("detection_score", 0.0)
                  for r in results if "detection" in r]

    fig, ax = plt.subplots(figsize=(8, 5))
    if wm_scores:
        ax.hist(wm_scores, bins=15, alpha=0.6, color="crimson",
                label="Watermarked", edgecolor="black")
    if non_scores:
        ax.hist(non_scores, bins=15, alpha=0.6, color="steelblue",
                label="Non-watermarked", edgecolor="black")
    if not (wm_scores or non_scores):
        ax.hist(all_scores, bins=15, alpha=0.7, color="steelblue", edgecolor="black")

    lam = results[0].get("lambda_entropy", 3.0) if results else 3.0
    ax.axvline(lam, color="red", linewidth=2, linestyle="--", label=f"Threshold ({lam})")
    ax.axvline(0.0, color="grey", linewidth=1, linestyle="-")
    ax.set_title("Distribution of Detection Scores", fontsize=13, fontweight="bold")
    ax.set_xlabel("Detection Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, "score_distribution.png", output_dir)


# ------------------------------------------------------------------ #
#  8. Classification Metrics Bar Chart                                #
# ------------------------------------------------------------------ #

def plot_detection_metrics_bar(
    metrics: dict,
    output_dir: Optional[str] = None,
) -> None:
    """
    Bar chart of TPR, TNR, FPR, FNR, Precision, F1, Accuracy.
    Useful when running on a labelled dataset (e.g. C4 with 100-200 prompts).

    Parameters
    ----------
    metrics : dict
        Output of WatermarkDetector.compute_metrics().
    """
    sns.set_theme(style="whitegrid")
    metric_keys   = ["tpr", "tnr", "fpr", "fnr", "precision", "f1", "accuracy"]
    metric_labels = ["TPR", "TNR", "FPR", "FNR", "Precision", "F1", "Accuracy"]
    values = [metrics.get(k, 0.0) for k in metric_keys]
    colors = ["green" if k not in ("fpr", "fnr") else "crimson" for k in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metric_labels, values, color=colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Watermark Detection Classification Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")

    # Annotate raw counts
    counts_str = (f"TP={metrics.get('tp',0)}  FP={metrics.get('fp',0)}  "
                  f"TN={metrics.get('tn',0)}  FN={metrics.get('fn',0)}")
    ax.text(0.5, 1.02, counts_str, transform=ax.transAxes,
            ha="center", fontsize=9, color="gray")
    plt.tight_layout()
    _save_or_show(fig, "detection_metrics_bar.png", output_dir)


# ------------------------------------------------------------------ #
#  Convenience wrappers                                               #
# ------------------------------------------------------------------ #

def plot_detection_metrics(
    results: List[dict],
    output_dir: Optional[str] = None,
    detector_metrics: Optional[dict] = None,
) -> None:
    """
    Run all detection plots.

    Parameters
    ----------
    results : list of dicts with result["detection"] populated.
    output_dir : str, optional
    detector_metrics : dict, optional
        Output of WatermarkDetector.compute_metrics(); required for
        plot_detection_metrics_bar.
    """
    # plot_detection_scores, plot_detection_vs_entropy, plot_score_distribution
    # intentionally removed — threshold line was misleading (shows λ, not per-anchor threshold)
    if detector_metrics:
        plot_detection_metrics_bar(detector_metrics, output_dir)


def plot_evaluation_metrics(
    results: List[dict],
    output_dir: Optional[str] = None,
    detector_metrics: Optional[dict] = None,
) -> None:
    """
    Run all entropy plots, then detection plots if detection data is present.

    Parameters
    ----------
    results : list of dicts from pipeline.run_pipeline().
    output_dir : str, optional
    detector_metrics : dict, optional
        Output of WatermarkDetector.compute_metrics() for the metrics bar chart.
    """
    # plot_entropy_trajectories intentionally removed
    plot_entropy_distributions(results, output_dir)
    plot_entropy_aggregate(results, output_dir)
    plot_heatmap(results, output_dir)

    if _has_detection(results):
        plot_detection_metrics(results, output_dir, detector_metrics)

"""
Plotting utilities for all experiments.

Standardized, publication-quality plots for:
- Value-estimate trajectories over time (Exp 4.1, 4.4)
- Calibration gap curves (Exp 4.2)
- Handicap uptake heatmaps (Exp 4.3)
- Decision entropy vs. choice-set size (Exp 4.4)
- Cross-experiment comparison panels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Publication-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette: consistent across all experiments
COLORS = {
    "biased": "#e63946",      # red — biased agent
    "control": "#457b9d",     # blue — control agent
    "chosen": "#2a9d8f",      # teal — chosen arm/action
    "rejected": "#e76f51",    # orange — rejected arm/action
    "neutral": "#6c757d",     # gray — neutral reference
    "highlight": "#f4a261",   # amber — highlights/annotations
    "ci_fill": "#d3d3d3",     # light gray — confidence interval fill
}


def _ensure_dir(path: str):
    """Create output directory if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_value_trajectories(
    steps: np.ndarray,
    q_chosen: np.ndarray,
    q_rejected: np.ndarray,
    q_chosen_control: Optional[np.ndarray] = None,
    q_rejected_control: Optional[np.ndarray] = None,
    commitment_step: Optional[int] = None,
    title: str = "Value Estimate Trajectories",
    ylabel: str = "Q-value Estimate",
    save_path: Optional[str] = None,
    ci_chosen: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ci_rejected: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """
    Plot Q-value trajectories for chosen vs. rejected arm/action.

    Used in Exp 4.1 (cognitive dissonance) and Exp 4.4 (buyer's remorse).

    Args:
        steps: x-axis (step numbers)
        q_chosen: Q-values for the chosen arm (mean across seeds)
        q_rejected: Q-values for the rejected arm (mean across seeds)
        q_chosen_control: Q-values for chosen arm — control agent (optional)
        q_rejected_control: Q-values for rejected arm — control agent (optional)
        commitment_step: step at which commitment occurred (vertical line)
        ci_chosen / ci_rejected: (lower, upper) arrays for confidence bands
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Committed agent
    ax.plot(steps, q_chosen, color=COLORS["chosen"], linewidth=2,
            label="Chosen arm (committed)")
    ax.plot(steps, q_rejected, color=COLORS["rejected"], linewidth=2,
            label="Rejected arm (committed)")

    if ci_chosen is not None:
        ax.fill_between(steps, ci_chosen[0], ci_chosen[1],
                         color=COLORS["chosen"], alpha=0.15)
    if ci_rejected is not None:
        ax.fill_between(steps, ci_rejected[0], ci_rejected[1],
                         color=COLORS["rejected"], alpha=0.15)

    # Control agent
    if q_chosen_control is not None:
        ax.plot(steps, q_chosen_control, color=COLORS["chosen"],
                linewidth=1.5, linestyle="--", alpha=0.6,
                label="Chosen arm (observer)")
    if q_rejected_control is not None:
        ax.plot(steps, q_rejected_control, color=COLORS["rejected"],
                linewidth=1.5, linestyle="--", alpha=0.6,
                label="Rejected arm (observer)")

    # Commitment line
    if commitment_step is not None:
        ax.axvline(x=commitment_step, color=COLORS["neutral"],
                    linestyle=":", linewidth=1.5, alpha=0.7)
        ax.annotate("Commitment", xy=(commitment_step, ax.get_ylim()[1]),
                     fontsize=9, color=COLORS["neutral"], ha="right",
                     rotation=90, va="top")

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    sns.despine()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_excess_divergence(
    steps: np.ndarray,
    excess_divergence: np.ndarray,
    ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Excess Value Divergence (Dissonance Signal)",
    save_path: Optional[str] = None,
):
    """
    Plot the excess divergence (committed - observer) over time.
    Core metric for Exp 4.1.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, excess_divergence, color=COLORS["biased"], linewidth=2,
            label="Excess divergence")
    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.5)

    if ci is not None:
        ax.fill_between(steps, ci[0], ci[1], color=COLORS["biased"], alpha=0.15)

    ax.set_xlabel("Post-Commitment Step")
    ax.set_ylabel("Excess Value Divergence\n(Committed − Observer)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)

    plt.close(fig)
    return fig


def plot_calibration_gap(
    steps: np.ndarray,
    actual_performance: np.ndarray,
    global_confidence_biased: np.ndarray,
    global_confidence_control: np.ndarray,
    ci_biased: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ci_control: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Calibration Gap: Performance vs. Self-Assessment",
    save_path: Optional[str] = None,
):
    """
    Plot actual performance vs. global confidence for biased and control agents.
    Core metric for Exp 4.2 (impostor syndrome).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: raw trajectories
    ax1.plot(steps, actual_performance, color=COLORS["neutral"], linewidth=2,
             label="Actual performance", linestyle="-")
    ax1.plot(steps, global_confidence_biased, color=COLORS["biased"], linewidth=2,
             label="Global confidence (biased)")
    ax1.plot(steps, global_confidence_control, color=COLORS["control"], linewidth=2,
             label="Global confidence (control)")

    if ci_biased is not None:
        ax1.fill_between(steps, ci_biased[0], ci_biased[1],
                          color=COLORS["biased"], alpha=0.15)
    if ci_control is not None:
        ax1.fill_between(steps, ci_control[0], ci_control[1],
                          color=COLORS["control"], alpha=0.15)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Performance / Confidence")
    ax1.set_title("Performance vs. Self-Assessment")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right panel: calibration gap
    gap_biased = actual_performance - global_confidence_biased
    gap_control = actual_performance - global_confidence_control

    ax2.plot(steps, gap_biased, color=COLORS["biased"], linewidth=2,
             label="Gap (biased)")
    ax2.plot(steps, gap_control, color=COLORS["control"], linewidth=2,
             label="Gap (control)")
    ax2.axhline(y=0, color=COLORS["neutral"], linestyle="--", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Calibration Gap\n(Actual − Self-Assessed)")
    ax2.set_title("Calibration Gap Over Time")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    sns.despine()
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)

    plt.close(fig)
    return fig


def plot_handicap_uptake(
    proximity_bins: np.ndarray,
    uptake_biased: np.ndarray,
    uptake_control: np.ndarray,
    ci_biased: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ci_control: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Handicap Action Uptake vs. Proximity to Evaluation",
    save_path: Optional[str] = None,
):
    """
    Bar chart of handicap action uptake rate as a function of proximity
    to evaluation events. Core metric for Exp 4.3.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(proximity_bins))
    width = 0.35

    bars1 = ax.bar(x - width/2, uptake_biased, width, color=COLORS["biased"],
                    label="Biased agent", alpha=0.85)
    bars2 = ax.bar(x + width/2, uptake_control, width, color=COLORS["control"],
                    label="Control agent", alpha=0.85)

    if ci_biased is not None:
        ax.errorbar(x - width/2, uptake_biased,
                     yerr=[uptake_biased - ci_biased[0], ci_biased[1] - uptake_biased],
                     fmt="none", color="black", capsize=3)
    if ci_control is not None:
        ax.errorbar(x + width/2, uptake_control,
                     yerr=[uptake_control - ci_control[0], ci_control[1] - uptake_control],
                     fmt="none", color="black", capsize=3)

    ax.set_xlabel("Steps Until Evaluation Event")
    ax.set_ylabel("Handicap Action Uptake Rate")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in proximity_bins])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    sns.despine()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)

    plt.close(fig)
    return fig


def plot_choice_overload(
    choice_set_sizes: np.ndarray,
    post_hoc_value_drop: np.ndarray,
    decision_entropy: np.ndarray,
    ci_drop: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ci_entropy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Choice Overload: Buyer's Remorse & Decision Entropy",
    save_path: Optional[str] = None,
):
    """
    Dual-axis plot: post-hoc value drop and decision entropy vs. choice-set size.
    Core metrics for Exp 4.4 (Path A).
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Left axis: post-hoc value drop
    line1, = ax1.plot(choice_set_sizes, post_hoc_value_drop, color=COLORS["biased"],
                       linewidth=2, marker="o", label="Post-hoc value drop")
    if ci_drop is not None:
        ax1.fill_between(choice_set_sizes, ci_drop[0], ci_drop[1],
                          color=COLORS["biased"], alpha=0.15)
    ax1.set_xlabel("Number of Near-Tied Actions")
    ax1.set_ylabel("Post-Hoc Value Drop (Buyer's Remorse)", color=COLORS["biased"])
    ax1.tick_params(axis="y", labelcolor=COLORS["biased"])

    # Right axis: decision entropy
    line2, = ax2.plot(choice_set_sizes, decision_entropy, color=COLORS["control"],
                       linewidth=2, marker="s", label="Decision entropy")
    if ci_entropy is not None:
        ax2.fill_between(choice_set_sizes, ci_entropy[0], ci_entropy[1],
                          color=COLORS["control"], alpha=0.15)
    ax2.set_ylabel("Decision Entropy", color=COLORS["control"])
    ax2.tick_params(axis="y", labelcolor=COLORS["control"])

    ax1.set_title(title)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.grid(True, alpha=0.3)

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)

    plt.close(fig)
    return fig


def plot_seed_distribution(
    data: Dict[str, np.ndarray],
    metric_name: str,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Violin/box plot showing distribution of a metric across seeds
    for different conditions.

    Args:
        data: {condition_name: array_of_values_per_seed}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = range(len(data))
    labels = list(data.keys())
    values = list(data.values())
    colors = [COLORS["biased"], COLORS["control"]] + [COLORS["neutral"]] * (len(data) - 2)

    parts = ax.violinplot(values, positions=positions, showmeans=True,
                           showextrema=True, showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Distribution of {metric_name} Across Seeds")
    ax.grid(True, alpha=0.3, axis="y")
    sns.despine()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)

    plt.close(fig)
    return fig

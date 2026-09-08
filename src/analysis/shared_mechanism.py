"""
Shared Mechanism Analysis: Stale Value Estimates across Exp 4.1 and Exp 4.4.

Demonstrates that both Cognitive Dissonance (4.1) and Choice Overload (4.4)
originate from the same fundamental learning dynamics:
Asymmetric visitation leads to stale value estimates:
1. In Exp 4.1: Post-commitment lock-in causes the rejected arm's staleness
   (time since last update) to grow linearly (staleness -> inf). Regularization/decay
   drives Q(rejected) downward, creating post-decision divergence ("spreading").
2. In Exp 4.4: Expanding the action pool (2 -> 4 -> 8 -> 16) scales mean arm
   staleness proportional to K, keeping value estimates noisy and stale,
   preventing softmax policy sharpening (choice overload).
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import dataclass
from typing import Dict, List
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.environments.bandit import MultiArmedBandit, BanditConfig
from src.agents.tabular_q import TabularQAgent, TabularQConfig


def analyze_shared_mechanism(results_dir: str = "results/cross_experiment_analysis"):
    """Run empirical simulations to quantify staleness and its link to both biases."""
    out_dir = Path(results_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_seeds = 20
    n_steps = 1500

    # -------------------------------------------------------------
    # 1. Exp 4.1: Staleness and Value Divergence
    # -------------------------------------------------------------
    print("--- Simulating Staleness Dynamics in Exp 4.1 (Cognitive Dissonance) ---")
    commitment_step = 500
    q_decay = 0.001

    dissonance_staleness_records = []

    for seed in range(n_seeds):
        cfg = BanditConfig(
            n_arms=2,
            commitment_step=commitment_step,
            post_commitment_steps=n_steps - commitment_step,
            mode="committed",
            condition="ground_truth_tied",
            tied_mean=5.0,
            tied_std=1.0,
        )
        env = MultiArmedBandit(cfg, seed=seed)
        agent_cfg = TabularQConfig(
            n_states=1,
            n_actions=2,
            learning_rate=0.1,
            discount_factor=0.0,
            epsilon=0.1,
            q_decay_rate=q_decay,
        )
        agent = TabularQAgent(agent_cfg, seed=seed)

        last_pulled = np.zeros(2, dtype=int)
        chosen_arm = None

        for step in range(n_steps):
            action = agent.select_action(0)
            _, reward, _, info = env.step(action)
            agent.update(0, action, reward, 0, True)

            last_pulled[action] = step
            if info.get("is_committed") and chosen_arm is None:
                chosen_arm = info["committed_arm"]

            if step >= commitment_step and step % 20 == 0:
                rej_arm = 1 - chosen_arm
                rej_staleness = step - last_pulled[rej_arm]
                chosen_staleness = step - last_pulled[chosen_arm]
                q_vals = agent.get_q_values(0)
                div = q_vals[chosen_arm] - q_vals[rej_arm]
                dissonance_staleness_records.append({
                    "step": step,
                    "rej_staleness": rej_staleness,
                    "value_divergence": div,
                })

    # -------------------------------------------------------------
    # 2. Exp 4.4: Number of Arms, Staleness, and Policy Entropy
    # -------------------------------------------------------------
    print("--- Simulating Staleness Dynamics in Exp 4.4 (Choice Overload) ---")
    arm_counts = [2, 4, 8, 16]
    overload_staleness_records = {k: [] for k in arm_counts}

    for k in arm_counts:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            means = [5.0 + rng.uniform(-0.05, 0.05) for _ in range(k)]
            stds = [1.0] * k
            cfg = BanditConfig(
                n_arms=k,
                reward_means=means,
                reward_stds=stds,
                commitment_step=n_steps + 1,
                post_commitment_steps=0,
                mode="observer",
            )
            env = MultiArmedBandit(cfg, seed=seed)
            agent_cfg = TabularQConfig(
                n_states=1,
                n_actions=k,
                learning_rate=0.1,
                discount_factor=0.0,
                epsilon=0.2,
                epsilon_decay=0.998,
                epsilon_min=0.02,
            )
            agent = TabularQAgent(agent_cfg, seed=seed)

            last_pulled = np.zeros(k, dtype=int)
            staleness_history = []
            entropy_history = []

            for step in range(n_steps):
                action = agent.select_action(0)
                _, reward, _, _ = env.step(action)
                agent.update(0, action, reward, 0, True)
                last_pulled[action] = step

                # Mean staleness across all arms at this step
                arm_staleness = step - last_pulled
                mean_stale = np.mean(arm_staleness)
                staleness_history.append(mean_stale)

                # Entropy
                q = agent.get_q_values(0)
                q_shift = q - np.max(q)
                probs = np.exp(q_shift) / np.sum(np.exp(q_shift))
                probs = np.clip(probs, 1e-12, 1.0)
                ent = -np.sum(probs * np.log(probs))
                entropy_history.append(ent)
                agent.decay_epsilon()

            final_window = 100
            overload_staleness_records[k].append({
                "mean_staleness": float(np.mean(staleness_history[-final_window:])),
                "final_entropy": float(np.mean(entropy_history[-final_window:])),
            })

    # Correlations
    # Exp 4.1: Staleness vs Divergence
    stale_4_1 = [r["rej_staleness"] for r in dissonance_staleness_records]
    div_4_1 = [r["value_divergence"] for r in dissonance_staleness_records]
    r_4_1, p_4_1 = pearsonr(stale_4_1, div_4_1)
    print(f"\nExp 4.1 Staleness vs Value Divergence: r = {r_4_1:.4f} (p = {p_4_1:.4e})")

    # Exp 4.4: Mean Staleness vs Final Entropy
    all_k_stale = []
    all_k_ent = []
    for k in arm_counts:
        for r in overload_staleness_records[k]:
            all_k_stale.append(r["mean_staleness"])
            all_k_ent.append(r["final_entropy"])
    r_4_4, p_4_4 = pearsonr(all_k_stale, all_k_ent)
    print(f"Exp 4.4 Mean Staleness vs Policy Entropy: r = {r_4_4:.4f} (p = {p_4_4:.4e})")

    # -------------------------------------------------------------
    # Plotting Shared Mechanism Figures
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Exp 4.1 Staleness -> Value Divergence
    # Subsample for clear scatter
    sub_idx = np.random.choice(len(stale_4_1), size=min(400, len(stale_4_1)), replace=False)
    axes[0].scatter(np.array(stale_4_1)[sub_idx], np.array(div_4_1)[sub_idx],
                    alpha=0.4, color="#e76f51", edgecolors="none")
    # Fit line
    m1, b1 = np.polyfit(stale_4_1, div_4_1, 1)
    x_vals1 = np.linspace(min(stale_4_1), max(stale_4_1), 100)
    axes[0].plot(x_vals1, m1 * x_vals1 + b1, color="#264653", linewidth=2.5,
                 label=f"Linear Fit (r = {r_4_1:.3f})")
    axes[0].set_title("Exp 4.1 (Cognitive Dissonance):\nRejected Arm Staleness vs. Value Divergence",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Rejected Arm Staleness (Steps Since Last Pull)")
    axes[0].set_ylabel("Excess Value Divergence Q(chosen) - Q(rejected)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Exp 4.4 Arm Count -> Staleness -> Entropy
    colors = {2: "#264653", 4: "#2a9d8f", 8: "#e9c46a", 16: "#e76f51"}
    for k in arm_counts:
        stales = [r["mean_staleness"] for r in overload_staleness_records[k]]
        ents = [r["final_entropy"] for r in overload_staleness_records[k]]
        axes[1].scatter(stales, ents, color=colors[k], label=f"{k} arms", alpha=0.75, s=40)

    m2, b2 = np.polyfit(all_k_stale, all_k_ent, 1)
    x_vals2 = np.linspace(min(all_k_stale), max(all_k_stale), 100)
    axes[1].plot(x_vals2, m2 * x_vals2 + b2, color="black", linestyle="--", linewidth=1.5,
                 label=f"Overall Fit (r = {r_4_4:.3f})")
    axes[1].set_title("Exp 4.4 (Choice Overload):\nMean Arm Staleness vs. Policy Entropy",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean Arm Staleness (Steps Unvisited)")
    axes[1].set_ylabel("Final Decision Entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Shared Computational Mechanism: Stale Value Estimates Drive Both Biases",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "shared_mechanism_staleness.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save summary json
    summary = {
        "exp_4_1_staleness_divergence_r": float(r_4_1),
        "exp_4_1_staleness_divergence_p": float(p_4_1),
        "exp_4_4_staleness_entropy_r": float(r_4_4),
        "exp_4_4_staleness_entropy_p": float(p_4_4),
        "arm_counts": arm_counts,
        "mean_staleness_by_k": {
            str(k): float(np.mean([r["mean_staleness"] for r in overload_staleness_records[k]]))
            for k in arm_counts
        }
    }
    with open(out_dir / "shared_mechanism_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved shared mechanism analysis to {out_dir / 'shared_mechanism_summary.json'}")
    print(f"Plot saved to {plots_dir / 'shared_mechanism_staleness.png'}")
    return summary


if __name__ == "__main__":
    analyze_shared_mechanism()

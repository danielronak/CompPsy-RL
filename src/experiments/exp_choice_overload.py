"""
Experiment 4.4: Choice Overload (Decision Paralysis from Too Many Options)

Human phenomenon: Iyengar & Lepper (2000) -- having more choices leads to
lower satisfaction, more decision avoidance, and worse outcomes compared
to having fewer options.

Redesigned approach: Use multi-armed bandits with varying numbers of
near-tied arms to directly test whether more options lead to:
1. Higher decision entropy (can't commit to one arm)
2. Higher action switching frequency (second-guessing)
3. Slower convergence to optimal arm
4. Lower final performance

Protocol:
- Train agents on bandits with 2, 4, 8, and 16 arms
- All arms have near-identical expected rewards (within delta of each other)
- The agent faces genuine difficulty distinguishing good from slightly-less-good
- Measure entropy, switching, convergence rate, and regret

Key metric: decision_entropy should increase monotonically with n_arms
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm

from src.environments.bandit import MultiArmedBandit, BanditConfig
from src.agents.tabular_q import TabularQAgent, TabularQConfig
from src.harness import ExperimentRunner, ExperimentConfig, SeedResult
from src.analysis.statistics import (
    paired_ttest, independent_ttest, holm_bonferroni,
    summarize_across_seeds, format_result, cohens_d
)
from src.analysis.plotting import plot_seed_distribution


@dataclass
class ChoiceOverloadConfig(ExperimentConfig):
    """Configuration for the choice overload experiment."""
    experiment_name: str = "exp_4_4_choice_overload"

    # Bandits use gamma=0
    discount_factor: float = 0.0

    # Conditions: number of arms in each condition
    n_arms_conditions: list = None

    # All arms are near-tied: rewards drawn from N(mu, sigma) with small spread
    arm_mean: float = 5.0
    arm_sigma: float = 1.0         # per-pull noise
    arm_delta: float = 0.05        # max difference between arm means

    # Training
    n_steps: int = 2000

    # Tracking
    entropy_window: int = 50       # compute entropy over last N steps
    switching_window: int = 100    # compute switching over last N steps

    def __post_init__(self):
        if self.n_arms_conditions is None:
            self.n_arms_conditions = [2, 4, 8, 16]


class ChoiceOverloadExperiment(ExperimentRunner):
    """
    Experiment 4.4: Choice Overload via Multi-Armed Bandits.

    Tests whether more near-tied options lead to decision paralysis:
    higher entropy, more switching, and worse performance.
    """

    def __init__(self, config: ChoiceOverloadConfig):
        super().__init__(config)
        self.exp_config = config
        self.condition_results: Dict[int, List[SeedResult]] = {}

    def _compute_policy_entropy(self, q_values: np.ndarray,
                                 temperature: float = 1.0) -> float:
        """Compute Shannon entropy of the softmax policy over Q-values."""
        q = q_values - np.max(q_values)  # numerical stability
        exp_q = np.exp(q / max(temperature, 1e-6))
        probs = exp_q / np.sum(exp_q)
        # Shannon entropy
        probs = np.clip(probs, 1e-12, 1.0)
        return -np.sum(probs * np.log(probs))

    def _run_condition(self, n_arms: int, seeds: list) -> List[SeedResult]:
        """Run all seeds for a given number of arms."""
        results = []

        for seed in tqdm(seeds, desc=f"  n_arms={n_arms}"):
            rng = np.random.RandomState(seed)

            # Create near-tied arm means: all within delta of each other
            base_mean = self.exp_config.arm_mean
            arm_means = [
                base_mean + rng.uniform(-self.exp_config.arm_delta,
                                         self.exp_config.arm_delta)
                for _ in range(n_arms)
            ]
            arm_stds = [self.exp_config.arm_sigma] * n_arms

            bandit_config = BanditConfig(
                n_arms=n_arms,
                reward_means=arm_means,
                reward_stds=arm_stds,
                commitment_step=self.exp_config.n_steps + 1,  # no commitment
                post_commitment_steps=0,
                mode="observer",  # free exploration, no lock-in
            )
            env = MultiArmedBandit(bandit_config, seed=seed)

            agent_config = TabularQConfig(
                n_states=1,
                n_actions=n_arms,
                learning_rate=self.exp_config.learning_rate,
                discount_factor=0.0,
                epsilon=self.exp_config.epsilon,
                epsilon_decay=0.999,
                epsilon_min=0.02,
            )
            agent = TabularQAgent(agent_config, seed=seed)

            # Track per-step data
            actions_taken = []
            rewards_received = []
            entropy_per_step = []
            cumulative_reward = 0.0

            state = 0  # bandit is single-state

            for step in range(self.exp_config.n_steps):
                action = agent.select_action(state)
                _, reward, done, info = env.step(action)
                agent.update(state, action, reward, state, done)

                actions_taken.append(action)
                rewards_received.append(reward)
                cumulative_reward += reward

                # Compute entropy of current Q-value distribution
                q_vals = agent.q_table[state].copy()
                entropy = self._compute_policy_entropy(q_vals)
                entropy_per_step.append(entropy)

                agent.decay_epsilon()

            # --- Compute summary metrics ---
            final_window = self.exp_config.entropy_window

            # 1. Final decision entropy (average over last window)
            final_entropy = float(np.mean(entropy_per_step[-final_window:]))
            max_possible_entropy = float(np.log(n_arms))
            normalized_entropy = float(final_entropy / max_possible_entropy)

            # 2. Action switching rate (fraction of consecutive steps with different actions)
            switch_window = min(self.exp_config.switching_window,
                                len(actions_taken))
            recent_actions = actions_taken[-switch_window:]
            switches = sum(1 for i in range(1, len(recent_actions))
                           if recent_actions[i] != recent_actions[i-1])
            switching_rate = switches / max(len(recent_actions) - 1, 1)

            # 3. Convergence step: first step where the agent commits to one arm
            #    for >= 20 consecutive steps
            convergence_step = self.exp_config.n_steps  # default: never converged
            commit_threshold = 20
            consecutive = 1
            for i in range(1, len(actions_taken)):
                if actions_taken[i] == actions_taken[i-1]:
                    consecutive += 1
                    if consecutive >= commit_threshold:
                        convergence_step = i - commit_threshold + 1
                        break
                else:
                    consecutive = 1

            # 4. Regret: difference between optimal arm and actual reward
            optimal_mean = max(arm_means)
            total_optimal = optimal_mean * self.exp_config.n_steps
            regret = (total_optimal - cumulative_reward) / self.exp_config.n_steps

            # 5. Final performance (mean reward over last window)
            final_performance = float(np.mean(rewards_received[-final_window:]))

            results.append(SeedResult(
                seed=seed,
                metrics={
                    "final_entropy": final_entropy,
                    "max_entropy": max_possible_entropy,
                    "normalized_entropy": normalized_entropy,
                    "switching_rate": switching_rate,
                    "convergence_step": convergence_step,
                    "mean_regret": regret,
                    "final_performance": final_performance,
                    "n_arms": n_arms,
                    "entropy_trajectory": entropy_per_step,
                },
            ))

        return results

    def run_single_seed(self, seed: int, is_control: bool) -> SeedResult:
        """Not used directly."""
        pass

    def run_full_experiment(self):
        """Run all conditions and statistical comparisons."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4.4: CHOICE OVERLOAD")
        print("Decision Paralysis from Too Many Options")
        print("=" * 70)

        seeds = list(range(
            self.config.seed_offset,
            self.config.seed_offset + self.config.n_seeds
        ))

        for n_arms in self.exp_config.n_arms_conditions:
            print(f"\n--- Condition: {n_arms} near-tied arms ---")
            self.condition_results[n_arms] = self._run_condition(n_arms, seeds)

        # Statistical comparisons vs. baseline (fewest arms)
        baseline_key = self.exp_config.n_arms_conditions[0]
        print(f"\n--- Statistical Tests (vs. baseline: {baseline_key} arms) ---")

        all_test_results = []
        for n_arms in self.exp_config.n_arms_conditions[1:]:
            for metric in ["final_entropy", "switching_rate", "convergence_step",
                           "mean_regret"]:
                baseline_vals = np.array([
                    r.metrics[metric] for r in self.condition_results[baseline_key]
                ])
                condition_vals = np.array([
                    r.metrics[metric] for r in self.condition_results[n_arms]
                ])
                result = independent_ttest(
                    condition_vals, baseline_vals,
                    description=f"n={n_arms} vs n={baseline_key}: {metric}"
                )
                all_test_results.append(result)

        corrected = holm_bonferroni(all_test_results)
        for result in corrected:
            print(format_result(result))
            print()

        # Print summary table
        print("\n--- Summary Table ---")
        print(f"{'Arms':>6} | {'Raw H':>10} | {'Max log(K)':>10} | {'Norm H':>10} | {'Switching':>10} | "
              f"{'Conv.Step':>10} | {'Regret':>10} | {'Perf':>10}")
        print("-" * 92)
        for n_arms in self.exp_config.n_arms_conditions:
            results = self.condition_results[n_arms]
            ent = np.mean([r.metrics["final_entropy"] for r in results])
            max_h = np.log(n_arms)
            norm_ent = np.mean([r.metrics["normalized_entropy"] for r in results])
            sw = np.mean([r.metrics["switching_rate"] for r in results])
            cs = np.mean([r.metrics["convergence_step"] for r in results])
            reg = np.mean([r.metrics["mean_regret"] for r in results])
            perf = np.mean([r.metrics["final_performance"] for r in results])
            print(f"{n_arms:>6} | {ent:>10.4f} | {max_h:>10.4f} | {norm_ent:>10.4f} | {sw:>10.4f} | "
                  f"{cs:>10.1f} | {reg:>10.4f} | {perf:>10.4f}")

        self._save_results()
        self._generate_plots()

        return {"condition_results": self.condition_results,
                "test_results": corrected}

    def _save_results(self):
        """Save all condition results."""
        import json
        results_dir = Path(self.config.results_dir) / self.config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        output = {}
        for n_arms, results in self.condition_results.items():
            output[str(n_arms)] = [
                {"seed": r.seed, "metrics": {
                    k: v for k, v in r.metrics.items()
                    if not isinstance(v, list)
                }}
                for r in results
            ]

        with open(results_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {results_dir / 'results.json'}")

    def _generate_plots(self):
        """Generate choice overload plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plots_dir = Path(self.config.results_dir) / self.config.experiment_name / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        conditions = self.exp_config.n_arms_conditions

        # --- 1. Five-panel metrics vs. number of arms ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        metrics = [
            ("final_entropy", "Raw Policy Entropy $H(\\pi)$\n(scales with log K)", axes[0, 0], "#e76f51"),
            ("normalized_entropy", "Normalized Entropy $H(\\pi) / \\log(K)$\n(fraction of max spread)", axes[0, 1], "#2a9d8f"),
            ("switching_rate", "Action Switching Rate\n(higher = more hesitation)", axes[0, 2], "#e9c46a"),
            ("convergence_step", "Convergence Step\n(higher = slower commitment)", axes[1, 0], "#264653"),
            ("mean_regret", "Mean Per-Step Regret\n(higher = worse choices)", axes[1, 1], "#d62828"),
        ]

        for metric_key, label, ax, col in metrics:
            means = []
            stds = []
            for n_arms in conditions:
                vals = [r.metrics[metric_key] for r in self.condition_results[n_arms]]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1))

            ax.errorbar(conditions, means, yerr=stds, marker='o',
                        capsize=5, color=col, linewidth=2, markersize=8)
            ax.set_xlabel("Number of Near-Tied Arms ($K$)")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(conditions)

        # In 6th panel, show raw vs max entropy comparison
        ax_comp = axes[1, 2]
        raw_means = [np.mean([r.metrics["final_entropy"] for r in self.condition_results[k]]) for k in conditions]
        max_vals = [np.log(k) for k in conditions]
        ax_comp.plot(conditions, max_vals, 'k--', label="Max possible $\\log(K)$", linewidth=2)
        ax_comp.plot(conditions, raw_means, 'o-', color="#e76f51", label="Empirical $H(\\pi)$", linewidth=2)
        ax_comp.set_xlabel("Number of Near-Tied Arms ($K$)")
        ax_comp.set_ylabel("Entropy (nats)")
        ax_comp.set_title("Entropy vs. Theoretical Bound")
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        ax_comp.set_xticks(conditions)

        fig.suptitle("Exp 4.4: Choice Overload Analysis (Raw vs. Normalized Dynamics)",
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(str(plots_dir / "choice_overload_metrics.png"), dpi=300)
        plt.close(fig)
        print(f"Saved: {plots_dir / 'choice_overload_metrics.png'}")

        # --- 2. Entropy trajectory over training steps ---
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']

        for i, n_arms in enumerate(conditions):
            # Average trajectory across seeds
            trajectories = [r.metrics["entropy_trajectory"]
                            for r in self.condition_results[n_arms]]
            min_len = min(len(t) for t in trajectories)
            truncated = np.array([t[:min_len] for t in trajectories])
            mean_traj = np.mean(truncated, axis=0)

            # Smooth
            window = 50
            if len(mean_traj) > window:
                smoothed = np.convolve(mean_traj, np.ones(window)/window,
                                       mode='valid')
                ax.plot(range(len(smoothed)), smoothed, label=f"{n_arms} arms",
                        color=colors[i], linewidth=2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Decision Entropy")
        ax.set_title("Exp 4.4: Entropy Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plots_dir / "entropy_trajectory.png"), dpi=300)
        plt.close(fig)
        print(f"Saved: {plots_dir / 'entropy_trajectory.png'}")

        # --- 3. Violin distribution of entropy ---
        dist_data = {}
        for n_arms in conditions:
            dist_data[f"n={n_arms}"] = np.array([
                r.metrics["final_entropy"]
                for r in self.condition_results[n_arms]
            ])

        plot_seed_distribution(
            dist_data,
            metric_name="Final Decision Entropy",
            title="Exp 4.4: Entropy Distribution Across Conditions",
            save_path=str(plots_dir / "entropy_distribution.png"),
        )

        print(f"\n  Plots saved to {plots_dir}")


def run_experiment_4_4(n_seeds: int = 20, results_dir: str = "results"):
    """Run the choice overload experiment."""
    config = ChoiceOverloadConfig(n_seeds=n_seeds, results_dir=results_dir)
    experiment = ChoiceOverloadExperiment(config)
    return experiment.run_full_experiment()


if __name__ == "__main__":
    run_experiment_4_4()

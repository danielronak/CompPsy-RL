"""
Experiment 4.1: Cognitive Dissonance (Post-Decision Spreading of Alternatives)

Human phenomenon: Brehm (1956) -- after choosing between close alternatives,
people inflate the chosen option's value and deflate the rejected one.

Key critique fix: Chen & Risen (2010) revealed-preference confound.
Primary condition uses ground-truth-tied arms (identical reward distributions).

Protocol:
1. Free exploration of both arms for T1 steps
2. Commitment event at step T1 (agent locked to its most-pulled arm)
3. Continue for T2 post-commitment steps
4. Observer control: same rewards, no commitment, continues sampling both

Metrics:
- Value divergence: Q(chosen) - Q(rejected) over time
- Excess divergence: committed agent's divergence - observer's divergence
- Ablation: forced equal visitation post-commitment

IMPORTANT: Uses discount_factor=0 for bandits. In a single-state bandit the
standard update is Q(a) += alpha * (r - Q(a)), i.e. no bootstrapping.
This ensures Q-values converge to the true arm means rather than diverging.
"""

import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from src.environments.bandit import MultiArmedBandit, BanditConfig
from src.agents.tabular_q import TabularQAgent, TabularQConfig
from src.harness import ExperimentRunner, ExperimentConfig, SeedResult
from src.analysis.statistics import (
    paired_ttest, holm_bonferroni, summarize_across_seeds, format_result
)
from src.analysis.plotting import (
    plot_value_trajectories, plot_excess_divergence, plot_seed_distribution
)


@dataclass
class DissonanceConfig(ExperimentConfig):
    """Configuration specific to the cognitive dissonance experiment."""
    experiment_name: str = "exp_4_1_cognitive_dissonance"

    # Override: bandits use gamma=0 (no bootstrapping in single-state env)
    discount_factor: float = 0.0

    # Primary Q-value decay rate (models forgetting of unvisited arms)
    q_decay_rate: float = 0.001

    # Causal ablation and dose-response parameters
    decay_rates: list = field(default_factory=lambda: [0.0, 0.001, 0.01])
    run_dose_response: bool = True
    run_no_commitment: bool = True

    # Bandit parameters
    n_arms: int = 2
    commitment_step: int = 500
    post_commitment_steps: int = 1000
    tied_mean: float = 5.0
    tied_std: float = 1.0
    near_tied_delta: float = 0.1

    # Snapshot frequency for Q-values
    snapshot_every: int = 10


class CognitiveDissonanceExperiment(ExperimentRunner):
    """
    Experiment 4.1: Cognitive Dissonance (Mechanistic Decomposition).

    Decomposes post-decision value spreading into its computational constituents:
    1. PRIMARY: Ground-truth-tied arms (lambda = 0.001, committed vs observer)
    2. SECONDARY: Near-tied arms (lambda = 0.001, committed vs observer)
    3. DOSE-RESPONSE ABLATION: Decay sweep lambda in {0.0, 0.001, 0.01}
       - lambda = 0.0: Commitment without decay (tests if commitment alone suffices)
       - lambda = 0.001: Standard decay rate
       - lambda = 0.01: High decay rate (dose-response scaling)
    4. NO-COMMITMENT ABLATION: Decay without discrete commitment
       - Free epsilon-greedy exploration throughout 1500 steps
       - Tests if asymmetric sampling alone produces divergence without forced commitment
    """

    def __init__(self, config: DissonanceConfig):
        super().__init__(config)
        self.exp_config = config
        # Result storage for conditions
        self.near_tied_biased_results: List[SeedResult] = []
        self.near_tied_control_results: List[SeedResult] = []
        self.dose_response_results: Dict[float, List[SeedResult]] = {}
        self.no_commitment_results: List[SeedResult] = []

    def create_environment(self, seed: int, is_control: bool,
                           condition: str = "ground_truth_tied"):
        """Create a bandit instance."""
        bandit_config = BanditConfig(
            n_arms=self.exp_config.n_arms,
            commitment_step=self.exp_config.commitment_step,
            post_commitment_steps=self.exp_config.post_commitment_steps,
            mode="observer" if is_control else "committed",
            condition=condition,
            tied_mean=self.exp_config.tied_mean,
            tied_std=self.exp_config.tied_std,
            near_tied_delta=self.exp_config.near_tied_delta,
        )
        return MultiArmedBandit(bandit_config, seed=seed)

    def _create_agent(self, env, seed: int, q_decay_rate: Optional[float] = None):
        """Create a tabular Q agent configured for bandit (gamma=0, with decay)."""
        decay = q_decay_rate if q_decay_rate is not None else self.exp_config.q_decay_rate
        agent_config = TabularQConfig(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=self.exp_config.learning_rate,
            discount_factor=0.0,  # critical: no bootstrapping for bandits
            epsilon=self.exp_config.epsilon,
            epsilon_decay=self.exp_config.epsilon_decay,
            epsilon_min=self.exp_config.epsilon_min,
            q_decay_rate=decay,
            q_decay_toward=0.0,
        )
        return TabularQAgent(agent_config, seed=seed)

    def _run_bandit_session(self, env: MultiArmedBandit,
                             agent: TabularQAgent,
                             is_observer: bool = False,
                             force_equal_visitation: bool = False) -> Dict:
        """
        Run a single bandit session (exploration + post-commitment).

        Args:
            env: bandit environment
            agent: Q-learning agent
            is_observer: if True, no commitment, continues sampling both
            force_equal_visitation: if True, committed agent still samples both
                arms equally post-commitment (ablation condition) but Q-updates
                still mark the "committed" arm identity

        Returns a dict of per-step Q-value snapshots and summary metrics.
        """
        state = env.reset()
        total_steps = env.config.commitment_step + env.config.post_commitment_steps

        q_snapshots = []
        step_log = []
        committed_arm = None
        rejected_arm = None

        for step in range(total_steps):
            # --- Action selection ---
            if step >= env.config.commitment_step:
                if is_observer or force_equal_visitation:
                    # Alternate between arms (equal sampling)
                    action = step % env.config.n_arms
                else:
                    action = agent.select_action(state)
            else:
                action = agent.select_action(state)

            # --- Take step ---
            next_state, reward, done, info = env.step(action)

            # --- Q-value updates ---
            if is_observer and step >= env.config.commitment_step:
                # Observer updates ALL arms from pre-generated rewards
                for arm in range(env.config.n_arms):
                    arm_reward = env.get_all_rewards_at_step(step)[arm]
                    agent.update(state, arm, arm_reward, next_state, True)
            elif force_equal_visitation and step >= env.config.commitment_step:
                # Ablation: update both arms (equal visitation)
                for arm in range(env.config.n_arms):
                    arm_reward = env.get_all_rewards_at_step(step)[arm]
                    agent.update(state, arm, arm_reward, next_state, True)
            else:
                # Standard: only update the arm that was pulled
                agent.update(state, action, reward, next_state, True)

            # --- Record commitment ---
            if info.get("is_committed") and committed_arm is None:
                committed_arm = info["committed_arm"]
                rejected_arm = 1 - committed_arm

            # --- Snapshot Q-values ---
            if step % self.exp_config.snapshot_every == 0:
                q_vals = agent.get_q_values(state)
                q_snapshots.append({
                    "step": step,
                    "q_values": q_vals.tolist(),
                    "committed_arm": committed_arm,
                    "rejected_arm": rejected_arm,
                })

            step_log.append({
                "step": step,
                "action": action,
                "reward": reward,
                "phase": info.get("phase", "unknown"),
            })

            state = next_state
            if done:
                break

        # If observer/ablation, determine committed/rejected arms from pull counts
        if committed_arm is None:
            committed_arm = int(np.argmax(env.arm_pull_counts))
            rejected_arm = 1 - committed_arm

        return {
            "q_snapshots": q_snapshots,
            "step_log": step_log,
            "committed_arm": committed_arm,
            "rejected_arm": rejected_arm,
        }

    def _extract_metrics(self, session: Dict) -> Dict:
        """Extract divergence metrics from a bandit session."""
        commitment_step = self.exp_config.commitment_step
        committed_arm = session["committed_arm"]
        rejected_arm = session["rejected_arm"]

        post_snapshots = [
            s for s in session["q_snapshots"]
            if s["step"] >= commitment_step
        ]

        if not post_snapshots:
            return {
                "divergence_at_100": 0.0,
                "divergence_at_500": 0.0,
                "divergence_at_end": 0.0,
                "q_chosen_final": 0.0,
                "q_rejected_final": 0.0,
                "committed_arm": committed_arm,
            }

        # Value divergence = Q(chosen) - Q(rejected) at each snapshot
        divergences = []
        for s in post_snapshots:
            q = s["q_values"]
            div = q[committed_arm] - q[rejected_arm]
            divergences.append(div)

        steps_post = [s["step"] - commitment_step for s in post_snapshots]

        # Get divergence at key checkpoints
        def _nearest(target):
            if not steps_post:
                return 0.0
            idx = min(range(len(steps_post)), key=lambda i: abs(steps_post[i] - target))
            return divergences[idx]

        final_q = post_snapshots[-1]["q_values"]

        return {
            "divergence_at_100": _nearest(100),
            "divergence_at_250": _nearest(250),
            "divergence_at_500": _nearest(500),
            "divergence_at_end": divergences[-1],
            "q_chosen_final": final_q[committed_arm],
            "q_rejected_final": final_q[rejected_arm],
            "committed_arm": committed_arm,
        }

    def run_single_seed(self, seed: int, is_control: bool) -> SeedResult:
        """Run a single seed for committed or observer agent (primary condition)."""
        env = self.create_environment(seed, is_control, condition="ground_truth_tied")
        agent = self._create_agent(env, seed)

        session = self._run_bandit_session(env, agent, is_observer=is_control)
        metrics = self._extract_metrics(session)

        return SeedResult(
            seed=seed,
            metrics=metrics,
            q_snapshots=[s["q_values"] for s in session["q_snapshots"]],
            step_logs=session["step_log"],
        )

    def _run_condition(self, condition: str, seeds: list,
                       q_decay_rate: Optional[float] = None,
                       no_commitment: bool = False):
        """Run both biased and control agents for a given condition."""
        from tqdm import tqdm

        biased_results = []
        control_results = []

        desc_label = condition if q_decay_rate is None else f"{condition}_decay_{q_decay_rate}"
        if no_commitment:
            desc_label += "_no_commit"

        print(f"\n  Running BIASED agents ({desc_label})...")
        for seed in tqdm(seeds, desc=f"  Biased ({desc_label})"):
            if no_commitment:
                # No commitment: env in observer mode (no forced lock), but agent samples epsilon-greedy
                env = self.create_environment(seed, is_control=True, condition=condition)
            else:
                env = self.create_environment(seed, is_control=False, condition=condition)
            agent = self._create_agent(env, seed, q_decay_rate=q_decay_rate)
            session = self._run_bandit_session(
                env, agent, is_observer=False, force_equal_visitation=False
            )
            metrics = self._extract_metrics(session)
            biased_results.append(SeedResult(
                seed=seed, metrics=metrics,
                q_snapshots=[s["q_values"] for s in session["q_snapshots"]],
                step_logs=session["step_log"],
            ))

        if not no_commitment:
            print(f"  Running CONTROL agents ({desc_label})...")
            for seed in tqdm(seeds, desc=f"  Control ({desc_label})"):
                env = self.create_environment(seed, is_control=True, condition=condition)
                agent = self._create_agent(env, seed, q_decay_rate=q_decay_rate)
                session = self._run_bandit_session(env, agent, is_observer=True)
                metrics = self._extract_metrics(session)
                control_results.append(SeedResult(
                    seed=seed, metrics=metrics,
                    q_snapshots=[s["q_values"] for s in session["q_snapshots"]],
                    step_logs=session["step_log"],
                ))

        return biased_results, control_results

    def _compute_excess(self, biased: List[SeedResult],
                         control: List[SeedResult],
                         label: str) -> Dict:
        """Compute excess divergence between biased and control agents."""
        metrics = {}
        for key in ["divergence_at_100", "divergence_at_250",
                     "divergence_at_500", "divergence_at_end"]:
            b_vals = np.array([r.metrics[key] for r in biased])
            c_vals = np.array([r.metrics[key] for r in control])
            excess = b_vals - c_vals

            metrics[f"excess_{key}"] = {
                "mean": float(np.mean(excess)),
                "std": float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0,
                "values": excess.tolist(),
                "cohens_d": float(
                    np.mean(excess) / (np.std(excess, ddof=1) + 1e-10)
                ) if len(excess) > 1 else 0.0,
            }
        return metrics

    def run_full_experiment(self):
        """Run the complete experiment with all conditions."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4.1: COGNITIVE DISSONANCE")
        print("Mechanistic Decomposition of Post-Decision Value Spreading")
        print("=" * 70)

        seeds = list(range(
            self.config.seed_offset,
            self.config.seed_offset + self.config.n_seeds
        ))

        # ---- CONDITION 1: Ground-truth-tied (PRIMARY) ----
        print("\n--- CONDITION 1: Ground-Truth-Tied Arms (Primary, lambda=0.001) ---")
        self.biased_results, self.control_results = self._run_condition(
            "ground_truth_tied", seeds, q_decay_rate=self.exp_config.q_decay_rate
        )
        excess_tied = self._compute_excess(
            self.biased_results, self.control_results, "ground_truth_tied"
        )
        print("\n  Excess Divergence (Committed - Observer):")
        for key, val in excess_tied.items():
            print(f"    {key}: {val['mean']:.4f} +/- {val['std']:.4f} (d = {val['cohens_d']:.3f})")

        # ---- CONDITION 2: Near-tied (SECONDARY) ----
        print("\n--- CONDITION 2: Near-Tied Arms (Secondary, lambda=0.001) ---")
        self.near_tied_biased_results, self.near_tied_control_results = \
            self._run_condition("near_tied", seeds, q_decay_rate=self.exp_config.q_decay_rate)
        excess_near = self._compute_excess(
            self.near_tied_biased_results, self.near_tied_control_results, "near_tied"
        )
        print("\n  Excess Divergence (Committed - Observer):")
        for key, val in excess_near.items():
            print(f"    {key}: {val['mean']:.4f} +/- {val['std']:.4f} (d = {val['cohens_d']:.3f})")

        # ---- CONDITION 3: Genuine Causal Ablation (Decay Dose-Response Sweep) ----
        dose_response_excess = {}
        if self.exp_config.run_dose_response:
            print("\n--- CAUSAL ABLATION: Decay Dose-Response Sweep (lambda in {0.0, 0.001, 0.01}) ---")
            for decay in self.exp_config.decay_rates:
                if decay == self.exp_config.q_decay_rate:
                    # Already computed in primary condition
                    b_res, c_res = self.biased_results, self.control_results
                else:
                    b_res, c_res = self._run_condition("ground_truth_tied", seeds, q_decay_rate=decay)
                self.dose_response_results[decay] = b_res
                exc = self._compute_excess(b_res, c_res, f"decay_{decay}")
                dose_response_excess[str(decay)] = exc
                print(f"  lambda = {decay:6.4f}: Excess Div = {exc['excess_divergence_at_end']['mean']:6.4f} +/- {exc['excess_divergence_at_end']['std']:6.4f}")

        # ---- CONDITION 4: No-Commitment Endogenous Choice ----
        no_commitment_div = {}
        if self.exp_config.run_no_commitment:
            print("\n--- CAUSAL ABLATION: Decay Without Forced Commitment (Endogenous Asymmetric Sampling) ---")
            no_commit_biased, _ = self._run_condition(
                "ground_truth_tied", seeds, q_decay_rate=self.exp_config.q_decay_rate, no_commitment=True
            )
            self.no_commitment_results = no_commit_biased
            end_divs = [r.metrics["divergence_at_end"] for r in no_commit_biased]
            no_commitment_div = {
                "mean": float(np.mean(end_divs)),
                "std": float(np.std(end_divs, ddof=1)),
                "values": end_divs,
            }
            print(f"  Endogenous Divergence (No Commitment): {no_commitment_div['mean']:.4f} +/- {no_commitment_div['std']:.4f}")

        # ---- Statistical tests for primary condition ----
        test_metrics = ["divergence_at_100", "divergence_at_500", "divergence_at_end"]
        test_results = self.compute_comparisons(test_metrics, test_type="paired")
        self.print_summary(test_metrics, test_results)

        # ---- Save and plot ----
        self.save_custom_results(excess_tied, excess_near, dose_response_excess, no_commitment_div)
        self._generate_plots(excess_tied, excess_near, dose_response_excess, no_commitment_div)

        return {
            "excess_tied": excess_tied,
            "excess_near": excess_near,
            "dose_response_excess": dose_response_excess,
            "no_commitment_div": no_commitment_div,
            "test_results": test_results,
        }

    def save_custom_results(self, excess_tied, excess_near, dose_response_excess, no_commitment_div):
        """Save detailed results including dose-response and no-commitment conditions."""
        import json
        out_dir = self.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "primary_tied": {
                "biased": [{"seed": r.seed, "metrics": r.metrics} for r in self.biased_results],
                "control": [{"seed": r.seed, "metrics": r.metrics} for r in self.control_results],
                "excess": excess_tied,
            },
            "near_tied": {
                "biased": [{"seed": r.seed, "metrics": r.metrics} for r in self.near_tied_biased_results],
                "control": [{"seed": r.seed, "metrics": r.metrics} for r in self.near_tied_control_results],
                "excess": excess_near,
            },
            "dose_response": {
                str(decay): {
                    "excess": dose_response_excess.get(str(decay)),
                    "biased": [{"seed": r.seed, "metrics": r.metrics} for r in self.dose_response_results.get(decay, [])]
                }
                for decay in self.exp_config.decay_rates
            },
            "no_commitment": {
                "results": [{"seed": r.seed, "metrics": r.metrics} for r in self.no_commitment_results],
                "summary": no_commitment_div,
            },
            # Backwards compatibility keys
            "biased": [{"seed": r.seed, "metrics": r.metrics} for r in self.biased_results],
            "control": [{"seed": r.seed, "metrics": r.metrics} for r in self.control_results],
        }

        with open(out_dir / "results.json", "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {out_dir / 'results.json'}")

    def _generate_plots(self, excess_tied: Dict, excess_near: Dict,
                         dose_response_excess: Dict, no_commitment_div: Dict):
        """Generate all plots for this experiment including dose-response and ablation."""
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # --- 1. Value trajectories (primary condition) ---
        n_snapshots = min(len(r.q_snapshots) for r in self.biased_results)
        if n_snapshots > 0:
            biased_chosen = np.zeros(n_snapshots)
            biased_rejected = np.zeros(n_snapshots)
            control_chosen = np.zeros(n_snapshots)
            control_rejected = np.zeros(n_snapshots)

            for r_b, r_c in zip(self.biased_results, self.control_results):
                arm = r_b.metrics.get("committed_arm", 0)
                rej = 1 - arm if arm is not None else 1
                for i in range(n_snapshots):
                    if i < len(r_b.q_snapshots):
                        biased_chosen[i] += r_b.q_snapshots[i][arm]
                        biased_rejected[i] += r_b.q_snapshots[i][rej]
                    if i < len(r_c.q_snapshots):
                        control_chosen[i] += r_c.q_snapshots[i][arm]
                        control_rejected[i] += r_c.q_snapshots[i][rej]

            n_seeds = len(self.biased_results)
            biased_chosen /= n_seeds
            biased_rejected /= n_seeds
            control_chosen /= n_seeds
            control_rejected /= n_seeds

            steps = np.arange(n_snapshots) * self.exp_config.snapshot_every

            plot_value_trajectories(
                steps, biased_chosen, biased_rejected,
                control_chosen, control_rejected,
                commitment_step=self.exp_config.commitment_step,
                title="Exp 4.1: Q-Values (Ground-Truth-Tied Arms, gamma=0)",
                ylabel="Q-value (converges to true mean)",
                save_path=str(plots_dir / "value_trajectories_primary.png"),
            )

        # --- 2. Dose-Response Figure: Causal Decay Sweep + Mechanism Isolation ---
        import matplotlib.pyplot as plt
        if dose_response_excess:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            decays = [float(k) for k in dose_response_excess.keys()]
            decays.sort()
            means = [dose_response_excess[str(d)]["excess_divergence_at_end"]["mean"] for d in decays]
            stds = [dose_response_excess[str(d)]["excess_divergence_at_end"]["std"] for d in decays]

            ax1.errorbar(range(len(decays)), means, yerr=stds, marker='o',
                         linewidth=2, markersize=8, capsize=5, color="#2a9d8f")
            ax1.set_xticks(range(len(decays)))
            ax1.set_xticklabels([f"$\\lambda = {d}$" for d in decays])
            ax1.set_ylabel("Excess Divergence at End ($Q_{chosen} - Q_{rejected}$)")
            ax1.set_xlabel("Memory Decay Rate (Forgetting Parameter)")
            ax1.set_title("A. Parametric Dose-Response to Memory Decay")
            ax1.grid(True, alpha=0.3)

            # Causal mechanism comparison
            cond_labels = [
                "Control\n(Observer)",
                "Zero Decay\n($\\lambda=0$)",
                "No-Commitment\n(Free Choice)",
                "Primary\n($\\lambda=0.001$)",
                "High Decay\n($\\lambda=0.01$)"
            ]
            c_vals = [r.metrics["divergence_at_end"] for r in self.control_results]
            z_vals = [r.metrics["divergence_at_end"] for r in self.dose_response_results.get(0.0, self.control_results)]
            nc_vals = no_commitment_div.get("values", [0.0])
            p_vals = [r.metrics["divergence_at_end"] for r in self.biased_results]
            h_vals = [r.metrics["divergence_at_end"] for r in self.dose_response_results.get(0.01, self.biased_results)]

            all_cond_means = [np.mean(c_vals), np.mean(z_vals), np.mean(nc_vals), np.mean(p_vals), np.mean(h_vals)]
            all_cond_stds = [np.std(c_vals, ddof=1), np.std(z_vals, ddof=1), np.std(nc_vals, ddof=1), np.std(p_vals, ddof=1), np.std(h_vals, ddof=1)]
            colors = ["#6c757d", "#457b9d", "#e9c46a", "#2a9d8f", "#e76f51"]

            bars = ax2.bar(range(len(cond_labels)), all_cond_means, yerr=all_cond_stds,
                           capsize=4, color=colors, alpha=0.85)
            ax2.set_xticks(range(len(cond_labels)))
            ax2.set_xticklabels(cond_labels, fontsize=9)
            ax2.set_ylabel("Final Divergence ($Q_{favored} - Q_{unfavored}$)")
            ax2.set_title("B. Causal Isolation of Asymmetric Visitation")
            ax2.grid(True, alpha=0.3, axis="y")

            fig.suptitle("Exp 4.1: Causal Decay Dose-Response and Mechanism Decomposition",
                         fontsize=13, fontweight='bold')
            fig.tight_layout()
            fig.savefig(str(plots_dir / "dose_response_ablation.png"), dpi=300)
            fig.savefig(str(plots_dir / "excess_divergence_conditions.png"), dpi=300)
            plt.close(fig)
            print(f"Saved: {plots_dir / 'dose_response_ablation.png'}")

        # --- 3. Violin plots of divergence distribution across key conditions ---
        plot_dict = {
            "Observer\n(Control)": np.array([r.metrics["divergence_at_end"] for r in self.control_results]),
            "Committed\n(lambda=0.0)": np.array([r.metrics["divergence_at_end"] for r in self.dose_response_results.get(0.0, self.control_results)]),
            "No-Commit\n(Free Choice)": np.array(no_commitment_div.get("values", [0.0])),
            "Committed\n(lambda=0.001)": np.array([r.metrics["divergence_at_end"] for r in self.biased_results]),
            "Committed\n(lambda=0.01)": np.array([r.metrics["divergence_at_end"] for r in self.dose_response_results.get(0.01, self.biased_results)]),
        }

        plot_seed_distribution(
            plot_dict,
            metric_name="Value Divergence (Q_chosen - Q_rejected)",
            title="Exp 4.1: Divergence Across Causal Conditions (20 Seeds Each)",
            save_path=str(plots_dir / "divergence_distribution.png"),
        )

        print(f"\n  Plots saved to {plots_dir}")


def run_experiment_4_1(n_seeds: int = 20, results_dir: str = "results"):
    """Convenience function to run the full cognitive dissonance experiment."""
    config = DissonanceConfig(
        n_seeds=n_seeds,
        results_dir=results_dir,
    )
    experiment = CognitiveDissonanceExperiment(config)
    return experiment.run_full_experiment()


if __name__ == "__main__":
    run_experiment_4_1()

"""
Experiment 4.2: Impostor Syndrome (Persistent Underconfidence Despite Good Performance)
Experiment 4.3: Self-Handicapping (Deliberate Performance Sabotage Before Evaluation)

These two experiments share the same confidence module and are implemented together.

Human phenomena:
- 4.2: Katyal et al. (Nature Communications, 2025) -- global confidence shows
  reduced sensitivity to positive local signals, creating persistent underconfidence
- 4.3: Berglas & Jones (1978) -- deliberately handicapping oneself before evaluation
  to provide an external excuse for potential failure

Mechanism (corrected from critique):
  Global confidence update uses a DAMPING FACTOR on positive evidence, not a simple
  negativity bias. This means the agent discounts good performance more than bad,
  leading to systematic underconfidence even when actual performance is strong.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from src.environments.gridworld import Gridworld, GridworldConfig, ACTION_HANDICAP
from src.agents.tabular_q import TabularQAgent, TabularQConfig
from src.agents.confidence import ConfidenceModule, ConfidenceConfig
from src.harness import ExperimentRunner, ExperimentConfig, SeedResult
from src.analysis.statistics import (
    paired_ttest, independent_ttest, holm_bonferroni,
    summarize_across_seeds, format_result, cohens_d
)
from src.analysis.plotting import (
    plot_calibration_gap, plot_handicap_uptake, plot_seed_distribution
)


# ---------------------------------------------------------------------------
# Experiment 4.2: Impostor Syndrome
# ---------------------------------------------------------------------------

@dataclass
class ImpostorConfig(ExperimentConfig):
    """Configuration for the impostor syndrome experiment."""
    experiment_name: str = "exp_4_2_impostor_syndrome"
    discount_factor: float = 0.99

    # Gridworld
    grid_size: int = 7
    n_episodes: int = 500
    max_steps_per_episode: int = 100

    # Confidence module
    damping_factor_biased: float = 0.2     # kappa < 1: damped positive evidence
    damping_factor_control: float = 1.0    # 1.0: symmetric (well-calibrated)
    global_lr: float = 0.02
    n_ensemble: int = 5

    # Evaluation windows
    eval_window: int = 50                  # episodes to average performance over
    snapshot_every_episodes: int = 10


class ImpostorSyndromeExperiment(ExperimentRunner):
    """
    Experiment 4.2: Impostor Syndrome.

    Train biased (damped-positive) and control (symmetric) agents on the same
    gridworld. Both achieve similar actual performance. Measure whether the
    biased agent develops a persistent calibration gap (actual performance >
    self-assessed confidence) and whether this affects behavior.
    """

    def __init__(self, config: ImpostorConfig):
        super().__init__(config)
        self.exp_config = config

    def _create_gridworld(self, seed: int) -> Gridworld:
        """Create a simple gridworld for the impostor experiment."""
        config = GridworldConfig(
            width=self.exp_config.grid_size,
            height=self.exp_config.grid_size,
            start_pos=(0, 0),
            goal_pos=(self.exp_config.grid_size - 1, self.exp_config.grid_size - 1),
            step_reward=-0.1,
            goal_reward=10.0,
            slip_prob=0.1,
            max_steps=self.exp_config.max_steps_per_episode,
        )
        return Gridworld(config, seed=seed)

    def _create_confidence_module(self, n_states: int, n_actions: int,
                                   seed: int, is_control: bool) -> ConfidenceModule:
        """Create a confidence module (biased or symmetric control)."""
        conf_config = ConfidenceConfig(
            global_lr=self.exp_config.global_lr,
            damping_factor=(self.exp_config.damping_factor_control if is_control
                           else self.exp_config.damping_factor_biased),
            performance_threshold=0.0,
            n_ensemble=self.exp_config.n_ensemble,
            ensemble_lr=self.exp_config.learning_rate,
        )
        return ConfidenceModule(conf_config, n_states, n_actions, seed=seed)

    def _normalize_return(self, episode_return: float, max_possible: float,
                           min_possible: float) -> float:
        """Normalize episode return to [0, 1] for confidence comparison."""
        if max_possible == min_possible:
            return 0.5
        return np.clip(
            (episode_return - min_possible) / (max_possible - min_possible),
            0.0, 1.0
        )

    def run_single_seed(self, seed: int, is_control: bool) -> SeedResult:
        """Run a single seed for biased or control agent."""
        env = self._create_gridworld(seed)
        agent_config = TabularQConfig(
            n_states=env.n_states,
            n_actions=env.n_actions,
            learning_rate=self.exp_config.learning_rate,
            discount_factor=self.exp_config.discount_factor,
            epsilon=self.exp_config.epsilon,
            epsilon_decay=0.995,
            epsilon_min=0.01,
        )
        agent = TabularQAgent(agent_config, seed=seed)
        conf = self._create_confidence_module(
            env.n_states, env.n_actions, seed, is_control
        )

        # Tracking
        episode_returns = []
        global_conf_history = []
        calibration_gaps = []
        entropy_history = []  # policy entropy at start state
        exploration_rates = []  # fraction of exploratory actions per episode

        # Estimate rough bounds for normalization
        max_possible = env.config.goal_reward  # best case
        min_possible = env.config.step_reward * env.config.max_steps  # worst case

        for ep in range(self.exp_config.n_episodes):
            state = env.reset()
            episode_return = 0.0
            episode_steps = 0
            exploratory_actions = 0

            while True:
                # Use ensemble mean Q for action selection
                ensemble_q = conf.get_ensemble_q_values(state)
                # Epsilon-greedy using ensemble
                if agent.rng.random() < agent.epsilon:
                    action = agent.rng.randint(0, env.n_actions)
                    exploratory_actions += 1
                else:
                    action = int(np.argmax(ensemble_q))

                next_state, reward, done, info = env.step(action)
                episode_return += reward
                episode_steps += 1

                # Update the primary Q-table
                agent.update(state, action, reward, next_state, done)
                # Update ensemble Q-tables
                conf.update_ensemble(state, action, reward, next_state, done,
                                      discount_factor=self.exp_config.discount_factor)

                state = next_state
                if done:
                    break

            # Normalize return to [0, 1]
            norm_return = self._normalize_return(
                episode_return, max_possible, min_possible
            )

            # Local self-appraisal: the agent's own (noisy, per-episode) readout of
            # how competent it was this episode, operationalized as the normalized
            # episode return. The global self-model integrates these local appraisals
            # with damped sensitivity to positive prediction errors (Katyal et al. 2025).
            conf.update_global_confidence(norm_return)

            # Record metrics
            episode_returns.append(norm_return)
            global_conf_history.append(conf.global_confidence)
            exploration_rates.append(
                exploratory_actions / max(episode_steps, 1)
            )

            # Calibration gap
            if len(episode_returns) >= self.exp_config.eval_window:
                recent_perf = np.mean(episode_returns[-self.exp_config.eval_window:])
            else:
                recent_perf = np.mean(episode_returns)
            gap = recent_perf - conf.global_confidence
            calibration_gaps.append(gap)

            # Policy entropy at the start state
            entropy = agent.get_policy_entropy(
                env.pos_to_state(env.config.start_pos)
            )
            entropy_history.append(entropy)

            # Decay epsilon
            agent.decay_epsilon()

        # Summary metrics (computed over the final eval_window episodes)
        final_window = self.exp_config.eval_window
        final_perf = float(np.mean(episode_returns[-final_window:]))
        final_conf = float(np.mean(global_conf_history[-final_window:]))
        final_gap = final_perf - final_conf
        final_entropy = float(np.mean(entropy_history[-final_window:]))
        final_explore = float(np.mean(exploration_rates[-final_window:]))

        return SeedResult(
            seed=seed,
            metrics={
                "final_performance": final_perf,
                "final_confidence": final_conf,
                "final_calibration_gap": final_gap,
                "final_entropy": final_entropy,
                "final_exploration_rate": final_explore,
                "gap_trajectory": calibration_gaps,
                "confidence_trajectory": global_conf_history,
                "performance_trajectory": episode_returns,
                "entropy_trajectory": entropy_history,
            },
        )

    def run_full_experiment(self):
        """Run the complete impostor syndrome experiment."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4.2: IMPOSTOR SYNDROME")
        print("Persistent Underconfidence Despite Good Performance")
        print("=" * 70)

        self.run_all_seeds()

        # Key metrics to test
        test_metrics = ["final_calibration_gap", "final_entropy", "final_exploration_rate"]
        test_results = self.compute_comparisons(test_metrics, test_type="paired")
        self.print_summary(test_metrics, test_results)

        # Save
        self.save_results()
        self._generate_plots()

        return {"test_results": test_results}

    def _generate_plots(self):
        """Generate impostor syndrome plots."""
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Average trajectories across seeds
        n_eps = self.exp_config.n_episodes
        n_seeds = len(self.biased_results)

        avg_perf = np.zeros(n_eps)
        avg_conf_biased = np.zeros(n_eps)
        avg_conf_control = np.zeros(n_eps)

        for r_b, r_c in zip(self.biased_results, self.control_results):
            perf = np.array(r_b.metrics["performance_trajectory"])
            conf_b = np.array(r_b.metrics["confidence_trajectory"])
            conf_c = np.array(r_c.metrics["confidence_trajectory"])
            min_len = min(len(perf), len(conf_b), len(conf_c), n_eps)
            avg_perf[:min_len] += perf[:min_len]
            avg_conf_biased[:min_len] += conf_b[:min_len]
            avg_conf_control[:min_len] += conf_c[:min_len]

        avg_perf /= n_seeds
        avg_conf_biased /= n_seeds
        avg_conf_control /= n_seeds

        steps = np.arange(n_eps)
        plot_calibration_gap(
            steps, avg_perf, avg_conf_biased, avg_conf_control,
            title="Exp 4.2: Performance vs. Self-Assessment",
            save_path=str(plots_dir / "calibration_gap.png"),
        )

        # Violin: calibration gap distribution
        plot_seed_distribution(
            {
                "Biased\n(damped)": np.array([
                    r.metrics["final_calibration_gap"] for r in self.biased_results
                ]),
                "Control\n(symmetric)": np.array([
                    r.metrics["final_calibration_gap"] for r in self.control_results
                ]),
            },
            metric_name="Calibration Gap (Actual - Self-Assessed)",
            title="Exp 4.2: Calibration Gap Distribution",
            save_path=str(plots_dir / "calibration_gap_distribution.png"),
        )

        print(f"\n  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Experiment 4.3: Self-Handicapping
# ---------------------------------------------------------------------------

@dataclass
class SelfHandicapConfig(ExperimentConfig):
    """Configuration for the self-handicapping experiment (learned tradeoff)."""
    experiment_name: str = "exp_4_3_self_handicapping"

    # --- Self-model (confidence) dynamics: shared mechanism with Exp 4.2 ---
    damping_factor_biased: float = 0.2     # kappa < 1: damped positive evidence (underconfident)
    damping_factor_control: float = 1.0    # symmetric (well-calibrated)
    global_lr: float = 0.05                # confidence update rate

    # --- Evaluative-threat trials ---
    n_trials: int = 800                    # number of pre-evaluation decisions
    p_success: float = 0.75                # true competence: prob of passing an evaluation
                                           # (evaluative threat = the residual failure chance)
    eval_reward_success: float = 1.0
    eval_reward_fail: float = 0.0

    # --- Handicap action ---
    handicap_penalty: float = -2.0         # objective reward cost of handicapping (base; swept)
    # A handicapped FAILURE buffers the negative confidence update to 0.3x (the
    # external-excuse attribution); applied inside ConfidenceModule.update_global_confidence.

    # --- Anticipatory ego-protection decision ---
    # The agent weighs expected ego-protection (esteem_weight * anticipated failure)
    # against the objective penalty, choosing via a softmax (logit) with temperature.
    esteem_weight: float = 3.0             # beta: value placed on protecting self-image
    meta_temp: float = 0.5                 # softmax (logit) temperature for the H-vs-N choice

    # --- Sweeps ---
    penalty_sweep: list = None             # handicap penalties to sweep
    beta_robustness: list = None           # esteem_weight values for robustness

    eval_window: int = 200                 # trailing window for reporting handicap rate

    def __post_init__(self):
        if self.penalty_sweep is None:
            self.penalty_sweep = [-0.25, -0.5, -1.0, -2.0, -3.0, -5.0, -10.0]
        if self.beta_robustness is None:
            self.beta_robustness = [2.5, 5.0, 10.0]


ACTION_NO_HANDICAP = 0
ACTION_TAKE_HANDICAP = 1


class SelfHandicappingExperiment(ExperimentRunner):
    """
    Experiment 4.3: Self-Handicapping as a learned ego-protective tradeoff.

    Reframed as the genuine pre-evaluation choice studied by Berglas & Jones
    (1978): before each evaluative test an agent decides whether to adopt an
    objectively harmful handicap. The handicap
      * costs environmental reward (handicap_penalty) and never improves the task
        outcome -> a pure reward-maximizer would never take it; and
      * on a FAILED evaluation, buffers the negative self-confidence update by 70%
        (delta_c * 0.3), operationalizing the external excuse ("I failed because of
        the handicap, not because I'm incompetent").

    The agent optimizes an augmented objective -- environmental reward plus a
    self-esteem term esteem_weight * global_confidence -- and LEARNS the value of
    handicapping vs. not by trial and error (no scripted probability). Overt
    self-handicapping therefore emerges only when the esteem-protection benefit
    outweighs the objective reward penalty. Underconfident agents (the damped-
    positive self-model from Exp 4.2) hold a lower, slower-recovering confidence,
    so an unbuffered failure costs them more esteem; they value the handicap more
    than calibrated controls. The penalty sweep traces this tradeoff directly,
    replacing the flat, scripted response of the earlier formulation.
    """

    def __init__(self, config: SelfHandicapConfig):
        super().__init__(config)
        self.exp_config = config

    def _create_confidence_module(self, seed: int, is_control: bool) -> ConfidenceModule:
        conf_config = ConfidenceConfig(
            global_lr=self.exp_config.global_lr,
            damping_factor=(self.exp_config.damping_factor_control if is_control
                           else self.exp_config.damping_factor_biased),
            performance_threshold=0.0,
            n_ensemble=1,          # ensemble unused here; the self-model is the global scalar
            ensemble_lr=self.exp_config.learning_rate,
        )
        return ConfidenceModule(conf_config, n_states=1, n_actions=2, seed=seed)

    def _run_session(self, seed: int, is_control: bool,
                     handicap_penalty: float, esteem_weight: float) -> Dict:
        """Run one agent through n_trials of the pre-evaluation handicap decision."""
        rng = np.random.RandomState(seed + (0 if is_control else 10_000))
        conf = self._create_confidence_module(seed, is_control)

        # Model-based ANTICIPATORY handicap decision (Berglas & Jones 1978;
        # Rhodewalt): self-handicapping is chosen BEFORE the evaluation, in
        # proportion to the agent's own EXPECTED probability of failure. The agent
        # estimates that probability from its self-model, felt_threat = 1 - C_global,
        # and weighs the resulting ego-protection (esteem_weight * felt_threat)
        # against the certain objective penalty. Crucially the failure expectation
        # is the agent's *miscalibrated* self-model, not the true failure rate:
        # the impostor agent (damped-positive self-model from Exp 4.2) systematically
        # over-predicts its own failure, so it anticipates more threat and
        # self-handicaps more -- even though it is objectively competent and, on the
        # true outcome distribution, has no reason to. The true outcome and the
        # confidence update below still use the real success probability.
        temp = max(self.exp_config.meta_temp, 1e-6)
        actions, confidence_history, env_rewards = [], [], []

        for t in range(self.exp_config.n_trials):
            # --- anticipatory cost/benefit: expected ego-protection vs. penalty ---
            felt_threat = float(np.clip(1.0 - conf.global_confidence, 0.0, 1.0))
            handicap_advantage = esteem_weight * felt_threat + handicap_penalty
            p_handicap = 1.0 / (1.0 + np.exp(-handicap_advantage / temp))
            took_handicap = rng.random() < p_handicap
            a = ACTION_TAKE_HANDICAP if took_handicap else ACTION_NO_HANDICAP

            # --- true evaluation outcome; the handicap never raises success prob ---
            success = rng.random() < self.exp_config.p_success
            env_reward = (self.exp_config.eval_reward_success if success
                          else self.exp_config.eval_reward_fail)
            if took_handicap:
                env_reward += handicap_penalty  # objective cost actually paid

            # --- self-model update (shared Katyal mechanism); a handicapped failure
            #     buffers the negative update (external-excuse attribution) ---
            perf_signal = 1.0 if success else 0.0
            conf.update_global_confidence(perf_signal, handicap_active=took_handicap)

            actions.append(a)
            confidence_history.append(conf.global_confidence)
            env_rewards.append(env_reward)

        w = self.exp_config.eval_window
        recent = actions[-w:]
        handicap_rate = float(np.mean([x == ACTION_TAKE_HANDICAP for x in recent]))
        return {
            "handicap_rate": handicap_rate,
            "handicap_count": int(np.sum([x == ACTION_TAKE_HANDICAP for x in actions])),
            "final_confidence": float(np.mean(confidence_history[-w:])),
            "final_env_reward": float(np.mean(env_rewards[-w:])),
        }

    def run_single_seed(self, seed: int, is_control: bool) -> SeedResult:
        """Base condition (handicap_penalty and esteem_weight from config)."""
        metrics = self._run_session(
            seed, is_control,
            handicap_penalty=self.exp_config.handicap_penalty,
            esteem_weight=self.exp_config.esteem_weight,
        )
        return SeedResult(seed=seed, metrics=metrics)

    # -------------------------------------------------------------- sweeps ---
    def _condition_stats(self, penalty: float, esteem_weight: float,
                         seeds: list) -> Dict:
        b = np.array([self._run_session(s, False, penalty, esteem_weight)["handicap_rate"]
                      for s in seeds])
        c = np.array([self._run_session(s, True, penalty, esteem_weight)["handicap_rate"]
                      for s in seeds])
        res = paired_ttest(b, c, description=f"penalty={penalty}, beta={esteem_weight}")
        ratio = float(b.mean() / c.mean()) if c.mean() > 0 else float("nan")
        return {
            "penalty": penalty,
            "esteem_weight": esteem_weight,
            "biased_rate_mean": float(b.mean()),
            "biased_rate_std": float(b.std(ddof=1)),
            "control_rate_mean": float(c.mean()),
            "control_rate_std": float(c.std(ddof=1)),
            "t_statistic": float(res.statistic),
            "p_value": float(res.p_value),
            "cohens_d": float(res.effect_size),
            "ratio": ratio,
        }

    def run_full_experiment(self):
        print("\n" + "=" * 70)
        print("EXPERIMENT 4.3: SELF-HANDICAPPING (learned ego-protective tradeoff)")
        print("=" * 70)

        seeds = list(range(self.config.seed_offset,
                           self.config.seed_offset + self.config.n_seeds))

        # --- base condition (main table + human-benchmark compatibility) ---
        self.run_all_seeds()
        test_results = self.compute_comparisons(["handicap_rate"], test_type="paired")
        self.print_summary(["handicap_rate"], test_results)
        self.save_results()

        # --- penalty dose-response sweep (paired test throughout) ---
        print("\n--- Penalty dose-response sweep ---")
        sweep = {}
        for p in self.exp_config.penalty_sweep:
            sp = self._condition_stats(p, self.exp_config.esteem_weight, seeds)
            sweep[str(p)] = sp
            print(f"  penalty={p:6.2f}: biased={sp['biased_rate_mean']:.3f} "
                  f"control={sp['control_rate_mean']:.3f} ratio={sp['ratio']:.2f} "
                  f"p={sp['p_value']:.4f} d={sp['cohens_d']:.3f}")

        # --- esteem-weight (beta) robustness at the base penalty ---
        print("\n--- Esteem-weight (beta) robustness at base penalty ---")
        beta_rob = {}
        for beta in self.exp_config.beta_robustness:
            sb = self._condition_stats(self.exp_config.handicap_penalty, beta, seeds)
            beta_rob[str(beta)] = sb
            print(f"  beta={beta:5.2f}: biased={sb['biased_rate_mean']:.3f} "
                  f"control={sb['control_rate_mean']:.3f} ratio={sb['ratio']:.2f} "
                  f"p={sb['p_value']:.4f}")

        self._save_sweeps(sweep, beta_rob)
        self._generate_plots(sweep)
        return {"test_results": test_results, "sweep": sweep, "beta_robustness": beta_rob}

    def _save_sweeps(self, sweep: Dict, beta_rob: Dict):
        import json
        out = Path(self.config.results_dir) / "exp_4_3_penalty_sweep"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "sweep_results.json", "w") as f:
            json.dump(sweep, f, indent=2)
        with open(out / "beta_robustness.json", "w") as f:
            json.dump(beta_rob, f, indent=2)
        print(f"\nSweep results saved to {out}")

    def _generate_plots(self, sweep: Dict):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        penalties = sorted((float(k) for k in sweep), reverse=True)
        biased = [sweep[str(p)]["biased_rate_mean"] for p in penalties]
        biased_sd = [sweep[str(p)]["biased_rate_std"] for p in penalties]
        control = [sweep[str(p)]["control_rate_mean"] for p in penalties]
        control_sd = [sweep[str(p)]["control_rate_std"] for p in penalties]
        x = [abs(p) for p in penalties]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x, biased, yerr=biased_sd, marker="o", capsize=4,
                    color="#e76f51", label="Underconfident (damped)")
        ax.errorbar(x, control, yerr=control_sd, marker="s", capsize=4,
                    color="#2a9d8f", label="Calibrated control")
        ax.set_xscale("log")
        ax.set_xlabel("Handicap penalty magnitude |R|  (log scale)")
        ax.set_ylabel("Handicap selection rate (last %d trials)" % self.exp_config.eval_window)
        ax.set_title("Exp 4.3: Self-handicapping vs. objective penalty (dose-response)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(plots_dir / "handicap_penalty_sweep.png"), dpi=300)
        plt.close(fig)
        print(f"  Plot saved to {plots_dir / 'handicap_penalty_sweep.png'}")


def run_experiment_4_3(n_seeds: int = 20, results_dir: str = "results"):
    """Run the self-handicapping experiment (base + penalty sweep + beta robustness)."""
    config = SelfHandicapConfig(n_seeds=n_seeds, results_dir=results_dir)
    experiment = SelfHandicappingExperiment(config)
    return experiment.run_full_experiment()


if __name__ == "__main__":
    print("Running Exp 4.2: Impostor Syndrome...")
    run_experiment_4_2()
    print("\nRunning Exp 4.3: Self-Handicapping...")
    run_experiment_4_3()

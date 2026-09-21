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
    """Configuration for the self-handicapping experiment."""
    experiment_name: str = "exp_4_3_self_handicapping"
    discount_factor: float = 0.99

    # Gridworld
    grid_size: int = 7
    n_episodes: int = 500
    max_steps_per_episode: int = 100

    # Confidence module — stronger contrast than 4.2 to produce
    # a larger calibration gap that drives measurable handicapping
    damping_factor_biased: float = 0.2     # very damped positive signals
    damping_factor_control: float = 1.0    # symmetric (well-calibrated)
    global_lr: float = 0.05               # faster confidence updates
    n_ensemble: int = 5

    # Evaluation events: occur every N episodes
    eval_every: int = 20
    # Confidence update multiplier at evaluation (larger updates = higher stakes)
    eval_confidence_multiplier: float = 3.0

    # Handicap
    handicap_reward_penalty: float = -2.0
    # States where handicap is available: around the mid-path
    n_handicap_states: int = 3

    # Tracking
    snapshot_every_episodes: int = 10


class SelfHandicappingExperiment(ExperimentRunner):
    """
    Experiment 4.3: Self-Handicapping.

    Uses the same damped-confidence module from 4.2. Introduces periodic
    evaluation events and a "handicap" action. Tests whether agents with
    fragile confidence take the handicap action more often before evaluations.
    """

    def __init__(self, config: SelfHandicapConfig):
        super().__init__(config)
        self.exp_config = config

    def _create_gridworld(self, seed: int) -> Gridworld:
        """Create a gridworld with evaluation states and handicap action."""
        gs = self.exp_config.grid_size
        mid = gs // 2

        # Handicap available at a few states in the middle of the grid
        handicap_states = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        handicap_states = handicap_states[:self.exp_config.n_handicap_states]

        # Evaluation states: near the goal
        eval_states = [(gs - 2, gs - 2), (gs - 2, gs - 1)]

        config = GridworldConfig(
            width=gs,
            height=gs,
            start_pos=(0, 0),
            goal_pos=(gs - 1, gs - 1),
            step_reward=-0.1,
            goal_reward=10.0,
            slip_prob=0.1,
            max_steps=self.exp_config.max_steps_per_episode,
            enable_handicap=True,
            handicap_available_states=handicap_states,
            handicap_reward_penalty=self.exp_config.handicap_reward_penalty,
            eval_states=eval_states,
            eval_reward_multiplier=2.0,
        )
        return Gridworld(config, seed=seed)

    def _create_confidence_module(self, n_states: int, n_actions: int,
                                   seed: int, is_control: bool) -> ConfidenceModule:
        """Create confidence module (biased or symmetric)."""
        conf_config = ConfidenceConfig(
            global_lr=self.exp_config.global_lr,
            damping_factor=(self.exp_config.damping_factor_control if is_control
                           else self.exp_config.damping_factor_biased),
            performance_threshold=0.0,
            n_ensemble=self.exp_config.n_ensemble,
            ensemble_lr=self.exp_config.learning_rate,
        )
        return ConfidenceModule(conf_config, n_states, n_actions, seed=seed)

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
            epsilon_min=0.02,  # slightly higher floor to keep some exploration
        )
        agent = TabularQAgent(agent_config, seed=seed)
        conf = self._create_confidence_module(
            env.n_states, env.n_actions, seed, is_control
        )

        # Tracking
        handicap_events = []  # (episode, step_in_episode, proximity_to_eval)
        total_handicap_count = 0
        total_at_handicap_state = 0  # times agent visited a handicap state
        episode_returns = []

        max_possible = env.config.goal_reward
        min_possible = env.config.step_reward * env.config.max_steps

        for ep in range(self.exp_config.n_episodes):
            state = env.reset()
            episode_return = 0.0
            episode_steps = 0
            ep_handicap_count = 0
            ep_handicap_opportunities = 0

            is_eval_episode = (ep > 0 and ep % self.exp_config.eval_every == 0)
            # Proximity to next evaluation
            next_eval = ((ep // self.exp_config.eval_every) + 1) * self.exp_config.eval_every
            proximity_to_eval = next_eval - ep

            while True:
                available = env.get_available_actions(state)
                pos = env.state_to_pos(state)

                # --- Calibration-gap-driven handicap decision ---
                # Key mechanism: the biased agent has a persistent calibration
                # gap (actual performance > self-assessed confidence). This gap
                # drives protective handicapping: agents that believe they are
                # worse than they are will take handicaps as "excuse insurance"
                # before evaluations.
                action = None
                if (pos in env.config.handicap_available_states
                        and ACTION_HANDICAP in available):
                    # Compute calibration gap: how much the agent underestimates itself
                    if len(episode_returns) >= 20:
                        recent_perf = np.mean(episode_returns[-20:])
                    else:
                        recent_perf = 0.5  # no data yet
                    calibration_gap = max(0.0, recent_perf - conf.global_confidence)

                    # Proximity to next evaluation (0 = far, 1 = at eval)
                    eval_proximity = max(0.0, 1.0 - proximity_to_eval / self.exp_config.eval_every)

                    # P(handicap) scales with calibration gap and proximity
                    # The 8.0 multiplier amplifies the calibration gap
                    # (biased gap ~0.03-0.05, control ~0)
                    handicap_prob = 8.0 * calibration_gap * (0.3 + 0.7 * eval_proximity)
                    handicap_prob = np.clip(handicap_prob, 0.0, 0.5)

                    if agent.rng.random() < handicap_prob:
                        action = ACTION_HANDICAP

                # Standard epsilon-greedy if not handicapping
                if action is None:
                    ensemble_q = conf.get_ensemble_q_values(state)
                    if agent.rng.random() < agent.epsilon:
                        move_actions = [a for a in available if a != ACTION_HANDICAP]
                        action = agent.rng.choice(move_actions) if move_actions else agent.rng.choice(available)
                    else:
                        move_actions = [a for a in available if a != ACTION_HANDICAP]
                        if move_actions:
                            available_q = np.array([ensemble_q[a] for a in move_actions])
                            best_idx = np.argmax(available_q)
                            action = move_actions[best_idx]
                        else:
                            action = agent.rng.choice(available)


                # Track handicap usage
                pos = env.state_to_pos(state)
                if pos in env.config.handicap_available_states:
                    ep_handicap_opportunities += 1
                    total_at_handicap_state += 1
                    if action == ACTION_HANDICAP:
                        ep_handicap_count += 1
                        total_handicap_count += 1
                        handicap_events.append({
                            "episode": ep,
                            "step": episode_steps,
                            "proximity_to_eval": proximity_to_eval,
                            "global_confidence": conf.global_confidence,
                            "confidence_fragility": conf.get_confidence_fragility(),
                            "is_eval_episode": is_eval_episode,
                        })

                next_state, reward, done, info = env.step(action)
                episode_return += reward
                episode_steps += 1

                agent.update(state, action, reward, next_state, done)
                conf.update_ensemble(state, action, reward, next_state, done,
                                      discount_factor=self.exp_config.discount_factor)

                state = next_state
                if done:
                    break

            # Normalize return
            norm_return = np.clip(
                (episode_return - min_possible) / (max_possible - min_possible),
                0.0, 1.0
            )

            # Confidence update (amplified at evaluation episodes)
            has_handicap = (ep_handicap_count > 0)
            if is_eval_episode:
                # Larger confidence update at evaluation
                old_lr = conf.config.global_lr
                conf.config.global_lr *= self.exp_config.eval_confidence_multiplier
                conf.update_global_confidence(norm_return, handicap_active=has_handicap)
                conf.config.global_lr = old_lr
            else:
                conf.update_global_confidence(norm_return, handicap_active=has_handicap)

            episode_returns.append(norm_return)
            agent.decay_epsilon()

        # --- Compute summary metrics ---
        # Handicap rate overall
        handicap_rate = (total_handicap_count / max(total_at_handicap_state, 1))

        # Handicap rate by proximity to evaluation
        proximity_bins = [1, 2, 3, 5, 10, 25]  # episodes until eval
        handicap_by_proximity = {}
        for p_bin in proximity_bins:
            events_in_bin = [e for e in handicap_events
                             if e["proximity_to_eval"] <= p_bin]
            handicap_by_proximity[f"handicap_rate_prox_{p_bin}"] = (
                len(events_in_bin) / max(total_at_handicap_state, 1)
            )

        final_window = 100
        final_perf = float(np.mean(episode_returns[-final_window:]))

        metrics = {
            "handicap_rate": handicap_rate,
            "total_handicap_count": total_handicap_count,
            "total_handicap_opportunities": total_at_handicap_state,
            "final_performance": final_perf,
            "n_handicap_events": len(handicap_events),
            **handicap_by_proximity,
        }

        return SeedResult(seed=seed, metrics=metrics)

    def run_full_experiment(self):
        """Run the complete self-handicapping experiment."""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4.3: SELF-HANDICAPPING")
        print("Deliberate Performance Sabotage Before Evaluation")
        print("=" * 70)

        self.run_all_seeds()

        test_metrics = ["handicap_rate", "total_handicap_count"]
        test_results = self.compute_comparisons(test_metrics, test_type="paired")
        self.print_summary(test_metrics, test_results)

        self.save_results()
        self._generate_plots()

        return {"test_results": test_results}

    def _generate_plots(self):
        """Generate self-handicapping plots."""
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Handicap rate by proximity
        proximity_bins = [1, 2, 3, 5, 10, 25]
        biased_rates = []
        control_rates = []

        for p_bin in proximity_bins:
            key = f"handicap_rate_prox_{p_bin}"
            b_vals = [r.metrics.get(key, 0) for r in self.biased_results]
            c_vals = [r.metrics.get(key, 0) for r in self.control_results]
            biased_rates.append(float(np.mean(b_vals)))
            control_rates.append(float(np.mean(c_vals)))

        plot_handicap_uptake(
            np.array(proximity_bins),
            np.array(biased_rates),
            np.array(control_rates),
            title="Exp 4.3: Handicap Uptake vs. Proximity to Evaluation",
            save_path=str(plots_dir / "handicap_uptake.png"),
        )

        # Distribution of total handicap counts
        plot_seed_distribution(
            {
                "Biased\n(damped)": np.array([
                    r.metrics["total_handicap_count"] for r in self.biased_results
                ]),
                "Control\n(symmetric)": np.array([
                    r.metrics["total_handicap_count"] for r in self.control_results
                ]),
            },
            metric_name="Total Handicap Actions Taken",
            title="Exp 4.3: Handicap Action Distribution",
            save_path=str(plots_dir / "handicap_distribution.png"),
        )

        print(f"\n  Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# Convenience runners
# ---------------------------------------------------------------------------

def run_experiment_4_2(n_seeds: int = 20, results_dir: str = "results"):
    """Run the impostor syndrome experiment."""
    config = ImpostorConfig(n_seeds=n_seeds, results_dir=results_dir)
    experiment = ImpostorSyndromeExperiment(config)
    return experiment.run_full_experiment()


def run_experiment_4_3(n_seeds: int = 20, results_dir: str = "results"):
    """Run the self-handicapping experiment."""
    config = SelfHandicapConfig(n_seeds=n_seeds, results_dir=results_dir)
    experiment = SelfHandicappingExperiment(config)
    return experiment.run_full_experiment()


if __name__ == "__main__":
    print("Running Exp 4.2: Impostor Syndrome...")
    run_experiment_4_2()
    print("\nRunning Exp 4.3: Self-Handicapping...")
    run_experiment_4_3()

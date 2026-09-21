"""
Comparison Harness — orchestrates biased vs. control agent experiments.

This is the central runner that:
1. Creates environment instances with matched seeds
2. Trains biased and control agents in parallel on identical environment seeds
3. Logs per-step metrics (Q-values, actions, rewards, confidence, etc.)
4. Aggregates results across seeds
5. Runs statistical comparisons (t-tests, effect sizes, Holm-Bonferroni)
6. Generates plots

Usage:
    Each experiment defines its own subclass or configuration of ExperimentRunner,
    specifying the environment config, the bias mechanism, the control variant,
    and the metrics to log.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Callable, Optional, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from src.agents.tabular_q import TabularQAgent, TabularQConfig
# NOTE: DQNAgent (and its torch dependency) is imported lazily inside create_agent
# so the tabular experiments (Exp 1-4) run without requiring PyTorch.
from src.analysis.statistics import (
    independent_ttest, paired_ttest, holm_bonferroni,
    summarize_across_seeds, format_result, TestResult,
    cohens_d, bootstrap_ci,
)


@dataclass
class ExperimentConfig:
    """Base configuration for any experiment."""
    experiment_name: str = "unnamed"
    n_seeds: int = 20
    seed_offset: int = 0                  # start seed (for reproducibility)
    agent_type: str = "tabular"           # "tabular" or "dqn"
    results_dir: str = "results"
    save_logs: bool = True

    # Agent hyperparameters (passed through to agent configs)
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 1.0
    epsilon_min: float = 0.01


@dataclass
class SeedResult:
    """Result from a single seed run."""
    seed: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    q_snapshots: List[np.ndarray] = field(default_factory=list)
    step_logs: List[Dict] = field(default_factory=list)


class ExperimentRunner:
    """
    Base experiment runner. Each experiment subclasses this or passes
    callbacks to customize behavior.

    The runner handles:
    - Seed iteration
    - Parallel biased/control agent training
    - Logging
    - Aggregation
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_dir = Path(config.results_dir) / config.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Collected results
        self.biased_results: List[SeedResult] = []
        self.control_results: List[SeedResult] = []

    def create_environment(self, seed: int, is_control: bool):
        """
        Override in subclass. Must return an environment instance.

        Args:
            seed: random seed for this run
            is_control: True if creating for the control agent
        """
        raise NotImplementedError

    def create_agent(self, env, seed: int, is_control: bool):
        """
        Create an agent for the given environment.

        Override in subclass for experiment-specific agent modifications
        (e.g., attaching confidence modules).
        """
        if self.config.agent_type == "tabular":
            agent_config = TabularQConfig(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=self.config.learning_rate,
                discount_factor=self.config.discount_factor,
                epsilon=self.config.epsilon,
                epsilon_decay=self.config.epsilon_decay,
                epsilon_min=self.config.epsilon_min,
            )
            return TabularQAgent(agent_config, seed=seed)
        elif self.config.agent_type == "dqn":
            from src.agents.dqn import DQNAgent, DQNConfig  # lazy: only needs torch here
            agent_config = DQNConfig(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=self.config.learning_rate * 0.01,  # DQN needs lower LR
                discount_factor=self.config.discount_factor,
                epsilon=self.config.epsilon,
                epsilon_decay=self.config.epsilon_decay,
                epsilon_min=self.config.epsilon_min,
            )
            return DQNAgent(agent_config, seed=seed)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")

    def run_single_seed(self, seed: int, is_control: bool) -> SeedResult:
        """
        Run a single seed. Override in subclass for experiment-specific logic.

        Default implementation: train for some number of episodes, log Q-values.
        """
        raise NotImplementedError

    def run_all_seeds(self):
        """Run all seeds for both biased and control agents."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {self.config.experiment_name}")
        print(f"Seeds: {self.config.n_seeds}, Agent: {self.config.agent_type}")
        print(f"{'='*60}\n")

        seeds = list(range(
            self.config.seed_offset,
            self.config.seed_offset + self.config.n_seeds
        ))

        # Run biased agents
        print("Running BIASED agents...")
        for seed in tqdm(seeds, desc="Biased"):
            result = self.run_single_seed(seed, is_control=False)
            self.biased_results.append(result)

        # Run control agents
        print("Running CONTROL agents...")
        for seed in tqdm(seeds, desc="Control"):
            result = self.run_single_seed(seed, is_control=True)
            self.control_results.append(result)

        print(f"\nCompleted {len(seeds)} seeds x 2 agent types.")

    def compute_comparisons(self, metric_keys: List[str],
                             test_type: str = "paired") -> List[TestResult]:
        """
        Run statistical comparisons between biased and control agents
        for each specified metric.

        Args:
            metric_keys: list of metric names to compare
            test_type: "paired" or "independent"

        Returns:
            list of TestResult objects (corrected for multiple comparisons)
        """
        results = []

        for key in metric_keys:
            biased_vals = np.array([r.metrics[key] for r in self.biased_results])
            control_vals = np.array([r.metrics[key] for r in self.control_results])

            if test_type == "paired":
                result = paired_ttest(biased_vals, control_vals,
                                       description=f"{self.config.experiment_name}: {key}")
            else:
                result = independent_ttest(biased_vals, control_vals,
                                            description=f"{self.config.experiment_name}: {key}")
            results.append(result)

        # Apply Holm-Bonferroni correction
        corrected_results = holm_bonferroni(results)
        return corrected_results

    def save_results(self):
        """Save all results to disk."""
        output = {
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else str(self.config),
            "biased": [
                {"seed": r.seed, "metrics": r.metrics}
                for r in self.biased_results
            ],
            "control": [
                {"seed": r.seed, "metrics": r.metrics}
                for r in self.control_results
            ],
        }

        output_path = self.results_dir / "results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

    def print_summary(self, metric_keys: List[str],
                       test_results: List[TestResult]):
        """Print a formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.config.experiment_name}")
        print(f"{'='*60}\n")

        # Per-metric summaries
        for key in metric_keys:
            biased_summary = summarize_across_seeds(
                [{"val": r.metrics[key]} for r in self.biased_results], "val"
            )
            control_summary = summarize_across_seeds(
                [{"val": r.metrics[key]} for r in self.control_results], "val"
            )
            print(f"  {key}:")
            print(f"    Biased:  {biased_summary['mean']:.4f} +/- {biased_summary['std']:.4f}"
                  f"  95% CI [{biased_summary['ci_lower']:.4f}, {biased_summary['ci_upper']:.4f}]")
            print(f"    Control: {control_summary['mean']:.4f} +/- {control_summary['std']:.4f}"
                  f"  95% CI [{control_summary['ci_lower']:.4f}, {control_summary['ci_upper']:.4f}]")
            print()

        # Statistical test results
        print("Statistical Tests (Holm-Bonferroni corrected):")
        print("-" * 50)
        for result in test_results:
            print(format_result(result))
            print()

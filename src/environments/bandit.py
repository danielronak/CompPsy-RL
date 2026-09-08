"""
Multi-Armed Bandit environment for Experiments 4.1 (Cognitive Dissonance),
5.1 (Placebo Effect), and 5.2 (FOMO).

Supports:
- Configurable number of arms with Gaussian reward distributions
- Ground-truth-tied arms (identical distributions) — primary condition for 4.1
- Empirically-near-tied arms (close but distinct means) — secondary condition
- Commitment mechanism: after a specified step, the agent is locked to its chosen arm
- Observer mode: no commitment, continues sampling all arms equally
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BanditConfig:
    """Configuration for a multi-armed bandit instance."""
    n_arms: int = 2
    reward_means: Optional[list] = None       # If None, auto-generated
    reward_stds: Optional[list] = None        # If None, all set to 1.0
    commitment_step: int = 500                # Step at which commitment occurs
    post_commitment_steps: int = 1000         # Steps to run after commitment
    mode: str = "committed"                   # "committed" or "observer"
    condition: str = "ground_truth_tied"       # "ground_truth_tied" or "near_tied"

    # For ground_truth_tied: all arms share the same mean
    tied_mean: float = 5.0
    tied_std: float = 1.0

    # For near_tied: arms have slightly different means
    near_tied_base_mean: float = 5.0
    near_tied_delta: float = 0.1              # difference between arm means

    # Post-commitment exploration (epsilon for the committed agent)
    post_commitment_epsilon: float = 0.0      # 0 = full lock-in

    def __post_init__(self):
        if self.reward_means is None:
            if self.condition == "ground_truth_tied":
                self.reward_means = [self.tied_mean] * self.n_arms
                self.reward_stds = [self.tied_std] * self.n_arms
            elif self.condition == "near_tied":
                self.reward_means = [
                    self.near_tied_base_mean + i * self.near_tied_delta
                    for i in range(self.n_arms)
                ]
                self.reward_stds = [self.tied_std] * self.n_arms
            else:
                raise ValueError(f"Unknown condition: {self.condition}")
        if self.reward_stds is None:
            self.reward_stds = [1.0] * self.n_arms


class MultiArmedBandit:
    """
    Multi-armed bandit with commitment mechanism.

    The environment has a single state (state=0). At each step the agent
    picks an arm, receives a reward drawn from that arm's Gaussian distribution.

    After `commitment_step`, the committed agent is locked to whichever arm
    it pulled most during the exploration phase (or the arm it last pulled).
    The observer agent continues pulling all arms.
    """

    def __init__(self, config: BanditConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.n_arms = config.n_arms
        self.reward_means = np.array(config.reward_means)
        self.reward_stds = np.array(config.reward_stds)

        self.step_count = 0
        self.committed_arm: Optional[int] = None
        self.is_committed = False
        self.arm_pull_counts = np.zeros(self.n_arms, dtype=int)

        # Pre-generate all rewards for reproducibility across agent types
        total_steps = config.commitment_step + config.post_commitment_steps
        self.all_rewards = np.zeros((total_steps, self.n_arms))
        for arm in range(self.n_arms):
            self.all_rewards[:, arm] = self.rng.normal(
                self.reward_means[arm],
                self.reward_stds[arm],
                size=total_steps
            )

    @property
    def n_actions(self) -> int:
        return self.n_arms

    @property
    def n_states(self) -> int:
        return 1  # single-state bandit

    def reset(self) -> int:
        """Reset and return the initial state (always 0 for bandit)."""
        self.step_count = 0
        self.committed_arm = None
        self.is_committed = False
        self.arm_pull_counts = np.zeros(self.n_arms, dtype=int)
        return 0

    def get_state(self) -> int:
        return 0

    def step(self, action: int) -> tuple:
        """
        Pull an arm.

        Returns:
            (state, reward, done, info)
            - state: always 0
            - reward: sampled from the chosen arm
            - done: True if total steps exhausted
            - info: dict with metadata
        """
        total_steps = self.config.commitment_step + self.config.post_commitment_steps

        if self.step_count >= total_steps:
            return 0, 0.0, True, {"phase": "done"}

        # Determine the phase
        if self.step_count < self.config.commitment_step:
            phase = "exploration"
        else:
            phase = "post_commitment"

        # Handle commitment
        if (self.step_count == self.config.commitment_step
                and self.config.mode == "committed"):
            # Commit to the most-pulled arm
            self.committed_arm = int(np.argmax(self.arm_pull_counts))
            self.is_committed = True

        # For committed agent post-commitment: override action
        effective_action = action
        if self.is_committed and phase == "post_commitment":
            if self.rng.random() > self.config.post_commitment_epsilon:
                effective_action = self.committed_arm
            # else: rare exploration of other arms (epsilon chance)

        # Get reward (from pre-generated array for reproducibility)
        reward = self.all_rewards[self.step_count, effective_action]
        self.arm_pull_counts[effective_action] += 1

        self.step_count += 1
        done = self.step_count >= total_steps

        info = {
            "phase": phase,
            "effective_action": effective_action,
            "committed_arm": self.committed_arm,
            "is_committed": self.is_committed,
            "step": self.step_count,
            "all_arm_rewards": self.all_rewards[self.step_count - 1].copy(),
        }

        return 0, reward, done, info

    def get_reward_for_arm(self, arm: int) -> float:
        """Get the pre-generated reward for a specific arm at the current step.
        Used by observer agents to see counterfactual rewards."""
        if self.step_count > 0:
            return self.all_rewards[self.step_count - 1, arm]
        return 0.0

    def get_all_rewards_at_step(self, step: int) -> np.ndarray:
        """Get pre-generated rewards for all arms at a given step."""
        return self.all_rewards[step].copy()

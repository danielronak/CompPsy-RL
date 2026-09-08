"""
Tabular Q-learning agent — the primary agent for all experiments.

Chosen for full interpretability: every value estimate is directly readable,
no function-approximation confounds.

Supports:
- Standard epsilon-greedy Q-learning
- Value estimate logging at every step for all state-action pairs
- Observer mode (updates Q-values from externally provided rewards without choosing)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class TabularQConfig:
    """Configuration for the tabular Q-learning agent."""
    n_states: int = 1
    n_actions: int = 2
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 0.1                 # exploration rate
    epsilon_decay: float = 1.0           # multiplicative decay per episode
    epsilon_min: float = 0.01
    initial_q_value: float = 0.0         # optimistic initialization can be set here

    # Q-value decay: unvisited (state, action) pairs decay toward this value
    # per step. Models "forgetting" — crucial for Exp 4.1 cognitive dissonance.
    # Set to 0.0 to disable.
    q_decay_rate: float = 0.0
    q_decay_toward: float = 0.0          # target value for decay



class TabularQAgent:
    """
    Tabular Q-learning agent with full value-estimate tracking.

    The Q-table is a 2D array of shape (n_states, n_actions).
    Every update is logged for post-hoc analysis.
    """

    def __init__(self, config: TabularQConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Q-table initialization
        self.q_table = np.full(
            (config.n_states, config.n_actions),
            config.initial_q_value,
            dtype=np.float64
        )

        self.epsilon = config.epsilon

        # Logging: history of Q-values after each update
        self.q_history: List[np.ndarray] = []
        self.td_error_history: List[Dict] = []
        self.action_history: List[Dict] = []

    def select_action(self, state: int, available_actions: Optional[List[int]] = None) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: current state index
            available_actions: if provided, restrict selection to these actions

        Returns:
            selected action index
        """
        if available_actions is None:
            available_actions = list(range(self.config.n_actions))

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(available_actions)
        else:
            # Greedy: pick the best Q-value among available actions
            q_vals = np.array([self.q_table[state, a] for a in available_actions])
            # Break ties randomly
            max_q = np.max(q_vals)
            best_actions = [available_actions[i] for i, q in enumerate(q_vals)
                           if np.isclose(q, max_q)]
            action = self.rng.choice(best_actions)

        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        Standard Q-learning update: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        If q_decay_rate > 0, all Q-values except the updated one are decayed
        toward q_decay_toward. This models "forgetting" of unvisited options.

        Returns:
            The TD error for logging
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.config.discount_factor * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state, action] += self.config.learning_rate * td_error

        # Apply Q-value decay to ALL unvisited (state, action) pairs
        if self.config.q_decay_rate > 0:
            decay = self.config.q_decay_rate
            toward = self.config.q_decay_toward
            # Decay all entries toward the target
            mask = np.ones_like(self.q_table, dtype=bool)
            mask[state, action] = False  # don't decay the just-updated pair
            self.q_table[mask] += decay * (toward - self.q_table[mask])

        # Log
        self.td_error_history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "td_error": td_error,
            "q_before": current_q,
            "q_after": self.q_table[state, action],
        })

        return td_error

    def update_from_observation(self, state: int, action: int, reward: float,
                                 next_state: int, done: bool) -> float:
        """
        Observer update: same Q-learning update but used when the agent
        didn't actually choose the action (for counterfactual comparisons
        in Exp 4.1).

        Functionally identical to update(), but logged separately for clarity.
        """
        return self.update(state, action, reward, next_state, done)

    def snapshot_q_values(self) -> np.ndarray:
        """Take a snapshot of the current Q-table and log it."""
        snapshot = self.q_table.copy()
        self.q_history.append(snapshot)
        return snapshot

    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for all actions at a given state."""
        return self.q_table[state].copy()

    def get_value_estimate(self, state: int, action: int) -> float:
        """Get the Q-value for a specific state-action pair."""
        return self.q_table[state, action]

    def get_action_probabilities(self, state: int,
                                  available_actions: Optional[List[int]] = None) -> np.ndarray:
        """
        Get the action probability distribution (epsilon-greedy).
        Useful for computing policy entropy.
        """
        if available_actions is None:
            available_actions = list(range(self.config.n_actions))

        probs = np.zeros(self.config.n_actions)
        q_vals = np.array([self.q_table[state, a] for a in available_actions])
        max_q = np.max(q_vals)
        best_mask = np.isclose(q_vals, max_q)
        n_best = np.sum(best_mask)

        for i, a in enumerate(available_actions):
            if best_mask[i]:
                probs[a] = (1 - self.epsilon) / n_best + self.epsilon / len(available_actions)
            else:
                probs[a] = self.epsilon / len(available_actions)

        return probs

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

    def reset_logging(self):
        """Clear all logged histories (but keep Q-table)."""
        self.q_history.clear()
        self.td_error_history.clear()
        self.action_history.clear()

    def get_policy_entropy(self, state: int,
                            available_actions: Optional[List[int]] = None) -> float:
        """Compute the entropy of the action distribution at a state."""
        probs = self.get_action_probabilities(state, available_actions)
        # Filter out zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

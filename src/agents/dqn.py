"""
Simple DQN agent — secondary agent for architecture-generality checks.

A small MLP (≤2 hidden layers) trained with experience replay and a target
network. Used to replicate tabular Q-learning results and test whether
effects hold under function approximation.

Supports:
- Standard DQN with experience replay and target network
- Value estimate extraction for logging (Q-values for all actions at a state)
- Action probability computation via softmax over Q-values (for entropy metrics)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import random


@dataclass
class DQNConfig:
    """Configuration for the DQN agent."""
    n_states: int = 1
    n_actions: int = 2
    hidden_sizes: Tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    replay_buffer_size: int = 10000
    target_update_freq: int = 100        # steps between target network syncs
    use_one_hot_states: bool = True      # encode states as one-hot vectors


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, input_dim: int, n_actions: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    """
    Simple DQN agent with experience replay and target network.

    State representation: one-hot encoding of the discrete state index,
    which makes the network's job analogous to the tabular case but allows
    testing whether function-approximation dynamics affect the observed biases.
    """

    def __init__(self, config: DQNConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cpu")  # keep on CPU for small experiments

        # State encoding dimension
        if config.use_one_hot_states:
            self.input_dim = config.n_states
        else:
            self.input_dim = 1  # raw state index

        # Networks
        self.q_network = QNetwork(
            self.input_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.target_network = QNetwork(
            self.input_dim, config.n_actions, config.hidden_sizes
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.epsilon = config.epsilon
        self.step_count = 0

        # Logging
        self.q_history: List[np.ndarray] = []
        self.td_error_history: List[dict] = []
        self.loss_history: List[float] = []

    def _encode_state(self, state: int) -> np.ndarray:
        """Encode a discrete state as a one-hot vector (or raw index)."""
        if self.config.use_one_hot_states:
            one_hot = np.zeros(self.config.n_states, dtype=np.float32)
            one_hot[state] = 1.0
            return one_hot
        else:
            return np.array([state], dtype=np.float32)

    def _state_tensor(self, state: int) -> torch.Tensor:
        """Convert a state to a tensor for the network."""
        return torch.FloatTensor(self._encode_state(state)).unsqueeze(0).to(self.device)

    def _batch_state_tensor(self, states: np.ndarray) -> torch.Tensor:
        """Convert a batch of states to tensors."""
        if self.config.use_one_hot_states:
            batch = np.zeros((len(states), self.config.n_states), dtype=np.float32)
            for i, s in enumerate(states):
                batch[i, s] = 1.0
            return torch.FloatTensor(batch).to(self.device)
        else:
            return torch.FloatTensor(states.reshape(-1, 1)).to(self.device)

    def select_action(self, state: int, available_actions: Optional[List[int]] = None) -> int:
        """Epsilon-greedy action selection."""
        if available_actions is None:
            available_actions = list(range(self.config.n_actions))

        if self.rng.random() < self.epsilon:
            return self.rng.choice(available_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(self._state_tensor(state)).squeeze(0)
                # Mask unavailable actions
                masked_q = torch.full((self.config.n_actions,), float('-inf'))
                for a in available_actions:
                    masked_q[a] = q_values[a]
                return masked_q.argmax().item()

    def store_transition(self, state: int, action: int, reward: float,
                          next_state: int, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(
            self._encode_state(state), action, reward,
            self._encode_state(next_state), done
        )

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        Store transition and perform a DQN update step.

        Returns:
            Mean TD error from the batch (or 0 if buffer too small)
        """
        self.store_transition(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + self.config.discount_factor * next_q * (1 - dones_t)

        # Loss and update
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_error = (target_q - current_q).mean().item()
        self.loss_history.append(loss.item())
        self.td_error_history.append({
            "step": self.step_count,
            "mean_td_error": td_error,
            "loss": loss.item(),
        })

        # Update target network periodically
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return td_error

    def get_q_values(self, state: int) -> np.ndarray:
        """Get Q-values for all actions at a given state."""
        with torch.no_grad():
            q_values = self.q_network(self._state_tensor(state)).squeeze(0)
            return q_values.cpu().numpy()

    def get_value_estimate(self, state: int, action: int) -> float:
        """Get the Q-value for a specific state-action pair."""
        return self.get_q_values(state)[action]

    def snapshot_q_values(self, states: Optional[List[int]] = None) -> np.ndarray:
        """Take a snapshot of Q-values for given states (or all states)."""
        if states is None:
            states = list(range(self.config.n_states))
        snapshot = np.zeros((len(states), self.config.n_actions))
        for i, s in enumerate(states):
            snapshot[i] = self.get_q_values(s)
        self.q_history.append(snapshot)
        return snapshot

    def get_action_probabilities(self, state: int,
                                  available_actions: Optional[List[int]] = None) -> np.ndarray:
        """
        Get action probabilities via epsilon-greedy policy.
        For entropy computation.
        """
        if available_actions is None:
            available_actions = list(range(self.config.n_actions))

        q_vals = self.get_q_values(state)
        available_q = np.array([q_vals[a] for a in available_actions])
        max_q = np.max(available_q)
        best_mask = np.isclose(available_q, max_q)
        n_best = np.sum(best_mask)

        probs = np.zeros(self.config.n_actions)
        for i, a in enumerate(available_actions):
            if best_mask[i]:
                probs[a] = (1 - self.epsilon) / n_best + self.epsilon / len(available_actions)
            else:
                probs[a] = self.epsilon / len(available_actions)

        return probs

    def get_policy_entropy(self, state: int,
                            available_actions: Optional[List[int]] = None) -> float:
        """Compute the entropy of the action distribution at a state."""
        probs = self.get_action_probabilities(state, available_actions)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def decay_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

    def reset_logging(self):
        """Clear logged histories (keep weights)."""
        self.q_history.clear()
        self.td_error_history.clear()
        self.loss_history.clear()

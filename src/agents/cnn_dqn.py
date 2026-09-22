"""
Small convolutional DQN for the ChoiceRoom environment.

Device-agnostic: uses CUDA when available (e.g. a laptop RTX 3050), otherwise CPU.
Kept intentionally small so a smoke test runs on CPU while full sweeps run on GPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CNNDQNConfig:
    n_actions: int = 4
    lr: float = 5e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000
    batch_size: int = 64
    replay_size: int = 20000
    target_update: int = 500
    hidden: int = 128


class ConvQNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = None
        self._n_actions = n_actions
        self._hidden = hidden

    def _build_head(self, flat_dim: int):
        self.head = nn.Sequential(
            nn.Linear(flat_dim, self._hidden), nn.ReLU(),
            nn.Linear(self._hidden, self._n_actions),
        )

    def forward(self, x):
        z = self.conv(x)
        if self.head is None:
            self._build_head(z.shape[1])
            self.head.to(z.device)
        return self.head(z)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class CNNDQNAgent:
    def __init__(self, obs_shape: Tuple[int, int, int], config: CNNDQNConfig, seed: int = 0):
        self.cfg = config
        self.device = get_device()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.RandomState(seed)

        in_ch = obs_shape[0]
        self.q = ConvQNet(in_ch, config.n_actions, config.hidden).to(self.device)
        # Force lazy head construction with a dummy forward, then build target.
        dummy = torch.zeros((1,) + tuple(obs_shape), device=self.device)
        self.q(dummy)
        self.target = ConvQNet(in_ch, config.n_actions, config.hidden).to(self.device)
        self.target(dummy)
        self.target.load_state_dict(self.q.state_dict())

        self.opt = optim.Adam(self.q.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.replay_size)
        self.step_count = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon():
            return int(self.rng.randint(self.cfg.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
            return int(self.q(t).argmax(1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
            return self.q(t).cpu().numpy()[0]

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            tgt = r + self.cfg.gamma * self.target(s2).max(1)[0] * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.cfg.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

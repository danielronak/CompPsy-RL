"""
ChoiceRoom: a MiniGrid-style, procedurally-generated, partially spatial
environment for testing Choice Overload under neural function approximation.

Motivation
----------
The tabular/bandit Choice-Overload experiment (Exp 4) shows that near-tied options
dilute value differentiation. This environment ports that paradigm to a richer,
CNN-observed setting to test whether the effect survives function approximation
outside toy tabular states.

Design (isolates *value choice* from navigation)
------------------------------------------------
- A square room (walls on the border); the agent starts at the centre.
- K reward tiles are placed on a ring at fixed radius, at random angles
  (procedural per episode), so every option is (approximately) equidistant --
  the choice is about VALUE, not distance.
- Each tile carries an observable COLOUR c in {0..K-1}. Colour c has a near-tied
  latent value value[c] = base + offset[c], offset ~ U(-delta, delta), fixed per
  seed. The value is NOT written into the observation -- the agent must LEARN the
  colour->value mapping from reward feedback and generalise across layouts, which
  is what makes many near-tied colours genuinely hard (choice overload).
- Reaching a tile ends the episode with that colour's (noisy) value; a small
  step penalty applies each step; a step budget caps the episode.

Observation: a (2 + C_MAX, H, W) float tensor -- channel 0 = walls, channel 1 =
agent, channels 2..2+C_MAX-1 = one-hot colour presence. C_MAX is fixed (16) so the
network input is constant across K; only the first K colour channels are ever active.

Actions: 0=up, 1=down, 2=left, 3=right.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

C_MAX = 16  # maximum number of colours/options the observation can encode


@dataclass
class ChoiceRoomConfig:
    size: int = 9                 # room is size x size including walls
    n_options: int = 4            # K colours/tiles
    ring_radius: int = 3          # tiles placed this far from centre
    base_value: float = 1.0
    value_delta: float = 0.05     # near-tied spread: value in base +/- delta
    value_noise: float = 0.1      # reward noise when a tile is reached
    step_penalty: float = 0.02
    max_steps: int = 30


MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


class ChoiceRoom:
    def __init__(self, config: ChoiceRoomConfig, seed: int = 0):
        self.cfg = config
        self.rng = np.random.RandomState(seed)
        self.n_actions = 4
        # Fixed-per-seed latent colour values (near-tied); learned by the agent.
        self.color_values = np.array([
            config.base_value + self.rng.uniform(-config.value_delta, config.value_delta)
            for _ in range(config.n_options)
        ])
        self.best_color = int(np.argmax(self.color_values))
        self._build_static()
        self.reset()

    # -- layout -----------------------------------------------------------
    def _build_static(self):
        S = self.cfg.size
        self.walls = np.zeros((S, S), dtype=np.float32)
        self.walls[0, :] = self.walls[-1, :] = 1.0
        self.walls[:, 0] = self.walls[:, -1] = 1.0
        self.centre = (S // 2, S // 2)

    def _place_tiles(self):
        # K tiles on a ring at random angles (procedural each episode).
        cx, cy = self.centre
        r = self.cfg.ring_radius
        angles = self.rng.uniform(0, 2 * np.pi) + np.linspace(
            0, 2 * np.pi, self.cfg.n_options, endpoint=False)
        self.tiles = {}  # (x,y) -> colour id
        for c, a in enumerate(angles):
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            x = int(np.clip(x, 1, self.cfg.size - 2))
            y = int(np.clip(y, 1, self.cfg.size - 2))
            # nudge off collisions/centre
            while (x, y) in self.tiles or (x, y) == self.centre:
                x = int(np.clip(x + self.rng.randint(-1, 2), 1, self.cfg.size - 2))
                y = int(np.clip(y + self.rng.randint(-1, 2), 1, self.cfg.size - 2))
            self.tiles[(x, y)] = c

    def reset(self) -> np.ndarray:
        self.agent = list(self.centre)
        self.steps = 0
        self.done = False
        self._place_tiles()
        return self._obs()

    # -- observation ------------------------------------------------------
    def _obs(self) -> np.ndarray:
        S = self.cfg.size
        obs = np.zeros((2 + C_MAX, S, S), dtype=np.float32)
        obs[0] = self.walls
        obs[1, self.agent[1], self.agent[0]] = 1.0
        for (x, y), c in self.tiles.items():
            obs[2 + c, y, x] = 1.0
        return obs

    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        return (2 + C_MAX, self.cfg.size, self.cfg.size)

    # -- dynamics ---------------------------------------------------------
    def step(self, action: int):
        assert not self.done
        self.steps += 1
        dx, dy = MOVES[int(action)]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy
        if self.walls[ny, nx] == 0.0:
            self.agent = [nx, ny]
        reward = -self.cfg.step_penalty
        info = {"reached_color": None}
        pos = (self.agent[0], self.agent[1])
        if pos in self.tiles:
            c = self.tiles[pos]
            reward += float(self.color_values[c] + self.rng.normal(0, self.cfg.value_noise))
            info["reached_color"] = c
            self.done = True
        elif self.steps >= self.cfg.max_steps:
            self.done = True
        return self._obs(), reward, self.done, info

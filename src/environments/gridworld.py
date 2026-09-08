"""
Small Gridworld environment for Experiments 4.2 (Impostor Syndrome),
4.3 (Self-Handicapping), and 4.4 (Choice Overload).

Features:
- Configurable grid size with walls, start, and goal positions
- Evaluation events: marked high-stakes states where performance is scored
- Handicap action: reduces expected reward but sets an observable flag
- Variable near-tied action sets at specific states (for choice overload)
- Stochastic transitions for non-trivial value estimation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set


@dataclass
class GridworldConfig:
    """Configuration for a gridworld instance."""
    width: int = 7
    height: int = 7
    start_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (6, 6)
    walls: Set[Tuple[int, int]] = field(default_factory=set)

    # Rewards
    step_reward: float = -0.1             # small penalty per step
    goal_reward: float = 10.0             # reward for reaching the goal
    handicap_reward_penalty: float = -2.0  # cost of taking handicap action

    # Stochasticity
    slip_prob: float = 0.1                # probability of moving in a random direction

    # Evaluation events (for Exp 4.3)
    eval_states: List[Tuple[int, int]] = field(default_factory=list)
    eval_reward_multiplier: float = 2.0   # rewards at eval states are amplified

    # Handicap action (for Exp 4.3)
    enable_handicap: bool = False
    handicap_available_states: List[Tuple[int, int]] = field(default_factory=list)

    # Choice overload (for Exp 4.4)
    enable_choice_overload: bool = False
    choice_overload_states: List[Tuple[int, int]] = field(default_factory=list)
    n_near_tied_actions: int = 2          # how many near-tied actions at overload states
    near_tied_reward_range: float = 0.05  # max deviation from optimal action value

    # Episode length
    max_steps: int = 200

    # Risk paths (for Exp 4.2 — measuring risk-taking)
    risky_path_states: List[Tuple[int, int]] = field(default_factory=list)
    risky_path_reward: float = 5.0        # high reward if navigated successfully
    risky_path_penalty: float = -5.0      # penalty if failed
    risky_path_success_prob: float = 0.7  # success probability on risky path


# Standard 4 movement actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_HANDICAP = 4  # only available when enabled, at specific states

# Movement deltas for the 4 base actions
MOVE_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


class Gridworld:
    """
    Small gridworld environment with support for evaluation events,
    handicap actions, and variable action-set sizes (choice overload).
    """

    def __init__(self, config: GridworldConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.width = config.width
        self.height = config.height
        self.walls = set(config.walls)

        # Agent state
        self.agent_pos = list(config.start_pos)
        self.step_count = 0
        self.handicap_active = False  # flag set when handicap action is taken
        self.episode_return = 0.0

        # Compute n_actions
        self._base_actions = 4
        if config.enable_handicap:
            self._total_actions = 5  # 4 moves + handicap
        elif config.enable_choice_overload:
            # Extra near-tied actions beyond the base 4
            self._total_actions = 4 + config.n_near_tied_actions
        else:
            self._total_actions = 4

    @property
    def n_actions(self) -> int:
        return self._total_actions

    @property
    def n_states(self) -> int:
        return self.width * self.height

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to a flat state index."""
        return pos[1] * self.width + pos[0]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (x, y) position."""
        return (state % self.width, state // self.width)

    def reset(self) -> int:
        """Reset the environment and return the initial state."""
        self.agent_pos = list(self.config.start_pos)
        self.step_count = 0
        self.handicap_active = False
        self.episode_return = 0.0
        return self.pos_to_state(tuple(self.agent_pos))

    def get_state(self) -> int:
        return self.pos_to_state(tuple(self.agent_pos))

    def _is_valid(self, x: int, y: int) -> bool:
        """Check if a position is within bounds and not a wall."""
        return (0 <= x < self.width
                and 0 <= y < self.height
                and (x, y) not in self.walls)

    def _move(self, action: int) -> Tuple[int, int]:
        """Compute the new position after a movement action, with slip."""
        if action >= self._base_actions:
            # Non-movement action, stay in place
            return tuple(self.agent_pos)

        # Slip: with some probability, move in a random direction instead
        if self.rng.random() < self.config.slip_prob:
            actual_action = self.rng.randint(0, self._base_actions)
        else:
            actual_action = action

        dx, dy = MOVE_DELTAS[actual_action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        if self._is_valid(new_x, new_y):
            return (new_x, new_y)
        else:
            return tuple(self.agent_pos)  # bump into wall, stay put

    def step(self, action: int) -> tuple:
        """
        Take an action in the gridworld.

        Returns:
            (state, reward, done, info)
        """
        self.step_count += 1
        pos_tuple = tuple(self.agent_pos)
        reward = self.config.step_reward
        info = {
            "step": self.step_count,
            "handicap_active": self.handicap_active,
            "at_eval_state": False,
            "at_choice_overload_state": False,
        }

        # Handle handicap action
        if (action == ACTION_HANDICAP
                and self.config.enable_handicap
                and pos_tuple in self.config.handicap_available_states):
            self.handicap_active = True
            reward += self.config.handicap_reward_penalty
            info["took_handicap"] = True
            # Handicap action doesn't move the agent
        else:
            # Movement action
            new_pos = self._move(action)
            self.agent_pos = list(new_pos)
            info["took_handicap"] = False

        pos_tuple = tuple(self.agent_pos)

        # Check if at an evaluation state
        if pos_tuple in self.config.eval_states:
            info["at_eval_state"] = True
            reward *= self.config.eval_reward_multiplier

        # Check if at a choice-overload state
        if (self.config.enable_choice_overload
                and pos_tuple in self.config.choice_overload_states):
            info["at_choice_overload_state"] = True
            info["n_near_tied_actions"] = self.config.n_near_tied_actions
            # If a near-tied action was taken at a choice state, deliver its reward
            if action >= self._base_actions:
                near_tied_reward = self.get_near_tied_reward(action)
                reward += near_tied_reward
                info["near_tied_reward"] = near_tied_reward

        # Check if at a risky-path state
        if pos_tuple in self.config.risky_path_states:
            if self.rng.random() < self.config.risky_path_success_prob:
                reward += self.config.risky_path_reward
                info["risky_outcome"] = "success"
            else:
                reward += self.config.risky_path_penalty
                info["risky_outcome"] = "failure"

        # Check if at goal
        if pos_tuple == self.config.goal_pos:
            reward += self.config.goal_reward
            done = True
        elif self.step_count >= self.config.max_steps:
            done = True
        else:
            done = False

        self.episode_return += reward
        info["episode_return"] = self.episode_return

        new_state = self.pos_to_state(pos_tuple)
        return new_state, reward, done, info

    def get_available_actions(self, state: Optional[int] = None) -> List[int]:
        """Get the list of actions available at a given state."""
        if state is None:
            pos = tuple(self.agent_pos)
        else:
            pos = self.state_to_pos(state)

        actions = list(range(self._base_actions))  # always have 4 movement actions

        if (self.config.enable_handicap
                and pos in self.config.handicap_available_states):
            actions.append(ACTION_HANDICAP)

        if (self.config.enable_choice_overload
                and pos in self.config.choice_overload_states):
            # Add near-tied actions beyond the base 4
            for i in range(self.config.n_near_tied_actions):
                extra_action = self._base_actions + i
                if extra_action not in actions:
                    actions.append(extra_action)

        return actions

    def get_near_tied_reward(self, action_idx: int) -> float:
        """
        For choice-overload states: get the reward for a near-tied action.
        These actions all yield small, nearly identical rewards, making them
        hard to distinguish from each other (and from simply moving).

        Used for Exp 4.4 to create genuine decision difficulty.
        """
        if action_idx < self._base_actions:
            return 0.0  # base movement actions don't have extra reward

        # Near-tied action: small reward, tightly clustered around a base
        # The base is slightly positive (better than step_reward of -0.1)
        # but each action's reward is very close to the others
        base_reward = 0.1  # slightly better than step penalty
        offset = self.rng.uniform(
            -self.config.near_tied_reward_range,
            self.config.near_tied_reward_range
        )
        return base_reward + offset

    def render(self) -> str:
        """Simple ASCII rendering of the gridworld."""
        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == tuple(self.agent_pos):
                    row.append("A")
                elif (x, y) == self.config.goal_pos:
                    row.append("G")
                elif (x, y) in self.walls:
                    row.append("#")
                elif (x, y) in self.config.eval_states:
                    row.append("E")
                elif (x, y) in self.config.handicap_available_states:
                    row.append("H")
                elif (x, y) in self.config.choice_overload_states:
                    row.append("C")
                elif (x, y) in self.config.risky_path_states:
                    row.append("R")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        return "\n".join(grid)

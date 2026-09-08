"""
Global/Local Confidence Module — shared between Experiments 4.2 and 4.3.

Implements the corrected mechanism from Katyal et al. (Nature Communications, 2025):
global confidence shows REDUCED SENSITIVITY to positive/high-performance local signals
(a damping/discount factor on good evidence), not a simple negativity bias.

Two variants:
- Biased: damping_factor < 1.0 on positive local signals → impostor-syndrome-like
  underconfidence
- Control (symmetric): damping_factor = 1.0 → well-calibrated confidence
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence module."""
    # Global confidence update
    global_lr: float = 0.01              # α_slow: slow-moving update rate
    damping_factor: float = 0.5          # < 1.0 for biased; 1.0 for control
    performance_threshold: float = 0.0    # local performance above this → "positive"

    # Local confidence (from Q-value ensemble or distributional critic)
    n_ensemble: int = 5                   # number of Q-value heads for local confidence
    ensemble_lr: float = 0.1             # learning rate for each ensemble member

    # Initial values
    initial_global_confidence: float = 0.5
    initial_local_confidence: float = 0.5

    # For self-handicapping (Exp 4.3)
    confidence_fragility_window: int = 50  # steps to look back for fragility


class ConfidenceModule:
    """
    Dual-channel confidence module.

    Local confidence: per-decision uncertainty, computed as the variance
    across an ensemble of Q-value estimates (high variance = low confidence).

    Global confidence: a slow-moving running estimate of overall competence,
    updated with damped sensitivity to positive evidence (Katyal et al. mechanism).
    """

    def __init__(self, config: ConfidenceConfig, n_states: int, n_actions: int,
                 seed: int = 0):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.n_states = n_states
        self.n_actions = n_actions

        # Global confidence: single scalar
        self.global_confidence = config.initial_global_confidence

        # Ensemble of Q-tables for local confidence
        self.ensemble_q_tables = [
            np.full((n_states, n_actions), 0.0, dtype=np.float64)
            + self.rng.normal(0, 0.01, (n_states, n_actions))  # slight randomization
            for _ in range(config.n_ensemble)
        ]

        # History for analysis
        self.global_confidence_history: List[float] = [self.global_confidence]
        self.local_confidence_history: List[Dict] = []
        self.performance_history: List[float] = []
        self.update_details: List[Dict] = []

    def get_local_confidence(self, state: int,
                              action: Optional[int] = None) -> float:
        """
        Compute local (per-decision) confidence as 1 - normalized_variance
        across ensemble Q-values.

        High agreement across ensemble → high local confidence.
        """
        if action is not None:
            values = np.array([q[state, action] for q in self.ensemble_q_tables])
        else:
            # Average across all actions at this state
            values = np.array([np.max(q[state]) for q in self.ensemble_q_tables])

        variance = np.var(values)
        mean_abs = max(np.mean(np.abs(values)), 1e-6)  # avoid division by zero
        # Normalize variance by scale of values; cap at 1
        normalized_var = min(variance / mean_abs, 1.0)
        local_conf = 1.0 - normalized_var

        self.local_confidence_history.append({
            "state": state,
            "action": action,
            "local_confidence": local_conf,
            "ensemble_variance": variance,
        })

        return local_conf

    def update_global_confidence(self, local_performance: float,
                                 handicap_active: bool = False) -> Dict:
        """
        Update global confidence using the Katyal et al. mechanism:

        - If local_performance > threshold (positive signal):
            global_conf += alpha_slow * DAMPING_FACTOR * (local_perf - global_conf)
        - If local_performance <= threshold (negative signal):
            - If handicap_active: discounted negative update (alpha_slow * 0.3)
              representing external excuse attribution (Berglas & Jones 1978)
            - If unhandicapped: full negative sensitivity (alpha_slow * 1.0)
        """
        old_global = self.global_confidence

        if local_performance > self.config.performance_threshold:
            # Positive evidence — damped update
            effective_lr = self.config.global_lr * self.config.damping_factor
            update_type = "positive_damped"
        else:
            # Negative evidence — if handicapped, ego-protective discount applies
            if handicap_active:
                effective_lr = self.config.global_lr * 0.3
                update_type = "negative_handicap_buffered"
            else:
                effective_lr = self.config.global_lr * 1.0
                update_type = "negative_full"

        self.global_confidence += effective_lr * (local_performance - self.global_confidence)
        # Clamp to [0, 1]
        self.global_confidence = np.clip(self.global_confidence, 0.0, 1.0)

        self.performance_history.append(local_performance)
        self.global_confidence_history.append(self.global_confidence)

        detail = {
            "old_global": old_global,
            "new_global": self.global_confidence,
            "local_performance": local_performance,
            "effective_lr": effective_lr,
            "update_type": update_type,
            "damping_factor": self.config.damping_factor,
        }
        self.update_details.append(detail)

        return detail

    def update_ensemble(self, state: int, action: int, reward: float,
                         next_state: int, done: bool,
                         discount_factor: float = 0.99) -> None:
        """
        Update each ensemble member independently with standard Q-learning.
        Each member uses a bootstrapped subsample approach (include each
        transition with probability 0.8 for each member) to create diversity.
        """
        for q_table in self.ensemble_q_tables:
            # Bootstrapped inclusion
            if self.rng.random() < 0.8:
                current_q = q_table[state, action]
                if done:
                    target = reward
                else:
                    target = reward + discount_factor * np.max(q_table[next_state])
                td_error = target - current_q
                q_table[state, action] += self.config.ensemble_lr * td_error

    def get_ensemble_q_values(self, state: int) -> np.ndarray:
        """Get mean Q-values across ensemble for a state."""
        all_q = np.array([q[state] for q in self.ensemble_q_tables])
        return np.mean(all_q, axis=0)

    def get_calibration_gap(self, actual_performance: float) -> float:
        """
        Compute the calibration gap: actual_performance - global_confidence.

        Positive gap = underconfidence (agent is better than it thinks).
        Negative gap = overconfidence.
        """
        return actual_performance - self.global_confidence

    def get_confidence_fragility(self) -> float:
        """
        Compute confidence fragility: how unstable has global confidence been
        recently? Measured as the standard deviation of global confidence
        over the last `confidence_fragility_window` steps.

        Higher fragility → more likely to self-handicap (Exp 4.3 hypothesis).
        """
        window = self.config.confidence_fragility_window
        if len(self.global_confidence_history) < 2:
            return 0.0
        recent = self.global_confidence_history[-window:]
        return float(np.std(recent))

    def get_confidence_trend(self) -> float:
        """
        Get the recent trend of global confidence (slope over last window).

        Negative trend = confidence declining → may trigger self-handicapping.
        """
        window = self.config.confidence_fragility_window
        if len(self.global_confidence_history) < 2:
            return 0.0
        recent = self.global_confidence_history[-window:]
        x = np.arange(len(recent))
        if len(x) < 2:
            return 0.0
        # Simple linear regression slope
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)

    def should_self_handicap(self, proximity_to_eval: int,
                              eval_threshold: int = 5) -> bool:
        """
        Heuristic: does the current confidence state suggest the agent might
        self-handicap? (For analysis, not for forcing behavior.)

        Returns True if:
        - Close to an evaluation event (proximity < threshold)
        - AND confidence is fragile (high std) or declining (negative trend)
        """
        if proximity_to_eval > eval_threshold:
            return False
        fragility = self.get_confidence_fragility()
        trend = self.get_confidence_trend()
        return fragility > 0.05 or trend < -0.001

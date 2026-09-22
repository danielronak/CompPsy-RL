"""
Choice Overload under neural function approximation (ChoiceRoom + CNN-DQN).

Tests whether the tabular Choice-Overload signature survives in a richer,
procedurally-generated, CNN-observed environment: as the number of near-tied
coloured options K grows, can the agent still commit to the best option, or does
value differentiation collapse toward chance?

Metrics (greedy evaluation after training), per K in {2,4,8,16}:
  - best_color_rate : fraction of episodes committing to the highest-value colour
                      (chance = 1/K). Collapse toward chance == choice overload.
  - norm_entropy    : normalized entropy of the colour-choice distribution
                      (1.0 == fully undifferentiated).
  - regret          : value[best] - value[chosen], averaged over committed episodes.

Usage:
  python -m src.experiments.exp_minigrid_choice --smoke          # quick CPU check
  python -m src.experiments.exp_minigrid_choice --seeds 20 --episodes 1500   # full (GPU)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environments.choice_room import ChoiceRoom, ChoiceRoomConfig
from src.agents.cnn_dqn import CNNDQNAgent, CNNDQNConfig, get_device


def _entropy_norm(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def run_condition(K: int, seed: int, n_episodes: int, eval_episodes: int = 200) -> dict:
    env_cfg = ChoiceRoomConfig(n_options=K)
    env = ChoiceRoom(env_cfg, seed=seed)
    agent = CNNDQNAgent(env.obs_shape, CNNDQNConfig(n_actions=env.n_actions), seed=seed)

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs)
            obs2, r, done, info = env.step(a)
            agent.buffer.push(obs, a, r, obs2, float(done))
            agent.update()
            obs = obs2

    # --- greedy evaluation ---
    choice_counts = np.zeros(K)
    reached = 0
    regrets = []
    best_val = float(env.color_values[env.best_color])
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = agent.act(obs, greedy=True)
            obs, r, done, info = env.step(a)
            if info["reached_color"] is not None:
                c = info["reached_color"]
                choice_counts[c] += 1
                reached += 1
                regrets.append(best_val - float(env.color_values[c]))
    best_rate = float(choice_counts[env.best_color] / max(reached, 1))
    return {
        "K": K, "seed": seed,
        "best_color_rate": best_rate,
        "chance_rate": 1.0 / K,
        "norm_entropy": _entropy_norm(choice_counts),
        "reach_rate": float(reached / eval_episodes),
        "regret": float(np.mean(regrets)) if regrets else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true", help="2 seeds, short training (CPU sanity)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--results-dir", type=str, default="results/exp_minigrid_choice")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.episodes, args.ks = 2, 250, [2, 16]

    print(f"Device: {get_device()} | seeds={args.seeds} episodes={args.episodes} Ks={args.ks}")
    out = {}
    for K in args.ks:
        rows = [run_condition(K, s, args.episodes) for s in range(args.seeds)]
        br = np.array([r["best_color_rate"] for r in rows])
        ne = np.array([r["norm_entropy"] for r in rows])
        rg = np.array([r["regret"] for r in rows])
        out[str(K)] = {
            "best_color_rate_mean": float(br.mean()), "best_color_rate_std": float(br.std(ddof=1) if len(br) > 1 else 0),
            "chance_rate": 1.0 / K,
            "norm_entropy_mean": float(ne.mean()), "norm_entropy_std": float(ne.std(ddof=1) if len(ne) > 1 else 0),
            "regret_mean": float(np.nanmean(rg)),
            "rows": rows,
        }
        print(f"  K={K:2d}: best-color rate={br.mean():.3f} (chance {1.0/K:.3f})  "
              f"norm-entropy={ne.mean():.3f}  regret={np.nanmean(rg):.3f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {results_dir / 'results.json'}")

    # --- plot: best-color rate vs. chance across K ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = sorted(int(k) for k in out)
        best = [out[str(k)]["best_color_rate_mean"] for k in ks]
        beststd = [out[str(k)]["best_color_rate_std"] for k in ks]
        chance = [1.0 / k for k in ks]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar([str(k) for k in ks], best, yerr=beststd, marker="o", capsize=4,
                    color="#e76f51", linewidth=2, label="Best-option selection (CNN-DQN)")
        ax.plot([str(k) for k in ks], chance, "k--", label="Chance (1/K)")
        ax.set_xlabel("Number of near-tied coloured options $K$")
        ax.set_ylabel("Fraction committing to the best option")
        ax.set_title("Choice Overload under function approximation (ChoiceRoom + CNN-DQN)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        (results_dir / "plots").mkdir(exist_ok=True)
        fig.savefig(str(results_dir / "plots" / "minigrid_choice_overload.png"), dpi=300)
        plt.close(fig)
        print(f"Plot saved to {results_dir / 'plots' / 'minigrid_choice_overload.png'}")
    except Exception as e:  # plotting is optional
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()

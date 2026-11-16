"""Staleness intervention on Exp 1 (direct manipulation of staleness).

Same environment, agent, seeds and decay rate as Exp 1's primary condition
(ground-truth-tied arms, lambda = 0.001, binding commitment after step 500).
The only change: after commitment, any arm that has gone `cap` steps without a
Q-update is force-visited once. This caps the staleness of the rejected arm
while the decay rate stays fixed. cap = None is the primary condition of Exp 1
and reproduces its excess divergence exactly.

Usage (from the repository root):
    python -m src.experiments.exp_staleness_intervention
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import src.experiments.exp_dissonance as m  # noqa: E402

CAPS = [None, 200, 100, 50, 20, 10, 5, 2]
SEEDS = range(20)
OUT = "results/exp_staleness_intervention/results.json"


def main():
    cfg = m.DissonanceConfig(n_seeds=20, save_logs=False)
    exp = m.CognitiveDissonanceExperiment(cfg)
    t1 = cfg.commitment_step
    total = t1 + cfg.post_commitment_steps
    end = total - cfg.snapshot_every          # last snapshot step used by Exp 1

    def run(seed, cap):
        env = exp.create_environment(seed, is_control=False, condition="ground_truth_tied")
        agent = exp._create_agent(env, seed)
        state = env.reset()
        last = np.full(2, -1)                 # step of each arm's last Q-update
        committed = None
        snaps, stale_rej, forced = {}, [], 0
        for step in range(total):
            action = agent.select_action(state)
            post = step >= t1
            if post and committed is None:
                committed = int(np.argmax(env.arm_pull_counts))
            was_forced = False
            if post and cap is not None and (step - last).max() >= cap:
                action = int(np.argmax(step - last)); forced += 1; was_forced = True
            nxt, reward, done, info = env.step(action)
            if was_forced:                    # a forced visit is paid from the visited arm's own draw
                reward = env.get_all_rewards_at_step(step)[action]
            else:                             # otherwise the environment pulls the committed arm
                action = info.get("effective_action", action)
            agent.update(state, action, reward, nxt, True)
            last[action] = step
            if post:
                stale_rej.append(step - last[1 - committed])
                if step in (t1 + 100, end):
                    q = agent.get_q_values(state)
                    snaps[step] = float(q[committed] - q[1 - committed])
            state = nxt
        return {"div100": snaps[t1 + 100], "div_end": snaps[end],
                "mean_staleness_rejected": float(np.mean(stale_rej)), "forced": forced}

    def observer(seed):
        env = exp.create_environment(seed, is_control=True, condition="ground_truth_tied")
        agent = exp._create_agent(env, seed)
        met = exp._extract_metrics(exp._run_bandit_session(env, agent, is_observer=True))
        return met["divergence_at_100"], met["divergence_at_end"]

    obs = np.array([observer(s) for s in SEEDS])
    runs = {str(cap): [run(s, cap) for s in SEEDS] for cap in CAPS}
    col = lambda rows, k: np.array([r[k] for r in rows])  # noqa: E731

    summary = {}
    base = {k: col(runs["None"], k) - obs[:, i] for i, k in enumerate(("div100", "div_end"))}
    print(" cap  staleness(rej)  forced   excess@100 (SD)   excess@end (SD)   paired t(19) vs no cap, at end")
    for cap, rows in runs.items():
        e100 = col(rows, "div100") - obs[:, 0]
        eend = col(rows, "div_end") - obs[:, 1]
        entry = {"mean_staleness_rejected": float(col(rows, "mean_staleness_rejected").mean()),
                 "forced_visits": float(col(rows, "forced").mean()),
                 "excess_at_100": {"mean": float(e100.mean()), "std": float(e100.std(ddof=1))},
                 "excess_at_end": {"mean": float(eend.mean()), "std": float(eend.std(ddof=1))}}
        line = (f" {cap:>4} {entry['mean_staleness_rejected']:11.1f} {entry['forced_visits']:9.1f}   "
                f"{e100.mean():+.3f} ({e100.std(ddof=1):.3f})   {eend.mean():+.3f} ({eend.std(ddof=1):.3f})   ")
        if cap != "None" and np.any(eend != base["div_end"]):
            t = stats.ttest_rel(eend, base["div_end"])
            entry["paired_t_vs_no_cap_at_end"] = {"t": float(t.statistic), "p": float(t.pvalue), "df": 19}
            line += f"t={t.statistic:.2f} p={t.pvalue:.1e}"
        summary[cap] = entry
        print(line)

    # within-seed monotonic trend across caps (one Spearman rho per seed)
    X = np.array([col(runs[str(c)], "mean_staleness_rejected") for c in CAPS])
    Y = np.array([col(runs[str(c)], "div_end") - obs[:, 1] for c in CAPS])
    rho = np.array([stats.spearmanr(X[:, i], Y[:, i]).statistic for i in range(len(obs))])
    summary["trend"] = {"mean_within_seed_spearman": float(rho.mean()),
                        "seeds_with_positive_rho": int((rho > 0).sum()), "n_seeds": int(len(rho))}
    print(f" trend: mean within-seed Spearman rho = {rho.mean():.2f}; positive in {(rho > 0).sum()}/{len(rho)} seeds")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "observer": obs.tolist(), "runs": runs}, open(OUT, "w"))
    print("saved", OUT)


if __name__ == "__main__":
    main()

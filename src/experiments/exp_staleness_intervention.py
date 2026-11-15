"""Staleness intervention on Exp 1 (exploratory; not yet reported in the paper).

Same environment, agent, seeds and decay rate as the paper's primary condition
(ground-truth-tied arms, lambda = 0.001). The only change: after step 500, any
arm that has gone `cap` steps without a Q-update is force-visited. cap = None
is the paper's primary condition and reproduces its +0.863 at 100 steps.

Two modes are run:
  free     the paper's actual design: after "commitment" the environment pays
           out from the committed arm, but the agent keeps selecting
           epsilon-greedily and credits the arm it selected.
  lock_in  commitment binds the agent: it selects and updates only the
           committed arm (unless a forced visit applies).

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
cfg = m.DissonanceConfig(n_seeds=20, save_logs=False)
exp = m.CognitiveDissonanceExperiment(cfg)
T1 = cfg.commitment_step
TOTAL = T1 + cfg.post_commitment_steps


def run(seed, cap, lock_in=False, decay=None):
    """One committed-agent session. lock_in=True makes commitment bind the
    agent's own updates (it selects only the committed arm unless forced)."""
    env = exp.create_environment(seed, is_control=False, condition="ground_truth_tied")
    agent = exp._create_agent(env, seed, q_decay_rate=decay)
    state = env.reset()
    last = np.full(2, -1)                 # step of each arm's last Q-update
    committed = None
    snaps, stale_rej, forced = {}, [], 0
    for step in range(TOTAL):
        action = agent.select_action(state)
        post = step >= T1
        if post and committed is None:
            committed = int(np.argmax(env.arm_pull_counts))
        if post and lock_in:
            action = committed
        was_forced = False
        if post and cap is not None:
            age = step - last
            if age.max() >= cap:
                action = int(np.argmax(age)); forced += 1; was_forced = True
        nxt, reward, done, info = env.step(action)
        if was_forced:                    # a forced visit is paid from the visited arm's own draw
            reward = env.get_all_rewards_at_step(step)[action]
        agent.update(state, action, reward, nxt, True)
        last[action] = step
        if post:
            stale_rej.append(step - last[1 - committed])
            if step - T1 in (100, cfg.post_commitment_steps - 1):
                q = agent.get_q_values(state)
                snaps[step - T1] = float(q[committed] - q[1 - committed])
        state = nxt
    return {"div100": snaps[100], "div_end": snaps[cfg.post_commitment_steps - 1],
            "mean_staleness_rejected": float(np.mean(stale_rej)), "forced": forced}


def observer(seed):
    env = exp.create_environment(seed, is_control=True, condition="ground_truth_tied")
    agent = exp._create_agent(env, seed)
    met = exp._extract_metrics(exp._run_bandit_session(env, agent, is_observer=True))
    return met["divergence_at_100"], met["divergence_at_end"]


out = {"observer": [observer(s) for s in SEEDS], "free": {}, "lock_in": {}, "lock_in_decay": {}}
for cap in CAPS:
    out["free"][str(cap)] = [run(s, cap) for s in SEEDS]
    out["lock_in"][str(cap)] = [run(s, cap, lock_in=True) for s in SEEDS]
for lam in (0.0, 0.001, 0.01):            # what Exp 1's decay sweep gives under true lock-in
    out["lock_in_decay"][str(lam)] = [run(s, None, lock_in=True, decay=lam) for s in SEEDS]
OUT = "results/exp_staleness_intervention/results.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("saved", OUT)

# ---- summary ----
d = out; obs = np.array(d["observer"])
col = lambda rows, k: np.array([r[k] for r in rows])
for mode in ("free", "lock_in"):
    print(f"\n=== {mode} (lambda=0.001, N=20) ===")
    print(" cap  stale(rej)  forced   excess@100 (SD)   excess@end (SD)   paired t(19) vs no cap: @100 | @end")
    b100 = col(d[mode]["None"], "div100") - obs[:, 0]; bend = col(d[mode]["None"], "div_end") - obs[:, 1]
    for cap, rows in d[mode].items():
        e100 = col(rows, "div100") - obs[:, 0]; eend = col(rows, "div_end") - obs[:, 1]
        line = f" {cap:>4} {col(rows,'mean_staleness_rejected').mean():9.1f} {col(rows,'forced').mean():8.1f}   {e100.mean():+.3f} ({e100.std(ddof=1):.3f})   {eend.mean():+.3f} ({eend.std(ddof=1):.3f})   "
        if cap != "None" and np.any(e100 != b100):
            a, b = stats.ttest_rel(e100, b100), stats.ttest_rel(eend, bend)
            line += f"t={a.statistic:.2f} p={a.pvalue:.1e} | t={b.statistic:.2f} p={b.pvalue:.1e}"
        print(line)
    # per-seed trend: slope of excess on log(mean staleness) across caps, tested against 0
    caps = list(d[mode]); X = np.log1p(np.array([col(d[mode][c], "mean_staleness_rejected") for c in caps]))
    for key, o in (("div100", 0), ("div_end", 1)):
        Y = np.array([col(d[mode][c], key) - obs[:, o] for c in caps])
        slopes = [np.polyfit(X[:, i], Y[:, i], 1)[0] for i in range(Y.shape[1])]
        r = [stats.spearmanr(X[:, i], Y[:, i]).statistic for i in range(Y.shape[1])]
        t = stats.ttest_1samp(slopes, 0)
        print(f"   trend {key}: per-seed slope vs log staleness mean {np.mean(slopes):+.3f}, t(19)={t.statistic:.2f}, p={t.pvalue:.1e}; mean per-seed Spearman rho={np.nanmean(r):.2f}")
print("\n=== true lock-in decay sweep ===")
for lam, rows in d["lock_in_decay"].items():
    e100 = col(rows, "div100") - obs[:, 0]; eend = col(rows, "div_end") - obs[:, 1]
    print(f" lambda={lam}: excess@100 {e100.mean():+.3f} ({e100.std(ddof=1):.3f}) t={stats.ttest_1samp(e100,0).statistic:.2f}   excess@end {eend.mean():+.3f} ({eend.std(ddof=1):.3f}) t={stats.ttest_1samp(eend,0).statistic:.2f}")

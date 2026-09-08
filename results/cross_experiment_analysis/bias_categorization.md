# Cross-Experiment Theoretical Framework: "Free" vs. "Architectural" Biases

## 1. Executive Summary

A central theoretical question in computational cognitive science and computational psychiatry is whether psychological biases in learning agents represent **inherent dynamical consequences** of standard reinforcement learning, or whether they require **specialized architectural inductive biases**.

Through our four empirical investigations, we demonstrate a clear dichotomy:
1. **"Free" Emergent Biases (Cognitive Dissonance & Choice Overload):** Require **zero** structural additions or bespoke loss terms. They emerge naturally from foundational reinforcement learning dynamics—specifically the interplay between asymmetric sampling and stale value estimates.
2. **"Architectural" Metacognitive Biases (Impostor Syndrome & Self-Handicapping):** Require explicit **hierarchical self-evaluative representations** (local decision-level confidence vs. global competence estimates) and asymmetric update rules that distort evidence integration.

---

## 2. Comparative Synthesis Table

| Dimension | Experiment 4.1: Cognitive Dissonance | Experiment 4.2: Impostor Syndrome | Experiment 4.3: Self-Handicapping | Experiment 4.4: Choice Overload |
| :--- | :--- | :--- | :--- | :--- |
| **Psychological Construct** | Post-decision spreading of alternatives | Persistent underconfidence despite competence | Pre-emptive defensive impediment selection | Decision paralysis & entropy from option proliferation |
| **Human Benchmark** | Brehm (1956); Chen & Risen (2010) | Katyal, Huys, Dolan & Fleming (2025, *Nat. Comms.*) | Berglas & Jones (1978) | Iyengar & Lepper (2000); Schwartz (2004) |
| **Bias Category** | **"Free" (Dynamics-Driven)** | **"Architectural" (Metacognitive)** | **"Architectural" (Metacognitive)** | **"Free" (Dynamics-Driven)** |
| **Required RL Substrate** | Standard Q-learning / DQN (decay / drift) | Dual-tier confidence module (Local + Global) | Dual-tier confidence + Threat evaluation action | Standard Q-learning / Softmax policy |
| **Computational Mechanism** | Asymmetric visitation post-commitment; rejected arm value stales and decays | Damped sensitivity to positive local confidence signals during global aggregation | Ego-protective action selection gated by low global confidence under evaluation threat | Exploration dilution across near-tied actions; $1/K$ refresh rate inflates stale value dispersion |
| **Primary Metric** | Excess value divergence $\Delta Q = Q_{\text{chosen}} - Q_{\text{rejected}}$ | Calibration gap $\mathcal{G} = \text{Objective Perf} - \text{Global Confidence}$ | Handicap action selection frequency | Softmax policy decision entropy $H(\pi)$ |
| **Key Empirical Result** | $\Delta Q = 3.32 \pm 0.45$ ($d = 2.21$, $p < 10^{-6}$) | $\mathcal{G} = 0.44 \pm 0.05$ vs $0.02$ ($d = 8.05$, $p \approx 0$) | Modest baseline effect in gridworld; requires high evaluative cost | Monotonic scaling $0.65 \to 1.99$ ($d > 16.5$, $p < 10^{-15}$) |
| **Ablation / Confound Control** | Forced equal visitation post-commitment yields $\Delta Q = 0.00$ | Symmetric update control agent achieves near-zero gap $\mathcal{G} = 0.02$ | Absence of evaluative threat eliminates handicap incentive | 2-arm baseline confirms minimal entropy and rapid commitment |
| **Function Approximation Generality** | Confirmed in Deep Q-Network (weight decay / buffer turnover) | Extensible via ensemble / distributional local confidence | Extensible via actor-critic meta-action | Confirmed in Deep Q-Network (entropy scaling holds) |

---

## 3. Mechanistic Deep-Dive

### 3.1 The "Free" Biases: Stale Value Dynamics
In both Experiment 4.1 and 4.4, the agent does not possess a "self-concept" or an intrinsic preference for consistency. Rather:
- **Asymmetric Sampling:** Once an action is committed or preferred, alternative actions cease to be sampled.
- **Staleness Accumulation:** Unsampled actions accumulate latency $\tau$. Under any continuous regularization (weight decay in DQN, synaptic decay in tabular Q, or non-stationary drift), stale values drift toward the prior or zero.
- **The Illusion of Rationalization:** When tested, the agent exhibits an apparent "rationalization" of its choice ($\Delta Q > 0$), not because it motivatedly inflated the winner, but because the unchosen options decayed into informational obsolescence.
- **The Anatomy of Paralysis:** As the choice set grows from $K=2$ to $K=16$, the sampling probability under $\epsilon$-greedy exploration dilutes to $\epsilon / K$. Arms remain unvisited for longer spans ($E[\tau] \propto K$). The resulting stale noise prevents the policy from breaking symmetry, trapping the agent in high decision entropy.

### 3.2 The "Architectural" Biases: Metacognitive Distortion
In contrast, neither Cognitive Dissonance nor standard Q-learning can generate Impostor Syndrome or Self-Handicapping:
- Standard agents are strictly self-consistent: their value estimates directly track discounted future reward. They cannot simultaneously perform well and believe they perform poorly.
- Generating persistent underconfidence requires separating **local performance tracking** from **global self-evaluation**.
- As empirically discovered by Katyal et al. (2025) in human clinical populations, the distortion is not general negativity, but an **asymmetric attenuation of positive evidence**. When an agent discounts successes while faithfully recording failures, its global self-assessment decouples from ground truth, generating a persistent calibration gap ($d = 8.05$).

---

## 4. Implications for AI Safety & Cognitive Architecture
1. **Spontaneous Emergence:** Standard RL agents deployed in recommendation systems, autonomous trading, or resource allocation will spontaneously exhibit post-decision rationalization and choice paralysis unless explicit counter-decay or exploration-refresh mechanisms are enforced.
2. **Metacognitive Vulnerability:** Incorporating human-like self-monitoring into AI agents (e.g., LLM reflection loops, RL self-critics) introduces the vulnerability to architectural cognitive distortions if evidence integration is not strictly symmetric.

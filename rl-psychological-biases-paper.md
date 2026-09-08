# Computational Analogues of Post-Decision and Self-Evaluative Biases in Deep Reinforcement Learning Agents

**Project type:** Research paper, intended for publication
**Status:** Pre-implementation / planning complete, ready to begin experiments
**Core inspiration:** Human psychological phenomena — specifically cognitive biases that arise *after* a decision is made or *about* one's own competence — reframed as testable computational hypotheses in reinforcement learning (RL) agents

---

## 0. One-paragraph summary (read this first)

Reinforcement learning has deep historical roots in behavioral psychology (reward prediction error, temporal-difference learning, and operant conditioning are close cousins), and a growing field called *computational psychiatry* already uses RL to model human biases like addiction, depression, and compulsivity. This project asks a narrower, less-explored question: can a handful of specific human *post-decision* and *self-evaluative* biases — cognitive dissonance, impostor syndrome, self-handicapping, and choice overload — be reproduced as emergent behavior in deep RL agents, without hand-engineering the bias directly into the reward function? The goal is not to claim RL agents "feel" these things, but to show that the same *learning-dynamics mechanisms* that produce these biases in humans (asymmetric feedback weighting, stale value estimates for unchosen options, credit-assignment quirks) can produce structurally similar behavioral and representational signatures in artificial agents — and to identify *why* mechanistically.

---

## 1. Motivation: why this is worth doing

- RL and psychology share a common mathematical ancestor: temporal-difference learning was explicitly inspired by animal conditioning experiments, and reward-prediction-error signals in RL models are widely used to explain dopamine activity in the brain.
- A large and active subfield — computational psychiatry / computational cognitive neuroscience — already uses RL to formally model human biases and disorders (e.g., habit vs. goal-directed control, addiction as a hijacked reward signal, depression as prevalence-driven learned helplessness).
- However, that literature is overwhelmingly focused on **learning-phase** biases (how agents update values from reward) rather than **decision-phase and self-evaluative** biases (how agents behave *after* a choice is made, or how they represent confidence in their own competence).
- This is a genuine, underexplored seam: post-decision rationalization (cognitive dissonance) and self-evaluation under uncertainty (impostor syndrome, self-handicapping) have rich human psychology literatures but essentially no direct deep-RL implementation.
- Framing it this way gives the paper both a hook (the psychological framing is intuitive and communicates well) and rigor (each phenomenon is reduced to a precise, falsifiable computational claim).

---

## 2. Related work landscape (what's already claimed — read before writing anything)

This section exists so that neither you nor a coding agent accidentally "discovers" something that's already published. Cite these explicitly in the paper's Related Work section, and use them to sharpen — not abandon — the novelty claims below.

### 2.1 Psychological phenomena already modeled in RL (adjacent, not this paper's core claim)
| Phenomenon | RL treatment | Notes |
|---|---|---|
| Action oscillation / behavioral instability | "Policy Inertia Controller," Chen et al., AAAI 2021 | Closest existing analog to "indecisiveness"; agents flip-flop between actions in near-identical states |
| Sunk cost fallacy | ICML 2025 paper on deep RL agents completing wasteful episodes | Direct deep-RL demonstration |
| Procrastination | TD-learning / compressed state-representation papers | Shows procrastination emerging from value-approximation error |
| Learned helplessness | Long history in computational psychiatry | Predicted by reward prevalence, not perceived controllability |
| Confirmation / positivity bias | RL agents that overweight confirming prediction errors | Can *outperform* unbiased agents under noise; linked to overconfidence and status-quo bias |
| Habit vs. goal-directed control | Daw, Dayan, Dolan et al. (model-free vs. model-based arbitration) | Foundational computational-psychiatry framework (OCD, addiction, compulsivity) |
| Loss aversion / prospect theory | Cumulative Prospect Theory folded into multi-agent RL value transforms | |
| Superstition | *"Superstition" in the Network: Deep RL Plays Deceptive Games* (Bontrager et al.) | Direct callback to Skinner's 1948 pigeon experiments |
| Partial reinforcement extinction effect (PREE) | Well established in animal/human behaviorism (Humphreys 1939; Ferster & Skinner 1957) | **Not yet ported to deep RL** — a real opportunity if you want a 5th/6th experiment later |
| Plasticity loss / primacy bias / capacity loss | Nikishin et al. (ICML 2022), Lyle et al., Abbas et al., and a large active subfield | **This is why "burnout" was excluded from scope** — see Section 6 |
| Paradox of choice (training-time) | *The Paradox of Choice: Using Attention in Hierarchical RL* | Shows fewer available options → faster learning in hierarchical RL. This paper is about **training efficiency**, not **decision-time indecisiveness** — the distinction this project's choice-overload experiment must maintain |
| Reward expectation / subjective reward shaping | *The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons* | Builds agents with subjective reward shaped by expectation/comparison — mechanistically adjacent to the optional placebo-effect extension (Section 5.1) |
| "Self-doubt" (terminology, not construct) | *Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery* | Explicitly defines "self-doubt" as a training-stability signal, **not** the psychological construct — leaves the actual construct open, see Section 4.2 |

### 2.2 Human cognitive-science grounding to cite for construct validity
- Post-decision dissonance / "spreading of alternatives" — Brehm (1956); classic Aesop's-fox framing used pedagogically in cognitive dissonance literature.
- Persistent underconfidence despite intact performance — "Distorted learning from local metacognition supports transdiagnostic underconfidence," *Nature Communications*, 2025. Mechanistic account: local (per-trial) and global (long-run) confidence update asymmetrically under biased feedback processing.
- Persistent over/underconfidence emerging from rational belief updating under ambiguous self-assessment — Lemoine, "Rationally Misplaced Confidence" (University of Arizona / NBER working paper).
- Choice-type effects on human RL — *Choice Type Impacts Human Reinforcement Learning*, Journal of Cognitive Neuroscience, 2023. Shows irrelevant credit assignment and slower value integration under ambiguous/larger choice spaces.
- Schwartz, B. — original "Paradox of Choice" behavioral-economics framing (decision paralysis, regret, and reduced satisfaction under many options).

---

## 3. Core research question

> **Do deep reinforcement learning agents, without the bias being explicitly hand-coded into the reward function, exhibit behavioral and representational signatures analogous to specific human post-decision and self-evaluative cognitive biases — and if so, what learning-dynamics mechanism produces each one?**

Every experiment in this paper must answer three things:
1. **What human phenomenon is being modeled**, with a citation to the psychology/cognitive-science literature defining it.
2. **What existing RL work is adjacent**, and precisely how this experiment differs from it (see Section 2.1 — do not skip this for any experiment).
3. **What mechanistic explanation** is being proposed — not just "the effect appears," but *why*, in terms of TD-error dynamics, function approximation, replay buffer composition, or credit assignment.

---

## 4. Core scope — four experiments (this is the paper's spine)

### 4.1 Cognitive Dissonance (Post-Decision Spreading of Alternatives)

**Human phenomenon:** After choosing between close alternatives, people resolve the discomfort of having rejected an attractive option by inflating their evaluation of the chosen option and devaluing the rejected one — even though nothing about the options themselves changed (Brehm, 1956).

**Gap:** No existing deep-RL paper models this directly.

**Operational definition:** After an agent commits to action *a* over a near-equal-value action *b* in the same state, does its own subsequent value estimate for *a* increase and for *b* decrease **beyond what the TD-error / realized reward alone would justify**?

**Experimental design:**
- Environment: repeated two-armed (or few-armed) bandit with two near-equal-value arms.
- Compare a normal learning agent (which stops sampling the rejected arm after committing) against a counterfactual "observer" agent that receives the same reward sequence without having "chosen" (i.e., continues to sample both arms, or has rewards injected without commitment).
- Track the value-estimate trajectory for both the chosen and rejected arm across post-commitment steps.
- Test whether divergence between chosen/rejected value estimates exceeds what's predicted by reward statistics alone.

**Hypothesis:** Standard deep RL agents will show value divergence for chosen-vs-rejected near-equal options beyond pure TD-error updating, driven by asymmetric visitation frequency post-commitment (stale value estimates decaying/drifting under whatever regularization or bootstrapping scheme is in use) — a mechanistic, non-mystical explanation for an effect that looks identical to human post-decision dissonance.

---

### 4.2 Impostor Syndrome (Persistent Underconfidence Despite Good Performance)

**Human phenomenon:** Persistent self-doubt and underestimation of one's own competence despite objectively strong and consistent performance; formally linked to asymmetric updating between local (per-instance) and global (long-run) self-assessment under biased feedback processing (*Nature Communications*, 2025).

**Gap:** No deep-RL implementation of the actual construct exists — the one paper using the term "self-doubt" in RL explicitly defines it as a training-stability signal, not this.

**Operational definition:** Give the agent two confidence channels:
- **Local confidence** — per-decision value-estimate confidence (e.g., from an ensemble of Q-networks or a distributional critic).
- **Global confidence** — a slower-moving running estimate of "how good am I at this task overall," updated on a separate, slower timescale, with an update rule that weights negative feedback more heavily than positive feedback (mirroring the human mechanism).

**Experimental design:**
- Train the agent normally; measure **actual performance** (e.g., percentile rank against other agents, or against its own historical best).
- Measure the **calibration gap**: global self-assessed competence vs. actual measured performance.
- Compare against a symmetric-feedback control agent (equal weighting of positive/negative feedback) to isolate the asymmetry as the causal driver.
- Behavioral readout: does the biased agent underexplore or under-commit to actions it is objectively good at, purely because of the lagging global-confidence signal?

**Hypothesis:** The asymmetric-feedback agent develops a persistent competence/self-assessment gap that does not close with more experience, and this gap measurably affects behavior (exploration rate, action confidence, risk-taking) even when raw performance matches the well-calibrated control agent.

---

### 4.3 Self-Handicapping (paired with 4.2 — shares the same architecture)

**Human phenomenon:** Deliberately sabotaging one's own performance ahead of an evaluation, so that failure can be attributed to the handicap rather than to a lack of ability — a defense mechanism that protects self-image at the cost of actual performance.

**Gap:** No existing RL treatment found.

**Why it belongs with 4.2, not as a standalone experiment:** it is the behavioral mirror image of impostor syndrome — same global-confidence module, opposite failure mode (impostor syndrome = doubt despite good performance; self-handicapping = deliberately *creating* an excuse for anticipated poor performance).

**Operational definition:** Reuse the global-confidence module from 4.2. Whenever the agent anticipates a high-stakes evaluation event that risks a large negative update to its global confidence estimate, does it choose an action that creates a plausible external "excuse" (an observable handicap) rather than the actual expected-value-maximizing action?

**Experimental design:**
- Introduce an environment feature representing an "evaluation event" (a state where performance will be visibly scored).
- Give the agent access to an action that measurably reduces expected reward but also introduces an observable "handicap" flag.
- Measure whether the agent's uptake of the handicapping action correlates with proximity to evaluation events and with the fragility of its current global-confidence estimate.

**Hypothesis:** Agents with the asymmetric global-confidence architecture from 4.2 will take the handicapping action disproportionately more often immediately before evaluation events than a control agent without the self-evaluation module, even though it costs expected reward.

---

### 4.4 Choice Overload / Paradox of Choice (decision-time framing)

**Human phenomenon:** Barry Schwartz's "paradox of choice" — more available options leads to decision paralysis, lower decision quality, and reduced post-decision satisfaction, even when a clearly optimal option is present.

**Gap — be precise about this one:** *The Paradox of Choice: Using Attention in Hierarchical RL* already exists and shows that restricting available choices speeds up **training**. That is a training-time efficiency result. This experiment is explicitly about a **fully trained** agent's decision quality/consistency **at evaluation time**, as the size of a near-tied option set grows — a different claim, and the paper must say so explicitly in Related Work to preempt a reviewer flagging it as duplicate.

**Operational definition:** Take a flat (non-hierarchical), already-converged agent. At evaluation time, vary the number of near-tied high-value actions available in a given state (e.g., 2 vs. 8 vs. 20 actions within ε of optimal value).

**Experimental design:**
- Measure **decision entropy** (spread of the action distribution) as a latency/indecision proxy.
- Measure **realized regret** — does average return drop as the choice set grows, even though the optimal action is always technically present?
- Measure **post-hoc value re-evaluation** ("buyer's remorse") — does the agent's own value estimate for the chosen action drop relative to the road not taken immediately after choosing? (This connects directly back to the cognitive dissonance experiment in 4.1 — consider running them on a shared codebase.)

**Hypothesis:** Performance and decision confidence degrade non-monotonically as the near-tied option set grows, even holding the optimal action's true value constant — mirroring the inverted-U regret curve found in human choice-overload studies.

---

## 5. Optional stretch extensions (only pursue after the core four are solid)

### 5.1 Placebo Effect (reward-expectation modulation)
- **Gap status:** Partial. *The pursuit of happiness* paper already builds RL agents with subjective reward shaped by expectation and comparison to a baseline (hedonic adaptation framing). The differentiator for this project would need to be a **cue-triggered, treatment-like structure** — an external signal that predicts reward but carries no actual information — rather than ongoing habituation/comparison.
- **Risk:** Moderate. Requires careful citation of the adjacent paper to avoid looking like a miss.

### 5.2 FOMO / Anticipatory Regret
- **Gap status:** Plausible, connects naturally to 4.1 and 4.4 (does a rejected alternative keep intruding on behavior after commitment).
- **Risk:** High. Regret-minimization is one of the most saturated subfields in RL (regret bounds, counterfactual regret minimization, regret-based exploration). The differentiator must be that the bias is *internally represented and behaviorally distorting* beyond what optimal regret-aware exploration would already predict — otherwise a reviewer will call this "just regret" and be right to.

**Recommendation:** Do not commit to either of these until the core four experiments (Section 4) are implemented and show clean results. If time/space allows, 5.1 is the safer addition of the two.

---

## 6. Explicitly excluded phenomena, and why (keep this section — it shows the reviewer/professor you did the diligence)

| Phenomenon | Why excluded |
|---|---|
| **Nostalgia** | Needs a fundamentally different methodological skeleton — long-horizon memory/replay valuation distorted by retrospection, not single-decision or self-evaluation bias. Would require a second experimental apparatus (memory architecture, time-decay of valuation) that shares nothing with the other four. Good idea for a *separate* paper. |
| **Groupthink / conformity cascades** | Requires multi-agent RL — observation of others' choices, a conformity metric, a social architecture. Different paper's worth of infrastructure. |
| **Burnout** (declining performance under sustained reward) | **Already claimed territory.** This is mechanistically identical to *plasticity loss / primacy bias / capacity loss*, an active, well-established deep-RL subfield (Nikishin et al. 2022; Lyle et al.; Abbas et al.) with specific mechanistic explanations already published (dead ReLU neurons, weight-norm growth, feature-rank collapse) and multiple mitigation methods. Re-labeling this as "burnout" would draw an immediate "isn't this just X" objection and should not be included. |

---

## 7. Shared methodological skeleton (use this for every experiment)

1. **Base agent:** Start with tabular Q-learning or a simple DQN/actor-critic for interpretability; only move to larger architectures once the effect is established in the simple case.
2. **Environments:** Bandits (for 4.1, 5.1, 5.2) and small gridworlds (for 4.2, 4.3, 4.4) — keep environments simple enough that value estimates are fully interpretable.
3. **Control vs. biased-agent paradigm:** Every experiment needs an unbiased control agent trained identically except for the specific mechanism under test (asymmetric feedback weighting, commitment-triggered value updates, etc.). The finding is the *difference* between biased and control agents, not just the biased agent's raw behavior.
4. **Metrics, two categories:**
   - *Representational* — value estimates, confidence calibration gaps, entropy of value distributions.
   - *Behavioral* — action choices, decision latency/entropy proxies, realized regret, exploration rate.
5. **Statistical validation:** Multiple random seeds (minimum 10–20 per condition), report effect sizes and confidence intervals, not just means. Run ablations to confirm the proposed mechanism (e.g., for 4.1, show the effect disappears if visitation frequency is held constant, to prove it's the *mechanism* you claim and not a confound).

---

## 8. Proposed paper structure

1. **Abstract**
2. **Introduction** — motivation (Section 1), research question (Section 3)
3. **Related Work** — Section 2, organized as "phenomena already modeled in RL" vs. "human cognitive-science grounding"
4. **Formalizing Psychological Constructs in RL** — a short methods section defining the shared skeleton (Section 7) before diving into individual experiments
5. **Experiments** — one subsection per phenomenon (4.1–4.4, plus 5.1/5.2 if included), each following: setup → hypothesis → results → mechanism
6. **Discussion** — cross-cutting themes: which biases emerged "for free" from standard learning dynamics vs. which required specific architectural choices; what this implies about the relationship between RL and human cognition
7. **Limitations** — be explicit that these are *computational analogues*, not claims that RL agents have subjective experience; discuss generalization beyond toy environments
8. **Conclusion**

---

## 9. Implementation roadmap (for a coding agent or for yourself)

- **Phase 0 — Setup:** Build the shared bandit and gridworld environments; implement the base agent (tabular Q-learning first) and the control/biased-agent comparison harness described in Section 7.
- **Phase 1 — Experiment 4.1 (Cognitive Dissonance):** Simplest to implement (plain bandit). Build this first to validate the overall pipeline (logging, statistical comparison, plotting value trajectories).
- **Phase 2 — Experiments 4.2 + 4.3 (Impostor Syndrome + Self-Handicapping):** Build the shared global/local confidence module once; instrument both experiments off the same module.
- **Phase 3 — Experiment 4.4 (Choice Overload):** Requires a converged baseline agent and an evaluation-time protocol for varying the near-tied action set size.
- **Phase 4 — Cross-experiment analysis:** Look for shared mechanisms across experiments (e.g., does the stale-value-estimate mechanism from 4.1 also explain part of 4.4's "buyer's remorse" metric?).
- **Phase 5 — Writeup:** Draft using the structure in Section 8.
- **Phase 6 (optional, only if time allows):** Sections 5.1 and/or 5.2.

---

## 10. Success criteria — what would make this publishable

- Each experiment has a **precise operational definition** that a skeptical reviewer can evaluate against the cited human-psychology definition.
- Each experiment includes an explicit **differentiation from the closest existing RL paper** (see Section 2.1 — do not skip this).
- Effects are **reproducible across seeds** and ideally across at least two agent architectures (e.g., tabular Q-learning and a small DQN).
- Each result comes with a **mechanistic explanation**, not just an observed correlation — this is what separates "cute analogy" papers from ones that get accepted.
- The paper explicitly engages with the "isn't this just [existing phenomenon]" objection for every experiment, especially 4.4 (vs. hierarchical-RL paradox-of-choice paper) and 5.2 (vs. regret-minimization literature) if included.

---

## 11. Known risks and open questions

- **Overclaiming risk:** Keep language disciplined — "computational analogue of X" or "a mechanism structurally similar to X," never "the agent feels X" or "the agent experiences X."
- **FOMO / regret overlap (5.2):** Highest risk of "prior work" rejection in the whole project. Only include if the internal-representation differentiation from Section 5.2 can be cleanly demonstrated.
- **Choice overload framing (4.4):** Must clearly state in the paper's first paragraph of that section that this is a decision-time, not training-time, claim, to preempt confusion with the existing hierarchical-RL paper.
- **Burnout temptation:** If, during experimentation, you notice something that looks like "burnout," check it against the plasticity-loss/primacy-bias literature (Section 2.1) before writing it up — it is very likely already explained.
- **Architecture generality:** Confirm whether the effects in 4.1–4.4 hold only for the specific agent architecture you test, or generalize — this affects how strong a claim the paper can make.

---

## 12. Key references (compile properly before submission — this is a working list, not a formatted bibliography)

- Chen, X. et al. "Policy Inertia Controller" — AAAI 2021 (action oscillation)
- Nikishin, E. et al. "The Primacy Bias in Deep Reinforcement Learning" — ICML 2022
- Bontrager, P. et al. "'Superstition' in the Network: Deep Reinforcement Learning Plays Deceptive Games" — AIIDE 2019
- Daw, N., Dayan, P., Dolan, R. et al. — model-free/model-based arbitration, habit vs. goal-directed control (foundational computational psychiatry work)
- Brehm, J. W. (1956) — post-decision dissonance / spreading of alternatives (foundational)
- "Distorted learning from local metacognition supports transdiagnostic underconfidence" — Nature Communications, 2025
- Lemoine, D. "Rationally Misplaced Confidence" — University of Arizona / NBER working paper
- "Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery" — arXiv, 2026
- "The Paradox of Choice: Using Attention in Hierarchical Reinforcement Learning" — arXiv 2201.09653
- "Choice Type Impacts Human Reinforcement Learning" — Journal of Cognitive Neuroscience, 2023
- "The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons" — PubMed, 2022
- Schwartz, B. — "The Paradox of Choice" (original behavioral economics framing)
- Humphreys, L. G. (1939) / Ferster & Skinner (1957) — partial reinforcement extinction effect (background, not yet ported to deep RL)
- Sutton, R. S. & Barto, A. G. — *Reinforcement Learning: An Introduction* (foundational RL/psychology connection)

**Note:** Verify full citations (page numbers, DOIs, exact venues) directly before submission — this list was compiled from search during project scoping and should be checked against the original sources.

---

## 13. How to use this document

- **Sending to a professor:** Sections 1–4 and 8 give the full scientific pitch; Section 6 shows you've done the differentiation work, which reviewers/advisors respond well to.
- **As a prompt for a coding agent:** Sections 7, 9, and the "Experimental design" subsections under Section 4 contain everything needed to start implementing — environments, metrics, comparison paradigm, and phased order.
- **For yourself:** Section 11 is worth re-reading before writing each section of the actual paper, to keep the claims disciplined.

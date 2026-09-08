# Computational Analogues of Post-Decision and Self-Evaluative Biases in Deep Reinforcement Learning Agents (v2 — revised after self-review)

**Project type:** Research paper, intended for publication
**Status:** Revised after literature-verification and methodology critique — ready to begin experiments
**Core inspiration:** Human psychological phenomena — specifically cognitive biases that arise *after* a decision is made or *about* one's own competence — reframed as testable computational hypotheses in reinforcement learning (RL) agents
**Revision note:** This version corrects citation errors, tightens one weak novelty claim, fixes a mechanism mismatch, adds a confound control, and specifies a multiple-comparisons approach, following a self-critique pass documented in `critique-report.md`.

---

## 0. One-paragraph summary

Reinforcement learning has deep historical roots in behavioral psychology, and a growing field called computational psychiatry already uses RL to model human biases like addiction, depression, and compulsivity. This project asks a narrower question: can specific human *post-decision* and *self-evaluative* biases — cognitive dissonance, impostor syndrome, self-handicapping, and (with caveats noted below) choice overload — be reproduced as emergent behavior in deep RL agents, without hand-engineering the bias directly into the reward function? The goal is not to claim RL agents "feel" these things, but to show that the same *learning-dynamics mechanisms* that produce these biases in humans can produce structurally similar behavioral and representational signatures in artificial agents — and to identify why, mechanistically, and to benchmark the resulting patterns against specific published human findings rather than against the psychological literature in general terms.

---

## 1. Motivation

- RL and psychology share a mathematical ancestor: temporal-difference learning was inspired by animal conditioning, and reward-prediction-error signals in RL are widely used to explain dopamine activity.
- Computational psychiatry already formally models many human biases with RL (habit vs. goal-directed control, addiction, prevalence-driven learned helplessness).
- That literature is overwhelmingly focused on the **learning phase** (how agents update values from reward) rather than the **decision phase and self-evaluation** (how agents behave after a choice, or represent confidence in their own competence).
- Post-decision rationalization and self-evaluation under uncertainty have rich human psychology literatures but essentially no direct deep-RL implementation — with one experiment (choice overload) now known to be more contested than originally thought; see Section 4.4.

---

## 2. Related work landscape (updated)

### 2.1 Psychological phenomena already modeled in RL
| Phenomenon | RL treatment | Verification status |
|---|---|---|
| Action oscillation / behavioral instability | "Policy Inertia Controller," Chen et al., AAAI 2021 | Not re-verified this pass (carried from earlier search) |
| Sunk cost fallacy | A 2025 ICML paper on deep RL agents completing wasteful episodes | **TODO — exact citation not yet confirmed; do not cite as-is** |
| Procrastination | TD-learning / compressed state-representation work | **TODO — exact citation not yet confirmed; do not cite as-is** |
| Learned helplessness | Computational psychiatry literature (reward-prevalence account) | **TODO — exact citation not yet confirmed; do not cite as-is** |
| Confirmation / positivity bias | RL agents overweighting confirming prediction errors | **TODO — exact citation not yet confirmed; do not cite as-is** |
| Habit vs. goal-directed control | Daw, Dayan, Dolan et al. — model-free/model-based arbitration | Not re-verified this pass, high prior confidence (well-known foundational work) |
| Superstition | Bontrager et al., *"Superstition" in the Network*, AIIDE 2019 | Not re-verified this pass (carried from earlier search) |
| Plasticity loss / primacy bias | **Verified this pass:** Nikishin, Schwarzer, D'Oro, Bacon, Courville, "The Primacy Bias in Deep Reinforcement Learning," ICML 2022 (arXiv:2205.07802) | ✅ Confirmed exact title, authors, venue, abstract match |
| Paradox of choice (training-time, hierarchical RL) | **Verified this pass:** Nica, Khetarpal, Precup, "The Paradox of Choice: Using Attention in Hierarchical Reinforcement Learning," NeurIPS 2022 workshop (arXiv:2201.09653) | ✅ Confirmed. Note: workshop paper, not main-track. |
| Paradox of choice (decision-time, bandit) — **newly found** | "Reinforcement Learning with Brain-Inspired Modulation can Improve Adaptation to Environmental Changes" (arXiv:2205.09729) — explicitly demonstrates a bandit-arm paradox-of-choice effect using Schwartz's own framing; a modulation rule helps with few arms, hurts with many | ✅ Confirmed, found in self-review pass. **This significantly narrows Section 4.4's novelty claim — see below.** |
| Reward expectation / subjective reward, incl. "too many similar options" degrading performance | **Verified this pass:** Dubey, Griffiths, Dayan, "The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons," *PLOS Computational Biology*, 2022 (corrected from earlier miscitation as "PubMed") | ✅ Confirmed exact venue/authors/abstract. Directly relevant to both 4.4 and 5.1. |
| "Self-doubt" (terminology, not construct) | "Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery" | Not re-verified this pass (carried from earlier search) |

### 2.2 Human cognitive-science grounding
- **Verified this pass:** Brehm, J. W. (1956). "Postdecision changes in the desirability of alternatives." *Journal of Abnormal and Social Psychology*, 52(3), 384–389. Confirmed exact title, journal, and pages.
- **Important addition from this pass:** Chen, M. K. & Risen, J. L. (2010) showed the classic Brehm free-choice paradigm can produce apparent preference change as a **statistical artifact** of revealed pre-existing preference, not genuine dissonance. This directly informs the confound control added to Section 4.1 below.
- **Verified this pass:** Katyal, Huys, Dolan & Fleming (2025). "Distorted learning from local metacognition supports transdiagnostic underconfidence." *Nature Communications*, 16, 1854. DOI: 10.1038/s41467-025-57040-0. Confirmed authors, exact title, venue, and pulled the actual abstract — see mechanism correction in Section 4.2.
- Schwartz, B. — original "Paradox of Choice" framing (not re-verified as a specific citation this pass, but the concept is well-established and widely cited).
- Choice Type Impacts Human Reinforcement Learning, *Journal of Cognitive Neuroscience*, 2023 — not re-verified this pass.

---

## 3. Core research question

> **Do deep reinforcement learning agents, without the bias being explicitly hand-coded into the reward function, exhibit behavioral and representational signatures analogous to specific human post-decision and self-evaluative cognitive biases, matching the *specific pattern* found in a named human study — and if so, what learning-dynamics mechanism produces each one?**

(Added "matching the specific pattern found in a named human study" per the domain-fit resolution in Section 6.)

---

## 4. Core scope

### 4.1 Cognitive Dissonance (Post-Decision Spreading of Alternatives) — **revised design**

**Human phenomenon:** Brehm (1956); post-decision "spreading of alternatives."

**Known controversy to control for:** Chen & Risen (2010) — apparent spreading can be a statistical artifact of revealed pre-existing preference rather than genuine post-decision change.

**Revised operational definition and design:**
- **Primary condition:** Construct bandit arms with a *verified identical ground-truth reward distribution* (not just empirically close estimates) so that any observed post-commitment value divergence cannot be explained by a real underlying preference the agent "discovered" through choosing.
- **Secondary condition:** Empirically near-tied arms (the original design), included for comparison but explicitly labeled as vulnerable to the revealed-preference confound.
- Compare a normal learning agent (which stops sampling the rejected arm after committing) against a counterfactual "observer" agent that receives the same reward sequence without having "chosen."
- Track value-estimate trajectories for chosen vs. rejected arms in both conditions.

**Hypothesis:** Value divergence appears even in the ground-truth-tied primary condition, ruling out the revealed-preference explanation, and is driven by asymmetric post-commitment visitation frequency (stale value estimates decaying under whatever regularization/bootstrapping scheme is used).

---

### 4.2 Impostor Syndrome — **mechanism corrected**

**Human phenomenon:** Persistent underconfidence despite good performance.

**Corrected mechanism (from verified source):** Katyal et al. (2025) found that global (long-run) self-assessed confidence shows **reduced sensitivity to local (per-instance) confidence and performance signals** — a damping/discounting effect on well-performing evidence — rather than a simple "weights negative feedback more than positive feedback" rule as the original brief proposed.

**Revised operational definition:**
- **Local confidence** — per-decision value-estimate confidence (e.g., ensemble of Q-networks or a distributional critic).
- **Global confidence** — a slower-moving running estimate of overall competence, updated with a **damping/discount factor applied specifically to positive/high-performance local signals** (not a general negative-over-positive asymmetry), directly mirroring the verified mechanism.

**Experimental design (unchanged from original):** measure the calibration gap between global self-assessed competence and actual performance; compare against a symmetric-updating control agent; check behavioral effects (exploration rate, under-commitment to actions the agent is objectively good at).

**Hypothesis:** The damped-positive-signal agent develops a persistent competence/self-assessment gap that tracks the qualitative shape of the human finding (gap driven specifically by discounted positive evidence, not general pessimism), distinguishing this mechanism from a simpler negativity-bias account.

---

### 4.3 Self-Handicapping (paired with 4.2, unchanged)

Same design as the original brief — reuses the corrected global-confidence module from 4.2. Citation status for the human construct itself was not re-verified this pass; standard psychology reference (Berglas & Jones, 1978) should be confirmed before submission.

---

### 4.4 Choice Overload / Paradox of Choice — **substantially narrowed claim, decision required**

**What changed:** Two additional papers found in this pass — the brain-inspired-modulation bandit paper and the Dubey et al. "too many similar options" finding — both already show decision-time (not just training-time) degradation with more options. The clean "training-time vs. decision-time" differentiation from the original brief no longer fully separates this project from prior work.

**Two honest paths forward — pick one before implementing:**

- **Path A (narrower but still viable):** Focus specifically on **post-hoc value re-evaluation / "buyer's remorse"** — does the agent's own value estimate for the chosen action drop relative to the road not taken, immediately after choosing, as a function of choice-set size? This connects directly to the 4.1 mechanism and hasn't been shown in either of the newly-found papers, which focus on realized performance/regret, not the agent's internal post-hoc value representation. This is the recommended path.
- **Path B (demote to optional):** If Path A doesn't pan out early in implementation, demote this experiment to an optional/discussion-section comparison against the two newly-found papers rather than a core claimed contribution, and reallocate paper space to 4.1–4.3.

---

## 5. Optional stretch extensions

### 5.1 Placebo Effect
Unchanged from original, but note the differentiation target is now sharper given the confirmed Dubey et al. paper: differentiate via a **cue-triggered, uninformative treatment signal** (placebo structure) vs. Dubey et al.'s ongoing habituation/comparison structure, which is continuous rather than cue-triggered.

### 5.2 FOMO / Anticipatory Regret
Unchanged — still the highest-risk optional extension; not re-assessed this pass.

---

## 6. Domain-fit: revised, more precise recommendation

**What the self-review found:** computational psychiatry has a genuine theory-driven/mechanistic sub-tradition that does not strictly require new patient data. But even that sub-tradition almost always ties its models to a **specific named clinical construct or symptom dimension**, not a general "this resembles a psychological phenomenon" framing.

**Revised recommendation (replaces the binary Path A/Path B framing from v1):** Keep the agent-only design (no new human data collection required), but require every experiment to report results **benchmarked against the specific quantitative pattern in a named human study** — e.g., for 4.2, compare the shape of the model's competence/self-assessment gap curve against the actual local-vs-global sensitivity curve reported in Katyal et al. (2025), not just a qualitative "agents show underconfidence too." This is achievable without new data collection and gives the paper a legitimate, specific claim of correspondence to computational psychiatry findings rather than a loose analogy. Target venues: computational psychiatry / computational cognitive science venues that accept this framing (verify specific ones before submission — not independently confirmed this pass), or cognitive-science/ML venues as a fallback if reviewers there don't accept the correspondence framing.

---

## 7. Shared methodological skeleton — updated

1. **Base agent:** Tabular Q-learning or simple DQN/actor-critic first.
2. **Environments:** Bandits (4.1, 5.1, 5.2) and small gridworlds (4.2, 4.3, 4.4).
3. **Control vs. biased-agent paradigm**, as before.
4. **Metrics:** Representational (value estimates, calibration gaps) and behavioral (action choices, entropy, regret).
5. **Statistical validation, revised:** Minimum 10–20 seeds per condition, effect sizes and confidence intervals, **plus explicit multiple-comparisons correction across the full set of core hypothesis tests (Holm-Bonferroni recommended as a simple, defensible default; pre-registration via OSF as a stronger alternative if timeline allows)**.

---

## 8. Proposed paper structure

Unchanged from v1: Abstract, Introduction, Related Work, Formalizing Psychological Constructs in RL, Experiments (4.1–4.4 with 4.4 following whichever path was chosen in Section 4.4), Discussion, Limitations, Conclusion.

---

## 9. Implementation roadmap — updated

- **Phase 0 — Setup:** Shared environments + base agent + control/biased-agent harness.
- **Phase 1 — Experiment 4.1:** Build the **ground-truth-tied primary condition first** (not the empirically-near-tied version), since it's the one that actually rules out the revealed-preference confound.
- **Phase 2 — Experiments 4.2 + 4.3:** Build the corrected damped-positive-signal confidence module; instrument both experiments off it.
- **Phase 3 — Experiment 4.4:** Implement Path A (post-hoc value re-evaluation) first; only proceed further if it shows a clean effect distinct from realized-regret metrics already covered by prior work.
- **Phase 4 — Cross-experiment analysis, benchmarked against named human studies per Section 6.**
- **Phase 5 — Writeup.**
- **Phase 6 (optional) — 5.1/5.2.**

---

## 10. Success criteria — updated

All original criteria, plus:
- Each experiment's results are reported **against the specific quantitative pattern of a named human study**, not just a qualitative resemblance claim (Section 6).
- Section 4.4 either shows a clean Path A effect distinct from the two newly-found papers, or is explicitly demoted to a discussion-section comparison rather than a claimed core contribution.
- All TODO citations in Section 2.1 are resolved to exact, verified references before submission.

---

## 11. Known risks — updated

- **New top risk:** Section 4.4 may not survive contact with the two newly-found papers even under Path A. Have a fallback plan (demote to discussion, Section 4.4 Path B) ready before investing significant implementation time.
- All risks from v1 remain (overclaiming language, FOMO/regret overlap, burnout temptation).
- **New:** several Related Work citations are marked TODO and must be resolved before any submission — do not carry them into a draft as-is.

---

## 12. Key references — updated status

**Verified this pass (safe to cite with confidence):**
- Brehm, J. W. (1956). Postdecision changes in the desirability of alternatives. *Journal of Abnormal and Social Psychology*, 52(3), 384–389.
- Chen, M. K. & Risen, J. L. (2010). [Critique of the free-choice paradigm — exact title/venue not pulled this pass, confirm before citing.]
- Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., & Courville, A. (2022). The Primacy Bias in Deep Reinforcement Learning. *ICML 2022*. arXiv:2205.07802.
- Nica, A., Khetarpal, K., & Precup, D. (2022). The Paradox of Choice: Using Attention in Hierarchical Reinforcement Learning. NeurIPS 2022 Workshop. arXiv:2201.09653.
- Dubey, R., Griffiths, T. L., & Dayan, P. (2022). The pursuit of happiness: A reinforcement learning perspective on habituation and comparisons. *PLOS Computational Biology*, 18(8). DOI: 10.1371/journal.pcbi.1010316.
- Katyal, S., Huys, Q. J. M., Dolan, R. J., & Fleming, S. M. (2025). Distorted learning from local metacognition supports transdiagnostic underconfidence. *Nature Communications*, 16, 1854. DOI: 10.1038/s41467-025-57040-0.
- "Reinforcement Learning with Brain-Inspired Modulation can Improve Adaptation to Environmental Changes." arXiv:2205.09729. [Full author list not pulled this pass — confirm before citing.]

**TODO — not verified, do not cite as-is:**
- Chen, X. et al. "Policy Inertia Controller" — AAAI 2021 (action oscillation)
- Sunk cost fallacy ICML 2025 paper
- Procrastination TD-learning papers
- Learned helplessness computational psychiatry citation
- Confirmation/positivity bias RL citation
- "Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery"
- Bontrager, P. et al., superstition in deep RL (AIIDE 2019) — high prior confidence, formal re-verification still recommended
- Daw, Dayan, Dolan et al. habit/goal arbitration — high prior confidence (foundational, well-known), formal re-verification still recommended
- Schwartz, B. "The Paradox of Choice" — original book/concept citation
- Choice Type Impacts Human Reinforcement Learning, *Journal of Cognitive Neuroscience*, 2023

---

## 13. How to use this document
Same as v1 — see `rl-psychological-biases-paper.md` Section 13. This v2 file supersedes v1 for all content; keep v1 only for change-tracking against `critique-report.md`.

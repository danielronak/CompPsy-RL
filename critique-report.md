# Critique Report: Computational Analogues of Post-Decision and Self-Evaluative Biases in Deep RL Agents

**Reviewer:** Claude (Sonnet 5), self-review of the project brief I originally drafted
**Method:** Live web search verification of citations + independent novelty search + methodological review
**Honesty note up front:** I verified the five most load-bearing citations directly (full text/abstract pulled and checked against my claims). Several secondary citations in the original brief were never given exact author/title/venue in the first place — I flag those explicitly below rather than pretend they're checked.

---

## CRITICAL

### C1. The choice-overload experiment's novelty claim is weaker than the brief stated
I found two prior papers closer to the proposed decision-time framing than the one already disclosed in the brief (Nica et al.'s hierarchical-RL paper):

- **"Reinforcement Learning with Brain-Inspired Modulation can Improve Adaptation to Environmental Changes"** (arXiv:2205.09729) explicitly implements a bandit-arm "paradox of choice" experiment using Schwartz's own framing, and shows that a modulation rule creates a bias that depresses perceived value of a previously high-reward arm, helping with few arms but *hurting* with many arms — i.e., decision-quality degradation as choice-set size grows. This is much closer to what Section 4.4 proposes than the hierarchical-RL paper the brief already cites.
- **Dubey, Griffiths & Dayan, "The pursuit of happiness"** (PLOS Computational Biology, 2022) — the same paper cited elsewhere in the brief for the placebo extension — explicitly reports that their comparison-based reward agents "perform sub-optimally... when there are too many similar options." That is a decision-time, too-many-options finding, in a paper the brief already leans on for something else.

**Verdict:** the "training-time vs. decision-time" differentiation the brief uses to defend Section 4.4 against the Nica et al. paper does **not** fully hold up against these two — both show decision-time degradation with more options, in agents that are already trained/adapting, not just during hierarchical option-learning. This experiment needs either a sharper differentiator (see revised brief) or should be treated as the weakest of the four core experiments and potentially demoted to optional/stretch status.

### C2. Several "citations" in the original brief were never fully specified
On re-inspection, entries like "ICML 2025 paper on deep RL agents completing wasteful episodes" (sunk cost), "TD-learning / compressed state-representation papers" (procrastination), and "computational psychiatry" (learned helplessness) were written generically in the original brief, without exact author names, titles, or verifiable venues. I did not re-derive these in this pass. **These cannot be cited in an actual paper as written** — they need to be tracked down to exact references or removed from the Related Work table until they are.

---

## MAJOR

### M1. The impostor-syndrome operational definition doesn't precisely match its cited mechanism
I pulled the actual abstract of Katyal, Huys, Dolan & Fleming (*Nature Communications*, 2025). The real mechanism is that **global confidence shows *reduced sensitivity* to local confidence/performance signals** — a damping/discounting effect — not simply "negative feedback weighted more heavily than positive feedback," which is what the original brief's operational definition proposed. These are related but not identical mechanisms, and a reviewer familiar with the source paper would notice the mismatch. The revised brief below corrects this.

### M2. The cognitive dissonance experiment has an unaddressed confound that mirrors a live controversy in the human literature
Chen & Risen (2010) showed that the classic Brehm free-choice paradigm can produce apparent "spreading of alternatives" as a **statistical artifact** — the act of choosing between "near-tied" options reveals a real, pre-existing (if slight) preference, and the second rating just restates that preference, rather than reflecting genuine dissonance-driven attitude change. The RL experiment in Section 4.1 has a direct structural analog: if the two bandit arms are only *empirically* near-tied (based on a noisy value estimate), the agent's post-commitment value divergence could just reflect that the tie was never real. **The original design has no control for this.** Fix: use arms with a verified identical ground-truth reward distribution (not just empirically close estimates) as the primary test condition, with empirically-near-tied arms as a secondary, weaker condition.

### M3. Multiple-comparisons correction was never specified
The original brief's methodology section mentions seeds, effect sizes, and confidence intervals, but never specifies a correction method for running four to six directional hypothesis tests in one paper. This needs to be explicit (e.g., Holm-Bonferroni across the core experiment set, or a pre-registration plan) or a reviewer will flag it.

### M4. Domain-fit is not as binary as the original brief implied
My research found that computational psychiatry does have a genuine **theory-driven/mechanistic sub-tradition** that doesn't strictly require new patient data (e.g., dynamical-systems models of symptom trajectories). So "you must add human data or retarget venues" was an oversimplification. But — even that mechanistic sub-tradition almost always frames itself explicitly around a **specific clinical construct or symptom dimension**, not a general "agent behavior resembles a named psychological phenomenon" framing. The realistic middle path: keep the agent-only design, but explicitly benchmark each experiment's *pattern* of results against a specific published human finding (e.g., the exact Katyal et al. underconfidence-gap curve) rather than just citing the human literature for inspiration. That's a real, achievable middle ground the original brief didn't articulate.

---

## MINOR

- The "pursuit of happiness" paper was cited as appearing in **"PubMed, 2022"** in the original brief's reference list — PubMed is an index, not a journal. Correct venue: *PLOS Computational Biology*, 2022 (Dubey, Griffiths, Dayan).
- The Nica et al. paper's exact venue is a NeurIPS 2022 workshop ("All Things Attention"), not a main-track publication — worth being precise about this in framing, since workshop papers carry different weight than main-track papers when arguing prior art either closes or doesn't close a gap.
- "Meta-Cognitive Reinforcement Learning with Self-Doubt and Recovery" (cited as arXiv 2026) and the Bontrager et al. superstition paper were not re-verified with a fresh search in this pass — they were verified earlier in our conversation but not re-checked now. Medium confidence, not zero.

---

## Overall confidence notes

I directly verified, with fresh searches in this pass: Brehm (1956), Nikishin et al. (ICML 2022), Katyal et al. (*Nature Communications*, 2025), Nica et al. (arXiv 2201.09653), and Dubey et al. (PLOS Comp Bio, 2022) — all five checked out as real, correctly attributed (with the one venue correction above), and I pulled actual mechanism details rather than trusting my own paraphrase. I did **not** re-verify the sunk-cost, procrastination, learned-helplessness, confirmation-bias, self-doubt, or superstition citations in this specific pass — those are carried over from earlier in our conversation at lower confidence and should not be treated as freshly checked.

---

## What changed in the revised brief (v2)

1. Added the two newly-found competing papers to Related Work (C1) and rewrote Section 4.4's differentiation argument to be honest about the narrower remaining gap.
2. Corrected the impostor-syndrome mechanism in Section 4.2 to match the actual "reduced sensitivity" finding rather than the simplified "asymmetric weighting" description (M1).
3. Added an explicit revealed-preference confound control to Section 4.1's experimental design (M2).
4. Added a named multiple-comparisons correction method to the shared methodology section (M3).
5. Rewrote the domain-fit section to reflect the more nuanced middle path — benchmark against specific published human findings rather than a binary "add human data or don't" framing (M4).
6. Corrected the "pursuit of happiness" citation venue (Minor).
7. Flagged the underspecified citations explicitly as TODO items rather than presenting them as settled (C2).

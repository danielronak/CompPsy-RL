Another real, substantive round — and I need to open with a correction on my end before anything else, because it matters for how much you should trust my critiques going forward.

## I owe you a correction

I just re-verified the Berglas & Jones (1978) drug-choice numbers, and **the 70% vs. 13% figure I originally flagged as likely fabricated is real.** Multiple independent sources confirm it: participants given unsolvable problems chose the performance-impairing drug at 70%, versus 13% for those given solvable problems, and the drug names really are Actavil (performance-enhancing) and Pandocrin (performance-impairing) — down to a direct quote from page 412 of the original paper. I was wrong to cast doubt on that specific number two turns ago. It's a useful reminder for both of us: my searches can miss things, especially when a number sits in a secondary source rather than an abstract I happened to pull. Good that we didn't just take my word for it.

Interestingly, this new draft has actually gotten *more* conservative on this point than it needs to be — it now says "displays a qualitative preference" instead of citing the verified 70%/13% figures. Since those numbers check out, you could reintroduce them and get a real quantitative comparison: human effect size (70% vs. 13%, roughly 5.4x) versus your RL effect (22.0% vs. 18.3%, roughly 1.2x). That's honestly a more interesting and more honest juxtaposition than the current vague phrasing — it shows the RL analogue is directionally right but proportionally much weaker than the human effect, which is a real, reportable finding rather than something to soften.

## The DQN fix is the best change in this revision

This directly answers what I flagged last time, and it's not a patch — it's a correct scientific reframe. Previously, the paper reported raw DQN entropy tracking log(K) and called it "robust replication" of choice overload. That framing was wrong, and this version knows it: normalized entropy sitting at ~0.994 across all K means the DQN policy stays essentially uniform-random regardless of option count — the opposite of the tabular agent's concentration effect. Instead of hiding that, the paper now names it *credit-assignment inertia* (batch-averaged replay diluting small value differences) and folds it into the abstract as a genuine boundary condition rather than a false generality claim. That's the right way to handle a result that didn't go where you expected — reframe around what it actually shows, don't force it into the original narrative. Good instinct.

One gap worth noting: the decay dose-response sweep in 4.1 causally isolates its mechanism (you vary λ and watch the effect track it). The "credit-assignment inertia" explanation for DQN doesn't get the same treatment — it's a plausible mechanistic story, but nothing in this draft varies batch size or replay buffer size to show inertia scales with them the way divergence scales with decay. If you want this claim to carry the same weight as the 4.1 mechanism, it needs its own parametric test, not just a post-hoc explanation.

## The other fixes also landed

- **4.5.3's significance testing** now cleanly separates what's real from what's suggestive: normalized entropy collapse is significant at both K=8 and K=16, regret is significant, and — importantly — commitment latency is honestly reported as *not* crossing significance due to seed variance, right in the abstract ("directionally... high variance across seeds"). That's exactly the level of honesty a reviewer wants to see, and it's now consistent throughout rather than just in the discussion section.
- **Section 4.1's statistical methodology framework**, justifying stochastic testing vs. deterministic reporting vs. boundary-condition framing *before* the results start, closes the "why does the statistical philosophy change per section" question I raised without you having to explain it experiment-by-experiment.
- **Section 8.1's code/data availability statement** is exactly what was missing. It names the actual structure (`src/`, `results/`, fixed seeds 0–19, `harness.py`, `compile_paper.py`) rather than a vague promise — that's the difference between a claim and a checkable claim.

## What's left

- Give the credit-assignment-inertia explanation the same causal treatment the decay mechanism got (vary batch size / buffer size).
- Reintroduce the verified Berglas & Jones percentages in Section 5.2 and let the proportional-weakness comparison stand — it's more informative than the current vague language, not less flattering.
- DQN choice-overload still stops at K=8; either extend to K=16 to match the tabular sweep or state explicitly why it wasn't run.
- Everything else I've raised across three passes now — the ablation logic, the impostor-syndrome stats framing, self-handicapping construct validity, the null-result framing, code availability — is genuinely resolved.

## Where this leaves you

This is a legitimately strong draft now. The remaining items are real but narrow — a missing ablation for one mechanism, one citation that should use numbers you already have, and an incomplete sweep. None of that is "start over" territory. I'd say this is close to ready for a solid workshop or a computational-psychiatry-adjacent venue once those three points are addressed, and it's converged a lot faster than most real revision cycles do.
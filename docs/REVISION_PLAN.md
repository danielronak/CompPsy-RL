# Revision plan: MoSICom 2026 paper 258 ("accepted with revisions")

This file is the hand-over for the reviewer revisions. It is written so that
anyone (or any model) can pick the work up part-way. Tick a box when a step is
done and committed.

## Where things are

- Paper source: `latex/paper.tex`, bibliography `latex/references.bib`.
- Work on branch `claude/practical-ritchie-nmivc5` (it is ahead of `main`;
  `main` still has the old author block and the Fig. 1 overlap).
- Word conversion scripts: `tools/docx_build/` (see "After the edits").
- Reviews: three reviewers (#1, #3, #4), all Good/Strong. No new experiments
  were requested.

## Rules for every edit

1. The paper must stay at **6 pages or fewer**. Before the revisions, page 6
   had about 20 free lines at the bottom of the right column. If a build goes
   to 7 pages, use the trims listed under "If it runs over 6 pages".
2. Do not change any reported number unless this plan says so. Every number
   was checked against `results/**/*.json`.
3. Keep the author's wording where it is not wrong. Reviewer 4 asked for
   proofreading, not restyling.
4. Build and check after each revision:

   ```bash
   cd latex
   pdflatex -interaction=nonstopmode paper.tex >/dev/null; bibtex paper >/dev/null
   pdflatex -interaction=nonstopmode paper.tex >/dev/null
   pdflatex -interaction=nonstopmode paper.tex | grep "Output written"   # must say 6 pages
   grep -c Overfull paper.log                                            # must be 0
   grep -ciE "undefined (citation|reference)" paper.log                  # must be 0
   pdffonts paper.pdf | awk 'NR>2 && $4=="no"' | wc -l                   # must be 0
   ```

5. Commit and push after each revision so progress is never lost.

## Facts checked in the code (use these, do not guess)

| Fact | Value | Source |
|---|---|---|
| Seeds per condition | 20 | all configs |
| "±" in the paper | sample SD across seeds (ddof = 1) | all experiment files |
| Exp 1 test | paired t-test, committed vs. observer, t(19) = 9.90, p = 6.2e-9; d = mean/SD of the paired difference | recomputed from `results/exp_4_1_cognitive_dissonance/results.json` |
| Exp 4, ChoiceRoom, DQN buffer tests | Student independent-samples t-test, df = 38 | `src/analysis/statistics.py`, `exp_dqn_replication.py` |
| Reported p-values | uncorrected (raw) | recomputed |
| Tabular learning rate | α = 0.1 in Exps 1–4 | `src/harness.py` |
| Exp 1 exploration | ε-greedy, ε = 0.1, constant | `DissonanceConfig` |
| **Exp 4 exploration** | **ε₀ = 0.1** (×0.999 per step, floor 0.02). The paper says ε₀ = 1.0, which is wrong. | `ChoiceOverloadConfig` (no override of the 0.1 default) |
| Exp 2 settings | 7×7 gridworld, 500 episodes, α = 0.1, γ = 0.99, ε = 0.1, α_global = 0.02 | results JSON config |
| Exp 3 settings | 800 trials, success probability 0.75, α_global = 0.05, reference β = 3, logistic temperature 0.5 | results JSON config |
| ChoiceRoom environment | 9×9 room, latent colour values 1.0 ± 0.05 | `src/environments/choice_room.py` |
| Exp 4 Holm-adjusted p (4 contrasts, K=16 vs K=2) | entropy 1.7e-21; deferral 3.4e-7; **regret 0.063**; latency 0.222 | recomputed |
| Latency p | 0.2225, so it rounds to **0.222** (paper says 0.223) | recomputed |
| Staleness r = 0.82 | pooled over 80 runs across the four K values | `shared_mechanism.py` |

Two of these are corrections to the paper, not only wording:

- **Exp 4 ε₀**: the paper states 1.0; the code that produced the results uses 0.1.
- **Exp 4 regret**: the paper says regret "increases significantly" (p = 0.031)
  while also saying Holm-Bonferroni correction is applied. After that
  correction p = 0.063, so it must be reported as a trend.

## The six revisions

Each item gives the text to find and what it becomes. "→" means "replace with".

### [x] Revision 1: moderate causal claims (Reviewers 1 and 3)

1. Abstract. "that scales monotonically with memory decay, ruling out the
   revealed-preference confound" → "that scales monotonically with memory
   decay and, by construction, cannot reflect revealed pre-existing preferences".
2. Abstract. "with a parametric replay-buffer ablation confirming this causal
   link" → "and a replay-buffer ablation shows that buffer size controls how
   well near-tied options are differentiated". (The ablation was run on
   choice overload, not on dissonance, so the old sentence was also inaccurate.)
3. Abstract, last sentence. → "These findings offer a candidate mechanistic
   account of how behavioral signatures resembling human cognitive
   distortions can emerge from the statistical mechanics of reinforcement
   learning in small agents, and identify boundary conditions on when such
   biases do and do not appear."
4. Introduction, contributions list:
   - "directly ruling out the \citet{chen2010choice} revealed-preference
     confound using ground-truth-identical alternatives" → "using
     ground-truth-identical alternatives so that the \citet{chen2010choice}
     revealed-preference confound cannot apply"
   - "(3) computational validation of the" → "(3) a computational
     implementation of the"
   - "causally drives objectively unjustified self-handicapping" → "is
     sufficient, in our model, to produce objectively unjustified
     self-handicapping"
   - "(6) a unified action-visitation-staleness mechanism underlying both
     Free biases" → "(6) action-visitation staleness as a candidate shared
     mechanism for both Free biases"
   - "(7) DQN replication isolating memory decay as the dominant amplifier"
     → "(7) a DQN replication indicating that memory decay is the main amplifier"
5. Exp 1 results:
   - "is directly proportional to the rate of memory decay" → "increases
     monotonically with the rate of memory decay" (three points; not proportional)
   - "showing that commitment alone is insufficient and that decay is the
     causal driver of divergence" → "showing that commitment alone is
     insufficient: in this model, removing decay removes the effect"
   - "confirming that biological commitment only initiates asymmetric future
     visitation" → "consistent with commitment acting by initiating
     asymmetric future visitation"
6. Table I header "Causal Role" → "Role". Fig. 1 caption "Zero-decay confirms
   that commitment alone is insufficient." → "With zero decay, commitment
   alone produces no reliable divergence."
7. Staleness subsection:
   - "Both Cognitive Dissonance and Choice Overload share a single
     mathematical root." → "We propose that Cognitive Dissonance and Choice
     Overload share a common root."
   - "rejected-arm staleness predicts excess value divergence" →
     "rejected-arm staleness correlates with excess value divergence";
     "mean staleness predicts policy entropy" → "mean staleness correlates
     with policy entropy"
   - "across 80 independent runs" → "across 80 runs pooled over the four
     values of $K$, so it partly reflects between-condition differences"
   - Add at the end of the paragraph: "These associations are correlational.
     The interventional evidence for the mechanism is the decay sweep of
     Exp 1, where removing decay removes the divergence; a direct
     manipulation of staleness itself is left to future work."
   - Fig. 3 caption: "predicts" → "correlates with" (twice); "Both Free
     biases share a common computational root." → "Both patterns are
     consistent with a shared computational root."
8. DQN subsection:
   - Title "Deep Q-Networks: Memory Substrate as Causal Driver" → "Deep
     Q-Networks: The Role of the Memory Substrate"
   - "confirmed this causally: normalized entropy scales monotonically" →
     "supports this account: normalized entropy increases monotonically with
     buffer size"
   - "This isolates biological memory decay as the dominant amplifier of
     human-like post-decision spreading." → "This points to memory decay as
     the main amplifier of post-decision spreading in these agents, although
     the tabular and deep agents also differ in function approximation, so
     the comparison is suggestive."
   - Fig. 5 caption: "causally isolating replay mixing as the driver of
     credit-assignment inertia" → "consistent with replay mixing driving
     credit-assignment inertia"
9. Discussion, first sentence: "These findings establish that cognitive
   dissonance and choice overload emerge" → "These findings indicate that
   signatures of cognitive dissonance and choice overload can emerge";
   "memory decay is the dominant amplifier" → "memory decay appears to be the
   main amplifier".

### [x] Revision 2: human comparisons are qualitative (Reviewers 1 and 3)

1. Abstract: "Agents using the human-verified asymmetric confidence-update
   rule" → "Agents given the asymmetric confidence-update rule reported in
   human studies"; "reproducing the human commitment collapse" →
   "qualitatively resembling the human commitment collapse".
2. Introduction: "structurally identical behavioral and representational
   trajectories" → "structurally analogous behavioral and representational
   trajectories". Background: "Midbrain dopamine mirrors TD prediction
   errors" → "Midbrain dopamine activity resembles TD prediction errors".
3. Exp 3: "a $4.77\times$ shift that closely tracks the human $5.4\times$
   shift toward performance-debilitating drugs ($70\%$ vs.\ $13\%$;
   \citet{berglas1978drug})" → "a $4.77\times$ shift, of similar order to
   the human $5.4\times$ shift toward performance-debilitating drugs ($70\%$
   vs.\ $13\%$; \citet{berglas1978drug}), although the agent's ratio varies
   with the penalty (Table~\ref{tab:exp43}), so the parallel is qualitative".
4. Exp 4: "the opt-out condition reproduces the human commitment collapse" →
   "the opt-out condition shows a decline in commitment in the same direction
   as the human one"; "mirroring the $30\%\to3\%$ purchase collapse" →
   "qualitatively paralleling the larger $30\%\to3\%$ purchase collapse".
5. Human Benchmark subsection, opening: "The agents' behavior aligns with
   published human data across all four experiments (Fig. 4):" → "The agents'
   behavior parallels published human data in direction across all four
   experiments (Fig. 4). These comparisons are qualitative: units, tasks, and
   populations differ, and several agent magnitudes depend on free
   parameters, so numerical proximity is not quantitative equivalence."
   Then: "reproduces the underconfidence pattern" → "has the same sign as the
   underconfidence pattern"; "mirrors the commitment collapse" → "parallels
   the commitment collapse"; "closely tracks \citet{berglas1978drug}" → "is of
   similar order to that of \citet{berglas1978drug}".
6. Fig. 4 caption: "closely tracking the human $5.4\times$ shift" → "of
   similar order to the human $5.4\times$ shift"; add "Comparisons are
   qualitative."

### [x] Revision 3: moderate generalization to larger AI systems (Reviewer 1)

Replace the whole "For AI alignment, ..." paragraph of the Discussion with:

> For AI alignment, these results suggest hypotheses, not established facts
> about larger systems: post-decision revaluation may arise in agents that
> combine asymmetric visitation with forgetting, and need not indicate
> deliberate misalignment; choice-overload-like degradation may appear in
> tool-use agents with large near-tied action sets; and asymmetric
> self-critique loops could decouple self-assessed competence from objective
> performance. Whether these effects hold at scale remains to be tested: our
> agents are small, and the deep-RL results were mixed, with dissonance
> largely absent in DQN. Within that scope, the results suggest that some
> human-like biases may reflect general trade-offs in learning from one's own
> choices under uncertainty.

(The old last sentence also claimed the biases were "seen across tabular and
deep architectures", which contradicts the DQN dissonance result.)

### [x] Revision 4: separate emergent from designed behaviour (Reviewer 3)

1. Abstract: "three out of four biases emerge robustly while the fourth marks
   an informative computational boundary" → "two biases (cognitive
   dissonance and choice overload) emerge from standard learning dynamics,
   while the other two appear only when a metacognitive self-model with an
   explicitly asymmetric update is built in".
2. Background, Architectural paragraph. Replace the sentence that starts "We
   also include an asymmetric update rule because" with: "A standard agent
   keeps its value estimates coherent with its behavior: it cannot act
   competently while representing that competence as poor. Impostor-like
   behavior therefore does not emerge on its own. We program an asymmetric
   evidence-filtering rule explicitly, so Exps 2 and 3 test the consequences
   of a designed mechanism, not its emergence."
3. Exp 2 results, after "under asymmetric weighting.": add "The gap is
   therefore a direct consequence of the programmed rule, not an emergent effect."
4. Discussion, first sentence: "while impostor syndrome requires asymmetric
   metacognitive evidence integration" → "while the impostor-like calibration
   gap and self-handicapping appear only with an explicitly programmed
   asymmetric self-model".

### [x] Revision 5: clearer statistics and implementation detail (Reviewers 1 and 3)

1. Replace the statistics paragraph at the start of "Empirical
   Investigations" with:

   > All experiments use $N=20$ independent seeds per condition; values are
   > mean $\pm$ sample standard deviation across seeds, and $d$ is Cohen's
   > $d$. Tabular agents use learning rate $\alpha=0.1$. For the stochastic
   > experiments (Exps 1, 4, and the deep-RL replications) we report
   > two-tailed $t$-tests: a paired test of committed against observer agents
   > in Exp 1 ($df=19$) and independent-samples tests between conditions
   > elsewhere ($df=38$). Reported $p$-values are uncorrected; we apply
   > Holm-Bonferroni correction across the contrasts reported within each
   > experiment and state where a result does not survive it. For the
   > deterministic self-model experiments (Exp 2, and the self-handicapping
   > mechanism of Exp 3), null-hypothesis tests would mistake a mechanical
   > certainty for statistical significance, so we report exact magnitudes,
   > dose-response curves, and 95\% CIs. Non-significant findings are
   > reported explicitly as boundary conditions.

2. Exp 1: "After $T_1=500$ steps of exploration" → "After $T_1=500$ steps of
   $\epsilon$-greedy exploration ($\epsilon=0.1$)". "($t=9.90$, $p<10^{-6}$,
   $d=2.21$)" → "(paired $t(19)=9.90$, $p<10^{-6}$, $d=2.21$)".
3. Exp 2: after "in a multi-task gridworld" add "($7\times7$, 500 episodes;
   $\gamma=0.99$, $\epsilon=0.1$, $\alpha_{\text{global}}=0.02$)".
4. Exp 3: after the sentence ending "against the objective penalty." add
   "Each run has 800 trials with true success probability $0.75$; the choice
   is logistic in this net value (esteem weight $\beta=3$, temperature $0.5$)."
5. Exp 4: **"$\epsilon_0=1.0$" → "$\epsilon_0=0.1$"** (factual correction).
6. Exp 4 regret (**correction**): "---policies concentrate unpredictably on a
   subset of options rather than becoming uniformly indecisive and
   consequently mean regret also increases significantly ($K=16$ vs.\ $K=2$:
   $t=2.24$, $p=0.031$)." → "---policies concentrate unpredictably on a
   subset of options rather than becoming uniformly indecisive. Mean regret
   also rises ($t=2.24$, uncorrected $p=0.031$), but this does not survive
   Holm-Bonferroni correction across the four Exp 4 contrasts (adjusted
   $p=0.063$), so we treat it as a trend."
7. Exp 4: "$t=-20.45$, $p=4.19\times10^{-22}$" → "$t(38)=-20.45$,
   $p=4.19\times10^{-22}$"; latency "$p=0.223$" → "$p=0.222$".
8. ChoiceRoom: after "solved by a convolutional DQN" add "($9\times9$ room;
   latent values $1.0\pm0.05$)".
9. Reproducibility: after "via \texttt{python run\_all.py}." add "All
   hyperparameters are set in the configuration classes of the repository."

### [x] Revision 6: proofread (Reviewer 4)

1. Abstract: "have always been known to be closely related" → "have long been
   known to be closely related"; "have not gotten as much attention" → "have
   received less attention"; "we investigate if four" → "we investigate
   whether four".
2. Exp 1: "A Tabular Q-learning agent" → "A tabular Q-learning agent"; "and
   high-decay yielded" → "and high decay yielded"; "When the Agent was not
   forced to commit, the divergence was observed to be ($+0.263\pm0.792$)" →
   "When the agent was not forced to commit, the divergence was $+0.263\pm0.792$".
3. Fig. 1 caption: "emerges endogenously($+0.263$)" → "emerges endogenously
   ($+0.263$)".
4. ChoiceRoom paragraph uses British spelling while the rest is American:
   "coloured" → "colored", "colour" → "color" (all forms), "generalise" →
   "generalize".
5. Reproducibility: "the sole responsibility of the author" → "the sole
   responsibility of the authors" (there are two authors now).
6. `references.bib`: protect capitals that the style lowercases:
   `{``Superstition''} in the Network` (reference [2]) and `{B}ayesian`
   (reference [10]).
7. Final pass: search the built PDF text for doubled words, a space before
   punctuation, and "??"; run a spell-check if `aspell` is available.

## If it runs over 6 pages

Already applied during Revision 1 (the build went to 7 pages): the reference
list was changed from `\small` with `\bibsep` 4pt to `\footnotesize` with
`\bibsep` 2pt, which is the standard IEEE 8pt reference size.

Also applied later to stay at 6 pages: Tables I, II and IV set in
`\footnotesize` like Table III (Revision 4); Fig. 5 panels balanced to equal
height; a sentence duplicated from Exp 3 cut from the benchmark section and
the Fig. 4 caption; Fig. 4 narrowed to `0.50\textwidth` (Revision 5). The
Fig. 5 source now sits above its subsection so the figure lands on page 5.
The finished paper has about 4 free lines on page 6.

If a later edit overflows, the remaining options, in order (item 1 is
already done):

1. Fig. 4: `width=0.54\textwidth` → `0.50\textwidth`.
2. Remove "Non-significant findings are reported explicitly as boundary
   conditions." from the statistics paragraph.
3. Remove the Reproducibility sentence added in Revision 5.9.
4. Fig. 3 and Fig. 2: `width=\linewidth` → `0.92\linewidth`.

## After the edits

- [x] Rebuild the PDF and run all four checks above.
- [x] Regenerate the Word file:

  ```bash
  cd tools/docx_build
  python3 prep.py ../../latex paper_pandoc.tex
  pandoc -o ref_default.docx --print-default-data-file reference.docx
  python3 refdoc.py ref_default.docx ref_ieee.docx
  pandoc paper_pandoc.tex -f latex -t docx --reference-doc=ref_ieee.docx \
         --resource-path=../../latex -o paper_raw.docx
  python3 post.py paper_raw.docx paper_ieee.docx
  ```

  `prep.py` reads `paper.aux` and `paper.bbl`, so build the PDF first. It
  stops with an error if a table cell contains math it cannot convert.
  Needs `pandoc` and `python-docx`.
- [x] Send the author the new PDF and Word file.
- [ ] The author must run the new PDF through IEEE PDF eXpress again (the
  approved copy has the old text) and upload the newly approved file.
- [ ] If CMT asks for a response to reviewers, write it from this file: each
  revision above names the reviewer and the exact change.

## Not in scope

Reviewer 3 asked for "stronger causal validation" of the staleness mechanism.
This plan answers by stating that the staleness evidence is correlational
(Revision 1.7). A new intervention experiment (forcing occasional visits to
the rejected arm and checking that divergence shrinks) would be a stronger
answer, but the author has not asked for it.

## Staleness intervention (run after the six revisions; NOT yet in the paper)

The author asked to try the experiment listed under "Not in scope". It is in
`src/experiments/exp_staleness_intervention.py`; results are in
`results/exp_staleness_intervention/results.json`. Run it from the repo root
with `python -m src.experiments.exp_staleness_intervention` (about a minute).

Design: Exp 1's primary condition unchanged (tied arms, lambda = 0.001, same
20 seeds), plus a staleness cap: after step 500, any arm that has gone `cap`
steps without an update is force-visited. No cap reproduces the paper's
+0.863 at 100 steps exactly.

Result in the paper's design, excess divergence at 100 steps (mean, SD):

| Cap (steps) | Mean staleness of rejected arm | Excess divergence | Paired t(19) vs. no cap |
|---|---|---|---|
| none | 15.8 | +0.863 (0.390) | baseline |
| 50 | 12.8 | +0.839 (0.360) | t = -1.08, p = 0.29 |
| 20 | 4.4 | +0.374 (0.641) | t = -4.27, p = 4.1e-4 |
| 10 | 2.3 | +0.185 (0.542) | t = -6.21, p = 5.8e-6 |
| 5 | 1.0 | -0.108 (0.518) | t = -7.15, p = 8.5e-7 |
| 2 | 0.5 | -0.133 (0.237) | t = -12.32, p = 1.7e-10 |

Divergence falls as staleness is capped, in all 20 seeds (mean within-seed
Spearman rho = 0.86), with the decay rate held fixed.

### Finding about Exp 1 that the author must decide on

The paper says that after 500 steps "the agent commits to the most-pulled
arm for 1000 steps". In the code (`src/environments/bandit.py` and
`_run_bandit_session` in `exp_dissonance.py`) the environment pays out from
the committed arm, but the agent keeps choosing epsilon-greedily and each
reward is credited to the arm the agent chose. After "commitment" the
rejected arm's estimate still receives 231 of 1000 updates on average (range
35 to 881 across seeds). With identical arms this is the same process as an
agent that never commits. The paper's numbers are what this code produces,
but the description of commitment does not match it.

If commitment is made to bind the agent (the `lock_in` mode of the new
script), Exp 1 gives larger and less noisy effects, but the zero-decay
control is no longer near zero:

| lambda | Excess at 100 steps | Excess at end |
|---|---|---|
| 0 | +0.339 (0.336) | +0.269 (0.439) |
| 0.001 | +1.326 (0.364) | +3.441 (0.383) |
| 0.01 | +4.342 (0.338) | +4.939 (0.378) |

Nothing in the paper has been changed for either point. Options are in the
chat reply of the session that added this section.

### Follow-up measurement: does the gap grow after the decision?

Chosen-minus-rejected gap in the committed agent (tied arms, lambda = 0.001,
20 seeds), measured at the decision (step 500), 100 steps later, and at the end:

| Version | At decision | +100 steps | End | Change by +100 |
|---|---|---|---|---|
| Paper's code (agent not bound) | +0.841 | +0.923 | +0.324 | +0.082, p = 0.56 |
| Binding commitment | +0.841 | +1.387 | +3.438 | +0.546, p = 1.8e-6 |

In the paper's code the gap is already present at the decision and does not
grow afterwards. The reported +0.863 is the gap the committed agent keeps
while the observer, which samples both arms equally, loses it. So the current
Exp 1 supports "self-determined sampling plus decay maintains a value gap
between identical options", but not "the gap spreads after the decision".
With binding commitment the gap does spread after the decision.

## Path 1 (chosen by the author): re-run Exp 1 with binding commitment

Goal: make the code do what the paper says ("the agent commits to the
most-pulled arm"), re-run everything that depends on it, and update the paper.
Tick each step when done and committed.

### Code

- [ ] `src/experiments/exp_dissonance.py`, `_run_bandit_session`: credit each
  reward to the arm that was actually pulled (`info["effective_action"]`), and
  log that arm in `step_log`. The environment already forces the committed
  arm after step 500, so this makes commitment bind the agent.
- [ ] `src/analysis/shared_mechanism.py`: the Exp 4.1 staleness simulation has
  the same crediting line; fix it the same way.
- [ ] `src/experiments/exp_staleness_intervention.py`: drop the old "free"
  mode; keep the cap sweep under binding commitment. With no cap it must
  reproduce the new primary numbers.
- The DQN dissonance run (`exp_dqn_replication.py`) already binds the agent
  and is **not** re-run. Exps 2, 3, 4 and ChoiceRoom do not use commitment.

### Re-run (needs numpy, scipy, matplotlib, seaborn, pandas, tqdm; no torch)

- [ ] `python run_all.py --exp 4.1`
- [ ] `python run_all.py --exp cross` (staleness analysis and benchmark figure)
- [ ] `python -m src.experiments.exp_staleness_intervention`
- [ ] Copy regenerated figures into `latex/figures/`:
  `dose_response_ablation.png` (Fig. 1), `shared_mechanism_staleness.png`
  (Fig. 3), `human_benchmarks_4panel.png` (Fig. 4).

### Expected numbers (from the exploratory run; replace with the re-run values)

Primary: about +1.33 at 100 steps and +3.44 at the end. Zero decay: about
+0.34 and +0.27, so it is no longer "near zero". High decay: about +4.34 and
+4.94. Endogenous and observer rows should not change.

### Paper edits that follow

- [ ] Exp 1 results paragraph, Table I, Fig. 1 caption: all numbers.
- [ ] Zero-decay wording: "commitment alone is insufficient" becomes a small
  selection effect on noisy estimates (the Chen and Risen artifact) that decay
  amplifies. Abstract and contribution (2) must not say the confound is
  excluded outright.
- [ ] Staleness section: new per-seed r; add the intervention result in place
  of "left to future work"; note the Exp 4.4 staleness simulation uses
  epsilon_0 = 0.2 (it differs from Exp 4's 0.1).
- [ ] Human benchmark sentence and Fig. 4 caption: new Exp 1 value; it no
  longer sits numerically close to Brehm's 0.81, so describe direction only.
- [ ] DQN section, abstract, Fig. 5 caption, Discussion: the tabular
  comparator (was +0.92) and the "about 88%" reduction, recomputed against
  the tabular raw gap at the end of the run.
- [ ] Keep 6 pages; run the four build checks; diff numbers against JSON.
- [ ] Regenerate the Word file; send both files.
- [ ] Tell the author what to say to the chairs (results in Exp 1 changed
  because a code/description mismatch was corrected).

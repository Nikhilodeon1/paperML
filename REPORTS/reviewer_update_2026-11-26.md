# TMLR revision: status update for the reviewer (26 November 2026)

**Purpose.** This reports what changed since your mock review, what is still open, and what we know is wrong in the
current draft. We would like your help ordering the remaining work. Please check the claims against the attached PDF
rather than taking this summary on trust.

**State of the PDF (`paper/main.pdf`).** It is a working draft and is not ready to submit. It has 39 pages, with the
main text, broader impact statement, and AI-use statement ending on page 12. Some table cells print `[pending]`. It
builds with no undefined references or citations. The prose linter reports 0 errors and 28 warnings. The last full test
run had 669 passes, 6 skips, and 4 failures. The failures appear to be in application and model code that the paper
does not use, but we did not check whether they predate this work, and we have not rerun the tests since the latest
edits.

**About the prose.** The abstract, introduction, related work, discussion, conclusion, and Section 5.4 were redrafted
with an AI coding assistant. You asked for these to be rewritten by hand in the authors' voice. That has **not** happened
yet.

---

## 1. What changed

- **Title:** now "What Held-Out Prediction Leaves Unidentified in a Bergman-Type Glucose Digital Twin".
- **Section 5.4 rebuilt** on a converged protocol: L-BFGS-B at every point of a 29-point grid, polished estimates, and
  the corrected real-pole projection.
- **Random-truth trace replicas added** (two seeds), next to the fitted-truth replica.
- **One coverage definition,** with width-aware companions.
- **Dalla Man control split** into model bias and coverage of the pseudo-true value.
- **Front matter rewritten** around the new results, and theory shortened and presented as textbook structure.
- **A table of the ten original hypotheses** is now in the main text (Table 2).
- **Seven references removed,** including the neural-operator, diffusion-tempering, HIPNO, and foundation-model
  citations and all four 2026 arXiv entries you asked us to check. We did not confirm that those four resolve.
- **Five references added** (White 1982, the oral minimal model, carbohydrate counting, SIMEX, data assimilation). We
  checked their metadata only, so they are cited as pointers.

### 1.1 Headline results now

**Area target (iAUC), real CGMacros data, 45 subjects.** These counts still use the old protocol (15-point grid, Adam
inner solver), and Section 5.1 does not say so.

| search box | k_e interior / bounded | k_a interior / bounded | S_I interior / bounded |
|---|---|---|---|
| 1x | 5 / 0 | 6 / 0 | 22 / 3 |
| 2x | 19 / 0 | 11 / 0 | 40 / 24 |
| 4x | 20 / 0 | 14 / 0 | 28 / 24 |

**Held-out area error, random-truth replicas.** Fits with the gut rates pinned to wrong box corners, with S_I refitted,
have 1.02 to 1.03 times the error of the true model. The three-parameter fit has 1.03. Peak time separates them only in
part (fit 1.10; corners 1.12 and 1.34). The trace ratio does not (fit 1.15; corners 1.14 to 1.16).

**Full trace, real data, converged.** The tau1 interval is bounded in 24 of 29 interior CGMacros subjects and the p
interval in 32 of 34. In the standardized-meal cohort the counts are 7 of 9 and 9 of 12. Shanghai is pending.

**Full trace, replicas, converged, 45 subjects.** "Covers" counts all subjects whose 95% interval, clipped to the box,
contains the truth.

| tau1, truths | logging CV | covers | bounded | median signed log error | median log width | rank correlation |
|---|---|---|---|---|---|---|
| random, seed 0 | 0 | 43 | 37 | 0.03 | 0.32 | 0.93 |
| random, seed 0 | 0.25 | 34 | 33 | 0.06 | 0.31 | 0.90 |
| random, seed 0 | 0.75 | 13 | 23 | 0.21 | 0.27 | 0.70 |
| random, seed 1 | 0 | 43 | 35 | 0.02 | 0.29 | 0.94 |
| random, seed 1 | 0.25 | 37 | 33 | 0.04 | 0.27 | 0.93 |
| random, seed 1 | 0.75 | 10 | 25 | 0.24 | 0.23 | 0.74 |
| fitted, seed 0 | 0 | 44 | 6 | 0.02 | 0.13 | 0.79 |
| fitted, seed 0 | 0.25 | 32 | 22 | 0.06 | 0.14 | 0.79 |
| fitted, seed 0 | 0.75 | pending | pending | pending | pending | pending |

For p, coverage on random truths is 42 and 41 with exact inputs, 40 and 41 at CV 0.25, and 22 and 13 at CV 0.75.

**How this answers your concern 3.** You suspected that the effect depended on truths sitting near the lower wall.

- **Boundedness did depend on it.** With fitted truths (median 3% of the range above the lower edge), bounded tau1
  intervals rise from 6 to 22. With random truths they fall from 37 to 23.
- **The loss of coverage did not.** It appears on all three truth sets for which we have results.
- **The likely mechanism for boundedness.** The estimate shifts toward larger tau1 in each case. Near the lower edge
  this closes intervals that were open below (28 open below at CV 0, 12 at CV 0.25). Mid-range it opens more intervals
  above (8 at CV 0, 22 at CV 0.75, seed 0).
- **What this does not do.** We did not run the separate errors-in-variables simulation with a known slope that you
  suggested, and we have not established why the shift goes toward slower absorption.

The old claim that "logging error raises the bounded share" is withdrawn as a general statement.

**Dalla Man generator, Bergman fit.** The pseudo-true tau1 has a median of 98.7 min, against a mean appearance time of
71.6 min for the generator. The median bias is 26.7 min, or 15.5 min if the sensor lag is added to the generator's
value. All 45 pseudo-true fits have k_e = k_a. Of 44 bounded intervals, 26 cover the pseudo-true value and none cover
the generator's value. The median distance from estimate to pseudo-true value is 3.6 min. Rank correlation is 0.95 with
the pseudo-true value and 0.78 with the generator's value. **These intervals use the old 15-point protocol,** so their
coverage is probably understated.

---

## 2. Status against your requests

### Must do

| # | request | status | detail |
|---|---|---|---|
| 1 | Converged solver everywhere; drop Adam and 15-point results | **Partial** | Only the new trace results are converged. **All iAUC profiles are not** (box table, S_I results, S_I replica and synthetic studies, perturbations), and Sections 5.1 and 5.2 do not label them. Neither are the centroid, residual-gut, exact-domain, and Dalla Man intervals; the main text labels those where it uses them. The appendix reports many old-protocol numbers. |
| 2 | Random-truth trace replica | **Done** | Two seeds at CV 0, 0.25, and 0.75. |
| 3 | Dalla Man: pseudo-true, sensor lag and free p2, DM-on-DM | **Partial** | Pseudo-true analysis done (old-protocol intervals). Sensor-lag variant, DM-on-DM, and a second generator distribution not run; no code yet. |
| 4 | One coverage metric, width-aware | **Mostly done** | Defined once, with coverage of narrow bounded intervals. Section 3 also promises an interval score that no table shows. Older coverage labels remain in the appendix. |
| 5 | 2x/4x primary, posterior contraction, no claims from 2 to 6 subjects | **Partial** | All three widths are in one main table, and no fraction is read from fewer than 10 interior subjects. No posterior contraction. |
| 6 | Prediction framing | **Mostly done** | Personal-mean losses on peak and trace, small real timing gain, and the honest floor-table reading are in. Margins were not redefined relative to the floor-to-baseline gap; we state that the margin is 65% of the gain and read the differences. The random-forest and SNPE comparison is gone from the main text but **still in the appendix**. |
| 7 | Errors-in-variables remedy, or matching recommendations | **Done by the second route** | Recommendations limit tau1 to ordering subjects and say no remedy was tested. |
| 8 | Narrow title and claims | **Done** | Please judge whether the claims are narrow enough. |
| 9 | Sensitivity analyses | **Partial** | Perturbations of p1, p2, and n by 30% (iAUC only) are computing; gamma and p3 ran earlier. Not run: sensor lag, per-meal k_e, isolated meals, subject-specific basal glucose. |
| 10 | Original hypotheses in the main text | **Done, with a flaw** | See Section 3, item 3. |

### Should and could do

| # | request | status |
|---|---|---|
| 11 | Reconcile theory with data | **Done** in Section 4 and the Discussion (information limit first, symmetry second). Two of the three supporting runs use the old protocol. |
| 12 | Cut or integrate F.7, F.9, baselines, gamma scaling | **Not done**, except that gamma scaling is shortened in the main text. |
| 13 | Rewrite prose; schematic; coverage-vs-CV figure | **Partial:** abstract cut to about 180 to 200 words and figure added (Figure 3). No schematic, and no rewrite by hand. |
| 14 | Consistency fixes and literature | **Partial:** literature mostly added as pointers; Hovorka-type identifiability is still missing. No full consistency audit of the appendix. |
| 15 | Move register and scripts to the repository | **Not done.** |
| 16 | Shorten Section 4 | **Done.** |

### Other items from your second list

| item | status |
|---|---|
| Regenerate stored log p profiles from before the projection fix | **Partial.** Section 5.4 is recomputed. Old appendix tables and figures have not all been regenerated or removed. |
| Shifted or looser control for the exact-domain result | **Not done.** The main claims no longer rest on that result. |
| Penalty weight in real units | **Done.** The penalty is negligible, and the paper says so. |
| Fisher tolerance check on all subjects | **Done** (45 subjects). |
| Shanghai caveats | **Partial.** Caveats stated; how far the Shanghai evidence depends on them is not quantified. |
| "Empirical audit" as the contribution | **Partial.** The contributions are now empirical, but the introduction does not use that framing explicitly. |
| Figure 3 (now appendix) caption with "either" | **Not done.** |

---

## 3. Problems we know about in the current draft

1. **Appendix.** It still reads as a research log: amendment mentions, hypothesis codes, inline status tags, dense blocks
   of numbers, and the side analyses you asked us to cut. Most linter warnings come from here.
2. **Mixed protocols.** Converged and 15-point results sit in one paper. The headline iAUC results are old-protocol and
   unlabeled, including the abstract's statement that no k_e or k_a interval is bounded. Our earlier numerical checks
   compared trace profiles; we should not assume they carry over to iAUC. The other old results are labeled in the main
   text but not consistently in the appendix.
3. **Exploratory labelling is inconsistent.** The caption of Table 2 says every analysis from Section 5.3 onward is
   exploratory, but H7 to H9 were fixed in advance and concern prediction. The text also never states in words that, of
   five primary endpoints, three were not met, one depends on the formulation, and one was met. The table still uses the
   term "formulation-dependent". "Fixed in version control" cannot be checked under blinding.
4. **Interval score.** Promised in Section 3 and absent from Section 5.4. Stored values for tau1 on random truths are
   0.32, 0.34, and 2.82 (seed 0) and 0.29, 0.32, and 4.74 (seed 1) at CV 0, 0.25, and 0.75.
5. **Leftover vocabulary.** "Polished" is still used in Sections 3 and 5.4.
6. **Unexplained results.** Why logging error shifts estimates toward slower absorption; a held-out ratio of 0.99, below
   the true-model floor, for one appendix control; and an S_I bias of about -0.2 in log units that is also present with
   exact carbohydrate, so logging error does not cause it.
7. **AI-use statement.** A claim that the authors verified every number and reference was removed because we have no
   record of that check. It will return only if the check is done.
8. **Record keeping.** The two newest macro files are not yet in the frozen number index. A separate checker reports
   hand-typed numbers in 22 files, mostly settings and the specification table; we have not reviewed that list.

---

## 4. Results still pending

| result | state | what it could change |
|---|---|---|
| Fitted truths, CV 0.75 | computing | The third row of the fitted-truth sweep. A different direction would weaken the two-direction account. |
| Shanghai real traces and replicas (CV 0 and 0.75) | one archive finished, not yet downloaded or checked | Whether the pattern replicates in a second cohort with noisier inputs. |
| Fitted truths, seeds 1 and 2 (CV 0 and 0.75) | computing | Whether the fitted-truth pattern depends on the seed. |
| Fixed-constant perturbations (iAUC) | computing | The S_I statement in Section 5.2 and the first limitation. Section 5.2 still says this was not run. |

Some interrupted runs left lock files that made later runs skip units while reporting no failures. A gap-filling pass
is running, and the summary script now writes no number for a sweep unless every subject is present.

---

## 5. Proposed roadmap

Effort figures are rough estimates, not measurements.

**Phase A: finish the current draft (about 1 to 2 working days, no new experiments).**
1. Merge the pending archives, check that every sweep is complete, and update Sections 5.2 and 5.4, then the abstract and
   Discussion if the results require it.
2. Fix items 3, 4, and 5 in Section 3 and the figure caption.
3. Clean up the appendix: remove the side analyses and the research-log material, and move the register to the repository.
4. Add the design schematic. Run the full consistency audit, linter, tests, and a clean build.

**Phase B: remove the mixed-protocol problem (compute plus possibly small code changes; not yet checked).**
5. Rerun the iAUC profiles (three box widths, S_I replica) and the Dalla Man intervals with the converged protocol. The
   profile code already has a converged-solver option, so the iAUC part looks like compute only (not yet tested on iAUC).
6. Rerun the centroid and residual-gut profiles the same way, or drop them from the argument.
7. Delete the exact-domain results, or rerun them with your shifted-domain control.

**Phase C: new experiments (new code needed; about several days or more).** Our provisional order:
8. A Bergman fit with sensor lag and free p2, plus a Dalla Man-on-Dalla-Man replica.
9. One errors-in-variables remedy on the replica (SIMEX or latent carbohydrate).
10. A second Dalla Man parameter distribution.
11. Posterior contraction as a box-free check.
12. Isolated-meal restriction and subject-specific basal glucose. We would leave per-meal k_e as a stated limitation.

**Phase D: authors.** Rewrite the interpretive passages by hand, confirm the AI-use statement, check the
pointer-only references, build the anonymized package, and get one outside read.

---

## 6. Questions for you

1. With the results in Section 1.1, what is your recommendation now, and which gaps still block acceptance?
2. If we finish Phases A, B, and D and state Phase C as limitations, is that enough for accept with minor revision? If
   not, which Phase C item is the minimum?
3. For each old-protocol result (iAUC, centroid, residual gut, exact domain, Dalla Man): must it be rerun, may it stay
   with a label, or should it be deleted? We assume the iAUC results must be rerun, since the claims rest on them.
4. Does the comparison of random and fitted truths satisfy your request to test the mechanism, or is the known-slope
   simulation still required?
5. Is the positive result (near-nominal coverage with exact inputs) framed fairly, given that it relies on the same model
   for generation and fitting?
6. Is the recommendation that tau1 is better for ordering subjects than for timing supported, when it rests on
   simulations only and the rank correlation drops to about 0.7 at CV 0.75?
7. For the appendix, what would you cut first, and is there anything on your cut list that you now think should stay?

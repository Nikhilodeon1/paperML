# Full report to the strategy agent: paperML, identifiability of a glucose digital twin (TMLR track)

Date: 2026-10-22. This supersedes `REPORTS/strategy_report_phase9.md`. Detailed numbers: `REPORTS/phase8.md`,
`phase9.md`, `phase10.md`, `phase11.md` (which covers Phases 11 and 12), `final.md` (generated verdict table),
`claims_ledger.md` (claim and reviewer-objection status).

## 1. One-paragraph status

The analysis is complete and the manuscript is a finished TMLR-format draft (34 pages, builds cleanly, every number
generated from a frozen result set, 0 hand-typed numbers, 0 undefined references). No further methodology work is
required. What remains is mechanical and on the user's side: place the official `tmlr.sty` and `tmlr.bst` in
`paper/`, commit and push the latest changes, then I rebuild under the real style, fix overfull boxes, re-run the
lint, write the cover note and give the densest paragraphs a last read. Nothing is submitted. My assessment is that
acceptance (possibly after one revision round) is more likely than not at TMLR and clearly better than the AISTATS
position, but I have no basis for a precise probability.

## 2. History in brief

* The earlier version (submitted to a conference, reviewed) made positive claims: parameters recoverable in some
  sense, error decomposition attributing the gap to the fitting objective, two model classes, a feature-leakage
  mechanism. Reviewers raised 14 objections (ledger section B).
* The revision was rebuilt as a pre-registered, machine-generated-numbers pipeline: plan in `PREREG.md` with
  amendments 1 to 8, each committed before the analyses it concerns (where an amendment declares post hoc
  hypotheses it says so). Result files are immutable per unit, frozen by SHA-256 index (five freezes so far). Product
  code under `ml/` was never touched.
* Target venue moved AISTATS to TMLR after the Phase 8 results (strategy decision): TMLR judges whether claims are
  supported, not novelty, which suits a careful negative-results paper.
* Phases: 8 (replica, replication on two more cohorts, coordinates, leakage with intervals), 9 (replica seeds,
  synthetic study, Shanghai prediction, second model class, optimizer checks), 10 (recalibrated cutoff,
  heteroskedastic noise), 11 (replica with smaller CGM noise, pooled calibration), 12 (replica with exact
  carbohydrate).

## 3. What the evidence says now

Three findings carry the paper:

1. **Gastric emptying (k_e) and carbohydrate absorption (k_a) cannot be separated from the iAUC.** Their exchange is
   an exact symmetry of the empty-gut cascade (Proposition 1, proved); the weakest Fisher direction lies along it in
   every subject of three cohorts (median |cos| 0.999 to 1.000); no profile interval for either rate is bounded under
   iAUC at three box widths and out to four times the upper bound; generic rank is 3 (ill-conditioned, not rank
   deficient). The symmetry is the glucose counterpart of the flip-flop ambiguity of oral pharmacokinetics (now
   cited); the paper's contribution is the measurement and the controls, not the symmetry.
2. **Insulin sensitivity S_I is only conditionally recoverable.** Bounded in 14% of interior subjects at the primary
   box (3 of 22); 20% pooled over five replica seeds when the model is exactly true; 38% in a random-truth synthetic
   study (52%, 37%, 25% by tercile of the true value). It depends on the noise level: on replica seed 0 the fraction
   is 26%, 37%, 49% as the replica CGM noise falls to one half and one quarter, still below the 80% the plan asked
   for. Heteroskedastic noise (two variants) does not bound S_I more often (0 of 15 and 0 of 2 interior subjects
   against 3 of 22). The weakness is not caused by carbohydrate logging error (15% bounded with exact carbohydrate).
3. **Prediction does not see any of this.** Gradient fitting and grid search differ by -1 mg/dL*min (95% interval -3
   to 1) on a 3025 error; three parameters versus S_I alone are equivalent within the pre-set margin of 150 in two
   cohorts (in Shanghai the three-parameter fit is better by 54, interval 30 to 94, inside the margin but not zero,
   which the paper states); a full-trace objective does not lower iAUC error.

Two qualifying results:

4. **Recoverability of the combination (tau1 = 1/k_e + 1/k_a, p = 1/(k_e k_a)) is not established.** On real data
   the trace bounds tau1 and p in 77% and 83% of interior subjects (Phase 8). Controls that failed: a true-model
   replica gives 65% and 48%; bounded replica intervals contain the truth in only 27% and 50%; 40% of trace fits are
   not likelihood optima; a cutoff calibrated on the replica lowers the real-data fractions to 41% and 62% (pooled
   over three replica seeds 48% and 62%). Phases 11 and 12 localized the cause of the undercoverage: with the
   carbohydrate given to the fit equal to the truth, the nominal 1.92 threshold covers (93% and 96%; empirical cutoff
   1.9) and the trace bounds only 17% and 23%; with the 25% logging error it bounds 65% and 48% but the intervals
   undercover. The input error makes trace intervals more often bounded and less often right. The real-data trace
   fractions (83%, 88%) are consistent with that mechanism, which is not shown. A second source of undercoverage at
   low noise even with exact carbohydrate (16% and 20%) is unidentified.
5. **A second model class (Dalla Man, float64 rebuild):** gut parameters bounded in 0 of 45 subjects, estimates on a
   bound in 31 to 40 of 45, Fisher effective rank 2 of 6. Narrow claim: gut parameters not recovered under iAUC in two
   gut structures; no replica was run for it and it has no exchange symmetry.

## 4. Verdict table (all pre-specified thresholds; none changed after any run)

H1 formulation-dependent; H2 met; H3 not met; H4 met; H5 not met; H6 met; H7 met; H8 met; H9 not met; H10 not met
(initialization factor tau 0.73 against 0.8; S_I first in 22 of 22); H11 not met; H12 not met; H13 not met; H14
reported (ratio 0.84, mixed); H15 met; H16 reported (exploratory); H17 not met; H18 not met; H19 met; H20 met; H21
not met; H22 met; H23 not met; H24 not met; H25 met. Of the 22 with a pass-or-fail rule, 10 are met and 12 not met.
Primary endpoints (H1, H3, H5, H7, H9): one met, three not met, one formulation-dependent. H13 to H25 were declared
after earlier results were seen (labelled post hoc throughout; thresholds fixed before each run).

## 5. What changed in the paper since the last report

* Restructured as decided: three findings lead; the scorecard moved to the appendix; claims table and abstract follow
  the same order.
* The combination claim was demoted to "not established" and the failed control is told in the main text.
* Positioning: HIPNO (Pan et al., arXiv 2608.10011, a hemodynamics paper that handles a scale symmetry) and GlucoFM
  (Li et al., arXiv 2605.30865) cited with one positioning sentence each; flip-flop pharmacokinetics (Gibaldi and
  Perrier) cited from memory; Kersting 2020 and Beck 2024 checked.
* Defects found in my own referee-style pass of the full PDF and fixed: the leakage check was described as Spearman
  but the code computes Pearson (text, caption and figure label now say Pearson; it also uses subjects with at least
  three meals, 97, not the ten-meal rule); duplicated verdict wording; a stale hypothesis range; an overstated
  "as well as" prediction claim; bare arXiv bibliography entries; anonymized package allow-list out of date.
* Reproducibility: anonymized code package built (`DigiTwin/anonymous_code.zip`, 27.6 MB, 6,183 files scanned, 0
  findings); internal planning notes excluded.

## 6. Reviewer objections (14)

Fully addressed: 1 (heuristics versus proof), 2 (ceiling confounded with the target), 3 (specification), 4 (error
decomposition, replaced by replica noise budgets), 5 (uncontrolled comparison), 6 (active bounds), 7 (equivalence
wording), 8 (joint identifiability), 10 (modality-wide ceiling claim not made), 11 (novelty positioning), 14
(presentation). Partial: 9 (second model class: gut parameters under iAUC only, no replica), 12 (other explanations:
carbohydrate error, heteroskedastic noise and misspecification via the replica covered; preprocessing beyond the
window not varied), 13 (design sweep not run; peak-timing objective not run as a fitting target). None ignored.

## 7. Risks, in the order I would worry about them

1. A reviewer reads the contribution as negative results on one model family, one observable family and three small
   cohorts. The limitations section leads with this; it will not remove the objection.
2. Post hoc hypotheses (H13 to H25) with thresholds fixed before each run but chosen knowing the earlier direction.
   Stated in the abstract-level scope and in the limitations.
3. Density: the experimental trail from H13 to H25 is long; a hostile reader may find it hard to follow. I can trim
   once the style files are in.
4. No independent reader has seen the paper. My review is not independent.
5. The symmetry is classical. The paper says so.
6. A second undercoverage source at low noise is unexplained, and the real-data trace fractions are only
   "consistent with" the input-error mechanism.

## 8. Decisions and inputs for the strategy side

1. **Stop experiments.** My recommendation: no further compute. Each recent run made the paper more defensible and
   less positive; more risks adding "not established" items. Only the low-noise undercoverage source and the
   not-run items (design sweep, Dalla Man replica, heteroskedastic trace) remain open, all declared as not run or
   not identified.
2. **One outside read before submitting** (a person who knows identifiability or CGM modeling), of the abstract,
   introduction and Section 5.5. This is the cheapest way to reduce risk.
3. **Reviewer suggestions on the TMLR form:** two or three names from identifiability or CGM modeling.
4. **Whether to post a preprint** (user decision).
5. **User actions pending:** commit and push the latest changes (the working tree held the Phase 11 and 12 text and
   the new table as of my last handoff; confirm), and supply `tmlr.sty` and `tmlr.bst`.

## 9. Remaining work (all small)

* After the style files: rebuild, fix overfull table boxes, lint, final read of the H23 to H25 paragraphs.
* Cover note and claims-and-evidence statement (the claims table already maps claims to evidence).
* Optional: shorten the experimental trail if the layout pushes the length far.

## 10. Process integrity and operations

* Every amendment preceded its runs; no threshold, rule or code path was changed after a first unit ran; earlier
  verdicts were never edited (corrections are logged in Amendments 5 onward); failed hypotheses are reported, not fixed.
* Pod lessons are in `docs/PHASE9.md` and memory: rebuild the venv in `/tmp` with a lean package set (no torch); prune
  `.jax_cache` on the 5 GB persistent disk; delete stale `*.claim` files before resuming; do not pull or commit while
  a queue runs; early ETAs in the runner log are inflated; the host is shared, so slow jobs are not necessarily broken.
* Amendment 6 was committed together with the code rather than alone (still before any Phase 10 result); Amendment 5
  likewise went into an earlier combined commit before Phase 9 ran.

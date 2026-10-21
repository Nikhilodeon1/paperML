# Report to the strategy agent: paperML after Phase 9 (TMLR track)

Date: 2026-10-10. Everything below is computed from the second frozen result set (`results/index.json`); the
detailed numbers are in `REPORTS/phase9.md`, the verdict table in `REPORTS/final.md`, the claim-by-claim
status in `REPORTS/claims_ledger.md`.

## 1. Where the project stands

* All compute is finished. Phase 9 (Amendment 5) ran to completion on a 30-core CPU pod, with two pod
  restarts resumed from per-unit result files. No Phase 9 unit failed. Results are merged locally and frozen
  a second time.
* A complete TMLR-format manuscript exists and builds (`paper/main.pdf`, 32 pages): zero hand-typed numbers
  (every number is a macro generated from the frozen results), zero to-do markers, zero pending macros, all
  citations resolve. The official `tmlr.sty` and `tmlr.bst` are not yet in the folder, so the build uses a
  fallback article class.
* Hypotheses graded: 19 with a pass or fail rule, 9 met, 10 not met; 1 formulation-dependent (H1), 2 reported
  without a rule (H14, H16). The paper reports all of them, failures included. H13 to H22 were declared after
  earlier results were seen and are labelled as such.
* Nothing from this session is committed. Amendment 5 is already in history (it must stay the first commit of
  the Phase 9 work); the remaining code, manuscript and reports are in the working tree.

## 2. What Phase 9 found

| hypothesis | verdict | what it says |
| --- | --- | --- |
| H6 (synthetic clause) | met | gradient diagnostic ranks parameters like the profile likelihood: Spearman 1.00, AUC 0.92 on real data and 0.98 on 90 random-truth synthetic subjects |
| H10 | not met | ordering stable across box width, log parameterization and L-BFGS (tau 1.00) but not across initializations (0.73, rule 0.8). S_I is always first, so only the order of the two gut rates changes, which the exchange symmetry predicts |
| H17 | not met | even when the model is exactly true, S_I is bounded in 20% of interior subjects pooled over five seeds (13 to 30% per seed); the bounded intervals are calibrated (91%). The S_I weakness is largely a property of the observable at this noise |
| H18 | not met | on the true-model replica the trace bounds tau1 in 65% and p in 48% (rules 70% and 50%); the bounded intervals contain the true value in only 27% and 50% |
| H19 | met (report) | random-truth synthetic study: S_I bounded in 38% of interior subjects, falling from 52% to 25% across terciles of the true S_I; k_e and k_a bounded in none of 90 subjects, and the estimates still sit on bounds |
| H20 | met | the prediction equivalences replicate in Shanghai (87 subjects): gradient vs grid -1 [-5, 4], three parameters vs S_I only -54 [-94, -30] (inside the margin, but not zero) |
| H21 | not met | trace-objective fits are likelihood optima to within 1.92 for only 60% of subjects (median gap 0.59, largest 41); the iAUC and centroid fits are fine (100%) |
| H22 | met | second model class (Dalla Man): kabs, kmax, kmin bounded in 0 of 45 subjects, estimates on a bound in 31 to 40 of 45, Fisher effective rank 2 of 6 |

## 3. The decisive finding: the "combination is recoverable" claim failed its own controls

The Phase 8 version of the story was: the gut rates cannot be separated, but their symmetric combinations
(tau1 = 1/k_e + 1/k_a and p = 1/(k_e k_a)) are recoverable from the full glucose trace (bounded in 77% and
83% of real subjects). Phase 9 tested that against a replica in which the model is true. The replica gives
fewer bounded intervals than the real data, those intervals mostly miss the true value, and the trace
optimizer does not reach an optimum for 40% of subjects. What survives is the ordering of the observables
(trace above centroid above area), which the replica reproduces. The absolute trace fractions are not evidence
of identifiability.

I rewrote the abstract, the introduction (contribution list and claims table), Section 5.4 and the
Discussion to say "not established", to report this as a control our own earlier result failed, and to list
the candidate explanations as untested (chi-square calibration of the profile threshold near the tied boundary
p = tau1^2/4, where about half to two thirds of cross-validation fits sit; noise-model misspecification for the
trace). I did not run tests to separate them.

## 4. What the paper claims now

1. k_e and k_a cannot be separated from the area: an exact symmetry of the empty-gut model (proved), the
   weakest Fisher direction lies along it in every subject of three cohorts, no profile interval for either
   rate is bounded under iAUC at three box widths or out to four times the upper bound. Strong, replicated,
   theory-backed.
2. S_I is only conditionally recoverable (14% of interior subjects at the primary box; about 20% when the
   model is exactly true; about 38% with truths spread over the box). Strong, with the replica and synthetic
   controls.
3. Prediction does not see any of this: gradient fit and grid search, and three parameters and one, are
   equivalent within a pre-set margin in two cohorts; a full-trace objective does not lower iAUC error.
4. A second model class shows the same non-identification of its gut parameters under iAUC (narrow claim; no
   replica was run for it).
5. Recovery of the combination from the trace is not established (above).
6. A protocol contribution: definitions of bounded, one-sided and flat intervals, the replica and synthetic
   instruments, and a pre-registered, amended plan with every failure shown.

The title ("Prediction Is Not Identification") and the main message survive. The part that did not survive is
the positive result that the paper used to soften its negatives.

## 5. Assessment for TMLR (my judgment, not a computed number)

* TMLR judges correctness and whether the claims are supported, not novelty or impact. This paper's claims
  now match its evidence, its negative results are reported with pre-set rules, and every number is
  traceable. Those are the properties TMLR reviewers check.
* The paper is now more defensible than at the Phase 8 freeze, and less exciting. It no longer has a
  positive identifiable-parameter result; its contributions are a proved symmetry, a protocol with controls,
  and a large body of negative evidence.
* Main risks, in the order I would worry about them:
  1. A reviewer reads the contribution as "one model family, one observable, three small cohorts, and
     negative results" and asks for more. The limitations section says this first; it will not make the
     objection disappear.
  2. H13 to H22 are post hoc. Thresholds were fixed before each analysis ran, but a skeptical reviewer will
     note they were chosen knowing the earlier direction. The paper states this.
  3. The trace result is now a hypothesis, not a finding, and some reviewers will ask why it is in the paper.
     The answer in the text is that it is the control that matters for anyone reproducing the work.
  4. Two planned analyses were not run (heteroskedastic noise sensitivity of the S_I profile, and the design
     sweep). The paper says so.
  5. Novelty objection from the earlier reviews (differentiable ODE inversion is established): handled by
     positioning against Kersting 2020 and Beck 2024 and by framing the paper as a protocol and an evidence
     base, not a method.
* I would put acceptance after revision as more likely than not at TMLR, clearly above the AISTATS
  position, but I do not have a basis for a precise probability and would not quote one.

## 6. Decisions and inputs needed from the strategy side

1. **Accept the demotion of the combination claim**, or authorise a follow-up that tries to settle it
   (cost: more compute and a new amendment; candidate analyses: profile intervals from a global optimizer, a
   likelihood-ratio calibration by simulation near the tied boundary, a noise-model check on the replica).
   My recommendation: do not delay submission for it; report it as an open, well-controlled question. If
   TMLR reviewers press, it is the natural revision experiment.
2. **Optional experiments** (each needs an amendment before it runs): the heteroskedastic-noise sensitivity and
   the design sweep (both already declared as not run); a replica or synthetic run for the Dalla Man model so
   the second-model-class claim is not limited to real data.
3. **Which paper did the reviewers mean by "HIPNO"?** The only HIPNO I found is a hemodynamics paper (arXiv
   2608.10011), which I did not cite. GlucoFM (arXiv 2605.30865) is cited.
4. **Submission mechanics** (user): drop in the official `tmlr.sty` and `tmlr.bst`; I then rebuild and fix
   overfull table boxes, build the anonymized package, and re-run the final lint. The user commits (first
   Amendment 5, then the rest) and decides on the public preprint.

## 7. Operational notes worth keeping

* The pod crashed twice during heavy jobs, apparently from host or disk conditions rather than the code (the
  cause was never confirmed). Resume worked each time because every unit is a result file. Lessons recorded in
  `docs/PHASE9.md` and in memory: rebuild the venv in `/tmp` with a lean package set (no torch), prune
  `.jax_cache` (it reached 2.8 GB of a 5 GB persistent disk), delete leftover `*.claim` files before resuming,
  never `git pull` while the queue runs.
* No threshold, rule or code path was changed after the first Phase 9 unit ran. Phase 8 verdicts were not
  edited; Phase 9 additions are layered on top of them and labelled.

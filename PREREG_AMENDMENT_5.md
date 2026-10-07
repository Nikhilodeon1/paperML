# Pre-registration amendment 5 (Phase 9, TMLR revision)

Amends `PREREG.md` and amendments 1 to 4. Committed before any Phase 9 analysis is run. No threshold from
an earlier document changes, and earlier verdicts are not edited: Phase 9 results are added to
`REPORTS/final.md` and described in `REPORTS/phase9.md`.

## What was seen before this amendment, stated plainly

Everything in the frozen Phase 8 results: `results/index.json` of 2026-10-05 00:58 local, 3,759 result
files, combined SHA-256 `2034a5dfb9d918b2a5f0dfe90486141a334caa94caa3544f037413042971efd3`, and the
summaries `results/phase2_summary.json` and `results/phase8_summary.json` built from them. Every
hypothesis below is therefore **post hoc** with respect to those results. Each threshold is fixed here,
before any Phase 9 work unit is run, and the verdict is the comparison of the observed number with it.

The venue changed from AISTATS to TMLR after the Phase 8 freeze. The reasons for the new analyses are the
weaknesses the Phase 8 results leave open, not a result that was disliked:

* the replica result (H13) rests on one random seed and one truth;
* the coordinate-profile result (H11) comes from fits where, for 20 to 32 subjects, the profile found a
  lower negative log-likelihood than the stored estimate, so the estimate was not an optimum;
* the prediction results (H7, H8, H9) come from one cohort;
* the second model class named in the submitted abstract has no result in the revision;
* the H6 synthetic clause and H10 were never run.

## Corrections to the Phase 8 record

**C1. Sign of a descriptive field.** `evaluation/phase8_summary.py` stored
`expected_log_ratio_if_S_I_absorbs_the_scale` as `-ln(scale)`. Scaling the logged carbohydrate by `s`
scales the insulin action needed to explain the same observed response by `s`, so the value is `+ln(s)`.
The code is fixed and the summary regenerated; no result file and no verdict changes, and the observed
medians (the quantities that were measured) are unaffected.

**C2. Shanghai 2x profile.** The Phase 8 run finished 85 of 87 subjects because two units exceeded the
runner's default per-unit limit of 1,800 s and were killed. Phase 9 re-runs the whole configuration with a
7,200 s limit. The Phase 8 result stays in place; the summaries use the most complete matching result set.

**C3. Cohort sizes.** The submitted abstract gave n = 45, 97 and 24. The revision's cohort rule (at least
10 meals per subject, 5 for Hall) gives 45, 87 and 15 subjects. The paper states the numbers produced by
the rule, computed from the cohort cache.

**C4. A summary section that mixed cohorts.** `profile_quality` in `evaluation/phase2_summary.py` read the
most complete A5 result set for the iAUC objective and box without checking the cohort. After Phase 8 added
Shanghai profiles (87 subjects at 1x, 85 at 2x) to the same analysis, the section described the Shanghai
set instead of the CGMacros set (for example, the counts of flat profiles under the four-times extension).
It is fixed to the match that `profile_lik.summarize` uses and the summary regenerated. No verdict, rule or
result file used that section, and every other summary reads its set through a cohort-aware match.

## Operational reading of H10 (original thresholds, unchanged)

`PREREG.md` states H10 as: Kendall tau at least 0.8 for the parameter ordering across 5 initializations,
3 bound settings, log versus linear parameterization, and Adam versus L-BFGS; `S_I` ranked first in at
least 95% of interior-converged subjects (read as "interior for `S_I`", Amendment 2). The words "the
parameter ordering" leave the computation open, so it is fixed here before the sweeps are run:

* the ordering is that of the three parameters by the bound-projected gradient diagnostic of the
  submitted paper (`projected_norm`, `evaluation/gradient_diag.py`); reference = Adam, rate space, the
  population start, the 1x box;
* for each subject and each factor, Kendall's tau-b between the reference ordering and the ordering under
  each level of the factor, averaged over the levels (five random initializations seeded 0 to 4; the 0.5x
  and 2x boxes; the log parameterization; L-BFGS);
* a factor passes when the median over subjects of that mean is at least 0.8. With three parameters tau
  takes only the values -1, -1/3, 1/3 and 1, so this means an identical ordering for most subjects;
* H10 is met when all four factors pass and `S_I` is ranked first by the reference diagnostic in at least
  95% of the subjects interior for `S_I`.

## New hypotheses

**H6 (synthetic clause).** Run as the random-truth replica of H19 below. `H6` is met when the real-data
clause of Phase 8 (median per-subject Spearman at least 0.7) holds and the synthetic clause (median
Spearman at least 0.7 and AUC at least 0.85 for detecting a profile-flat parameter, on the synthetic
subjects) holds. Otherwise it is not met. Until the synthetic study is run it stays "partially evaluated".

**H17 (replica stability).** The H13 replica (truth = each subject's regularized prediction-fit estimate
moved at least 5% inside the box, real meal schedules, AR(1) CGM noise from the subject's residuals,
carbohydrate logging error of CV 0.25) is repeated with seeds 1 to 4 and pooled with seed 0. Pooled over the
five seeds, the H13 rule is applied: `S_I` bounded in at least 50% of interior subjects and `k_e`, `k_a` in
at most 25%. Reported with it: the per-seed range of the `S_I` fraction, the number of seeds meeting the
`S_I` clause, and the coverage of the true value by the bounded `S_I` intervals (calibrated if at least
0.90).

**H18 (identifiable combinations under a true model).** Coordinate profiles in (log `S_I`, log `tau1`,
log `p`) with L-BFGS-polished estimates, on the seed-0 replica of H13, 1x box. Under the trace objective,
`tau1` is bounded in at least 70% and `p` in at least 50% of the subjects interior for that parameter (the
thresholds of the trace clauses of H11). iAUC and iAUC+centroid are reported
without a threshold. Reading: met means a true model with this noise gives at least the coordinate
identifiability the real data showed, so the real-data result is not evidence of anything beyond the
observable; not met means a true model would not, and the real-data result needs another explanation.

**H19 (synthetic recovery with known truth).** Random-truth replica: each synthetic subject has the real
meal schedule, true `S_I`, `k_e` and `k_a` drawn independently and uniformly inside the box (at least 5% of
its width from either bound), CGM noise from the subject's own residuals, and carbohydrate error of CV
0.25. Seeds 0 and 1 are pooled (each seed redraws every truth). The iAUC profile (primary box) and the
gradient diagnostic are computed on each synthetic subject. The H6 synthetic clause above is evaluated on
these. Reported without a threshold: the bounded fraction of each parameter among subjects interior for it,
by tercile of the true value; the coverage of the true value by bounded intervals; the median absolute log
error of the estimate.

**H20 (prediction replication on Shanghai).** The A9 protocol (five repeats of five-fold cross-validation
within subject, SHA-256 seeded folds, iAUC target) on the Shanghai subjects with at least 10 meals, cells
`grad3`, `grad1`, `grid3`, `grid1`, `personal_mean`, `population`, `persistence`. The paired subject-level
difference in held-out iAUC MAE, 90% bootstrap interval inside (-150, +150) mg/dL*min for both
`grad3` versus `grid3` (the H7 rule) and `grad3` versus `grad1` (the H8 rule). The margin is the same
absolute margin as in H7 and H8; the personal-mean MAE is reported so the scale of the margin is visible.
Reported without a threshold: every other pair, peak-time and trace error.

**H21 (are the coordinate estimates likelihood optima).** For the coordinate fits (1x box, real data,
objectives iAUC+centroid and trace), the maximum-likelihood estimate is re-optimized by L-BFGS-B in log
coordinates from the Adam estimate and from five seeded random points, and the best is kept. The gap is
`NLL(Adam estimate) - NLL(best)`. Rule: the gap is below 1.92 (the interval threshold) in at least 90% of
subjects, for each of the two objectives. If not met, the coordinate result of H11 is reported as limited by
the optimizer. The H11 clauses recomputed on the polished estimates are reported next to the Phase 8
numbers; the Phase 8 H11 verdict is not changed.

**H22 (second model class).** The Dalla Man (2007) meal model as implemented in `simulation/dalla_man.py`
(three-compartment gut with a nonlinear gastric-emptying law, two glucose and two insulin compartments,
delayed insulin action on glucose production, subcutaneous sensor compartment), rebuilt in float64, fitted
per subject to the same iAUC data of the 45 CGMacros subjects. Parameters `Vmx`, `kabs`, `kmax`, `kmin`, `f`,
`Td` in the ranges of `personalization/gradient_fit_dalla_man.py`; independent Gaussian errors with one
variance per subject, estimated from the Adam fit and then held fixed; maximum likelihood by Adam then
L-BFGS-B. Full Fisher matrix in log coordinates and 95% profile intervals (rise 1.92) for `Vmx`, `kabs`,
`kmax`, `kmin` over their boxes (11 grid points, the other parameters re-optimized). Rule: `kabs`, `kmax` and
`kmin` are each bounded in at most 25% of subjects. Met means the non-identifiability of the gut parameters
under iAUC replicates in a second model class. `Vmx` and the Fisher spectrum are reported without a threshold.

## Report-only items declared in advance

* Hall, trace objective, profile intervals at the 1x box (the Phase 8 run did the Fisher matrix only).
* The Holm-adjusted p-values of the twelve subject-level correlation tests per box (the Phase 8 table
  reported raw p-values only).
* The recomputed carbohydrate-scale reference value (C1).
* The bounded-interval fractions of the iAUC profiles recomputed from the stored profile grids at the 90% and
  99% interval levels (rises of 1.35 and 3.32), at the three boxes (`evaluation/delta_sensitivity.py`). No
  new fit is run; the 95% level of the plan stays the one every verdict uses.
* Everything H21 reports about the polished estimates, for the iAUC objective and the tied model as well.

## What is not added

The heteroskedastic-noise sensitivity of the `S_I` profile and the design sweep (A15) are not run in Phase
9, and the paper says so.

## Procedure

One result file per work unit, resumable, 30 workers, per-unit limit 7,200 s, queue order as in
`scripts/pod_phase9.sh` (H10, H21/H11, H18, H19/H6, H20, H22, H17, report-only). The queue may be stopped
at any point; each analysis is used if it completed and is dropped, with the paper saying it was not run,
if it did not. A second freeze (`results/index.json`) follows the last completed analysis. No threshold is
changed after the first Phase 9 unit is run, and no hypothesis is rescued by changing code or thresholds.

## Reporting commitments for the manuscript

The manuscript states which hypotheses were pre-registered (`PREREG.md`, amendments 1 to 3) and which were
declared after results were seen (H13 to H22); gives every verdict, failures included, beside its rule;
restricts each claim to what the cohort, window and model it was shown on support; and lists every analysis
that was planned and not run.

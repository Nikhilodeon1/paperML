# Pre-registration amendment 9 (Phase 13, TMLR revision)

Amends `PREREG.md` and amendments 1 to 8. Committed before any Phase 13 analysis is run. No threshold from an earlier
document changes, and earlier verdicts are not edited: Phase 13 results are added to `REPORTS/final.md` and described in
`REPORTS/phase13.md`.

## What was seen before this amendment, stated plainly

Everything in the fifth freeze (`results/index.json`, Phase 12) and the summaries through `phase11_summary.json`, and
an internal referee read of the manuscript PDF. The referee's criticisms that call for new analyses are the reason for
every item below. All items are **post hoc**. Each rule is fixed here, before any Phase 13 unit is run, and the items
without a rule are reported whatever they show.

The referee also asked for changes of framing and reporting (lead with the information content of the iAUC instead of
the exchange symmetry; report the three box widths and intervals for the insulin-sensitivity result; compare polished
fits with polished fits; scope the prediction claim to iAUC prediction; drop a tally that mixes planned and post hoc
hypotheses). They need no new analysis, change no hypothesis, rule or verdict, and are recorded in the deviations
appendix of the manuscript.

## Report-only analyses (no rule)

**R1. Prediction against parameter error.** The claim that held-out prediction hides the identifiability failures is
measured directly. Data: replica seed 0 with the subjects' fitted truths (the A9 replica results already stored at CGM
noise on and carbohydrate error 0.25), and random-truth replicas, seeds 0 and 1, for which the held-out cross-validation
of A9 (cells `grad3`, `grad1`, `personal_mean`, protocol unchanged) is run in Phase 13. For each subject: the held-out
iAUC mean absolute error of `grad3` (mean over repeats); the parameter error of the iAUC maximum-likelihood estimate of
the A5 replica profile result of the same replica, as the root mean square of the three absolute log errors of `S_I`,
`k_e` and `k_a`, and the separate log errors of `k_e` and `k_a`; and the noise floor, the held-out mean absolute error of
the model at the *true* parameters with the carbohydrate values that the fit is given, on the same meals. Reported: the
medians of the three errors across subjects, the ratio of the `grad3` error to the floor, and the Spearman correlation
between parameter error and held-out error across subjects with a subject-level bootstrap interval. Reading: prediction
near the floor together with a large parameter error and a weak correlation shows that prediction does not reveal the
failure; a strong correlation or a clear gap to the floor shows that it partly does, and the paper states what comes out.

**R2. Intervals for the headline fractions.** Wilson intervals with counts for every fraction in the abstract and the
contributions; the `S_I` result at all three box widths with their interior sets; subject-clustered percentile
bootstrap intervals (resampling the 45 subjects, each carrying its five replica seeds) for the pooled replica fractions,
since the seeds reuse the same subjects and meal schedules; and the comparison of the polished real-data trace fractions
with the polished replica fractions with their intervals. No new fit.

**R3. Prior-meal overlap.** For the CGMacros logs: the median gap between a meal and the previous logged meal of the same
subject, and the shares of meals whose previous meal began within three, four and six hours (the observation window is
three hours after the meal). Descriptive; it bounds how often the empty-gut convention is violated.

**R4. Numerical checks.** (a) For the first ten CGMacros subjects in identifier order, the Fisher information of the iAUC
objective at the primary box is recomputed with the solver tolerance tightened from 1e-4 to 1e-8, and the eigenvalues
(relative difference) and the cosine between the weakest timing eigenvector and the exchange direction are compared with
the stored values. (b) For the same subjects the iAUC profiles are recomputed with 500 inner Adam steps instead of 100, and
the number of profiles whose class (bounded, one-sided, flat) changes is reported.

**R5. A trivial insulin-sensitivity proxy for the HbA1c check.** For each Shanghai subject of the leakage analysis the mean
over meals of observed iAUC per gram of logged carbohydrate. Its Pearson correlation with HbA1c, raw and with fasting
glucose partialled out, with bootstrap intervals over subjects, is reported beside the SNPE and gradient estimates.

**R6. Carbohydrate-error sweep.** The trace coordinate profiles with polished estimates (the configuration of H18) on replica
seed 0 with carbohydrate error CV 0.1, 0.5 and 0.75, CGM noise at its estimate. Reported with the existing CV 0 and 0.25:
the bounded fractions of `tau1` and `p` among interior subjects and the coverage of the true value by the nominal
interval. The question is descriptive: whether any CV reproduces both the real-data fractions and a calibrated interval.

## New hypothesis

**H26 (are the polished trace estimates global optima?).** The trace coordinate profiles with polished estimates are rerun
with 30 random starting points in place of 5, on the real data and on replica seed 0 (primary box, otherwise the
configuration of H21). For the real data, the rule is: the negative log-likelihood of the Phase 9 polished estimate
exceeds that of the 30-start estimate by less than 1.92 in at least 90% of subjects. Met means the earlier polish was
already close to a global optimum. Not met means it was not, and the polished fractions of H11 and H18 are reported at the
30-start estimates beside the Phase 9 values. The bounded fractions of `tau1` and `p` at 30 starts are reported for both data sets.

## What is not added

A day-continuous replica (non-isolated meals), a perturbation of the fixed parameters `gamma` and `p3`, and a real-data
test of carbohydrate error in the cohort with standardized meals are not run, and the paper lists them as limitations. The
design sweep and a replica for the second gut model remain not run.

## Procedure

`scripts/pod_phase13.sh` runs, in this order: the random-truth cross-validation (R1), the carbohydrate-error sweep (R6),
and the two 30-start trace profiles (H26). R2 to R5 are computed locally from stored files and from two short local
recomputations (R4) by `evaluation/phase13_summary.py`. Resumable, per-unit limit 7,200 s. A sixth freeze follows. No
threshold is changed after the first Phase 13 unit is run, and no hypothesis is rescued by changing code or thresholds.
An analysis not finished is dropped and the paper says it was not run.

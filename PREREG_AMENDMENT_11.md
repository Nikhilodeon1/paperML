# Pre-registration amendment 11 (Phase 15, TMLR revision, external-review preparation)

Amends `PREREG.md` and amendments 1 to 10. To be committed before any Phase 15 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 15 results are added beside them.

## What was seen before this amendment, stated plainly

Everything through the seventh freeze (Phase 14 results, `results/phase14_summary.json`): in particular that a longer inner
optimization changes no profile class and does not restore coverage at low noise (H27 met, H28 not met), and that the
trace profiles are calibrated when they are rarely bounded and often bounded when they are not. While building the code
for this amendment, the following were looked at and none of them is used as a result: the simulated glucose difference
under exchange of the two rates for a population-default subject at 0, 2, 5 and 10% residual intestinal content (maximum
difference 0, 1.1, 2.7 and 5.4 mg/dL); one unit of the likelihood-at-the-truth analysis on a stored Phase 14 result
(CGMacros-001, quarter noise, exact carbohydrate: half likelihood ratio at the truth 0.45, no optimizer gap); and two
one-subject smoke tests of the profile code with reduced settings (grid of five points, twenty inner steps, 60-step fits)
whose profile classes were printed to see that the code ran. All items are **post hoc**. Each rule is fixed here, before any
Phase 15 unit is run, and the items without a rule are reported whatever they show.

## Why

Phase 14 left two things open. First, the trace profiles have never returned a bounded and a calibrated interval together,
so the manuscript cannot say that the detector can identify anything when the answer is known: the gut rates are separated
only by the exchange symmetry, and a design that breaks that symmetry should make them identifiable, by Proposition 2.
Second, the low-noise undercoverage with exact carbohydrate is unexplained, and a longer inner optimization is not the
cause. The likelihood at the true parameters separates two remaining explanations: a polished estimate that is not the
optimum, and a likelihood that places the truth far from its own optimum. A reviewer also asked what a wrong fixed
parameter does, which the manuscript lists as a limitation.

## New hypotheses

**H29 (a positive control: can the detector identify the gut rates when the symmetry is broken by design?).** The replica is
changed in one way: every meal starts with a fraction of its own glucose mass already in the intestine, at t = 0 (30 minutes
before the meal), in the truth and in the fit (`jax_engine.RESIDUAL_GUT_FRACTION`; the fit is given the same fraction, so
the model is exactly true). Data: random-truth replica seed 0 (truths drawn across the box, not near its edge), exact
carbohydrate (`carb_cv` 0), CGM noise at its estimate, 45 subjects. Fit: trace objective, rate parameterization, primary
box, polished estimates (five random starts), 500 inner steps, profile intervals as in H5. Rule, at a residual of **10%**:
among subjects interior for the parameter, the interval for `k_e` and the interval for `k_a` are each bounded in at least
**50%** of subjects, **and** the nominal interval (excess at the true value below 1.92) contains the true value for at least
**90%** of all subjects, for `k_e` and for `k_a`. Met means that when the symmetry is broken by design, with exact inputs and the
estimated noise, the detector returns bounded and calibrated intervals for the gut rates, so a bounded interval can be
believed. Not met means it does not, and the manuscript says which part failed (bounded fraction or coverage).

**H30 (is the polished estimate the optimum, relative to the truth, at low noise?).** For the trace coordinate profiles of
Phase 14 on replica seed 0 with exact carbohydrate and the CGM noise at one quarter (500 inner steps), the negative
log-likelihood is evaluated at the true parameters and an L-BFGS-B run is started from the truth (`evaluation.truth_calibration`).
Rule: the stored polished estimate exceeds the best optimum found from the truth by less than 1.92 in at least **90%** of
subjects. Met means the optimizer did not miss a better optimum, and the low-noise undercoverage belongs to the likelihood
(its calibration at the truth) and not to the estimate. Not met means the polished estimate was often not the optimum at low
noise, which would explain the undercoverage and would require the Phase 9 and Phase 14 polished fractions at low noise to be
read with that in mind.

## Report-only analyses (no rule)

**R10. The likelihood at the truth.** For the exact-carbohydrate replica at the estimated noise, at one quarter of it, and the
replica with carbohydrate error 0.25 (all Phase 14 coordinate profiles): the median and 90th percentile of half the
likelihood-ratio statistic at the truth (the negative log-likelihood at the truth minus the lowest value found) against the
corresponding quantiles of half a chi-square with three degrees of freedom, the share above 1.92, and the optimizer gap.

**R11. The dose response of the positive control.** The same profiles at 0, 5 and 20% residual (seed 0) and 10% on seed 1:
bounded fractions and coverage for `S_I`, `k_e` and `k_a`, and the coverage of the bounded intervals.

**R12. A wrong fixed parameter.** The standard replica (seed 0, carbohydrate error 0.25, truths near the fitted values) with
the truth's `gamma` (insulin secretion) multiplied by 1.3 and by 0.7, and the truth's `p3` (insulin action gain, `x_gain`)
multiplied by 1.3 and by 0.7, the fit keeping the population values. For `S_I` under the iAUC, as in H13: the bounded fraction
among interior subjects, the coverage of the true value by the nominal interval and by the bounded intervals, and the median
absolute and signed log error of the estimate, beside the unperturbed replica.

## What is not added

A positive control based on a liquid and a solid meal, a day-continuous replica, a replica for the second gut model and a
real-data test of carbohydrate error remain limitations of the manuscript and are not run. The residual content here is a
design choice for a control, not a model of free-living meals, and the manuscript says so.

## Procedure

`scripts/pod_phase15.sh` runs, in this order: the five residual-gut profile runs, the four perturbed-parameter profile runs and
the three likelihood-at-the-truth runs. It must run on the disk that holds the Phase 14 results. Resumable, per-unit limit
7,200 s. The summary is `evaluation/phase15_summary.py` (`python -m evaluation.make_paper_assets --phase15 --latex` builds
it). An eighth freeze follows. No threshold is changed after the first Phase 15 unit is run, and no hypothesis is rescued by
changing code or thresholds. Profile files of residual-gut and perturbed replicas record the replica keys `residual_gut` and
`perturb`, and result selection separates sets by them.

## Addendum 1 (written after the first launch of Phase 15 and before any Phase 15 result was read)

The first launch ran the residual-gut runs of H29 and R11 with 500 inner steps in place of the default 100, as written above. After
more than two hours, 5 of the 45 units of the first run (10% residual, seed 0) had finished and the others had gone more than
7,200 s without a heartbeat and were being killed and restarted; the Phase 14 runs, which have the same number of grid points,
took about 14 minutes in total. None of the finished units was read, and no Phase 15 summary was built. The runs were stopped.

Two settings of the profile runs change, and nothing else:

* the inner optimization uses the default **100 steps**. Phase 14 showed that 500 steps in place of 100 change no profile class
  (0 of 90 profiles in each of four settings) and do not change the coverage, so the budget does not affect the quantities
  H29 and R11 read;
* `extension_points` is set to **0**. The extension of the `k_e` and `k_a` grids to four times the upper bound is a diagnostic
  outside the classification (`extended`, never used by H5, H29 or any rule), so the in-box classes, the bounded fractions and
  the coverage that H29 reads are unchanged.

Every rule, threshold and data set above stays as written: H29 (10% residual, seed 0, `k_e` and `k_a` each bounded in at
least 50% of interior subjects and covered in at least 90% of subjects), H30, R10 to R12. The 500-step files of the abandoned
launch stay on disk; result selection separates sets by the number of inner steps, so they are not used. The likelihood-at-the-truth
runs (H30, R10) still read the Phase 14 coordinate profiles at 500 steps, unchanged. The 100-step profile files of
the residual-gut and perturbed replicas record `inner_steps` 100 and are selected by it.

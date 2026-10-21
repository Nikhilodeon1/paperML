# Pre-registration amendment 6 (Phase 10, TMLR revision)

Amends `PREREG.md` and amendments 1 to 5. Committed before any Phase 10 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 10 results are added to
`REPORTS/final.md` and described in `REPORTS/phase10.md`.

## What was seen before this amendment, stated plainly

Everything in the Phase 9 freeze: `results/index.json` of 2026-10-09 19:45 local, 5,405 result files, combined
SHA-256 `c80f0bfc1db54b4d20ed23027b99f8f7f432291f1ab1d8b4c1c25fefe121f1b7`, with `results/phase9_summary.json`.
In particular the verdicts of H17 to H22 (H18 and H21 not met: on a true-model replica the polished trace fits
bound `tau1` in 65% and `p` in 48% of interior subjects, the bounded intervals contain the true value in 27% and
50% of cases, and the Adam trace estimate is a likelihood optimum to within 1.92 for 60% of subjects). Both new
analyses below are therefore **post hoc**. Each rule is fixed here, before any Phase 10 work unit is run.

The reasons for the analyses are the two points the Phase 9 results leave open:

* the failed calibration of the coordinate intervals under the trace objective raises the question whether the
  nominal profile threshold (a rise of 1.92 in the negative log-likelihood) is simply the wrong cutoff for these
  profiles, and whether the real-data trace result survives a cutoff calibrated on the replica;
* the sensitivity of the `S_I` profile to heteroskedastic noise was declared in Amendment 4 as part of H16 and
  not run in Phase 9.

## New hypothesis

**H23 (recalibrated cutoff).** The nominal interval cutoff `delta = 1.92` is replaced, per parameter, by the
empirical 95th percentile of the profile excess at the true value on true-model replica subjects, and the
classification of the *real-data* profiles is recomputed from their stored grids. No new fit is run.

* *Statistic.* For a replica subject and a parameter with true value `t`, the excess is
  `profile(t) - reference`, with `profile` interpolated linearly in the logarithm of the grid and `reference`
  the stored reference (the lower of the loss at the estimate and the profile minimum). A subject whose true
  value lies outside the in-box grid is excluded and counted.
* *Cutoff.* The 95th percentile (linear interpolation) of the excesses, with a 95% percentile-bootstrap
  interval from resampling real subjects (the replica seeds share subjects), using the resample count of the
  other analyses.
* *Sets.* (a) `S_I` under iAUC at the primary box: replica seeds 0 to 4 pooled, applied to the real-data iAUC
  profiles at the primary box. (b) `tau1` and `p` under the trace objective in coordinates with polished
  estimates, primary box: replica seed 0, applied to the real-data polished coordinate profiles.
* *Reading and rule.* H23 is met when both parts hold, at the calibrated cutoffs and among subjects interior for
  the parameter (Amendment 2 definition): (a) the `S_I` interval is bounded in fewer than 50% (the weakness
  conclusion of H13 and H17 is unchanged); (b) `tau1` is bounded in at least 70% and `p` in at least 50% (the
  trace clauses of H11 and H18). Otherwise not met. A met verdict would mean the real-data trace result
  survives a calibration that the nominal cutoff did not give it; a not-met verdict on part (b) would mean it
  does not.
* *Reported with it, without threshold.* The cutoffs, their intervals and the number of excluded subjects; the
  bounded fractions at the nominal and the calibrated cutoff on the real data and on the replica; the coverage
  of the true value at the calibrated cutoff on the replica (95% by construction in sample, shown so that the
  statistic is checked); the same for `k_e` and `k_a` is not computed, since no interval for either is bounded
  at any cutoff in the stored profiles.
* *Limit stated in advance.* The cutoff is calibrated on a replica whose truths are the fitted values of the real
  subjects, so it is a calibration for replica-like truths; it is not a general correction.

**H16 (heteroskedastic noise, `S_I` profile; completes the exploratory item of Amendment 4).** The iAUC
objective of the 45 CGMacros subjects, primary box, real data, is refitted by maximum likelihood (pilot fit, frozen
noise, then unregularized fit, as in Amendment 1) under two heteroskedastic noise models in which the standard
deviation of the residual of meal `i` is `sigma * s_i^e`, with `s_i = max(observed iAUC_i, 0.1 * mean observed iAUC)`
and `e = 1/2` (`root`: variance linear in the area) or `e = 1` (`proportional`: error proportional to the area).
`sigma` is estimated from the pilot residuals divided by `s_i^e` and then frozen. Only the profile of `S_I`
is computed (15 grid points, the other parameters re-optimized, interval at the nominal cutoff of 1.92).
Reported, without a pass or fail rule: the fraction of subjects with a bounded `S_I` interval among those
interior for `S_I`, with Wilson intervals, under each structure and under the homoskedastic baseline of Phase 2;
the per-subject changes of classification. For context only, the `S_I` clause of H5 (at least 80% bounded) is
stated beside the numbers: if it remains unmet under both structures the paper says the `S_I` result does not
depend on the variance structure tried; otherwise it says it does and reports by how much.

## What is not added

The design sweep (A15) and a replica or synthetic study for the Dalla Man model are not run, and the paper says
so. The heteroskedastic analysis covers the iAUC objective only; no heteroskedastic model for the trace is
defined.

## Presentation change recorded here

The manuscript is restructured so that the main text leads with three findings (the exchange symmetry, the
conditional recoverability of `S_I`, and the insensitivity of held-out prediction to both) and the scorecard of
hypotheses is in an appendix. This changes no hypothesis, rule or verdict.

## Procedure

H23 is computed from stored result files by `evaluation/recalibration.py`; H16 is run by
`python -m evaluation.runner evaluation.profile_lik` with `noise_structure` set to `root` and `proportional` and
`profile_only` set to `["insulin_sensitivity"]`, 45 units each, resumable. A third freeze (`results/index.json`)
follows. No threshold is changed after the first Phase 10 unit is run, and no hypothesis is rescued by changing
code or thresholds. The work is time-boxed to one day; an analysis not finished is dropped and the paper says it was
not run.

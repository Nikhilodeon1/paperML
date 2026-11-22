# Pre-registration amendment 14 (Phase 18, TMLR revision, robustness of the input-error result)

Amends `PREREG.md` and amendments 1 to 13. To be committed before any Phase 18 analysis is run. No threshold from an earlier
document changes, and earlier verdicts are not edited: Phase 18 results are added beside them.

## What was seen before this amendment, stated plainly

Everything through the tenth freeze (Phases 15 to 17), and a second outside read of a condensed version of the manuscript. That
read asked for the carbohydrate-error result of Section 5.4 to be explained and replicated, and it raised several points that
this amendment tests. While preparing the code, the following were looked at from stored results and none is used as a result:

* In the carbohydrate-error sweep (trace coordinate profiles, replica seed 0) the pilot residual standard deviation is almost
  unchanged by the error: the median is 6.045 at error 0 and 6.051 at error 0.1 (mg/dL, innovation scale), and the negative
  log-likelihood at the estimate is about 665 at both. The autocorrelation is about 0.99 throughout.
* The cutoff on the profile that would cover the truth for 90% of subjects, expressed as a factor on the noise standard
  deviation, is 0.99 at error 0, 1.2 at 0.1, 3.4 at 0.25, 9.3 at 0.5 and 16.8 at 0.75 for tau1, and 0.99, 0.88, 1.9, 4.0 and 7.0
  for p. At those cutoffs the bounded fraction is 20, 21, 12, 0 and 0% for tau1.
* The median signed log error of `S_I` under the iAUC on the exact-carbohydrate replica is -0.20, with 19 of 45 estimates on a
  bound, so the offset of the unperturbed replica in the perturbation analysis does not come from carbohydrate error.
* One smoke test of the new code on one subject (CGMacros-001, error 0.75, a reduced grid): the noise standard deviation
  re-estimated at the polished estimate was 4.428 against 4.429 at the pilot, and the profile class of tau1 was unchanged.

All of this is **post hoc**. Each rule below is fixed before any Phase 18 unit is run, and the items without a rule are reported
whatever they show.

## Why

The carbohydrate-error result (intervals for tau1 and p become more often bounded and less often right as the logging error
grows) is one of the paper's three main findings and rests on one replica seed, one cohort and an independent per-meal error.
The stored numbers above suggest a mechanism: the fitted parameters absorb the input error, so the residual does not grow, the
frozen noise model does not widen the interval, and the estimate moves by more than the interval. Phase 18 tests the parts of
that account that stored results cannot, and it asks whether the pattern replicates.

## New code

* `profile_lik` option `noise_source`: "pilot" (the default, every earlier result) or "final", which estimates the residual standard
  deviation and the autocorrelation again at the polished estimate, rebuilds the objective with them and polishes once more from
  that estimate before profiling. Payloads record `noise_source`, and result selection separates sets by it.
* Replica key `carb_bias`: a systematic logging error, one unit-mean lognormal multiplier per subject (coefficient of variation
  `carb_bias`) shared by every meal of the subject, in addition to any per-meal `carb_cv`.
* `prediction_cv` option `fold_scheme`: "random" (every earlier result), "blocked" (contiguous blocks of meals in time order, the
  other blocks train) and "forward" (a block is held out and only earlier blocks train). Payloads record `fold_scheme`.

## New hypotheses

**H35 (is the undercoverage a noise-level matter?).** Replica seed 0, carbohydrate error 0.25, trace coordinate profiles with
polished estimates and `noise_source` "final". Rule: the nominal interval (excess at the true value below 1.92) contains the true
value for at least **90%** of subjects, for tau1 and for p. Met means that estimating the noise at the fit repairs the coverage
and the earlier result is a matter of the pilot noise estimate. Not met means input error is not a noise-level matter and the
mechanism above stands. The expected outcome, given the smoke test, is not met.

**H36 (does the pattern replicate across seeds?).** Replica seeds 1 and 2 (seed 0 exists), trace coordinate profiles with polished
estimates at carbohydrate error 0 and 0.75. Rule: on each seed, for tau1 and for p, the bounded fraction among interior subjects at
0.75 exceeds that at 0 by at least **30 points** and the nominal coverage at 0.75 is at least **30 points** below that at 0.
(Seed 0 gives 71 and 82 points for tau1 and 55 and 65 for p.)

**H37 (is a systematic error absorbed?).** Replica seed 0, no per-meal error and a systematic error `carb_bias` 0.25, trace
coordinate profiles with polished estimates. Rule: the nominal interval contains the true value for at least **90%** of subjects,
for tau1 and for p. Met means a pure scale error in the logged carbohydrate is absorbed by `S_I` and leaves the timing
combination alone, so that the per-meal independent error is what breaks the intervals. Not met means a systematic error breaks
them too.

**H38 (does the pattern replicate on a second cohort?).** A replica built from the Shanghai subjects (at least ten meals, seed 0),
trace coordinate profiles with polished estimates at carbohydrate error 0 and 0.75. Rule: the same as H36, **30 points** both ways,
for tau1 and for p.

## Report-only analyses (no rule)

**R17. The systematic error.** The bounded fractions, coverage and median log error of `S_I` under the iAUC for the H37 replica.

**R18. The Shanghai real data.** Bounded fractions among interior subjects for `S_I`, `k_e` and `k_a` under the iAUC (the primary box,
Phase 8 profiles) and for tau1 and p under the trace (polished coordinate profiles).

**R19. Chronological folds.** On CGMacros, the mean held-out iAUC error of the gradient fit of three parameters, the gradient fit of
`S_I` alone, the personal mean and the population values under random, blocked and forward folds, and the paired difference of
the three-parameter fit from the personal mean and from the `S_I`-only fit with a bootstrap interval over subjects.

## What is not added

A sandwich or bootstrap interval, a learned or amortized estimator of the personal parameters, a replica for the second gut model,
per-subject basal glucose and a sensor-lag parameter are not run. They remain limitations of the manuscript.

## Procedure

`scripts/pod_phase18.sh` runs, in order: the two re-estimated-noise runs, the two systematic-error runs, the six extra-seed runs,
the two Shanghai real-data runs, the two Shanghai replica runs and the two chronological cross-validation runs. Resumable, per-unit
limit 7,200 s. The summary is `evaluation/phase18_summary.py` (`python -m evaluation.make_paper_assets --phase18 --latex` builds it).
An eleventh freeze follows. No threshold is changed after the first Phase 18 unit is run, and no hypothesis is rescued by changing
code or thresholds. Replica runs have no stored fit and refit every subject, so they take longer than the profile runs of Phases
14 to 17. Earlier files count as the pilot noise model, random folds and no systematic error.

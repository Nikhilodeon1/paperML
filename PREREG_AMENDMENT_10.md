# Pre-registration amendment 10 (Phase 14, TMLR revision, external-review preparation)

Amends `PREREG.md` and amendments 1 to 9. To be committed before any Phase 14 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 14 results are added beside them.

## What was seen before this amendment, stated plainly

Everything through the sixth freeze (Phase 13 results, `results/phase13_summary.json`). In preparing the manuscript for an
external read, the stored diagnostics of the existing 100-step trace coordinate profiles were inspected, which are fields
already in every A5 result file: the number of in-box grid points flagged `unsettled` (inner optimization still
improving by more than 0.05 in negative log-likelihood) and the flag `better_than_theta_hat`. For the polished trace
coordinate profiles on replica seed 0 the flagged share of in-box grid points was 5.7% with exact carbohydrate at the
estimated noise and 22.1% with exact carbohydrate at a quarter of the noise (7.5% and 21.1% with carbohydrate error 0.25
at the estimated noise and at a quarter of it). That rise at low noise is the reason for R7 below. A smoke test of the
R9 code on the stored files was run and its structure checked (counts of subjects and result keys); the numbers it
produces were not read. All items are **post hoc**. Each rule is fixed here, before any Phase 14 unit is run, and the
items without a rule are reported whatever they show.

## Why

An unsettled inner optimization overstates a profile: the profile minimum at a grid point is too high, which makes a flat
parameter look bounded, and a spuriously bounded interval need not contain the truth. The manuscript reports two
patterns that this mechanism could produce: trace intervals that are bounded more often and cover the truth less often
as the carbohydrate error grows, and a second source of undercoverage at low noise with exact carbohydrate that the
manuscript calls unidentified. It also reports bounded fractions for the real data (83% and 88%) that depend on the same
inner optimizer. A longer inner optimization is the direct test, and it can also show, in the case with exact inputs, low
noise and enough inner steps, whether the detector can return "bounded" and "covers the truth" together at all.

## New hypotheses

**H27 (do the real-data trace fractions depend on the inner optimizer budget?).** The trace coordinate profiles with polished
estimates (primary box, five starts, grid and everything else as in H18) are rerun on the real data with 500 inner Adam
steps in place of 100. Rule: the bounded fractions of `tau1` and of `p`, among subjects interior for the parameter, each
differ from the 100-step fractions (83% and 88%) by at most 10 percentage points. Met means the real-data fractions do not
depend materially on the inner budget. Not met means they do, and the paper reports the 500-step fractions beside the
100-step ones and says which of its statements they change.

**H28 (does a longer inner optimization restore calibration at low noise?).** The same profiles at 500 inner steps on replica
seed 0 with exact carbohydrate (`carb_cv` 0) and the CGM noise scaled to one quarter. Rule: the nominal interval contains
the true value for at least 90% of subjects, for `tau1` and for `p`. Met means the second source of undercoverage was the
inner optimizer. Not met means it was not, and the manuscript keeps the statement that a second source is unidentified.

## Report-only analyses (no rule)

**R7. The inner-optimizer budget in full.** The 500-step runs of H27 and H28, and the same on replica seed 0 with carbohydrate
error 0.25 at the estimated noise and with exact carbohydrate at the estimated noise. For every setting, with the stored
100-step run beside it: bounded fractions of `tau1` and `p` among interior subjects with Wilson intervals, the
coverage of the true value by the nominal interval and by the bounded intervals, the empirical cutoff, the share of
flagged grid points, the number of profiles whose class changes between 100 and 500 steps and the direction of the change.

**R8. The detector under ideal inputs.** From the 500-step run with exact carbohydrate and a quarter of the noise: how often
`tau1` and `p` are bounded and whether the bounded intervals contain the truth. The reading is a positive control. If the
intervals are calibrated and often bounded there, the detector can identify the combination when the inputs are exact and
the noise is low, and the real-data failure is a failure of inputs and noise. If they are calibrated and rarely bounded,
the combination is not identifiable under these conditions either, and the manuscript says so.

**R9. Peak-time and trace error against the floor of the true model.** For the random-truth replicas (seeds 0 and 1) and for the
replica with fitted truths where the stored cross-validation results allow it: per subject, the held-out peak-time error and
the held-out trace root-mean-square error of the iAUC-fitted cells `grad3` and `grad1`, divided by the same quantity for the
model at the true parameters on the same meals (the A16 floor). Reported: the medians, the ratios, their Spearman correlation
with the timing-parameter error with a subject-level bootstrap interval, and the paired difference between `grad3` and
`grad1` in units of the floor. This extends R1 from the iAUC to the peak time and the trace, from stored files only.

## What is not added

A positive control that breaks the exchange symmetry by design (residual intestinal content, a liquid and a solid meal), a
day-continuous replica, a perturbation of the fixed parameters `gamma` and `p3`, a replica for the second gut model, and a
real-data test of carbohydrate error remain limitations of the manuscript and are not run.

## Procedure

`scripts/pod_phase14.sh` runs the four 500-step profile runs in the order quarter-noise exact, estimated-noise exact,
carbohydrate error 0.25, real data. Resumable, per-unit limit 7,200 s. The summary is `evaluation/phase14_summary.py`
(`python -m evaluation.make_paper_assets --phase14 --latex` builds it). A seventh freeze follows. No threshold is changed
after the first Phase 14 unit is run, and no hypothesis is rescued by changing code or thresholds. The profile-result files
now record `inner_steps`, and result selection separates sets by it; earlier files without the field count as 100.

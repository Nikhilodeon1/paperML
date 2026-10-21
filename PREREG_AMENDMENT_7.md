# Pre-registration amendment 7 (Phase 11, TMLR revision)

Amends `PREREG.md` and amendments 1 to 6. Committed before any Phase 11 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 11 results are added to
`REPORTS/final.md` and described in `REPORTS/phase11.md`.

## What was seen before this amendment, stated plainly

Everything in the Phase 10 freeze (`results/index.json` of 2026-10-21, third freeze) and the summaries
`results/phase9_summary.json` and `results/phase10_summary.json`. In particular: on the true-model replica the
polished trace fits bound `tau1` in 65% and `p` in 48% of interior subjects against 83% and 88% on the real data
(H18 not met); the nominal 1.92 interval contains the true value for 62% (`tau1`) and 82% (`p`) of replica
subjects; with a cutoff calibrated on the replica the real-data fractions are 41% and 62% (H23 not met); and the
calibrated cutoffs for the coordinates rest on 45 replica subjects, with wide bootstrap intervals. The new
analyses are therefore **post hoc**. Each rule is fixed here, before any Phase 11 unit is run.

The reasons are two open points in the paper: the real data show more bounded coordinate intervals than the
replica, and the paper does not say why; and the coordinate cutoffs are poorly determined.

## Replica with a smaller CGM noise scale

The replica of Amendment 4 draws CGM noise from the subject's residuals around the real maximum-likelihood
fit, which contain model misfit as well as sensor noise, so the replica noise is an upper bound on sensor
noise. The replica configuration gains an optional key `cgm_scale` (default 1, which reproduces every earlier
replica exactly). The CGM noise of the replica is multiplied by `cgm_scale`; the random draws are the same, so
scales are compared on common random numbers. The carbohydrate error (CV 0.25), the truths and the meal
schedules do not change. The fit estimates its own frozen noise model from the replica data, as before.

## New hypothesis

**H24 (is the replica noise too large?).** Replica seed 0 with `cgm_scale` equal to 0.5 and to 0.25, two
analyses at each scale at the primary box: (a) the iAUC profile (rate parameterization), and (b) the trace
coordinate profile with L-BFGS-polished estimates, the configuration of H18. Reported at each scale, together
with the scale 1 results of H13 and H18, without a threshold: the bounded fraction of `S_I` among subjects
interior for it, under iAUC, and of `k_e` and `k_a`; the bounded fraction of `tau1` and `p` under the trace among
subjects interior for each; the share of replica subjects whose true value lies inside the nominal interval; the
coverage of the bounded intervals; and the calibrated cutoff of Amendment 6 with its interval.

Rule. H24 is **met** when, at some scale in the grid, both of these hold for the trace coordinate profiles:

* the bounded fractions among interior subjects reach at least 73% for `tau1` and at least 78% for `p` (ten
  percentage points below the 83% and 88% of the polished real-data fits, Phase 9); and
* the nominal interval (rise 1.92) contains the true value for at least 90% of replica subjects, for both
  `tau1` and `p`.

Otherwise **not met**. Reading: met means a smaller replica noise reproduces the real-data fractions with a
nominal cutoff that covers, so the earlier gap between the replica and the real data is a noise-level effect;
not met means it is not explained by the noise levels tried. A not-met verdict is reported as such. The iAUC part
carries no rule: it is reported so that the paper can say whether the `S_I` result depends on the noise scale.

## Report-only item: more replica seeds for the coordinate calibration

Replica seeds 1 and 2 are run for the trace coordinate profiles with polished estimates (the configuration of
H18, scale 1). The calibrated cutoffs of H23 are recomputed pooling seeds 0 to 2 and the real-data fractions at
those cutoffs are reported next to the Amendment 6 numbers. No verdict is re-graded: H23 keeps the verdict of
Amendment 6, and the pooled numbers are reported in addition, to say how well the cutoffs were determined.

## What is not added

The design sweep (A15), a replica or synthetic study for the Dalla Man model, and heteroskedastic models for
the trace are not run, and the paper says so.

## Procedure

`scripts/pod_phase11.sh` runs, in this order: the two trace coordinate profiles of H24, the two iAUC profiles of
H24, then replica seeds 1 and 2 for the trace coordinates; `evaluation/noise_scale.py` computes the summaries from
stored files. Resumable, per-unit limit 7,200 s. A fourth freeze follows. No threshold is changed after the first
Phase 11 unit is run, and no hypothesis is rescued by changing code or thresholds. The work is time-boxed to one
day; an analysis not finished is dropped and the paper says it was not run.

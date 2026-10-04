# Pre-registration amendment 4 (Phase 8)

Amends `PREREG.md` and amendments 1 to 3. Committed before any Phase 8 analysis is run. No threshold from
an earlier document changes, and earlier verdicts are not edited: Phase 8 results are added to
`REPORTS/final.md`.

**Timing of this amendment.** H13 to H16 are declared AFTER the CGMacros results in
`results/phase2_summary.json` were seen. They are motivated by the failure of the `S_I` clause of H5 (a
bounded interval in 14% of interior subjects at the primary box) and by reviewer point (4), the undefined
error decomposition. H11 and H12 (Amendment 1) are run in Phase 8 for the first time.

## New hypotheses

**H13 (well-specified replica, identifiability).** The truth is set to each subject's regularized
prediction-fit estimate (moved at least 5% inside the box), with the real meal schedules, AR(1) CGM noise
estimated from that subject's real residuals, and carbohydrate logging error of CV 0.25. Among replica
subjects interior for `S_I` at the primary box, the fraction with a bounded `S_I` profile interval is at
least 50%, and for `k_e` and `k_a` at most 25%. Interpretation: an `S_I` fraction of at least 50% attributes
the real-data shortfall (14%) to misspecification or unmodelled variability; at most 25%, the `S_I`
weakness is inherent to the observable at this noise level; in between, reported as mixed.

**H14 (replica, prediction).** Held-out iAUC MAE of the `grad3` pipeline on replica data at the same
noise, divided by the real-data MAE. At most 0.7: a substantial part of the real held-out error is not
explained by noise or optimization. At least 0.9: noise accounts for almost all of it. In between: mixed.
No additive attribution is claimed in any case.

**H15 (replication on other cohorts, iAUC, primary box).** For Shanghai and for Hall, the `k_e` and `k_a`
profile intervals are bounded in at most 25% of subjects, and the weak Fisher direction has absolute cosine
at least 0.8 with the exchange direction in at least 80% of subjects. `S_I` results are reported with no
threshold.

**H16 (exploratory, no thresholds).** The `S_I` sensitivity-saturation curve; carbohydrate-scale
sensitivity; heteroskedastic noise-model sensitivity.

## Procedure

One result file per work unit, resumable, content-hashed cohort cache, 30 workers, the node shut down after
the last queued job. Compute and analysis finish by 2026-10-05 12:00 local time and results are frozen.
Anything not in `results/index.json` at the freeze is dropped, and the paper says it was not run.
Priority if time is short: T0, T1, T2, T3, T4, T5(i)(ii), T6, T7, T5(iii); cut from the bottom.

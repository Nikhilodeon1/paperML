# Pre-registration amendment 8 (Phase 12, TMLR revision)

Amends `PREREG.md` and amendments 1 to 7. Committed before any Phase 12 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 12 results are added to
`REPORTS/final.md` and described in `REPORTS/phase12.md`.

## What was seen before this amendment, stated plainly

Everything in the fourth freeze (`results/index.json` after the Phase 11 merge, 2026-10-21) and
`results/phase11_summary.json`. In particular H24 was **not met**: reducing the replica CGM noise to one half and one
quarter raised the bounded fractions of `tau1` and `p` under the trace (from 65% and 48% to 73% and 71%, and 69%
and 81%) while the nominal interval covered the true value *less* often (`tau1`: 62%, 33%, 11%; `p`: 82%, 51%,
22%), and the cutoff that would cover rose from 22.7 to 68.5 and 183.9 for `tau1`. The `S_I` fraction under iAUC
rose from 26% to 37% and 49%. H25 is therefore **post hoc**; its rule is fixed here, before any Phase 12 unit is run.

The reason: a miscoverage that grows as the noise falls is what an error that the noise does not remove would
produce. The one such error in the replica is the carbohydrate logging error (lognormal, CV 0.25), which enters
the fit as a wrong input. This amendment tests that candidate directly.

## New hypothesis

**H25 (is the undercoverage caused by the carbohydrate logging error?).** Replica seed 0 with the carbohydrate given
to the fit equal to the true carbohydrate (`carb_cv` 0), at CGM noise scales 1 (the estimated noise, as in H13 and
H18) and 0.25. At each scale, two analyses at the primary box: the iAUC profile (rate parameterization) and the trace
coordinate profile with L-BFGS-polished estimates (the configuration of H18). Everything else (truths, meal
schedules, random draws, polishing, the nominal cutoff of 1.92) is as in H18 and H24.

Rule. H25 is **met** when, at CGM noise scale 1 with exact carbohydrate, under the trace the nominal interval
(rise 1.92) contains the true value for at least 90% of replica subjects, for both `tau1` and `p`. Otherwise
**not met**. Reading: met means the undercoverage of the nominal interval seen in H18 and H24 is attributable to
the logging error of the carbohydrate input, so the profile threshold itself is adequate when the inputs are exact;
not met means the nominal threshold undercovers even with exact inputs, so the cause is the likelihood or the
optimizer. A not-met verdict is reported as such.

Reported with it, without a threshold: at both scales, the bounded fractions of `tau1`, `p` and `S_I` among
interior subjects under the trace; the coverage of the true value by the nominal interval for `S_I` as well; the
calibrated cutoffs of Amendment 6; and under iAUC the bounded fraction of `S_I`, `k_e` and `k_a` among interior
subjects and the coverage of the true `S_I`. The two cells at scale 1 and 0.25 with CV 0.25 (H18, H24) and the two
cells with CV 0 form a two-by-two comparison of CGM noise and logging error on seed 0; the paper reports it as such
and does not claim an additive attribution.

## What is not added

The design sweep (A15), a replica or synthetic study for the Dalla Man model, and heteroskedastic models for the
trace are not run, and the paper says so. No further seed is run for H25.

## Procedure

`scripts/pod_phase12.sh` runs the two trace coordinate profiles first and the two iAUC profiles second;
`evaluation/noise_scale.py` computes the summary from stored files. Resumable, per-unit limit 7,200 s. A fifth
freeze follows. No threshold is changed after the first Phase 12 unit is run, and no hypothesis is rescued by
changing code or thresholds. An analysis not finished is dropped and the paper says it was not run.

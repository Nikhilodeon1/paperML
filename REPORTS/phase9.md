# Phase 9 report (Amendment 5, TMLR revision)

Source: `results/phase9_summary.json` and `results/P7_paper_numbers`, built from the second freeze
(`results/index.json`, 2026-10-09). The queue ran on a 30-worker CPU pod and was resumed twice after pod
restarts; every analysis finished (no failed unit in any Phase 9 configuration; the `.failed.json` files in the
merged `results/` belong to earlier configurations). Thresholds are those of `PREREG_AMENDMENT_5.md`; no
threshold, code path or hypothesis was changed after the first Phase 9 unit ran.

## Verdicts

| hypothesis | verdict | deciding numbers |
| --- | --- | --- |
| H6 (synthetic clause) | **met** | real data: median Spearman 1.00, AUC 0.92; 90 synthetic subjects (seeds 0, 1): median Spearman 1.00, AUC 0.98 (182 of 270 pairs profile-flat). With three parameters a Spearman coefficient takes few values |
| H10 | **not met** | median over subjects of the mean Kendall tau-b with the reference ordering: initialization 0.73 (fails; five seeds, identical ordering in 53 to 82% of subjects), box 1.00, log parameterization 1.00, L-BFGS 1.00. S_I first in 22 of 22 interior subjects at every level. With S_I always first, only the order of k_e and k_a can change |
| H17 | **not met** | pooled over replica seeds 0 to 4: S_I bounded in 32 of 157 interior subjects (20%, Wilson 15 to 27%), per seed 13 to 30%, no seed reaches 50%; timing rates bounded in 0%; coverage of bounded S_I intervals 29 of 32 (91%, calibrated). Reading: the S_I weakness is largely inherent to the observable at this noise level |
| H18 | **not met** | polished coordinate profiles on the seed-0 replica, trace: tau1 bounded in 65% (22 of 34; rule 70%), p in 48% (16 of 33; rule 50%); centroid 12% and 0%; iAUC 0% and 0%. Coverage of the true value by bounded trace intervals: tau1 27% (6 of 22), p 50% (8 of 16), S_I 85% (28 of 33) |
| H19 | **met** (report) | 90 random-truth subjects: S_I bounded in 38% of interior subjects (52, 37, 25% by tercile of the true S_I), coverage 84% (27 of 32), median absolute log error 0.19; k_e and k_a bounded in 0 of 90 (estimates interior in only 3 and 5), median log errors 0.36 and 0.48 |
| H20 | **met** (post hoc) | Shanghai, 87 subjects, five repeats: grad3 - grid3 -0.85 [-5.5, 3.6] (90% interval inside +/-90); grad3 - grad1 -54 [-94, -30] (90% interval -86 to -33, inside +/-90 and +/-150); grad3 - personal mean -38 [-163, 103]; cell means 3,854 (grad3), 3,908 (grad1), 3,892 (personal mean), 4,220 (population), 5,535 (persistence) |
| H21 | **not met** | gap (Adam estimate minus polished) below 1.92 in 100% (iAUC), 100% (centroid) and 60% (trace); median gap 0.59 and largest 41.2 under the trace. Polished H11 clauses: trace tau1 83%, p 88% (interior); centroid 23% and 8% |
| H22 | **met** | Dalla Man, 45 subjects: kabs, kmax, kmin bounded in 0 of 45 (estimates on a bound in 40, 31, 34); Vmx bounded in 2 of 45 (4%); median Fisher effective rank 2 of 6, median condition number 8.8e12; median polish gap 0.009 |

## What the new results do to the paper

1. **The combination claim failed its controls.** The Phase 8 reading that (tau1, p) is recoverable from the
   trace does not survive: a true-model replica gives fewer bounded intervals than the real data (H18), those
   intervals cover the truth in 27% and 50% of cases, and 40% of the trace fits are not optima (H21). The
   ordering of observables (trace above centroid above area) is reproduced; the absolute trace fractions are not
   evidence of identifiability. The manuscript (abstract, introduction table, Section 5.4, Discussion) now says
   "not established", reports the failed control, and lists candidate explanations as untested (the chi-square
   calibration near the tied boundary p = tau1^2/4, noise-model misspecification for the trace).
2. **S_I is inherently weak at this noise level** (H17, H19), more strongly than Phase 8 alone showed.
3. **Prediction equivalence replicates** (H20), with one detail the paper states: in Shanghai the
   three-parameter fit is better than the S_I-only fit by 54 [30, 94], inside the margin but not zero, so the
   claim that the timing parameters add nothing was softened to "add less than the margin".
4. **The second model class** (H22) supports a narrow claim only: gut parameters not identified under iAUC in
   two structurally different models. No replica was run for it.
5. **The gradient diagnostic** (H6 met, H10 not met): reliable against profile strength, ordering of the two
   timing rates unstable across initializations, consistent with the exchange symmetry.

## Report-only items (declared in Amendment 5)

* Hall, trace objective, 1x box: S_I bounded in 3 of 5 interior subjects, k_e in 1 of 11, k_a in 0 of 8.
* Shanghai 2x profile re-run covers all 87 subjects (Phase 8: 85); S_I bounded in 45% of all subjects at 2x
  (7% at 1x); k_e, k_a bounded in 0%.
* Holm-adjusted correlation p-values are in `generated/correlates.tex`; the carbohydrate-scale reference values
  carry the corrected sign (C1).
* Interval-level sensitivity (`delta_sensitivity`): see `paper/generated/levels.tex`; S_I among interior subjects
  at the primary box is 23%, 14%, 0% at the 90, 95, 99% levels; k_e and k_a 0% at every level.

## Not run (declared in Amendment 5)

Heteroskedastic-noise sensitivity of the S_I profile and the design sweep (A15). The paper says so
(`app_register.tex`, `sec_discussion.tex`).

## Still to do before submission

* Place the official `tmlr.sty` and `tmlr.bst` in `paper/` and rebuild (the build uses a fallback article class
  while they are absent; overfull boxes in wide tables need a check under the real style).
* Citations: `kersting2020differentiable` and `beck2024diffusion` checked against the venue and author lists;
  GlucoFM added (arXiv 2605.30865, Li et al.). The reviewers' "HIPNO" could not be matched to a glucose paper:
  the only HIPNO found is a hemodynamics paper (arXiv 2608.10011), which is not cited. Ask which paper was meant.
* Commit `PREREG_AMENDMENT_5.md` is already in history; commit the Phase 9 code, manuscript and reports; keep
  `results/` ignored and archive the frozen `index.json` hashes.
* Rebuild the anonymized package (`evaluation/package_anonymized.py`).

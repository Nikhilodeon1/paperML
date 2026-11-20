# Pre-registration amendment 13 (Phase 17, TMLR revision, grid resolution of the profiles)

Amends `PREREG.md` and amendments 1 to 12. To be committed before any Phase 17 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 17 results are added beside them.

## What was seen before this amendment, stated plainly

Everything through the ninth freeze (Phase 15 and Phase 16 results, `results/phase15_summary.json` and
`results/phase16_summary.json`), in particular:

* H31 met, H32 met, H33 met. At a quarter of the noise the profile excess at the true value, computed directly at that value,
  was within 1.92 in 45 of 45 subjects for tau1 and 44 of 45 for p, whether the inner solve used Adam or L-BFGS-B and whether it
  started at the estimate or at the truth (the median difference between solvers and between starts was below 0.001). The
  stored grid read 5.3 and 3.4 at the median, so about 4.8 and 3.1 of that is interpolation across the 15-point grid.
* With a converged inner solver on the 15-point grid the real-data bounded fractions and every profile class were unchanged
  (0 of 90 classes changed in each of the four settings). On the 29-point grid at a quarter of the noise, nominal coverage was
  0.91 for tau1 and for p, and the bounded fractions were 80% and 71% against 61% and 60% on the 15-point grid.
* The Adam and L-BFGS-B profile values agree to about 1e-4 at the median; in a minority of points one solver is lower than
  the other (up to 32 in a few real-data points where the continuation of L-BFGS-B stayed at a worse inner optimum).
* Phase 15, complete: H29 not met; at 0, 5, 10 and 20% residual `k_e` was bounded in 1 of 32, 1 of 28, 4 of 31 and 12 of 31
  interior subjects, and `k_a` in 1 of 36, 0 of 35, 2 of 31 and 7 of 31, with coverage 0.98 to 1.00 throughout; on seed 1 at 10%
  the counts were 5 of 33 and 2 of 22.

All of this is **post hoc**. Each rule below is fixed before any Phase 17 unit is run.

## Why

The 29-point result at a quarter of the noise raises one question the first twelve amendments cannot answer: the bounded
fraction of a profile depends on how many grid points fall inside its narrow band, so every bounded fraction in the paper was
computed on a 15-point grid and could depend on that choice. At a quarter of the noise the 15-point grid understated the
bounded fractions by about 20 points. If the real-data fractions or the positive control of H29 have the same dependence, the
manuscript has to say so. Phase 17 recomputes them on a 29-point grid with the same Adam inner solver that produced the
published numbers.

## New hypotheses

**H34 (the real-data bounded fractions do not depend on the grid).** Real-data trace coordinate profiles (CGMacros, 45
subjects, primary box, polished estimates, Adam inner solver at 100 steps) on a **29-point** grid against the 15-point grid of
the paper. Rule: the bounded fraction among interior subjects differs by at most **10 percentage points** for tau1 and for p.
Met means the headline real-data fractions do not depend on the grid resolution at this level. Not met means they do, and the
manuscript reports both and states which one it treats as the primary reading.

## Report-only analyses (no rule)

**R15. The same grid comparison on the replicas.** Exact carbohydrate at the estimated noise and carbohydrate error 0.25, tau1
and p, 15 points against 29: bounded fractions among interior subjects, nominal coverage, coverage of the bounded intervals,
and the number of profile classes that change.

**R16. The positive control on a finer grid.** The residual-gut replica of H29 (random truths, seed 0, exact carbohydrate, the
estimated noise) at **0% and 10%** residual content, trace profiles in rate coordinates as in Amendment 11, on a 29-point grid
against the 15-point grid: bounded fractions of `S_I`, `k_e` and `k_a` among interior subjects and coverage of the nominal
interval. The verdict on H29 is not revisited, whatever this shows.

## What is not added

No new inner solver, replica or objective. The `k_e` and `k_a` profiles are not recomputed with L-BFGS-B. Grid sizes other
than 15 and 29 are not tried.

## Procedure

`scripts/pod_phase17.sh` runs the five profile runs: real data, exact carbohydrate and carbohydrate error 0.25 (coordinate
profiles, 29 points), and the residual-gut replica at 10% and 0% (rate coordinates, 29 points, `extension_points` 0). Resumable,
per-unit limit 7,200 s. The summary is `evaluation/phase17_summary.py` (`python -m evaluation.make_paper_assets --phase17
--latex` builds it). A tenth freeze follows. No threshold is changed after the first Phase 17 unit is run. Result files record
`grid_points`, and result selection separates sets by it (Amendment 12).

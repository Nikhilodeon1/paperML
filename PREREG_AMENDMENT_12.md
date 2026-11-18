# Pre-registration amendment 12 (Phase 16, TMLR revision, inner solver of the profiles)

Amends `PREREG.md` and amendments 1 to 11. To be committed before any Phase 16 analysis is run. No threshold from an
earlier document changes, and earlier verdicts are not edited: Phase 16 results are added beside them.

## What was seen before this amendment, stated plainly

Everything through the eighth freeze (Phase 15 results, `results/phase15_summary.json`), in particular:

* H29 not met: at 10% residual intestinal content, `k_e` and `k_a` were bounded in 4 of 31 and 2 of 31 interior subjects
  (13% and 6%), with coverage 100%. H30 met: the stored polished estimate was within delta of the optimum found from the
  truth in 45 of 45 subjects (largest gap 0.0006). R10: half the likelihood ratio at the true parameters had median 0.74,
  1.02 and 1.60 for exact carbohydrate, a quarter of the noise and carbohydrate error 0.25, against a median of 1.18 for half
  a chi-square with three degrees of freedom, and exceeded 1.92 in 13%, 20% and 38% of subjects.
* R12: scaling the truth's `insulin_secretion` or its `x_gain` by the same factor gave the same `S_I` estimate in 44 of 45
  subjects, so the two perturbations are one experiment (a change of the scale of insulin action). Coverage of the nominal
  interval for `S_I` was 0.87 on the unperturbed replica, 0.91 at 1.3 and 0.64 at 0.7.
* **Post hoc, from stored data and in no rule:** at a quarter of the noise the profile excess at the true value read off the
  stored Phase 14 coordinate grid (median 5.3 for tau1 and 3.4 for p; above 1.92 in 84% and 80% of subjects) is far above the
  likelihood ratio at the true parameters (median 1.0; above 1.92 in 20%). With exact carbohydrate at the estimated noise and
  with carbohydrate error 0.25 the two agree. A profile value cannot exceed the likelihood at a feasible point that has the
  same coordinate, so the stored excess at the truth is overstated at low noise, and the Phase 14 coverage of 16% and 20%
  (H28) is not a property of the likelihood.
* Two one-subject smoke tests of the code written for this amendment (CGMacros-001, a quarter of the noise): the excess at
  the true value of tau1 was 3.30 from the stored grid and 0.13 from a direct solve at the true value (Adam and L-BFGS-B
  agreeing to six digits); and the converged-solver profile ran to the end, with 15 of 15 and 14 of 15 grid points flagged as
  converged. They show that the code runs and are not used as results.

## Why

Phase 15 left one question that decides how much of the low-noise calibration story survives: the stored profile at the true
value is above the likelihood at the true value, so something in the construction of the profile (the interpolation between
the 15 grid points, the fixed-step inner solver, or the starting point of the inner solve) overstates it. The same
construction produces the bounded fractions of the real-data results. A reviewer can fairly ask whether those fractions are a
property of the data or of an inner solver that stops after a fixed number of Adam steps. Phase 14 showed that more Adam
steps change no profile class, which does not show that Adam reaches the inner minimum. Phase 16 answers both questions with
a converged solver.

## New hypotheses

**H31 (the true value is inside the band when the profile is computed directly).** Replica seed 0, exact carbohydrate, CGM
noise at a quarter of its estimate, the stored Phase 14 coordinate-profile results (500 inner steps) as inputs, 45 subjects.
For each subject and for each of tau1 and p, the profile is computed at the true coordinate alone by a converged L-BFGS-B
solve of the other two coordinates started from the polished estimate (`evaluation.profile_at_truth`, `lbfgsb_hat`), and its
excess is taken above the reference of the stored profile. Rule: the excess is at most **1.92** in at least **90%** of
subjects, for tau1 and for p. Met means a correctly computed profile covers the truth at low noise and the Phase 14
undercoverage belongs to the grid construction. Not met means that, even computed directly from the estimate, the profile
leaves the truth outside the band, and the manuscript says so (a likelihood problem, or an inner optimum that the estimate's
start does not reach).

**H32 (the real-data bounded fractions do not depend on the inner solver).** Real-data trace coordinate profiles (CGMacros,
45 subjects, primary box, polished estimates, 15 grid points), tau1 and p profiled by the converged solver
(`inner_solver` "lbfgsb": L-BFGS-B at every grid point, started from the nearest solved neighbour on the side of the
estimate), against the default Adam profiles at 100 inner steps. Rule: the bounded fraction among interior subjects differs
by at most **10 percentage points** for tau1 and for p. Met means the headline real-data fractions are not an artifact of the
fixed-step solver. Not met means they are, and the manuscript reports the converged fractions beside the Adam ones.

**H33 (a converged solver and a finer grid restore calibration at low noise).** Replica seed 0, exact carbohydrate, a quarter
of the noise, tau1 and p profiled by the converged solver on **29** grid points in place of 15. Rule: the nominal interval
(excess at the true value, interpolated on the grid as in H28, below 1.92) contains the true value for at least **90%** of
subjects, for tau1 and for p. Met means the grid-based interval of the paper is calibrated at low noise once the grid is fine
enough. Not met means grid interpolation or the solver is still not enough at 29 points.

## Report-only analyses (no rule)

**R13. The profile at the true value, four ways.** For the exact-carbohydrate replica at the estimated noise, at a quarter of
it, and with carbohydrate error 0.25: for tau1 and p, the share of subjects above 1.92 and the median of (i) the excess
interpolated from the stored 15-point grid, (ii) the stored Adam inner routine (500 steps, from the estimate) run at the
true value alone, (iii) the converged solve from the estimate, (iv) the converged solve from the true free coordinates; and
the medians of the three differences (i) minus (ii), (ii) minus (iii) and (iii) minus (iv), which are the share due to the
grid interpolation, to the fixed-step solver and to the starting point.

**R14. The converged-solver profiles.** For the real data and for the three replicas of R13, tau1 and p: bounded fractions
among interior subjects, nominal coverage, coverage of the bounded intervals, the number of grid points the solver reports as
not converged, and the number of profiles whose class (bounded, one-sided, flat) differs from the default Adam profile, with
the direction of the change.

## What is not added

The gut-rate coordinates `k_e` and `k_a` are not recomputed with the converged solver: H29 is closed, and Phase 15 already
reports them as not bounded under a broken symmetry. A fresh positive control (a liquid and a solid meal, a day-continuous
replica, a replica for the second gut model) remains a limitation of the manuscript. No new prediction experiment is run.

## Procedure

`scripts/pod_phase16.sh` runs, in this order: the three profile-at-the-truth runs (A19), then the converged-solver profile
runs: a quarter of the noise at 29 grid points, the real data, a quarter of the noise at 15 points, exact carbohydrate, and
carbohydrate error 0.25. It must run on the disk that holds the Phase 14 results (the inputs of A19). Resumable, per-unit limit
7,200 s. The summary is `evaluation/phase16_summary.py` (`python -m evaluation.make_paper_assets --phase16 --latex` builds
it). A ninth freeze follows. No threshold is changed after the first Phase 16 unit is run, and no hypothesis is rescued by
changing code or thresholds. Result files record `inner_solver` and `grid_points`, and result selection separates sets by
them; payloads that predate this amendment count as Adam with 15 points.

The first eleven amendments' code paths are unchanged by default: `inner_solver` defaults to "adam" and `grid_points` to 15.

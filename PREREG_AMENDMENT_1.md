# Pre-registration amendment 1

Amends `PREREG.md` (committed at `421b657`). Committed **before any Phase 2 or later analysis was
run**: at the time of this commit `results/` contained only `A0_benchmark` and `A0_scaling`, both of
which are compute measurements and neither of which is an analysis of the hypotheses. No Phase 2
result had been produced, and none had been viewed. The thresholds in `PREREG.md` are unchanged; this
amendment adds two hypotheses and records three method corrections that change results and were
therefore adopted before any result existed.

## New hypotheses

**H11 — the identifiable combination.** In coordinates `(log S_I, log tau1, log p)` with
`tau1 = 1/k_e + 1/k_a` and `p = 1/(k_e k_a)`, the profile-likelihood 95% CI for `tau1` is bounded in
at least **70%** of interior-converged subjects under iAUC+centroid and under the full trace. `p` is
bounded in at most **30%** of subjects under iAUC+centroid and in at least **50%** under the full
trace.

**H12 — the tied-rate model loses nothing on prediction.** A two-parameter model
`(S_I, tau1)` with `k_e = k_a = 2 / tau1` has held-out iAUC MAE within the **+/-150 mg/dL*min**
equivalence margin of the three-parameter model (paired, subject-level), and its `tau1` profile is
bounded in at least **70%** of subjects under iAUC+centroid.

H11 and H12 are secondary endpoints. The primary endpoints remain H1, H3, H5 under iAUC, H7 and H9.

## Method corrections adopted before any Phase 2 analysis

These change numbers. Each is recorded here rather than applied silently.

**B1 — maximum-likelihood estimate for every identifiability analysis.** `theta_hat_ML` is fitted
with `lambda = 0`, warm-started from the regularized fit, and is what A4 (Fisher), A5 (profile
likelihood), A6 (the ladder), A16 and the interior-versus-bound classification use. The regularized
fit is used **only** for prediction (A9). The Phase 2.5 gate check that the profile minimum sits at
`theta_hat` refers to `theta_hat_ML`.

*Why:* a penalty toward the population parameters shrinks every profile, so a regularized estimate
makes parameters look better identified than the likelihood supports. Reporting identifiability from a
regularized fit would be the first thing a reviewer checked.

**B2 — inverse residual variance, not observed variance, as the per-observable weight.** Two passes
per subject: a pilot fit, then `sigma_k^2` estimated for each observable `k` from the residuals at the
pilot estimate, then a refit with those weights frozen. The Fisher matrix and the profile likelihood
use block-diagonal weights with the same `sigma_k^2`.

*Why:* the condition number of the Schur-complement timing block, which is the H3 metric, depends on
the relative weighting of the observables. Weighting by the spread of the observed values rather than
by the spread of the residuals would make H3 an artifact of how variable the cohort happens to be.

**B3 — AR(1) whitening of the trace objective.** CGM residuals are strongly autocorrelated. The trace
objective, its Fisher matrix and its profile likelihood use Prais-Winsten whitening: `rho` and the
innovation variance are estimated per subject from pilot residuals, then frozen. The distribution of
`rho` is reported. Held-out trace RMSE is still computed on **unwhitened** residuals, because it is a
prediction-error summary rather than a likelihood.

*Why:* about 36 samples per meal that are nearly a random walk carry far less information than 36
independent ones. Treating them as independent would overstate what the trace reveals and bias both
H3 and H5 in favour of the trace rung -- which is the direction that would most flatter the paper.

## Scope changes that do not affect any hypothesis

Recorded for completeness; neither alters a threshold or a decision rule.

* **Multistart is a full-data analysis, not a cross-validation cell.** Ten random initializations are
  used for the inference gap in A10 only, and `multistart grad3` is removed from the A9 cell list.
  H7 and H8 never referred to it.
* **Grid cells are precomputed per meal.** The loss is additive over meals, so every meal is simulated
  once per grid point and each fold is a sum over the right subset. This is an identity, not an
  approximation. Grid resolution stays 20 x 8 x 8 unless the measured projection exceeds about two
  hours across all subjects, in which case 12 x 6 x 6 is used and reported.

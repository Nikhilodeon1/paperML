# Pre-registration: AISTATS revision, identifiability of glucose-insulin parameters from postprandial CGM

Committed before any Phase 2 or later analysis is run. **After this commit the thresholds and
decision rules below are frozen.** If a threshold later looks wrong, the run stops and reports;
the threshold is not edited. A hypothesis that fails is recorded as failed in the results and in
the phase report. Code and thresholds are never tuned to rescue a prediction.

Free parameters throughout: insulin sensitivity `S_I`, gastric emptying `k_e`, carbohydrate
absorption `k_a`, with the bounds declared in `personalization/gradient_fit.py` and exported
verbatim to `SPEC_EXPORT.md`.

## Hypotheses

**H1 — Full-trace structural analysis.** `S_I` identifiable; `k_e` and `k_a` locally but not
globally identifiable (swap). Decision rule: report the tool output verbatim. If `k_e` and `k_a`
are globally identifiable, delete all swap framing from the paper.

**H2 — Area observable nearly blind to timing.** In the linearized model with a 6 h window, the
ratio `|d iAUC / d log k|` to `|d iAUC / d log S_I|` is at most 0.02 for both `k_e` and `k_a`. In
the nonlinear model with the 3 h window, the median of that ratio across subjects is at most 0.25.

**H3 — Timing information increases along the observable ladder** iAUC, then iAUC+centroid, then
full trace. Metric: condition number of the Schur-complement timing block of the Fisher matrix
(log coordinates, after profiling out `S_I`). Prediction: medians decrease at each step. Test:
paired Wilcoxon per step, Holm corrected, alpha 0.05.

**H4 — Direction of the weak mode.** For iAUC+centroid, the weak eigenvector of the timing block
has median absolute cosine at least 0.8 with `(1/k_a, -1/k_e)` normalized, in log coordinates.

**H5 — Profile-likelihood 95% confidence intervals.** The fraction of subjects with a bounded CI
is at most 25% for `k_e` and for `k_a` under iAUC, and at least 50% under full trace; for `S_I` it
is at least 80% under iAUC among interior-converged subjects.

**H6 — Gradient diagnostic agrees with profile likelihood.** Median per-subject Spearman
correlation between parameter rankings at least 0.7; AUC at least 0.85 for detecting
profile-flat parameters in the synthetic study.

**H7 — Optimizer irrelevance.** Subject-level paired differences in held-out iAUC MAE, gradient
(3 parameters) versus exhaustive 3-D grid, and gradient (`S_I` only) versus 1-D grid, have a 90%
bootstrap CI inside (-150, +150) mg/dL*min. Sensitivity margins 90 and 300 are also reported.

**H8 — Extra parameters add nothing under iAUC.** Gradient (3 parameters) versus gradient (`S_I`
only), same equivalence test as H7.

**H9 — Objective matters.** A full-trace fit lowers held-out iAUC MAE relative to an iAUC fit by
more than 150 mg/dL*min (paired Wilcoxon, Holm). If it does not, report "ceiling persists across
observables for this model".

**H10 — Diagnostic robustness.** Kendall tau at least 0.8 for the parameter ordering across 5
initializations, 3 bound settings, log versus linear parameterization, and Adam versus L-BFGS;
`S_I` ranked first in at least 95% of interior-converged subjects.

## Primary endpoints

H1, H3, H5 (iAUC), H7, H9. The remaining hypotheses are secondary.

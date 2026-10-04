# Phase 2 report (gate G1), with the Phase 4c prediction results

Every number below is in `results/phase2_summary.json`, produced by `python -m evaluation.phase2_summary`
from stored results only. Compute ran on a 32-core node: the full queue finished with no failed unit.
The analysis plan was fixed in version control before the analyses were run; deviations are
`PREREG_AMENDMENT_1.md`, `_2.md` and `_3.md`, and `paper/appendix_deviations.tex`. No threshold was
changed. Nothing was tuned to rescue a hypothesis.

## Verdicts against the plan

| | endpoint? | result | verdict |
| --- | --- | --- | --- |
| H1 swap structure | primary | planned formulation: `k_e`, `k_a` global; empty-gut model: exact exchange symmetry | **formulation-dependent, not met** (deviation, Amendment 3) |
| H3 timing information rises along the ladder | primary | at each objective's own estimate the median condition number *rises* from iAUC to iAUC+centroid in all three boxes | **not met** |
| H4 weak direction along `(1/k_a, -1/k_e)` | secondary | median abs cosine 1.00 for iAUC+centroid, 100% of subjects above 0.8, all boxes | **met** |
| H5 timing parameters unbounded under iAUC | primary | `k_e`, `k_a` bounded in 0 of 45 subjects in every box (Wilson upper limit 8%) | **met** |
| H5 `S_I` bounded in at least 80% (interior for `S_I`) | primary | 0/15 (0.5x), 3/22 = 14% (1x), 24/40 = 60% (2x) | **not met** at any box |
| H5 timing bounded in at least 50% under the trace | primary | profile not run for the trace | **not evaluated** |
| H7 optimizer irrelevance | primary | gradient vs grid, both pairs equivalent at margins 90, 150, 300 | **met**, robust to 500 steps |
| H8 extra parameters add nothing | secondary | `grad3` - `grad1` = -34 [-68, -12], equivalent at 90 | **met** (but `grad3` is slightly better) |
| H9 trace lowers held-out iAUC error by more than 150 | primary | trace is *worse* by +32 [+2, +67] | **not met**: ceiling persists |
| H2, H6, H10, H11, H12 | secondary | not run | **not evaluated** |

Primary endpoints: H7 met; H1 formulation-dependent; H3, H5 (`S_I` clause) and H9 not met.

## 1. Bounds: the stop rule and the box sweep

The stop rule "more than 20% of subjects on a bound for `S_I`" fires at two of three boxes. Maximum-likelihood
fit, iAUC objective, 45 subjects:

| | 0.5x | 1x (primary) | 2x |
| --- | --- | --- | --- |
| `S_I` on a bound | 30 (67%) | 23 (51%) | 5 (11%) |
| `k_e` on a bound | 45 (100%) | 40 (89%) | 26 (58%) |
| `k_a` on a bound | 44 (98%) | 39 (87%) | 34 (76%) |
| interior for `S_I` (Amendment 2) | 15 | 22 | 40 |
| interior-converged, original definition | 0 | 0 | 0 |

The original definition gives an empty set in every box, so the clause as first written is unevaluable.
Pinning of `S_I` falls steeply with box width; pinning of `k_a` does not (98% to 76%). Unlike the shorthand
used in planning, `k_e` also falls (100% to 58%), so "pinned in every box at 75-89%" is not what the data show.

Optimizer check (A4b): the Adam estimate's negative log-likelihood is within 0.006 (0.5x), 0.25 (1x) and
1.03 (2x) of a multistart L-BFGS in log coordinates, always below the 1.92 threshold, and within 0.1 for
100%, 96% and 89% of subjects. The stored estimates are likelihood optima at the resolution that matters.

Fisher spectrum in log coordinates, medians (1x): 3.9e-08, 7.1e-03, 14.8. Three orders of magnitude of
spread are between the middle and the largest eigenvalue, and about five between the middle and the smallest.

## 2. Gradient diagnostic with bound projection (2.2)

The separation of `S_I` from the timing parameters **survives** projection and survives on the interior
set. At 1x, on the 22 subjects interior for `S_I`, the projected diagnostic ranks `S_I` first in 100%
(86% unprojected) and the median timing-to-`S_I` ratio is 0.006. At 2x on the 40 interior subjects: 97%.
Raw gradients overstate the timing parameters, as expected; projection widens, not narrows, the gap.

## 3. Generic rank (2.3)

Full rank (3) at tolerance 1e-6 for every observable set, in 198-200 of 200 parameter draws (iAUC: 198).
The distinction is conditioning, smallest over largest singular value: iAUC 8e-5, iAUC+centroid 4e-3,
iAUC+peak 6e-3, trace 2.5e-2. So iAUC is **locally identifiable but ill-conditioned**; the exchange
symmetry of the empty-gut model is the only structural statement the analyses support.

## 4. Structural identifiability (2.4) and H1

All seven runs, with inputs and tool versions, are in `julia/SI_RUNS.md`. `S_I` is globally identifiable
in every run where the constants are fixed. `k_e` and `k_a` are global only when the intestinal initial
state is a generic known value (run 1 and the planned formulation) and local otherwise; in canonical
coordinates `k_e + k_a` and `k_e k_a` are global, which leaves the unordered pair. The engine itself:
exchanging the rates changes glucose by at most 0.009 mg/dL for an empty gut. Residual gut content breaks
the symmetry linearly:

| initial gut content, % of meal | 0 | 0.5 | 1 | 2 | 5 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| max glucose change (mg/dL) | 0.009 | 0.60 | 1.2 | 2.4 | 5.9 | 11.8 |
| median (mg/dL) | 0.003 | 0.32 | 0.63 | 1.3 | 3.1 | 6.3 |

About 5% residual content is enough to exceed CGM noise, so the exchange symmetry is a property of the
idealized model that free-living data would partly break.

## 5. Profile likelihood (2.5), iAUC objective

Fraction of subjects with a bounded 95% interval (`n = 45`, Wilson in brackets):

| | 0.5x | 1x | 2x |
| --- | --- | --- | --- |
| `S_I` | 0 [0, 0.08] | 3 = 7% [0.02, 0.18] | 24 = 53% [0.39, 0.67] |
| `k_e` | 0 [0, 0.08] | 0 [0, 0.08] | 0 [0, 0.08] |
| `k_a` | 0 [0, 0.08] | 0 [0, 0.08] | 0 [0, 0.08] |

For `S_I` most subjects are *one-sided* (rises on one side only: 34, 38, 21): the interval extends beyond the
box. **The bounded fraction depends on the box**, which is the opposite of what a purely identifiable
parameter would do, and the H5 clause for `S_I` fails at every box. The timing parameters are flat in 82-93%
of subjects and one-sided in the rest, in every box. The diagnostic extension to four times the upper bound
leaves 39-41 of 45 flat at 1x, so the flatness does not end just outside the physiological range.

Quality of the profiles (reported, not hidden): no profile point lowered the loss below the stored estimate
by more than 0.66 (1x) or 1.03 (2x), always under the 1.92 threshold. Inner optimizations still falling at
the end of the run are absent at 0.5x and 1x and appear at 2x (`S_I` 27 grid points of 675, `k_e` 10, `k_a` 9;
8, 11 and 7 subjects have the profile minimum away from the estimate). The 2x `S_I` figure is therefore the
least certain of the nine, and an unsettled point overstates a profile, which flatters identifiability.

## 6. Pinned subjects are not only a flat likelihood (flag from the plan)

Subjects with `k_e` or `k_a` at the upper bound (1x: 32 and 33 of 45) have much larger observed responses
(mean observed iAUC about 5,600 vs 2,600, p < 0.001), are under-predicted in held-out data by about 1,000
mg/dL*min (CI -1,438 to -705, against -84 for the others, p = 0.002) and eat marginally more carbohydrate
(55.6 vs 53.3 g, p = 0.07). At 2x the pattern is the same (-781, CI -1,047 to -562). So the pinning is
partly an area deficit: the fit pushes timing to the fast limit because it cannot reach the observed
excursion. A flat likelihood and a mis-specified area are both present. "Bounds bite in every box" is not
written as a finding.

## 7. The observable ladder (3.1, H3, H4)

Median condition number of the Schur-complement timing block:

| | iAUC | iAUC+centroid | trace |
| --- | --- | --- | --- |
| 1x, own estimate | 4.3e5 | 2.8e7 | 8.8e7 |
| 1x, common reference | 3.3e5 | 8.7e4 | 8.1e2 |
| 0.5x, own estimate | 2.3e5 | 4.0e5 | 6.4e3 |
| 0.5x, common reference | 2.3e5 | 2.1e5 | 1.2e3 |
| 2x, own estimate | 7.8e3 | 5.0e14 | 1.1e14 |
| 2x, common reference | 7.8e3 | 1.0e5 | 4.1e2 |

At each objective's own estimate the centroid step goes the wrong way in every box (adjusted p < 0.001),
because those fits sit on bounds where the information matrix is nearly singular (2x: 1e14). At the common
reference the trace is consistently two orders of magnitude better conditioned than iAUC, and the centroid
step is in the predicted direction only at 0.5x (adjusted p = 0.0008; at 1x p = 0.054). H3 is **not met**
as pre-registered (own estimate, 1x). The robust statement is the second one: the full trace carries about
two orders of magnitude more timing information than the area, at any box, and the centroid adds none that is
reliable.

## 8. Prediction (A9, 45 subjects, five repeats, 5-fold, 150 steps)

Held-out iAUC MAE (mg/dL*min), mean over subjects: gradient three parameters 3025, exhaustive grid three
parameters 3026, gradient `S_I` only 3059, grid `S_I` only 3059, trace objective 3057, random forest 3154,
personal mean 3257, population 3359, SNPE 3420, persistence 4508. Spread across fold assignments is a median
of 37 within a subject. Gradient three parameters beats the personal mean by 232 [97, 398] (adjusted p =
0.032, 35 of 45 subjects) and the SNPE by 395.

Optimizer (H7): gradient vs grid, three parameters: -1.1 [-2.9, +0.7]; `S_I` only: 0.0 [-0.5, +0.5]. Both
equivalent at 90, 150 and 300. The inference gap confirms why: the gradient training loss is within 1% of the
grid minimum in 96.6% of folds (`S_I`-only: 100%), and lower than the grid's in 65%.
The 500-step check (135 units: `grad3`, `grad1`, repeats 0-2) leaves every equivalence verdict unchanged
(`grad3` vs `grid3` -2.2 [-4.4, +0.1]); H7 and H8 are robust to optimization budget.

Objective (H9): the trace objective does not improve held-out iAUC (+32 [+2, +67]). It improves its own
targets: trace RMSE 31.2 vs 35.9 and peak-time MAE 41.6 vs 47.5 minutes (personal mean: 32.0 and 41.0).
On peak time and trace RMSE the personal mean is as good as any model, so the model adds little there.

## 9. What was not run

The trace and centroid profile likelihoods (H5 trace clause, H11), the tied-rate model (H12), the moment
checks and window sweep (H2, Phase 3.2), the diagnostic-validation and robustness sweeps (H6, H10, Phase 5),
the inference-gap replica with noise (A10, beyond the loss-gap check above), synthetic recovery, replication
cohorts, design sweep, and Phase 7. None has a result. H2 and H6 in particular are needed before the paper
can make its diagnostic claims.

## Decision for the next step (gate G1)

1. The `S_I` clause of H5 fails and depends on the box. The paper cannot claim `S_I` is identifiable at
   the 80% level; it can claim that `S_I` is constrained from one side in most subjects and bounded in
   about half at the widest box, with the caveat in section 5.
2. The headline conclusion that survives every box is the timing parameters: flat, ill-conditioned, and
   needing the full trace to constrain. The prediction ceiling (H7, H9) is supported.
3. Before writing, run the trace and centroid profile likelihoods and the moment checks. These are the
   missing pieces for H5 (trace), H2 and H11, and take under an hour on the node used.

# Revision writeup: what changed since the submitted draft, and what the data now say

Status date: 2026-10-04. Everything below is computed from stored result files by code in this repository;
the numbers are in `results/phase2_summary.json` and the verdict table is `REPORTS/final.md`. The draft
that was submitted and rejected is the baseline. The revision work was set up against six reviewer
objections, and the build was framed as a revision for AISTATS; if the AAAI feedback you have in mind is
a different list, it needs to be checked against the six below, which are the only objections recorded in
the project.

## 1. Summary in one page

The submitted paper argued that insulin sensitivity `S_I` is identifiable from a postprandial CGM meal
response while gastric emptying `k_e` and carbohydrate absorption `k_a` are not, that prediction hits a
ceiling because the fitting target is the baseline-subtracted iAUC, and that the choice of optimizer does
not matter. The evidence was a gradient-magnitude diagnostic, the diagonal of a Fisher matrix, one fixed
fold assignment, and an error decomposition nobody could define.

The revision replaced every one of those with a defined, tested method and re-ran the analysis on the
45 CGMacros subjects. What survives:

* **`k_e` and `k_a` are not identifiable from iAUC, and the reason is exact.** Their exchange is an
  exact symmetry of the empty-gut model, the weak Fisher direction lies along the exchange direction in
  every subject (cosine 0.999), and the profile likelihood is flat for them in 82 to 93% of subjects at
  every box width, still flat out to four times the upper bound.
* **The prediction ceiling is real and not an artifact of the optimizer.** A gradient fit and an
  exhaustive grid search of the same loss predict held-out iAUC equally well (difference -1.1 mg/dL*min,
  90% interval inside +/-90), at 150 and at 500 optimizer steps.
* **A trace objective does not lift the ceiling** for held-out iAUC (it is slightly worse, +32), although
  the full trace is about two orders of magnitude better conditioned for the timing parameters.

What does not survive, or must be rewritten:

* **"`S_I` is identifiable" fails its own test.** Among subjects whose `S_I` estimate is interior, the
  profile interval is bounded in 14% (primary box), against the 80% the plan required. Most `S_I`
  intervals are bounded on one side only, and the bounded fraction depends on box width (0, 7, 53% across
  three widths). The honest claim is that `S_I` is constrained from one side and its identifiability
  depends on the prior range.
* **Timing information does not rise monotonically along the ladder** at each objective's own estimate
  (the centroid step goes the wrong way). It does rise at a common reference point, but only the full trace
  makes a large, consistent difference.
* **Pinned parameters are not only a flat likelihood.** Subjects with timing parameters at the upper bound
  have much larger responses and are under-predicted by about 1,000 mg/dL*min, so part of the pinning is an
  area deficit the model cannot reproduce.
* **The structural claim depends on a convention.** A structural-identifiability tool says `k_e` and `k_a`
  are globally identifiable when the gut starts at a generic nonzero state, and locally identifiable only
  otherwise. This is recorded as a deviation from the plan, not as a pass.

Of the five primary endpoints, one is met (H7), one is formulation-dependent (H1), three are not met
(H3, H5 `S_I` clause, H9). Nothing was tuned to change any of these.

## 2. What was wrong with the draft, and what replaced it

| reviewer point | what the draft did | what the revision does |
| --- | --- | --- |
| (1) gradient magnitude and a Fisher diagonal are local heuristics | argued non-identifiability from gradient size and the Fisher diagonal | full Fisher matrix with eigenvectors, Schur complement for the timing block, profile likelihood, generic rank, structural-identifiability runs, an analytic exchange symmetry |
| (2) the ceiling is confounded with the iAUC target | one objective | three-rung ladder (iAUC, iAUC + centroid, full trace) in Fisher, profile and held-out prediction |
| (3) mathematics under-specified | prose | the model, bounds, loss, observation operators and constants exported from the code to `SPEC_EXPORT.md` and a LaTeX table; observables defined in `simulation/jax_observables.py`; controllable canonical form |
| (4) the 79/21/0 decomposition was undefined | an additive decomposition | **not replaced yet** (see section 8); the loss-gap check on the folds gives one defined quantity |
| (5) held-out comparison uncontrolled, one fixed fold assignment | one fold assignment | five-fold cross-validation repeated five times, SHA-256 seeded folds identical for every method, paired bootstrap intervals, Holm correction, equivalence tests at three margins |
| (6) the `S_I` gradient may reflect an active bound | not examined | maximum-likelihood fits, a box sweep at three widths, a per-parameter interior definition, projected gradients, bound-aware profile likelihood |

## 3. Method changes

**Analysis plan fixed in version control before the analyses were run.** Ten hypotheses with thresholds
(`PREREG.md`), then three amendments, each listed with its reason in `paper/appendix_deviations.tex`:

* Amendment 1 (before any identifiability analysis): two secondary hypotheses (the identifiable
  combination `tau1`, a tied-rate model); unregularized maximum-likelihood estimates for every
  identifiability analysis; inverse-residual-variance weights from a pilot fit; first-order
  autocorrelation whitening of the trace objective; multistart moved to a full-data analysis.
* Amendment 2 (after the first Fisher results were seen): "interior" redefined per parameter, because the
  original definition gives an empty set whenever a non-identifiable parameter sits on a bound; 0.5, 1 and
  2 times boxes reported side by side; the original box kept primary for every cross-method comparison.
* Amendment 3 (after the structural-identifiability runs were seen): H1 labelled formulation-dependent;
  initial-gut-content sweep; a lint banning "structurally non-identifiable" outside the exchange-symmetry
  context; the optimizer-budget check declared in advance.

No numerical threshold changed.

**Engine and objectives.** A differentiable JAX/diffrax engine (float64) with a canonical-form variant
that never takes square roots of the rates, a linearized engine at the fasting fixed point, smooth and
hard versions of iAUC, centroid, peak time and trace, and a general fitting routine with a frozen noise
model. A bug found on the way: scaling the parameter box linearly gave a negative `S_I` lower bound at the
wide setting; the box is now scaled in log space.

**Computation.** A resumable runner (one result file per work unit, atomic writes, heartbeats, graceful
stop), a content-hashed cohort cache so a rented node needs no raw data, deterministic folds, and a
pipeline in which every number in the paper is a macro generated from a frozen, hashed result file, with
a lint that fails the build on a hand-typed number.

## 4. Structural identifiability

Seven runs of StructuralIdentifiability.jl (Julia 1.13.1, package 0.5.34), tabulated with inputs and raw
output in `julia/SI_RUNS.md`.

| setup | `S_I` | `k_e`, `k_a` |
| --- | --- | --- |
| meal as input, all initial conditions generic known | global | global |
| planned formulation (meal as an initial condition, intestinal state generic) | global | **global** |
| meal as input, gut initial conditions unconstrained or unknown | global | **local** |
| canonical coordinates `(k_e + k_a, k_e k_a)` | global | the two combinations are global, leaving the unordered pair |

The engine starts every meal with an empty gut, where the transfer function from meal rate to appearance
is `f k_e k_a / ((s + k_e)(s + k_a))`, exactly symmetric. Numerically, exchanging the rates changes simulated
glucose by at most 0.009 mg/dL. A nonzero intestinal initial content breaks the symmetry linearly
(`evaluation/gut_sweep.py`):

| residual gut content, % of the meal | 0 | 0.5 | 1 | 2 | 5 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| largest glucose change under the exchange (mg/dL) | 0.009 | 0.60 | 1.2 | 2.4 | 5.9 | 11.8 |

At about 5% residual content the exchange exceeds typical CGM noise, so in free-living data the symmetry
is partly broken. The planned rule ("if globally identifiable, delete the swap framing") fired for the
planned formulation, so H1 is reported as formulation-dependent rather than met.

**Generic rank is full** (rank 3 at tolerance 1e-6) for every observable set, so iAUC is locally
identifiable but ill-conditioned. Smallest over largest singular value: iAUC 8e-5, iAUC + centroid 4e-3,
iAUC + peak 6e-3, full trace 2.5e-2.

## 5. Why an area is blind to timing (H2, met)

For the gut input, the zeroth moment is 1 for every pair of rates (all carbohydrate arrives), the first is
`1/k_e + 1/k_a`, and the second depends on the pair only through that sum and `1/k_e^2 + 1/k_a^2`; verified
numerically and symbolically to 3e-16. So an area carries no information about how the arrival is split
between the two rates, only about the sum. Sensitivity ratio `|d iAUC / d log k| / |d iAUC / d log S_I|`,
median over 45 subjects:

| window after the meal (min) | 90 | 180 | 240 | 360 |
| --- | --- | --- | --- | --- |
| linearized engine, `k_e` / `k_a` | 1.56 / 1.73 | 0.12 / 0.17 | 0.04 / 0.06 | 0.004 / 0.007 |
| nonlinear engine, `k_e` / `k_a` | 1.13 / 1.25 | 0.097 / 0.141 | 0.015 / 0.017 | 0.055 / 0.065 |

H2 is met (below 0.02 linearized at six hours; below 0.25 nonlinear at three hours). The result depends on
the window: at 90 minutes the area is more sensitive to timing than to `S_I`, so a short window is
exactly where timing is visible. The nonlinear ratio is not monotone in the window.

## 6. Identifiability results (CGMacros, 45 subjects, iAUC objective)

Subjects whose maximum-likelihood estimate is on a bound, three box widths (the 1x box is the primary):

| | 0.5x | 1x | 2x |
| --- | --- | --- | --- |
| `S_I` | 67% | 51% | 11% |
| `k_e` | 100% | 89% | 58% |
| `k_a` | 98% | 87% | 76% |

The plan's stop rule (more than 20% of subjects on a bound for `S_I`) fires at two of three widths. The
Adam estimates are likelihood optima: against a multistart L-BFGS the negative log-likelihood is within
0.006, 0.25 and 1.03 at the three widths, always below the 1.92 interval threshold.

Fisher information in log coordinates: median eigenvalues 3.9e-8, 7.1e-3, 14.8. The weak direction has a
median cosine of 0.9986 (iAUC), 1.0000 (iAUC + centroid) and 1.0000 (trace) with the exchange direction
`(1/k_a, -1/k_e)`, in 100% of subjects: H4 is met.

Profile-likelihood intervals, fraction of subjects with a bounded 95% interval:

| | 0.5x | 1x | 2x |
| --- | --- | --- | --- |
| `S_I` | 0% | 7% | 53% |
| `k_e`, `k_a` | 0% | 0% | 0% |
| `S_I`, among subjects interior for `S_I` | 0 of 15 | 3 of 22 | 24 of 40 |

H5 holds for the timing parameters and fails for `S_I`. The extension to four times the upper bound leaves
39 to 41 of 45 timing profiles flat. Quality: no profile point improved on the stored estimate by more than
1.03 (below the 1.92 threshold); some inner optimizations had not settled at the 2x width (27 of 675
`S_I` grid points), which overstates a profile and so flatters identifiability. The 2x `S_I` figure is the
least certain number in the study.

**Bias check.** Subjects with `k_e` or `k_a` on the upper bound (32 and 33 of 45 at the primary width) have
larger observed responses (mean iAUC about 5,600 against 2,600) and are under-predicted by about 1,000
mg/dL*min (95% interval -1,438 to -705, against -84 for the rest). The pinning is partly an area deficit.

**Gradient diagnostic with bound projection.** Projecting out the components that push through an active
bound strengthens the separation: `S_I` is the largest in 100% of subjects (86% raw) at the primary width,
and the median timing-to-`S_I` ratio is 0.006. So the draft's Figure 1 survives the active-bound objection.

## 7. The observable ladder (H3)

Median condition number of the timing-block information:

| | iAUC | iAUC + centroid | trace |
| --- | --- | --- | --- |
| 1x, at each objective's own estimate | 4.3e5 | 2.8e7 | 8.8e7 |
| 1x, at a common reference point | 3.3e5 | 8.7e4 | 8.1e2 |
| 0.5x, common reference | 2.3e5 | 2.1e5 | 1.2e3 |
| 2x, common reference | 7.8e3 | 1.0e5 | 4.1e2 |

H3 is **not met**. At each objective's own estimate the centroid step goes the wrong way in all three
boxes, because those fits sit on bounds where the information is nearly singular. At a common reference
point the trace is consistently two orders of magnitude better conditioned than iAUC, but the centroid
step helps only at the narrowest width. The robust statement: the full trace carries far more timing
information than the area; the centroid adds none that is reliable.

## 8. Prediction (CGMacros, 45 subjects, five repeats of five-fold cross-validation)

Held-out iAUC mean absolute error (mg/dL*min): gradient three parameters 3025, exhaustive grid 3026,
gradient `S_I` only 3059, trace objective 3057, random forest 3154, personal mean 3257, population
default 3359, SNPE 3420, persistence 4508. These reproduce the draft's single-fold-assignment numbers
(3001, 3007, 3150, 3404) to within about 1%, so the earlier comparison was not an artifact of one
assignment; the repeats now give the spread (median within-subject standard deviation 37).

| question | result |
| --- | --- |
| H7 gradient vs grid, three parameters | -1.1 [-2.9, +0.7]; equivalent at margins 90, 150, 300 |
| H7 gradient vs grid, `S_I` only | 0.0 [-0.5, +0.5]; equivalent at all three margins |
| H8 three parameters vs `S_I` only | -34 [-68, -12]; equivalent at all margins, although `grad3` is slightly better |
| H9 trace objective vs iAUC objective | +32 [+2, +67]; does not lower held-out iAUC error: ceiling persists |
| gradient vs personal mean | -232 [-398, -97], 35 of 45 subjects, adjusted p = 0.032 |
| gradient vs random forest / SNPE | -129 [-220, -54] / -395 [-612, -254] |

The optimizer is not the reason for the equivalence: the gradient training loss is within 1% of the grid
minimum in 96.6% of folds (100% for `S_I` only), and 500 steps changes no verdict. The trace objective
improves what it targets (trace error 31.2 against 35.9 mg/dL; peak-time error 41.6 against 47.5 min), but
the personal mean (32.0 and 41.0) is as good as any model on those two, so the model adds little there.

## 9. Scorecard

| | endpoint | verdict |
| --- | --- | --- |
| H1 swap structure | primary | formulation-dependent (deviation) |
| H2 area blind to timing | secondary | met |
| H3 timing information along the ladder | primary | **not met** |
| H4 weak direction along the exchange direction | secondary | met |
| H5 `k_e`, `k_a` unbounded under iAUC | primary (iAUC) | met |
| H5 `S_I` bounded in at least 80% | primary (iAUC) | **not met** at all three widths |
| H5 timing bounded in at least 50% under the trace | primary | **not yet evaluated** (run on the node; archive not merged) |
| H7 optimizer irrelevance | primary | met, robust to 500 steps |
| H8 extra parameters add nothing | secondary | met |
| H9 trace lowers iAUC error by more than 150 | primary | **not met** |
| H6, H10 diagnostic validation and robustness | secondary | not run |
| H11, H12 identifiable combination, tied-rate model | secondary | not run |

## 10. What was not done

* **Reviewer point (4), the 79/21/0 decomposition,** is not replaced. The planned replacement (a
  well-specified replica with noise, reported as "the gap not explained by noise or optimization") was not
  built. The cleanest option for the paper is to drop the decomposition and report the quantities above.
* **Phase 5 (H6, H10):** diagnostic against profile likelihood, robustness sweeps across initializations,
  boxes, parameterizations and optimizers. Not run. The paper cannot claim the gradient diagnostic is
  validated; it can say the diagnostic's S_I-versus-timing separation survives bound projection.
* **Leakage intervals and clinical correlations (A13, A14), synthetic recovery, replication on Hall and
  Shanghai, design sweep (Phase 6):** not run. Every result here is CGMacros only.
* **H11 and H12:** coordinates and the tied-rate model were implemented and tested, but neither the
  profile in those coordinates nor the tied-rate prediction cells were run.
* **Trace and centroid profile likelihoods:** run on the compute node; the archive has not been merged, so
  the H5 trace clause and the profile rungs of the ladder are not yet in the numbers above.
* **The manuscript itself has not been rewritten.** What exists is the machinery: macros, 8 figures,
  6 generated tables, a deviations appendix, a verdict report, and an anonymized code package.
  Figure 5 (diagnostic against profile) is absent because Phase 5 was not run.

## 11. Limits that should be stated in the paper

One cohort (45 subjects, 5-minute CGM, carbohydrates logged by the participants). The 2x box result for
`S_I` rests on profiles that had not all settled. The primary-endpoint failures are reported as failures.
H1 depends on the gut initial-condition convention. About 5% of CGMacros meals have an observed iAUC of
exactly zero, which are kept for the area objective and excluded from the centroid.

## 12. Reproducibility

Every analysis is a resumable runner module writing one result file per work unit under
`results/<analysis>/<config hash>/`, with the code version, configuration and environment recorded.
Deterministic folds (SHA-256), a content-hashed cohort cache (no raw data needed on a node), an end-to-end
`reproduce_all_aistats` script with a three-subject smoke test (run successfully on the compute node),
`make_paper_assets` which fails if a number is typed by hand or a banned term appears, and an anonymized
package builder that refuses to write an archive if it finds a name, address, home directory or key. The
whole compute queue (225 cross-validation units, nine profile runs, six ladder Fisher runs, diagnostics, the
500-step check, the moment checks) ran in under 4 hours on a 30-worker node with no failed unit.

## 13. Next steps

1. Merge the trace and centroid profile archive, rebuild the assets, and complete the H5 trace clause.
2. Decide how the paper handles point (4): drop the decomposition, or build the replica.
3. Rewrite the manuscript around what the data support (section 1), including the deviations appendix.
4. Rebuild and test the anonymized package on the final results; look at the figures.
5. If time allows, Phase 5 (H6, H10), the most valuable missing piece for the diagnostic claims.

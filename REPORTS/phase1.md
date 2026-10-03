# Phase 1 report — observables, objectives, the generalized fit, and two new engines

Includes Patch 1. Every number below is measured.

## 0. Patch status

**No Phase 2 analysis had been run or viewed when the patch arrived.** `results/` contained
`A0_benchmark` and `A0_scaling` only, both compute measurements. `PREREG_AMENDMENT_1.md` was therefore
committed first, at **`c1845b6`**, before any of the corrections below were applied, and the
pre-registration status is clean.

What the patch changed in Phase 1 work that already existed:

| item | before the patch | after |
| --- | --- | --- |
| B2 | `iauc_centroid` standardized each term by the variance of the OBSERVED values | inverse residual variance, estimated at a pilot fit and frozen |
| B3 | trace residuals treated as independent | Prais-Winsten AR(1) whitening, `rho` frozen from the pilot |
| B1 | one regularized fit served everything | pilot (regularized) then `theta_hat_ML` (`lam = 0`, likelihood scale) for all identifiability work |
| D1/D2/D3 | rates only | canonical engine plus `(log S_I, log tau1, log p)` and the tied-rate model |
| C4 | `intra_op_parallelism_threads=1` removed from `XLA_FLAGS` as inert | reinstated as specified; the finding is recorded in the code comment |

C1 (precomputed grids), C2 (multistart out of the cross-validation cells), C3 (launch order) and C5
(Julia manifests) belong to Phases 2 and 4 and are not implemented here. C2 is already reflected in the
plan: the 51.6 h multistart cell is gone from the cross-validation list.

## 1.3 Regression against the submitted fit — PASS

Worst absolute parameter difference over 5 subjects x 3 parameters, new code against the untouched
`personalization/gradient_fit.py`:

**2.835e-07**, target < 1e-6.

### What the regression exposed

Reproducing the old result exactly required a `legacy_full_window` flag, because the submitted fit
integrated the **simulated** iAUC over the whole 0–210 min trajectory -- the 30 min pre-meal baseline
and a 30 min tail included -- while the **observed** iAUC was integrated over 0–180 min after the meal.
The two sides of the residual were not the same functional.

The effect is real but immaterial: **about 2.1 mg/dL*min**, 0.04% of a 60 g meal, nearly all of it the
smooth positive part accumulating its floor of `ln 2 / 10 = 0.069` mg/dL over the 30 min of pre-meal
samples. With the corrected window the worst parameter shift is 1.245e-03, which is three orders of
magnitude below the width of any bound. Reported rather than buried; corrected going forward.

A separate check: the new hard `iauc` operator reproduces the loader's own value to **1.1e-11** across
all 1640 CGMacros meals, so the bottom rung of the ladder is the quantity the submitted results were
fitted to.

## 1.4 Linearized engine — and the operating point that does not work

The first attempt linearized about the fasting fixed point and was **67% wrong even for a 1 g meal**,
with the error getting worse as the meal shrank. That is diagnostic rather than merely bad, and the
cause is worth stating because it constrains how H2 can be argued.

The engine smooths the insulin-secretion switch as `beta * softplus_relu(G - 90)` with a sharpness of 10
per mg/dL, so its **transition half-width is 0.1 mg/dL**:

| `G - 90` (mg/dL) | -0.1 | 0.0 | 0.1 | 0.5 | 2.0 | 10.0 |
| --- | --- | --- | --- | --- | --- | --- |
| gain, as a fraction of the active branch | 27% | 50% | 73% | 99% | 100% | 100% |

At the fasting point the gain is exactly half. A 1 g meal delivers 900 mg into a distribution volume of
about 95 dL, an excursion of order 9 mg/dL -- **ninety transition widths**. Every real meal spends its
whole excursion on the active branch, so linearizing at fasting gives a secretion gain three times too
small, under-disposes glucose, and roughly doubles the area. Making the meal smaller does not help: it
moves the excursion *into* the transition region.

`regime="active"` is therefore the default and the only regime used for H2. It takes the above-threshold
branch of the secretion softplus and shuts the fasting switch (already 0.007 at 15 uU/mL of insulin).
That is the classical linear minimal model, which is what the H2 proposition is a statement about.

### Verifying the Jacobian separately from the approximation

Two different questions were being conflated, so they are now answered separately.

**Is `A` correct?** Compared against the same vector field solved nonlinearly:

| meal (g) | nonlinear iAUC | linear iAUC | relative | error / meal² |
| --- | --- | --- | --- | --- |
| 0.01 | 1.330310 | 1.330388 | 0.006% | 0.785 |
| 0.1 | 13.296061 | 13.303883 | 0.059% | 0.782 |
| 1.0 | 132.266391 | 133.038827 | **0.584%** | 0.772 |
| 5.0 | 646.874056 | 665.194137 | 2.832% | 0.733 |
| 20.0 | 2412.731832 | 2660.776504 | 10.281% | 0.620 |

**0.584% at 1 g, inside the 1% target**, and the last column is the decisive evidence: the error is
constant per unit meal **squared**, which is the signature of a correct first-order expansion. A merely
small error could hide a wrong Jacobian; an error quadratic in the perturbation cannot.

**How far does linearity hold against the full engine?** 1.46% at 1 g, 2.9% at 5 g, 26% at 60 g. The
growth is the genuinely nonlinear `-x G` disposal term. Against the full engine the basal regime is 67%
wrong at 1 g where the active regime is 1.46%, and a test pins that gap so the default cannot be changed
back silently.

The linearization is stable, its residual at the operating point is **8.6e-16**, and the two gut rates
appear exactly as eigenvalues (−0.022 and −0.026 for a population-default subject), which is an
independent check that the gut block was linearized correctly.

## 1.5 One-subject Jacobian — the ladder, visible as numbers

`d observable / d log theta` at `theta_hat_ML`, CGMacros-003, meal 0 (81 g):

| observable | d/dlog S_I | d/dlog k_e | d/dlog k_a | \|k_e\|/\|S_I\| | \|k_a\|/\|S_I\| |
| --- | --- | --- | --- | --- | --- |
| **iauc** | −2003.45 | 151.48 | 266.14 | **0.076** | **0.133** |
| **centroid** | −15.54 | −18.61 | −20.54 | 1.198 | 1.322 |
| **peak_time** | −27.52 | −43.78 | −43.24 | 1.591 | 1.571 |
| **trace** (norm over 37 samples) | 90.31 | 90.78 | 102.22 | 1.005 | 1.132 |

This is the ladder argument as arithmetic rather than assertion. The area is dominated by insulin
sensitivity and sees the gut rates at roughly a tenth of the strength; the centroid and peak time invert
that; the trace is balanced. Both iAUC ratios are below the 0.25 that H2 predicts for the nonlinear 3 h
window, though one subject and one meal is not a result.

Caveat recorded with it: this subject sits on both rate bounds and the fit did not meet the convergence
criterion. The Jacobian is still well defined, but it is evaluated on the boundary.

## D1 Canonical engine — the square root never gets formed

The gut cascade is realized in controllable canonical form in `(sigma1, c) = (k_e + k_a, k_e k_a)`, which
has the same transfer function `f c / (s^2 + sigma1 s + c)` and is a polynomial in the parameters, so it
is analytic everywhere -- including at `k_e = k_a`, where recovering the rates by square root is not.

**Verification on 5 subjects x 2 meal sizes:** worst relative difference in glucose **3.76e-05** and in
`Ra` **3.30e-05**, both inside the 1e-4 target. The residual is solver tolerance (`rtol = atol = 1e-4`),
not a modelling difference: the two forms are the same linear system in different bases.

**The trap, demonstrated.** Differentiating the root map at `k_e = k_a`:

```
d k_e / d sigma1  via sqrt            -> nan
d tau1  / d sigma1 in canonical coords ->  1748.25
```

The discriminant on the tied boundary is exactly 0.0 with a gradient of −6.8e-21, i.e. finite. A test
asserts the NaN, so the reason this module exists stays documented in executable form.

**Scope check before building it:** inside the JAX engine the individual `stomach_glucose_mg` and
`gut_glucose_mg` states are read only by `vector_field` itself. No analysis reads them. The numpy
reference engine keeps the cascade and is untouched.

## D2/D3 Coordinates and the tied-rate model

| | range |
| --- | --- |
| `tau1 = 1/k_e + 1/k_a` | 56.25 – 150.00 min |
| `p = 1/(k_e k_a)` | 781.2 – 5555.6 |
| tied-rate `tau1` (from the rate intervals' intersection) | 62.50 – 133.33 min |

Round-trips are exact at every corner of the original box, and the real-pole condition `p <= tau1^2 / 4`
holds across the whole box, as it must.

**The projection works and lands on the tied model.** A step to `(tau1, p) = (60, 5000)` violates the
constraint (`tau1^2 / 4 = 900`); projection returns `p = 900.00` exactly, which is `k_e = k_a = 0.03333`,
and `on_tied_boundary` reports True. The projection is differentiable at the clipped point (gradient
3.0: one from `log tau1` directly, two from the ceiling it is pinned to).

**One decision recorded.** The fitted box is the **rectangular hull** of the image of the original
`(k_e, k_a)` box, not the image itself, which is a curvilinear region. Some corners of the hull
correspond to rate pairs the original box excluded -- the projected point above is one: `k = 0.0333`
exceeds the `carb_absorption` upper bound of 0.032, and `rates_in_original_box` duly reports
`inside_original_box: False`. The looser box is the deliberate direction, because a bound that cuts into
the likelihood manufactures apparent identifiability, which is reviewer point 6. Every fit records
whether its implied rates fell inside the original box, so the looseness is visible rather than assumed
harmless.

## B2/B3 as applied, with the numbers

**`rho = 0.976`** on the first subject's trace residuals, with an innovation standard deviation of
4.49 mg/dL. That is nearly a random walk, and it is why B3 matters: without whitening, 37 samples per
meal would enter the likelihood as 37 independent observations when they carry far less, inflating
everything the trace rung appears to reveal and biasing H3 and H5 in its favour.

**Calibration check.** With sigmas estimated at the same point, a correctly scaled negative
log-likelihood should land near `n_residuals / 2`:

| rung | residuals | NLL at `theta_hat_ML` | `n/2` |
| --- | --- | --- | --- |
| iauc | 41 | 19.0 | 20.5 |
| iauc_centroid | 81 | 37.5 | 40.5 |
| trace | 1517 | 758.5 | 758.5 |

A test asserts this within 35% for all three rungs. Had the weights not been applied, or the residual
count been wrong, every profile-likelihood interval built on this scale would have been mis-sized.

**Measured sigmas** for that subject: iAUC 1787.9 mg/dL*min, centroid 32.2 min, trace innovation
4.49 mg/dL. The spread across observables is a factor of 50, which is exactly why an unweighted sum of
the two residual families would have been an iAUC fit with rounding error attached.

## A defect in the middle rung, found and fixed

**82 of the 1640 CGMacros meals (5.0%) have an observed iAUC of exactly zero** -- glucose never rose
above the pre-meal baseline. For those the centroid is not a measurement but the floor of a zero-over-
zero ratio, and it reads as **0 min**: an impossibly early peak that would have dragged the timing fit
earlier for every meal the subject ate. The centroid residual is now scored only where an excursion
exists, and the centroid variance is computed on the same subset so the timing term is not quietly
down-weighted. No meal is excluded from the iAUC rung, where "no rise" is a perfectly good observation.

This is not a tunable threshold: it is exactly the set on which the statistic is undefined. Both
behaviours are pinned by tests.

## Early signal, not a result

On one subject fitted with `iauc_centroid`, all three parameterizations agree and **both the rate fit
and the coordinate fit land on `k_e = k_a`** -- the rate fit at 0.02273 for both, the coordinate fit on
the tied boundary. The tied two-parameter model reaches an NLL of 28.60 against 26.87 for the
three-parameter one, a difference of 1.7 on a scale where the chi-square threshold for one degree of
freedom is 1.92.

That is the shape H11 and H12 predict. It is one subject with no interval attached and it is recorded
here only so the Phase 2 numbers can be compared against a prior expectation rather than read as a
surprise.

## Tests

**46 new tests, all passing** (`tests/test_phase1.py`), plus 29 for the observables
(`tests/test_observables.py`). The ones worth naming:

- the canonical engine reproduces the cascade to 1e-4, and is differentiable at equal rates where the
  root map returns NaN;
- the linearization error is **quadratic** in the perturbation, not merely small;
- the basal regime is wrong by more than 30% while the active regime is within 1%, so the default cannot
  be switched back without a failure that explains itself;
- the generalized fit reproduces the submitted fit to 1e-6;
- the AR(1) estimate recovers a known `rho` of 0.8 to within 0.05, is pooled within meals and not across
  them (exactly 5/7 if the boundary pair were included), and reports zero rather than a negative
  coefficient;
- the NLL lands near `n/2` on all three rungs;
- a random start without a seed is refused, and seeded starts are reproducible.

Two test constructions were wrong on the first run and were corrected rather than the modules: an
assertion that `rho` would be exactly 1 where the estimator caps it at 0.99, and an alternating-sign
series that has zero rather than negative lag-1 correlation.

## Next

Phase 2 gates on G1 after 2.1, 2.2, 2.3 and 2.4. Items to carry forward:

1. `theta_hat_ML` is in place, so 2.2 can report the fraction of subjects on a bound for the
   **unregularized** estimate, which is the comparison that matters for reviewer point 6.
2. The Fisher Jacobian should be built as `vmap(jacrev(one_meal))`, about one gradient over all meals,
   rather than one reverse pass per meal. The Phase 2 gate check on 3 subjects will measure it.
3. Julia is needed for 2.4 (`StructuralIdentifiability.jl`), with `Project.toml` and `Manifest.toml`
   committed alongside the script and its raw output, per C5. A sympy fallback is documented if the
   toolchain is declined.

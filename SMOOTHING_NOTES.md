# Smoothing decisions — differentiable JAX engine (`simulation/jax_engine.py`)

Every nonsmooth operation in the numpy engine that lies on the gradient path for glucose inference,
and how it is made differentiable. Each is a deliberate modeling choice with a physiological
justification, not a hack. Validated: numpy↔jax glucose agreement max 1.5% / mean 0.6% on a 75g
meal; iAUC identical; `jax.grad(iAUC)/dSi` matches finite differences to 0.0%.

## Decision 0 (DIVERGENCE FROM BRIEF) — state clamping removed from the RHS, not soft-clamped
The brief proposed replacing the numpy per-step hard clamp with a `tanh` `soft_clamp` on **all**
state variables. **We did not do this, because the proposed `soft_clamp` is mathematically wrong
for physiological ranges.** `soft_clamp(x, lo, hi)` is near-identity only when `x` is near the
range midpoint; for glucose with bounds (15, 700) the midpoint is 357, so `soft_clamp(90)` returns
≈134 — a 49% distortion of a normal fasting glucose. Applying it corrupted the dynamics (glucose
collapsed to ~27 mg/dL). 

Correct approach: the numpy per-step clamp was an **explicit-Euler artifact**, not physiology — it
never activates in normal operation (glucose stays 70–200 for a meal, far inside 15–700). In the
continuous ODE we use **raw state in the vector field** and enforce boundedness only where it is
real: by **projecting θ to plausible parameter ranges during optimization** (`gradient_fit`), and
optionally guarding pathological runs. `soft_clamp` is kept as a helper for that projection, not
applied to the state. This is more faithful to the physiology AND removes a large gradient bias.

## Decision 1 — `max(0, ·)` rectifiers → softplus  (`softplus_relu`, β=10)
`softplus_relu(x) = softplus(βx)/β`. <1% error for |x|>0.5, exact sign, smooth gradient. Applied to:
- insulin secretion `insulin_secretion · max(0, glucose − 90)` (Bergman β-cell term)
- hepatic output `0.10 · max(0, cortisol − 15)` (cortisol here is constant baseline, term ≈0)
- gluconeogenesis `0.08 · max(0, 72 − glucose) · fasted`
Physiological justification: enzymatic/secretory thresholds are soft, not hard hinges.

## Decision 2 — `_clamp01` ramps → smooth sigmoid  (`smooth_clamp01`, sharpness 6)
`fasted = clamp01((13 − insulin)/6)` and `glycogen_low = clamp01((350 − glycogen)/150)` become
`sigmoid(6·(z − 0.5))`. Justification: metabolic-state transitions (fed↔fasted) are graded.

## Decision 3 — state-value discontinuity → sigmoid dose-response  (`sigmoid_switch`)
`+0.15 if insulin > 15 else 0` (glycogen restoration) → `0.15 · sigmoid(10·(insulin − 15))`.
Justification: insulin-stimulated glycogen synthesis is a sigmoidal dose-response; the if/else was
a linearization. Sharpness 10 ⇒ 99% of max at insulin 15.5, 1% at 14.5 — preserves the threshold.
(The `bac > 0` elimination switch is in the alcohol module, ported when alcohol is added.)

## Decision 4 — meal impulse: Dirac state-add → Gaussian rate pulse  (`_meal_rate`, σ=2 min)
The numpy meal instantaneously adds `carbs·1000·0.90·blunt` mg to `stomach_glucose_mg`. In the ODE
this becomes a Gaussian rate pulse of the same total mass centered at the meal time (σ=2 min),
which is both physiologically truer (food enters the stomach over a couple of minutes) and
differentiable (incl. w.r.t. meal time). Mass is conserved; iAUC matches the numpy engine to 0%.
The `blunt = 1/(1 + 0.08·fibre + 0.005·fat)` factor is identical to the numpy engine (calibrated).

## Decision 5 — observation operator (`jax_observation.py`)
`peak_glucose`/`peak_time` argmax → soft-argmax via `softmax(glucose/T)` (T tunable). `iAUC` uses
`softplus_relu` above baseline + trapezoid — already smooth.

## Decision 6 — time/window switches (category C) — handled as inputs, not smoothed
Asleep/awake and exercise on/off are discontinuous in **time**, not state. For per-meal CGM
inference there is no exercise/sleep event in the window, so they don't appear on the gradient
path. When the multi-system port needs them, they enter as piecewise-constant time-indexed inputs
in `args` (diffrax handles time-discontinuous forcing), no smoothing required.

## Scope of the differentiable engine (this stage)
IN: metabolic (Bergman glucose/insulin/X + 2-compartment gut) + substrate (fasting/ketosis) — the
CGMacros per-meal inference target. Cortisol = diurnal baseline constant (no acute stressor in
CGMacros; hepatic coupling inactive below cortisol 15). caffeine=0, no exercise.
DEFERRED: alcohol / autonomic (HR/BP/HRV) / circadian / thermoregulation / hydration modules
(extend the same `vector_field`); pharmacology (open `substances` dict ≠ fixed JAX array).

"""A7: why an area is nearly blind to timing, and how far that survives nonlinearity (H2).

Four parts, all from the engine and its linearization, none from fitted data except where stated.

(a) **Linearized engine, long window.** The ratio of `|d iAUC / d log k|` (for `k_e` and for `k_a`) to
    `|d iAUC / d log S_I|`, with the window out to six hours. In a linear system the area under a
    response to a fixed input is a statement about the zeroth moment of the response, and the zeroth
    moment of the gut cascade does not depend on the rates: all the carbohydrate arrives eventually. H2
    predicts a ratio of at most 0.02 at six hours.
(b) **Nonlinear engine, windows 90, 180, 240, 360 min.** The same ratio, per subject (median over the
    subject's meals), then median and interquartile range over subjects. The three-hour window is the
    one the analyses use; H2 predicts a median of at most 0.25 there. A window that cuts the response
    short makes the area depend on timing, which is why the ratio is reported against window length.
(c) **The moments of the gut input** (part of the proposition, checked numerically and symbolically): for
    a hypoexponential input with rates `k_e`, `k_a`, the zeroth moment is 1 for every pair, the first
    moment is `1/k_e + 1/k_a`, and the second depends on the pair only through that sum and
    `1/k_e^2 + 1/k_a^2`.
(d) The ridge cosine for the centroid objective is H4 and is computed in `evaluation/ladder.py`.

(a) and (b) are evaluated at the subject's population-default parameters and, for (b), also at the
regularized fit (the A4 `theta_pilot`), so a result that held only at the default point would show.

Run:  python -m evaluation.runner evaluation.moment_checks --workers 6
      python -m evaluation.moment_checks            # part (c) only, instant
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import equinox as eqx            # noqa: E402
import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from personalization.subject_loss import TARGETS, base_params, build_params  # noqa: E402
from simulation.jax_engine import run_meal                              # noqa: E402
from simulation.jax_engine_linear import iauc_sensitivity               # noqa: E402
from simulation.jax_observables import Window, iauc                     # noqa: E402

ANALYSIS_ID = "A7_moment_checks"
WINDOWS = (90, 180, 240, 360)
H2_LINEAR_MAX = 0.02
H2_NONLINEAR_MAX = 0.25


def default_config() -> dict:
    return {"cohort": "cgmacros", "min_meals": 10, "limit": None, "windows": list(WINDOWS),
            "max_meals": 12, "regime": "active"}


def units(config: dict) -> list[str]:
    return [s.subject_id for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                              limit=config["limit"])]


def _pick(records, max_meals: int):
    """Evenly spaced meals, deterministically."""
    if len(records) <= max_meals:
        return list(records)
    index = np.linspace(0, len(records) - 1, max_meals).round().astype(int)
    return [records[i] for i in sorted(set(index.tolist()))]


def nonlinear_ratios(params, record, window_min: float) -> dict:
    """`|d iAUC / d log k|` over `|d iAUC / d log S_I|` for one meal in the full nonlinear engine."""
    window = Window(post_min=window_min)
    duration = window.meal_time_min + window_min

    def area(log_scale):
        scaled = eqx.tree_at(
            lambda m: (m.insulin_sensitivity, m.gastric_emptying, m.carb_absorption), params,
            (params.insulin_sensitivity * jnp.exp(log_scale[0]),
             params.gastric_emptying * jnp.exp(log_scale[1]),
             params.carb_absorption * jnp.exp(log_scale[2])))
        ts, glucose = run_meal(scaled, record["carbs_g"], record["fat_g"], record["fiber_g"],
                               duration_min=duration)
        return iauc(ts, glucose, window, beta=None)

    grad = np.asarray(jax.grad(area)(jnp.zeros(3)), dtype=float)
    reference = abs(grad[0])
    return {"d_iauc_d_log": dict(zip(TARGETS, grad.tolist())),
            "ratio_ke": abs(grad[1]) / reference if reference > 0 else float("inf"),
            "ratio_ka": abs(grad[2]) / reference if reference > 0 else float("inf")}


def run_unit(unit: str, config: dict) -> dict:
    subject = next(s for s in load_cohort(config["cohort"], min_meals=config["min_meals"],
                                          limit=config["limit"]) if s.subject_id == unit)
    base = base_params(subject.profile)
    meals = _pick(list(subject.records), config["max_meals"])
    carbs = float(np.median([r["carbs_g"] for r in subject.records]))

    from evaluation.ladder import a4_results
    stored = a4_results("iauc", 1.0).get(unit)
    fitted = None
    if stored is not None:
        theta = np.array([stored["theta_pilot"][n] for n in TARGETS], dtype=float)
        fitted = build_params(base, jnp.asarray(theta))

    out = {"subject_id": unit, "n_meals_used": len(meals), "median_carbs_g": carbs,
           "linear": {}, "nonlinear_default": {}, "nonlinear_fitted": {}}
    for w in config["windows"]:
        lin = iauc_sensitivity(base, carbs, window_min=float(w), regime=config["regime"])
        out["linear"][str(w)] = {"ratio_ke": lin["ratio_to_si"]["gastric_emptying"],
                                 "ratio_ka": lin["ratio_to_si"]["carb_absorption"],
                                 "d_iauc_d_log": lin["d_iauc_d_log"]}
        for key, params in (("nonlinear_default", base), ("nonlinear_fitted", fitted)):
            if params is None:
                continue
            rows = [nonlinear_ratios(params, r, float(w)) for r in meals]
            out[key][str(w)] = {
                "median_ratio_ke": float(np.median([r["ratio_ke"] for r in rows])),
                "median_ratio_ka": float(np.median([r["ratio_ka"] for r in rows])),
                "n": len(rows)}
    out["macros"] = {}
    return out


# --- (c) moments of the gut input -----------------------------------------------------------------

def moments_numeric(ke: float, ka: float, horizon: float | None = None, n: int = 400001):
    """Raw moments 0, 1, 2 of the impulse response `ke ka / (ka - ke) (e^{-ke t} - e^{-ka t})`."""
    from scipy.integrate import simpson
    slow = min(ke, ka)
    horizon = horizon or 60.0 / slow
    t = np.linspace(0.0, horizon, n)
    if abs(ka - ke) < 1e-12:
        h = ke * ke * t * np.exp(-ke * t)
    else:
        h = ke * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))
    return tuple(float(simpson(h * t ** k, x=t)) for k in range(3))


def moments_check(pairs=((0.015, 0.045), (0.03, 0.02), (0.02, 0.03), (0.04, 0.012), (0.025, 0.025))):
    """Numerical check of the three statements, and the symbolic derivation of the second moment."""
    import sympy as sp

    rows = []
    for ke, ka in pairs:
        m0, m1, m2 = moments_numeric(ke, ka)
        tau = 1 / ke + 1 / ka
        second = (1 / ke ** 2 + 1 / ka ** 2) + tau ** 2
        rows.append({"k_e": ke, "k_a": ka, "m0": m0, "m1": m1, "m2": m2,
                     "tau1": tau, "m1_error": m1 - tau, "m2_formula": second,
                     "m2_error": m2 - second, "m0_error": m0 - 1.0})
    ke, ka, s = sp.symbols("k_e k_a s", positive=True)
    transfer = ke * ka / ((s + ke) * (s + ka))
    d1 = -sp.diff(transfer, s).subs(s, 0)
    d2 = sp.diff(transfer, s, 2).subs(s, 0)
    symbolic = {
        "m0": str(sp.simplify(transfer.subs(s, 0))),
        "m1": str(sp.simplify(d1)),
        "m2_minus_formula": str(sp.simplify(d2 - ((1 / ke ** 2 + 1 / ka ** 2) + (1 / ke + 1 / ka) ** 2))),
    }
    # Pairs with equal first moment but different second moment, to show the second moment separates them.
    return {"pairs": rows, "symbolic": symbolic,
            "max_abs_error": float(max(max(abs(r["m0_error"]), abs(r["m1_error"]) / r["tau1"],
                                           abs(r["m2_error"]) / r["m2_formula"]) for r in rows))}


# --- summaries -------------------------------------------------------------------------------------

def summarize() -> dict:
    from evaluation.results_io import largest_matching
    rows = list(largest_matching(ANALYSIS_ID, lambda p: "linear" in p).values())
    if not rows:
        return {"n": 0}

    def stat(values):
        v = np.asarray(values, dtype=float)
        return {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                "q3": float(np.percentile(v, 75)), "max": float(v.max()), "n": int(v.size)}

    windows = sorted({int(w) for r in rows for w in r["linear"]})
    out = {"n_subjects": len(rows), "windows": windows, "linear": {}, "nonlinear_default": {},
           "nonlinear_fitted": {}}
    for w in windows:
        w = str(w)
        out["linear"][w] = {"ke": stat([r["linear"][w]["ratio_ke"] for r in rows]),
                            "ka": stat([r["linear"][w]["ratio_ka"] for r in rows])}
        for key in ("nonlinear_default", "nonlinear_fitted"):
            have = [r[key][w] for r in rows if w in r.get(key, {})]
            if have:
                out[key][w] = {"ke": stat([h["median_ratio_ke"] for h in have]),
                               "ka": stat([h["median_ratio_ka"] for h in have])}
    lin360 = out["linear"].get("360")
    nl180 = out["nonlinear_default"].get("180")
    out["h2"] = {
        "linear_360_ke_median": lin360["ke"]["median"] if lin360 else None,
        "linear_360_ka_median": lin360["ka"]["median"] if lin360 else None,
        "linear_met": bool(lin360 and lin360["ke"]["median"] <= H2_LINEAR_MAX
                           and lin360["ka"]["median"] <= H2_LINEAR_MAX),
        "nonlinear_180_ke_median": nl180["ke"]["median"] if nl180 else None,
        "nonlinear_180_ka_median": nl180["ka"]["median"] if nl180 else None,
        "nonlinear_met": bool(nl180 and nl180["ke"]["median"] <= H2_NONLINEAR_MAX
                              and nl180["ka"]["median"] <= H2_NONLINEAR_MAX),
    }
    out["h2"]["met"] = bool(out["h2"]["linear_met"] and out["h2"]["nonlinear_met"])
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(moments_check(), indent=2))

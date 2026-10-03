"""Event-log sensitivity — how much does sloppy meal logging cost the prediction?

The event log is the binding constraint on the whole product (a forward model needs inputs),
and its fidelity is a UX decision with an accuracy price:

  - timing error   -> cheap to avoid (phone timestamps meals automatically)
  - carb-AMOUNT    -> photo + LLM-vision estimation (real engineering cost)
  - carb-TYPE (GI) -> a food-database lookup in the logging path (big UX investment)

This measures which error the ENGINE's postprandial prediction is most sensitive to, so the
logging UX is chosen from a number rather than an intuition. Metric = iAUC and peak-timing
error vs a clean-log reference (same machinery as forward_validation).

WHAT THIS IS AND IS NOT
-----------------------
Reference and perturbed runs are BOTH the engine, so this measures the SIMULATOR's tolerance
to input error — precisely the right question for "how good must logging be". It says nothing
about real human physiology; a real person's glucose could be more or less forgiving. Treat
the crossover points as engineering guidance, not clinical fact.

Run:  python -m evaluation.logging_sensitivity
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from simulation import Simulator, PhysioParams, Schedule, Food, Sleep
from simulation.foods import food_defs
from simulation.observation import observe_series, spec_for
from evaluation.forward_validation import postprandial_metrics

DAY_MIN = 24 * 60
_PROFILE = {"weight_kg": 82.0, "height_cm": 178.0, "age": 41.0, "sex": "male"}

# A realistic day of NAMED foods (so GI is real and carb-type perturbation is meaningful).
# grams chosen so each meal is a substantial carb load.
_MEALS = [
    {"t_min": 8 * 60, "food": "oats_cooked", "grams": 300},
    {"t_min": 13 * 60, "food": "white_rice_cooked", "grams": 350},
    {"t_min": 19 * 60, "food": "pasta_cooked", "grams": 320},
]
# Same-GI-band swaps for the carb-TYPE test: what a coarse bucket would confuse.
# low-GI stand-in <-> high-GI stand-in at equal grams.
_TYPE_SWAP = {"oats_cooked": "white_bread", "white_rice_cooked": "lentils_cooked",
              "pasta_cooked": "potato_boiled"}


def _params():
    return PhysioParams.from_profile(_PROFILE)


def _run(meals: list[dict]):
    s = Schedule()
    s.add(Sleep(0, 7 * 60))
    for m in meals:
        s.add(Food(float(m["t_min"]), str(m["food"]), grams=float(m["grams"])))
    return Simulator(_params()).run(s, duration_min=DAY_MIN, dt=1.0, record_every=5,
                                    outputs=["glucose_mg_dl"])


def _glucose(meals):
    return observe_series(_run(meals), spec_for("cgm", "glucose"), step_min=5.0)


def _meal_iauc_peak(obs, meal_t):
    m = postprandial_metrics(obs["values"], obs["t0_min"], obs["step_min"], meal_t)
    return (m["iauc"], m["peak"], m["peak_time_min"]) if m else (None, None, None)


# --- perturbations a logging method would introduce ------------------------------------

def perturb_carbs(meals, frac, rng):
    out = []
    for m in meals:
        f = 1.0 + rng.uniform(-frac, frac)
        out.append({**m, "grams": max(1.0, m["grams"] * f)})
    return out


def perturb_timing(meals, minutes, rng):
    out = []
    for m in meals:
        out.append({**m, "t_min": float(np.clip(m["t_min"] + rng.uniform(-minutes, minutes),
                                                 60, DAY_MIN - 60))})
    return out


def perturb_type(meals, rng, p=1.0):
    """Swap each food for its same-band confusable (what a coarse 'medium carb' bucket does),
    keeping grams identical so ONLY glycemic type changes."""
    out = []
    for m in meals:
        swap = _TYPE_SWAP.get(m["food"], m["food"]) if rng.random() < p else m["food"]
        out.append({**m, "food": swap})
    return out


@dataclass
class Cost:
    label: str
    iauc_mae: float
    iauc_pct: float
    peak_time_mae: float


def _cost(label, ref_obs, perturbed_obs_list, meal_times) -> Cost:
    """Mean per-meal iAUC error and peak-timing error of perturbed runs vs the clean ref."""
    iauc_errs, pt_errs, ref_iaucs = [], [], []
    for meals_obs in perturbed_obs_list:
        for t in meal_times:
            ri, _, rpt = _meal_iauc_peak(ref_obs, t)
            pi, _, ppt = _meal_iauc_peak(meals_obs, t)
            if ri is None or pi is None:
                continue
            iauc_errs.append(abs(ri - pi))
            pt_errs.append(abs(rpt - ppt))
            ref_iaucs.append(ri)
    iauc_mae = statistics.fmean(iauc_errs) if iauc_errs else float("nan")
    denom = statistics.fmean(ref_iaucs) if ref_iaucs else 1.0
    return Cost(label, round(iauc_mae, 1), round(100 * iauc_mae / max(denom, 1e-9), 1),
                round(statistics.fmean(pt_errs), 1) if pt_errs else float("nan"))


def run_sensitivity(n_trials: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ref_obs = _glucose(_MEALS)
    meal_times = [m["t_min"] for m in _MEALS]

    def batch(fn):
        return [_glucose(fn()) for _ in range(n_trials)]

    results = {"single": [], "interaction": []}
    for frac in (0.30, 0.50, 0.75):
        results["single"].append(
            _cost(f"carb amount +/-{int(frac*100)}%",
                  ref_obs, batch(lambda f=frac: perturb_carbs(_MEALS, f, rng)), meal_times))
    for mins in (15, 30, 60):
        results["single"].append(
            _cost(f"meal timing +/-{mins}min",
                  ref_obs, batch(lambda mm=mins: perturb_timing(_MEALS, mm, rng)), meal_times))
    results["single"].append(
        _cost("carb TYPE (GI) swap",
              ref_obs, batch(lambda: perturb_type(_MEALS, rng)), meal_times))

    # interaction: 30min late AND 50% carb error, together vs the linear sum of each alone
    both = _cost("timing 30min + carb 50% (together)",
                 ref_obs, batch(lambda: perturb_carbs(perturb_timing(_MEALS, 30, rng), 0.50, rng)),
                 meal_times)
    results["interaction"] = both
    return results


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    r = run_sensitivity()
    print("=" * 74)
    print("EVENT-LOG SENSITIVITY — engine's postprandial error vs logging sloppiness")
    print("=" * 74)
    print(f"  {'perturbation':<34}{'iAUC MAE':>11}{'% of iAUC':>11}{'peak-time':>11}")
    print(f"  {'-'*67}")
    for c in sorted(r["single"], key=lambda c: c.iauc_pct):
        print(f"  {c.label:<34}{c.iauc_mae:>11.0f}{c.iauc_pct:>10.1f}%{c.peak_time_mae:>9.0f}m")
    b = r["interaction"]
    print(f"\n  {b.label:<34}{b.iauc_mae:>11.0f}{b.iauc_pct:>10.1f}%{b.peak_time_mae:>9.0f}m")

    # decision hint
    singles = {c.label: c for c in r["single"]}
    carb50 = singles["carb amount +/-50%"].iauc_pct
    time30 = singles["meal timing +/-30min"].iauc_pct
    typ = singles["carb TYPE (GI) swap"].iauc_pct
    dominant = max([("carb amount", carb50), ("meal timing", time30), ("carb type", typ)],
                   key=lambda x: x[1])
    print(f"\n  dominant error source (iAUC): {dominant[0]} ({dominant[1]:.1f}%)")
    print("  decision:")
    print("    timing dominates -> auto-timestamp; carb buckets are fine")
    print("    amount dominates -> photo + vision carb estimation is worth it")
    print("    type   dominates -> food-DB lookup needed (biggest UX cost)")
    print()
    print("  NOTE: reference and perturbed runs are BOTH the engine, so this is the")
    print("  SIMULATOR's input-error tolerance, not a claim about real physiology.")
    print("=" * 74)


if __name__ == "__main__":
    main()

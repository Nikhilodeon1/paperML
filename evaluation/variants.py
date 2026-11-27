"""Subject and input variants for the Stage 2 and Stage 3 reruns (decision rules R13 to R16).

Each variant changes one thing and is selected by one configuration key, which every stored result records:

* `min_gap_h`      keep only meals more than this many hours after the previous logged meal (CGMacros).
* `basal`          `"subject"` sets basal glucose to the subject's own pre-meal level instead of 90 mg/dL.
* `window_min`     a shorter observation window after the meal.
* `sensor_lag_min` a first-order sensor lag with this time constant inside the fitted model.
* replica `calibrate: "regression"` replaces the logged carbohydrate by its regression-calibrated value.

Meal gaps need the meal times, which the committed cohort cache does not keep. `build_meal_gaps` derives them from the
raw archive and stores only the gaps in hours (no dates), so a pod needs no dataset.

Run:  python -m evaluation.variants build-gaps      # on a machine that has the CGMacros archive
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

from evaluation.cohort_data import Subject, load_cohort

GAPS_PATH = Path(__file__).resolve().parents[1] / "cohort_cache" / "cgmacros_meal_gaps.json"
MIN_LOG_VARIANCE = 1e-4           # floor on the estimated variance of log true load (decision rule R13)
PRE_MEAL_SAMPLES = 6              # -30 to -5 min on the 5 minute grid, the observables' baseline window


def regression_calibrate(logged, carb_cv: float) -> np.ndarray:
    """E[X | W] for a logged load W = X * U, with U lognormal of unit mean and coefficient of variation `carb_cv`.

    log X is taken as normal across one subject's meals; its mean and variance are estimated from the logged loads and
    the error variance is taken as known. Meals logged with no carbohydrate stay at zero.
    """
    logged = np.asarray(logged, dtype=float)
    out = logged.copy()
    positive = logged > 0
    if carb_cv <= 0 or positive.sum() < 2:
        return out
    error_variance = np.log1p(carb_cv ** 2)
    log_w = np.log(logged[positive])
    mean_x = log_w.mean() + error_variance / 2.0                    # E[log U] = -error_variance / 2
    variance_x = max(log_w.var(ddof=1) - error_variance, MIN_LOG_VARIANCE)
    weight = variance_x / (variance_x + error_variance)
    posterior_mean = mean_x + weight * (log_w + error_variance / 2.0 - mean_x)
    out[positive] = np.exp(posterior_mean + variance_x * (1.0 - weight) / 2.0)
    return out


def subject_basal(records) -> float:
    """The median over meals of the mean glucose in the 30 minutes before the meal."""
    return float(np.median([np.mean(r["glucose"]["values"][:PRE_MEAL_SAMPLES]) for r in records]))


def with_basal(params, value: float):
    import equinox as eqx
    import jax.numpy as jnp
    return eqx.tree_at(lambda p: p.basal_glucose, params, jnp.asarray(float(value), params.basal_glucose.dtype))


def sensor_lag(glucose, step_min: float, time_constant_min: float):
    """First-order lag, dS/dt = (G - S) / T, on an evenly sampled trace, starting at rest (S = G at the first sample)."""
    import jax
    import jax.numpy as jnp
    keep = jnp.exp(-step_min / time_constant_min)

    def step(state, value):
        state = keep * state + (1.0 - keep) * value
        return state, state

    _, rest = jax.lax.scan(step, glucose[0], glucose[1:])
    return jnp.concatenate([glucose[:1], rest])


@lru_cache(maxsize=1)
def meal_gaps() -> dict:
    return json.loads(GAPS_PATH.read_text(encoding="utf-8"))


def isolate(subject: Subject, min_gap_h: float) -> Subject:
    """The subject restricted to meals more than `min_gap_h` hours after the previous logged meal."""
    if subject.cohort != "cgmacros":
        raise ValueError("meal gaps are stored for CGMacros only")
    gaps = meal_gaps()[subject.subject_id]
    if len(gaps) != subject.n_meals:
        raise ValueError(f"{subject.subject_id}: {len(gaps)} gaps for {subject.n_meals} meals")
    records = tuple(r for r, gap in zip(subject.records, gaps) if gap is None or gap > min_gap_h)
    return dataclasses.replace(subject, records=records)


def build_meal_gaps() -> dict:
    """Hours from the previous logged meal (any logged meal, including ones without a usable window) to each cached meal."""
    from evaluation import cgmacros
    cached = {s.subject_id: s for s in load_cohort("cgmacros", min_meals=10)}
    out = {}
    for subject_id, meals in cgmacros.subjects():
        if subject_id not in cached:
            continue
        ordered = sorted(meals, key=lambda m: m.t)
        previous, gaps, carbs = None, [], []
        for meal in ordered:
            gap = None if previous is None else (meal.t - previous) / timedelta(hours=1)
            previous = meal.t
            if meal.real_iauc() is not None:                       # the same rule that built the cached records
                gaps.append(None if gap is None else round(float(gap), 4))
                carbs.append(float(meal.carbs_g))
        expected = [r["carbs_g"] for r in cached[subject_id].records]
        if len(carbs) != len(expected) or not np.allclose(carbs, expected):
            raise ValueError(f"{subject_id}: raw meal log does not line up with the cached meals")
        out[subject_id] = gaps
    missing = set(cached) - set(out)
    if missing:
        raise ValueError(f"no raw meal log for {sorted(missing)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build-gaps",))
    ap.parse_args(argv)
    gaps = build_meal_gaps()
    GAPS_PATH.write_text(json.dumps(gaps, sort_keys=True), encoding="utf-8")
    print(f"wrote {GAPS_PATH.name}: {len(gaps)} subjects, {sum(map(len, gaps.values()))} meals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

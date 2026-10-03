"""One way to load each cohort, into one record schema, so every analysis sees the same data.

Previously each script loaded a cohort itself and applied its own minimum-meal filter, so "the 45
subjects" meant a slightly different set depending on which script produced the number. And the
three loaders returned three different shapes: CGMacros keyed iAUC as `iauc` with a `glucose` block,
Hall and Shanghai as `observed_iAUC` with a `cgm_curve` list. Anything working across cohorts had to
know which was which.

Here the selection rule is one function with explicit arguments, the order is sorted by subject
identifier, and every cohort arrives as `Subject` objects with identical record keys. The parse comes
from `evaluation/cohort_cache` -- the committed, hash-checked cache if present, the archives
otherwise -- so a machine with no datasets behaves the same as the one the cache was built on.

The cohorts differ in ways that matter and are carried on the object rather than smoothed over:

* **CGMacros** -- n=45, 5 min CGM, carbohydrates from a photographed, nutritionist-coded diet log,
  iAUC over 0-180 min.
* **Hall 2018** -- n=24, 5 min CGM, standardized meals so carbohydrates are known by design, but
  iAUC over a fixed 0-**145** min window. A cross-cohort iAUC comparison has to say that.
* **ShanghaiT2DM** -- n=97, sensor sampled every **15** min and the stored 5-min grid was filled by
  linear interpolation, carbohydrates estimated from a free-text dietary record. A trace objective
  here must not treat interpolated points as independent observations; `grid_is_interpolated` says
  so on the object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from evaluation.cohort_cache import load as load_cohort_cache

__all__ = ["Subject", "load_cgmacros", "load_hall", "load_shanghai", "load_cohort",
           "cohort_summary"]


@dataclass(frozen=True)
class Subject:
    """One subject: the meals that produced a usable postprandial window, plus the biometrics.

    `records` entries always carry `carbs_g`, `fat_g`, `fiber_g`, `protein_g`, `iauc` and a
    `glucose` block of `{values, t0_min, step_min, meal_t_min}`, whichever cohort they came from.
    """
    subject_id: str
    cohort: str
    records: tuple[dict, ...]
    profile: dict
    sampling_min: float
    window_min: float
    grid_is_interpolated: bool = False
    clinical: dict = field(default_factory=dict)

    @property
    def n_meals(self) -> int:
        return len(self.records)

    def iauc(self) -> np.ndarray:
        return np.array([r["iauc"] for r in self.records], dtype=float)

    def carbs(self) -> np.ndarray:
        return np.array([r["carbs_g"] for r in self.records], dtype=float)

    def traces(self) -> np.ndarray:
        """Meals x samples array of the glucose window, NaN where a meal's window is shorter."""
        width = max(len(r["glucose"]["values"]) for r in self.records)
        out = np.full((len(self.records), width), np.nan)
        for i, r in enumerate(self.records):
            values = r["glucose"]["values"]
            out[i, :len(values)] = values
        return out


def _record(carbs, fat, fiber, protein, iauc, values, t0_min, step_min) -> dict:
    return {"carbs_g": float(carbs), "fat_g": float(fat), "fiber_g": float(fiber),
            "protein_g": float(protein), "iauc": float(iauc),
            "glucose": {"values": [float(v) for v in values], "t0_min": float(t0_min),
                        "step_min": float(step_min), "meal_t_min": 0.0}}


def _subjects_from_cache(cohort: str, min_meals: int) -> tuple[Subject, ...]:
    payload = load_cohort_cache(cohort)
    sampling = float(payload["sampling_min"])
    window = float(payload["window_min"])
    interpolated = bool(payload.get("grid_is_interpolated", False))
    pre = float(payload.get("pre_meal_min", 30.0))

    out: list[Subject] = []
    for entry in payload["subjects"]:
        if cohort == "cgmacros":
            records = [
                _record(r["carbs_g"], r["fat_g"], r["fiber_g"], r.get("protein_g", 0.0), r["iauc"],
                        r["glucose"]["values"], r["glucose"]["t0_min"], r["glucose"]["step_min"])
                for r in entry["records"]]
            profile = entry["profile"]
            raw = entry.get("clinical") or {}
            # Normalized to the same key names the other cohorts use, so a cross-cohort analysis
            # does not have to know that one loader spelled it `hba1c` and another `HbA1c`.
            clinical = {"HbA1c": raw.get("hba1c"), "HOMA_IR": raw.get("homa_ir"),
                        "fasting_glucose": raw.get("fasting_glucose"),
                        "fasting_insulin": raw.get("fasting_insulin"),
                        "bmi": raw.get("bmi"), "dx_group": raw.get("status")}
            clinical = {k: v for k, v in clinical.items() if v is not None}
        else:
            records = [
                _record(m["carbs_g"], m.get("fat_g", 0.0), m.get("fiber_g", 0.0),
                        m.get("protein_g", 0.0), m["observed_iAUC"], m["cgm_curve"], -pre, 5.0)
                for m in entry["meals"]]
            demographics = entry.get("demographics") or {}
            profile = {"weight_kg": demographics.get("weight_kg"),
                       "height_cm": demographics.get("height_cm"),
                       "age": demographics.get("age") or 45,
                       "sex": demographics.get("sex") or "male"}
            clinical = {k: entry.get(k) for k in ("HbA1c", "SSPG", "HOMA_IR", "dx_group")
                        if entry.get(k) is not None}

        # A subject with no usable weight or height cannot have population-default parameters built
        # for them, so they cannot be fitted at all; dropped here with the reason visible in the
        # cohort summary rather than failing deep inside a worker.
        if not profile.get("weight_kg") or not profile.get("height_cm"):
            continue
        if len(records) < min_meals:
            continue
        out.append(Subject(subject_id=entry["subject_id"], cohort=cohort,
                           records=tuple(records), profile=profile, sampling_min=sampling,
                           window_min=window, grid_is_interpolated=interpolated,
                           clinical=clinical))
    out.sort(key=lambda s: s.subject_id)
    return tuple(out)


@lru_cache(maxsize=8)
def load_cohort(cohort: str, min_meals: int = 10, limit: int | None = None) -> tuple[Subject, ...]:
    """Subjects of one cohort, sorted, filtered, normalized. Cached per process."""
    subjects = _subjects_from_cache(cohort, min_meals)
    return subjects[:limit] if limit else subjects


def load_cgmacros(limit: int | None = None, min_meals: int = 10,
                  use_cache: bool = True) -> tuple[Subject, ...]:
    """CGMacros subjects with at least `min_meals` usable meals and complete biometrics.

    `use_cache` is accepted for compatibility with earlier callers; the committed cohort cache is
    always preferred and verified, so there is no longer a reason to bypass it.
    """
    del use_cache
    return load_cohort("cgmacros", min_meals=min_meals, limit=limit)


def load_hall(limit: int | None = None, min_meals: int = 3) -> tuple[Subject, ...]:
    return load_cohort("hall", min_meals=min_meals, limit=limit)


def load_shanghai(limit: int | None = None, min_meals: int = 3) -> tuple[Subject, ...]:
    return load_cohort("shanghai", min_meals=min_meals, limit=limit)


def cohort_summary(subjects_: tuple[Subject, ...]) -> dict:
    """The counts the paper has to report for each cohort, computed rather than remembered."""
    if not subjects_:
        return {"cohort": None, "n_subjects": 0, "n_meals_total": 0}
    meals = np.array([s.n_meals for s in subjects_], dtype=float)
    carbs = np.concatenate([s.carbs() for s in subjects_])
    first = subjects_[0]
    return {
        "cohort": first.cohort,
        "n_subjects": len(subjects_),
        "n_meals_total": int(meals.sum()),
        "meals_per_subject": {"median": float(np.median(meals)), "min": float(meals.min()),
                              "max": float(meals.max())},
        "carbs_g": {"median": float(np.median(carbs)),
                    "iqr": [float(np.percentile(carbs, 25)), float(np.percentile(carbs, 75))]},
        "sampling_min": first.sampling_min,
        "window_min": first.window_min,
        "grid_is_interpolated": first.grid_is_interpolated,
        "n_with_hba1c": sum(1 for s in subjects_ if s.clinical.get("HbA1c") is not None),
        "subject_ids": [s.subject_id for s in subjects_],
    }

"""Where an analysis gets its subject from: a real cohort, a scaled copy of one, or a simulated replica.

Phase 8 needs the same fitting, Fisher and profile machinery to run on three kinds of data without each
analysis learning about the other two:

* a **real** subject from the cohort cache, with the sampling grid of its own cohort (`window_for`);
* a **carbohydrate-scaled** copy (`config["carb_scale"]`): every logged carbohydrate multiplied by a
  constant, the observed glucose untouched, to see how much a systematic logging error moves `S_I`;
* a **well-specified replica** (`config["replica"]`): the model is the truth. Each real meal is simulated
  at the subject's regularized prediction-fit estimate, CGM noise and carbohydrate logging error are
  added, and the analysis is run on that. Whatever the real-data analysis finds that the replica does
  not find is not explained by noise at that level, optimization, or the observable; whatever the
  replica also shows is.

Phase 9 adds a second truth for the replica, `replica["truth"] == "random"`: instead of each subject's own
fit, the true parameters are drawn independently and uniformly inside the box (at least 5% of its width
from either bound), per subject and seed. That is a synthetic recovery study with known truth, so whether
a parameter is recoverable can be read against where in the box it lies, which the fit-truth replica
cannot show because its timing truths sit near the upper bound.

The replica's noise is the subject's own: AR(1) residuals from the real trace against the maximum-
likelihood fit (fallback sigma 10 mg/dL, rho 0.7 when that estimate is unstable). The carbohydrate error
is lognormal with unit-mean multiplier and coefficient of variation `carb_cv`; the TRUTH uses the true
carbohydrates and the FIT is given the noisy ones. Every random stream is seeded from a SHA-256 of the
unit, the replica seed and the stream's name, so switching one noise source on or off does not change
the draws of the other.
"""
from __future__ import annotations

import dataclasses
import hashlib
from functools import lru_cache

import numpy as np

from evaluation.cohort_data import Subject, load_cohort

FALLBACK_SIGMA_MG_DL = 10.0
FALLBACK_RHO = 0.7
INSIDE_FRACTION = 0.05          # truth moved at least this far inside the box, as a fraction of its width


def window_for(subject: Subject):
    """The observation window of the subject's cohort, on that cohort's own sampling grid."""
    from simulation.jax_observables import Window
    stride = max(1, int(round(subject.sampling_min / 5.0))) if subject.sampling_min >= 10 else 1
    return Window(post_min=float(subject.window_min), stride=stride)


def _rng(*parts) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _cohort_subject(config: dict, unit: str) -> Subject:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config.get("limit"))
    subject = next((s for s in subjects if s.subject_id == unit), None)
    if subject is None:
        raise KeyError(f"no subject {unit!r} in cohort {config['cohort']!r}")
    return subject


def get_subject(config: dict, unit: str) -> Subject:
    subject = _cohort_subject(config, unit)
    scale = config.get("carb_scale")
    if scale not in (None, 1, 1.0):
        records = tuple({**r, "carbs_g": r["carbs_g"] * float(scale)} for r in subject.records)
        subject = dataclasses.replace(subject, records=records)
    replica = config.get("replica")
    if replica:
        subject = _replica_subject(subject, replica, config)
    return subject


# --------------------------------------------------------------------------------------------------
# Replica
# --------------------------------------------------------------------------------------------------

def _reference_fit(unit: str, box: float = 1.0) -> dict:
    from evaluation.ladder import a4_results
    row = a4_results("iauc", box).get(unit)
    if row is None:
        raise RuntimeError(f"no A4 iAUC result for {unit} at box {box}: the replica truth is the "
                           f"regularized prediction fit stored there")
    return row


def true_parameters(subject: Subject, unit: str, box: float = 1.0) -> dict:
    """The regularized prediction-fit estimate, moved at least 5% of the box width inside the box."""
    from personalization.objectives import _scaled_bounds
    from personalization.subject_loss import TARGETS
    row = _reference_fit(unit, box)
    lower, upper = (np.asarray(v, dtype=float) for v in _scaled_bounds(box))
    margin = INSIDE_FRACTION * (upper - lower)
    theta = np.array([row["theta_pilot"][n] for n in TARGETS], dtype=float)
    theta = np.clip(theta, lower + margin, upper - margin)
    return dict(zip(TARGETS, theta.tolist()))


def random_parameters(unit: str, seed: int, box: float = 1.0) -> dict:
    """Independent uniform draws inside the box, at least 5% of its width from either bound."""
    from personalization.objectives import _scaled_bounds
    from personalization.subject_loss import TARGETS
    lower, upper = (np.asarray(v, dtype=float) for v in _scaled_bounds(box))
    margin = INSIDE_FRACTION * (upper - lower)
    theta = _rng(unit, seed, "truth").uniform(lower + margin, upper - margin)
    return dict(zip(TARGETS, theta.tolist()))


def _simulate(subject: Subject, theta: dict, carbs=None):
    import jax.numpy as jnp
    from personalization.objectives import simulate_meals
    from personalization.subject_loss import TARGETS, base_params, build_params
    base = base_params(subject.profile)
    params = build_params(base, jnp.asarray([theta[n] for n in TARGETS]))
    records = subject.records
    carbs = np.array([r["carbs_g"] for r in records]) if carbs is None else np.asarray(carbs)
    fat = np.array([r["fat_g"] for r in records])
    fiber = np.array([r["fiber_g"] for r in records])
    return np.asarray(simulate_meals(params, jnp.asarray(carbs), jnp.asarray(fat), jnp.asarray(fiber),
                                     210.0, 5.0), dtype=float)


def real_residual_noise(subject: Subject, unit: str) -> dict:
    """AR(1) coefficient and innovation sd of the real trace around the maximum-likelihood fit."""
    from personalization.noise_model import estimate_ar1
    row = _reference_fit(unit)
    try:
        simulated = _simulate(subject, row["theta_ml"])
        n = min(len(r["glucose"]["values"]) for r in subject.records)
        real = np.array([r["glucose"]["values"][:n] for r in subject.records], dtype=float)
        residual = real - simulated[:, :n]
        rho, innovation = estimate_ar1(residual)
        stable = bool(np.isfinite(rho) and np.isfinite(innovation) and 0.5 < innovation < 60.0)
    except Exception:
        stable = False
    if not stable:
        return {"rho": FALLBACK_RHO, "innovation_sd": FALLBACK_SIGMA_MG_DL * np.sqrt(1 - FALLBACK_RHO ** 2),
                "fallback": True}
    return {"rho": float(rho), "innovation_sd": float(innovation), "fallback": False}


def _ar1_noise(rng, rho: float, innovation_sd: float, shape: tuple[int, int]) -> np.ndarray:
    meals, n = shape
    out = np.zeros(shape)
    out[:, 0] = rng.normal(0.0, innovation_sd / np.sqrt(max(1.0 - rho ** 2, 1e-6)), meals)
    for t in range(1, n):
        out[:, t] = rho * out[:, t - 1] + rng.normal(0.0, innovation_sd, meals)
    return out


@lru_cache(maxsize=256)
def _replica_cached(cohort: str, min_meals: int, unit: str, seed: int, cgm: bool, carb_cv: float,
                    limit, truth_mode: str = "fit") -> tuple:
    config = {"cohort": cohort, "min_meals": min_meals, "limit": limit}
    subject = _cohort_subject(config, unit)
    truth = (random_parameters(unit, seed) if truth_mode == "random"
             else true_parameters(subject, unit))
    clean = _simulate(subject, truth)                                   # 210 min on a 5 min grid
    n_samples = clean.shape[1]
    glucose = clean.copy()
    noise = {"rho": None, "innovation_sd": None, "fallback": None}
    if cgm:
        noise = real_residual_noise(subject, unit)
        glucose = glucose + _ar1_noise(_rng(unit, seed, "cgm"), noise["rho"], noise["innovation_sd"],
                                       glucose.shape)
    carbs = np.array([r["carbs_g"] for r in subject.records], dtype=float)
    if carb_cv > 0:
        sigma2 = np.log1p(carb_cv ** 2)
        multiplier = _rng(unit, seed, "carb").lognormal(-0.5 * sigma2, np.sqrt(sigma2), carbs.shape)
        logged = carbs * multiplier
    else:
        logged = carbs
    from simulation.jax_observables import Window, iauc
    window = Window()
    records = []
    for i, r in enumerate(subject.records):
        values = glucose[i]
        area = float(iauc(None, np.asarray(values), window, beta=None))
        records.append({**r, "carbs_g": float(logged[i]), "iauc": area,
                        "glucose": {"values": [float(v) for v in values], "t0_min": -30.0,
                                    "step_min": 5.0, "meal_t_min": 0.0}})
    return tuple(records), truth, noise, n_samples


def _replica_subject(subject: Subject, replica: dict, config: dict) -> Subject:
    records, _, _, _ = _replica_cached(config["cohort"], config["min_meals"], subject.subject_id,
                                       int(replica.get("seed", 0)), bool(replica.get("cgm", True)),
                                       float(replica.get("carb_cv", 0.25)), config.get("limit"),
                                       str(replica.get("truth", "fit")))
    # The replica is always simulated on the 5 minute grid with the default window, whatever the
    # cohort it was built from; it is a CGMacros-style replica.
    return dataclasses.replace(subject, records=tuple(records), sampling_min=5.0, window_min=180.0,
                               grid_is_interpolated=False)


def replica_truth(config: dict, unit: str) -> dict:
    replica = config["replica"]
    _, truth, noise, _ = _replica_cached(config["cohort"], config["min_meals"], unit,
                                         int(replica.get("seed", 0)), bool(replica.get("cgm", True)),
                                         float(replica.get("carb_cv", 0.25)), config.get("limit"),
                                         str(replica.get("truth", "fit")))
    return {"theta_true": truth, "noise": noise}


def truth_in_coordinates(theta_true: dict) -> dict:
    """The true parameters in the natural units of the coordinate profiles: `S_I`, `tau1`, `p`."""
    from personalization import coords as co
    tau1, p = co.rates_to_tau_p(theta_true["gastric_emptying"], theta_true["carb_absorption"])
    return {"log_insulin_sensitivity": float(theta_true["insulin_sensitivity"]),
            "log_tau1": float(tau1), "log_p": float(p)}

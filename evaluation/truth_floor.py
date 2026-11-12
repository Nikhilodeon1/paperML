"""The prediction floor of a replica (Amendment 9, R1): how well does the TRUE model predict its own data?

On a replica the true parameters are known, so the held-out error of a fit can be compared with the error of the
model at the truth. That error is the floor: it is what the model makes of the logged carbohydrate and of the
sensor noise when it has the right parameters, and no fit can be expected to go below it by more than chance. A fit
whose held-out error is near the floor while its parameters are far from the truth is the case the paper
claims to exist; a fit whose error rises with its parameter error would show that prediction does see the failure.

One unit is one subject. No fitting happens here, so the analysis is cheap. The units read the same replica
as the prediction and profile analyses (`evaluation.subject_source`), so the floor is on the very meals the fits saw.

Run:  python -m evaluation.runner evaluation.truth_floor --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
"""
from __future__ import annotations

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

from evaluation import prediction_cv                                     # noqa: E402
from evaluation.cohort_data import load_cohort                           # noqa: E402
from evaluation.subject_source import get_subject, replica_truth         # noqa: E402
from personalization.subject_loss import TARGETS                         # noqa: E402

ANALYSIS_ID = "A16_truth_floor"


def default_config() -> dict:
    base = prediction_cv.default_config()
    return {k: base[k] for k in ("cohort", "min_meals", "limit", "steps", "learning_rate", "lam", "grid_shape",
                                 "grid1_points", "beta", "carb_scale", "replica")} | {
        "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}, "cohort": "cgmacros"}


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    return [s.subject_id for s in subjects]


def run_unit(unit: str, config: dict) -> dict:
    """Predictions of the model at the true parameters, for every meal of the replica subject."""
    if not config.get("replica"):
        raise ValueError("the floor is defined for a replica only")
    truth = replica_truth(config, unit)
    theta = np.asarray([truth["theta_true"][name] for name in TARGETS], dtype=float)
    subject = get_subject(config, unit)
    ctx = prediction_cv.SubjectContext(subject, config)
    out = ctx.predict(theta)
    obs = np.asarray(ctx.obs_iauc, dtype=float)
    pred = np.asarray(out["iauc"], dtype=float)
    return {
        "subject_id": unit, "cohort": config["cohort"], "replica": config["replica"], "n_meals": int(ctx.n),
        "theta_true": truth["theta_true"], "targets": list(TARGETS),
        "pred_iauc": pred.tolist(), "obs_iauc": obs.tolist(),
        "iauc_mae": float(np.mean(np.abs(pred - obs))),
        "pred_peak": np.asarray(out["peak"], dtype=float).tolist(),
        "obs_peak": np.asarray(ctx.obs_peak, dtype=float).tolist(),
        "trace_rmse": float(np.mean(np.asarray(out["trace_rmse"], dtype=float))),
        "macros": {},
    }

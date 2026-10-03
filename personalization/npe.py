"""Amortized parameter inference — the SBI scaffold (summary-stat route).

Simulation-Based Inference turns the engine (a forward model theta -> glucose curve) into an
inverse model: given a user's observed meal response, infer the posterior over their physiology
parameters. Trained ONCE on simulated data, it then fits a new user in milliseconds — vs SMC's
per-user minutes (`personalization/particle_fit.py` is the exact baseline this must match).

Design (per the plan): SUMMARY-STATISTIC input, not raw CGM. With only 45 real subjects for
eventual validation, five interpretable stats (iAUC, peak, peak-time, baseline, early slope)
generalize better than a learned encoder over 288 raw samples, and we already compute most of
them. The observed carb load is passed alongside the stats so the estimator is conditioned on
what was eaten (real meals vary).

This module is the DEPENDENCY-FREE substrate:
  - `SUMMARY_NAMES`, `summary_stats`  — the observation operator's summary
  - `PRIOR`, `simulate_summary`       — the sbi "simulator": theta -> summary
  - `generate_training_set`           — sample prior -> simulate -> (theta, x) for training
  - `AmortizedEstimator`              — a scikit RandomForest theta_hat(x): amortized POINT
                                         inference, no torch. Proves the summary stats are
                                         informative and that amortization works.

The full SNPE (calibrated posteriors via a normalizing flow) is the `sbi`/torch upgrade on top
of this substrate — same simulator, same summary, same training set. Kept separate so the
torch dependency is only pulled when quantified uncertainty is actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.observation import observe_series, spec_for

_PRE, _WINDOW = 30.0, 180.0
_PROFILE = {"weight_kg": 82, "height_cm": 178, "age": 45, "sex": "male"}

PARAM_NAMES = ["insulin_sensitivity", "gastric_emptying", "carb_absorption"]
PRIOR = {"insulin_sensitivity": (0.30, 1.60), "gastric_emptying": (0.015, 0.040),
         "carb_absorption": (0.012, 0.032)}
SUMMARY_NAMES = ["iauc", "peak", "peak_time_min", "baseline", "early_slope"]


def summary_stats(values, t0_min, step_min, meal_t_min=_PRE) -> np.ndarray:
    """Five interpretable stats of a postprandial window — the NPE input (plus carbs)."""
    pre, seg, times = [], [], []
    for i, v in enumerate(values):
        t = t0_min + i * step_min
        if meal_t_min - _PRE <= t < meal_t_min:
            pre.append(float(v))
        if meal_t_min <= t <= meal_t_min + _WINDOW:
            seg.append(float(v)); times.append(t - meal_t_min)
    if len(seg) < 3:
        return np.array([np.nan] * 5)
    base = float(np.mean(pre)) if pre else seg[0]
    incr = [max(0.0, v - base) for v in seg]
    iauc = step_min * (sum(incr) - 0.5 * (incr[0] + incr[-1]))
    peak = max(seg); peak_time = times[seg.index(peak)]
    # early slope: rise over the first ~45 min (mg/dL per min)
    j = min(range(len(times)), key=lambda k: abs(times[k] - 45.0))
    early_slope = (seg[j] - base) / max(times[j], 1e-6)
    return np.array([iauc, peak, peak_time, base, early_slope])


def _params(theta: dict) -> PhysioParams:
    p = PhysioParams.from_profile(_PROFILE)
    for k, v in theta.items():
        setattr(p, k, float(v))
    return p


# Demographics are amortized too, so a new user of any body size is handled without retraining
# (and so it matches the per-subject-demographics grid fit it is compared against).
_DEMO_RANGES = {"weight_kg": (50.0, 140.0), "height_cm": (150.0, 195.0), "age": (18.0, 72.0)}
FEATURE_NAMES = SUMMARY_NAMES + ["carbs", "weight_kg", "height_cm", "age", "sex_male"]


def simulate_summary(theta: np.ndarray, carbs: float, profile: dict | None = None,
                     rng=None, noise_sd: float = 10.0) -> np.ndarray:
    """The sbi simulator: parameter vector -> summary stats of a `carbs`-gram meal response for
    a person of the given demographics. `rng` adds CGM noise so the estimator accounts for it."""
    p = PhysioParams.from_profile(profile or _PROFILE)
    for k, v in zip(PARAM_NAMES, theta):
        setattr(p, k, float(v))
    s = Schedule(); s.add(Meal(_PRE, carbs_g=float(carbs)))
    traj = Simulator(p).run(s, duration_min=_PRE + _WINDOW, dt=1.0, record_every=5,
                            outputs=["glucose_mg_dl"])
    obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
    vals = obs["values"]
    if rng is not None and noise_sd > 0:
        vals = list(np.asarray(vals) + rng.normal(0, noise_sd, len(vals)))
    return summary_stats(vals, obs["t0_min"], 5.0, _PRE)


def features_for(summary: np.ndarray, carbs: float, profile: dict) -> np.ndarray:
    """Assemble one feature row (summary stats + meal + demographics) — the estimator input."""
    return np.array([*summary, float(carbs), float(profile.get("weight_kg", 82)),
                     float(profile.get("height_cm", 178)), float(profile.get("age", 45)),
                     1.0 if profile.get("sex", "male") == "male" else 0.0])


def sample_prior(n: int, rng) -> np.ndarray:
    lo = np.array([PRIOR[k][0] for k in PARAM_NAMES])
    hi = np.array([PRIOR[k][1] for k in PARAM_NAMES])
    return rng.uniform(lo, hi, size=(n, len(PARAM_NAMES)))


def generate_training_set(n: int, seed: int = 0, carb_range=(30.0, 90.0)):
    """Sample theta ~ prior + random meal + random demographics, simulate, summarise ->
    (theta[n,3], features[n, len(FEATURE_NAMES)])."""
    rng = np.random.default_rng(seed)
    theta = sample_prior(n, rng)
    carbs = rng.uniform(*carb_range, size=n)
    x = np.empty((n, len(FEATURE_NAMES)))
    for i in range(n):
        prof = {"weight_kg": rng.uniform(*_DEMO_RANGES["weight_kg"]),
                "height_cm": rng.uniform(*_DEMO_RANGES["height_cm"]),
                "age": rng.uniform(*_DEMO_RANGES["age"]),
                "sex": "male" if rng.random() < 0.5 else "female"}
        x[i] = features_for(simulate_summary(theta[i], carbs[i], prof, rng=rng), carbs[i], prof)
    good = ~np.isnan(x).any(axis=1)
    return theta[good], x[good]


@dataclass
class AmortizedEstimator:
    """RandomForest theta_hat(summary, carbs) — amortized POINT inference, no torch.

    Not a calibrated posterior (that is the sbi/SNPE upgrade), but it proves the two things the
    scaffold needs to show: the summary stats carry enough signal to recover theta, and
    inference is instant after a one-time train (vs SMC per-user minutes)."""
    model: object = None
    r2: dict = None

    def fit(self, theta: np.ndarray, x: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                           n_jobs=-1, random_state=0).fit(x, theta)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(np.atleast_2d(x))

    def infer_from_meals(self, meals: list[dict], profile: dict | None = None) -> dict:
        """A user's logged meals + demographics -> one theta estimate (median over meals)."""
        profile = profile or _PROFILE
        feats = []
        for m in meals:
            g = m.get("glucose") or {}
            vals = g.get("values", m.get("values"))
            if vals is None:
                continue
            s = summary_stats(vals, g.get("t0_min", 0.0), g.get("step_min", 5.0),
                              g.get("meal_t_min", _PRE))
            if not np.isnan(s).any():
                feats.append(features_for(s, m.get("carbs_g", 60.0), profile))
        if not feats:
            return {}
        est = np.median(self.model.predict(np.array(feats)), axis=0)
        return dict(zip(PARAM_NAMES, [round(float(v), 4) for v in est]))

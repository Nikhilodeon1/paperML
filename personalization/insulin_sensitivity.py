"""Per-user insulin sensitivity from logged meal responses — with a confidence gate.

The k-fold result (evaluation/kfold_calibration.py) showed a fitted Si predicts held-out meals
better than a personal constant — but ONLY when the fit is trustworthy. A user with two logged
meals has a wide, noisy posterior; using that noisy Si would be worse than the population
prior. So this learner GATES: it returns a fitted Si only when there is enough data AND the fit
beats the prior on that user's own meals; otherwise it hands back the population prior and says
so (`source`). The live simulate path uses the value only when `source == "fitted"`.

Consumes `user["meal_responses"]`: a list of logged postprandial windows, each
    {carbs_g, protein_g, fat_g, fiber_g, glucose: {values, t0_min, step_min, meal_t_min}}
which is exactly the shape the CGMacros loader and the observation model produce, so real
CGM+food-log data drops straight in. Synthetic users have none -> population prior (honest).
"""

from __future__ import annotations

import numpy as np

from simulation import Simulator, PhysioParams, Schedule, Meal
from simulation.observation import observe_series, spec_for

_PRE, _WINDOW = 30.0, 180.0
_SI_GRID = np.linspace(0.2, 1.6, 15)
_MIN_MEALS = 4                 # below this the posterior is too wide to beat the prior
_IMPROVE = 0.90                # fitted must cut error to <=90% of the prior's to be used


def _iauc(values, t0_min, step_min, meal_t_min) -> float | None:
    """Baseline-subtracted incremental AUC over 0-3h (baseline = mean of the pre-meal window)."""
    pre, seg = [], []
    for i, v in enumerate(values):
        t = t0_min + i * step_min
        if meal_t_min - _PRE <= t < meal_t_min:
            pre.append(float(v))
        if meal_t_min <= t <= meal_t_min + _WINDOW:
            seg.append(float(v))
    if len(seg) < 3:
        return None
    base = float(np.mean(pre)) if pre else seg[0]
    incr = [max(0.0, v - base) for v in seg]
    return step_min * (sum(incr) - 0.5 * (incr[0] + incr[-1]))


def _predict_iauc(carbs, si, profile, protein=0.0, fat=0.0, fiber=0.0) -> float:
    p = PhysioParams.from_profile(profile)
    p.insulin_sensitivity = float(np.clip(si, 0.05, 3.0))
    s = Schedule()
    s.add(Meal(_PRE, carbs_g=carbs, protein_g=protein, fat_g=fat, fiber_g=fiber))
    traj = Simulator(p).run(s, duration_min=_PRE + _WINDOW, dt=1.0, record_every=5,
                            outputs=["glucose_mg_dl"])
    obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
    return _iauc(obs["values"], obs["t0_min"], 5.0, _PRE) or 0.0


def _real_iauc(mr: dict) -> float | None:
    g = mr.get("glucose") or {}
    if "values" not in g:
        return None
    return _iauc(g["values"], g.get("t0_min", 0.0), g.get("step_min", 5.0),
                g.get("meal_t_min", _PRE))


def learn_insulin_sensitivity(user: dict) -> dict | None:
    """Fit Si from the user's logged meals; return a gated result or None (=> use prior).

    {value, sd, n, source}. source: 'fitted' (trustworthy, live path uses it), 'weak' (fit ran
    but doesn't beat the prior enough / too few meals -> value is the prior), 'prior' (no data).
    """
    profile = user.get("profile") or {}
    if not (profile.get("weight_kg") and profile.get("height_cm")):
        return None
    prof = {"weight_kg": float(profile["weight_kg"]), "height_cm": float(profile["height_cm"]),
            "age": float(profile.get("age") or 45), "sex": profile.get("sex", "male")}
    prior_si = PhysioParams.from_profile(prof).insulin_sensitivity

    reals = []
    for mr in user.get("meal_responses", []):
        r = _real_iauc(mr)
        if r is not None and mr.get("carbs_g"):
            reals.append((mr, r))
    if len(reals) < _MIN_MEALS:
        return {"value": round(prior_si, 3), "sd": None, "n": len(reals), "source": "prior"}

    def total_err(si):
        return sum(abs(_predict_iauc(m["carbs_g"], si, prof, m.get("protein_g", 0.0),
                                     m.get("fat_g", 0.0), m.get("fiber_g", 0.0)) - r)
                   for m, r in reals)

    errs = np.array([total_err(si) for si in _SI_GRID])
    i = int(errs.argmin())
    si, best = float(_SI_GRID[i]), float(errs[i])
    prior_err = total_err(prior_si)

    # posterior width from the error curve (softmax over -error): a wide spread = untrustworthy
    ll = -errs / (best + 1e-9)
    w = np.exp(ll - ll.max()); w /= w.sum()
    mu = float((w * _SI_GRID).sum())
    sd = float(np.sqrt((w * (_SI_GRID - mu) ** 2).sum()))

    # The improvement-over-prior gate is the meaningful confidence signal: use the fitted Si
    # only when it predicts the user's own meals materially better than the population prior.
    # (sd is kept as a loose sanity bound — its absolute scale is not well calibrated, so it is
    # not the primary gate. A fit that clearly beats the prior is worth using even if somewhat
    # wide; when the user's Si ~= the prior, the improvement is small and we keep the prior.)
    trustworthy = best <= _IMPROVE * prior_err and sd < 0.5
    return {"value": round(si if trustworthy else prior_si, 3), "sd": round(sd, 3),
            "n": len(reals), "source": "fitted" if trustworthy else "weak"}


# --- test/synthetic helper --------------------------------------------------------------

def make_meal_responses(true_si: float, n: int = 8, profile: dict | None = None,
                        noise_sd: float = 10.0, seed: int = 0) -> list[dict]:
    """Logged meal responses for a person with a known hidden Si (engine + CGM noise)."""
    prof = profile or {"weight_kg": 82, "height_cm": 178, "age": 45, "sex": "male"}
    rng = np.random.default_rng(seed)
    p = PhysioParams.from_profile(prof); p.insulin_sensitivity = float(true_si)
    out = []
    for k in range(n):
        carbs = float(rng.integers(35, 90))
        s = Schedule(); s.add(Meal(_PRE, carbs_g=carbs))
        traj = Simulator(p).run(s, duration_min=_PRE + _WINDOW, dt=1.0, record_every=5,
                                outputs=["glucose_mg_dl"])
        obs = observe_series(traj, spec_for("cgm", "glucose"), step_min=5.0)
        vals = [round(v + rng.normal(0, noise_sd), 1) for v in obs["values"]]
        out.append({"carbs_g": carbs, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0,
                    "glucose": {"values": vals, "t0_min": obs["t0_min"], "step_min": 5.0,
                                "meal_t_min": _PRE}})
    return out

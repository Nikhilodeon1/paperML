"""Uncertainty quantification — honest confidence bands on the simulation.

The engine is deterministic given a PhysioParams, but those parameters are themselves
uncertain: population physiology varies person to person, and even a personalized value is
a posterior with a spread. UQ propagates that parameter uncertainty through the simulation
by Monte-Carlo — sample many plausible parameter sets, run the engine for each, and report
the median trajectory with a 5-95% band. So an answer becomes "glucose peaks around 155
(120-185)", not a single false-precise number.

Crucially the bands are PERSONALIZED: `ParamUncertainty.from_user` tightens a parameter's
spread to the user's Bayesian posterior SD where we have one, so the more data we have on a
person, the sharper their forecast — the payoff of the personalization layer, made visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .params import PhysioParams
from .engine import Simulator
from .inputs import Schedule


@dataclass(frozen=True)
class Spec:
    sd: float
    kind: str = "lognormal_rel"        # "lognormal_rel" (multiplicative) | "normal_abs"


# Population inter-individual + measurement uncertainty for the parameters that most drive
# the outputs (relative SDs unless noted). Deliberately conservative but physiological.
_DEFAULTS: dict[str, Spec] = {
    "insulin_sensitivity": Spec(0.28),
    "rmr_kcal_min": Spec(0.10),
    "glucose_effectiveness": Spec(0.22),
    "insulin_secretion": Spec(0.25),
    "carb_absorption": Spec(0.22),
    "gastric_emptying": Spec(0.22),
    "alcohol_beta_g_dl_min": Spec(0.18),
    "hr_rest": Spec(4.0, "normal_abs"),
}


class ParamUncertainty:
    def __init__(self, specs: dict[str, Spec] | None = None):
        self.specs: dict[str, Spec] = dict(specs or _DEFAULTS)

    @classmethod
    def from_user(cls, user: dict) -> "ParamUncertainty":
        """Tighten spreads to the user's learned posteriors where available (fewer data ->
        wider bands, more data -> sharper)."""
        u = cls()
        d = user.get("derived") or {}
        m = d.get("metabolic") or {}
        if m.get("rmr_multiplier_sd") is not None:
            # posterior SD of the RMR multiplier -> relative uncertainty on RMR, capped.
            # (Only parameters we actually LEARN get tightened — we don't infer glucose
            # kinetics from weigh-ins, so those stay at population uncertainty. Honest.)
            u.specs["rmr_kcal_min"] = Spec(min(0.12, max(0.02, float(m["rmr_multiplier_sd"]))))
        h = d.get("hepatic") or {}
        if h.get("elimination_beta_sd") is not None and h.get("elimination_beta"):
            rel = float(h["elimination_beta_sd"]) / max(1e-6, float(h["elimination_beta"]))
            u.specs["alcohol_beta_g_dl_min"] = Spec(min(0.25, max(0.03, rel)))
        # learned resting-HR baseline -> tighten HR uncertainty to the observed SEM
        wl = d.get("wearable") or {}
        if wl.get("resting_hr", {}).get("sd") is not None:
            u.specs["hr_rest"] = Spec(max(0.5, float(wl["resting_hr"]["sd"])), "normal_abs")
        return u


def sample_params(base: PhysioParams, unc: ParamUncertainty, rng: np.random.Generator) -> PhysioParams:
    """One draw of a plausible PhysioParams (re-clamped via __post_init__ through replace)."""
    over: dict[str, float] = {}
    for name, spec in unc.specs.items():
        val = getattr(base, name, None)
        if val is None:
            continue
        if spec.kind == "normal_abs":
            over[name] = val + float(rng.normal(0, spec.sd))
        else:                                       # lognormal_rel (multiplicative, positive)
            over[name] = val * math.exp(float(rng.normal(0, spec.sd)))
    return replace(base, **over)


@dataclass
class Ensemble:
    times_min: list[float]
    bands: dict[str, dict]             # var -> {"median":[...], "lo":[...], "hi":[...]}
    n: int
    _peaks: dict = field(default_factory=dict)
    _ends: dict = field(default_factory=dict)

    def summary(self, variables=None) -> dict:
        variables = variables or ("glucose_mg_dl", "insulin_uU_ml", "heart_rate_bpm",
                                  "sbp_mmhg", "cortisol_ug_dl", "bac_g_dl", "ketones_mmol_l",
                                  "alertness")
        def _r(x):                                  # adaptive rounding (BAC needs 3 dp)
            x = float(x)
            return round(x, 3) if abs(x) < 1 else round(x, 1)
        out = {}
        for v in variables:
            if v in self._peaks:
                pk, en = self._peaks[v], self._ends[v]
                out[v] = {
                    "peak": {"median": _r(np.median(pk)), "lo": _r(np.percentile(pk, 5)),
                             "hi": _r(np.percentile(pk, 95))},
                    "end": {"median": _r(np.median(en)), "lo": _r(np.percentile(en, 5)),
                            "hi": _r(np.percentile(en, 95))},
                }
        return out


def run_ensemble(base: PhysioParams, schedule: Schedule, duration_min: float, *,
                 n: int = 120, unc: ParamUncertainty | None = None,
                 outputs: list[str] | None = None, dt: float = 1.0, seed: int = 0) -> Ensemble:
    """Monte-Carlo the simulation over parameter uncertainty; return median + 5-95% bands."""
    unc = unc or ParamUncertainty()
    n = max(20, min(int(n), 400))
    rng = np.random.default_rng(seed)

    runs = [Simulator(sample_params(base, unc, rng)).run(schedule, duration_min, dt=dt,
                                                         outputs=outputs) for _ in range(n)]
    times = runs[0].times_min
    variables = list(runs[0].series)
    bands: dict[str, dict] = {}
    peaks: dict[str, np.ndarray] = {}
    ends: dict[str, np.ndarray] = {}
    for v in variables:
        a = np.array([r.series[v] for r in runs])          # (n, T)
        bands[v] = {"median": np.round(np.median(a, 0), 3).tolist(),
                    "lo": np.round(np.percentile(a, 5, 0), 3).tolist(),
                    "hi": np.round(np.percentile(a, 95, 0), 3).tolist()}
        peaks[v] = a.max(axis=1)
        ends[v] = a[:, -1]
    return Ensemble(times_min=times, bands=bands, n=n, _peaks=peaks, _ends=ends)

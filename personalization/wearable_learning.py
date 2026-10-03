"""Per-user parameter learning from wearable history.

Learns each user's personal physiological baselines — resting heart rate, HRV, sleep
efficiency — from their wearable time-series, as a posterior (mean + shrinking uncertainty
with more days of data). These feed straight into the simulation engine's PhysioParams
(so the digital twin behaves like the real body) and into the uncertainty layer (so a
well-observed user gets sharper forecasts).

Works identically on synthetic or real wearable data — the same `wearable.daily` schema a
Galaxy Watch will POST — so the whole pipeline is built and validated now, before any
device is connected.
"""

from __future__ import annotations

import statistics

# (wearable key, PhysioParams field it maps to, population SD prior)
_SIGNALS = [
    ("resting_hr", "hr_rest", 8.0),
    ("hrv_rmssd", "hrv_rest", 15.0),
    ("sleep_efficiency", None, 6.0),
    ("steps", None, 2500.0),
]


def learn_wearable(user: dict) -> dict:
    """Posterior baselines from `wearable.daily`: {signal: {mean, sd(=SEM), n}}. Empty when
    too little data (the twin then uses population priors — honest)."""
    daily = user.get("wearable", {}).get("daily", [])
    out: dict = {}
    for key, _param, pop_sd in _SIGNALS:
        vals = [d[key] for d in daily if isinstance(d.get(key), (int, float))]
        if len(vals) >= 3:
            mean = statistics.mean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else pop_sd
            sem = sd / (len(vals) ** 0.5)              # posterior SD shrinks with more days
            out[key] = {"mean": round(mean, 1), "sd": round(max(0.1, sem), 2),
                        "day_sd": round(sd, 1), "n": len(vals)}
    return out


def learned_params(user: dict) -> dict:
    """PhysioParams overrides from the learned wearable baselines (for the engine)."""
    w = (user.get("derived", {}) or {}).get("wearable", {})
    over: dict = {}
    for key, param, _ in _SIGNALS:
        if param and key in w:
            over[param] = w[key]["mean"]
    return over

"""BodyState — the single shared physiological state the whole simulation evolves.

This is the heart of the framework. Every module reads from and writes to ONE state
vector, so systems are genuinely coupled: a stressor raising cortisol here is the same
cortisol the metabolism module reads to release glucose, and the same glucose the
autonomic module sees. That shared state is exactly what a real body simulator needs and
what a bag of independent estimators lacks.

Variables carry physiological units in their names. Baselines are the fasted, rested,
sober, awake resting state; modules perturb them. Extra variables can be added freely —
modules only touch the ones they declare.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace


@dataclass
class BodyState:
    t_min: float = 0.0                 # minutes since simulation start

    # --- metabolic / glucose-insulin ---
    glucose_mg_dl: float = 90.0        # plasma glucose
    insulin_uU_ml: float = 10.0        # plasma insulin
    x_insulin_action: float = 0.0      # remote insulin action (Bergman X), 1/min
    stomach_glucose_mg: float = 0.0    # carbohydrate in the stomach (gastric emptying)
    gut_glucose_mg: float = 0.0        # glucose in the gut lumen awaiting absorption
    glycogen_g: float = 450.0          # liver + muscle glycogen stores
    ketones_mmol_l: float = 0.1        # blood ketones (beta-hydroxybutyrate)
    sc_insulin_u: float = 0.0          # exogenous (injected) insulin in the subcut depot, units

    # --- fluid balance + thermoregulation ---
    water_deficit_ml: float = 0.0      # body-water deficit (dehydration), ml
    core_temp_c: float = 37.0          # core body temperature

    # --- substances ---
    bac_g_dl: float = 0.0              # blood alcohol concentration
    gut_alcohol_g: float = 0.0         # ethanol in the gut awaiting absorption
    caffeine_mg: float = 0.0           # plasma caffeine

    # --- neuroendocrine / autonomic ---
    cortisol_ug_dl: float = 11.0       # circulating cortisol (stress axis)
    heart_rate_bpm: float = 60.0
    hrv_rmssd_ms: float = 45.0         # heart-rate variability (parasympathetic tone)
    sbp_mmhg: float = 118.0
    dbp_mmhg: float = 76.0

    # --- energy + circadian ---
    energy_expended_kcal: float = 0.0  # cumulative expenditure since t=0
    sleep_pressure: float = 0.30       # homeostatic sleep drive (0..~1, Process S)
    alertness: float = 0.85            # 0..1, subjective/vigilance proxy

    # --- open, data-driven substance compartments (mg) ---
    # amount of each drug/substance in body / gut, keyed by name (+"__gut"). Populated by
    # the pharmacology module from knowledge_base/substances.json — new substances need no
    # new state field, so novel "what does X do to me" questions are answerable by data.
    substances: dict = field(default_factory=dict)

    def copy(self) -> "BodyState":
        return replace(self)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # vector helpers for the integrator ------------------------------------
    _INTEGRATED = (
        "glucose_mg_dl", "insulin_uU_ml", "x_insulin_action", "stomach_glucose_mg",
        "gut_glucose_mg", "glycogen_g", "ketones_mmol_l", "sc_insulin_u", "water_deficit_ml", "bac_g_dl",
        "gut_alcohol_g", "caffeine_mg", "cortisol_ug_dl", "heart_rate_bpm", "hrv_rmssd_ms",
        "sbp_mmhg", "dbp_mmhg", "core_temp_c", "energy_expended_kcal", "sleep_pressure",
        "alertness",
    )

    # hard physiological bounds — outputs SATURATE here rather than exploding, so even
    # absurd inputs (500g meal, 5000mg caffeine, 50 METs) yield extreme-but-bounded,
    # never-nonsensical numbers. (lo, hi) per variable.
    _BOUNDS = {
        "glucose_mg_dl": (15.0, 700.0), "insulin_uU_ml": (0.0, 1000.0),
        "x_insulin_action": (0.0, 5.0), "stomach_glucose_mg": (0.0, 1e6),
        "gut_glucose_mg": (0.0, 1e6), "glycogen_g": (0.0, 900.0),
        "ketones_mmol_l": (0.0, 10.0), "sc_insulin_u": (0.0, 200.0),
        "water_deficit_ml": (0.0, 15000.0), "bac_g_dl": (0.0, 1.0),
        "gut_alcohol_g": (0.0, 1e5), "caffeine_mg": (0.0, 1e5),
        "core_temp_c": (30.0, 43.0),
        "cortisol_ug_dl": (0.5, 100.0), "heart_rate_bpm": (25.0, 240.0),
        "hrv_rmssd_ms": (2.0, 200.0),
        "sbp_mmhg": (60.0, 270.0), "dbp_mmhg": (30.0, 170.0),
        "energy_expended_kcal": (0.0, 1e7), "sleep_pressure": (0.0, 1.2),
        "alertness": (0.0, 1.0),
    }

    def apply_rates(self, rates: dict[str, float], dt: float) -> None:
        """Advance every integrated variable by dt * summed-rate, then clamp to hard
        physiological bounds. Non-finite rates are treated as zero so NaN/inf can never
        propagate through the trajectory."""
        import math
        for name in self._INTEGRATED:
            r = rates.get(name, 0.0)
            if not math.isfinite(r):
                r = 0.0
            v = getattr(self, name) + r * dt
            lo, hi = self._BOUNDS.get(name, (float("-inf"), float("inf")))
            setattr(self, name, min(hi, max(lo, v)))
        # generic substance compartments (keys like "substances.nicotine")
        for key, r in rates.items():
            if key.startswith("substances."):
                if not math.isfinite(r):
                    r = 0.0
                sub = key.split(".", 1)[1]
                self.substances[sub] = max(0.0, self.substances.get(sub, 0.0) + r * dt)

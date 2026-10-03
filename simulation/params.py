"""PhysioParams — the per-person parameters that make the simulation THIS body.

The framework's personalization hook: same equations, different constants per user.
Anything Layer-3 learns about a user (insulin sensitivity, RMR multiplier, alcohol
elimination rate, resting HR, fitness) flows in here, so the simulated body behaves like
the real one. `from_profile` builds a sensible default set from a stored user profile;
learned parameters override the population defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysioParams:
    # anthropometrics
    weight_kg: float = 78.0
    height_cm: float = 178.0
    age: float = 35.0
    sex: str = "male"                  # "male" | "female"

    # metabolic (glucose-insulin minimal model + energy)
    insulin_sensitivity: float = 1.0   # multiplier on insulin action (Si); <1 = resistant
    circadian_amp: float = 0.0         # how much worse a meal hits in the evening (0 = off)
    rmr_kcal_min: float = 1.06         # resting energy expenditure per minute (~1525/day)
    glucose_effectiveness: float = 0.026   # p1, glucose self-disposal (1/min)
    x_decay: float = 0.028             # p2 (1/min)
    x_gain: float = 1.3e-5             # p3 base gain
    insulin_clearance: float = 0.16    # n (1/min)
    insulin_secretion: float = 0.16    # beta: uU/mL per (mg/dL over baseline) per min
    gastric_emptying: float = 0.026    # kempt, stomach -> gut (1/min)
    carb_absorption: float = 0.022     # kabs, gut glucose -> plasma (1/min)

    # cardiovascular + fitness
    hr_rest: float = 60.0
    sbp_rest: float = 118.0
    dbp_rest: float = 76.0
    vo2max: float = 40.0               # ml/kg/min — cardiorespiratory fitness
    hrv_rest: float = 45.0             # resting HRV (RMSSD, ms)
    cycle_day: int = 0                 # menstrual-cycle day (1-28); 0 = not tracked/applicable

    @property
    def luteal(self) -> bool:
        """Luteal phase (post-ovulation): progesterone-driven ~0.3C higher core temp and
        somewhat lower HRV. Only for a female with a tracked cycle day."""
        return self.sex == "female" and 15 <= self.cycle_day <= 28

    @property
    def luteal_temp_offset(self) -> float:
        return 0.32 if self.luteal else 0.0

    # substances
    alcohol_beta_g_dl_min: float = 0.00025   # ~0.015 g/dL per hour zero-order elimination
    alcohol_absorption: float = 0.12         # gut ethanol -> blood (1/min); calibrated so
                                             # peak BAC matches the validated Widmark module
    caffeine_halflife_min: float = 300.0     # ~5 h

    def __post_init__(self):
        """Clamp every parameter into a sane physiological range so the engine can never
        be handed garbage (0 kg, negative age, unknown sex, absurd sensitivities). Robust
        by construction: bad input degrades to the nearest plausible body, never a crash."""
        def clamp(v, lo, hi, default):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            if v != v:                                  # NaN
                return default
            return min(hi, max(lo, v))
        self.weight_kg = clamp(self.weight_kg, 30.0, 400.0, 78.0)
        self.height_cm = clamp(self.height_cm, 90.0, 230.0, 178.0)
        self.age = clamp(self.age, 5.0, 110.0, 35.0)
        self.sex = self.sex if self.sex in ("male", "female") else "male"
        self.insulin_sensitivity = clamp(self.insulin_sensitivity, 0.05, 3.0, 1.0)
        self.circadian_amp = clamp(self.circadian_amp, 0.0, 0.6, 0.0)
        self.rmr_kcal_min = clamp(self.rmr_kcal_min, 0.4, 3.0, 1.06)
        self.insulin_secretion = clamp(self.insulin_secretion, 0.02, 1.0, 0.16)
        self.hr_rest = clamp(self.hr_rest, 35.0, 120.0, 60.0)
        self.sbp_rest = clamp(self.sbp_rest, 80.0, 200.0, 118.0)
        self.dbp_rest = clamp(self.dbp_rest, 45.0, 130.0, 76.0)
        self.vo2max = clamp(self.vo2max, 15.0, 85.0, 40.0)
        self.hrv_rest = clamp(self.hrv_rest, 8.0, 150.0, 45.0)
        self.alcohol_beta_g_dl_min = clamp(self.alcohol_beta_g_dl_min, 0.0001, 0.001, 0.00025)

    @property
    def hr_max(self) -> float:
        return 220.0 - self.age

    @property
    def widmark_r(self) -> float:
        return 0.68 if self.sex == "male" else 0.55

    @classmethod
    def from_profile(cls, profile: dict, learned: dict | None = None) -> "PhysioParams":
        """Build params from a stored user profile (+ optional learned Layer-3 params)."""
        sex = profile.get("sex", "male")
        w = float(profile.get("weight_kg", 78))
        h = float(profile.get("height_cm", 178))
        age = float(profile.get("age", 35))
        # Mifflin-St Jeor RMR -> per-minute
        s = 5 if sex == "male" else -161
        rmr = (10 * w + 6.25 * h - 5 * age + s)
        # fitness from activity level + age (VO2max declines ~0.4 ml/kg/min per year)
        activity = str(profile.get("activity", "sedentary"))
        base_vo2 = {"sedentary": 33.0, "moderate": 42.0, "active": 52.0}.get(activity, 38.0)
        if sex == "female":
            base_vo2 -= 6.0
        vo2 = base_vo2 - 0.4 * max(0.0, age - 30.0)
        hrv = max(12.0, 55.0 - 0.45 * max(0.0, age - 25.0) + {"active": 10.0, "moderate": 3.0}.get(activity, 0.0))
        p = cls(weight_kg=w, height_cm=h, age=age, sex=sex, rmr_kcal_min=max(0.6, rmr / 1440.0),
                hr_rest=float(profile.get("resting_hr") or 60),
                sbp_rest=float(profile.get("sbp") or 118),
                dbp_rest=float(profile.get("dbp") or 76), vo2max=vo2, hrv_rest=hrv,
                cycle_day=int(profile.get("cycle_day") or 0))
        # diabetic / prediabetic -> lower insulin sensitivity + secretion
        if profile.get("diabetic"):
            p.insulin_sensitivity *= 0.45
            p.insulin_secretion *= 0.6
        elif (profile.get("hba1c") or 0) >= 5.7:
            p.insulin_sensitivity *= 0.7
        if learned:
            for k, v in learned.items():
                if hasattr(p, k) and v is not None:
                    setattr(p, k, v)
        return p

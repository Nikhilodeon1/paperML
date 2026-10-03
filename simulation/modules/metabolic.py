"""Metabolic module — glucose / insulin / energy, coupled to stress & activity.

Glucose-insulin dynamics follow the Bergman minimal model (Bergman 1979): plasma glucose
G, plasma insulin I, and a remote insulin-action compartment X. Carbohydrate from a meal
enters a gut compartment and is absorbed into plasma. On top of the classic model we add
the couplings a whole-body sim needs:

  - cortisol and caffeine raise hepatic glucose output (stress/■stimulant hyperglycaemia),
  - exercise drives insulin-INDEPENDENT glucose uptake and burns glycogen,
  - insulin sensitivity (Si) is a personal parameter (diabetic/prediabetic lower it).

Energy expenditure accumulates from RMR scaled by activity (METs). 'Strong but simplified'
(brief 2): the shapes and directions are right and the peaks are tuned to be physiological;
absolute precision would need a fuller model (e.g. Dalla Man 2007).
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow, Meal, Food, Insulin

_GB = 90.0      # basal glucose (mg/dL)
_IB = 10.0      # basal insulin (uU/mL)


def _circadian_resistance(clock_hour: float) -> float:
    """0 in the morning (best insulin sensitivity ~6am) rising to 1 late evening (~10pm).
    Multiplied by `circadian_amp` to set how much the same meal hits harder at night."""
    return min(1.0, max(0.0, (float(clock_hour) - 6.0) / 16.0))
_INSULIN_KA = 0.025          # rapid-acting SC absorption rate (1/min); Tmax ~50-60 min
_INSULIN_PLASMA = 44.0       # uU/mL added per unit/min absorbed (tuned: 10 IU -> ~+70 uU/mL peak)


class MetabolicModule(Module):
    name = "metabolic"
    writes = ("glucose_mg_dl", "insulin_uU_ml", "x_insulin_action", "stomach_glucose_mg",
              "gut_glucose_mg", "glycogen_g", "sc_insulin_u", "energy_expended_kcal")
    reads = ("cortisol_ug_dl", "caffeine_mg")

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        if isinstance(event, Meal):
            # ~90% of carbohydrate reaches plasma as glucose; enters the stomach first.
            # Fat and fibre blunt/delay the postprandial rise (slower gastric emptying, lower
            # 0-3h iAUC). Factor is 1.0 for a pure-carb meal, so the 75g reference calibration
            # (no fat/fibre) is unchanged. Coefficients (fibre 0.08, fat 0.005) are calibrated:
            # 5-fold CV on CGMacros (evaluation/kfold_calibration.py) — fibre dominates, fat is
            # minor, and this makes the personalized engine beat a personal constant on
            # held-out meals (5/5 folds). Fibre converges cleanly; a richer gut model (Dalla
            # Man) is the eventual upgrade past this lumped factor.
            blunt = 1.0 / (1.0 + 0.08 * event.fiber_g + 0.005 * event.fat_g)
            s.stomach_glucose_mg += event.carbs_g * 1000.0 * 0.90 * blunt
        elif isinstance(event, Food):
            # a named food -> its glycemic-index/fibre-adjusted effective carbohydrate
            from ..foods import effective_glycemic_carbs
            eff = effective_glycemic_carbs(event.food, event.grams)
            s.stomach_glucose_mg += eff * 1000.0 * 0.90
        elif isinstance(event, Insulin):
            # injected insulin lands in a subcutaneous depot, absorbed into plasma over ~1h
            s.sc_insulin_u += max(0.0, event.units)

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        Vg = 1.6 * p.weight_kg                      # glucose distribution volume (dL)
        # two-compartment gut: stomach --(gastric emptying)--> gut --(absorption)--> plasma,
        # which delays and rounds the postprandial peak to a realistic ~45-60 min.
        emptying = p.gastric_emptying * s.stomach_glucose_mg
        Ra = p.carb_absorption * s.gut_glucose_mg   # gut -> plasma appearance (mg/min)

        # counter-regulatory hepatic glucose output (stress + caffeine + exercise)
        # only cortisol ABOVE the normal diurnal range (acute stress) drives extra glucose,
        # so the ordinary morning cortisol peak doesn't spuriously raise resting glucose.
        hepatic = (0.10 * max(0.0, s.cortisol_ug_dl - 15.0)
                   + 0.09 * (s.caffeine_mg / 200.0)
                   + (0.45 if w.exercise_mets > 3 else 0.0))
        # insulin-independent uptake during exercise (mass-action in glucose)
        ex_uptake = 0.11 * w.exercise_mets * (s.glucose_mg_dl / _GB) if w.exercise_mets > 0 else 0.0

        dG = (-(p.glucose_effectiveness + s.x_insulin_action) * s.glucose_mg_dl
              + p.glucose_effectiveness * _GB
              + Ra / Vg + hepatic - ex_uptake)
        # exogenous insulin: subcutaneous depot absorbs into plasma (adds to dI); the rest of
        # the Bergman machinery (X action, insulin sensitivity) turns it into glucose lowering.
        insulin_absorb = _INSULIN_KA * s.sc_insulin_u                    # units/min
        dI = -p.insulin_clearance * (s.insulin_uU_ml - _IB) \
            + p.insulin_secretion * max(0.0, s.glucose_mg_dl - _GB) \
            + _INSULIN_PLASMA * insulin_absorb
        dSC = -insulin_absorb
        # circadian glucose tolerance: insulin sensitivity is best in the morning and worse in
        # the evening (~20-30% worse; well documented). `circadian_amp` scales this and is 0.0
        # by default (so the reference calibrations are unchanged) — calibrated on real meal
        # timestamps in evaluation/kfold_calibration.
        si_eff = p.insulin_sensitivity * (1.0 - p.circadian_amp * _circadian_resistance(w.clock_hour))
        dX = -p.x_decay * s.x_insulin_action \
            + p.x_gain * si_eff * (s.insulin_uU_ml - _IB)
        dStomach = -emptying
        dGut = emptying - p.carb_absorption * s.gut_glucose_mg

        # glycogen: burned during exercise, slowly restored when insulin is elevated (fed)
        dGly = (-0.35 * w.exercise_mets if w.exercise_mets > 0 else 0.0) \
            + (0.15 if s.insulin_uU_ml > _IB + 5 else 0.0)

        # energy expenditure (kcal/min): RMR scaled by activity, small stress/caffeine bump
        mets = max(1.0, w.exercise_mets)
        dE = p.rmr_kcal_min * mets * (1.0 + 0.03 * (s.caffeine_mg / 200.0))

        return {"glucose_mg_dl": dG, "insulin_uU_ml": dI, "x_insulin_action": dX,
                "stomach_glucose_mg": dStomach, "gut_glucose_mg": dGut,
                "glycogen_g": dGly, "sc_insulin_u": dSC, "energy_expended_kcal": dE}

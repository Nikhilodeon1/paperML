"""Autonomic module — heart rate & blood pressure as an integrator of the whole body.

HR and BP relax toward a target that is the sum of drivers coming from OTHER systems:
exercise, cortisol (stress), caffeine, and hypoglycaemia. Because those drivers live in
the shared state, the same stressor or coffee that shifts glucose here also shifts heart
rate — the coupling that makes this a body simulator, not a set of gauges. First-order
relaxation with a short time constant; magnitudes are physiological approximations.
"""

from __future__ import annotations

import math

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow

_BASE_CORT = 11.0


def _relax_rate(current: float, target: float, tau: float, dt: float) -> float:
    """Rate whose explicit-Euler step reproduces the EXACT exponential relaxation
    `x(t+dt) = target + (x - target)*exp(-dt/tau)`.

    HR relaxes with tau=0.8 min while the engine steps at dt=1.0, so the naive rate
    `(target-x)/tau` has dt/tau = 1.25 and the Euler update factor (1 - dt/tau) goes
    NEGATIVE: the state overshoots and rings. Measured, that produced a 65 bpm one-step
    spike at exercise onset and a 225 bpm phantom peak. This form is exact for a
    piecewise-constant target and unconditionally stable at any dt, with no solver change.

    (Exact only while this module is the sole writer of the variable; `pharmacology` also
    contributes to HR/BP, and summed rates are then approximate — but still stable.)
    """
    if dt <= 0.0 or tau <= 0.0:
        return 0.0
    return (target - current) * (1.0 - math.exp(-dt / tau)) / dt


def _toward_max(base: float, modifiers: float, ceiling: float) -> float:
    """Add sub-maximal drivers (caffeine, cortisol, heat...) so they approach `ceiling`
    asymptotically instead of stacking straight through it.

    The old form summed every driver on top of `hr_rest + (hr_max-hr_rest)*ex_frac`, so a
    40-year-old (HRmax 180) reached 220 bpm — 122% of HRmax — even with a perfect
    integrator. At rest there is plenty of headroom so behaviour is unchanged (+10 bpm per
    200 mg caffeine); near max effort the headroom vanishes and HR saturates at HRmax.
    """
    headroom = max(0.0, ceiling - base)
    if headroom <= 1e-6:
        return ceiling
    return base + headroom * (1.0 - math.exp(-max(0.0, modifiers) / headroom))


class AutonomicModule(Module):
    name = "autonomic"
    writes = ("heart_rate_bpm", "hrv_rmssd_ms", "sbp_mmhg", "dbp_mmhg")
    reads = ("caffeine_mg", "cortisol_ug_dl", "glucose_mg_dl", "water_deficit_ml",
             "core_temp_c", "bac_g_dl")

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        from .hydration import pct_dehydration
        # exercise pushes HR toward a fraction of heart-rate reserve set by intensity
        # RELATIVE TO FITNESS: the same METs are a smaller %VO2max (lower HR) for a fitter
        # person, so effort scales by VO2max.
        ex_frac = min(1.0, (w.exercise_mets * 3.5) / max(10.0, p.vo2max))
        hypo = max(0.0, 70.0 - s.glucose_mg_dl) * 0.4      # tachycardia when low
        dehy = pct_dehydration(s, p)                        # cardiovascular drift
        # effort sets the base; everything else is a sub-maximal driver that can only take
        # HR toward HRmax, never past it (see _toward_max).
        hr_base = p.hr_rest + (p.hr_max - p.hr_rest) * ex_frac
        hr_drivers = (10.0 * (s.caffeine_mg / 200.0)       # ~+8-12 bpm per 200 mg (Benowitz)
                      + 2.2 * max(0.0, s.cortisol_ug_dl - _BASE_CORT)
                      + hypo + 4.0 * dehy
                      + 10.0 * max(0.0, s.core_temp_c - 37.0))   # hyperthermic drift
        hr_target = _toward_max(hr_base, hr_drivers, p.hr_max)
        tau = 0.8 if w.exercise_mets > 0 else 1.6
        dHR = _relax_rate(s.heart_rate_bpm, hr_target, tau, w.dt)

        sbp_target = (p.sbp_rest + 45.0 * ex_frac
                      + 3.0 * max(0.0, s.cortisol_ug_dl - _BASE_CORT)
                      + 5.0 * (s.caffeine_mg / 200.0)
                      - 5.0 * dehy)
        dbp_target = (p.dbp_rest + 12.0 * ex_frac
                      + 1.5 * max(0.0, s.cortisol_ug_dl - _BASE_CORT))
        dSBP = _relax_rate(s.sbp_mmhg, sbp_target, 1.5, w.dt)
        dDBP = _relax_rate(s.dbp_mmhg, dbp_target, 1.5, w.dt)

        # HRV (RMSSD): parasympathetic tone. Crushed during exercise (recovers slowly
        # afterward = HR/HRV recovery), and suppressed by stress, alcohol, dehydration and
        # heat. Scales off the person's resting HRV (fitness).
        supp = (1.0 - 0.85 * ex_frac)
        supp *= max(0.35, 1.0 - 0.03 * max(0.0, s.cortisol_ug_dl - _BASE_CORT))
        supp *= max(0.4, 1.0 - 6.0 * s.bac_g_dl)                       # alcohol lowers HRV
        supp *= max(0.6, 1.0 - 0.05 * pct_dehydration(s, p) - 0.3 * max(0.0, s.core_temp_c - 37.5))
        if p.luteal:
            supp *= 0.9                                                # luteal-phase HRV dip
        hrv_target = p.hrv_rest * min(1.2, max(0.05, supp))
        tau_hrv = 2.5 if w.exercise_mets > 0 else 12.0                 # slow rebound (recovery)
        dHRV = _relax_rate(s.hrv_rmssd_ms, hrv_target, tau_hrv, w.dt)
        return {"heart_rate_bpm": dHR, "hrv_rmssd_ms": dHRV, "sbp_mmhg": dSBP, "dbp_mmhg": dDBP}

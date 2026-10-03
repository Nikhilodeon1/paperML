"""Simulator — the time-stepping engine that evolves the body forward.

Registers a set of modules, then marches the shared BodyState through time: at each step
it applies any impulse stimuli, asks every module for its rate contributions, sums them
per state variable, and integrates. The result is a Trajectory — the full time-series of
the simulated body, which downstream apps (the health predictor being the first) read.

Integration is fixed-step (default 1 min) explicit Euler — robust and fast for these
stiff-but-tame physiological ODEs at this granularity; swap in RK4 here if ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import BodyState
from .params import PhysioParams
from .inputs import Schedule
from .module import Module


@dataclass
class Trajectory:
    times_min: list[float]
    series: dict[str, list[float]]     # variable -> values aligned to times_min
    params: PhysioParams

    def final(self) -> dict:
        return {k: v[-1] for k, v in self.series.items()}

    def peak(self, var: str) -> float:
        return max(self.series[var])

    def at(self, var: str, t_min: float) -> float:
        i = min(range(len(self.times_min)), key=lambda j: abs(self.times_min[j] - t_min))
        return self.series[var][i]

    # Per-variable rounding: a flat 1 dp silently destroyed small-magnitude variables —
    # a real 0.039 g/dL blood alcohol rounded to 0.0, so drink scenarios reported no BAC.
    _PRECISION = {"bac_g_dl": 3, "ketones_mmol_l": 2, "sleep_pressure": 2,
                  "alertness": 2, "core_temp_c": 2, "x_insulin_action": 4}

    def summary(self) -> dict:
        """Compact, human/LLM-friendly digest of the run (peaks, ranges, endpoints)."""
        def rng(v):
            p = self._PRECISION.get(v, 1)
            xs = self.series[v]
            return {"min": round(min(xs), p), "max": round(max(xs), p),
                    "end": round(xs[-1], p)}
        return {v: rng(v) for v in ("glucose_mg_dl", "insulin_uU_ml", "heart_rate_bpm",
                                    "hrv_rmssd_ms", "cortisol_ug_dl", "bac_g_dl", "caffeine_mg",
                                    "glycogen_g", "ketones_mmol_l", "core_temp_c",
                                    "energy_expended_kcal", "sleep_pressure", "alertness")}


class Simulator:
    def __init__(self, params: PhysioParams, modules: list[Module] | None = None):
        self.params = params
        self.modules = modules if modules is not None else default_modules()

    def modules_for(self, outputs: list[str] | None) -> list[Module]:
        """Strategic selection: the minimal set of modules needed to produce `outputs`,
        via the transitive closure of module read/write dependencies. `outputs=None` runs
        everything. This is how the engine avoids simulating what a question doesn't need
        (e.g. a glucose question skips the alcohol, autonomic and sleep modules)."""
        if not outputs:
            return self.modules
        writers: dict[str, list[Module]] = {}
        for m in self.modules:
            for v in m.writes:
                writers.setdefault(v, []).append(m)
        needed: set = set()
        frontier = list(outputs)
        while frontier:
            var = frontier.pop()
            for m in writers.get(var, []):
                if m not in needed:
                    needed.add(m)
                    frontier.extend(m.reads)
        return [m for m in self.modules if m in needed]

    def run(self, schedule: Schedule, duration_min: float, dt: float = 1.0,
            state: BodyState | None = None, record_every: int = 5,
            outputs: list[str] | None = None, start_hour: float = 8.0) -> Trajectory:
        p = self.params
        s = state or BodyState()
        modules = self.modules_for(outputs)
        # guard against runaway / degenerate runs
        duration_min = max(0.0, min(float(duration_min), 14 * 24 * 60))   # cap at 14 days
        dt = min(max(float(dt), 0.1), 5.0)
        # start HR/BP/HRV at the person's resting values
        s.heart_rate_bpm, s.sbp_mmhg, s.dbp_mmhg = p.hr_rest, p.sbp_rest, p.dbp_rest
        s.hrv_rmssd_ms = p.hrv_rest
        s.core_temp_c = 37.0 + p.luteal_temp_offset

        times: list[float] = []
        # only numeric state fields form time-series (skip the substances dict etc.)
        _numeric = [k for k, v in s.as_dict().items() if isinstance(v, (int, float))]
        series: dict[str, list[float]] = {k: [] for k in _numeric}

        def record():
            times.append(round(s.t_min, 2))
            for k in _numeric:
                series[k].append(round(getattr(s, k), 4))

        n_steps = int(duration_min / dt)
        record()
        for step in range(n_steps):
            t0 = s.t_min
            # 1) impulse stimuli deposited at the start of this interval
            for ev in schedule.impulses_in(t0, t0 + dt):
                for m in modules:
                    m.on_impulse(s, p, ev)
            # 2) sum rate contributions from every module (shared-state coupling)
            w = schedule.window(t0)
            w.clock_hour = (start_hour * 60.0 + t0) / 60.0 % 24.0
            w.dt = dt                       # modules with tau < dt use an exact exponential step
            rates: dict[str, float] = {}
            for m in modules:
                for var, r in m.derivatives(s, p, w).items():
                    rates[var] = rates.get(var, 0.0) + r
            # 3) integrate
            s.apply_rates(rates, dt)
            s.t_min = t0 + dt
            if (step + 1) % record_every == 0:
                record()
        if times[-1] != round(s.t_min, 2):
            record()
        return Trajectory(times_min=times, series=series, params=p)


def default_modules() -> list[Module]:
    from .modules.metabolic import MetabolicModule
    from .modules.autonomic import AutonomicModule
    from .modules.stress import StressModule
    from .modules.stimulants import CaffeineModule
    from .modules.alcohol import AlcoholModule
    from .modules.circadian import CircadianModule
    from .modules.hydration import HydrationModule
    from .modules.pharmacology import PharmacologyModule
    from .modules.substrate import SubstrateModule
    from .modules.thermoregulation import ThermoregulationModule
    return [MetabolicModule(), SubstrateModule(), StressModule(), CaffeineModule(),
            AlcoholModule(), ThermoregulationModule(), HydrationModule(),
            PharmacologyModule(), AutonomicModule(), CircadianModule()]

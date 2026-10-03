"""Stimuli — the inputs that drive the simulated body over time.

Two kinds:
  - IMPULSE events (a meal, a drink, a coffee) deposit a bolus into a state compartment
    at a single instant (e.g. carbs into the gut).
  - INTERVAL events (exercise, an acute stressor, a sleep window) hold a level over a
    span of time and are exposed to the modules each step via the `InputWindow`.

A `Schedule` is just a timeline of events; `window(t)` tells the engine what's active.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---- impulse events --------------------------------------------------------
@dataclass
class Meal:
    t_min: float
    carbs_g: float
    protein_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0


@dataclass
class Food:
    t_min: float
    food: str                          # must match a key in knowledge_base/foods.json
    grams: float = 0.0                 # 0 => use the food's typical serving


@dataclass
class Drink:
    t_min: float
    standard_drinks: float = 1.0       # 14 g ethanol each


@dataclass
class Caffeine:
    t_min: float
    mg: float = 95.0                   # ~1 cup of coffee


@dataclass
class Water:
    t_min: float
    ml: float = 250.0                  # a glass of water (rehydration)


@dataclass
class Dose:
    t_min: float
    substance: str                     # must match a key in knowledge_base/substances.json
    mg: float = 0.0                    # 0 => use the substance's typical dose


@dataclass
class Insulin:
    t_min: float
    units: float = 0.0                 # exogenous rapid-acting insulin bolus (IU)


# ---- interval events -------------------------------------------------------
@dataclass
class Exercise:
    start_min: float
    end_min: float
    intensity_mets: float = 6.0        # METs (brisk walk ~4, run ~9-11)


@dataclass
class Stressor:
    start_min: float
    end_min: float
    level: float = 0.6                 # 0..1 acute psychological stress


@dataclass
class Sleep:
    start_min: float
    end_min: float


@dataclass
class Ambient:
    start_min: float
    end_min: float
    temp_c: float = 21.0               # environmental temperature (heat/cold exposure)


@dataclass
class InputWindow:
    """What is active at the current instant (passed to every module each step)."""
    exercise_mets: float = 0.0         # 0 if at rest
    stress_level: float = 0.0          # 0..1
    asleep: bool = False
    ambient_c: float = 21.0            # comfortable room temperature by default
    clock_hour: float = 8.0            # wall-clock hour (0-24) for circadian rhythms
    dt: float = 1.0                    # the integrator's current step (min). Lets a module
                                       # with a time constant shorter than dt return an
                                       # EXACT exponential update instead of a stiff rate.


@dataclass
class Schedule:
    meals: list[Meal] = field(default_factory=list)
    foods: list[Food] = field(default_factory=list)
    drinks: list[Drink] = field(default_factory=list)
    caffeine: list[Caffeine] = field(default_factory=list)
    water: list[Water] = field(default_factory=list)
    doses: list[Dose] = field(default_factory=list)
    insulin: list[Insulin] = field(default_factory=list)
    exercise: list[Exercise] = field(default_factory=list)
    stressors: list[Stressor] = field(default_factory=list)
    sleep: list[Sleep] = field(default_factory=list)
    ambient: list[Ambient] = field(default_factory=list)

    def add(self, ev) -> "Schedule":
        {Meal: self.meals, Food: self.foods, Drink: self.drinks, Caffeine: self.caffeine,
         Water: self.water, Dose: self.doses, Insulin: self.insulin, Exercise: self.exercise,
         Stressor: self.stressors, Sleep: self.sleep, Ambient: self.ambient}[type(ev)].append(ev)
        return self

    def impulses_in(self, t0: float, t1: float):
        """Yield (event) for impulse events with t in [t0, t1)."""
        for ev in (*self.meals, *self.foods, *self.drinks, *self.caffeine, *self.water,
                   *self.doses, *self.insulin):
            if t0 <= ev.t_min < t1:
                yield ev

    def window(self, t: float) -> InputWindow:
        mets = max((e.intensity_mets for e in self.exercise if e.start_min <= t < e.end_min),
                   default=0.0)
        stress = max((s.level for s in self.stressors if s.start_min <= t < s.end_min),
                     default=0.0)
        asleep = any(s.start_min <= t < s.end_min for s in self.sleep)
        ambient = next((a.temp_c for a in self.ambient if a.start_min <= t < a.end_min), 21.0)
        return InputWindow(exercise_mets=mets, stress_level=stress, asleep=asleep,
                           ambient_c=ambient)

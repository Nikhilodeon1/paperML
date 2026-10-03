"""Horizon body-simulation framework.

An app-agnostic engine that evolves a shared physiological BodyState forward in time under
stimuli (meals, drinks, caffeine, exercise, stress, sleep), with coupled mechanistic
modules and per-user personalization. The health-prediction app is its first client; the
same engine is meant to power many.

    from simulation import Simulator, PhysioParams, Schedule, Meal, Exercise
    sim = Simulator(PhysioParams.from_profile(user["profile"]))
    traj = sim.run(Schedule().add(Meal(0, carbs_g=60)), duration_min=180)
"""

from .state import BodyState
from .params import PhysioParams
from .inputs import (Schedule, Meal, Food, Drink, Caffeine, Water, Dose, Insulin, Exercise,
                     Stressor, Sleep, Ambient, InputWindow)
from .engine import Simulator, Trajectory, default_modules
from .module import Module

__all__ = ["BodyState", "PhysioParams", "Schedule", "Meal", "Food", "Drink", "Caffeine",
           "Water", "Dose", "Insulin", "Exercise", "Stressor", "Sleep", "Ambient", "InputWindow",
           "Simulator", "Trajectory", "default_modules", "Module"]

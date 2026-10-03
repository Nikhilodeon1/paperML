"""Module base — one physiological subsystem.

A module contributes rate-of-change terms for the state variables it governs and, at
event time, may deposit boluses from impulse stimuli. Modules never see each other
directly: they communicate ONLY through the shared BodyState, which is what makes the
system coupled and extensible — add a module, have it read/write shared variables, and
it automatically interacts with everything else.
"""

from __future__ import annotations

from .state import BodyState
from .params import PhysioParams
from .inputs import InputWindow


class Module:
    name: str = "module"
    # State variables this module writes, and the OTHER-module state it reads. The engine
    # uses these to run only the modules needed for a requested output (strategic sim).
    writes: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        """Return {state_variable: d/dt contribution} at the current instant. The engine
        SUMS these across all modules before integrating, so several modules can drive the
        same variable (e.g. glucose is pushed down by insulin action and up by cortisol)."""
        return {}

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        """Apply an instantaneous stimulus (meal/drink/caffeine) to the state. Modules
        ignore events they don't care about."""
        return None

"""Alcohol module — Widmark pharmacokinetics inside the shared state.

A drink deposits ethanol into the gut; it is absorbed first-order into blood and cleared
zero-order (the hallmark of alcohol metabolism) at the person's elimination rate. This is
the same Widmark model already validated in `modules/hepatic.py`, re-expressed as a state
module so BAC coexists with — and can couple to — glucose, HR and sleep. Personal `beta`
(elimination) and `r` (distribution) come from PhysioParams.
"""

from __future__ import annotations

from ..module import Module
from ..state import BodyState
from ..params import PhysioParams
from ..inputs import InputWindow, Drink

_GRAMS_PER_DRINK = 14.0


class AlcoholModule(Module):
    name = "alcohol"
    writes = ("bac_g_dl", "gut_alcohol_g")
    reads = ()

    def on_impulse(self, s: BodyState, p: PhysioParams, event) -> None:
        if isinstance(event, Drink):
            s.gut_alcohol_g += event.standard_drinks * _GRAMS_PER_DRINK

    def derivatives(self, s: BodyState, p: PhysioParams, w: InputWindow) -> dict[str, float]:
        absorb_g = p.alcohol_absorption * s.gut_alcohol_g           # g/min into blood
        # grams -> BAC(g/dL): Vd = r * body water mass; 1 g in (r*W) kg -> g/dL via /10
        vd_dl = p.widmark_r * p.weight_kg * 10.0
        d_bac_from_abs = absorb_g / vd_dl
        elim = p.alcohol_beta_g_dl_min if s.bac_g_dl > 0 else 0.0   # zero-order
        return {"gut_alcohol_g": -absorb_g, "bac_g_dl": d_bac_from_abs - elim}

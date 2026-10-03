"""Layer 1 — Knowledge base.

Static, shared physiological coefficients/equations with citations. NOT a model.
Modules (Layer 2) query this for their baseline parameters; per-user adjustments
live in Layer 3 (personalization), never here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_KB_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Parameter:
    """A single population-level coefficient with its uncertainty and source.

    `population_sd` is the prior spread: it drives how wide a confidence interval
    starts, and serves as the prior for Layer 3 Bayesian personalization.
    """

    name: str
    value: float
    unit: str
    population_sd: float
    plausible_range: tuple[float, float]
    citation: str
    notes: str = ""

    def clamp(self, x: float) -> float:
        lo, hi = self.plausible_range
        return max(lo, min(hi, x))


@lru_cache(maxsize=None)
def load_system(system: str) -> dict[str, Parameter]:
    """Load all parameters for one physiological system (e.g. 'hepatic').

    Raises if any parameter is missing a citation — uncited coefficients are
    forbidden by the brief (section 3).
    """
    path = _KB_DIR / f"{system}.json"
    if not path.exists():
        raise FileNotFoundError(f"No knowledge base file for system '{system}': {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    params: dict[str, Parameter] = {}
    for name, p in raw.get("parameters", {}).items():
        if not p.get("citation"):
            raise ValueError(f"Parameter '{name}' in {path.name} has no citation.")
        params[name] = Parameter(
            name=name,
            value=float(p["value"]),
            unit=p["unit"],
            population_sd=float(p["population_sd"]),
            plausible_range=tuple(p["plausible_range"]),  # type: ignore[arg-type]
            citation=p["citation"],
            notes=p.get("notes", ""),
        )
    return params


def get(system: str, name: str) -> Parameter:
    """Convenience: fetch one parameter."""
    return load_system(system)[name]


@lru_cache(maxsize=None)
def load_raw(system: str) -> dict:
    """Return the full parsed JSON for a system, including nested sections such as
    `cross_system_modifiers` and `biomarkers` that `load_system` does not flatten."""
    path = _KB_DIR / f"{system}.json"
    if not path.exists():
        raise FileNotFoundError(f"No knowledge base file for system '{system}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_from_dict(name: str, p: dict) -> Parameter:
    """Build a Parameter from a raw dict (e.g. a nested modifier). Citation required."""
    if not p.get("citation"):
        raise ValueError(f"Parameter '{name}' has no citation.")
    return Parameter(
        name=name,
        value=float(p["value"]),
        unit=p["unit"],
        population_sd=float(p["population_sd"]),
        plausible_range=tuple(p["plausible_range"]),  # type: ignore[arg-type]
        citation=p["citation"],
        notes=p.get("notes", ""),
    )

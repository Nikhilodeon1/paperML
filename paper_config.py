"""Central configuration for the Horizon SBI paper branch.

Single source of truth for random seeds, parameter prior ranges, and artifact paths, so every
result produced by ``evaluation/reproduce_all.py`` is deterministic and consistent across the
three estimators (SMC / RF / SNPE).

Import and call :func:`set_all_seeds` at the top of every script that has any stochastic step.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

SEED = 42

# --- Parameter prior support ------------------------------------------------------------------
# The SINGLE canonical range used by SMC, RF and SNPE alike, so the three estimators are directly
# comparable (Table 1). This matches ``personalization/npe.py::PRIOR`` (the RF baseline this work
# is measured against) and spans the RF-shrinkage probe points up to Si=1.6 (Figure 3).
#
# NOTE (brief vs codebase): the implementation brief floated three different ranges for Si
# (``[0.1, 2.5]`` in Block 1b, ``particle_fit``'s ``[0.30, 1.40]``, and npe.py's ``[0.30, 1.60]``).
# We standardize on npe.py's range because (a) a head-to-head comparison requires one shared
# prior, and (b) the RF-shrinkage analysis probes true Si up to 1.6. Documented here so the
# choice is explicit and reproducible.
PARAM_NAMES = ["insulin_sensitivity", "gastric_emptying", "carb_absorption"]
PRIOR: dict[str, tuple[float, float]] = {
    "insulin_sensitivity": (0.30, 1.60),
    "gastric_emptying": (0.015, 0.040),
    "carb_absorption": (0.012, 0.032),
}

# Meal carbohydrate range sampled during training-data generation (grams).
CARB_RANGE = (30.0, 90.0)

# --- Paths ------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "evaluation" / "artifacts"
FIGURES = ROOT / "evaluation" / "figures"
SNPE_POSTERIOR = ARTIFACTS / "snpe_posterior.pkl"


def ensure_dirs() -> None:
    """Create the artifact/figure output directories if they do not exist."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int = SEED) -> None:
    """Lock every stochastic source used in the paper (python ``random``, numpy, torch)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def prior_bounds() -> tuple[list[float], list[float]]:
    """Return ``(low, high)`` lists over :data:`PARAM_NAMES` in canonical order."""
    lo = [PRIOR[k][0] for k in PARAM_NAMES]
    hi = [PRIOR[k][1] for k in PARAM_NAMES]
    return lo, hi

"""Does the bounded-interval result depend on the confidence level of the profile interval?

The plan fixes the interval level at 95% (a rise of 1.92 in the negative log-likelihood). A reviewer can ask
whether "no interval for k_e or k_a is bounded" and "S_I is bounded for a minority" are artefacts of that
choice. The stored profiles contain every grid value, so the classification is recomputed at the 90% and 99%
levels without running anything. Report-only (Amendment 5); no threshold of the plan is touched.

Run:  python -m evaluation.delta_sensitivity
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from evaluation import profile_lik
from evaluation.identifiability_tools import classify_profile
from evaluation.results_io import largest_matching
from evaluation.stats_utils import wilson_ci

LEVELS = (0.90, 0.95, 0.99)
PARAMS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")


def delta_for(level: float) -> float:
    """Half the chi-square quantile with one degree of freedom."""
    return 0.5 * float(stats.chi2.ppf(level, 1))


def summarize(box: float = 1.0, objective: str = "iauc") -> dict:
    config = {**profile_lik.default_config(), "objective": objective, "bounds_scale": box}
    rows = list(largest_matching(profile_lik.ANALYSIS_ID, lambda p: profile_lik._matches(p, config)).values())
    if not rows:
        return {"n_subjects": 0}
    out = {"n_subjects": len(rows), "objective": objective, "bounds_scale": box, "levels": {}}
    for level in LEVELS:
        delta = delta_for(level)
        block = {"delta": delta, "parameters": {}}
        for parameter in PARAMS:
            verdicts, interior = [], []
            for r in rows:
                e = r["profiles"][parameter]
                n_in = e["n_in_box"]
                v = classify_profile(np.asarray(e["grid"])[:n_in], np.asarray(e["profile"])[:n_in],
                                     delta=delta, reference=e["reference"])["verdict"]
                verdicts.append(v)
                interior.append(r["distance_to_bound"][parameter] > 0.01)
            verdicts, interior = np.array(verdicts), np.array(interior)
            bounded = verdicts == "identifiable"
            ci_all = wilson_ci(int(bounded.sum()), len(rows))
            n_in = int(interior.sum())
            ci_in = wilson_ci(int((bounded & interior).sum()), n_in) if n_in else None
            block["parameters"][parameter] = {
                "bounded": int(bounded.sum()), "fraction": float(bounded.mean()),
                "wilson": [ci_all.low, ci_all.high],
                "interior_n": n_in, "interior_bounded": int((bounded & interior).sum()),
                "interior_fraction": (float((bounded & interior).sum() / n_in) if n_in else None),
                "interior_wilson": [ci_in.low, ci_in.high] if ci_in else None}
        out["levels"][f"{int(round(100 * level))}"] = block
    return out

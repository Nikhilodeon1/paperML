"""Collects every Phase 2 / Phase 4c number into one JSON file, from stored results only.

Nothing is fitted here. Run after the compute has finished:

    python -m evaluation.phase2_summary            # writes results/phase2_summary.json

Each section is computed by the module named in its key, so the report can quote the file and a reader
can follow any number to the code that made it.
"""
from __future__ import annotations

import json

import numpy as np

from evaluation import bound_handling as bh
from evaluation import ladder, optimizer_check, prediction_stats as ps, profile_lik
from evaluation.results_io import RESULTS, largest_matching

BOXES = (0.5, 1.0, 2.0)
PARAMS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")


def profile_quality(scale: float) -> dict:
    rows = largest_matching("A5_profile", lambda p: p["objective"] == "iauc"
                            and abs(p["bounds_scale"] - scale) < 1e-12)
    out = {}
    for parameter in PARAMS:
        improvement = [r["profiles"][parameter]["loss_at_theta_hat"]
                       - min(r["profiles"][parameter]["profile"]) for r in rows.values()]
        rise = [r["profiles"][parameter]["classification"]["max_rise"] for r in rows.values()]
        entry = {"improvement_over_theta_hat_max": float(np.max(improvement)),
                 "improvement_over_theta_hat_median": float(np.median(improvement)),
                 "max_rise_in_box_median": float(np.median(rise)),
                 "max_rise_in_box_q90": float(np.percentile(rise, 90))}
        if parameter != "insulin_sensitivity":
            verdicts = [r["profiles"][parameter]["extended"]["verdict"] for r in rows.values()]
            entry["four_times_extension"] = {v: verdicts.count(v) for v in set(verdicts)}
        out[parameter] = entry
    return out


def eigenvalues(scale: float, objective: str = "iauc") -> dict:
    rows = ladder.a4_results(objective, scale)
    ev = np.array([r["fisher"]["eigenvalues"] for r in rows.values()])
    return {f"lambda{i + 1}": {"median": float(np.median(ev[:, i])),
                               "q1": float(np.percentile(ev[:, i], 25)),
                               "q3": float(np.percentile(ev[:, i], 75))} for i in range(3)}


def main() -> None:
    summary = {
        "bounds_sweep_fisher_fit": bh.sweep(),
        "eigenvalues": {f"{s:g}x": eigenvalues(s) for s in BOXES},
        "profile_likelihood": {},
        "profile_quality": {f"{s:g}x": profile_quality(s) for s in BOXES},
        "optimizer_check": {},
        "ladder": {f"{s:g}x": ladder.summarize(s) for s in BOXES},
        "gradient_diagnostic": {f"{s:g}x": bh.projected_gradient_check(s) for s in BOXES},
        "pinned_bias": {f"{s:g}x": bh.pinned_bias(s) for s in (1.0, 2.0)},
        "prediction": ps.analyse(),
        "prediction_peak_time": ps.analyse(metric="peak"),
        "prediction_trace_rmse": ps.analyse(metric="trace"),
        "inference_gap": ps.inference_gap(),
        "steps_sensitivity": ps.steps_sensitivity(),
    }
    config = profile_lik.default_config()
    summary["profile_likelihood_trace"] = {}
    summary["profile_likelihood_iauc_centroid"] = {}
    for s in BOXES:
        summary["profile_likelihood"][f"{s:g}x"] = profile_lik.summarize({**config, "bounds_scale": s})
        for objective, key in (("trace", "profile_likelihood_trace"),
                               ("iauc_centroid", "profile_likelihood_iauc_centroid")):
            summary[key][f"{s:g}x"] = profile_lik.summarize(
                {**config, "bounds_scale": s, "objective": objective})
        summary["optimizer_check"][f"{s:g}x"] = optimizer_check.summarize(
            {**optimizer_check.default_config(), "bounds_scale": s})
    from evaluation import moment_checks
    summary["moment_checks"] = moment_checks.summarize()
    summary["moments_of_gut_input"] = moment_checks.moments_check()
    summary["generic_rank"] = {
        k: {"rank_counts": d["rank_counts"], "smallest_over_largest_median": d["smallest_over_largest_median"],
            "singular_values_median": d["singular_values"]["median"]}
        for k, d in largest_matching("A3_generic_rank", lambda p: "observables" in p).items()}
    summary["gut_sweep"] = next(iter(largest_matching(
        "A2_gut_sweep", lambda p: "sweep" in p).values()), {})
    summary["swap_check"] = next(iter(largest_matching(
        "A2_swap_check", lambda p: "pairs" in p).values()), {})
    path = RESULTS / "phase2_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Completeness manifest: which units every reported sweep must contain, and a check that fails loudly on any gap.

A sweep is complete when the set of subject IDs with a stored result equals the set of subject IDs of its cohort. Counts
are not enough, and a sweep that is merely "mostly there" is not summarised: interrupted runs once left claim files that
made later runs skip units while reporting no failure.

Run:  python -m evaluation.manifest [--set phase21|stage2]   # prints every registered sweep, writes results/manifest.json,
                                               # and exits 1 if any sweep is incomplete
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

from evaluation import profile_lik
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS

OUT = RESULTS / "manifest.json"


class IncompleteSweep(RuntimeError):
    """Raised when a summary would be built from a sweep with missing or unexpected units."""


def converged(replica: dict | None, **extra) -> dict:
    """The trace-coordinate profile configuration of the converged protocol."""
    return {**profile_lik.default_config(), "bounds_scale": 1.0, "objective": "trace", "parameterization": "coords",
            "polish": True, "grid_points": 29, "inner_solver": "lbfgsb", "replica": replica, **extra}


def phase21_sweeps() -> dict[str, dict]:
    """The converged trace sweeps of Amendment 17, under the job names of the pod scripts."""
    shanghai = {"cohort": "shanghai", "min_meals": 10}
    out = {"real": converged(None),
           "hall_real": converged(None, cohort="hall", min_meals=5),
           "shanghai_real": converged(None, **shanghai)}
    for cv in (0.0, 0.75):
        out[f"shanghai_rep{cv}"] = converged({"seed": 0, "cgm": True, "carb_cv": cv}, **shanghai)
    for cv in (0.0, 0.1, 0.25, 0.5, 0.75):
        out[f"fit{cv}_s0"] = converged({"seed": 0, "cgm": True, "carb_cv": cv})
    for cv in (0.0, 0.25, 0.75):
        for seed in (0, 1):
            out[f"rand{cv}_s{seed}"] = converged({"seed": seed, "cgm": True, "carb_cv": cv, "truth": "random"})
    for cv in (0.0, 0.75):
        for seed in (1, 2):
            out[f"fit{cv}_s{seed}"] = converged({"seed": seed, "cgm": True, "carb_cv": cv})
    return out


def _stage2(objective: str, replica: dict | None = None, parameterization: str = "rates", **extra) -> dict:
    """A rate-coordinate (or centroid, or residual-gut) profile under the converged protocol. The grid stops at the box:
    points beyond it would lower the reference value that the in-box interval is read against."""
    return {**converged(replica), "objective": objective, "parameterization": parameterization, "extension_points": 0,
            "lbfgsb_restarts": 2, **extra}


PERTURBED = ("glucose_effectiveness", "x_decay", "insulin_clearance", "insulin_secretion", "x_gain")


def _replica(seed: int = 0, carb_cv: float = 0.25, **extra) -> dict:
    return {"seed": seed, "cgm": True, "carb_cv": carb_cv, **extra}


def stage2_sweeps() -> dict[str, dict]:
    """Every old-protocol result the paper keeps, rerun under the converged protocol (decision rules R5 to R10)."""
    out = {f"iauc_box{box:g}": _stage2("iauc", bounds_scale=box) for box in (1.0, 2.0, 4.0)}
    for cohort, min_meals in (("shanghai", 10), ("hall", 5)):
        for box in (1.0, 2.0):
            out[f"iauc_{cohort}_box{box:g}"] = _stage2("iauc", cohort=cohort, min_meals=min_meals, bounds_scale=box)
    for seed in range(5):
        out[f"iauc_rep_s{seed}"] = _stage2("iauc", _replica(seed))
    out["iauc_rep_exact"] = _stage2("iauc", _replica(carb_cv=0.0))
    for scale in (0.5, 0.25):
        out[f"iauc_rep_noise{scale:g}"] = _stage2("iauc", _replica(cgm_scale=scale))
    for seed in (0, 1):
        out[f"iauc_synth_s{seed}"] = _stage2("iauc", _replica(seed, truth="random"))
    for constant in PERTURBED:
        for factor in (0.7, 1.3):
            out[f"iauc_pert_{constant}_{factor:g}"] = _stage2("iauc", _replica(perturb={constant: factor}))
    out["centroid_real"] = _stage2("iauc_centroid", parameterization="coords")
    out["centroid_rep"] = _stage2("iauc_centroid", _replica(), parameterization="coords")
    for seed, residual in ((0, 0.0), (0, 0.05), (0, 0.1), (0, 0.2), (1, 0.1)):
        replica = _replica(seed, carb_cv=0.0, truth="random")
        if residual:
            replica["residual_gut"] = residual
        out[f"residual{residual:g}_s{seed}"] = _stage2("trace", replica)
    return out


@lru_cache(maxsize=None)
def _cohort_units(cohort: str, min_meals: int, limit) -> tuple[str, ...]:
    return tuple(profile_lik.units({**profile_lik.default_config(), "cohort": cohort, "min_meals": min_meals,
                                    "limit": limit}))


def expected_units(config: dict) -> list[str]:
    return list(_cohort_units(config["cohort"], config["min_meals"], config.get("limit")))


def found_units(config: dict) -> set[str]:
    return {row["subject_id"] for row in _rows(config)}


def check(config: dict) -> dict:
    expected, found = expected_units(config), found_units(config)
    return {"expected": len(expected), "found": len(found),
            "missing": sorted(set(expected) - found), "unexpected": sorted(found - set(expected))}


def is_complete(report: dict) -> bool:
    return report["expected"] > 0 and not report["missing"] and not report["unexpected"]


def spurious_bounded(rows: list[dict]) -> list[tuple[str, str]]:
    """(subject, parameter) pairs classed bounded although no in-box grid point lies below the threshold, which can
    happen when the reference value comes from outside the box. Such an interval is empty inside the box."""
    flagged = []
    for row in rows:
        for name, entry in row["profiles"].items():
            verdict = entry["classification"]
            if verdict["verdict"] == "identifiable" and min(entry["profile"][: entry["n_in_box"]]) > verdict["threshold"]:
                flagged.append((row["subject_id"], name))
    return flagged


def require(sweeps: dict[str, dict]) -> None:
    """Raise `IncompleteSweep`, naming every incomplete sweep and its first missing units."""
    gaps = []
    for name, config in sweeps.items():
        report = check(config)
        if not is_complete(report):
            gaps.append(f"{name}: {report['found']}/{report['expected']} units"
                        f", missing {report['missing'][:5]}, unexpected {report['unexpected'][:5]}")
    if gaps:
        raise IncompleteSweep("incomplete sweeps:\n  " + "\n  ".join(gaps))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="group", choices=("phase21", "stage2", "all"), default="all")
    group = ap.parse_args(argv).group
    sweeps = {}
    if group in ("phase21", "all"):
        sweeps.update(phase21_sweeps())
    if group in ("stage2", "all"):
        sweeps.update(stage2_sweeps())
    reports = {name: check(config) for name, config in sweeps.items()}
    for name, config in sweeps.items():
        reports[name]["spurious_bounded"] = spurious_bounded(_rows(config))
    OUT.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    for name, report in reports.items():
        status = "complete" if is_complete(report) else "INCOMPLETE"
        if report["spurious_bounded"]:
            status = "SPURIOUS"
        print(f"{status:10s} {name:34s} {report['found']}/{report['expected']}")
    bad = sum(not is_complete(r) or bool(r["spurious_bounded"]) for r in reports.values())
    print(f"{len(reports) - bad} of {len(reports)} sweeps complete and clean")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

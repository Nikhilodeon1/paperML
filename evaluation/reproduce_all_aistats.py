"""Reproduce the revision's analyses end to end.

Full mode runs every queued analysis in priority order with the resumable runner, then builds the paper
assets and the final report. It is the same list `scripts/pod_run_all.sh` runs on a rented node.

Quick mode is a smoke test: every analysis is run on three subjects (`--gate 3`) with the cheapest
settings that still exercise the whole code path, into a TEMPORARY results directory so nothing in
`results/` is touched, and the run fails if any unit fails. It proves the pipeline runs on the machine
it is run on; it produces no number for the paper.

Run:  python -m evaluation.reproduce_all_aistats --quick
      python -m evaluation.reproduce_all_aistats --workers 30
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from evaluation.results_io import ROOT

BOXES = ("1.0", "2.0", "0.5")

# (module, extra --set arguments). Order is priority order: the prediction comparison first, then the
# identifiability analyses, the ladder, the diagnostics, and the optimizer-budget check last.
JOBS: list[tuple[str, list[str]]] = [
    ("evaluation.generic_rank", []),
    ("evaluation.prediction_cv", []),
    *[("evaluation.fisher_full", [f"objective={o}", f"bounds_scale={b}"])
      for o in ("iauc", "iauc_centroid", "trace") for b in BOXES],
    *[("evaluation.profile_lik", [f"objective={o}", f"bounds_scale={b}"])
      for o in ("iauc", "trace", "iauc_centroid") for b in BOXES],
    *[("evaluation.ladder", [f"bounds_scale={b}"]) for b in BOXES],
    *[("evaluation.gradient_diag", [f"bounds_scale={b}"]) for b in BOXES],
    *[("evaluation.optimizer_check", [f"bounds_scale={b}"]) for b in BOXES],
    ("evaluation.moment_checks", ["max_meals=8"]),
    ("evaluation.prediction_cv", ["steps=500", "repeats=3", 'cells=["grad3","grad1"]']),
]

# Cheap overrides for the smoke test only.
QUICK_SETS = {
    "evaluation.generic_rank": ["n_theta=20"],
    "evaluation.prediction_cv": ["repeats=1"],
    "evaluation.profile_lik": ["steps=20", "grid_points=7", "extension_points=2", "fit_steps=60",
                               "pilot_steps=30"],
    "evaluation.fisher_full": ["steps=60", "pilot_steps=30"],
    "evaluation.gradient_diag": ["steps=30"],
    "evaluation.moment_checks": ["max_meals=2", "windows=[180]"],
    "evaluation.optimizer_check": ["maxiter=20", "n_random_starts=1"],
}


def _run(module: str, sets: list[str], workers: int, env: dict, gate: int | None) -> int:
    command = [sys.executable, "-m", "evaluation.runner", module, "--workers", str(workers)]
    if gate:
        command += ["--gate", str(gate)]
    for item in sets:
        command += ["--set", item]
    print(f"\n=== {time.strftime('%H:%M:%S')} {module} {' '.join(sets)}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="3-subject smoke test in a temporary directory")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-assets", action="store_true")
    args = ap.parse_args(argv)

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failures = []
    if args.quick:
        workers = args.workers or 3
        scratch = tempfile.mkdtemp(prefix="aistats_quick_")
        env["HORIZON_RESULTS_DIR"] = str(Path(scratch) / "results")
        env["HORIZON_LOGS_DIR"] = str(Path(scratch) / "logs")
        print(f"quick mode: results go to {scratch}")
        # The ladder and the profile likelihood read earlier Fisher results, so a quick run uses the
        # first-listed box only and each objective once.
        seen = set()
        for module, sets in JOBS:
            key = (module, tuple(s for s in sets if s.startswith("objective")))
            if module in ("evaluation.fisher_full", "evaluation.profile_lik") and key in seen:
                continue
            if module in ("evaluation.ladder", "evaluation.gradient_diag", "evaluation.optimizer_check") \
                    and (module,) in seen:
                continue
            if "steps=500" in sets:
                continue
            seen.add(key if module in ("evaluation.fisher_full", "evaluation.profile_lik") else (module,))
            quick = [s for s in sets if not s.startswith("bounds_scale")] + QUICK_SETS.get(module, [])
            quick.append("limit=3")
            if module == "evaluation.generic_rank":
                quick = QUICK_SETS[module]
            if _run(module, quick, workers, env, gate=3 if module != "evaluation.generic_rank" else None):
                failures.append(module)
        print("\nquick run:", "FAILED " + ", ".join(failures) if failures else "all analyses ran")
        return 1 if failures else 0

    workers = args.workers or max(1, (os.cpu_count() or 2) // 2 - 1)
    for module, sets in JOBS:
        if _run(module, sets, workers, env, gate=None):
            failures.append(f"{module} {' '.join(sets)}")
    if not args.skip_assets:
        for module in ("evaluation.make_paper_assets", "evaluation.final_report"):
            if subprocess.run([sys.executable, "-m", module], cwd=ROOT, env=env).returncode:
                failures.append(module)
    print("\nFAILED: " + "; ".join(failures) if failures else "\nall steps ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

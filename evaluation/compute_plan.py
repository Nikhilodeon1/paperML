"""Combine the measured unit costs with the measured parallel speed-up into one honest plan.

`benchmark_compute` divides serial time by the number of worker processes, which assumes a worker
is as fast when fourteen of them run as when one does. `benchmark_scaling` measures that it is not:
on 8 physical cores presented as 16 logical ones, fourteen workers deliver about eight times the
throughput of one, not fourteen. This module divides by the MEASURED speed-up instead, and flags any
job whose projection exceeds the agreed 12-hour ceiling so a reduced plan can be proposed before
the job starts rather than after it overruns.

Run:  python -m evaluation.compute_plan
      python -m evaluation.compute_plan --workers 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.benchmark_compute import compute_plan
from evaluation.results_io import RESULTS

BUDGET_HOURS = 12.0


def _latest(analysis_id: str, unit: str) -> dict:
    """The most recent result file for an analysis, across configuration hashes."""
    candidates = sorted((RESULTS / analysis_id).glob(f"*/{unit}.json"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"no {analysis_id}/{unit}.json under {RESULTS}; run the benchmark first.")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))["payload"]


def build(workers: int | None = None) -> dict:
    benchmark = _latest("A0_benchmark", "benchmark")
    scaling = _latest("A0_scaling", "scaling")

    runs = {r["n_workers"]: r for r in scaling["runs"]}
    if workers is None:
        workers = scaling.get("recommended_workers") or max(runs)
    if workers not in runs:
        raise SystemExit(f"no scaling measurement for {workers} workers; measured {sorted(runs)}")
    effective = runs[workers].get("effective_workers")
    if not effective:
        raise SystemExit("the scaling result has no single-worker baseline to normalize against")

    plan = compute_plan(benchmark["unit_cost_seconds"], workers=effective)
    over = [job for job in plan if job["projected_wall_hours"] > BUDGET_HOURS]
    return {
        "workers": workers,
        "effective_parallelism": effective,
        "nominal_parallelism": workers,
        "efficiency": effective / workers,
        "unit_cost_seconds": benchmark["unit_cost_seconds"],
        "scaling_measured": {k: round(v.get("effective_workers", float("nan")), 2)
                             for k, v in sorted(runs.items())},
        "plan": plan,
        "total_wall_hours": round(sum(j["projected_wall_hours"] for j in plan), 2),
        "over_budget": [j["job"] for j in over],
        "budget_hours": BUDGET_HOURS,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    result = build(args.workers)

    print("=" * 88)
    print(f"COMPUTE PLAN  --  {result['workers']} worker processes, measured speed-up "
          f"{result['effective_parallelism']:.1f}x ({result['efficiency']:.0%} efficiency)")
    print(f"  measured speed-up by worker count: {result['scaling_measured']}")
    print("-" * 88)
    print(f"  {'job':58} {'serial h':>9} {'wall h':>8}")
    for job in result["plan"]:
        flag = "  OVER BUDGET" if job["projected_wall_hours"] > result["budget_hours"] else ""
        print(f"  {job['job'][:58]:58} {job['serial_hours']:>9.2f} "
              f"{job['projected_wall_hours']:>8.2f}{flag}")
    print("-" * 88)
    print(f"  {'TOTAL':58} {'':>9} {result['total_wall_hours']:>8.2f}")
    if result["over_budget"]:
        print(f"  OVER THE {result['budget_hours']:.0f}-HOUR CEILING: "
              + "; ".join(result["over_budget"]))
    print("=" * 88)
    print(json.dumps({k: v for k, v in result.items() if k != "plan"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

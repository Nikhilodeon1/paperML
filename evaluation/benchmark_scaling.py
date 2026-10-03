"""How much throughput extra worker processes actually buy on this machine.

The single-process benchmark gives a unit cost; the compute plan divides by a worker count. That
division is only honest if a worker runs as fast when fourteen of them are running as it does
alone, and on a machine with 8 physical cores presented as 16 logical ones it does not. This script
measures the degradation instead of assuming it: it runs the same fit in `n` concurrent
single-threaded processes and reports throughput in fits per minute.

The result is the number the plan should divide by -- an EFFECTIVE worker count, which is
`throughput(n) / throughput(1)` and is generally well below `n`.

Run:  python -m evaluation.benchmark_scaling --workers 1 4 8 14
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ID = "A0_scaling"

# One subject, one 500-step fit, and nothing else: the unit the plan is denominated in.
_WORKER_ARGS = ["-m", "evaluation.smoke_fit", "--subjects", "1", "--steps", "500"]


def _env() -> dict:
    """One BLAS and one XLA thread per worker, so `n` workers use `n` cores and not n x 16."""
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "PYTHONIOENCODING": "utf-8",
    }


def measure(n_workers: int, timeout: int = 1800) -> dict:
    """Wall time for `n_workers` concurrent fits, started together."""
    env = _env()
    start = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, *_WORKER_ARGS], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(n_workers)]
    results, failures = [], []
    for proc in procs:
        out, err = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            failures.append(err[-500:])
            continue
        try:
            results.append(json.loads(out[out.find("{"):]))
        except (ValueError, json.JSONDecodeError):
            failures.append(out[-500:])
    wall = time.perf_counter() - start

    thetas = [r["subjects"][0]["theta"] for r in results if r.get("subjects")]
    spread = 0.0
    if len(thetas) > 1:
        for key in thetas[0]:
            values = [t[key] for t in thetas]
            spread = max(spread, max(values) - min(values))

    return {
        "n_workers": n_workers,
        "wall_seconds": wall,
        "completed": len(results),
        "failed": len(failures),
        "failures": failures[:2],
        "fits_per_minute": (len(results) / wall * 60.0) if wall > 0 else float("nan"),
        "seconds_per_fit_wall": (wall / len(results)) if results else float("nan"),
        # All workers fit the SAME subject, so their answers must agree; a spread here would mean
        # the result depends on how many processes happened to be running.
        "max_theta_spread_across_workers": spread,
    }


def run(worker_counts: list[int]) -> dict:
    out = {"cores_logical": os.cpu_count(), "runs": []}
    for n in worker_counts:
        result = measure(n)
        out["runs"].append(result)
        print(f"  {n:>3} workers  wall {result['wall_seconds']:7.1f}s  "
              f"{result['fits_per_minute']:6.2f} fits/min  "
              f"{result['completed']} ok, {result['failed']} failed", flush=True)
    base = next((r["fits_per_minute"] for r in out["runs"] if r["n_workers"] == 1), None)
    if base:
        for r in out["runs"]:
            r["effective_workers"] = r["fits_per_minute"] / base
        best = max(out["runs"], key=lambda r: r["fits_per_minute"])
        out["recommended_workers"] = best["n_workers"]
        out["effective_workers_at_recommended"] = best["fits_per_minute"] / base
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 14])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    print(f"SCALING -- {os.cpu_count()} logical cores")
    result = run(args.workers)
    if "recommended_workers" in result:
        print(f"  recommended {result['recommended_workers']} workers, "
              f"effective parallelism {result['effective_workers_at_recommended']:.1f}x")
    if args.save:
        from evaluation.results_io import save_result
        print("saved", save_result(ANALYSIS_ID, result, {"workers": args.workers},
                                   unit="scaling", overwrite=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

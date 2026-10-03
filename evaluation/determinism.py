"""Run a small version of a job twice and prove the two outputs agree.

Required of every script before its full run. The point is not that floating-point arithmetic is
exactly reproducible -- it is, on one machine with one thread count -- but that nothing in the job
depends on an unseeded source: a salted `hash`, a fresh `default_rng()`, dictionary iteration over
a set, a wall-clock seed, or thread-count-dependent reduction order.

Two modes:

* `max_abs_difference(a, b)` walks two nested structures and returns the largest absolute
  difference between corresponding numbers, plus every structural mismatch it found. A structural
  mismatch is a failure however small the numbers are.
* `run_module_twice(module, args)` runs a module as a subprocess twice, with DIFFERENT
  `PYTHONHASHSEED` values, and compares the JSON it prints. The differing hash seed is what catches
  a salted `hash`; running twice in one process would not.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

__all__ = ["max_abs_difference", "run_module_twice", "assert_deterministic"]


def _walk(a, b, path: str, state: dict) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            state["mismatches"].append(
                f"{path}: keys differ ({sorted(set(a) ^ set(b))[:5]})")
            return
        for key in sorted(a, key=str):
            _walk(a[key], b[key], f"{path}.{key}", state)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            state["mismatches"].append(f"{path}: lengths {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", state)
        return
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            state["mismatches"].append(f"{path}: {a!r} vs {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return          # NaN in both places is agreement, not a difference
        diff = abs(float(a) - float(b))
        if not math.isfinite(diff):
            state["mismatches"].append(f"{path}: {a!r} vs {b!r}")
            return
        if diff > state["max_diff"]:
            state["max_diff"] = diff
            state["worst_path"] = path
        return
    if a != b:
        state["mismatches"].append(f"{path}: {a!r} vs {b!r}")


def max_abs_difference(a, b) -> dict:
    """`{max_diff, worst_path, mismatches}` between two nested structures.

    Keys that carry provenance rather than results -- timestamps, elapsed times, the environment
    block -- are excluded, because they legitimately differ between two runs and comparing them
    would make every check fail for the wrong reason.
    """
    ignore = {"timestamp", "frozen_at", "environment", "seconds", "elapsed", "wall_seconds",
              "compile_plus_fit_500_s", "profile_first_call_s", "grid_first_call_s"}

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items()
                    if k not in ignore and not str(k).endswith("_s")}
        if isinstance(obj, (list, tuple)):
            return [strip(v) for v in obj]
        return obj

    state = {"max_diff": 0.0, "worst_path": "", "mismatches": []}
    _walk(strip(a), strip(b), "", state)
    return state


def run_module_twice(module: str, args: list[str] | None = None, timeout: int = 3600,
                     hash_seeds: tuple[str, str] = ("0", "98765")) -> dict:
    """Run `python -m <module> <args>` twice and compare the JSON on stdout.

    The module must print a JSON document as the LAST thing on stdout; anything before the final
    opening brace is treated as log output and ignored, so a script may print a human-readable
    table as well.
    """
    outputs = []
    for seed in hash_seeds:
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, "-m", module, *(args or [])], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{module} failed with hash seed {seed}:\n{proc.stderr[-4000:]}")
        start = proc.stdout.find("{")
        if start < 0:
            raise RuntimeError(f"{module} printed no JSON document:\n{proc.stdout[-2000:]}")
        outputs.append(json.loads(proc.stdout[start:]))
    result = max_abs_difference(*outputs)
    result["module"] = module
    result["hash_seeds"] = list(hash_seeds)
    return result


def assert_deterministic(a, b, tol: float = 1e-6) -> None:
    """Raise with the worst offender named if two runs disagree by more than `tol`."""
    result = max_abs_difference(a, b)
    if result["mismatches"]:
        raise AssertionError("structural mismatch between runs:\n  "
                             + "\n  ".join(result["mismatches"][:10]))
    if result["max_diff"] > tol:
        raise AssertionError(
            f"runs differ by {result['max_diff']:.3e} at {result['worst_path']!r}, "
            f"tolerance {tol:.1e}")

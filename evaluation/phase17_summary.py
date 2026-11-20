"""Phase 17 (Amendment 13): do the bounded fractions of the paper depend on the 15-point grid?

H34   real data, trace coordinate profiles: the tau1 and p bounded fractions on a 29-point grid stay within 10 points of
      those on the 15-point grid of the paper (same Adam inner solver, 100 steps)
R15   the same comparison on the exact-carbohydrate and carbohydrate-error replicas
R16   the positive control of Amendment 11 (0% and 10% residual, rate coordinates) on a 29-point grid

Nothing is fitted here. A missing result set gives `{"n": 0}` for its block.

Run:  python -m evaluation.phase17_summary          # writes results/phase17_summary.json
"""
from __future__ import annotations

import json

from evaluation.phase13_summary import _block, _coords
from evaluation.phase14_summary import COORDS, class_changes
from evaluation.phase15_summary import _rates_config, p1_replica, rates_block
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS

H34_POINTS = 10.0
GRIDS = (15, 29)
REPLICAS = {"real": None, "exact": {"seed": 0, "cgm": True, "carb_cv": 0.0},
            "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}
RESIDUALS = (0.0, 0.10)


def _config(replica: dict | None, grid: int) -> dict:
    return _coords({"steps": 100, "grid_points": grid, **({"replica": replica} if replica else {})})


def r15() -> dict:
    out = {}
    for tag, replica in REPLICAS.items():
        rows = {g: _rows(_config(replica, g)) for g in GRIDS}
        out[tag] = {f"grid_{g}": _block(rows[g]) for g in GRIDS}
        out[tag]["class_changes"] = class_changes(rows[15], rows[29]) if rows[15] and rows[29] else {"n": 0}
    return out


def r16() -> dict:
    out = {}
    for residual in RESIDUALS:
        replica = p1_replica(residual)
        out[f"residual_{residual:g}"] = {f"grid_{g}": rates_block(_rows({**_rates_config(replica), "grid_points": g}))
                                         for g in GRIDS}
    return out


def _fraction(block: dict, name: str):
    return (block.get(name) or {}).get("bounded_interior", {}).get("fraction")


def h34(r15_out: dict) -> dict:
    """H34: real data, bounded fractions of tau1 and p on 29 points within 10 points of those on 15 points."""
    rule = {"points": H34_POINTS, "grid_points": [15, 29]}
    real = r15_out.get("real", {})
    a, b = real.get("grid_15", {}), real.get("grid_29", {})
    if not a.get("n") or not b.get("n"):
        return {"status": "not evaluated", "rule": rule}
    diff = {}
    for name, _ in COORDS:
        fa, fb = _fraction(a, name), _fraction(b, name)
        if fa is None or fb is None:
            return {"status": "not evaluated", "rule": rule}
        diff[name] = 100.0 * (fb - fa)
    return {"status": "met" if all(abs(v) <= H34_POINTS for v in diff.values()) else "not met", "rule": rule,
            "difference_points": diff}


def summarize() -> dict:
    first = r15()
    return {"r15": first, "h34": h34(first), "r16": r16()}


def main() -> int:
    path = RESULTS / "phase17_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

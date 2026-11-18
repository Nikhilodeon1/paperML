"""Phase 16 (Amendment 12): is the inner solver of the profiles the reason for the low-noise undercoverage?

R13 / H31  the profile at the true value, four ways (A19), at a quarter of the noise, at the estimated noise and with
           carbohydrate error: H31 is that a converged inner solve from the estimate puts the true value inside the
           delta band in at least 90% of subjects, for tau1 and for p, at a quarter of the noise
R14        the trace coordinate profiles of Phase 14 recomputed with the converged inner solver (A5, inner_solver
           "lbfgsb", tau1 and p only), on the real data and on the three replicas: bounded fractions, nominal
           coverage, coverage of the bounded intervals, class changes against the Adam profiles, points that did not converge
H32        real data: the tau1 and p bounded fractions of the converged profiles stay within 10 points of those of the
           Adam profiles at the default inner budget
H33        a quarter of the noise, exact carbohydrate, 29 grid points: nominal coverage of at least 90% for tau1 and p

Nothing is fitted here. A missing result set gives `{"n": 0}` for its block.

Run:  python -m evaluation.phase16_summary          # writes results/phase16_summary.json
"""
from __future__ import annotations

import json

from evaluation import profile_at_truth
from evaluation.identifiability_tools import CHI2_DELTA
from evaluation.phase13_summary import _block, _coords
from evaluation.phase14_summary import COORDS, class_changes, settling
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS

H31_SHARE, H32_POINTS, H33_COVERAGE = 0.90, 10.0, 0.90
PROFILE_ONLY = ["log_tau1", "log_p"]
REPLICAS = {"quarter": {"seed": 0, "cgm": True, "carb_cv": 0.0, "cgm_scale": 0.25},
            "exact": {"seed": 0, "cgm": True, "carb_cv": 0.0},
            "replica": {"seed": 0, "cgm": True, "carb_cv": 0.25}}
SETTINGS = {"real": None, **REPLICAS}


def _adam_config(replica: dict | None) -> dict:
    """The Phase 9 / Phase 14 default-budget profiles the converged ones are compared with."""
    return _coords({"steps": 100, **({"replica": replica} if replica else {})})


def _converged_config(replica: dict | None, grid_points: int = 15) -> dict:
    return _coords({"steps": 100, "inner_solver": "lbfgsb", "profile_only": PROFILE_ONLY, "grid_points": grid_points,
                    **({"replica": replica} if replica else {})})


def _not_converged(rows: list[dict]) -> dict:
    points = bad = 0
    for row in rows:
        for entry in row["profiles"].values():
            flags = entry.get("solver_converged") or []
            points += len(flags)
            bad += sum(not f for f in flags)
    return {"points": points, "not_converged": bad, "fraction": (bad / points) if points else None}


def r13() -> dict:
    return {tag: profile_at_truth.summarize({"replica": replica}) for tag, replica in REPLICAS.items()}


def h31(r13_out: dict) -> dict:
    """H31: at a quarter of the noise a converged solve from the estimate keeps the true value inside delta in >= 90%."""
    rule = {"share": H31_SHARE}
    block = r13_out.get("quarter", {})
    if not block.get("n"):
        return {"status": "not evaluated", "rule": rule}
    share = {}
    for name in profile_at_truth.COORDINATES:
        value = (block.get(name) or {}).get("lbfgsb_hat", {}).get("share_within_delta")
        if value is None:
            return {"status": "not evaluated", "rule": rule}
        share[name] = value
    return {"status": "met" if all(v >= H31_SHARE for v in share.values()) else "not met", "rule": rule, "share": share}


def r14() -> dict:
    out = {}
    for tag, replica in SETTINGS.items():
        adam = _rows(_adam_config(replica))
        solved = _rows(_converged_config(replica))
        out[tag] = {"adam": _block(adam), "lbfgsb": _block(solved),
                    "lbfgsb_settling": _not_converged(solved), "adam_settling": settling(adam),
                    "class_changes": class_changes(adam, solved) if adam and solved else {"n": 0}}
    out["quarter_fine"] = {"lbfgsb": _block(_rows(_converged_config(REPLICAS["quarter"], 29)))}
    return out


def _fraction(block: dict, name: str):
    return (block.get(name) or {}).get("bounded_interior", {}).get("fraction")


def h32(r14_out: dict) -> dict:
    """H32: real data, converged versus Adam profiles, bounded fractions within 10 points."""
    rule = {"points": H32_POINTS}
    real = r14_out.get("real", {})
    a, b = real.get("adam", {}), real.get("lbfgsb", {})
    if not a.get("n") or not b.get("n"):
        return {"status": "not evaluated", "rule": rule}
    diff = {}
    for name, _ in COORDS:
        fa, fb = _fraction(a, name), _fraction(b, name)
        if fa is None or fb is None:
            return {"status": "not evaluated", "rule": rule}
        diff[name] = 100.0 * (fb - fa)
    return {"status": "met" if all(abs(v) <= H32_POINTS for v in diff.values()) else "not met", "rule": rule,
            "difference_points": diff}


def h33(r14_out: dict) -> dict:
    """H33: a quarter of the noise, exact carbohydrate, 29 grid points, converged solver: nominal coverage >= 90%."""
    rule = {"coverage": H33_COVERAGE, "grid_points": 29}
    block = r14_out.get("quarter_fine", {}).get("lbfgsb", {})
    if not block.get("n"):
        return {"status": "not evaluated", "rule": rule}
    cover = {name: (block[name]["nominal_coverage"] or {}).get("fraction") for name, _ in COORDS}
    if any(v is None for v in cover.values()):
        return {"status": "not evaluated", "rule": rule}
    return {"status": "met" if all(v >= H33_COVERAGE for v in cover.values()) else "not met", "rule": rule,
            "coverage": cover}


def summarize() -> dict:
    first, second = r13(), r14()
    return {"r13": first, "h31": h31(first), "r14": second, "h32": h32(second), "h33": h33(second), "delta": CHI2_DELTA}


def main() -> int:
    path = RESULTS / "phase16_summary.json"
    path.write_text(json.dumps(summarize(), indent=2, default=float), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

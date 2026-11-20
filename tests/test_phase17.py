"""Phase 17 (Amendment 13): the pieces that can be tested without result files."""
from __future__ import annotations

from evaluation import hypotheses, phase17_summary as p17


def _fractions(tau, p):
    return {"n": 45, "tau1": {"bounded_interior": {"fraction": tau}}, "p": {"bounded_interior": {"fraction": p}}}


def test_h34_is_ten_points_either_way():
    ok = {"real": {"grid_15": _fractions(0.83, 0.88), "grid_29": _fractions(0.90, 0.85)}}
    assert p17.h34(ok)["status"] == "met"
    bad = {"real": {"grid_15": _fractions(0.83, 0.88), "grid_29": _fractions(0.95, 0.88)}}
    assert p17.h34(bad)["status"] == "not met"
    assert p17.h34({})["status"] == "not evaluated"


def test_configs_differ_only_by_grid_and_replica():
    a, b = p17._config(None, 15), p17._config(None, 29)
    assert a["grid_points"] == 15 and b["grid_points"] == 29 and a["parameterization"] == "coords"
    assert {k: v for k, v in a.items() if k != "grid_points"} == {k: v for k, v in b.items() if k != "grid_points"}
    assert p17._config({"seed": 0}, 15)["replica"] == {"seed": 0}


def test_h34_is_not_evaluated_without_results():
    assert hypotheses.h34({})["status"] == "not evaluated"
    met = hypotheses.h34({"phase17": {"h34": {"status": "met", "difference_points": {"tau1": 1.0, "p": -2.0}}}})
    assert met["status"] == "met"

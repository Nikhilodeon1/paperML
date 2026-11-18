"""Phase 16 (Amendment 12): the pieces that can be tested without result files."""
from __future__ import annotations

import numpy as np

from evaluation import hypotheses, phase16_summary as p16, profile_lik
from evaluation.profile_solver import profile_lbfgsb


def test_converged_profile_of_a_quadratic_is_exact():
    import jax.numpy as jnp
    centre = np.array([0.3, -0.2, 0.5])
    scale = np.array([1.0, 4.0, 9.0])

    def loss(phi):
        return 0.5 * jnp.sum(jnp.asarray(scale) * (phi - jnp.asarray(centre)) ** 2) + 0.4 * (phi[0] - centre[0]) * (phi[1] - centre[1])

    lower, upper = np.full(3, -3.0), np.full(3, 3.0)
    grid = np.linspace(-1.0, 1.5, 11)
    out = profile_lbfgsb(loss, centre, 0, grid, lower, upper)
    # profile over coordinate 0 of a quadratic: minimize over the other two analytically
    h = np.array([[scale[1], 0.0], [0.0, scale[2]]])
    for g, value in zip(grid, out["profile"]):
        d0 = g - centre[0]
        best_1 = -0.4 * d0 / scale[1]
        expected = 0.5 * scale[0] * d0 ** 2 + 0.4 * d0 * best_1 + 0.5 * scale[1] * best_1 ** 2
        assert abs(value - expected) < 1e-6
    assert all(out["converged"]) and out["optimizer"] == "lbfgsb"


def test_result_sets_are_separated_by_solver_and_grid():
    base = {**profile_lik.default_config(), "objective": "trace", "parameterization": "coords", "polish": True}
    payload = {"objective": "trace", "bounds_scale": 1.0, "cohort": "cgmacros", "parameterization": "coords",
               "inner_steps": 100, "replica": None, "carb_scale": None, "noise_structure": "homoskedastic",
               "profile_only": None, "polish": {"start_values": [0.0] * 6}}
    assert profile_lik._matches(payload, base)                                   # an old payload counts as Adam, 15 points
    assert not profile_lik._matches(payload, {**base, "inner_solver": "lbfgsb"})
    assert not profile_lik._matches(payload, {**base, "grid_points": 29})
    assert profile_lik._matches({**payload, "inner_solver": "lbfgsb", "grid_points": 29},
                               {**base, "inner_solver": "lbfgsb", "grid_points": 29})


def _share(tau, p):
    def one(v):
        return {"lbfgsb_hat": {"share_within_delta": v}}
    return {"n": 45, "log_tau1": one(tau), "log_p": one(p)}


def test_h31_needs_ninety_percent_for_both_coordinates():
    assert p16.h31({"quarter": _share(0.95, 0.91)})["status"] == "met"
    assert p16.h31({"quarter": _share(0.95, 0.80)})["status"] == "not met"
    assert p16.h31({})["status"] == "not evaluated"


def _fractions(tau, p):
    return {"n": 45, "tau1": {"bounded_interior": {"fraction": tau}}, "p": {"bounded_interior": {"fraction": p}}}


def test_h32_is_ten_points_either_way():
    ok = {"real": {"adam": _fractions(0.83, 0.88), "lbfgsb": _fractions(0.75, 0.95)}}
    assert p16.h32(ok)["status"] == "met"
    bad = {"real": {"adam": _fractions(0.83, 0.88), "lbfgsb": _fractions(0.60, 0.88)}}
    assert p16.h32(bad)["status"] == "not met"
    assert p16.h32({})["status"] == "not evaluated"


def test_h33_is_nominal_coverage_at_29_grid_points():
    def block(a, b):
        return {"n": 45, "tau1": {"nominal_coverage": {"fraction": a}}, "p": {"nominal_coverage": {"fraction": b}}}
    assert p16.h33({"quarter_fine": {"lbfgsb": block(0.93, 0.91)}})["status"] == "met"
    assert p16.h33({"quarter_fine": {"lbfgsb": block(0.93, 0.50)}})["status"] == "not met"
    assert p16.h33({})["status"] == "not evaluated"


def test_not_converged_counts_points_flagged_by_the_solver():
    rows = [{"profiles": {"a": {"solver_converged": [True, True, False]}, "b": {"solver_converged": [True]}}}]
    out = p16._not_converged(rows)
    assert out["points"] == 4 and out["not_converged"] == 1
    assert p16._not_converged([{"profiles": {"a": {}}}])["fraction"] is None


def test_hypotheses_are_not_evaluated_without_results():
    for fn in (hypotheses.h31, hypotheses.h32, hypotheses.h33):
        assert fn({})["status"] == "not evaluated"
    met = hypotheses.h31({"phase16": {"h31": {"status": "met", "share": {"log_tau1": 0.95, "log_p": 0.93}}}})
    assert met["status"] == "met"

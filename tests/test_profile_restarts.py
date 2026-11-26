"""When L-BFGS-B stops without success, other starting points are tried and the lowest value is kept."""
import numpy as np

from evaluation import profile_solver

LOWER, UPPER = np.array([0.0, 0.0]), np.array([4.0, 4.0])


def _fake(calls, converged):
    """A stand-in inner solve: starts above 2.5 in the free coordinate reach the lower basin."""
    def solve(value_and_grad, free_index, start, index, value, lower, upper, maxiter=200, project=None):
        calls.append(float(start[1]))
        loss = 1.0 if start[1] > 2.5 else 5.0
        theta = np.array([value, start[1]])
        return {"loss": loss, "theta": theta, "converged": converged, "iterations": 1}
    return solve


def _profile(monkeypatch, converged, restarts):
    calls = []
    monkeypatch.setattr(profile_solver, "fixed_coordinate_minimum", _fake(calls, converged))
    monkeypatch.setattr(profile_solver, "reduced_value_and_grad", lambda *a: (None, np.array([1])))
    out = profile_solver.profile_lbfgsb(None, np.array([2.0, 1.0]), 0, np.array([2.0]), LOWER, UPPER,
                                        restarts=restarts)
    return out, calls


def test_an_unconverged_point_takes_the_lower_value_from_a_restart(monkeypatch):
    out, calls = _profile(monkeypatch, converged=False, restarts=2)
    assert out["profile"] == [1.0]
    assert calls == [1.0, 1.0, 3.0]                       # warm start, then 25% and 75% of the box


def test_a_converged_point_is_not_restarted(monkeypatch):
    out, calls = _profile(monkeypatch, converged=True, restarts=2)
    assert out["profile"] == [5.0] and calls == [1.0]


def test_without_restarts_the_solver_behaves_as_before(monkeypatch):
    out, calls = _profile(monkeypatch, converged=False, restarts=0)
    assert out["profile"] == [5.0] and calls == [1.0]

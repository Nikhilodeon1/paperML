"""Correctness gate for the identifiability machinery, on problems with analytic answers.

Nothing in Phase 2 onward is trustworthy unless these pass. Each toy is a case where the Fisher
matrix and the profile likelihood can be written down by hand, so a bug in the generic tools shows
up as a failed assertion here rather than as a plausible-looking but wrong eigenvalue spectrum on
the real engine.

The toys are chosen to mirror the structure the paper claims about the ODE model:

* **Toy A** -- one observable that depends only on `1/k_e + 1/k_a`. The mean transit time of a
  two-compartment chain is exactly this, so an observable that sees only the mean has a flat
  direction along which the two rate constants trade off. Fisher must be rank 1, the informative
  direction must point along `(-1/k_e, -1/k_a)` in log coordinates, and the null direction along
  `(1/k_a, -1/k_e)`.
* **Toy B** -- adding a second observable that depends on `1/k_e^2 + 1/k_a^2` restores full rank,
  except exactly on the diagonal `k_e = k_a`, where the two observables become symmetric functions
  of the same quantity and the swap degeneracy returns. This is the structure behind H1.
* **Toy C** -- profile likelihoods: flat along Toy A's null direction, and exactly the analytic
  quadratic for a coefficient of a linear regression.
"""
from __future__ import annotations

import math

import pytest

from evaluation.jax_config import configure

configure()   # float64 before any jax array exists; see evaluation/jax_config.py

import jax                      # noqa: E402
import jax.numpy as jnp        # noqa: E402
import numpy as np             # noqa: E402

from evaluation.identifiability_tools import (                      # noqa: E402
    CHI2_DELTA, classify_profile, fisher_matrix, numerical_rank, profile_likelihood,
    schur_complement,
)

# --- the toy observables ------------------------------------------------------------------------

def mean_transit(theta):
    """Toy A: `h = 1/k_e + 1/k_a`, the mean transit time of a two-compartment chain."""
    return jnp.array([1.0 / theta[0] + 1.0 / theta[1]])


def two_moments(theta):
    """Toy B: the mean transit time together with `1/k_e^2 + 1/k_a^2`."""
    return jnp.array([1.0 / theta[0] + 1.0 / theta[1],
                      1.0 / theta[0] ** 2 + 1.0 / theta[1] ** 2])


THETA = jnp.array([0.025, 0.018])      # the engine's own gastric / absorption scale


# --- Toy A --------------------------------------------------------------------------------------

def test_toy_a_fisher_is_rank_one():
    """A single observable cannot inform two parameters: exactly one non-zero eigenvalue."""
    res = fisher_matrix(mean_transit, THETA, sigma=1.0, log_coords=True,
                        names=("k_e", "k_a"))
    assert res.eigenvalues.shape == (2,)
    assert res.eigenvalue_ratio < 1e-8, (
        f"expected a flat direction, got eigenvalues {res.eigenvalues}")
    assert numerical_rank(np.linalg.svd(res.fisher, compute_uv=False), rtol=1e-6) == 1


def test_toy_a_informative_direction_is_analytic():
    """The informative eigenvector points along `(-1/k_e, -1/k_a)` in log coordinates.

    In log coordinates `d h / d log k = -1/k`, so the Jacobian row IS that vector and the top
    eigenvector of `J^T J` must be parallel to it.
    """
    res = fisher_matrix(mean_transit, THETA, sigma=1.0, log_coords=True)
    k_e, k_a = float(THETA[0]), float(THETA[1])
    expected = np.array([-1.0 / k_e, -1.0 / k_a])
    expected /= np.linalg.norm(expected)
    top = np.asarray(res.eigenvectors[:, -1], dtype=float)
    cosine = abs(float(expected @ top))
    assert cosine > 0.999, f"informative direction off: cosine {cosine:.6f}"


def test_toy_a_null_direction_is_the_tradeoff():
    """The flat direction is `(1/k_a, -1/k_e)`: the two rates compensating each other.

    This is the analytic statement behind the swap claim -- increasing one rate and decreasing the
    other leaves the mean transit time unchanged to first order.
    """
    res = fisher_matrix(mean_transit, THETA, sigma=1.0, log_coords=True)
    k_e, k_a = float(THETA[0]), float(THETA[1])
    expected = np.array([1.0 / k_a, -1.0 / k_e])
    expected /= np.linalg.norm(expected)
    null = res.weak_direction()
    cosine = abs(float(expected @ null))
    assert cosine > 0.999, f"null direction off: cosine {cosine:.6f}"


def test_toy_a_schur_complement_removes_all_information():
    """Profiling out one of two parameters linked by a single observable leaves nothing.

    The Schur complement of a rank-1 matrix after eliminating one coordinate is zero. This is the
    test that distinguishes the Schur complement from the raw sub-block: the raw `(k_a, k_a)` entry
    is large and non-zero, which is exactly the misreading the reviewers objected to.
    """
    res = fisher_matrix(mean_transit, THETA, sigma=1.0, log_coords=True)
    raw_subblock = res.fisher[1, 1]
    profiled = schur_complement(res.fisher, keep=[1], eliminate=[0])
    assert raw_subblock > 1.0, "the toy should have a large raw diagonal entry"
    assert abs(float(profiled[0, 0])) < 1e-8 * float(raw_subblock), (
        f"Schur complement should vanish, got {profiled[0, 0]!r}")


# --- Toy B --------------------------------------------------------------------------------------

def test_toy_b_full_rank_off_the_diagonal():
    """Two independent moments identify both rates when the rates differ."""
    res = fisher_matrix(two_moments, THETA, sigma=1.0, log_coords=True)
    assert numerical_rank(np.linalg.svd(res.fisher, compute_uv=False), rtol=1e-6) == 2
    assert res.eigenvalue_ratio > 1e-6, (
        f"expected full rank away from k_e == k_a, eigenvalues {res.eigenvalues}")


def test_toy_b_degenerate_on_the_diagonal():
    """At `k_e == k_a` the swap degeneracy returns and the matrix drops to rank 1.

    Both observables are symmetric in the two rates, so at the symmetric point their gradients are
    parallel. This is the structural reason the paper expects a LOCAL but not GLOBAL identifiability
    verdict for the pair.
    """
    equal = jnp.array([0.02, 0.02])
    res = fisher_matrix(two_moments, equal, sigma=1.0, log_coords=True)
    assert res.eigenvalue_ratio < 1e-8, (
        f"expected rank deficiency at k_e == k_a, eigenvalues {res.eigenvalues}")


@pytest.mark.parametrize("separation", [1.0, 1.05, 1.2, 2.0])
def test_toy_b_rank_degrades_smoothly_toward_the_diagonal(separation):
    """Conditioning should worsen monotonically as the two rates approach each other."""
    theta = jnp.array([0.02 * separation, 0.02])
    res = fisher_matrix(two_moments, theta, sigma=1.0, log_coords=True)
    if separation == 1.0:
        assert res.eigenvalue_ratio < 1e-8
    else:
        assert res.eigenvalue_ratio > 0.0


def test_toy_b_conditioning_is_monotone_in_separation():
    ratios = []
    for sep in (1.02, 1.1, 1.5, 2.5):
        res = fisher_matrix(two_moments, jnp.array([0.02 * sep, 0.02]), sigma=1.0,
                            log_coords=True)
        ratios.append(res.eigenvalue_ratio)
    assert all(a < b for a, b in zip(ratios, ratios[1:])), (
        f"eigenvalue ratio should improve as the rates separate, got {ratios}")


# --- Toy C: profile likelihood ------------------------------------------------------------------

def test_toy_c_profile_is_flat_along_the_null_direction():
    """Toy A: profiling either rate is flat, because the other absorbs the change exactly.

    The inner optimization has to find the compensating value; if it does, the profile stays at the
    minimum across the whole box. A profile that rises here means the inner optimizer is failing,
    not that the parameter is identifiable -- which is why this assertion guards every later
    profile-likelihood result.
    """
    target = float(mean_transit(THETA)[0])

    def loss(theta):
        return 0.5 * jnp.sum((mean_transit(theta) - target) ** 2) / 1.0 ** 2

    # The box must be wide enough for the compensating value to exist at every grid point;
    # otherwise the profile rises because the bound stopped it, which the next test covers.
    grid = jnp.linspace(0.015, 0.040, 15)
    res = profile_likelihood(loss, THETA, index=0, grid=grid,
                             lower=jnp.array([0.015, 0.005]), upper=jnp.array([0.040, 0.100]),
                             steps=400, learning_rate=0.002, optimizer="lbfgs")
    rise = max(res["delta"])
    verdict = classify_profile(res["grid"], res["profile"], delta=CHI2_DELTA)
    assert rise < 1e-6, f"profile should be flat, maximum rise {rise:.3e}"
    assert verdict["verdict"] == "flat", verdict


def test_toy_c_a_bound_can_fake_identifiability():
    """A profile can rise purely because the compensating parameter hit a bound.

    Toy A with a NARROW box on `k_a`: at the low end of the `k_e` grid the value of `k_a` that
    would cancel the change lies outside the box, so the profile rises there even though the
    direction is structurally flat. The rise is an artifact of the box, not information in the data.

    This is reviewer point 6 in miniature, and it is why every later analysis reports an
    at-a-bound flag per parameter and splits subjects into an interior-converged set and the full
    set: a one-sided profile whose rising side coincides with an active bound means nothing.
    """
    target = float(mean_transit(THETA)[0])

    def loss(theta):
        return 0.5 * jnp.sum((mean_transit(theta) - target) ** 2)

    grid = jnp.linspace(0.015, 0.040, 15)
    narrow = profile_likelihood(loss, THETA, index=0, grid=grid,
                                lower=jnp.array([0.015, 0.012]), upper=jnp.array([0.040, 0.032]),
                                steps=400, learning_rate=0.002, optimizer="lbfgs")
    deltas = np.asarray(narrow["delta"])
    thetas = np.asarray(narrow["theta"])
    at_upper = np.isclose(thetas[:, 1], 0.032, rtol=0, atol=1e-9)

    assert deltas.max() > 1.0, "the narrow box should visibly distort the profile"
    assert at_upper.any(), "expected the compensating parameter to be pinned at its upper bound"
    # Every point that rose is a point where the compensator was pinned.
    assert np.all(at_upper[deltas > 1e-6]), (
        "profile rose where the compensating parameter was NOT at a bound; that would be real "
        "information and contradicts the flat structure")


def test_toy_c_profile_matches_the_analytic_quadratic():
    """Linear regression: the profile over one coefficient is exactly a known parabola.

    For `y = X beta + noise` with known `sigma`, profiling coefficient `j` gives
    `0.5 * (beta_j - beta_j_hat)^2 / v_jj` where `v_jj` is the `j`-th diagonal entry of
    `sigma^2 (X^T X)^{-1}`. Matching that curve tests the inner optimizer, the threshold scale and
    the interval interpolation at once: the 95% interval must come out at
    `beta_j_hat +/- 1.96 * sqrt(v_jj)`.
    """
    rng = np.random.default_rng(0)
    n, sigma = 60, 0.5
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta_true = np.array([1.0, 2.0, -0.5])
    y = X @ beta_true + rng.normal(scale=sigma, size=n)

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    cov = sigma ** 2 * np.linalg.inv(X.T @ X)
    j = 1
    se = math.sqrt(cov[j, j])

    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    def loss(beta):
        resid = yj - Xj @ beta
        return 0.5 * jnp.sum(resid ** 2) / sigma ** 2

    grid = jnp.asarray(beta_hat[j] + se * np.linspace(-3.0, 3.0, 15))
    res = profile_likelihood(loss, jnp.asarray(beta_hat), index=j, grid=grid,
                             steps=300, learning_rate=0.05, optimizer="lbfgs")

    analytic = 0.5 * (np.asarray(res["grid"]) - beta_hat[j]) ** 2 / cov[j, j]
    observed = np.asarray(res["delta"])
    assert np.allclose(observed, analytic, atol=1e-4, rtol=1e-3), (
        f"profile deviates from the analytic quadratic:\n{observed}\nvs\n{analytic}")

    verdict = classify_profile(res["grid"], res["profile"], delta=CHI2_DELTA)
    assert verdict["verdict"] == "identifiable", verdict
    half_width = 0.5 * verdict["width"]
    assert abs(half_width - 1.96 * se) < 0.02 * se, (
        f"95% half-width {half_width:.5f} should be about {1.96 * se:.5f}")


def test_classify_profile_detects_one_sided():
    """A profile that rises on one side only must be reported as one-sided, not as an interval."""
    grid = np.linspace(0.0, 1.0, 11)
    profile = np.where(grid < 0.5, 0.0, 10.0 * (grid - 0.5) ** 2)
    verdict = classify_profile(grid, profile, delta=CHI2_DELTA)
    assert verdict["verdict"] == "one-sided", verdict
    assert verdict["ci_low"] is None and verdict["ci_high"] is not None
    assert verdict["width"] is None


def test_fisher_matrix_rejects_mismatched_sigma():
    with pytest.raises(ValueError, match="entries for"):
        fisher_matrix(two_moments, THETA, sigma=np.array([1.0, 2.0, 3.0]))


def test_fisher_log_coordinates_differ_from_natural():
    """The two parameterizations must give genuinely different matrices (column scaling applied)."""
    nat = fisher_matrix(mean_transit, THETA, log_coords=False)
    log = fisher_matrix(mean_transit, THETA, log_coords=True)
    scale = np.asarray(THETA, dtype=float)
    assert np.allclose(log.jacobian, nat.jacobian * scale[None, :], rtol=1e-12)


def test_x64_is_actually_enabled():
    """Every assertion above about 1e-8 ratios depends on this."""
    assert jnp.zeros(1).dtype == jnp.float64, "float64 is not enabled; see evaluation/jax_config"
    assert jax.config.jax_enable_x64 is True

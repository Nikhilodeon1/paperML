"""Phase 1: the noise model, the three objectives, the reparameterizations, and the two new engines.

The expensive assertions are the ones that would silently corrupt every later phase if they were wrong:
that the canonical engine reproduces the cascade it replaces, that the coordinate maps round-trip, that
the linearization is a correct first-order approximation (error quadratic in the perturbation, not
merely small), and that the generalized fit still reproduces the submitted results when pointed at the
submitted objective definition.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.jax_config import configure

configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402

from evaluation.cohort_data import load_cgmacros                       # noqa: E402
from personalization import coords as co                               # noqa: E402
from personalization import noise_model as nm                          # noqa: E402
from personalization.objectives import (                               # noqa: E402
    LADDER, ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402
from simulation.jax_observables import Window                          # noqa: E402


@pytest.fixture(scope="module")
def subject():
    try:
        subjects = load_cgmacros()
    except FileNotFoundError as exc:
        pytest.skip(f"CGMacros not available: {exc}")
    if not subjects:
        pytest.skip("CGMacros loaded but empty")
    return subjects[0]


# --- the noise model (amendment B2, B3) -----------------------------------------------------------

def test_sigma_uses_n_minus_p():
    residuals = np.array([10.0, -10.0, 10.0, -10.0, 10.0])
    sigma, dof = nm.estimate_sigma(residuals, n_parameters=3, observable="iauc")
    assert dof == 2
    assert sigma == pytest.approx(math.sqrt(500.0 / 2.0))


def test_sigma_is_floored():
    """A near-perfect pilot fit must not produce an infinite weight on that observable."""
    sigma, _ = nm.estimate_sigma(np.zeros(40), n_parameters=3, observable="iauc")
    assert sigma == nm.SIGMA_FLOOR["iauc"]


def test_ar1_recovers_a_known_coefficient():
    """Simulate an AR(1) series with a known rho and check the estimate comes back."""
    rng = np.random.default_rng(0)
    true_rho, n_meals, n_samples = 0.8, 60, 37
    rows = np.zeros((n_meals, n_samples))
    for i in range(n_meals):
        value = rng.normal()
        for t in range(n_samples):
            value = true_rho * value + rng.normal(scale=1.0)
            rows[i, t] = value
    rho, innovation = nm.estimate_ar1(rows)
    assert rho == pytest.approx(true_rho, abs=0.05), rho
    assert innovation == pytest.approx(1.0, rel=0.15), innovation


def test_ar1_is_pooled_within_meals_not_across_them():
    """Two meals whose ends and starts differ wildly must not be paired across the boundary.

    Within each meal the series is perfectly persistent, so the raw estimate is 1 and the reported value
    is the cap. Pairing the last sample of the first meal with the first of the second would introduce a
    strongly negative product and drag the estimate well below the cap.
    """
    rows = np.array([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]])
    rho, _ = nm.estimate_ar1(rows)
    assert rho == pytest.approx(nm.RHO_MAX, abs=1e-12)

    # What pooling across the boundary would look like: 7 lag-1 pairs, six of them +1 and the
    # boundary pair -1, giving exactly 5/7 instead of 1.
    flattened = rows.reshape(1, -1)
    assert nm.estimate_ar1(flattened)[0] == pytest.approx(5.0 / 7.0, rel=1e-12)


def test_ar1_clips_negative_rho():
    """A genuinely anti-persistent series is reported as zero rather than whitened with a negative rho.

    Whitening with rho < 0 would amplify high-frequency noise instead of removing low-frequency
    structure, and on a CGM trace a negative lag-1 coefficient means noise is dominating rather than
    that glucose truly alternates.
    """
    rng = np.random.default_rng(1)
    rows = np.zeros((40, 37))
    for i in range(40):
        value = rng.normal()
        for t in range(37):
            value = -0.7 * value + rng.normal()
            rows[i, t] = value
    assert nm.estimate_ar1(rows)[0] == 0.0


def test_whitening_removes_lag_one_structure():
    rng = np.random.default_rng(2)
    rows = np.cumsum(rng.normal(size=(40, 37)), axis=1)      # a random walk: rho near 1
    rho, _ = nm.estimate_ar1(rows)
    assert rho > 0.9
    whitened = nm.whiten_rows(rows, rho)
    after, _ = nm.estimate_ar1(whitened)
    assert abs(after) < 0.3, f"whitening left rho at {after}"


def test_whitening_preserves_shape():
    rows = np.arange(12.0).reshape(3, 4)
    assert nm.whiten_rows(rows, 0.5).shape == rows.shape


def test_unit_noise_is_unweighted():
    model = nm.unit_noise()
    assert model.for_observable("iauc") == 1.0
    assert model.rho == 0.0
    assert model.whitened is False


# --- the objectives -------------------------------------------------------------------------------

def test_nll_objective_refuses_regularization():
    """The chi-square threshold is a likelihood statement; a penalty would narrow every interval."""
    with pytest.raises(ValueError, match="unregularized"):
        ObjectiveSpec(normalization="nll", lam=0.01)


def test_unknown_normalization_and_parameterization_are_refused():
    with pytest.raises(ValueError, match="unknown normalization"):
        ObjectiveSpec(normalization="sse")
    with pytest.raises(ValueError, match="unknown parameterization"):
        ObjectiveSpec(parameterization="canonical")


def test_legacy_window_only_in_rate_space():
    with pytest.raises(ValueError, match="submitted rate-space fit"):
        ObjectiveSpec(parameterization="coords", legacy_full_window=True)


def test_theta_names_follow_the_parameterization():
    assert ObjectiveSpec().theta_names == TARGETS
    assert ObjectiveSpec(parameterization="coords").theta_names == co.COORD_NAMES
    assert ObjectiveSpec(parameterization="tied").theta_names == co.TIED_COORD_NAMES


@pytest.mark.slow
def test_observed_iauc_matches_the_loader(subject):
    """The bottom rung must be the quantity the submitted results were fitted to."""
    observed = observed_values(subject.records, Window())
    stored = np.array([r["iauc"] for r in subject.records])
    assert np.allclose(observed["iauc"], stored, atol=1e-8)


@pytest.mark.slow
def test_meals_with_no_excursion_are_excluded_from_the_centroid(subject):
    observed = observed_values(subject.records, Window())
    undefined = observed["iauc"] <= 0.0
    assert np.all(observed["centroid_defined"][undefined] == 0.0)
    assert np.all(observed["centroid_defined"][~undefined] == 1.0)


@pytest.mark.slow
@pytest.mark.parametrize("name", LADDER)
def test_every_rung_has_a_finite_loss_and_gradient(subject, name):
    records = list(subject.records)[:12]
    base = base_params(subject.profile)
    arrays = subject_arrays(records)
    observed = observed_values(records, Window())
    objective = build_objective(ObjectiveSpec(name=name), base, arrays, observed)
    value = float(objective.loss(objective.theta0))
    grad = np.asarray(jax.grad(objective.loss)(objective.theta0), dtype=float)
    assert np.isfinite(value)
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0.0)


@pytest.mark.slow
def test_nll_at_the_estimate_is_about_half_the_residual_count(subject):
    """A calibration check: with sigma estimated at the same point, the NLL should be about n/2.

    If it comes out far from that, either the weights are not being applied or the residual count is
    wrong, and every profile-likelihood interval built on it would be mis-scaled.
    """
    from personalization.fit_general import fit_ml

    for name in LADDER:
        result = fit_ml(subject, ObjectiveSpec(name=name), steps=200, pilot_steps=100)
        expected = result.ml.n_residuals / 2.0
        assert result.ml.final_loss == pytest.approx(expected, rel=0.35), (
            f"{name}: NLL {result.ml.final_loss:.1f} against n/2 = {expected:.1f}")


@pytest.mark.slow
def test_trace_whitening_is_actually_applied(subject):
    """The AR(1) coefficient on a real CGM trace is large, and the whitened loss must differ."""
    from personalization.fit_general import fit_ml

    result = fit_ml(subject, ObjectiveSpec(name="trace"), steps=150, pilot_steps=80)
    assert result.noise.whitened is True
    assert result.noise.rho > 0.5, f"expected strong autocorrelation, got rho = {result.noise.rho}"

    records = list(subject.records)
    base = base_params(subject.profile)
    arrays = subject_arrays(records)
    observed = observed_values(records, Window())
    spec = ObjectiveSpec(name="trace", lam=0.0, normalization="nll")
    whitened = build_objective(spec, base, arrays, observed, noise=result.noise)
    plain = build_objective(ObjectiveSpec(name="trace", lam=0.0, normalization="nll",
                                          whiten_trace=False), base, arrays, observed,
                            noise=result.noise)
    theta = whitened.theta0
    assert float(whitened.loss(theta)) != pytest.approx(float(plain.loss(theta)), rel=1e-6)


# --- coordinates (amendment D2, D3) ---------------------------------------------------------------

@pytest.mark.parametrize("k_e,k_a", [(0.015, 0.012), (0.040, 0.032), (0.026, 0.022), (0.02, 0.02)])
def test_coordinate_round_trip(k_e, k_a):
    tau1, p = co.rates_to_tau_p(k_e, k_a)
    back_e, back_a = co.tau_p_to_rates(tau1, p)
    assert float(back_e) == pytest.approx(max(k_e, k_a), rel=1e-9)
    assert float(back_a) == pytest.approx(min(k_e, k_a), rel=1e-9)


def test_real_pole_condition_holds_on_the_whole_rate_box():
    for k_e in np.linspace(0.015, 0.040, 7):
        for k_a in np.linspace(0.012, 0.032, 7):
            tau1, p = co.rates_to_tau_p(k_e, k_a)
            assert p <= tau1 ** 2 / 4.0 + 1e-12


def test_equal_rates_sit_exactly_on_the_boundary():
    tau1, p = co.rates_to_tau_p(0.02, 0.02)
    assert p == pytest.approx(tau1 ** 2 / 4.0, rel=1e-12)
    assert co.on_tied_boundary(np.log([1.0, tau1, p]))


def test_projection_enforces_real_poles():
    lower, upper = co.log_bounds()
    violating = jnp.asarray([0.0, math.log(60.0), math.log(5000.0)])
    projected = np.asarray(co.project_log(violating, lower, upper), dtype=float)
    tau1, p = math.exp(projected[1]), math.exp(projected[2])
    assert p <= tau1 ** 2 / 4.0 + 1e-9
    assert co.on_tied_boundary(projected)


def test_projection_is_differentiable_everywhere():
    """The whole point of the reparameterization: no square root, so no NaN at equal rates."""
    lower, upper = co.log_bounds()

    def total(log_tau):
        theta = jnp.stack([jnp.float64(0.0), log_tau, jnp.float64(math.log(5000.0))])
        return jnp.sum(co.project_log(theta, lower, upper))

    assert np.isfinite(float(jax.grad(total)(jnp.float64(math.log(60.0)))))


def test_the_root_map_is_the_thing_that_fails():
    """Evidence for why the canonical form exists: differentiating the roots gives NaN at k_e = k_a."""
    from simulation.jax_engine_canonical import canonical_to_rates

    def first_rate(sigma1):
        return canonical_to_rates(sigma1, (sigma1 / 2.0) ** 2)[0]

    assert not np.isfinite(float(jax.grad(first_rate)(jnp.float64(0.05))))


def test_tied_map_is_consistent_and_analytic():
    for tau1 in (62.5, 100.0, 133.0):
        sigma1, c = co.tied_to_canonical(tau1)
        assert float(sigma1 ** 2 - 4.0 * c) == pytest.approx(0.0, abs=1e-18)
        k = 2.0 / tau1
        assert float(sigma1) == pytest.approx(2.0 * k, rel=1e-12)
        assert float(c) == pytest.approx(k * k, rel=1e-12)
    assert np.isfinite(float(jax.grad(lambda t: sum(co.tied_to_canonical(t)))(jnp.float64(100.0))))


def test_bounds_contain_the_image_of_the_rate_box():
    lower, upper = co.log_bounds()
    for k_e in (0.015, 0.0275, 0.040):
        for k_a in (0.012, 0.022, 0.032):
            tau1, p = co.rates_to_tau_p(k_e, k_a)
            point = np.log([1.0, tau1, p])
            assert np.all(point >= lower - 1e-9) and np.all(point <= upper + 1e-9)


def test_tied_range_is_the_rate_intersection():
    lo, hi = co.tied_tau_range()
    assert lo == pytest.approx(2.0 / 0.032)
    assert hi == pytest.approx(2.0 / 0.015)


def test_rates_outside_the_original_box_are_reported():
    """The fitted box is the hull of the image, so some of it is outside the original box."""
    inside = co.rates_in_original_box(*co.rates_to_tau_p(0.026, 0.022))
    assert inside["inside_original_box"] is True
    outside = co.rates_in_original_box(60.0, 900.0)       # the tied point k = 1/30
    assert outside["real_poles"] is True
    assert outside["inside_original_box"] is False


# --- the canonical engine (amendment D1) ----------------------------------------------------------

@pytest.mark.slow
def test_canonical_engine_reproduces_the_cascade(subject):
    """The two forms are the same linear system in different bases, so they must agree to solver
    tolerance. Any larger difference means the realization is wrong, not approximate."""
    from simulation.jax_engine import run_meal as run_cascade
    from simulation.jax_engine_canonical import CanonicalParams
    from simulation.jax_engine_canonical import run_meal as run_canonical

    params = base_params(subject.profile)
    canonical = CanonicalParams.from_jax_params(params)
    for carbs in (20.0, 60.0):
        _, cascade = run_cascade(params, carbs, meal_time=30.0, duration_min=210.0, step_min=5.0)
        _, canon = run_canonical(canonical, carbs, meal_time=30.0, duration_min=210.0,
                                 step_min=5.0)
        relative = float(np.max(np.abs(np.asarray(cascade) - np.asarray(canon)))
                         / np.max(np.abs(np.asarray(cascade))))
        assert relative < 1e-4, f"{carbs} g: relative difference {relative:.2e}"


@pytest.mark.slow
def test_canonical_engine_is_differentiable_at_equal_rates(subject):
    """At k_e = k_a the cascade parameterization has a singular root map; this one does not."""
    from simulation.jax_engine_canonical import CanonicalParams
    from simulation.jax_engine_canonical import run_meal as run_canonical
    import equinox as eqx

    params = base_params(subject.profile)
    canonical = CanonicalParams.from_jax_params(params)

    def area(log_tau):
        sigma1, c = co.tied_to_canonical(jnp.exp(log_tau))
        p = eqx.tree_at(lambda m: (m.sigma1, m.c), canonical, (sigma1, c))
        _, glucose = run_canonical(p, 60.0, meal_time=30.0, duration_min=210.0, step_min=5.0)
        return jnp.sum(glucose)

    gradient = float(jax.grad(area)(jnp.float64(math.log(100.0))))
    assert np.isfinite(gradient) and gradient != 0.0


# --- the linearized engine (Phase 1.4) -----------------------------------------------------------

@pytest.mark.slow
def test_linearization_error_is_second_order(subject):
    """The decisive check that the Jacobian is right: the error must scale as the meal SQUARED.

    A merely small error could hide a wrong Jacobian. An error proportional to the square of the
    perturbation is the signature of a correct first-order expansion.
    """
    from simulation.jax_engine_linear import linearize, run_meal_fast, run_meal_linear
    from simulation.jax_observables import iauc

    params = base_params(subject.profile)
    model = linearize(params, regime="active")
    window = Window()

    ratios = []
    for carbs in (0.05, 0.5, 5.0):
        _, nonlinear = run_meal_fast(params, carbs)
        _, linear = run_meal_linear(params, carbs, model=model)
        a_nl = float(iauc(None, nonlinear, window, beta=None))
        a_li = float(iauc(None, linear, window, beta=None))
        ratios.append(abs(a_nl - a_li) / carbs ** 2)

    assert ratios[0] == pytest.approx(ratios[1], rel=0.25), ratios
    assert ratios[1] == pytest.approx(ratios[2], rel=0.25), ratios


@pytest.mark.slow
def test_linear_and_nonlinear_agree_within_one_percent_at_one_gram(subject):
    from simulation.jax_engine_linear import linearize, run_meal_fast, run_meal_linear
    from simulation.jax_observables import iauc

    params = base_params(subject.profile)
    model = linearize(params, regime="active")
    _, nonlinear = run_meal_fast(params, 1.0)
    _, linear = run_meal_linear(params, 1.0, model=model)
    a_nl = float(iauc(None, nonlinear, Window(), beta=None))
    a_li = float(iauc(None, linear, Window(), beta=None))
    assert abs(a_nl - a_li) / a_nl < 0.01


@pytest.mark.slow
def test_the_basal_regime_is_the_wrong_operating_point(subject):
    """Documents the finding: linearizing at the fasting point halves the secretion gain.

    Kept as a test so that if someone later switches the default regime, this fails and says why.
    """
    from simulation.jax_engine_linear import linearize, run_meal_fast, run_meal_linear
    from simulation.jax_observables import iauc

    params = base_params(subject.profile)
    _, nonlinear = run_meal_fast(params, 1.0, regime="active")
    reference = float(iauc(None, nonlinear, Window(), beta=None))

    active = float(iauc(None, run_meal_linear(
        params, 1.0, model=linearize(params, regime="active"))[1], Window(), beta=None))
    basal = float(iauc(None, run_meal_linear(
        params, 1.0, model=linearize(params, regime="basal"))[1], Window(), beta=None))

    assert abs(active - reference) / reference < 0.01
    assert abs(basal - reference) / reference > 0.3


@pytest.mark.slow
def test_linear_model_is_stable_and_shows_the_gut_rates_as_eigenvalues(subject):
    from simulation.jax_engine_linear import linearize

    params = base_params(subject.profile)
    model = linearize(params, regime="active")
    real = np.sort(np.real(model.eigenvalues))
    assert np.all(real < 0.0), real
    assert model.residual < 1e-10, model.residual
    for rate in (float(params.gastric_emptying), float(params.carb_absorption)):
        assert np.min(np.abs(real + rate)) < 1e-6, (
            f"the gut rate {rate} should appear as an eigenvalue of the linearization; got {real}")


# --- the generalized fit (Phase 1.3) -------------------------------------------------------------

@pytest.mark.slow
def test_fit_general_reproduces_the_submitted_fit():
    """The regression the whole refactor rests on, against the untouched original module."""
    from personalization.fit_general import fit_subject
    from personalization.gradient_fit import fit_parameters

    worst = 0.0
    for subject_ in load_cgmacros()[:5]:
        meals = [{"carbs_g": r["carbs_g"], "fat_g": r["fat_g"], "fiber_g": r["fiber_g"],
                  "observed_iAUC": r["iauc"]} for r in subject_.records]
        old = fit_parameters(meals, n_steps=150, base=base_params(subject_.profile))["theta"]
        new = fit_subject(subject_, ObjectiveSpec(legacy_full_window=True), steps=150).theta
        worst = max(worst, max(abs(old[k] - new[k]) for k in TARGETS))
    assert worst < 1e-6, f"worst parameter difference {worst:.3e}"


@pytest.mark.slow
def test_ml_fit_is_unregularized_and_warm_started(subject):
    from personalization.fit_general import fit_ml

    result = fit_ml(subject, ObjectiveSpec(name="iauc"), steps=200, pilot_steps=100)
    assert result.spec_pilot["lam"] == pytest.approx(0.01)
    assert result.spec_pilot["normalization"] == "mean"
    assert result.spec_ml["lam"] == 0.0
    assert result.spec_ml["normalization"] == "nll"
    assert result.ml.init == "warm"
    assert result.ml.extras["theta_pilot"] == result.pilot.theta


@pytest.mark.slow
def test_bound_flags_and_projected_gradient_are_reported(subject):
    from personalization.fit_general import fit_subject

    result = fit_subject(subject, ObjectiveSpec(), steps=200)
    assert set(result.at_bound) == set(TARGETS)
    assert result.n_at_bound == sum(1 for v in result.at_bound.values() if v is not None)
    assert result.projected_grad_norm >= 0.0
    assert result.interior == (result.converged and result.n_at_bound == 0)


@pytest.mark.slow
@pytest.mark.parametrize("parameterization", ["rates", "coords", "tied"])
def test_each_parameterization_fits_and_reports_natural_units(subject, parameterization):
    from personalization.fit_general import fit_ml

    result = fit_ml(subject, ObjectiveSpec(name="iauc", parameterization=parameterization),
                    steps=150, pilot_steps=80)
    natural = result.ml.theta_full
    assert 0.0 < natural["insulin_sensitivity"] < 3.0
    assert np.isfinite(result.ml.final_loss)
    if parameterization != "rates":
        assert natural["tau1"] > 0.0
        assert "inside_original_box" in natural


@pytest.mark.slow
def test_random_start_needs_a_seed(subject):
    from personalization.fit_general import fit, fit_ml  # noqa: F401
    from personalization.objectives import build_objective

    records = list(subject.records)[:10]
    objective = build_objective(ObjectiveSpec(), base_params(subject.profile),
                               subject_arrays(records), observed_values(records, Window()))
    with pytest.raises(ValueError, match="needs a seed"):
        fit(objective, steps=5, init="random", seed=None)


@pytest.mark.slow
def test_random_starts_are_reproducible_and_different(subject):
    from personalization.fit_general import fit
    from personalization.objectives import build_objective

    records = list(subject.records)[:10]
    objective = build_objective(ObjectiveSpec(), base_params(subject.profile),
                               subject_arrays(records), observed_values(records, Window()))
    a = fit(objective, steps=20, init="random", seed=7).theta
    b = fit(objective, steps=20, init="random", seed=7).theta
    c = fit(objective, steps=20, init="random", seed=8).theta
    assert a == b
    assert a != c

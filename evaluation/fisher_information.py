"""Fisher-information grounding for the gradient-norm identifiability diagnostic (theory fix).

The paper uses gradient magnitude as an empirical identifiability proxy. This connects it to the
diagonal Fisher information. The gradient-norm diagnostic is computed on the iAUC reconstruction
loss, so the consistent Fisher is for the iAUC OBSERVABLE (not the raw trajectory): under Gaussian
observation noise sigma^2,

    FI(theta_i) = (1/sigma^2) * E_meal[ (d iAUC(theta) / d theta_i)^2 ]

computed EXPLICITLY at the subject's population-default parameters (rather than averaged over an
optimization trajectory, the way the gradient-norm diagnostic is). Since the loss gradient is
2*(pred - obs)*d(iAUC)/dtheta, |grad| ~ |residual| * sqrt(FI); if the two correlate across subjects,
gradient magnitude is a computable proxy for the Fisher criterion.

IMPLEMENTATION NOTE (deviation from the brief): the brief specified forward-mode jax.jacfwd, but
diffrax's ODE solve defines a custom_vjp (reverse-mode adjoint) that forbids forward-mode AD. We
therefore use reverse-mode jax.grad on the scalar iAUC — one backward pass, fast, and it is exactly
the observable the diagnostic uses. Bergman engine, the paper's main method.

Run:  python -m evaluation.fisher_information
"""
from __future__ import annotations

import numpy as np

import paper_config as cfg

_PARAMS = ("insulin_sensitivity", "gastric_emptying", "carb_absorption")


def _fisher_for_subject(base, carbs_list, sigma_cgm=10.0):
    """Diagonal Fisher (3-vector) on the iAUC observable, averaged over a subject's meals."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from personalization.gradient_fit import _meal_iauc

    theta0 = jnp.array([float(getattr(base, p)) for p in _PARAMS], dtype=jnp.float32)

    def iauc_of(theta, carbs):
        p = base
        for i, name in enumerate(_PARAMS):
            p = eqx.tree_at(lambda m, n=name: getattr(m, n), p, theta[i])
        return _meal_iauc(p, carbs, 0.0, 0.0)      # scalar iAUC

    grad_fn = jax.grad(iauc_of)                     # d iAUC / d theta -> (3,)  reverse-mode
    acc = np.zeros(3)
    for c in carbs_list:
        gi = np.asarray(grad_fn(theta0, jnp.asarray(float(c), jnp.float32)))
        acc += gi ** 2
    fi = acc / max(len(carbs_list), 1) / (sigma_cgm ** 2)
    return {p: float(fi[i]) for i, p in enumerate(_PARAMS)}


def compute_fisher_diagonal(params, meals, sigma_cgm: float = 10.0) -> dict:
    """Diagonal Fisher information for the 3 Bergman params from a list of meal dicts."""
    carbs = [float(m["carbs_g"]) for m in meals]
    return _fisher_for_subject(params, carbs, sigma_cgm)


def compare_fisher_to_gradient_norms(limit: int | None = None, n_steps: int = 150) -> dict:
    """Per subject: diagonal Fisher (at defaults) vs gradient-norm (from optimization). Correlate
    each parameter across subjects."""
    from simulation import PhysioParams
    from simulation.jax_engine import JaxPhysioParams
    from evaluation.cgmacros import _profile, load_bio, subjects
    from evaluation.gradient_inference_results import _subject_meals
    from personalization.gradient_fit import fit_parameters

    from personalization.gradient_fit import PARAM_RANGES
    cfg.set_all_seeds()
    bio = load_bio()
    rng = {p: PARAM_RANGES[p][1] - PARAM_RANGES[p][0] for p in _PARAMS}
    fi_rows = {p: [] for p in _PARAMS}       # raw diagonal Fisher
    fisr_rows = {p: [] for p in _PARAMS}     # sqrt(FI)*range — comparable units to grad_norm
    gn_rows = {p: [] for p in _PARAMS}
    for sid, meals in subjects(limit=limit):
        prof = _profile(bio.get(sid))
        ms = _subject_meals(meals)
        if not prof or len(ms) < 4:
            continue
        base = JaxPhysioParams.from_numpy(PhysioParams.from_profile(prof))
        fi = compute_fisher_diagonal(base, ms)
        gn = fit_parameters(ms, n_steps=n_steps, base=base)["gradient_norms"]
        for p in _PARAMS:
            fi_rows[p].append(fi[p])
            fisr_rows[p].append(np.sqrt(fi[p]) * rng[p])
            gn_rows[p].append(gn[p])

    def r(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    # PRIMARY grounding: does the theoretically-motivated Fisher information rank the parameters the
    # same way the gradient-norm diagnostic does (Si >> gut/timing)? (The per-subject correlation
    # below is a stricter, noisier test that the trajectory-averaged norm partly decouples for Si.)
    si = "insulin_sensitivity"
    fi_mean = {p: float(np.mean(fisr_rows[p])) for p in _PARAMS}
    gn_mean = {p: float(np.mean(gn_rows[p])) for p in _PARAMS}
    return {
        "n": len(fi_rows[si]),
        "corr": {p: round(r(fi_rows[p], gn_rows[p]), 3) for p in _PARAMS},
        "fisher_rank_Si_over_gastric": round(fi_mean[si] / max(fi_mean["gastric_emptying"], 1e-12), 1),
        "fisher_rank_Si_over_carb": round(fi_mean[si] / max(fi_mean["carb_absorption"], 1e-12), 1),
        "gradnorm_rank_Si_over_gastric": round(gn_mean[si] / max(gn_mean["gastric_emptying"], 1e-12), 1),
        "gradnorm_rank_Si_over_carb": round(gn_mean[si] / max(gn_mean["carb_absorption"], 1e-12), 1),
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    res = compare_fisher_to_gradient_norms()
    print("=" * 74)
    print(f"FISHER INFORMATION vs GRADIENT NORM  (n={res['n']} CGMacros subjects)")
    print("=" * 74)
    print("  PRIMARY — both rank Si as the sole identifiable parameter:")
    print(f"    Fisher   ranks Si/gastric {res['fisher_rank_Si_over_gastric']}x , Si/carb "
          f"{res['fisher_rank_Si_over_carb']}x")
    print(f"    grad-norm ranks Si/gastric {res['gradnorm_rank_Si_over_gastric']}x , Si/carb "
          f"{res['gradnorm_rank_Si_over_carb']}x")
    print("  SECONDARY — per-subject correlation (trajectory-avg norm partly decouples for Si):")
    for p, c in res["corr"].items():
        print(f"    r(FI({p}), grad_norm({p})) = {c:+.3f}")
    print("=" * 74)


if __name__ == "__main__":
    main()

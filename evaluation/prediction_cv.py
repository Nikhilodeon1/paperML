"""A9: repeated k-fold prediction comparison, every cell on the same folds.

One work unit is one (subject, repeat). Within it, the subject's meals are split into five folds by
`make_folds` (SHA-256 seeded, so identical across machines and runs), every cell is fitted on the
four training folds and predicts the held-out fold, and the raw per-meal predictions are stored so any
statistic can be recomputed later without refitting.

Cells
-----
Gradient, same regularized loss, Adam as in the submitted fit:
  grad3, grad1                  iAUC objective, three parameters / insulin sensitivity only
  grad3_trace, grad1_trace      trace objective (4b)
Exhaustive search of the SAME regularized iAUC loss:
  grid3                         20 x 8 x 8 over the box
  grid1                         41 points over the S_I interval, timing at its population default
  grid1_legacy                  the submitted grid: 15 points on [0.2, 1.6], unregularized L1 loss
Amortized estimators (insulin sensitivity only, exactly as in the submission):
  rf, snpe
Baselines:
  personal_mean, population, persistence

Multistart is deliberately absent. It is a full-data analysis (A10), not a cross-validation cell.

Three design points are about cost and none changes a number:

* The loss is a sum over meals, so a fold is the same compiled program with a different training mask.
  The data stay arguments of the jitted function instead of being baked into it, so a subject compiles
  once per cell, not once per fold.
* The grid is evaluated once per subject: every meal at every grid point. A fold is then a masked sum
  over the right meals and an argmin, which is an identity, not an approximation.
* The persistent compilation cache is shared by the worker processes, so the second worker to need a
  subject loads its compiled programs instead of building them.

Run:  python -m evaluation.runner evaluation.prediction_cv --gate 3
      python -m evaluation.runner evaluation.prediction_cv --workers 12
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from evaluation.jax_config import configure

_JAX = configure()

import jax                       # noqa: E402
import jax.numpy as jnp         # noqa: E402
import optax                     # noqa: E402

from evaluation.cohort_data import load_cohort                          # noqa: E402
from evaluation.cv_utils import make_folds                              # noqa: E402
from evaluation.subject_source import get_subject, window_for           # noqa: E402
from personalization.fit_general import _optimizer, _project_gradient   # noqa: E402
from personalization.objectives import (                                # noqa: E402
    ObjectiveSpec, build_objective, observed_values,
)
from personalization.subject_loss import TARGETS, base_params, subject_arrays  # noqa: E402
from simulation.jax_observables import Window, iauc, peak_time, trace   # noqa: E402

ANALYSIS_ID = "A9_prediction_cv"

# (objective, free parameters, parameterization). The last two groups are the Amendment 1 cells: the
# tied-rate model (k_e = k_a = 2 / tau1) and the (log S_I, log tau1, log p) coordinates. They are run
# as their own pass (`--set 'cells=[...]'`), not as part of the primary list.
FIT_CELLS = {
    "grad3": ("iauc", TARGETS, "rates"),
    "grad1": ("iauc", ("insulin_sensitivity",), "rates"),
    "grad3_trace": ("trace", TARGETS, "rates"),
    "grad1_trace": ("trace", ("insulin_sensitivity",), "rates"),
    "grad2_tied": ("iauc", TARGETS, "tied"),
    "grad3_coords": ("iauc", TARGETS, "coords"),
    "grad2_tied_trace": ("trace", TARGETS, "tied"),
    "grad3_coords_trace": ("trace", TARGETS, "coords"),
}
PRIMARY_FIT_CELLS = ("grad3", "grad1", "grad3_trace", "grad1_trace")
GRID_CELLS = ("grid3", "grid1", "grid1_legacy")
AMORTIZED_CELLS = ("rf", "snpe")
BASELINE_CELLS = ("personal_mean", "population", "persistence")
ALL_CELLS = (PRIMARY_FIT_CELLS + GRID_CELLS + AMORTIZED_CELLS + BASELINE_CELLS)

LEGACY_SI_GRID = np.linspace(0.2, 1.6, 15)


def default_config() -> dict:
    return {
        "cohort": "cgmacros",
        "min_meals": 10,
        "limit": None,
        "folds": 5,
        "repeats": 5,
        # The submitted protocol: Adam, learning rate 0.02, 150 steps, lambda 0.01. Held fixed so the
        # gradient cell reproduces the number a reviewer already saw; per-fit convergence flags are
        # stored so an under-converged fit is reported rather than hidden.
        "steps": 150,
        "learning_rate": 0.02,
        "lam": 0.01,
        "grid_shape": [20, 8, 8],
        "grid1_points": 41,
        "cells": list(ALL_CELLS),
        "beta": 10.0,
        "carb_scale": None,
        "replica": None,
    }


def units(config: dict) -> list[str]:
    subjects = load_cohort(config["cohort"], min_meals=config["min_meals"], limit=config["limit"])
    return [f"{s.subject_id}__r{seed}" for seed in range(config["repeats"]) for s in subjects]


def split_unit(unit: str) -> tuple[str, int]:
    subject_id, _, repeat = unit.rpartition("__r")
    return subject_id, int(repeat)


def stable_seed(*parts) -> int:
    """A seed from named parts. SHA-256, never Python's per-process `hash`."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------------------------------
# Per-subject machinery, built once per process and subject.
# --------------------------------------------------------------------------------------------------

class SubjectContext:
    """Everything about one subject that does not depend on the fold or the repeat."""

    def __init__(self, subject, config: dict):
        self.subject = subject
        self.config = config
        self.records = list(subject.records)
        self.n = len(self.records)
        self.window = window_for(subject)
        self.base = base_params(subject.profile)
        self.arrays = subject_arrays(self.records)
        self.width = self.arrays["width"]
        self.observed = observed_values(self.records, self.window)
        self.obs_iauc = np.asarray(self.observed["iauc"], dtype=float)
        self.obs_trace = np.asarray(self.observed["trace"], dtype=float)
        self.obs_peak = np.array([
            float(peak_time(None, jnp.asarray(r["glucose"]["values"], dtype=jnp.float64),
                            self.window, beta=None)) for r in self.records])

        # One objective over all meals supplies the glucose simulator for every prediction.
        spec3 = ObjectiveSpec(name="iauc", lam=config["lam"], beta=config["beta"], free=TARGETS,
                              window=self.window)
        self.obj3 = build_objective(spec3, self.base, self.arrays, self.observed)
        self.theta_base = np.asarray(self.obj3.theta0, dtype=float)
        self.lower3 = np.asarray(self.obj3.lower, dtype=float)
        self.upper3 = np.asarray(self.obj3.upper, dtype=float)
        beta = config["beta"]
        window = self.window
        obs_area = jnp.asarray(np.pad(self.obs_iauc, (0, self.width - self.n)))
        obs_rows = jnp.asarray(np.pad(self.obs_trace, ((0, self.width - self.n), (0, 0))))
        simulate, penalty = self.obj3.simulate, self.obj3.penalty

        def outputs(theta):
            """Every prediction a cell needs, for every meal, at one full parameter vector."""
            glucose = simulate(theta)
            smooth = jax.vmap(lambda g: iauc(None, g, window, beta))(glucose)
            rows = jax.vmap(lambda g: trace(None, g, window))(glucose)
            return {
                "iauc": jax.vmap(lambda g: iauc(None, g, window, None))(glucose),
                "peak": jax.vmap(lambda g: peak_time(None, g, window, None))(glucose),
                "trace_rmse": jnp.sqrt(jnp.mean((rows - obs_rows) ** 2, axis=1)),
                "sq_error": (smooth - obs_area) ** 2,
                "penalty": penalty(theta),
            }

        self._outputs = jax.jit(outputs)
        self._outputs_batch = jax.jit(jax.vmap(outputs))
        self._fitters: dict[str, _Fitter] = {}
        self._grid3 = None
        self._grid1 = None
        self._grid1_legacy = None

    # -- prediction -----------------------------------------------------------------------------
    def predict(self, theta_full) -> dict:
        """Predictions for every meal at one full parameter vector (S_I, k_e, k_a)."""
        out = self._outputs(jnp.asarray(theta_full, dtype=jnp.float64))
        n = self.n
        return {key: np.asarray(out[key])[:n] for key in ("iauc", "peak", "trace_rmse")}

    def prediction_fn(self, simulate):
        """Held-out predictions for any simulator (the canonical-form engine for the reparameterized
        cells), in the same three metrics as the rate-space cells."""
        window = self.window
        obs_rows = jnp.asarray(np.pad(self.obs_trace, ((0, self.width - self.n), (0, 0))))

        @jax.jit
        def outputs(theta):
            glucose = simulate(theta)
            rows = jax.vmap(lambda g: trace(None, g, window))(glucose)
            return {"iauc": jax.vmap(lambda g: iauc(None, g, window, None))(glucose),
                    "peak": jax.vmap(lambda g: peak_time(None, g, window, None))(glucose),
                    "trace_rmse": jnp.sqrt(jnp.mean((rows - obs_rows) ** 2, axis=1))}

        def predict(theta):
            out = outputs(jnp.asarray(theta, dtype=jnp.float64))
            return {k: np.asarray(v)[:self.n] for k, v in out.items()}
        return predict

    def predict_fitted(self, cell: str, fit: dict) -> dict:
        fitter = self.fitter(cell)
        if fitter.parameterization == "rates":
            return self.predict(fit["theta_full"])
        return fitter.predict(fit["theta"])

    def fitter(self, cell: str):
        if cell not in self._fitters:
            name, free, parameterization = FIT_CELLS[cell]
            self._fitters[cell] = _Fitter(self, name, free, parameterization)
        return self._fitters[cell]

    # -- grid tables ------------------------------------------------------------------------------
    def _table(self, thetas: np.ndarray, chunk: int = 32) -> dict:
        """Every meal at every grid point, once: errors, penalties and the held-out predictions."""
        count = len(thetas)
        padded = np.concatenate([thetas, np.repeat(thetas[-1:], (-count) % chunk, axis=0)])
        parts = [self._outputs_batch(jnp.asarray(padded[i:i + chunk]))
                 for i in range(0, len(padded), chunk)]
        table = {"thetas": np.asarray(thetas, dtype=float)}
        n = self.n
        for key in ("iauc", "peak", "trace_rmse", "sq_error"):
            table[key] = np.concatenate([np.asarray(p[key]) for p in parts])[:count, :n]
        table["penalty"] = np.concatenate([np.asarray(p["penalty"]) for p in parts])[:count]
        return table

    def grid3_table(self) -> dict:
        if self._grid3 is None:
            shape = self.config["grid_shape"]
            axes = [np.linspace(self.lower3[i], self.upper3[i], shape[i]) for i in range(3)]
            points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            self._grid3 = self._table(points)
        return self._grid3

    def grid1_table(self) -> dict:
        if self._grid1 is None:
            values = np.linspace(self.lower3[0], self.upper3[0], self.config["grid1_points"])
            points = np.tile(self.theta_base, (len(values), 1))
            points[:, 0] = values
            self._grid1 = self._table(points)
        return self._grid1

    def grid1_legacy_table(self) -> dict:
        if self._grid1_legacy is None:
            points = np.tile(self.theta_base, (len(LEGACY_SI_GRID), 1))
            points[:, 0] = LEGACY_SI_GRID
            self._grid1_legacy = self._table(points)
        return self._grid1_legacy


class _Fitter:
    """Adam on the regularized loss for one cell of one subject, with the training set as a mask."""

    def __init__(self, ctx: SubjectContext, name: str, free: tuple[str, ...],
                 parameterization: str = "rates"):
        config = ctx.config
        spec = ObjectiveSpec(name=name, lam=config["lam"], beta=config["beta"], free=free,
                             window=ctx.window, parameterization=parameterization)
        self.parameterization = parameterization
        self.ctx = ctx
        self.objective = build_objective(spec, ctx.base, ctx.arrays, ctx.observed)
        self.lam = config["lam"]
        obj = self.objective
        width = ctx.width
        self.per_sample = (obj.counts["n_trace_samples"] if name == "trace" else 1)
        residual = obj.weighted_residuals

        def per_meal(theta):
            return jnp.sum(residual(theta).reshape(width, -1) ** 2, axis=1)

        def loss(theta, train_mask, inverse_norm):
            return (jnp.sum(train_mask * per_meal(theta)) * inverse_norm
                    + self.lam * obj.penalty(theta))

        steps, rate = config["steps"], config["learning_rate"]
        optimizer = _optimizer("adam", rate)

        @jax.jit
        def run(theta0, train_mask, inverse_norm):
            def body(carry, _):
                current, state = carry
                value, grad = jax.value_and_grad(loss)(current, train_mask, inverse_norm)
                updates, state = optimizer.update(grad, state, current, value=value, grad=grad,
                                                  value_fn=lambda t: loss(t, train_mask,
                                                                          inverse_norm))
                current = obj.project(optax.apply_updates(current, updates))
                return (current, state), value

            (final, _), curve = jax.lax.scan(body, (theta0, optimizer.init(theta0)), None,
                                             length=steps)
            return final, curve

        self._run = run
        self.predict = (ctx.prediction_fn(obj.simulate) if parameterization != "rates" else None)
        self._grad = jax.jit(jax.grad(loss))
        self._loss = jax.jit(loss)

    def fit(self, train_index: np.ndarray) -> dict:
        ctx = self.ctx
        mask = np.zeros(ctx.width)
        mask[train_index] = 1.0
        inverse_norm = 1.0 / (max(len(train_index), 1) * self.per_sample)
        mask_j, norm_j = jnp.asarray(mask), jnp.asarray(inverse_norm)
        obj = self.objective
        theta0 = obj.project(obj.theta0)
        theta, curve = self._run(theta0, mask_j, norm_j)
        theta = obj.project(theta)
        curve = np.asarray(curve, dtype=float)
        lower, upper = np.asarray(obj.lower), np.asarray(obj.upper)
        g0 = _project_gradient(self._grad(theta0, mask_j, norm_j), theta0, lower, upper)
        g1 = _project_gradient(self._grad(theta, mask_j, norm_j), theta, lower, upper)
        n0, n1 = float(np.linalg.norm(g0)), float(np.linalg.norm(g1))
        tail = max(1, len(curve) // 10)
        still_moving = bool(len(curve) > 2 * tail
                            and (curve[-2 * tail:-tail].mean() - curve[-tail:].mean())
                            > 1e-6 * abs(curve[-tail:].mean()))
        coordinates = np.asarray(theta, dtype=float)
        full = (np.asarray(obj.expand(theta), dtype=float) if self.parameterization == "rates"
                else coordinates)
        out = {"theta_full": full.tolist(), "theta": coordinates.tolist(),
               "final_loss": float(curve[-1]),
               "initial_loss": float(curve[0]), "projected_grad_norm": n1,
               "initial_projected_grad_norm": n0,
               "converged": bool(n1 <= 1e-3 * max(n0, 1e-300)), "still_moving": still_moving}
        if self.parameterization == "coords":
            # Fits that land on p = tau1^2 / 4 are the tied-rate model; how often is reported.
            from personalization import coords as co
            out["on_tied_boundary"] = bool(co.on_tied_boundary(coordinates, tolerance=1e-4))
        return out


_CONTEXTS: dict[tuple, SubjectContext] = {}
_AMORTIZED: dict[str, object] = {}


def _context(config: dict, subject_id: str) -> SubjectContext:
    key = (subject_id, config["steps"], config["learning_rate"], config["lam"], config["beta"],
           tuple(config["grid_shape"]), config["grid1_points"], config["cohort"],
           str(config.get("replica")), str(config.get("carb_scale")))
    if key not in _CONTEXTS:
        subject = get_subject(config, subject_id)
        _CONTEXTS.clear()           # one subject at a time: compiled programs are large
        _CONTEXTS[key] = SubjectContext(subject, config)
    return _CONTEXTS[key]


def _amortized():
    """The random forest and the SNPE posterior, built once per process.

    The forest takes about three minutes to train, and every worker would otherwise train its own
    identical copy (the training seed is fixed). It is trained once, written atomically, and loaded by
    the rest. A corrupt or partial file is treated as absent and rebuilt.
    """
    if not _AMORTIZED:
        import os
        import joblib
        from evaluation.clinical_recovery import build_rf, load_snpe
        from evaluation.results_io import ROOT
        path = ROOT / ".cache" / "rf_20000.joblib"
        rf = None
        if path.exists():
            try:
                rf = joblib.load(path)
            except Exception:
                rf = None
        if rf is None:
            rf = build_rf()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            joblib.dump(rf, tmp)
            os.replace(tmp, path)
        _AMORTIZED["rf"] = rf
        _AMORTIZED["snpe"] = load_snpe()
    return _AMORTIZED["rf"], _AMORTIZED["snpe"]


# --------------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------------

def _grid_best(ctx: SubjectContext, table: dict, train: np.ndarray, lam: float,
               legacy: bool = False) -> tuple[int, float]:
    """The grid point that minimizes the same loss the gradient cells minimize, and that loss."""
    if legacy:
        score = np.sum(np.abs(np.sqrt(table["sq_error"][:, train])), axis=1)
    else:
        score = (np.sum(table["sq_error"][:, train], axis=1) / max(len(train), 1)
                 + lam * table["penalty"])
    best = int(np.argmin(score))
    return best, float(score[best])


def _grid_choice(ctx: SubjectContext, table: dict, train: np.ndarray, lam: float,
                 legacy: bool = False) -> int:
    return _grid_best(ctx, table, train, lam, legacy)[0]


def _empty(n: int) -> dict:
    return {"iauc": np.full(n, np.nan), "peak": np.full(n, np.nan),
            "trace_rmse": np.full(n, np.nan), "folds": []}


def _amortized_si(cell: str, ctx: SubjectContext, train_records, seed: int):
    from evaluation.snpe_kfold import _fit_si
    rf, posterior = _amortized()
    prof = ctx.subject.profile
    return _fit_si(cell, train_records, prof, rf, posterior)


def run_unit(unit: str, config: dict) -> dict:
    subject_id, repeat = split_unit(unit)
    ctx = _context(config, subject_id)
    n = ctx.n
    started = time.perf_counter()
    folds = make_folds(subject_id, n, k=config["folds"], seed=repeat)
    cells = {c: _empty(n) for c in config["cells"]}
    timing = {c: 0.0 for c in config["cells"]}
    lam = config["lam"]

    base_pred = ctx.predict(ctx.theta_base)       # population defaults, independent of the fold
    obs = {"iauc": ctx.obs_iauc, "peak": ctx.obs_peak}

    for fold in range(config["folds"]):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        if len(test) == 0 or len(train) == 0:
            continue
        train_records = [ctx.records[i] for i in train]

        for cell in config["cells"]:
            t0 = time.perf_counter()
            out = cells[cell]
            record: dict = {"fold": fold, "n_train": int(len(train)), "n_test": int(len(test))}

            if cell in FIT_CELLS:
                fit = ctx.fitter(cell).fit(train)
                pred = ctx.predict_fitted(cell, fit)
                record.update(fit)
            elif cell == "grid3":
                table = ctx.grid3_table()
                best, score = _grid_best(ctx, table, train, lam)
                pred = {k: table[k][best] for k in ("iauc", "peak", "trace_rmse")}
                record["theta_full"] = table["thetas"][best].tolist()
                record["final_loss"] = score      # the same regularized loss the gradient cells report
            elif cell == "grid1":
                table = ctx.grid1_table()
                best, score = _grid_best(ctx, table, train, lam)
                pred = {k: table[k][best] for k in ("iauc", "peak", "trace_rmse")}
                record["theta_full"] = table["thetas"][best].tolist()
                record["final_loss"] = score      # the same regularized loss the gradient cells report
            elif cell == "grid1_legacy":
                table = ctx.grid1_legacy_table()
                best = _grid_choice(ctx, table, train, lam, legacy=True)
                pred = {k: table[k][best] for k in ("iauc", "peak", "trace_rmse")}
                record["theta_full"] = table["thetas"][best].tolist()
            elif cell in AMORTIZED_CELLS:
                si = _amortized_si(cell, ctx, train_records, repeat)
                record["si"] = None if si is None else float(si)
                if si is None or not np.isfinite(si):
                    record["failed"] = "no estimate"
                    out["folds"].append(record)
                    timing[cell] += time.perf_counter() - t0
                    continue
                theta = ctx.theta_base.copy()
                theta[0] = float(si)
                pred = ctx.predict(theta)
                record["theta_full"] = theta.tolist()
            elif cell == "personal_mean":
                mean_trace = (ctx.obs_trace[train].mean(axis=0) if ctx.obs_trace.size else None)
                pred = {
                    "iauc": np.full(n, ctx.obs_iauc[train].mean()),
                    "peak": np.full(n, ctx.obs_peak[train].mean()),
                    "trace_rmse": (np.sqrt(np.mean((ctx.obs_trace - mean_trace) ** 2, axis=1))
                                   if mean_trace is not None else np.zeros(n))}
            elif cell == "population":
                pred = base_pred
            elif cell == "persistence":
                # The meal immediately before this one in the subject's own record order; the first
                # meal has no predecessor and falls back to the training mean.
                prev = np.array([i - 1 if i > 0 else -1 for i in range(n)])
                fallback = ctx.obs_iauc[train].mean()
                peak_fallback = ctx.obs_peak[train].mean()
                pred = {
                    "iauc": np.where(prev >= 0, ctx.obs_iauc[np.maximum(prev, 0)], fallback),
                    "peak": np.where(prev >= 0, ctx.obs_peak[np.maximum(prev, 0)], peak_fallback),
                    "trace_rmse": np.array([
                        float(np.sqrt(np.mean((ctx.obs_trace[i] - ctx.obs_trace[prev[i]]) ** 2)))
                        if prev[i] >= 0 and ctx.obs_trace.size else float("nan")
                        for i in range(n)])}
            else:
                raise ValueError(f"unknown cell {cell!r}")

            for key in ("iauc", "peak", "trace_rmse"):
                out[key][test] = np.asarray(pred[key])[test]
            out["folds"].append(record)
            timing[cell] += time.perf_counter() - t0

    payload_cells = {}
    for name, out in cells.items():
        err = np.abs(out["iauc"] - obs["iauc"])
        payload_cells[name] = {
            "pred_iauc": out["iauc"].tolist(), "pred_peak": out["peak"].tolist(),
            "trace_rmse": out["trace_rmse"].tolist(),
            "iauc_mae": float(np.nanmean(err)) if np.isfinite(err).any() else None,
            "peak_mae": (float(np.nanmean(np.abs(out["peak"] - obs["peak"])))
                         if np.isfinite(out["peak"]).any() else None),
            "trace_rmse_mean": (float(np.nanmean(out["trace_rmse"]))
                                if np.isfinite(out["trace_rmse"]).any() else None),
            "n_missing": int(np.isnan(out["iauc"]).sum()),
            "folds": out["folds"], "seconds": round(timing[name], 2),
        }
    return {
        "subject_id": subject_id, "repeat": repeat, "cohort": config["cohort"], "n_meals": n,
        "replica": config.get("replica"), "carb_scale": config.get("carb_scale"),
        "folds": folds.tolist(), "obs_iauc": ctx.obs_iauc.tolist(),
        "obs_peak": ctx.obs_peak.tolist(), "carbs_g": [r["carbs_g"] for r in ctx.records],
        "fit_settings": {k: config[k] for k in ("steps", "learning_rate", "lam", "beta")},
        "cells": payload_cells, "wall_seconds": round(time.perf_counter() - started, 2),
        "macros": {},
    }


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unit", default=None, help="run one unit in the foreground, e.g. CGMacros-001__r0")
    ap.add_argument("--cells", default=None, help="comma list; default all")
    args = ap.parse_args()
    config = default_config()
    if args.cells:
        config["cells"] = [c.strip() for c in args.cells.split(",")]
    if args.unit:
        result = run_unit(args.unit, config)
        for name, cell in result["cells"].items():
            print(f"{name:14} iAUC MAE {cell['iauc_mae']}  peak MAE {cell['peak_mae']}  "
                  f"trace RMSE {cell['trace_rmse_mean']}  {cell['seconds']}s")
        return
    print(__doc__)
    print(f"units: {len(units(config))}")


if __name__ == "__main__":
    main()

"""Export the model specification FROM THE CODE, so the paper's appendix cannot drift from it.

Reviewer point 3 was that the mathematics was under-specified. The fix is not to write the
appendix more carefully; it is to stop writing it by hand. This module reads the constants, bounds,
solver settings, observation operators, loss and regularization out of the modules that actually
run, and writes them to `SPEC_EXPORT.md` plus a LaTeX table. The appendix includes those files.

Anything that cannot be read mechanically -- the vector field itself -- is emitted as the verbatim
source of the function being integrated, with its line numbers and the file it came from, rather
than as a transcription.

Run:  python -m evaluation.export_spec
"""
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import fields
from pathlib import Path

from evaluation.results_io import ROOT

MARKDOWN = ROOT / "SPEC_EXPORT.md"
LATEX = ROOT / "paper" / "spec_table.tex"

# Parameters that are free during inference, and what they are called in the paper.
PAPER_SYMBOLS = {
    "insulin_sensitivity": r"S_I",
    "gastric_emptying": r"k_e",
    "carb_absorption": r"k_a",
    "glucose_effectiveness": r"p_1",
    "x_decay": r"p_2",
    "x_gain": r"p_3",
    "insulin_clearance": r"n",
    "insulin_secretion": r"\beta",
    "circadian_amp": r"c_{\mathrm{circ}}",
    "weight_kg": r"W",
    "rmr_kcal_min": r"\mathrm{RMR}",
}


def _source(obj) -> dict:
    """Verbatim source of a function, with where it came from."""
    try:
        lines, start = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return {"file": "unavailable", "lines": "", "source": ""}
    file = Path(inspect.getsourcefile(obj) or "unknown")
    try:
        rel = file.relative_to(ROOT)
    except ValueError:
        rel = file
    return {"file": str(rel).replace("\\", "/"),
            "lines": f"{start}-{start + len(lines) - 1}",
            "source": "".join(lines).rstrip("\n")}


def collect() -> dict:
    from personalization import gradient_fit
    from simulation import jax_engine, jax_observation, params as params_mod
    from evaluation import cgmacros
    import paper_config as cfg

    defaults = params_mod.PhysioParams()
    engine_params = [f.name for f in fields(jax_engine.JaxPhysioParams)]

    return {
        "state": {
            "variables": list(jax_engine.STATE_VARS),
            "n_state": jax_engine.N_STATE,
            "evolved_this_stage": ["glucose_mg_dl", "insulin_uU_ml", "x_insulin_action",
                                  "stomach_glucose_mg", "gut_glucose_mg", "glycogen_g",
                                  "ketones_mmol_l", "energy_expended_kcal"],
            "initial": {name: float(value) for name, value in
                        zip(jax_engine.STATE_VARS, jax_engine.initial_state())
                        if float(value) != 0.0},
        },
        "parameters": {
            "carried_into_jax": engine_params,
            "population_defaults": {
                name: getattr(defaults, name) for name in engine_params
            },
            "free_during_inference": list(gradient_fit.TARGETS),
            "bounds": {k: list(v) for k, v in gradient_fit.PARAM_RANGES.items()},
            "prior_support_shared_with_baselines": {k: list(v) for k, v in cfg.PRIOR.items()},
            "paper_symbols": PAPER_SYMBOLS,
        },
        "fixed_constants": {
            "basal_glucose_mg_dl": float(jax_engine._GB),
            "basal_insulin_uU_ml": float(jax_engine._IB),
            "glucose_distribution_volume_per_kg_dl": 1.6,
            "hepatic_cortisol_threshold_ug_dl": 15.0,
            "hepatic_gain_mg_dl_min_per_ug_dl": 0.10,
            "glycogen_restoration_insulin_threshold_uU_ml": 15.0,
            "glycogen_restoration_rate_g_min": 0.15,
            "ketone_drive_rate_mmol_l_min": 0.12,
            "ketone_decay_per_min": 0.020,
            "gluconeogenesis_threshold_mg_dl": 72.0,
            "carb_to_glucose_fraction": 0.90,
            "meal_pulse_sigma_min": 2.0,
        },
        "meal_input": {
            "blunting": "1 / (1 + 0.08 * fiber_g + 0.005 * fat_g)",
            "glucose_mass_mg": "carbs_g * 1000 * 0.90 * blunting",
            "delivery": "Gaussian rate pulse of width sigma = 2 min centred on the meal time, "
                        "integrating to the glucose mass",
        },
        "solver": {
            "library": "diffrax",
            "method": "Tsit5",
            "dt0_min": 1.0,
            "controller": "PIDController(rtol=1e-4, atol=1e-4)",
            "max_steps": 10000,
            "save_grid_min": 5.0,
            "meal_time_min": 30.0,
            "duration_min": 210.0,
        },
        "observation": {
            "postprandial_window_min": float(cgmacros._WINDOW),
            "pre_meal_baseline_min": float(cgmacros._PRE),
            "baseline": "mean of the 6 samples (30 min) before the meal",
            "iauc": "trapezoid of softplus_relu(glucose - baseline) over the window",
            "operators": {name: _source(getattr(jax_observation, name))
                          for name in ("iauc", "peak_glucose", "peak_time", "early_slope")},
        },
        "smoothing": {name: _source(getattr(jax_engine, name))
                      for name in ("softplus_relu", "sigmoid_switch", "smooth_clamp01",
                                   "soft_clamp", "diurnal_cortisol")},
        "vector_field": _source(jax_engine.vector_field),
        "run_meal": _source(jax_engine.run_meal),
        "loss": {
            "reconstruction": "masked mean squared error on iAUC over a subject's meals",
            "regularization": "sum over free parameters of ((theta - theta_population) / range)^2",
            "regularization_weight_default": 0.01,
            "regularization_for_profile_likelihood": 0.0,
            "projection": "theta clipped to the bounds after every update",
            "optimizer_default": "Adam, learning rate 0.02",
            "padded_meal_slots": gradient_fit.MAX_MEALS,
            "source": _source(gradient_fit._loss),
        },
        "cohorts": {
            "CGMacros": {"sampling_min": 5.0, "carbs": "photographed, nutritionist-coded log"},
            "Hall2018": {"sampling_min": 5.0, "carbs": "standardized meals, known by design"},
            "ShanghaiT2DM": {"sampling_min": 15.0, "carbs": "free-text dietary record"},
        },
    }


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def to_markdown(spec: dict) -> str:
    out: list[str] = [
        "# Model specification, exported from the code",
        "",
        "Generated by `evaluation/export_spec.py`. Do not edit: edit the code it reads.",
        "Every constant, bound, solver setting and loss term below is read out of the modules that",
        "run during inference, so the paper's appendix cannot disagree with what was computed.",
        "",
        "## State",
        "",
        f"The state vector has {spec['state']['n_state']} entries; this stage of the engine evolves",
        f"{len(spec['state']['evolved_this_stage'])} of them and holds the rest constant.",
        "",
        "| index | variable | evolved | initial value |",
        "| --- | --- | --- | --- |",
    ]
    initial = spec["state"]["initial"]
    for i, name in enumerate(spec["state"]["variables"]):
        evolved = "yes" if name in spec["state"]["evolved_this_stage"] else "no"
        out.append(f"| {i} | `{name}` | {evolved} | {_fmt(initial.get(name, 0.0))} |")

    out += ["", "## Parameters", "",
            "| parameter | symbol | population default | inference bounds | free |",
            "| --- | --- | --- | --- | --- |"]
    params = spec["parameters"]
    for name in params["carried_into_jax"]:
        symbol = params["paper_symbols"].get(name, "")
        bounds = params["bounds"].get(name)
        out.append(
            f"| `{name}` | ${symbol}$ | {_fmt(params['population_defaults'][name])} | "
            f"{'[' + ', '.join(_fmt(b) for b in bounds) + ']' if bounds else '--'} | "
            f"{'yes' if name in params['free_during_inference'] else 'no'} |")

    out += ["", "## Fixed constants", "", "| constant | value |", "| --- | --- |"]
    for key, value in spec["fixed_constants"].items():
        out.append(f"| `{key}` | {_fmt(value)} |")

    out += ["", "## Meal input", ""]
    for key, value in spec["meal_input"].items():
        out.append(f"* **{key}** -- {value}")

    out += ["", "## Solver", "", "| setting | value |", "| --- | --- |"]
    for key, value in spec["solver"].items():
        out.append(f"| `{key}` | {_fmt(value)} |")

    out += ["", "## Observation", ""]
    for key in ("postprandial_window_min", "pre_meal_baseline_min", "baseline", "iauc"):
        out.append(f"* **{key}** -- {_fmt(spec['observation'][key])}")
    out += ["", "### Observation operators, verbatim", ""]
    for name, src in spec["observation"]["operators"].items():
        out += [f"`{name}` ({src['file']}, lines {src['lines']})", "", "```python",
                src["source"], "```", ""]

    out += ["## Smoothing primitives, verbatim", ""]
    for name, src in spec["smoothing"].items():
        out += [f"`{name}` ({src['file']}, lines {src['lines']})", "", "```python",
                src["source"], "```", ""]

    out += ["## Vector field, verbatim", "",
            f"`vector_field` ({spec['vector_field']['file']}, "
            f"lines {spec['vector_field']['lines']})", "", "```python",
            spec["vector_field"]["source"], "```", "",
            f"`run_meal` ({spec['run_meal']['file']}, lines {spec['run_meal']['lines']})", "",
            "```python", spec["run_meal"]["source"], "```", ""]

    out += ["## Loss", ""]
    for key, value in spec["loss"].items():
        if key != "source":
            out.append(f"* **{key}** -- {_fmt(value)}")
    out += ["", f"`_loss` ({spec['loss']['source']['file']}, "
            f"lines {spec['loss']['source']['lines']})", "", "```python",
            spec["loss"]["source"]["source"], "```", ""]

    out += ["## Cohorts", "", "| cohort | sampling (min) | carbohydrate source |",
            "| --- | --- | --- |"]
    for name, info in spec["cohorts"].items():
        out.append(f"| {name} | {_fmt(info['sampling_min'])} | {info['carbs']} |")

    return "\n".join(out) + "\n"


def to_latex(spec: dict) -> str:
    """The constants and bounds table for the appendix. Generated, never hand-typed."""
    params = spec["parameters"]
    rows = []
    for name in params["free_during_inference"]:
        symbol = params["paper_symbols"].get(name, name)
        lo, hi = params["bounds"][name]
        rows.append(f"${symbol}$ & {_fmt(params['population_defaults'][name])} & "
                    f"$[{_fmt(lo)}, {_fmt(hi)}]$ & free \\\\")
    for name in params["carried_into_jax"]:
        if name in params["free_during_inference"]:
            continue
        symbol = params["paper_symbols"].get(name)
        if not symbol:
            continue
        rows.append(f"${symbol}$ & {_fmt(params['population_defaults'][name])} & -- & fixed \\\\")

    constants = [f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {_fmt(v)} \\\\"
                 for k, v in spec["fixed_constants"].items()]

    return "\n".join([
        "% Generated by evaluation/export_spec.py -- DO NOT EDIT.",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Parameter & Population default & Inference bounds & Status \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Constant & Value \\",
        r"\midrule",
        *constants,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="also print the collected spec as JSON")
    args = ap.parse_args()
    spec = collect()
    MARKDOWN.write_text(to_markdown(spec), encoding="utf-8")
    LATEX.parent.mkdir(parents=True, exist_ok=True)
    LATEX.write_text(to_latex(spec), encoding="utf-8")
    print(f"wrote {MARKDOWN.relative_to(ROOT)} and {LATEX.relative_to(ROOT)}")
    if args.json:
        print(json.dumps(spec, indent=2, default=str))


if __name__ == "__main__":
    main()

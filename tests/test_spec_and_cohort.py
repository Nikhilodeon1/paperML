"""The exported specification and the cohort loader.

The specification export exists so the paper's appendix is generated rather than transcribed; these
tests check that it actually reads the live code, so a changed constant changes the export instead
of leaving a stale appendix behind.
"""
from __future__ import annotations

import pytest

from evaluation import export_spec
from evaluation.cohort_data import cohort_summary, load_cgmacros
from evaluation.results_io import ROOT


@pytest.fixture(scope="module")
def spec():
    return export_spec.collect()


def test_spec_reads_the_live_bounds(spec):
    """The bounds in the export must BE the bounds the fit uses, not a copy."""
    from personalization.gradient_fit import PARAM_RANGES

    assert spec["parameters"]["bounds"] == {k: list(v) for k, v in PARAM_RANGES.items()}
    assert spec["parameters"]["free_during_inference"] == ["insulin_sensitivity",
                                                          "gastric_emptying", "carb_absorption"]


def test_spec_reads_the_live_constants(spec):
    from simulation.jax_engine import _GB, _IB

    assert spec["fixed_constants"]["basal_glucose_mg_dl"] == float(_GB)
    assert spec["fixed_constants"]["basal_insulin_uU_ml"] == float(_IB)


def test_spec_carries_the_vector_field_verbatim(spec):
    """Not a transcription: the exported text must be the source that is integrated."""
    import inspect

    from simulation.jax_engine import vector_field

    assert spec["vector_field"]["source"].strip() == inspect.getsource(vector_field).strip()
    assert spec["vector_field"]["file"].endswith("simulation/jax_engine.py")
    assert "-" in spec["vector_field"]["lines"]


def test_spec_records_the_solver_settings(spec):
    assert spec["solver"]["method"] == "Tsit5"
    assert spec["solver"]["save_grid_min"] == 5.0
    assert spec["solver"]["duration_min"] == 210.0


def test_spec_records_each_cohort_sampling_interval(spec):
    assert spec["cohorts"]["ShanghaiT2DM"]["sampling_min"] == 15.0
    assert spec["cohorts"]["CGMacros"]["sampling_min"] == 5.0


def test_markdown_and_latex_render(spec):
    markdown = export_spec.to_markdown(spec)
    latex = export_spec.to_latex(spec)
    assert "## Vector field, verbatim" in markdown
    assert "| `insulin_sensitivity` | $S_I$ |" in markdown
    assert r"\begin{tabular}" in latex
    assert "$S_I$" in latex
    assert "DO NOT EDIT" in latex


def test_exported_files_are_present_and_current():
    """The committed export must match what the current code would produce."""
    path = ROOT / "SPEC_EXPORT.md"
    if not path.exists():
        pytest.skip("SPEC_EXPORT.md not generated yet")
    assert path.read_text(encoding="utf-8") == export_spec.to_markdown(export_spec.collect())


# --- cohort loader ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cgmacros():
    try:
        subjects = load_cgmacros()
    except FileNotFoundError as exc:
        pytest.skip(f"CGMacros not available: {exc}")
    if not subjects:
        pytest.skip("CGMacros loaded but empty")
    return subjects


def test_cohort_is_sorted_and_deduplicated(cgmacros):
    ids = [s.subject_id for s in cgmacros]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_every_subject_meets_the_meal_minimum(cgmacros):
    assert all(s.n_meals >= 10 for s in cgmacros)


def test_meal_count_fits_the_padded_width(cgmacros):
    """`MAX_MEALS` must still exceed the busiest subject, or meals would be silently dropped."""
    from personalization.gradient_fit import MAX_MEALS

    assert max(s.n_meals for s in cgmacros) <= MAX_MEALS


def test_traces_are_on_the_expected_grid(cgmacros):
    """43 samples at 5 min covers -30 to +180 min, which is what the observables assume."""
    record = cgmacros[0].records[0]
    assert record["glucose"]["step_min"] == 5.0
    assert record["glucose"]["t0_min"] == -30.0
    assert len(record["glucose"]["values"]) == 43


def test_summary_counts_are_consistent(cgmacros):
    summary = cohort_summary(cgmacros)
    assert summary["n_subjects"] == len(cgmacros)
    assert summary["n_meals_total"] == sum(s.n_meals for s in cgmacros)
    assert summary["sampling_min"] == 5.0


def test_loading_twice_gives_the_same_object_identity(cgmacros):
    """The in-process cache must be in play; re-parsing takes about 40 seconds."""
    assert load_cgmacros() is cgmacros

"""Phase 7: the verdict logic, the macro names, and the anonymization scan.

These are the pieces that sit between a result file and a sentence in the paper, so the failure to
guard against is a verdict computed against the wrong threshold, or a macro name TeX would reject.
"""
from __future__ import annotations

import re

import pytest

from evaluation import hypotheses as hyp
from evaluation import package_anonymized as anon
from evaluation import paper_numbers as pn


def _summary(**overrides):
    base = {
        "prediction": {
            "h7_h8": {
                "grad3_vs_grid3": {"mean_difference": -1.0, "equivalent_at": {"90": True, "150": True, "300": True}},
                "grad1_vs_grid1": {"mean_difference": 0.0, "equivalent_at": {"90": True, "150": True, "300": True}},
                "grad3_vs_grad1": {"mean_difference": -34.0, "equivalent_at": {"90": True, "150": True, "300": True}},
            },
            "h9": {"grad3_trace_vs_grad3": {"lowers_error_by_more_than_margin": False, "p_holm": 1.0}},
        },
        "steps_sensitivity": {"robust_to_budget": True},
        "profile_likelihood": {"1x": {"n_subjects": 45, "parameters": {
            "insulin_sensitivity": {"fraction": 0.07, "wilson": [0.02, 0.18]},
            "gastric_emptying": {"fraction": 0.0, "wilson": [0, 0.08]},
            "carb_absorption": {"fraction": 0.0, "wilson": [0, 0.08]}},
            "S_I_bounded_among_interior_for_S_I": {"bounded": 3, "n": 22, "fraction": 3 / 22,
                                                   "wilson": [0.05, 0.33]}}},
        "ladder": {"1x": {"h3_own_theta_hat": {"h3_met": False, "median_condition_number": {}},
                          "h4_cosine_timing_block": {"iauc_centroid": {"median": 0.99, "n": 45}}}},
    }
    base.update(overrides)
    return base


def test_h7_met_and_h9_not_met_follow_the_plan_rules():
    v = hyp.evaluate_all(_summary(), moment={})
    assert v["H7"]["status"] == "met" and v["H8"]["status"] == "met"
    assert v["H9"]["status"] == "not met"
    assert v["H4"]["status"] == "met"
    assert v["H3"]["status"] == "not met"


def test_h5_requires_both_clauses_and_reports_not_evaluated_for_the_trace():
    v = hyp.evaluate_all(_summary(), moment={})
    assert v["H5"]["status"] == "not met"               # S_I clause 14% < 80%
    assert "timing/trace: not evaluated" in v["H5"]["note"]
    good = _summary()
    good["profile_likelihood"]["1x"]["S_I_bounded_among_interior_for_S_I"].update(fraction=0.9)
    assert hyp.evaluate_all(good, moment={})["H5"]["note"].startswith("timing/iAUC: met; S_I: met")


def test_h7_fails_when_a_pair_is_not_equivalent():
    s = _summary()
    s["prediction"]["h7_h8"]["grad3_vs_grid3"]["equivalent_at"]["150"] = False
    assert hyp.evaluate_all(s, moment={})["H7"]["status"] == "not met"


def test_unrun_hypotheses_are_not_evaluated_not_guessed():
    v = hyp.evaluate_all({}, moment={})
    for name in ("H6", "H10", "H11", "H12", "H2"):
        assert v[name]["status"] == "not evaluated"
    assert v["H1"]["status"] == "formulation-dependent"
    assert sum(1 for r in v.values() if r["primary"]) == 5


def test_h2_uses_both_thresholds():
    ok = {"h2": {"met": True, "linear_360_ke_median": 0.004}}
    assert hyp.evaluate_all(_summary(), moment=ok)["H2"]["status"] == "met"
    bad = {"h2": {"met": False}}
    assert hyp.evaluate_all(_summary(), moment=bad)["H2"]["status"] == "not met"


def test_macro_names_are_alphabetic_only_for_every_cell_and_verdict():
    summary = {
        "prediction": {
            "cells": {c: {"mean": 3000.0, "sd": 10.0} for c in
                      ("grad3", "grad1_trace", "grid1_legacy", "personal_mean")},
            "comparisons": [{"first": "grad3_trace", "second": "grad3", "group": "superiority",
                             "mean_difference": 30.0, "ci95": {"low": 1.0, "high": 60.0},
                             "wins_first": 20, "n": 45, "p_holm": 0.5}]},
        "bounds_sweep_fisher_fit": {"1x": {"n_subjects": 45, "pinned": {
            p: {"n": 10, "fraction": 0.2, "lower": 5, "upper": 5} for p in pn.PARAM_WORD},
            "interior_for_parameter": {"insulin_sensitivity": {"n": 22}},
            "interior_converged_original": 0}},
    }
    macros = pn.build(summary, hyp.evaluate_all({}, moment={}))
    bad = [name for name in macros if not re.fullmatch(r"[A-Za-z]+", name)]
    assert not bad, bad
    assert "resMaeGradThree" in macros and "resMaeGridOneLegacy" in macros
    assert macros["resVerdictHSeven"]["value"] == "not evaluated"


def test_scientific_notation_macro_typesets():
    assert pn.sci(2.8e7) == r"$2.8\times10^{7}$"
    assert pn.sci(8.15e-5) == r"$8.2\times10^{-5}$"
    assert pn.sci(float("inf")) == r"\infty"


def test_cell_word_maps_digits():
    assert pn.cell_word("grad3_trace") == "GradThreeTrace"
    assert pn.cell_word("grid1_legacy") == "GridOneLegacy"


@pytest.mark.parametrize("text, label", [
    ("write to someone@example.org please", "e-mail address"),
    (r"C:\Users\someone\Codes", "windows home directory"),
    ("/home/someone/project", "unix home directory"),
    ("running on jupyter-someone-40mail", "hosted notebook name"),
])
def test_generic_identifier_shapes_are_caught(text, label):
    assert anon.GENERIC[label].search(text)


def test_scan_reports_forbidden_words_and_not_clean_files(tmp_path, monkeypatch):
    clean = tmp_path / "ok.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    dirty = tmp_path / "bad.py"
    dirty.write_text("# written by Jane Surnameson\n", encoding="utf-8")
    monkeypatch.setattr(anon, "ROOT", tmp_path)
    monkeypatch.setenv("ANON_FORBIDDEN", "surnameson")
    findings = anon.scan([clean, dirty])
    assert len(findings) == 1 and "bad.py:1" in findings[0]


def test_package_allow_list_excludes_product_code_and_data():
    files = {str(p.relative_to(anon.ROOT)).replace("\\", "/") for p in anon.candidate_files()}
    assert not any(f.startswith(("user_data/", "data/", "logs/", ".venv/")) for f in files)
    assert "personalization/auth.py" not in files
    assert "evaluation/runner.py" in files

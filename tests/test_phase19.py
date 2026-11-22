"""Phase 19 (Amendment 15, report-only): the curvature diagnostic and its summary, without result files."""
from __future__ import annotations

import numpy as np

from evaluation import carb_curvature, paper_numbers, phase19_summary as p19


def test_curvature_summary_is_empty_without_results(monkeypatch):
    monkeypatch.setattr(p19, "largest_matching", lambda analysis_id, predicate: {})
    out = p19.curvature()
    assert all(block == {"n": 0} for block in out.values())


def test_schur_ratio_is_paired_per_subject(monkeypatch):
    def rows(scale):
        return {"u1": {"timing_schur_eigenvalues": [1.0 * scale, 10.0 * scale], "hessian_eigenvalues": [1.0, 2.0, 3.0],
                       "load_ratio_squared": 1.0 + scale / 10, "noise_innovation_sd": 4.0, "subject_id": "u1",
                       "replica": {"seed": 0, "cgm": True, "carb_cv": 0.0}}}
    monkeypatch.setattr(p19, "_curvature_rows", lambda cv: rows(1.0 if cv == 0.0 else 2.0))
    out = p19.curvature()
    assert out["0.75"]["schur_ratio_median"] == [2.0, 2.0]
    assert out["0.75"]["schur_trace_ratio_median"] == 2.0


def test_macros_are_alphabetic_and_empty_for_no_summary():
    assert paper_numbers._phase19_macros({}) == {}
    macros = paper_numbers._phase19_macros({"r22": {"n": 45, "median_log_error": -0.2, "pinned": 19},
                                            "r23": {"n": 45, "S_I": 3, "k_e": 0, "k_a": 0},
                                            "r21": {"0.25": {"n": 45, "load_ratio_squared": 1.06, "median_sigma": 6.0,
                                                             "schur_trace_ratio_median": 1.1, "schur_eig_median": [60.0, 330.0],
                                                             "schur_ratio_median": [1.0, 1.2]}}})
    assert macros["resExactSIPinned"]["value"] == 19 and macros["resRealIaucSIBoundedAll"]["value"] == 3
    assert all(name.isalpha() for name in macros)
    assert macros["resCurvQuarterSchurHighRatio"]["value"] == 1.2


def test_load_ratio_of_unit_mean_multipliers_exceeds_one():
    rng = np.random.default_rng(0)
    cv = 0.75
    sigma2 = np.log1p(cv ** 2)
    m = rng.lognormal(-0.5 * sigma2, np.sqrt(sigma2), 200_000)
    assert abs(m.mean() - 1) < 0.02 and abs((m ** 2).mean() - (1 + cv ** 2)) < 0.1
    assert carb_curvature.ANALYSIS_ID == "A20_carb_curvature"

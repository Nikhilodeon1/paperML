"""Smoke tests for figure generation (Block 8) — files created, no visual assertions."""
from __future__ import annotations

import pytest

import paper_config as cfg


def test_figures_module_imports_and_config():
    from evaluation import figures
    assert "snpe" in figures.COLORS and "rf" in figures.COLORS
    assert cfg.FIGURES.name == "figures"


@pytest.mark.slow
def test_clinical_recovery_figure_generates():
    if not cfg.SNPE_POSTERIOR.exists():
        pytest.skip("no trained SNPE artifact")
    from evaluation import figures
    path = figures.fig_clinical_recovery()
    assert path.exists()
    assert path.suffix == ".pdf"
    assert path.with_suffix(".png").exists()

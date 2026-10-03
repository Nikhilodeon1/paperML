"""The results pipeline: provenance, resumability, freezing, macro generation, number linting.

These tests run against a temporary results directory, never the real one, so running the suite can
never touch a committed result file.
"""
from __future__ import annotations

import json

import pytest

from evaluation import freeze_results, lint_numbers, make_macros, results_io


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Redirect every module that holds a reference to the results directory."""
    target = tmp_path / "results"
    target.mkdir()
    monkeypatch.setattr(results_io, "RESULTS", target)
    monkeypatch.setattr(results_io, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(freeze_results, "RESULTS", target)
    monkeypatch.setattr(freeze_results, "INDEX", target / "index.json")
    monkeypatch.setattr(freeze_results, "ROOT", tmp_path)
    monkeypatch.setattr(results_io, "ROOT", tmp_path)
    return target


CONFIG = {"objective": "iauc", "steps": 500, "free": ["S_I", "k_e", "k_a"]}


def test_config_hash_is_stable_and_order_independent():
    a = results_io.config_hash({"b": 2, "a": 1}, seed=0)
    b = results_io.config_hash({"a": 1, "b": 2}, seed=0)
    assert a == b
    assert results_io.config_hash({"a": 1}, seed=0) != results_io.config_hash({"a": 1}, seed=1)
    assert results_io.config_hash({"a": 1}) != results_io.config_hash({"a": 1.0000001})


def test_save_and_load_round_trip(results_dir):
    path = results_io.save_result("A5", {"profile": [1.0, 2.0]}, CONFIG, unit="CGMacros-001")
    assert path.exists()
    doc = results_io.load_result("A5", CONFIG, unit="CGMacros-001")
    assert doc["payload"]["profile"] == [1.0, 2.0]
    assert doc["analysis_id"] == "A5"
    assert doc["config_hash"] == results_io.config_hash(CONFIG)
    assert doc["commit"]
    assert doc["environment"]["packages"]["numpy"] != "absent"


def test_result_files_are_not_overwritten(results_dir):
    results_io.save_result("A5", {"v": 1}, CONFIG, unit="s1")
    with pytest.raises(FileExistsError, match="not overwritten"):
        results_io.save_result("A5", {"v": 2}, CONFIG, unit="s1")
    results_io.save_result("A5", {"v": 2}, CONFIG, unit="s1", overwrite=True)
    assert results_io.load_result("A5", CONFIG, unit="s1")["payload"]["v"] == 2


def test_a_changed_config_lands_in_a_new_directory(results_dir):
    results_io.save_result("A5", {"v": 1}, CONFIG, unit="s1")
    other = {**CONFIG, "steps": 100}
    results_io.save_result("A5", {"v": 2}, other, unit="s1")
    directories = sorted(p.name for p in (results_dir / "A5").iterdir())
    assert len(directories) == 2, directories


def test_unit_done_skips_completed_work(results_dir):
    assert results_io.unit_done("A5", CONFIG, "s1") is False
    results_io.save_result("A5", {"v": 1}, CONFIG, unit="s1")
    assert results_io.unit_done("A5", CONFIG, "s1") is True


def test_a_truncated_file_counts_as_not_done(results_dir):
    """A job killed mid-write must be redone, not carried into the paper half-written."""
    path = results_io.save_result("A5", {"v": 1}, CONFIG, unit="s1")
    path.write_text('{"analysis_id": "A5", "payl', encoding="utf-8")
    assert results_io.unit_done("A5", CONFIG, "s1") is False


def test_unit_names_with_separators_are_made_safe(results_dir):
    path = results_io.save_result("A5", {"v": 1}, CONFIG, unit="sub/ject 01")
    assert path.name == "sub_ject_01.json"
    assert results_io.load_result("A5", CONFIG, unit="sub/ject 01")["unit"] == "sub/ject 01"


def test_load_units_collects_everything_and_flags_unreadable(results_dir):
    for i in range(3):
        results_io.save_result("A5", {"v": i}, CONFIG, unit=f"s{i}")
    bad = results_io.analysis_dir("A5", CONFIG) / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    units = results_io.load_units("A5", CONFIG)
    assert {"s0", "s1", "s2"} <= set(units)
    assert units["_unreadable"]["files"] == ["broken.json"]


def test_log_writes_a_timestamped_line(results_dir, tmp_path):
    results_io.log("A5", "started 45 subjects")
    text = (tmp_path / "logs" / "A5.log").read_text(encoding="utf-8")
    assert "started 45 subjects" in text
    assert "[A5]" in text


# --- freezing and macro generation --------------------------------------------------------------

def test_freeze_collects_macros_and_hashes(results_dir):
    results_io.save_result("A5", {"macros": {"resProfileFlatFraction": 0.78}}, CONFIG, unit="s1")
    results_io.save_result("A9", {"macros": {
        "resGradVsGridDiff": {"value": -12.0, "low": -88.0, "high": 61.0, "fmt": ".0f"}}},
        CONFIG, unit="s1")
    index = freeze_results.freeze()
    assert index["n_files"] == 2
    assert index["macros"]["resProfileFlatFraction"]["value"] == 0.78
    assert index["macros"]["resGradVsGridDiff"]["low"] == -88.0
    assert len(index["sha256_over_all"]) == 64
    assert index["collisions"] == []


def test_freeze_check_detects_a_modified_result(results_dir):
    path = results_io.save_result("A5", {"macros": {"resX": 1.0}}, CONFIG, unit="s1")
    freeze_results.freeze()
    assert freeze_results.check() == 0
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["payload"]["macros"]["resX"] = 2.0
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert freeze_results.check() == 1


def test_a_macro_claimed_twice_with_different_values_is_an_error(results_dir):
    results_io.save_result("A5", {"macros": {"resX": 1.0}}, CONFIG, unit="s1")
    results_io.save_result("A9", {"macros": {"resX": 2.0}}, CONFIG, unit="s1")
    with pytest.raises(SystemExit, match="declared twice"):
        freeze_results.freeze()


def test_make_macros_emits_interval_endpoints_separately(results_dir, tmp_path, monkeypatch):
    results_io.save_result("A9", {"macros": {
        "resGradVsGridDiff": {"value": -12.4, "low": -88.0, "high": 61.0, "fmt": ".0f",
                              "unit": "mg/dL*min"}}}, CONFIG, unit="s1")
    freeze_results.freeze()
    monkeypatch.setattr(make_macros, "INDEX", tmp_path / "results" / "index.json")
    text, count = make_macros.build()
    assert count == 3
    assert r"\newcommand{\resGradVsGridDiff}{-12}" in text
    assert r"\newcommand{\resGradVsGridDiffLow}{-88}" in text
    assert r"\newcommand{\resGradVsGridDiffHigh}{61}" in text
    assert "mg/dL*min" in text


def test_make_macros_rejects_names_tex_cannot_define(results_dir, tmp_path, monkeypatch):
    results_io.save_result("A9", {"macros": {"res_table_2": 1.0}}, CONFIG, unit="s1")
    freeze_results.freeze()
    monkeypatch.setattr(make_macros, "INDEX", tmp_path / "results" / "index.json")
    with pytest.raises(SystemExit, match="alphabetic"):
        make_macros.build()


def test_make_macros_marks_a_missing_or_nan_value_loudly(results_dir, tmp_path, monkeypatch):
    results_io.save_result("A9", {"macros": {"resMissing": {"value": None},
                                             "resNaN": {"value": float("nan")}}},
                           CONFIG, unit="s1")
    freeze_results.freeze()
    monkeypatch.setattr(make_macros, "INDEX", tmp_path / "results" / "index.json")
    text, _ = make_macros.build()
    assert r"\textbf{??}" in text and r"\textbf{NaN}" in text


# --- number linting -----------------------------------------------------------------------------

def test_lint_flags_a_hand_typed_result_number():
    tex = r"The gradient fit reached a held-out MAE of 1270 mg/dL$\cdot$min."
    findings = lint_numbers.scan_text(tex, allow=set())
    assert [f["number"] for f in findings] == ["1270"]


def test_lint_accepts_a_macro():
    tex = r"The gradient fit reached a held-out MAE of \resTableTwoGradMAE mg/dL$\cdot$min."
    assert lint_numbers.scan_text(tex, allow=set()) == []


def test_lint_ignores_math_citations_and_years():
    tex = "\n".join([
        r"Following \citet{dallaman2007} and \cite{bergman1979,caumo2000}, we set $k_e = 0.025$.",
        r"\begin{equation} \frac{dG}{dt} = -(p_1 + X) G + p_1 G_b \end{equation}",
        r"The 2007 model and the 1979 minimal model agree \citep[see][p.~3]{smith2020}.",
        r"\includegraphics[width=0.48\columnwidth]{fig1.pdf}",
        r"% a comment with 1234 in it",
        r"\label{tab:results2}",
    ])
    assert lint_numbers.scan_text(tex, allow=set()) == []


def test_lint_ignores_verbatim_blocks():
    tex = "\n".join([r"\begin{verbatim}", "steps = 500", r"\end{verbatim}"])
    assert lint_numbers.scan_text(tex, allow=set()) == []


def test_lint_respects_the_allowlist():
    tex = r"We use 5-fold cross-validation repeated 5 times."
    assert len(lint_numbers.scan_text(tex, allow=set())) == 2
    assert lint_numbers.scan_text(tex, allow={"5"}) == []


def test_lint_reports_line_numbers():
    tex = "clean line\n\nthe median was 418\n"
    findings = lint_numbers.scan_text(tex, allow=set(), name="paper/main.tex")
    assert findings[0]["line"] == 3
    assert findings[0]["file"] == "paper/main.tex"

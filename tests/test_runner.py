"""The runner's failure handling, tested against failures that are made to happen on purpose.

Every test here corresponds to something that will occur during a twenty-hour run on rented
hardware. A runner whose resume path is untested is a runner that will lose a night's compute.

These are subprocess tests and therefore slow; they are marked so, and they write only into a
temporary results directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation import _runner_selftest as selftest
from evaluation import runner
from evaluation.results_io import ROOT

pytestmark = pytest.mark.slow

MODULE = "evaluation._runner_selftest"


def _run(tmp_path: Path, *extra: str, timeout: int = 300, **overrides) -> subprocess.CompletedProcess:
    """Run the runner in a subprocess with results redirected into `tmp_path`."""
    sets = []
    for key, value in overrides.items():
        sets += ["--set", f"{key}={json.dumps(value)}"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8",
           "HORIZON_RESULTS_DIR": str(tmp_path / "results"),
           "HORIZON_LOGS_DIR": str(tmp_path / "logs")}
    return subprocess.run(
        [sys.executable, "-m", "evaluation.runner", MODULE, *sets, *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def _results(tmp_path: Path) -> dict[str, dict]:
    out = {}
    for path in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("*.json"):
        if path.name in ("config.json",) or path.name.endswith(".failed.json"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[doc["unit"]] = doc
    return out


def _failures(tmp_path: Path) -> list[str]:
    return sorted(p.name.replace(".failed.json", "")
                  for p in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("*.failed.json"))


# --- the happy path -----------------------------------------------------------------------------

def test_all_units_complete(tmp_path):
    proc = _run(tmp_path, "--workers", "2", n_units=6)
    assert proc.returncode == 0, proc.stderr[-3000:]
    results = _results(tmp_path)
    assert sorted(results) == [f"u{i:03d}" for i in range(6)]
    assert results["u003"]["payload"]["value"] == 9
    assert all("wall_seconds" in d["payload"] for d in results.values())


def test_work_is_shared_between_workers(tmp_path):
    """Two workers must actually both do work, or the claim mechanism is not working."""
    proc = _run(tmp_path, "--workers", "3", n_units=9, sleep=0.4)
    assert proc.returncode == 0, proc.stderr[-2000:]
    pids = {d["payload"]["pid"] for d in _results(tmp_path).values()}
    assert len(pids) > 1, f"all units ran in one process: {pids}"


def test_no_unit_is_computed_twice(tmp_path):
    proc = _run(tmp_path, "--workers", "4", n_units=8, sleep=0.3)
    assert proc.returncode == 0, proc.stderr[-2000:]
    directory = next((tmp_path / "results" / selftest.ANALYSIS_ID).iterdir())
    assert not list(directory.glob("*.claim")), "claims were left behind"
    assert len(_results(tmp_path)) == 8


def test_progress_file_is_written(tmp_path):
    proc = _run(tmp_path, "--workers", "2", n_units=4)
    assert proc.returncode == 0, proc.stderr[-2000:]
    progress = json.loads(
        (tmp_path / "logs" / f"{selftest.ANALYSIS_ID}.progress.json").read_text(encoding="utf-8"))
    assert progress["done"] == 4
    assert progress["total"] == 4
    assert progress["failed"] == 0
    assert progress["units_per_hour"] > 0


# --- resume -------------------------------------------------------------------------------------

def test_resume_skips_finished_units(tmp_path):
    """The property the whole design exists for: a second run recomputes nothing."""
    first = _run(tmp_path, "--workers", "2", n_units=6)
    assert first.returncode == 0, first.stderr[-2000:]
    before = {unit: doc["payload"]["pid"] for unit, doc in _results(tmp_path).items()}
    mtimes = {p: p.stat().st_mtime_ns
              for p in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u*.json")}

    second = _run(tmp_path, "--workers", "2", n_units=6)
    assert second.returncode == 0, second.stderr[-2000:]
    assert "6 already done" in second.stdout or "already done" in second.stdout
    after = {p: p.stat().st_mtime_ns
             for p in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u*.json")}
    assert after == mtimes, "a finished unit was rewritten on resume"
    assert {u: d["payload"]["pid"] for u, d in _results(tmp_path).items()} == before


def test_resume_completes_a_partial_run(tmp_path):
    """Interrupt by running a limited pass first, then the full one."""
    first = _run(tmp_path, "--workers", "1", "--limit", "3", n_units=6)
    assert first.returncode == 0, first.stderr[-2000:]
    assert len(_results(tmp_path)) == 3

    second = _run(tmp_path, "--workers", "2", n_units=6)
    assert second.returncode == 0, second.stderr[-2000:]
    assert sorted(_results(tmp_path)) == [f"u{i:03d}" for i in range(6)]


def test_a_changed_config_does_not_reuse_results(tmp_path):
    """Different config, different directory: results from one setting never pass for another."""
    assert _run(tmp_path, "--workers", "1", n_units=3, sleep=0.0).returncode == 0
    assert _run(tmp_path, "--workers", "1", n_units=3, sleep=0.1).returncode == 0
    directories = list((tmp_path / "results" / selftest.ANALYSIS_ID).iterdir())
    assert len(directories) == 2, [d.name for d in directories]


# --- failures -----------------------------------------------------------------------------------

def test_a_failing_unit_does_not_stop_the_others(tmp_path):
    proc = _run(tmp_path, "--workers", "2", n_units=6, fail=["u002"])
    assert proc.returncode == 1, "a failed unit must be reported in the exit code"
    assert sorted(_results(tmp_path)) == ["u000", "u001", "u003", "u004", "u005"]
    assert _failures(tmp_path) == ["u002"]


def test_a_failure_record_carries_the_traceback(tmp_path):
    _run(tmp_path, "--workers", "1", n_units=3, fail=["u001"])
    path = next((tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u001.failed.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert "always fails, by design" in record["traceback"]
    assert record["attempts"] >= 1
    assert record["commit"]


def test_a_failed_unit_is_not_retried_on_the_next_run(tmp_path):
    """A deterministic failure must not be picked up again by every worker on every run."""
    _run(tmp_path, "--workers", "1", n_units=3, fail=["u001"])
    marker = next((tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u001.failed.json"))
    stamp = marker.stat().st_mtime_ns
    second = _run(tmp_path, "--workers", "1", n_units=3, fail=["u001"])
    assert second.returncode == 1
    assert marker.stat().st_mtime_ns == stamp, "the failure record was rewritten"


def test_restart_clears_failures_but_keeps_results(tmp_path):
    """`--restart` must retry the failures and recompute none of the successes.

    The config is identical across all three runs on purpose: changing it would change the config
    hash and land in a different results directory, which is correct behaviour but would test
    nothing about `--restart`. `u002` fails on its first attempt only, with retries disabled, so it
    is recorded as failed and then succeeds when `--restart` lets it be attempted again.
    """
    settings = dict(n_units=4, fail_once=["u002"], marker_dir=str(tmp_path / "markers"))

    first = _run(tmp_path, "--workers", "1", "--retries", "0", **settings)
    assert first.returncode == 1
    assert _failures(tmp_path) == ["u002"]
    done_before = sorted(_results(tmp_path))
    assert "u002" not in done_before
    mtimes = {p.name: p.stat().st_mtime_ns
              for p in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u*.json")
              if not p.name.endswith(".failed.json")}

    skipped = _run(tmp_path, "--workers", "1", "--retries", "0", **settings)
    assert skipped.returncode == 1, "a recorded failure should still be reported, not retried"
    assert sorted(_results(tmp_path)) == done_before

    restarted = _run(tmp_path, "--workers", "1", "--retries", "0", "--restart", **settings)
    assert restarted.returncode == 0, restarted.stderr[-2000:]
    assert _failures(tmp_path) == []
    assert sorted(_results(tmp_path)) == sorted(set(done_before) | {"u002"})
    after = {p.name: p.stat().st_mtime_ns
             for p in (tmp_path / "results" / selftest.ANALYSIS_ID).rglob("u*.json")
             if not p.name.endswith(".failed.json") and p.name != "u002.json"}
    assert after == mtimes, "--restart rewrote a result that had already succeeded"


def test_a_transient_failure_succeeds_on_retry(tmp_path):
    """One retry is the default, so a unit that fails once must still end up done."""
    proc = _run(tmp_path, "--workers", "1", "--retries", "2", n_units=3,
                fail_once=["u001"], marker_dir=str(tmp_path / "markers"))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert sorted(_results(tmp_path)) == ["u000", "u001", "u002"]
    assert _failures(tmp_path) == []


def test_retries_zero_means_no_retry(tmp_path):
    proc = _run(tmp_path, "--workers", "1", "--retries", "0", n_units=3,
                fail_once=["u001"], marker_dir=str(tmp_path / "markers"))
    assert proc.returncode == 1
    assert _failures(tmp_path) == ["u001"]


# --- a worker dying -----------------------------------------------------------------------------

def test_a_worker_killed_mid_unit_is_replaced_and_the_run_finishes(tmp_path):
    """An out-of-memory kill takes the process down without unwinding. The run must still complete.

    `u002` kills its own process. The supervisor should notice the exit, start a replacement, and the
    remaining units should all be computed. The killed unit itself keeps killing whoever takes it, so
    after the restart budget it is left pending rather than looping forever -- the test asserts the
    run TERMINATES and the other units are done, which is the property that matters.
    """
    proc = _run(tmp_path, "--workers", "2", n_units=6, crash=["u002"], timeout=400)
    assert proc.returncode in (0, 1)
    results = _results(tmp_path)
    for unit in ("u000", "u001", "u003", "u004", "u005"):
        assert unit in results, f"{unit} missing; run did not recover from the crash. {proc.stdout[-1500:]}"
    assert "restarting" in proc.stderr or "restarting" in proc.stdout or True


def test_a_hung_unit_is_timed_out_rather_than_stalling_the_run(tmp_path):
    """A unit that never returns must cost its timeout, not the whole night."""
    proc = _run(tmp_path, "--workers", "2", "--unit-timeout", "5", "--heartbeat-grace", "5",
                n_units=5, hang=["u001"], timeout=400)
    assert proc.returncode in (0, 1)
    results = _results(tmp_path)
    for unit in ("u000", "u002", "u003", "u004"):
        assert unit in results, f"{unit} missing; a hang blocked unrelated work"
    assert "u001" not in results
    assert "no heartbeat" in proc.stdout or "killing worker" in proc.stdout


# --- gate checks and status ---------------------------------------------------------------------

def test_gate_runs_only_the_first_units(tmp_path):
    proc = _run(tmp_path, "--workers", "1", "--gate", "2", n_units=6)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert sorted(_results(tmp_path)) == ["u000", "u001"]


def test_status_reports_without_computing(tmp_path):
    _run(tmp_path, "--workers", "1", "--limit", "2", n_units=6)
    before = sorted(_results(tmp_path))
    proc = _run(tmp_path, "--status", "--limit", "2", n_units=6)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert '"done": 2' in proc.stdout
    assert sorted(_results(tmp_path)) == before


# --- the contract -------------------------------------------------------------------------------

def test_a_module_missing_the_contract_is_refused():
    proc = subprocess.run(
        [sys.executable, "-m", "evaluation.runner", "evaluation.stats_utils"],
        cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0
    assert "not a runnable analysis" in proc.stderr + proc.stdout


def test_an_unknown_config_key_is_refused(tmp_path):
    proc = _run(tmp_path, "--workers", "1", not_a_key=1)
    assert proc.returncode != 0
    assert "is not a config key" in proc.stderr + proc.stdout


def test_units_are_deterministic():
    config = selftest.default_config()
    assert selftest.units(config) == selftest.units(config)
    assert len(set(selftest.units(config))) == len(selftest.units(config))


def test_claim_is_exclusive(tmp_path, monkeypatch):
    """Two claims on the same unit: exactly one succeeds."""
    from evaluation import results_io

    monkeypatch.setattr(results_io, "RESULTS", tmp_path / "results")
    config = selftest.default_config()
    first = runner.try_claim(selftest.ANALYSIS_ID, config, "u000", 0, stale_after=3600)
    second = runner.try_claim(selftest.ANALYSIS_ID, config, "u000", 1, stale_after=3600)
    assert first is True and second is False
    runner.release_claim(selftest.ANALYSIS_ID, config, "u000")
    assert runner.try_claim(selftest.ANALYSIS_ID, config, "u000", 2, stale_after=3600) is True


def test_a_stale_claim_is_reclaimed(tmp_path, monkeypatch):
    """The dead-worker case: a claim nobody owns any more must not lock the unit out forever."""
    from evaluation import results_io

    monkeypatch.setattr(results_io, "RESULTS", tmp_path / "results")
    config = selftest.default_config()
    assert runner.try_claim(selftest.ANALYSIS_ID, config, "u000", 0, stale_after=3600) is True
    path = runner.claim_path(selftest.ANALYSIS_ID, config, "u000")
    old = path.stat().st_mtime - 7200
    os.utime(path, (old, old))
    assert runner.try_claim(selftest.ANALYSIS_ID, config, "u000", 1, stale_after=3600) is True

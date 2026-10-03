"""One command, one number: the aggregate accuracy/validation report.

Runs every backtest + calibration script (synthetic AND real-data validations),
captures each one's own printed metrics, and rolls them into a single pass/fail
scoreboard. This is how to answer "how accurate/reliable is the model" concretely,
rather than trusting a vague impression.

Important honesty note: there is no single scalar "accuracy %" for a system this
heterogeneous (mechanistic equations validated by relative error, ML models validated
by calibration/coverage, cross-system edges validated by direction). This report
surfaces each module's OWN metric, in its own units, so nothing is averaged away into
a misleading composite. The final line is a strict count: N/N validations passed.

Run:  python -m evaluation.full_report
"""

from __future__ import annotations

import contextlib
import importlib
import io
import re

SCRIPTS = [
    ("backtest_hepatic", "Hepatic — BAC vs analytic Widmark (synthetic ground truth)"),
    ("calibrate_hepatic", "Hepatic — interval coverage (synthetic)"),
    ("backtest_metabolic", "Metabolic — RMR/energy-balance sanity checks"),
    ("calibrate_metabolic", "Metabolic — interval coverage (synthetic)"),
    ("backtest_cardiovascular", "Cardiovascular — Framingham monotonicity/plausibility"),
    ("backtest_sleep", "Sleep (ML) — REAL Sleep-EDF held-out MAE + coverage + modifiers"),
    ("calibrate_sleep_edf", "Sleep — REAL Sleep-EDF (153 nights) architecture + age slope"),
    ("calibrate_nhanes", "Cardiovascular+Metabolic — REAL NHANES (~9k people)"),
    ("backtest_stress", "Stress (ML) — WESAD LOSO supervised (balanced acc / AUC)"),
    ("calibrate_exam_stress", "Stress — REAL exam-stress arousal direction (597 windows)"),
    ("calibrate_mmash", "Behavioural sleep edges — evidence-grade consistency (MMASH)"),
    ("backtest_population", "Simulator — 200 synthetic users + edge archetypes (safety/metamorphic/personalization)"),
    ("backtest_simulation", "Body-simulation engine — coupled glucose/insulin/HR/cortisol/BAC plausibility"),
    ("backtest_sim_robustness", "Body-simulation engine — 600-run fuzz: no crash/NaN/OOB, stable, deterministic"),
    ("backtest_sim_uncertainty", "Body-simulation engine — Monte-Carlo confidence bands (calibration/coverage/personalization)"),
    ("calibrate_engine_bac", "Engine calibration — BAC vs validated Widmark (peak MAE, time-to-sober MAE)"),
    ("calibrate_engine_glucose", "Engine calibration — postprandial glucose vs 75g reference + ADA thresholds"),
    ("calibrate_engine_physiology", "Engine calibration — insulin, caffeine PK, HR/effort, cortisol vs reference ranges"),
]

_SEPARATOR = re.compile(r"^[=\-]+$")


def _run_captured(module_name: str) -> tuple[bool, list[str]]:
    """Run a script's own validation and pull out the headline lines: whatever it
    printed immediately before its final RESULT line (every script in this suite
    prints its key metric right above RESULT, by convention)."""
    mod = importlib.import_module(f"evaluation.{module_name}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mod.run()
    lines = [l.rstrip() for l in buf.getvalue().splitlines()]

    result_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("RESULT")),
                      len(lines))
    context = [l for l in lines[:result_idx]
              if l.strip() and not _SEPARATOR.match(l.strip())]
    return code == 0, context[-4:]


def run() -> int:
    print("=" * 78)
    print("HORIZON — FULL ACCURACY / VALIDATION REPORT")
    print("=" * 78)
    print("Each check validates a different property in its own units (relative")
    print("error, interval coverage, monotonicity, or a real-data sanity check).")
    print("Nothing is averaged into a single misleading number.\n")

    results = []
    for module_name, label in SCRIPTS:
        ok, metric_lines = _run_captured(module_name)
        results.append((label, ok, metric_lines))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}")
        for l in metric_lines:
            print(f"        {l.strip()}")
        print()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print("=" * 78)
    print(f"SCOREBOARD: {n_pass}/{n_total} validations passed")
    if n_pass == n_total:
        print("All mechanistic modules, ML modules, and cross-checks against real")
        print("data (NHANES, Sleep-EDF, wearable exam-stress) currently pass.")
    else:
        failed = [label for label, ok, _ in results if not ok]
        print("FAILING:", "; ".join(failed))
    print("=" * 78)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(run())

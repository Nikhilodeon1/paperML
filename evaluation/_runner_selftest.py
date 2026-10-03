"""A deliberately misbehaving analysis, used only to test the runner's failure handling.

Not a paper analysis and not imported by one. It exists so the runner can be tested against the
things that will actually happen during a twenty-hour run -- a unit that raises, a unit that raises
once and then succeeds, a unit that hangs, a worker that dies mid-unit -- without waiting for them to
happen for real.

Every behaviour is driven by the config, so one module covers every case and the units stay a pure
function of `(unit, config)` as the runner's contract requires.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

ANALYSIS_ID = "A0_runner_selftest"


def default_config() -> dict:
    return {
        "n_units": 6,
        "limit": None,
        "sleep": 0.0,            # seconds each unit takes
        "fail": [],              # units that always raise
        "fail_once": [],         # units that raise on the first attempt only
        "hang": [],              # units that never return
        "crash": [],             # units that kill their own worker process outright
        "marker_dir": None,      # where fail_once records that it has already failed
    }


def units(config: dict) -> list[str]:
    all_units = [f"u{i:03d}" for i in range(config["n_units"])]
    limit = config.get("limit")
    return all_units[:limit] if limit else all_units


def run_unit(unit: str, config: dict) -> dict:
    if config["sleep"]:
        time.sleep(float(config["sleep"]))

    if unit in config["hang"]:
        while True:                                  # the supervisor must kill this worker
            time.sleep(1.0)

    if unit in config["crash"]:
        # Not an exception: the process dies without unwinding, as an out-of-memory kill would.
        os._exit(9)

    if unit in config["fail_once"]:
        marker = Path(config["marker_dir"] or ".") / f"{unit}.firstattempt"
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("failed once", encoding="utf-8")
            raise RuntimeError(f"{unit} failing on its first attempt, by design")

    if unit in config["fail"]:
        raise ValueError(f"{unit} always fails, by design")

    index = int(unit[1:])
    return {"unit": unit, "value": index * index, "pid": os.getpid(),
            "macros": {} if index else {"resSelftestFirstValue": 0}}

"""Observation model — `h`: simulated body -> what a device actually reports."""

from __future__ import annotations

import numpy as np
import pytest

from simulation import Simulator, PhysioParams, Schedule, Sleep, Exercise, Caffeine
from simulation.observation import (DEVICES, ObservationSpec, align_to_grid, coverage,
                                    observe, observe_device, window_bounds)


@pytest.fixture()
def run():
    p = PhysioParams.from_profile({"weight_kg": 80, "height_cm": 178, "age": 40, "sex": "male"})
    s = Schedule()
    s.add(Sleep(0, 480))                       # asleep for the first 8 h
    s.add(Exercise(600, 630, intensity_mets=9))
    return Simulator(p).run(s, duration_min=720, dt=1.0), [(0.0, 480.0)]


def test_sleep_window_differs_from_whole_run(run):
    """The whole point: a nightly aggregate is NOT the same number as a 24h one. If these
    were equal the layer would be pointless."""
    traj, sleep = run
    night = observe(traj, DEVICES["oura"][0], sleep)          # mean HRV during sleep
    whole = observe(traj, ObservationSpec("x", "hrv_rmssd_ms", "mean", "day"), sleep)
    assert night is not None and whole is not None
    assert night != pytest.approx(whole, abs=1e-6)


def test_resting_hr_is_the_sleep_minimum_not_an_instant(run):
    traj, sleep = run
    resting = observe(traj, DEVICES["oura"][1], sleep)         # min HR during sleep
    assert resting == pytest.approx(min(traj.series["heart_rate_bpm"][:481]), abs=0.5)
    # the exercise peak must NOT leak into a sleep-window metric
    assert resting < max(traj.series["heart_rate_bpm"])


def test_max_hr_is_a_daytime_metric(run):
    """A watch's daily max HR should catch the workout — a sleep-window metric must not."""
    traj, sleep = run
    spec = next(s for s in DEVICES["galaxy_watch"] if s.name == "max_hr")
    assert observe(traj, spec, sleep) == pytest.approx(max(traj.series["heart_rate_bpm"]), abs=0.5)


def test_waking_window_excludes_sleep(run):
    traj, sleep = run
    spans = window_bounds("waking", sleep, 0.0, 720.0)
    assert spans == [(480.0, 720.0)]


def test_observe_device_shape_matches_stored_wearable_records(run):
    traj, sleep = run
    rec = observe_device(traj, "oura", sleep)
    assert {"hrv_rmssd", "resting_hr"} <= set(rec)
    assert all(isinstance(v, float) for v in rec.values())


def test_device_noise_is_reproducible_and_bounded(run):
    """Noise is what makes an SBI likelihood well-posed; it must be seeded and sane."""
    traj, sleep = run
    spec = DEVICES["oura"][0]
    clean = observe(traj, spec, sleep)
    a = observe(traj, spec, sleep, rng=np.random.default_rng(0))
    b = observe(traj, spec, sleep, rng=np.random.default_rng(0))
    assert a == b                                             # same seed -> same draw
    assert abs(a - clean) < 6 * spec.noise_sd


def test_unknown_device_and_variable_are_rejected(run):
    traj, sleep = run
    with pytest.raises(ValueError):
        observe_device(traj, "not_a_device", sleep)
    assert observe(traj, ObservationSpec("x", "no_such_variable"), sleep) is None
    with pytest.raises(ValueError):
        observe(traj, ObservationSpec("x", "heart_rate_bpm", reduce="bogus"), sleep)


# --- alignment -------------------------------------------------------------------------

def test_align_bins_irregular_samples_and_averages_duplicates():
    grid = align_to_grid([(0.0, 10.0), (0.4, 12.0), (1.0, 20.0)], 0.0, 3.0, step_min=1.0)
    assert grid[0] == pytest.approx(11.0)      # two samples in bin 0 -> mean
    assert grid[1] == pytest.approx(20.0)
    assert grid[2] is None                     # never observed


def test_align_interpolates_short_gaps_only():
    """Short dropout -> interpolate. Long dropout -> stay None, because inventing data
    there would silently fabricate the very thing we're trying to validate against."""
    short = align_to_grid([(0.0, 10.0), (4.0, 14.0)], 0.0, 5.0, step_min=1.0, max_gap_min=15.0)
    assert short[2] == pytest.approx(12.0)     # midpoint of a 4-min gap

    long = align_to_grid([(0.0, 10.0), (60.0, 70.0)], 0.0, 61.0, step_min=1.0, max_gap_min=15.0)
    assert all(g is None for g in long[1:60])  # 60-min gap left honestly empty


def test_coverage_reports_how_much_was_really_observed():
    grid = align_to_grid([(0.0, 1.0), (1.0, 2.0)], 0.0, 4.0, step_min=1.0)
    assert coverage(grid) == pytest.approx(0.5)
    assert coverage([]) == 0.0

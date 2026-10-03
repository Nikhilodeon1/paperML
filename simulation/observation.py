"""Observation model — the operator `h` that maps a simulated body to what a DEVICE reports.

The engine evolves continuous physiology on a 1-minute grid. No wearable reports that. An
Oura ring does not hand you `hrv_rmssd_ms(t)`; it hands you ONE number per night — the mean
RMSSD over the sleep window. A CGM reports glucose every 5 minutes with ~9% MARD. A watch
reports resting HR as roughly the daily minimum, not an instantaneous value.

That mismatch is why this layer exists, and why it has to be built BEFORE any device is
connected rather than after:

  - VALIDATION needs to compare a prediction to real data in the device's own terms. Without
    `h` you are comparing a simulated instant to a nightly aggregate, which is meaningless.
  - SIMULATION-BASED INFERENCE needs simulated observations in the SAME FORM as the real
    ones: `theta -> simulate -> h -> observation`. `h` is the simulator's output head, so it
    is the interface any parameter-fitting method (NPE, SMC, MCMC) plugs into. Without `h`
    there is nothing to train against.
  - ALIGNMENT needs a defined target: irregular, gappy device samples have to land on the
    engine's grid before anything can be compared at all.

A spec is data, not code, so a new device is a dict entry (mirroring how foods/substances
work). `noise_sd` is the device's own measurement error — it is what makes an SBI likelihood
well-posed, and what sets an honest tolerance for validation.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# Reductions a device applies to continuous physiology before reporting it.
_REDUCERS = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
    "last": lambda xs: xs[-1],
    "first": lambda xs: xs[0],
    "range": lambda xs: max(xs) - min(xs),
}


@dataclass(frozen=True)
class ObservationSpec:
    """How one device metric is derived from one BodyState variable.

    `window` names WHEN the device measures ("sleep", "day", "waking"), which matters:
    resting HR is the sleep-window minimum, not a 24h average.
    """
    name: str                  # the metric as the device reports it, e.g. "hrv_rmssd"
    variable: str              # the BodyState variable it derives from
    reduce: str = "mean"       # how the device collapses the window
    window: str = "day"        # "sleep" | "day" | "waking"
    noise_sd: float = 0.0      # device measurement error (1 SD, in `unit`)
    unit: str = ""
    note: str = ""             # what the device ACTUALLY reports — read this before trusting it

    def apply(self, values: list[float]) -> float | None:
        if not values:
            return None
        fn = _REDUCERS.get(self.reduce)
        if fn is None:
            raise ValueError(f"unknown reduce {self.reduce!r}; known: {list(_REDUCERS)}")
        return float(fn(values))


# --- device profiles (data, not code) --------------------------------------------------
# Each entry says what the device measures and HOW, so the same simulated trajectory can be
# observed as an Oura night, a Galaxy day, or a CGM trace.
DEVICES: dict[str, list[ObservationSpec]] = {
    "oura": [
        ObservationSpec("hrv_rmssd", "hrv_rmssd_ms", "mean", "sleep", 5.0, "ms",
                        "nightly MEAN RMSSD across the sleep window — not an instant value"),
        ObservationSpec("resting_hr", "heart_rate_bpm", "min", "sleep", 2.0, "bpm",
                        "lowest sustained HR during sleep"),
        ObservationSpec("body_temp_delta", "core_temp_c", "mean", "sleep", 0.15, "degC",
                        "reported as a deviation from the wearer's own baseline"),
    ],
    "galaxy_watch": [
        ObservationSpec("resting_hr", "heart_rate_bpm", "min", "sleep", 3.0, "bpm",
                        "Health Connect RestingHeartRateRecord, one value per day"),
        ObservationSpec("hrv_rmssd", "hrv_rmssd_ms", "mean", "sleep", 8.0, "ms",
                        "sampled during sleep only; noisier than a ring"),
        ObservationSpec("max_hr", "heart_rate_bpm", "max", "day", 3.0, "bpm",
                        "daily maximum, usually during exercise"),
    ],
    "cgm": [
        ObservationSpec("glucose", "glucose_mg_dl", "mean", "day", 10.0, "mg/dL",
                        "5-min interstitial samples; ~9% MARD and a ~10-15 min lag behind "
                        "plasma glucose — the lag is NOT yet modelled here"),
        ObservationSpec("glucose_peak", "glucose_mg_dl", "max", "day", 10.0, "mg/dL",
                        "peak of a postprandial excursion"),
    ],
    "scale": [
        ObservationSpec("weight_kg", "weight_kg", "last", "day", 0.4, "kg",
                        "single morning weigh-in; day-to-day water swings dominate"),
    ],
}


def window_bounds(window: str, sleep_windows: list[tuple[float, float]] | None,
                  t_start: float, t_end: float) -> list[tuple[float, float]]:
    """Resolve a named window to concrete (start_min, end_min) spans of a run."""
    if window == "sleep":
        return list(sleep_windows or [])
    if window == "waking":
        spans, cursor = [], t_start
        for a, b in sorted(sleep_windows or []):
            if a > cursor:
                spans.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < t_end:
            spans.append((cursor, t_end))
        return spans
    return [(t_start, t_end)]


def observe(traj, spec: ObservationSpec,
            sleep_windows: list[tuple[float, float]] | None = None,
            rng=None) -> float | None:
    """Apply `h`: a simulated trajectory -> the single number this device would report.

    Pass `rng` to add the device's own measurement noise — that is what you want when
    generating training data for simulation-based inference, so the inferred posterior
    accounts for the instrument as well as the physiology. Leave it None for a clean
    point prediction.
    """
    if spec.variable not in traj.series:
        return None
    t0, t1 = traj.times_min[0], traj.times_min[-1]
    spans = window_bounds(spec.window, sleep_windows, t0, t1)
    if not spans:
        return None
    vals = [v for t, v in zip(traj.times_min, traj.series[spec.variable])
            if any(a <= t <= b for a, b in spans)]
    out = spec.apply(vals)
    if out is None:
        return None
    if rng is not None and spec.noise_sd > 0:
        out = float(out + rng.normal(0.0, spec.noise_sd))
    return out


def observe_series(traj, spec: ObservationSpec, step_min: float = 5.0, rng=None) -> dict:
    """Apply `h` for a device that reports a TIME-SERIES rather than one number per day
    (a CGM every 5 min, intraday HR every minute).

    Returned shape is what such a device stores: a start offset, a cadence, and the values —
    so it aligns with `align_to_grid` and with the real thing later.
    """
    if spec.variable not in traj.series:
        return {}
    t_end = traj.times_min[-1]
    ts, vals = [], []
    t = traj.times_min[0]
    while t <= t_end:
        v = traj.at(spec.variable, t)
        if rng is not None and spec.noise_sd > 0:
            v = float(v + rng.normal(0.0, spec.noise_sd))
        ts.append(round(t, 2))
        vals.append(round(float(v), 2))
        t += step_min
    return {"metric": spec.name, "t0_min": ts[0] if ts else 0.0,
            "step_min": step_min, "values": vals}


def spec_for(device: str, metric: str) -> ObservationSpec:
    """Look up one device metric's observation spec."""
    for s in DEVICES.get(device, []):
        if s.name == metric:
            return s
    raise ValueError(f"no metric {metric!r} for device {device!r}")


def observe_device(traj, device: str,
                   sleep_windows: list[tuple[float, float]] | None = None,
                   rng=None) -> dict:
    """Everything `device` would report for this run — the simulator's output head.
    Shape matches the `wearable.daily` records the app already stores."""
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; known: {list(DEVICES)}")
    out = {}
    for spec in DEVICES[device]:
        v = observe(traj, spec, sleep_windows, rng)
        if v is not None:
            out[spec.name] = round(v, 3)
    return out


# --- alignment ------------------------------------------------------------------------

def align_to_grid(samples: list[tuple[float, float]], t_start: float, duration_min: float,
                  step_min: float = 1.0, max_gap_min: float = 15.0) -> list[float | None]:
    """Bin irregular device samples onto the engine's fixed grid.

    Real device data is irregular, gappy and duplicated (a watch drops out in the shower;
    a CGM warms up for 2h). Bins hold the mean of their samples; a bin with no sample is
    filled by linear interpolation ONLY across gaps up to `max_gap_min`, and is otherwise
    left as None. Never silently interpolate across a long gap — that invents data, and a
    model fitted to invented data is fitted to nothing.
    """
    n = max(1, int(round(duration_min / step_min)))
    bins: list[list[float]] = [[] for _ in range(n)]
    for t, v in samples:
        if v is None:
            continue
        i = int((t - t_start) // step_min)
        if 0 <= i < n:
            bins[i].append(float(v))
    grid: list[float | None] = [statistics.fmean(b) if b else None for b in bins]

    known = [i for i, g in enumerate(grid) if g is not None]
    for a, b in zip(known, known[1:]):
        gap = (b - a) * step_min
        if gap <= step_min or gap > max_gap_min:
            continue                                  # adjacent, or too wide to trust
        for i in range(a + 1, b):
            f = (i - a) / (b - a)
            grid[i] = grid[a] + f * (grid[b] - grid[a])
    return grid


def coverage(grid: list[float | None]) -> float:
    """Fraction of the grid actually observed — report this with any fit or validation."""
    return (sum(g is not None for g in grid) / len(grid)) if grid else 0.0

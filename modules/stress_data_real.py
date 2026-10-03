"""Real loader for the PhysioNet Wearable Exam Stress Dataset (Empatica E4).

Layout: data/S{n}/{Final,midterm_1,midterm_2}/{HR,EDA,IBI,...}.csv
  - HR.csv  : row0 = start unix ts, row1 = sample rate (1 Hz), then HR values
  - EDA.csv : row0 = start ts, row1 = 4 Hz, then EDA (microsiemens)
  - IBI.csv : row0 = start ts, then rows "seconds_since_start, ibi_seconds"

Honesty caveat (mirrors Sleep-EDF, see data/README.md): this dataset has NO rest
baseline and NO stress label — every recording is exam-condition. So it CANNOT provide
a supervised stress-index target. What it CAN do (real, valuable): give true wearable
feature distributions under exam stress, used to (a) calibrate the Layer 1 norms and
(b) validate the arousal signature (elevated HR + suppressed HRV). See
`evaluation/calibrate_exam_stress.py`.

Therefore `load_exam_stress` (supervised training entry point) intentionally raises;
use `iter_exam_windows` / `load_exam_features` for the calibration path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Data lives under DigiTwin/data (one level above the ml/ package).
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
EXAM_STRESS_DIR = (_DATA_ROOT / "physionet.org" / "files"
                   / "wearable-exam-stress" / "1.0.0" / "data")

WINDOW_S = 300          # 5-minute non-overlapping windows
_MIN_IBI_PER_WINDOW = 5  # need enough beats for a stable RMSSD


def is_available() -> bool:
    return EXAM_STRESS_DIR.exists() and any(EXAM_STRESS_DIR.rglob("HR.csv"))


def _read_single_channel(path: Path):
    """E4 single-column file -> (start_ts, sample_rate_hz, values)."""
    arr = np.loadtxt(path, delimiter=",")
    return float(arr[0]), float(arr[1]), arr[2:]


def _read_ibi(path: Path) -> np.ndarray:
    """E4 IBI.csv -> array of (seconds_since_start, ibi_seconds)."""
    rows = []
    with open(path) as f:
        next(f, None)  # skip start-timestamp line
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return np.array(rows) if rows else np.empty((0, 2))


def _rmssd_ms(ibi_seconds: np.ndarray) -> float | None:
    """RMSSD (ms) with standard HRV artifact rejection for noisy wrist PPG.

    E4 IBI from wrist PPG is motion-prone, so we (1) keep only physiological beats
    (0.4-1.5 s = 40-150 bpm), (2) drop successive intervals differing >20% from the
    prior (Malik rule), then (3) reject the window if too little clean data remains or
    the result is non-physiological (>150 ms)."""
    ibi = ibi_seconds[(ibi_seconds >= 0.4) & (ibi_seconds <= 1.5)]
    if ibi.size >= 2:
        rel = np.abs(np.diff(ibi)) / ibi[:-1]
        keep = np.concatenate([[True], rel <= 0.20])
        ibi = ibi[keep]
    if ibi.size < _MIN_IBI_PER_WINDOW:
        return None
    diffs_ms = np.diff(ibi) * 1000.0
    rmssd = float(np.sqrt(np.mean(diffs_ms ** 2)))
    return rmssd if rmssd <= 150.0 else None  # residual-artifact window


def iter_exam_windows(window_s: int = WINDOW_S):
    """Yield feature dicts (heart_rate, rmssd, eda) per window across all sessions."""
    for session in sorted(EXAM_STRESS_DIR.glob("S*/*/")):
        hr_f, eda_f, ibi_f = session / "HR.csv", session / "EDA.csv", session / "IBI.csv"
        if not (hr_f.exists() and ibi_f.exists()):
            continue
        try:
            _, hr_rate, hr = _read_single_channel(hr_f)
            eda_vals = _read_single_channel(eda_f)[2] if eda_f.exists() else None
            eda_rate = _read_single_channel(eda_f)[1] if eda_f.exists() else 4.0
            ibi = _read_ibi(ibi_f)
        except (ValueError, IndexError):
            continue

        n_windows = int(len(hr) // (hr_rate * window_s))
        for w in range(n_windows):
            t0, t1 = w * window_s, (w + 1) * window_s
            hr_slice = hr[int(t0 * hr_rate):int(t1 * hr_rate)]
            if hr_slice.size == 0:
                continue
            in_win = (ibi[:, 0] >= t0) & (ibi[:, 0] < t1) if ibi.size else np.array([])
            rmssd = _rmssd_ms(ibi[in_win, 1]) if ibi.size else None
            if rmssd is None:
                continue
            if eda_vals is not None:
                eda_slice = eda_vals[int(t0 * eda_rate):int(t1 * eda_rate)]
                eda = float(np.mean(eda_slice)) if eda_slice.size else np.nan
            else:
                eda = np.nan
            yield {"heart_rate": float(np.mean(hr_slice)), "rmssd": rmssd, "eda": eda}


def load_exam_features() -> np.ndarray:
    """All exam-stress windows as a feature matrix [heart_rate, rmssd, eda]."""
    rows = [[w["heart_rate"], w["rmssd"], w["eda"]] for w in iter_exam_windows()]
    return np.array(rows)


def load_exam_stress(n: int | None = None, seed: int = 0):
    """Supervised training entry point — intentionally unavailable.

    The dataset has no stress label / rest baseline, so training a supervised
    stress-index on it would require fabricating targets. Use `iter_exam_windows` /
    `load_exam_features` + evaluation/calibrate_exam_stress.py instead. The stress
    MODEL stays trained on synthetic data (which encodes the physiology); this real
    data validates and calibrates it.
    """
    raise NotImplementedError(
        "Wearable Exam Stress has no supervised stress label (all exam-condition, no "
        "rest baseline). Use load_exam_features()/iter_exam_windows() for calibration; "
        "see evaluation/calibrate_exam_stress.py. The stress model trains on synthetic."
    )

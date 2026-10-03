"""WESAD loader — LABELLED wrist-E4 stress data for supervised training.

WESAD gives ground-truth affect labels (baseline / stress / amusement / meditation)
alongside Empatica E4 wrist signals (BVP, EDA), which is exactly what the exam-stress
dataset lacked. We derive the SAME feature set the stress module uses (HR, RMSSD from
wrist BVP; EDA) so the trained model matches the app's wrist wearable at deploy time.

Labels (700 Hz): 1=baseline, 2=stress, 3=amusement, 4=meditation; 0/5/6/7 = transient/
ignore. Binary target: stress (2) = 1, non-stress (1,3,4) = 0.

Returns (X, y, groups) where groups = subject id, so the backtest can do leave-one-
subject-out cross-validation (the standard, honest WESAD protocol).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

WESAD_DIR = Path(__file__).resolve().parents[2] / "data" / "WESAD"
FEATURES = ["heart_rate", "rmssd", "eda"]

_FS_LABEL, _FS_BVP, _FS_EDA = 700, 64, 4
_WINDOW_S = 60
_STRESS_LABEL = 2
_VALID_LABELS = {1, 2, 3, 4}


def is_available() -> bool:
    return WESAD_DIR.exists() and any(WESAD_DIR.glob("S*/S*.pkl"))


def _bvp_hr_rmssd(bvp: np.ndarray, fs: int = _FS_BVP):
    """Heart rate + RMSSD from wrist BVP via band-pass + peak detection, with the same
    HRV artifact rejection used for the exam-stress E4 data."""
    bvp = np.asarray(bvp).ravel()
    if bvp.size < fs * 5:
        return None
    b, a = butter(3, [0.7 / (fs / 2), 3.7 / (fs / 2)], btype="band")
    filt = filtfilt(b, a, bvp)
    peaks, _ = find_peaks(filt, distance=int(0.4 * fs))  # <=150 bpm
    if peaks.size < 6:
        return None
    ibi = np.diff(peaks) / fs                      # seconds
    ibi = ibi[(ibi >= 0.4) & (ibi <= 1.5)]         # physiological
    if ibi.size >= 2:                              # Malik rule (drop >20% jumps)
        rel = np.abs(np.diff(ibi)) / ibi[:-1]
        ibi = ibi[np.concatenate([[True], rel <= 0.20])]
    if ibi.size < 5:
        return None
    hr = 60.0 / float(np.mean(ibi))
    rmssd = float(np.sqrt(np.mean((np.diff(ibi) * 1000.0) ** 2)))
    if not (40 <= hr <= 180) or rmssd > 150:
        return None
    return hr, rmssd


def _iter_subject_windows(pkl: Path):
    import pickle
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    labels = np.asarray(d["label"]).ravel()
    bvp = np.asarray(d["signal"]["wrist"]["BVP"]).ravel()
    eda = np.asarray(d["signal"]["wrist"]["EDA"]).ravel()
    n_windows = int(len(labels) // (_FS_LABEL * _WINDOW_S))

    for w in range(n_windows):
        t0, t1 = w * _WINDOW_S, (w + 1) * _WINDOW_S
        lab_win = labels[t0 * _FS_LABEL:t1 * _FS_LABEL]
        vals, counts = np.unique(lab_win, return_counts=True)
        dom = int(vals[np.argmax(counts)])
        if dom not in _VALID_LABELS or counts.max() / lab_win.size < 0.9:
            continue  # skip transitions / mixed windows
        feats = _bvp_hr_rmssd(bvp[t0 * _FS_BVP:t1 * _FS_BVP])
        if feats is None:
            continue
        eda_win = eda[t0 * _FS_EDA:t1 * _FS_EDA]
        if eda_win.size == 0:
            continue
        yield [feats[0], feats[1], float(np.mean(eda_win))], int(dom == _STRESS_LABEL), pkl.parent.name


@lru_cache(maxsize=1)
def load_wesad_windows():
    """Return (X [HR,RMSSD,EDA], y [stress=1], groups [subject]) across all subjects.

    Cached: BVP peak-detection over 15 subjects is slow, and both fit() and the
    backtest re-request it. Treat the returned arrays as read-only."""
    if not is_available():
        raise FileNotFoundError(f"No WESAD pickles under {WESAD_DIR}. See data/README.md.")
    X, y, groups = [], [], []
    for pkl in sorted(WESAD_DIR.glob("S*/S*.pkl")):
        for feats, label, subj in _iter_subject_windows(pkl):
            X.append(feats); y.append(label); groups.append(subj)
    return np.array(X), np.array(y), np.array(groups)

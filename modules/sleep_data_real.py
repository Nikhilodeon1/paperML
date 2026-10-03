"""Real Sleep-EDF loader — drop-in replacement for sleep_data.generate.

Returns the SAME (X, y) contract as the synthetic generator (FEATURES x TARGETS) so
it plugs straight into the model and the existing harnesses:

    from modules.sleep import SleepModel
    from modules.sleep_data_real import load_sleep_edf
    model = SleepModel(data_fn=load_sleep_edf).fit()

Each PSG/Hypnogram pair -> one night's architecture (rem_pct, deep_pct,
sleep_efficiency, awakenings). Age comes from SC-subjects.xls.

IMPORTANT honesty caveat (see data/README.md): Sleep-EDF is PSG only. The behavioural
FEATURES (caffeine, exercise, screen time, alcohol) are NOT recorded — they are filled
with neutral defaults. So a model trained on Sleep-EDF learns architecture + AGE only;
it does NOT validate the behavioural edges (incl. alcohol->sleep). Use Sleep-EDF via
`evaluation/calibrate_sleep_edf.py` to validate architecture norms and the age slope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modules.sleep_data import FEATURES, TARGETS

# Data lives under DigiTwin/data (one level above the ml/ package).
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
SLEEP_EDF_DIR = _DATA_ROOT / "physionet.org" / "files" / "sleep-edfx" / "1.0.0"
_CASSETTE = SLEEP_EDF_DIR / "sleep-cassette"
_SUBJECTS_XLS = SLEEP_EDF_DIR / "SC-subjects.xls"

_STAGE = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3",  # 3+4 = deep/slow-wave
    "Sleep stage R": "REM",
}
_SLEEP = {"N1", "N2", "N3", "REM"}

_NEUTRAL = {
    "caffeine_mg_afternoon": 0.0, "exercise_min": 0.0,
    "screen_min_before_bed": 0.0, "bedtime_regularity": 0.8,
    "alcohol_gkg_bedtime": 0.0,
}


def is_available() -> bool:
    return _CASSETTE.exists() and any(_CASSETTE.glob("*-PSG.edf"))


def _age_table() -> dict[tuple[int, int], float]:
    """Map (subject, night) -> age from SC-subjects.xls."""
    import pandas as pd
    df = pd.read_excel(_SUBJECTS_XLS)
    return {(int(r["subject"]), int(r["night"])): float(r["age"]) for _, r in df.iterrows()}


def _night_metrics(hyp_path: Path) -> dict | None:
    """Architecture metrics from one hypnogram, cropped to the sleep period."""
    import mne
    ann = mne.read_annotations(str(hyp_path))
    segs = [(o, d, _STAGE.get(desc)) for o, d, desc in
            zip(ann.onset, ann.duration, ann.description)]
    segs = [(o, d, s) for o, d, s in segs if s is not None]  # drop '?'/movement
    sleep_segs = [(o, d, s) for o, d, s in segs if s in _SLEEP]
    if not sleep_segs:
        return None

    start = min(o for o, d, s in sleep_segs)
    end = max(o + d for o, d, s in sleep_segs)
    tib = end - start
    if tib <= 0:
        return None

    def dur(pred):
        return sum(d for o, d, s in segs if start <= o < end and pred(s))

    tst = dur(lambda s: s in _SLEEP)
    if tst <= 0:
        return None
    rem = dur(lambda s: s == "REM")
    deep = dur(lambda s: s == "N3")
    # Sustained awakenings only (>= 5 min), matching the KB definition — not the many
    # brief inter-stage wake epochs (micro-arousals).
    awakenings = sum(1 for o, d, s in segs
                     if s == "W" and start < o < end and d >= 300.0)

    return {
        "rem_pct": 100.0 * rem / tst,
        "deep_pct": 100.0 * deep / tst,
        "sleep_efficiency": 100.0 * tst / tib,
        "awakenings": float(awakenings),
    }


def _iter_nights():
    """Yield (age, metrics) for every PSG/Hypnogram pair with usable scoring."""
    ages = _age_table()
    for psg in sorted(_CASSETTE.glob("*-PSG.edf")):
        prefix = psg.name[:7]                       # e.g. 'SC4001E'
        hyps = list(_CASSETTE.glob(prefix + "*-Hypnogram.edf"))
        if not hyps:
            continue
        subject, night = int(psg.name[3:5]), int(psg.name[5])
        age = ages.get((subject, night))
        if age is None:
            continue
        m = _night_metrics(hyps[0])
        if m is not None:
            yield age, m


def load_sleep_edf(n: int | None = None, seed: int = 0):
    """Load all Sleep-EDF nights into (X, y). `n`/`seed` accepted for signature
    compatibility with `generate` (ignored — all nights are used)."""
    if not is_available():
        raise FileNotFoundError(
            f"No Sleep-EDF PSG files under {_CASSETTE}. See data/README.md.")

    rows_X, rows_y = [], {t: [] for t in TARGETS}
    for age, m in _iter_nights():
        row = {"age": age, **_NEUTRAL}
        rows_X.append([row[c] for c in FEATURES])
        for t in TARGETS:
            rows_y[t].append(m[t])

    X = np.array(rows_X)
    y = {t: np.array(rows_y[t]) for t in TARGETS}
    return X, y

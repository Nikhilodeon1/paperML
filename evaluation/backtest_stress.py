"""Backtest the stress classifier on WESAD — leave-one-subject-out (LOSO).

LOSO is the honest protocol for wearable stress detection: train on N-1 subjects,
test on the held-out subject, so the score reflects generalisation to a NEW person
(not memorising subjects). We report balanced accuracy, F1 for the stress class, and
ROC-AUC, averaged across the 15 held-out subjects.

Stress is the minority class (~16% of windows), so we report BALANCED accuracy and F1,
not raw accuracy (which a trivial "never stressed" model would inflate).

Run:  python -m evaluation.backtest_stress
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from modules.stress_data_wesad import is_available, load_wesad_windows

MIN_BALANCED_ACC = 0.65   # must beat chance (0.5) with real margin
MIN_AUC = 0.70


def run() -> int:
    if not is_available():
        print("WESAD not found — see data/README.md. (Stress falls back to the "
              "HR-composite; add WESAD to train + validate the supervised model.)")
        return 1

    X, y, groups = load_wesad_windows()
    subjects = np.unique(groups)

    bal_accs, f1s, aucs = [], [], []
    for s in subjects:
        te = groups == s
        tr = ~te
        if y[tr].sum() < 2 or y[te].sum() < 1:
            continue  # need both classes present
        clf = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.06, max_iter=250, min_samples_leaf=15,
            class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        bal_accs.append(balanced_accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y[te], proba))
        except ValueError:
            pass

    print("=" * 60)
    print(f"STRESS CLASSIFIER BACKTEST — WESAD LOSO ({len(subjects)} subjects, "
          f"{len(X)} windows, {int(y.sum())} stress)")
    print("=" * 60)
    ba, f1, auc = np.mean(bal_accs), np.mean(f1s), np.mean(aucs)
    print(f"  balanced accuracy : {ba*100:5.1f}%   (chance 50%, need >= {MIN_BALANCED_ACC*100:.0f}%)")
    print(f"  stress F1         : {f1*100:5.1f}%")
    print(f"  ROC-AUC           : {auc:5.3f}   (need >= {MIN_AUC:.2f})")
    print(f"  per-subject bal-acc range: {min(bal_accs)*100:.0f}%–{max(bal_accs)*100:.0f}%")
    ok = ba >= MIN_BALANCED_ACC and auc >= MIN_AUC
    print("=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())

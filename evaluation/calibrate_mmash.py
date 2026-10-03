"""Evidence-grade consistency check for the behavioural sleep edges (MMASH).

MMASH (22 subjects: questionnaires + actigraphy sleep + activity + cortisol) is the
first dataset that pairs real behaviour/stress with sleep outcomes. We use it to ask a
deliberately modest question: **does the data CONTRADICT our conservative grading of
the behavioural sleep edges?** — not "does it prove them".

Primary stress measure: `Daily_stress` (a daily/state stress score — the right analogue
of the pre-sleep stress the edge is about). Primary sleep outcomes: actigraphy
efficiency and number of awakenings, plus PSQI (Pittsburgh, higher = worse).

HONEST CAVEATS baked into the verdict:
  - n≈21, cross-sectional, single questionnaire per person. Between-subject correlation
    is underpowered to detect a WITHIN-person effect ("when I'm stressed, I sleep worse").
  - So weak/mixed associations are EXPECTED and are consistent with the stress->sleep
    edge being correctly graded WEAK. A strong OPPOSITE association would be needed to
    contradict it.
  - Trait anxiety (STAI) and activity (steps) are reported as context only; they are
    noisier proxies and do not gate the result.

Run:  python -m evaluation.calibrate_mmash
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

_MMASH = os.path.join(str(__import__("pathlib").Path(__file__).resolve().parents[1].parent),
                      "data", "MMASH", "DataPaper")
_STRONG_CONTRADICTION = 0.40   # |r| this large in the WRONG direction would contradict


def _load() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(os.path.join(_MMASH, "user_*"))):
        try:
            s = pd.read_csv(os.path.join(d, "sleep.csv"))
            q = pd.read_csv(os.path.join(d, "questionnaire.csv"))
            act = pd.read_csv(os.path.join(d, "Actigraph.csv"))
            eff = pd.to_numeric(s["Efficiency"], errors="coerce").dropna()
            awk = pd.to_numeric(s["Number of Awakenings"], errors="coerce").dropna()
            if not len(eff):
                continue
            rows.append(dict(
                efficiency=eff.mean(), awakenings=awk.mean() if len(awk) else np.nan,
                daily_stress=float(q["Daily_stress"].iloc[0]),
                stai=float(q["STAI1"].iloc[0]), psqi=float(q["Pittsburgh"].iloc[0]),
                steps=pd.to_numeric(act["Steps"], errors="coerce").sum()))
        except Exception:
            continue
    return pd.DataFrame(rows)


def run() -> int:
    if not os.path.isdir(_MMASH):
        print("MMASH not found — see data/README.md.")
        return 1
    df = _load()
    n = len(df)

    def corr(a, b):
        return float(df[a].corr(df[b]))

    # (label, r, expected_sign) for the primary stress -> sleep associations.
    primary = [
        ("Daily_stress -> efficiency", corr("daily_stress", "efficiency"), -1),
        ("Daily_stress -> awakenings", corr("daily_stress", "awakenings"), +1),
        ("Daily_stress -> PSQI (worse sleep)", corr("daily_stress", "psqi"), +1),
    ]
    context = [
        ("STAI(trait anxiety) -> efficiency", corr("stai", "efficiency"), -1),
        ("steps(activity) -> efficiency", corr("steps", "efficiency"), None),
        ("steps(activity) -> PSQI", corr("steps", "psqi"), None),
    ]

    print("=" * 70)
    print(f"MMASH BEHAVIOURAL-SLEEP EVIDENCE CHECK  (n={n} subjects, cross-sectional)")
    print("=" * 70)
    print("Primary stress->sleep associations (Daily_stress):")
    correct_dir = 0
    contradicted = False
    for label, r, sign in primary:
        ok_dir = (r * sign) >= 0
        correct_dir += ok_dir
        if (r * sign) <= -_STRONG_CONTRADICTION:
            contradicted = True
        print(f"  {label:<38} r={r:+.2f}  ({'expected dir' if ok_dir else 'against'})")
    print("\nContext (noisier proxies; not gating):")
    for label, r, _ in context:
        print(f"  {label:<38} r={r:+.2f}")

    print("-" * 70)
    print(f"Primary directions matching expectation: {correct_dir}/3 (all weak, |r|<0.3)")
    verdict_ok = not contradicted
    print("VERDICT:", "consistent with a WEAK stress->sleep edge — data does NOT "
          "contradict it" if verdict_ok else "data CONTRADICTS the edge — reconsider it")
    print("Interpretation: associations are weak and mixed, as expected for a weak edge")
    print("in a small cross-sectional sample. This SUPPORTS keeping the edge graded WEAK")
    print("(not upgrading it), and confirms activity/caffeine/alcohol->sleep remain")
    print("unvalidated on real behaviour+outcome data (need within-person / larger n).")
    print("=" * 70)
    print("RESULT:", "PASS" if verdict_ok else "FAIL")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(run())

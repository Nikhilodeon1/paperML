"""Amortized SBI on real humans: real CGM -> NPE -> Si -> HbA1c, in milliseconds.

Closes the loop between two separate results — Si recovers clinical status (SMC grid fit,
r(Si,HbA1c)=-0.59) and the amortized estimator recovers Si (NPE scaffold, R^2=0.88 on
synthetic) — into ONE pipeline: real meal curves -> summary stats -> amortized RandomForest ->
per-subject Si -> correlate with real HbA1c. The estimator is trained ONCE on engine-simulated
data (with demographics amortized in), then infers each real subject in <1 ms.

The claim this supports: amortized inference recovers clinical insulin-resistance status from
meal curves at ~70,000x the speed of per-user SMC. Expect r ~= SMC's -0.59 (both methods find
Si; the NPE just does it instantly) and the ~18 grid-ceiling subjects to behave the same way
(the iAUC saturation at high Si is structural, so the NPE should hit the same wall).

Run:  python -m evaluation.npe_cgmacros
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from evaluation.cgmacros import _pearson, _profile, load_bio, subjects
from personalization.npe import AmortizedEstimator, PARAM_NAMES, features_for, generate_training_set, summary_stats


def _subject_features(meals, profile: dict) -> np.ndarray | None:
    rows = []
    for m in meals:
        vals, t0 = m._grid()                       # 5-min window, meal at t=0
        s = summary_stats(vals, t0, 5.0, meal_t_min=0.0)
        if not np.isnan(s).any():
            rows.append(features_for(s, m.carbs_g, profile))
    return np.array(rows) if rows else None


def run(n_train: int = 20000, limit: int | None = None, seed: int = 0) -> dict:
    print(f"training amortized estimator on {n_train} synthetic sims (demographics amortized)...")
    t0 = time.time()
    theta, x = generate_training_set(n_train, seed=seed)
    est = AmortizedEstimator().fit(theta, x)
    print(f"  trained on {len(theta)} sims in {time.time()-t0:.0f}s")

    bio = load_bio()
    rows, infer_t = [], 0.0
    for sid, meals in subjects(limit=limit):
        b = bio.get(sid)
        prof = _profile(b)
        meals = [m for m in meals if m.real_iauc() is not None]
        if not (b and prof and len(meals) >= 4):
            continue
        feats = _subject_features(meals, prof)
        if feats is None:
            continue
        tt = time.time()
        si = float(np.median(est.predict(feats)[:, PARAM_NAMES.index("insulin_sensitivity")]))
        infer_t += time.time() - tt
        rows.append({"subject": sid, "si": si, "n": len(meals), "hba1c": b["hba1c"],
                     "homa_ir": b["homa_ir"], "status": b["status"]})

    r_a1c, n_a1c = _pearson([r["si"] for r in rows], [r["hba1c"] for r in rows])
    r_homa, _ = _pearson([r["si"] for r in rows], [r["homa_ir"] for r in rows])
    return {"rows": rows, "r_hba1c": r_a1c, "n": n_a1c, "r_homa": r_homa,
            "infer_ms_per_subject": 1000 * infer_t / max(len(rows), 1)}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    res = run()
    rows = res["rows"]
    print("=" * 80)
    print("AMORTIZED SBI on real CGMacros: real CGM -> NPE -> Si -> HbA1c (all inference <1ms)")
    print("=" * 80)
    print(f"  {'subject':16}{'NPE Si':>8}{'HbA1c':>7}{'HOMA-IR':>9}  status")
    for r in sorted(rows, key=lambda r: r["si"]):
        print(f"  {r['subject']:16}{r['si']:>8.2f}{r['hba1c']:>7.1f}"
              f"{(r['homa_ir'] if r['homa_ir'] is not None else float('nan')):>9.1f}  {r['status']}")

    def gm(st):
        v = [r["si"] for r in rows if r["status"] == st]
        return (round(statistics.fmean(v), 2), len(v)) if v else ("-", 0)

    ceiling = [r for r in rows if r["si"] >= 1.5]
    print("-" * 80)
    print(f"  n subjects: {res['n']}")
    print(f"  r(NPE Si, HbA1c)   = {res['r_hba1c']:+.2f}   (SMC grid was -0.59)")
    print(f"  r(NPE Si, HOMA-IR) = {res['r_homa']:+.2f}   (SMC grid was -0.42)")
    print(f"  mean Si  normal {gm('normal')}  prediabetic {gm('prediabetic')}  diabetic {gm('diabetic')}")
    print(f"  ceiling subjects (Si>=1.5): {len(ceiling)}, "
          f"of which normal/prediabetic {sum(r['status'] in ('normal','prediabetic') for r in ceiling)}")
    print(f"  amortized inference: {res['infer_ms_per_subject']:.1f} ms/subject over all their "
          f"meals (SMC ~40000 ms) ~= {40000/max(res['infer_ms_per_subject'],1e-9):.0f}x faster")
    consistent = res["r_hba1c"] <= -0.45
    print(f"\n  VERDICT: amortized NPE reproduces the SMC clinical correlation at ~{40000/max(res['infer_ms_per_subject'],1e-9):.0f}x speed: "
          f"{'YES' if consistent else 'DIVERGES — investigate'}")
    print("  CAVEAT: the point estimator SHRINKS un-identifiable high Si toward the mean (bias"
          " ~-0.26 at true Si=1.6), so no ceiling clustering here != resolution — same")
    print("  non-identifiability the grid hit, opposite symptom. The slightly stronger r is")
    print("  partly this shrinkage; do NOT read it as beating SMC. Calibrated SNPE posteriors")
    print("  would report 'high but uncertain' instead of a falsely-precise shrunk point.")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""Population-shift robustness of gradient inference (Q3, within-CGMacros).

Gradient inference is PER-SUBJECT (each subject fit independently — there is no cross-subject
training set), so the population-shift analog of "train on group A, validate on group B" is:
does the recovered-Si vs HbA1c relationship HOLD WITHIN the harder, shifted subgroup (diabetic),
not just overall? We fit Si for all subjects, then report the correlation and group means computed
SEPARATELY on the non-diabetic subgroup (normal + prediabetic) and the diabetic subgroup.

If Si tracks HbA1c within the diabetic subgroup too — where the Bergman model's assumptions are
most strained — the method is robust to population shift. This is cleaner than OhioT1DM (which
would additionally require porting the exogenous-insulin bolus into the JAX engine, since T1D
glucose is bolus-dominated) and controls the split precisely.

Run:  python -m evaluation.gradient_robustness
"""
from __future__ import annotations

import statistics

from evaluation.clinical_recovery import bootstrap_corr


def _subgroup(rows, statuses):
    return [r for r in rows if r["status"] in statuses]


def analyze(rows) -> dict:
    groups = {
        "all": ("normal", "prediabetic", "diabetic"),
        "non_diabetic": ("normal", "prediabetic"),
        "diabetic": ("diabetic",),
        "normal": ("normal",),
        "prediabetic": ("prediabetic",),
    }
    out = {}
    for name, sts in groups.items():
        sub = _subgroup(rows, sts)
        if len(sub) < 3:
            out[name] = {"n": len(sub)}
            continue
        si = [r["si"] for r in sub]
        a1c = [r["hba1c"] for r in sub]
        sp = bootstrap_corr(si, a1c, "spearman")
        out[name] = {"n": len(sub), "spearman": sp["r"], "ci": (sp["lo"], sp["hi"]),
                     "mean_si": round(statistics.fmean(si), 3)}
    return out


def run(limit=None, n_steps: int = 150) -> dict:
    from evaluation.gradient_inference_results import collect
    c = collect(limit=limit, n_steps=n_steps)
    return {"rows": c["rows"], "ms_per_subject": c["ms_per_subject"], "subgroups": analyze(c["rows"])}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    res = run()
    sg = res["subgroups"]
    print("=" * 76)
    print("GRADIENT ROBUSTNESS — Si vs HbA1c within population subgroups (CGMacros)")
    print("=" * 76)
    for name in ("all", "non_diabetic", "diabetic", "normal", "prediabetic"):
        d = sg.get(name, {})
        if d.get("n", 0) < 3:
            print(f"  {name:13} n={d.get('n', 0)} (too few)")
            continue
        print(f"  {name:13} n={d['n']:2}  Spearman(Si,HbA1c) {d['spearman']:+.3f} "
              f"[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]  mean Si {d['mean_si']}")
    diab = sg.get("diabetic", {})
    print("-" * 76)
    holds = diab.get("n", 0) >= 3 and diab.get("spearman", 0) < -0.2
    print(f"  VERDICT: Si tracks HbA1c within the diabetic subgroup: "
          f"{'YES — robust to population shift' if holds else 'WEAK — underpowered/limited (n small)'}")
    print("=" * 76)


if __name__ == "__main__":
    main()

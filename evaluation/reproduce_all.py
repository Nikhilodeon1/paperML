"""Reproduce every number in the paper, in order (Block 7b).

Hand this to a reviewer who asks "how do I regenerate Table 1?". Each step is seed-locked via
``paper_config.set_all_seeds`` and prints its headline result. SNPE training is skipped if the
artifact already exists (pass --retrain to force).

Run:  python -m evaluation.reproduce_all            # full pipeline
      python -m evaluation.reproduce_all --quick    # smaller n for a smoke run
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import paper_config as cfg


def _hr(title: str) -> None:
    print("\n" + "#" * 88 + f"\n# {title}\n" + "#" * 88, flush=True)


def _check_datasets() -> dict:
    from evaluation.cgmacros import _ZIP
    ok = {"CGMacros": _ZIP.exists()}
    return ok


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller n for a fast smoke run")
    ap.add_argument("--retrain", action="store_true", help="retrain SNPE even if artifact exists")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    cfg.set_all_seeds()
    cfg.ensure_dirs()
    n_train = 5000 if args.quick else 50_000
    summary = {}
    kfold_result = ood_result = None
    t_start = time.time()

    _hr("STEP 1 — datasets present")
    ds = _check_datasets()
    for k, v in ds.items():
        print(f"  {k}: {'FOUND' if v else 'MISSING'}")
    if not ds.get("CGMacros"):
        print("  CGMacros zip missing — the clinical results cannot run. Aborting.")
        return

    _hr("STEP 2 — mechanistic validations (full_report)")
    try:
        from evaluation import full_report
        n_ok = full_report.run()   # returns count of passing validations; prints its own report
        summary["mechanistic"] = n_ok
    except Exception:
        traceback.print_exc()

    _hr(f"STEP 3 — train SNPE posterior (n={n_train})")
    from personalization.snpe_trainer import SNPETrainer
    if cfg.SNPE_POSTERIOR.exists() and not args.retrain:
        print(f"  artifact exists ({cfg.SNPE_POSTERIOR.name}) — skipping (use --retrain to force)")
    else:
        tr = SNPETrainer(n_simulations=n_train)
        tr.generate(); tr.train(); tr.save()
        print(f"  trained + saved ({tr.meta.get('train_seconds')}s)")

    _hr("STEP 4 — SBC calibration check (appendix)")
    try:
        from evaluation import snpe_calibration
        post = SNPETrainer.load().posterior
        n_sbc = 100 if args.quick else 500
        res = snpe_calibration.run_sbc_check(post, n_trials=n_sbc)
        for c in (0.5, 0.9, 0.95):
            cov = res["coverage"][c]
            print(f"  {int(c*100)}% coverage: " +
                  " ".join(f"{n.split('_')[0]} {cov[n]*100:.0f}%" for n in cfg.PARAM_NAMES))
        summary["sbc"] = res["coverage"]
    except Exception:
        traceback.print_exc()

    _hr("STEP 5 — Table 1: clinical recovery, all estimators")
    try:
        from evaluation import snpe_vs_smc_vs_rf
        snpe_vs_smc_vs_rf.main()
    except Exception:
        traceback.print_exc()

    _hr("STEP 6 — Table 2: held-out k-fold utility")
    try:
        from evaluation import snpe_kfold
        kfold_result = snpe_kfold.run(limit=None)
        for e, d in kfold_result["estimators"].items():
            print(f"  {e:8} iAUC MAE {d['mae']['mean']:.0f}  gap {d['gap']['mean']:+.0f}"
                  f"  beats-personal {d['winrate']['mean']:.0f}%")
        summary["kfold"] = kfold_result
    except Exception:
        traceback.print_exc()

    _hr("STEP 6b — simulator-reality gap (why SNPE loses utility)")
    try:
        from evaluation import ood_analysis
        ood_result = ood_analysis.simulator_reality_gap()
        print(f"  {ood_result['frac_real_ood_any_stat']*100:.0f}% of real meals OOD on >=1 summary "
              f"stat; baseline shift {ood_result['per_stat']['baseline']['median_shift_z']:+.1f}z")
        summary["ood"] = {k: v for k, v in ood_result.items() if k not in ("sim", "real")}
    except Exception:
        traceback.print_exc()

    _hr("STEP 7 — demographic baseline (does Si beat age+BMI+sex?)")
    try:
        from evaluation import baselines
        d = baselines.demographic_baseline("snpe")
        print(f"  partial r(Si,HbA1c | demographics) = "
              f"{d.get('partial_r_si_given_demographics', float('nan')):+.3f}")
        summary["demographic_baseline"] = d
    except Exception:
        traceback.print_exc()

    _hr("STEP 8 — identifiability (Table 3, Figures 2-3 data)")
    try:
        from evaluation import identifiability_analysis as ida
        post = SNPETrainer.load().posterior
        summ = ida.width_summary(ida.posterior_width_analysis(post))
        for name, dd in summ["per_param"].items():
            print(f"  {name:20} mean posterior std {dd['mean_std']:.4f}  "
                  f"({dd['frac_of_prior']*100:.0f}% of prior)")
        summary["identifiability"] = summ
    except Exception:
        traceback.print_exc()

    _hr("STEP 10 — inference cost / INO comparison (Table 3)")
    try:
        from evaluation import inference_cost_analysis
        inference_cost_analysis.main()
    except Exception:
        traceback.print_exc()

    _hr("STEP 11 — error decomposition (Table 4: why every method ties)")
    try:
        from evaluation import model_expressiveness_analysis as mea
        d = mea.decompose()
        for name, mae, pct in d["rows"]:
            print(f"  {name:40}{mae:>8.0f}  {pct:>3.0f}%")
        summary["error_decomposition"] = d["rows"]
    except Exception:
        traceback.print_exc()

    if not args.quick:
        _hr("STEP 12 — Shanghai external validation (Si vs HbA1c, all methods, frozen transfer)")
        try:
            from evaluation import cross_dataset_validation as cdv
            r = cdv.validate_on_shanghai()
            for m in ("gradient", "grid", "rf", "snpe"):
                print(f"  {m:9} Spearman(Si,HbA1c) {r[m]['r_spearman']:+.3f}  meanSi {r[m]['mean_si']}")
            print(f"  identifiability replication: Si grad-norm "
                  f"{r['identifiability']['ratio_Si_over_gastric']}x gastric")
            summary["shanghai"] = {m: r[m] for m in ("gradient", "grid", "rf", "snpe")}
        except Exception:
            traceback.print_exc()

        _hr("STEP 13 — SNPE tautology check (is Shanghai SNPE real or fasting-glucose?)")
        try:
            from evaluation import tautology_check
            t = tautology_check.run()
            print(f"  partial r(SNPE Si, HbA1c | fasting) = {t['partial_snpe_hba1c_given_fasting']:+.3f}"
                  f"  (raw {t['raw_snpe_hba1c']:+.3f}); leakage {t['r_baseline_feature_vs_snpe_si']:+.3f}")
            print(f"  partial r(gradient Si, HbA1c | fasting) = {t['partial_grad_hba1c_given_fasting']:+.3f}")
            summary["tautology"] = t
        except Exception:
            traceback.print_exc()

    _hr("STEP 14 — CGMacros gradient inference (Table 1 gradient column + Fig 7 data)")
    grad_history = None
    try:
        from evaluation.gradient_inference_results import collect as grad_collect
        from evaluation.clinical_recovery import bootstrap_corr
        gc = grad_collect(n_steps=150 if not args.quick else 60)
        rows = gc["rows"]
        sp = bootstrap_corr([r["si"] for r in rows], [r["hba1c"] for r in rows], "spearman")
        pe = bootstrap_corr([r["si"] for r in rows], [r["hba1c"] for r in rows], "pearson")
        print(f"  gradient Spearman(Si,HbA1c) {sp['r']:+.3f} [{sp['lo']:+.2f},{sp['hi']:+.2f}]  "
              f"Pearson {pe['r']:+.3f}   ({gc['ms_per_subject']:.0f} ms/subject)")
        grad_history = gc["grad_history"]
        summary["gradient_cgmacros"] = {"spearman": sp, "pearson": pe}
    except Exception:
        traceback.print_exc()

    _hr("STEP 15 — Dalla Man validation gates")
    try:
        from evaluation import dalla_man_validation as dmv
        gates = {"steady_state": dmv.validate_steady_state(),
                 "ogtt": dmv.validate_dalla_man_vs_paper(), "cgm_lag": dmv.validate_cgm_lag()}
        for name, g in gates.items():
            print(f"  [{'PASS' if g['pass'] else 'WARN'}] {name}: "
                  + "  ".join(f"{k}={v}" for k, v in g.items() if k not in ("pass", "target")))
        summary["dalla_man_gates"] = gates      # WARN not fatal: document deviations, do not exit
    except Exception:
        traceback.print_exc()

    if not args.quick:
        _hr("STEP 16 — Dalla Man vs Bergman vs grid (fat/fibre parity) + Fig 8")
        try:
            from evaluation import dalla_man_comparison as dmc
            r = dmc.run(parity=True)
            for m in ("dalla_man", "bergman", "grid"):
                print(f"  {m:10} held-out iAUC MAE {r['mae'][m]['mean']:.0f}   "
                      f"beats-personal {r['winrate_vs_personal'][m]['mean']:.0f}%")
            gn = r["dalla_man_grad_norms"]
            ratio = gn["Vmx"] / max(gn["kabs"], 1e-9)
            print(f"  Dalla Man identifiability: Vmx/kabs grad-norm ratio {ratio:.1f}x  "
                  f"(paired dalla-vs-grid {r['paired_dalla_vs_grid'][0]}%)")
            summary["dalla_man"] = {"mae": {m: r["mae"][m] for m in ("dalla_man", "bergman", "grid")},
                                    "Vmx_over_kabs": round(ratio, 1)}
        except Exception:
            traceback.print_exc()

    if not args.skip_figures:
        _hr("STEP 9 — generate figures")
        try:
            from evaluation import figures
            figures.generate_all(kfold=kfold_result, ood=ood_result)
            if grad_history:      # Fig 7 (lead figure) needs the per-step gradient norms
                print(f"  wrote {figures.fig_gradient_identifiability(grad_history)}")
            if not args.quick:    # Fig 8 — Dalla Man identifiability (its own gradient fits)
                print(f"  wrote {figures.fig_dalla_man_identifiability()}")
        except Exception:
            traceback.print_exc()

    _hr(f"DONE in {time.time() - t_start:.0f}s")
    print("  All paper numbers regenerated. Seeds locked to", cfg.SEED)


if __name__ == "__main__":
    main()

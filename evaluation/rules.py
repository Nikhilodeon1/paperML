"""Apply the pre-written decision rules (REPORTS/decision_rules_2026-11-26.md) to the stored sweeps.

Each rule is a small function of counts, and `evaluate` feeds it from complete sweeps only: a rule whose sweeps are not
all complete is reported as waiting. R4 and R12 are decisions rather than computations, and R17 and R18 concern how
results are reported, so they have no function here.

Run:  python -m evaluation.rules        # writes REPORTS/rule_outcomes.md and results/rule_outcomes.json
"""
from __future__ import annotations

import json
import math

import numpy as np

from evaluation import manifest
from evaluation.recalibration import _rows
from evaluation.results_io import RESULTS, ROOT

INTERIOR = 0.01
REPORT = ROOT / "REPORTS" / "rule_outcomes.md"
OUT = RESULTS / "rule_outcomes.json"
SI, KE, KA = "insulin_sensitivity", "gastric_emptying", "carb_absorption"
FITTED_QUARTER_SHIFT = 0.064                                       # R1's threshold, from the run that preceded it


# --------------------------------------------------------------------------------------------------
# The rules, as functions of numbers
# --------------------------------------------------------------------------------------------------

def r1(shift_075: float, covers_075: int, covers_exact: int) -> dict:
    return {"account": "stands" if shift_075 > FITTED_QUARTER_SHIFT else "cut the explanation",
            "coverage_sentence": "keep" if covers_075 < covers_exact else "replace with counts"}


def agrees(block: dict) -> bool:
    return block["covers_075"] < block["covers_0"] and block["shift_075"] > block["shift_0"]


def r2(seeds: list[dict]) -> str:
    return "held on every fitted-truth seed" if all(map(agrees, seeds)) else "report as seed dependence"


def r3(block: dict, exact_covers: int, n: int) -> dict:
    return {"replicates": agrees(block),
            "near_nominal_stated_for": "CGMacros and Shanghai" if exact_covers >= 0.8 * n else "CGMacros only"}


def r5(counts: dict, n: int) -> str:
    """`counts`: box width -> (bounded k_e, bounded k_a)."""
    if all(max(v) == 0 for v in counts.values()):
        return "keep: no interval bounded"
    if max(counts[1.0]) == 0:
        return f"say at most {max(max(v) for v in counts.values())} of {n}; keep zero for the primary box"
    return "state the count; remove the zero statement"


def r6(bounded: dict) -> str:
    return "box-dependent" if any(bounded[b] != bounded[1.0] for b in bounded) else "remove box-dependent"


def r7(shift_exact: float, shift_noisy: float, shift_interior: float) -> dict:
    return {"present_with_exact_input": bool(shift_exact * shift_noisy > 0 and abs(shift_exact) >= 0.5 * abs(shift_noisy)),
            "attributed_to_pinning": bool(abs(shift_interior) < abs(shift_noisy) / 3.0)}


def r8_material(base: dict, perturbed: dict) -> bool:
    return bool(abs(perturbed["coverage_pct"] - base["coverage_pct"]) >= 15.0
                or abs(perturbed["shift"] - base["shift"]) >= 0.2 - 1e-12)


def r9(bounded: int, interior: int) -> str:
    if interior and bounded >= 0.5 * interior:
        return "remove and rewrite the information-limit reading"
    return "keep, with the new count"


def r10(rates: dict) -> str:
    met = all(v["interior"] and v["bounded"] >= 0.5 * v["interior"] and v["covers"] >= 0.9 * v["n"]
              for v in rates.values())
    return "rewrite the information-limit reading" if met else "keep, with the new counts"


def r11(covers_generator: int, covers_pseudo_bounded: int, bounded: int) -> dict:
    return {"zero_generator_statement": "keep" if covers_generator == 0 else "give the count",
            "short_of_nominal_statement": "keep" if bounded and covers_pseudo_bounded < 0.9 * bounded else "remove"}


def r13(seeds: list[dict]) -> str:
    for s in seeds:
        halved = abs(s["shift_calibrated"]) <= 0.5 * abs(s["shift_naive"])
        recovered = s["covers_calibrated"] - s["covers_naive"] >= 0.5 * (s["covers_exact"] - s["covers_naive"])
        if not (halved and recovered):
            return "mechanism not established"
    return "supported"


def material_variant(block: dict) -> bool:
    """R14 and R15."""
    return bool(block["iauc_bounded_variant"] > block["iauc_bounded_full"]
                or abs(block["trace_share_variant"] - block["trace_share_full"]) >= 15.0)


def r19(exact_random, covers_075_vs_exact, widths_random, fit_bounded, random_bounded, rank_075) -> dict:
    return {"near_nominal": all(c >= 41 for c in exact_random),
            "coverage_falls": all(at_075 < exact for at_075, exact in covers_075_vs_exact),
            "without_widening": all(at_075 <= exact for exact, at_075 in widths_random),
            "opposite_directions": bool(fit_bounded[1] > fit_bounded[0]
                                        and all(at_075 < exact for exact, at_075 in random_bounded)),
            "ordering": all(r >= 0.6 for r in rank_075)}


# --------------------------------------------------------------------------------------------------
# Numbers from stored rows
# --------------------------------------------------------------------------------------------------

def clipped_interval(row: dict, key: str) -> tuple[float, float]:
    """The profile interval in natural units, with an open side running to the edge of the box."""
    low, high = row["box"][key]
    if row.get("parameterization", "rates") != "rates":
        low, high = math.exp(low), math.exp(high)
    cls = row["profiles"][key]["classification"]
    ci_low, ci_high = cls.get("ci_low"), cls.get("ci_high")
    return (low if ci_low is None else max(ci_low, low), high if ci_high is None else min(ci_high, high))


def covers(row: dict, key: str, truth: float | None = None) -> bool:
    value = row["profiles"][key]["truth"] if truth is None else truth
    low, high = clipped_interval(row, key)
    return bool(low <= value <= high)


def is_bounded(row: dict, key: str) -> bool:
    return row["profiles"][key]["classification"]["verdict"] == "identifiable"


def log_estimate(row: dict, key: str) -> float:
    value = row["theta_ml"][key]
    return math.log(value) if row.get("parameterization", "rates") == "rates" else value


def shift(rows: list[dict], key: str) -> float:
    return float(np.median([log_estimate(r, key) - math.log(r["profiles"][key]["truth"]) for r in rows]))


def n_covers(rows, key) -> int:
    return sum(covers(r, key) for r in rows)


def n_bounded(rows, key) -> int:
    return sum(is_bounded(r, key) for r in rows)


def interior(rows, key) -> list[dict]:
    return [r for r in rows if r["distance_to_bound"][key] > INTERIOR]


def share_bounded_interior(rows, key) -> float:
    inner = interior(rows, key)
    return 100.0 * n_bounded(inner, key) / len(inner) if inner else float("nan")


def median_width(rows, key) -> float:
    return float(np.median([math.log(hi / lo) for lo, hi in (clipped_interval(r, key) for r in rows)]))


def rank(rows, key) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr([log_estimate(r, key) for r in rows],
                           [math.log(r["profiles"][key]["truth"]) for r in rows])[0])


def complete_rows(name: str) -> list[dict] | None:
    """The rows of a registered sweep, or None unless it is complete."""
    if name in manifest.other_sweeps():
        if not manifest.is_complete(manifest.check_other(name)):
            return None
        from evaluation import dalla_man_identifiability as dm
        from evaluation.results_io import largest_matching
        _, config = manifest.other_sweeps()[name]
        return list(largest_matching(dm.ANALYSIS_ID, lambda p: dm.matches(p, config)).values())
    config = manifest.profile_sweeps()[name]
    if not manifest.is_complete(manifest.check(config)):
        return None
    rows = _rows(config)
    if manifest.spurious_bounded(rows):
        raise RuntimeError(f"{name}: a profile is classed bounded with no in-box point below the threshold")
    return rows


class _Waiting(Exception):
    pass


def _need(name: str) -> list[dict]:
    rows = complete_rows(name)
    if rows is None:
        raise _Waiting(name)
    return rows


def _block(prefix: str, suffix: str) -> dict:
    zero, high = _need(f"{prefix}0.0{suffix}"), _need(f"{prefix}0.75{suffix}")
    return {"covers_0": n_covers(zero, "log_tau1"), "covers_075": n_covers(high, "log_tau1"),
            "shift_0": shift(zero, "log_tau1"), "shift_075": shift(high, "log_tau1")}


def _e_r1():
    high, exact = _need("fit0.75_s0"), _need("fit0.0_s0")
    numbers = {"shift_075": shift(high, "log_tau1"), "covers_075": n_covers(high, "log_tau1"),
               "covers_exact": n_covers(exact, "log_tau1")}
    return numbers, r1(**numbers)


def _e_r2():
    seeds = [_block("fit", f"_s{seed}") for seed in (1, 2)]
    return {"seeds 1 and 2": seeds}, r2(seeds)


def _e_r3():
    block, exact = _block("shanghai_rep", ""), _need("shanghai_rep0.0")
    real = _need("shanghai_real")
    counts = {k: {"interior": len(interior(real, k)), "bounded_interior": n_bounded(interior(real, k), k),
                  "bounded_all": n_bounded(real, k)} for k in ("log_tau1", "log_p")}
    return {**block, "n": len(exact), "real": counts}, r3(block, block["covers_0"], len(exact))


def _e_r5_r6():
    rows = {box: _need(f"iauc_box{box:g}") for box in (1.0, 2.0, 4.0)}
    table = {f"{box:g}x": {k: {"interior": len(interior(r, k)), "bounded": n_bounded(r, k)} for k in (KE, KA, SI)}
             for box, r in rows.items()}
    rates = {box: (n_bounded(r, KE), n_bounded(r, KA)) for box, r in rows.items()}
    return table, {"R5": r5(rates, len(rows[1.0])), "R6": r6({box: n_bounded(r, SI) for box, r in rows.items()})}


def _si_block(rows):
    return {"bounded_interior": n_bounded(interior(rows, SI), SI), "interior": len(interior(rows, SI)),
            "coverage_pct": 100.0 * n_covers(rows, SI) / len(rows), "shift": shift(rows, SI),
            "rate_bounded": [n_bounded(rows, KE), n_bounded(rows, KA)]}


def _e_r7():
    seeds = {s: _si_block(_need(f"iauc_rep_s{s}")) for s in range(5)}
    noisy, exact = _need("iauc_rep_s0"), _need("iauc_rep_exact")
    numbers = {"shift_exact": shift(exact, SI), "shift_noisy": shift(noisy, SI),
               "shift_interior": shift(interior(noisy, SI), SI)}
    others = {n: _si_block(_need(n)) for n in ("iauc_rep_exact", "iauc_rep_noise0.5", "iauc_rep_noise0.25",
                                                 "iauc_synth_s0", "iauc_synth_s1")}
    return {"replica seeds": seeds, "other replicas": others, **numbers}, r7(**numbers)


def _e_r8():
    base = _si_block(_need("iauc_rep_s0"))
    blocks, verdicts = {}, {}
    for constant in manifest.PERTURBED:
        for factor in (0.7, 1.3):
            name = f"iauc_pert_{constant}_{factor:g}"
            blocks[name] = _si_block(_need(name))
            verdicts[name] = {"material_for_S_I": r8_material(base, blocks[name]),
                              "bounded_rates": blocks[name]["rate_bounded"]}
    return {"unperturbed": base, **blocks}, verdicts


def _e_r9():
    rows = _need("centroid_real")
    inner = interior(rows, "log_tau1")
    numbers = {"bounded": n_bounded(inner, "log_tau1"), "interior": len(inner)}
    replica = _need("centroid_rep")
    return {**numbers, "replica_covers": n_covers(replica, "log_tau1")}, r9(**numbers)


def _e_r10():
    names = {"0%": "residual0_s0", "5%": "residual0.05_s0", "10%": "residual0.1_s0", "20%": "residual0.2_s0",
             "10% seed 1": "residual0.1_s1"}
    table = {}
    for label, name in names.items():
        rows = _need(name)
        table[label] = {k: {"bounded": n_bounded(interior(rows, k), k), "interior": len(interior(rows, k)),
                            "covers": n_covers(rows, k), "n": len(rows)} for k in (KE, KA)}
    return table, r10(table["10%"])


def _dm_numbers(suffix: str) -> dict:
    noisy = {r["subject_id"]: r for r in _need(f"dm_cgm{suffix}")}
    clean = {r["subject_id"]: r for r in _need(f"dm_clean{suffix}")}
    ids = sorted(noisy)
    pseudo = {i: math.exp(clean[i]["theta_ml"]["log_tau1"]) for i in ids}
    truth = {i: noisy[i]["profiles"]["log_tau1"]["truth"] for i in ids}
    bounded = [i for i in ids if is_bounded(noisy[i], "log_tau1")]
    from scipy.stats import spearmanr
    estimate = [math.exp(noisy[i]["theta_ml"]["log_tau1"]) for i in ids]
    return {"n": len(ids), "bounded": len(bounded),
            "covers_generator": sum(covers(noisy[i], "log_tau1") for i in ids),
            "covers_generator_bounded": sum(covers(noisy[i], "log_tau1") for i in bounded),
            "covers_pseudo": sum(covers(noisy[i], "log_tau1", pseudo[i]) for i in ids),
            "covers_pseudo_bounded": sum(covers(noisy[i], "log_tau1", pseudo[i]) for i in bounded),
            "median_bias_min": float(np.median([pseudo[i] - truth[i] for i in ids])),
            "median_pseudo_min": float(np.median(list(pseudo.values()))),
            "median_generator_min": float(np.median(list(truth.values()))),
            "median_abs_estimate_minus_pseudo_min": float(np.median([abs(e - pseudo[i]) for e, i in zip(estimate, ids)])),
            "rank_with_generator": float(spearmanr(estimate, [truth[i] for i in ids])[0]),
            "rank_with_pseudo": float(spearmanr(estimate, [pseudo[i] for i in ids])[0])}


def _e_r11():
    numbers = _dm_numbers("")
    verdict = r11(numbers["covers_generator"], numbers["covers_pseudo_bounded"], numbers["bounded"])
    real = _need("dm_real_iauc")
    numbers["real_data_six_parameter_bounded"] = {k: n_bounded(real, k) for k in real[0]["profiles"]}
    return numbers, verdict


def _e_r13():
    seeds = []
    for seed in (0, 1):
        exact, naive = _need(f"rand0.0_s{seed}"), _need(f"rand0.75_s{seed}")
        calibrated = _need(f"calibrated0.75_s{seed}")
        seeds.append({"shift_naive": shift(naive, "log_tau1"), "shift_calibrated": shift(calibrated, "log_tau1"),
                      "covers_naive": n_covers(naive, "log_tau1"),
                      "covers_calibrated": n_covers(calibrated, "log_tau1"),
                      "covers_exact": n_covers(exact, "log_tau1")})
    quarter = {}
    for seed in (0, 1):
        naive, calibrated = _need(f"rand0.25_s{seed}"), _need(f"calibrated0.25_s{seed}")
        quarter[seed] = {"covers_naive": n_covers(naive, "log_tau1"), "covers_calibrated": n_covers(calibrated, "log_tau1"),
                         "shift_naive": shift(naive, "log_tau1"), "shift_calibrated": shift(calibrated, "log_tau1")}
    return {"CV 0.75": seeds, "CV 0.25": quarter}, r13(seeds)


def _compare(variant_iauc, variant_trace, full_iauc, full_trace) -> dict:
    ids = {r["subject_id"] for r in variant_trace}
    full_trace = [r for r in full_trace if r["subject_id"] in ids]
    ids_iauc = {r["subject_id"] for r in variant_iauc}
    full_iauc = [r for r in full_iauc if r["subject_id"] in ids_iauc]
    return {"subjects": len(ids),
            "iauc_bounded_variant": max(n_bounded(variant_iauc, KE), n_bounded(variant_iauc, KA)),
            "iauc_bounded_full": max(n_bounded(full_iauc, KE), n_bounded(full_iauc, KA)),
            "trace_share_variant": share_bounded_interior(variant_trace, "log_tau1"),
            "trace_share_full": share_bounded_interior(full_trace, "log_tau1")}


def _e_r14():
    blocks = {f"{h:g} h": _compare(_need(f"gap{h:g}_iauc"), _need(f"gap{h:g}_trace"), _need("iauc_box1"), _need("real"))
              for h in (4.0, 6.0)}
    return blocks, {k: "material" if material_variant(v) else "did not depend on it" for k, v in blocks.items()}


def _e_r15():
    blocks = {"CGMacros": _compare(_need("basal_iauc"), _need("basal_trace"), _need("iauc_box1"), _need("real")),
              "Shanghai": _compare(_need("basal_shanghai_iauc"), _need("basal_shanghai_trace"),
                                   _need("iauc_shanghai_box1"), _need("shanghai_real"))}
    return blocks, {k: "material" if material_variant(v) else "did not depend on it" for k, v in blocks.items()}


def _e_r16():
    plain, lagged = _dm_numbers(""), _dm_numbers("_lag")
    return {"no lag": plain, "10 min lag": lagged}, {
        "median_bias_change_min": lagged["median_bias_min"] - plain["median_bias_min"]}


def _e_window():
    table = {}
    for name in ("window60", "window90", "window120", "iauc_box1"):
        rows = _need(name)
        table[name] = {k: {"interior": len(interior(rows, k)), "bounded": n_bounded(rows, k)} for k in (KE, KA)}
    any_bounded = any(v[k]["bounded"] for v in table.values() for k in (KE, KA))
    return table, "give the counts" if any_bounded else "keep: none bounded from 60 to 180 minutes"


def _e_r19():
    random = {(cv, s): _need(f"rand{cv}_s{s}") for cv in (0.0, 0.75) for s in (0, 1)}
    fit = {cv: _need(f"fit{cv}_s0") for cv in (0.0, 0.25, 0.75)}
    extra = {(cv, s): _need(f"fit{cv}_s{s}") for cv in (0.0, 0.75) for s in (1, 2)}
    pairs = [(random[0.75, s], random[0.0, s]) for s in (0, 1)] + [(fit[0.75], fit[0.0])]
    pairs += [(extra[0.75, s], extra[0.0, s]) for s in (1, 2)]
    numbers = {
        "exact_random": [n_covers(random[0.0, s], "log_tau1") for s in (0, 1)],
        "covers_075_vs_exact": [(n_covers(a, "log_tau1"), n_covers(b, "log_tau1")) for a, b in pairs],
        "widths_random": [(median_width(random[0.0, s], "log_tau1"), median_width(random[0.75, s], "log_tau1"))
                          for s in (0, 1)],
        "fit_bounded": (n_bounded(fit[0.0], "log_tau1"), n_bounded(fit[0.25], "log_tau1")),
        "random_bounded": [(n_bounded(random[0.0, s], "log_tau1"), n_bounded(random[0.75, s], "log_tau1"))
                           for s in (0, 1)],
        "rank_075": [rank(random[0.75, s], "log_tau1") for s in (0, 1)]}
    return numbers, r19(**numbers)


EVALUATIONS = {"R1": _e_r1, "R2": _e_r2, "R3": _e_r3, "R5 and R6": _e_r5_r6, "R7": _e_r7, "R8": _e_r8, "R9": _e_r9,
               "R10": _e_r10, "R11": _e_r11, "R13": _e_r13, "R14": _e_r14, "R15": _e_r15, "R16": _e_r16,
               "Window sweep": _e_window, "R19": _e_r19}


def evaluate() -> dict:
    out = {}
    for label, function in EVALUATIONS.items():
        try:
            numbers, verdict = function()
        except _Waiting as waiting:
            out[label] = {"status": "waiting for data", "first_missing_sweep": str(waiting)}
            continue
        out[label] = {"status": "applied", "verdict": verdict, "numbers": numbers}
    return out


def main() -> int:
    outcomes = evaluate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(outcomes, indent=2, default=float), encoding="utf-8")
    lines = ["# Rule outcomes (generated by `python -m evaluation.rules`; do not edit)", ""]
    for label, block in outcomes.items():
        lines.append(f"## {label}: {block['status']}")
        if block["status"] == "applied":
            lines += ["", f"**Verdict:** `{json.dumps(block['verdict'], default=float)}`", "",
                      "```json", json.dumps(block["numbers"], indent=1, default=float), "```"]
        else:
            lines.append(f"First incomplete sweep: `{block['first_missing_sweep']}`")
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    applied = sum(b["status"] == "applied" for b in outcomes.values())
    for label, block in outcomes.items():
        print(f"{label:14s} {block['status']}" + (f": {json.dumps(block['verdict'], default=float)[:150]}"
                                                  if block["status"] == "applied" else ""))
    print(f"{applied} of {len(outcomes)} rule groups applied")
    return 0 if applied == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Numbers and statement checks of the workshop paper.

Final mode reads only complete sweeps of the converged protocol (the same registry as the journal paper) and stops if a
sweep is incomplete or if a sentence the paper prints is no longer true of the numbers. Draft mode reads the earlier
stored runs, so the text can be written and laid out before the reruns exist. A draft is never to be submitted.

Run:  python workshop_b2e/ws_numbers.py [--draft]      # writes ws_numbers.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from evaluation import manifest, profile_lik, rules  # noqa: E402
from evaluation.recalibration import _rows  # noqa: E402

OUT = HERE / "ws_numbers.json"
SUBJECTS = 45
CV = {0.0: "Zero", 0.25: "Quarter", 0.75: "ThreeQuarter"}
TRUTHS = {"RandA": ("rand", 0), "RandB": ("rand", 1), "Fit": ("fit", 0)}
BOXES = {1.0: "One", 2.0: "Two", 4.0: "Four"}


def trace_rows(name: str, draft: bool):
    if not draft:
        return rules.complete_rows(name)
    rows = _rows({**manifest.phase21_sweeps()[name], "lbfgsb_restarts": 0})      # the earlier solver, same grid
    return rows if len(rows) == SUBJECTS else None


def box_rows(box: float, draft: bool):
    if not draft:
        return rules.complete_rows(f"iauc_box{box:g}")
    rows = _rows({**profile_lik.default_config(), "objective": "iauc", "bounds_scale": box})   # the earlier protocol
    return rows if len(rows) == SUBJECTS else None


def need(rows, name: str):
    if rows is None:
        raise SystemExit(f"incomplete sweep: {name}. Download the pod results first, or build with --draft.")
    return rows


def numbers(draft: bool) -> tuple[dict, dict, dict]:
    """Macros, the raw values the statements are checked on, and the per-sweep blocks for the figure."""
    macros: dict = {"wsN": str(SUBJECTS)}
    raw: dict = {}
    boxes = {}
    for box, word in BOXES.items():
        rows = box_rows(box, draft)
        if rows is None and not (draft and box == 4.0):
            need(rows, f"iauc_box{box:g}")
        if rows is None:
            continue
        boxes[box] = rows
        for key, tag in ((rules.KE, "Ke"), (rules.KA, "Ka"), (rules.SI, "SI")):
            inner = rules.interior(rows, key)
            macros[f"wsBox{word}{tag}Int"] = str(len(inner))
            macros[f"wsBox{word}{tag}Bd"] = str(rules.n_bounded(rows, key))
    macros["wsBoxRows"] = " ".join(
        f"${box:g}\\times$ & " + " & ".join(macros[f"wsBox{BOXES[box]}{tag}{what}"]
                                              for tag in ("Ke", "Ka", "SI") for what in ("Int", "Bd")) + " \\\\"
        for box in sorted(boxes))
    macros["wsBoxRange"] = "up to four times as wide" if 4.0 in boxes else "up to twice as wide"
    raw["rate_bounded"] = sum(rules.n_bounded(r, k) for r in boxes.values() for k in (rules.KE, rules.KA))
    raw["boxes"] = sorted(boxes)
    raw["si_bounded"] = {box: rules.n_bounded(r, rules.SI) for box, r in boxes.items()}

    real = need(trace_rows("real", draft), "real")
    for key, tag in (("log_tau1", "Tau"), ("log_p", "P")):
        inner = rules.interior(real, key)
        macros[f"wsReal{tag}Int"] = str(len(inner))
        macros[f"wsReal{tag}Bd"] = str(rules.n_bounded(inner, key))

    blocks = {}
    for label, (truth, seed) in TRUTHS.items():
        for cv, word in CV.items():
            name = f"{truth}{cv}_s{seed}"
            rows = need(trace_rows(name, draft), name)
            lo_hi = rows[0]["box"]["log_tau1"]
            block = {
                "cover": rules.n_covers(rows, "log_tau1"), "coverP": rules.n_covers(rows, "log_p"),
                "bounded": rules.n_bounded(rows, "log_tau1"), "shift": rules.shift(rows, "log_tau1"),
                "width": rules.median_width(rows, "log_tau1"), "rho": rules.rank(rows, "log_tau1"),
                "position": float(np.median([(math.log(r["profiles"]["log_tau1"]["truth"]) - lo_hi[0])
                                             / (lo_hi[1] - lo_hi[0]) for r in rows])),
            }
            narrow = [r for r in rows if rules.is_bounded(r, "log_tau1")
                      and math.log(np.divide(*rules.clipped_interval(r, "log_tau1")[::-1])) <= 0.5]
            block["narrow"], block["narrowCover"] = len(narrow), sum(rules.covers(r, "log_tau1") for r in narrow)
            blocks[label, cv] = block
            prefix = f"ws{label}{word}"
            for field, fmt in (("cover", "d"), ("coverP", "d"), ("bounded", "d"), ("narrow", "d"),
                               ("narrowCover", "d"), ("shift", ".2f"), ("width", ".2f"), ("rho", ".2f")):
                macros[prefix + field[0].upper() + field[1:]] = format(block[field], fmt)
    macros["wsFitPosPct"] = format(100 * blocks["Fit", 0.0]["position"], ".0f")
    macros["wsRandPosPct"] = format(100 * blocks["RandA", 0.0]["position"], ".0f")
    return macros, raw, blocks


def statements(raw: dict, blocks: dict) -> dict[str, bool]:
    """Every sentence of the paper whose truth depends on a rerun, as a test on the numbers."""
    random = ("RandA", "RandB")
    return {
        "no interval for k_e or k_a is bounded at any box width shown": raw["rate_bounded"] == 0,
        "the S_I bounded count depends on the box": len(set(raw["si_bounded"].values())) > 1,
        "with exact inputs tau1 coverage is close to nominal on both random-truth seeds":
            all(blocks[t, 0.0]["cover"] >= 41 for t in random),
        "coverage falls with logging error on every truth set":
            all(blocks[t, 0.75]["cover"] < blocks[t, 0.0]["cover"] for t in TRUTHS),
        "estimates shift toward larger tau1 as the error grows":
            all(blocks[t, 0.75]["shift"] > blocks[t, 0.0]["shift"] for t in TRUTHS),
        "intervals do not widen (random truths)": all(blocks[t, 0.75]["width"] <= blocks[t, 0.0]["width"] for t in random),
        "bounded counts rise from CV 0 to 0.25 with fitted truths":
            blocks["Fit", 0.25]["bounded"] > blocks["Fit", 0.0]["bounded"],
        "bounded counts fall from CV 0 to 0.75 with random truths":
            all(blocks[t, 0.75]["bounded"] < blocks[t, 0.0]["bounded"] for t in random),
        "subject ordering degrades less than coverage (rank correlation at least 0.6 at CV 0.75)":
            all(blocks[t, 0.75]["rho"] >= 0.6 for t in random),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args(argv)
    macros, raw, blocks = numbers(args.draft)
    failed = [text for text, holds in statements(raw, blocks).items() if not holds]
    for text in failed:
        print(f"STATEMENT NO LONGER HOLDS: {text}")
    if failed and not args.draft:
        raise SystemExit("rewrite the sentences above before building the final paper")
    OUT.write_text(json.dumps({"draft": args.draft, "boxes": raw["boxes"], "macros": macros}, indent=2), encoding="utf-8")
    print(f"{'DRAFT' if args.draft else 'FINAL'}: wrote {len(macros)} numbers; {len(failed)} statements failing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

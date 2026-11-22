"""Build the single-file AAAI submission from `paper_src.tex`.

The AAAI kit requires one .tex file with no \\input, so this script inlines every macro the paper uses (the generated
result macros of the main manuscript and the numbers of the workshop figures), copies the style files, the figures and a
bibliography made of verified entries, and compiles with pdflatex and bibtex.

Run:  python workshop_caleai/build.py            # writes workshop_caleai/build/Anonymous.tex and .pdf
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
KIT = HERE / "AuthorKit27"
NAME = "Anonymous"
KEYS = ("zeevi2015personalized", "berry2020human", "bergman1979", "dallaman2007", "raue2009structural",
        "yamaoka1978statistical", "cutler1978theory", "das2025cgmacros", "zhao2023shanghai", "hall2018glucotypes",
        "cobelli1980parameter", "miao2011identifiability", "wieland2021structural", "gutenkunst2007universally",
        "erdos2024leveraging", "eisenberg2014determining", "boxenbaum1998flipflop")
DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
          "8": "Eight", "9": "Nine"}
FORMAT = {"WindowMin": ".0f", "MealGrams": ".0f", "PairTauOne": ".0f", "IaucSurfaceRange": ".0f", "IaucRangeOverNoise": ".2f",
          "IaucNoiseSd": ".0f", "TraceNoiseSd": ".1f", "PairIaucSpreadPct": ".1f", "PairPeakTimeSpread": ".0f",
          "PairPeakGlucoseSpread": ".1f", "PairTraceMaxDiff": ".1f", "PairPA": ".0f", "PairPB": ".0f", "PairPC": ".0f",
          "PvpRhoIauc": ".2f", "PvpRhoPeak": ".2f", "PvpRhoTrace": ".2f"}


def word(key: str) -> str:
    """Macro names contain letters only: `AbsorbPinned07` becomes `AbsorbPinnedZeroSeven`."""
    return re.sub(r"\d", lambda m: DIGITS[m.group()], key)


def figure_macros() -> dict[str, str]:
    numbers = json.loads((HERE / "figure_numbers.json").read_text(encoding="utf-8"))
    out = {}
    for key, value in numbers.items():
        if key.startswith("Absorb"):
            text = f"{value:.2f}" if key.startswith("AbsorbMedian") else f"{int(value)}"
        elif key.startswith(("SweepAll", "RealIauc", "SweepCalBounded")):
            text = f"{float(value):.0f}"
        elif key.startswith(("Seed", "Sys", "Final")):
            text = f"{float(value):.0f}"
        elif key.startswith("Chrono"):
            text = f"${float(value):.0f}$"
        elif key.startswith(("SweepSigma",)):
            text = f"{float(value):.1f}"
        elif key.startswith("Eig"):
            mantissa, exponent = f"{float(value):.1e}".split("e")
            text = f"${mantissa}\\times10^{{{int(exponent)}}}$"
        else:
            text = format(float(value), FORMAT.get(key, ".2f"))
        out["fig" + word(key)] = text
    return out


def result_macros() -> dict[str, str]:
    text = (ROOT / "paper" / "results_macros.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(res[A-Za-z]+)\}\{(.*?)\}\s*(?:%.*)?$", text, re.M)}


def flatten(src: str) -> str:
    used = sorted(set(re.findall(r"\\((?:res|fig)[A-Za-z]+)", src)))
    available = {**result_macros(), **figure_macros()}
    missing = [name for name in used if name not in available]
    if missing:
        raise SystemExit("macros missing: " + ", ".join(missing))
    lines = ["% Numbers of the paper, generated from the stored results (inlined: the kit allows a single source file)."]
    lines += [f"\\newcommand{{\\{name}}}{{{available[name]}}}" for name in used]
    return src.replace("%%MACROS%%", "\n".join(lines))


def bibliography() -> str:
    source = (ROOT / "paper" / "references.bib").read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=@\w+\s*\{)", source)
    chosen = [b for b in blocks if any(re.match(rf"@\w+\s*\{{\s*{k}\s*,", b) for k in KEYS)]
    found = {re.match(r"@\w+\s*\{\s*([^,\s]+)", b).group(1) for b in chosen}
    if found != set(KEYS):
        raise SystemExit("bibliography entries missing: " + ", ".join(sorted(set(KEYS) - found)))
    return "\n".join(chosen)


def main() -> int:
    if BUILD.exists():
        shutil.rmtree(BUILD, ignore_errors=True)       # a viewer or shell may hold the folder open on Windows
    (BUILD / "figures").mkdir(parents=True, exist_ok=True)
    for name in ("aaai2027.sty", "aaai2027.bst"):
        shutil.copy(KIT / name, BUILD / name)
    for pdf in (HERE / "figures").glob("fig_*.pdf"):
        shutil.copy(pdf, BUILD / "figures" / pdf.name)
    (BUILD / "caleai.bib").write_text(bibliography(), encoding="utf-8")
    (BUILD / f"{NAME}.tex").write_text(flatten((HERE / "paper_src.tex").read_text(encoding="utf-8")), encoding="utf-8")
    for command in (["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"], ["bibtex", NAME],
                    ["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"], ["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"]):
        result = subprocess.run(command, cwd=BUILD, capture_output=True, text=True)
        if result.returncode and command[0] == "pdflatex":
            print(result.stdout[-2500:])
            return 1
    print("built", BUILD / f"{NAME}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())

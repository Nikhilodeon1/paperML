"""Prose lint for the manuscript: machine tells, em dashes, announcing sentences, rhythm.

The word list and the rules come from the `tmlr-paper-voice` skill. Hard findings fail the build; the rhythm
numbers are printed for the author.

Run:  python -m evaluation.lint_prose
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from evaluation.results_io import ROOT

PAPER = ROOT / "paper"
MAIN = ["sec_abstract.tex", "sec_intro.tex", "sec_related.tex", "sec_setting.tex", "sec_theory.tex",
        "sec_results.tex", "sec_discussion.tex"]
TELLS = ["delve", "crucial", "pivotal", "landscape", "tapestry", "underscore", "leverage", "seamless",
         "comprehensive", "notably", "it is important to note", "it is worth noting", "in today's", "plays a key role",
         "serves as", "a testament to", "moreover", "furthermore", "not only", "additionally", "importantly",
         "remarkably", "intriguing", "shed light", "paves the way", "in the realm of", "dramatically",
         "undoubtedly", "clearly demonstrates"]
ANNOUNCE = [r"\bin this section,? we\b", r"\bthis section (states|gives|describes|presents)\b", r"\bthe section is organi[sz]ed\b",
            r"\bwe now (turn|describe|present)\b", r"\bthe remainder of this paper\b"]
EM_DASH = re.compile(r"---|\u2014")
NOVEL = re.compile(r"\bnovel\b", re.I)
ROBUST = re.compile(r"\brobust(ly|ness)?\b", re.I)


def prose(text: str) -> str:
    """Strip comments, math, commands and environments that are not prose."""
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)
    text = re.sub(r"\\begin\{(figure|table|equation|align|tabularx|tabular|enumerate|itemize|proposition|definition)\*?\}.*?\\end\{\1\*?\}",
                  " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " X ", text)
    text = re.sub(r"\\(cite[a-z]*|ref|eqref|label)\*?(\[[^\]]*\])*\{[^}]*\}", " ", text)
    text = re.sub(r"\\res[A-Za-z]+", " 0 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    return re.sub(r"[{}~]", " ", text)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9])", re.sub(r"\s+", " ", text))
    return [p.strip() for p in parts if len(p.split()) >= 3]


def lint(files: list[str]) -> tuple[list[str], list[str]]:
    hard, info = [], []
    for name in files:
        raw = (PAPER / name).read_text(encoding="utf-8")
        for number, line in enumerate(raw.splitlines(), 1):
            body = re.sub(r"(?<!\\)%.*$", "", line)
            body = re.sub(r"\\(label|ref|eqref|includegraphics|safefigure|input)\*?(\[[^\]]*\])?\{[^}]*\}", "", body)
            low = body.lower()
            for word in TELLS:
                if word in low:
                    hard.append(f"{name}:{number}: '{word}'")
            if EM_DASH.search(body):
                hard.append(f"{name}:{number}: em dash")
            if NOVEL.search(body):
                hard.append(f"{name}:{number}: 'novel'")
            if ROBUST.search(body) and "statistic" not in low:
                hard.append(f"{name}:{number}: 'robust' (use only in the statistical sense)")
            for pattern in ANNOUNCE:
                if re.search(pattern, low):
                    hard.append(f"{name}:{number}: announcing sentence")
        text = prose(raw)
        lengths = [len(s.split()) for s in sentences(text)]
        if lengths:
            info.append(f"{name:18} {sum(len(s.split()) for s in sentences(text)):5d} words, {len(lengths):3d} sentences, "
                        f"mean {statistics.mean(lengths):4.1f}, median {statistics.median(lengths):4.1f}, "
                        f"max {max(lengths):3d}, share over 40 words {sum(l > 40 for l in lengths) / len(lengths):4.0%}")
    return hard, info


def main() -> int:
    hard, info = lint(MAIN)
    for line in info:
        print("    " + line)
    for line in hard:
        print("    " + line)
    print(f"    {len(hard)} prose problems")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())

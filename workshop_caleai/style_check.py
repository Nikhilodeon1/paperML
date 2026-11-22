"""Small style audit for the workshop source, after the checks of the ml-paper-writer skill.

Errors: dashes, banned words, contractions, unresolved markers other than the stated AI-use placeholder.
Warnings: density of Tier B words, participial tails, stock transitions, lists of three, candid openers,
not-X-but-Y, summary sentences, and the rate of first-person plural.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HARD = (r"\b(?:delv(?:e|es|ed|ing)|tapestry|meticulous(?:ly)?|groundbreaking|transformative|paramount|unparalleled|invaluable"
        r"|multifaceted|testament|embark\w*|endeavou?rs?|bolster\w*|plethora|burgeoning|commendable|noteworthy|myriad|pivotal"
        r"|underscor\w+|intricate\w*|seamless\w*|holistic\w*|realm|unlock\w*|nuanced?|harness\w*|cornerstone|cutting-edge"
        r"|game-chang\w+|ever-evolving|revolutioni[sz]\w+)\b|pav(?:e|es|ed|ing) the way|at the forefront|in today's|rapidly evolving")
WATCH = set("""capabilities enabling enhance enhances enhanced enhancing comprehensive remarkable remarkably ensuring offers integrates
integrating highlighting advancements addressing struggle struggles emerged primarily achieving preserving maintaining demonstrating
inherent thereby insights diverse extensive effectively exhibit exhibits potential tailored foster fosters fostering showcase
showcases showcasing vital crucial additionally notably substantial employs facilitate facilitates facilitating""".split())
PART = (r",\s+(?:thereby |thus |while )?(?:enabling|allowing|ensuring|highlighting|demonstrating|providing|offering|making|leading"
        r"|resulting|reflecting|suggesting|indicating|underscoring|showing|achieving|improving|reducing|enhancing|facilitating"
        r"|capturing|revealing|paving|opening|yielding|supporting|promoting)\b")
TRANS = r"(?:^|(?<=[.!?] ))(?:Moreover|Furthermore|Additionally|In addition|Notably|Importantly|Interestingly|Crucially|Overall|Consequently|Hence|Thus)\b"
ANTI = r"\bnot (?:just|merely|simply)\b|\bnot only\b[^.;]{0,120}?\bbut (?:also|rather)\b"
SUMMARY = r"^(?:Overall|In summary|In conclusion|Taken together|Together|Collectively)\b"


def prose(text: str) -> str:
    text = text.split("\\begin{document}", 1)[-1]
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    text = re.sub(r"\\begin\{(?:table\*?|figure\*?|enumerate)\}.*?\\end\{(?:table\*?|figure\*?|enumerate)\}", lambda m: m.group(0)
                  if "enumerate" in m.group(0) else " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " X ", text)
    text = re.sub(r"\\(?:cite[a-z]*|ref|label|bibliography)\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:res|fig)[A-Za-z]+\{\}|\\(?:res|fig)[A-Za-z]+", " 1 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?|[{}\\~&]", " ", text)
    return text


def main(path: str) -> int:
    raw = Path(path).read_text(encoding="utf-8")
    body = prose(raw)
    errors = 0
    checks = [("dash", r"—|(?<=\s)–(?=\s)|---|(?<=\s)--(?=\s)", 0), ("banned word", HARD, re.I),
              ("contraction", r"\b\w+(?:n't|'re|'ll|'ve|'d|'m)\b|\b(?:it|that|there|here|what|let)'s\b", re.I),
              ("marker", r"\[CITE-NEEDED[^\]]*\]|\bTODO\b|\bFIXME\b|\?\?", 0)]
    for tag, pattern, flags in checks:
        for m in re.finditer(pattern, body, flags):
            errors += 1
            print(f"ERROR [{tag}] ...{body[max(0, m.start() - 30):m.end() + 30].strip()}...")
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", body.lower())
    n = max(len(words), 1)
    k = 1000.0 / n
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body)) if len(s.split()) >= 3]
    rates = {
        "we/our per 1k (human median 21, flag below 11)": k * len(re.findall(r"\b(?:we|our|ours)\b", body, re.I)),
        "Tier B words % (flag above 0.55)": 100.0 * sum(w in WATCH for w in words) / n,
        "participial tails per 1k (flag above 2.5)": k * len(re.findall(PART, body)),
        "stock transitions % of sentences (flag above 10)": 100.0 * sum(bool(re.match(TRANS, s)) for s in sentences) / max(len(sentences), 1),
        "lists of three per 1k (flag above 3.0)": k * len(re.findall(r"\b\w+, \w+,? (?:and|or) \w+\b", body)),
        "not-X-but-Y per 1k (flag above 0.6)": k * len(re.findall(ANTI, body, re.I)),
        "summary sentences per 1k (flag above 1.0)": k * sum(bool(re.match(SUMMARY, s)) for s in sentences),
    }
    print(f"{n} words, {len(sentences)} sentences, mean sentence length {sum(len(s.split()) for s in sentences) / max(len(sentences), 1):.1f}")
    for name, value in rates.items():
        print(f"  {name}: {value:.2f}")
    print("placeholders left:", re.findall(r"\[AUTHOR:[^\]]*\]", raw))
    print(f"{errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "paper_src.tex")))

"""Freeze the result files the paper is built from.

Writes `results/index.json`: the list of every result file that contributes a number to the paper,
each with its own SHA-256, plus one SHA-256 over all of them in a fixed order, plus the collected
`macros` declarations. The paper is then generated from that index and nothing else, so a figure or
a sentence cannot quietly disagree with the result file behind it.

A macro name declared twice with different values is an error, not a last-writer-wins: two
analyses claiming the same paper-facing scalar means one of them is stale.

Run:  python -m evaluation.freeze_results            # freeze everything under results/
      python -m evaluation.freeze_results --check    # verify the existing index still matches
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from evaluation.results_io import RESULTS, ROOT, git_commit

INDEX = RESULTS / "index.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict:
    """Walk `results/`, hash every result file, and gather the macro declarations."""
    files: list[dict] = []
    macros: dict[str, dict] = {}
    collisions: list[str] = []

    for path in sorted(RESULTS.rglob("*.json")):
        if path == INDEX or path.name.endswith(".partial"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            files.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                          "sha256": _sha256(path), "unreadable": str(exc)})
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        files.append({
            "path": rel,
            "sha256": _sha256(path),
            "analysis_id": doc.get("analysis_id"),
            "config_hash": doc.get("config_hash"),
            "unit": doc.get("unit"),
            "commit": doc.get("commit"),
        })
        declared = (doc.get("payload") or {}).get("macros") or {}
        for name, spec in declared.items():
            entry = spec if isinstance(spec, dict) else {"value": spec}
            entry = {**entry, "source": rel, "analysis_id": doc.get("analysis_id")}
            if name in macros and macros[name].get("value") != entry.get("value"):
                collisions.append(
                    f"{name}: {macros[name]['source']} says {macros[name].get('value')!r}, "
                    f"{rel} says {entry.get('value')!r}")
            macros[name] = entry

    combined = hashlib.sha256()
    for item in files:
        combined.update(item["path"].encode("utf-8"))
        combined.update(item["sha256"].encode("utf-8"))

    return {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "n_files": len(files),
        "sha256_over_all": combined.hexdigest(),
        "files": files,
        "macros": dict(sorted(macros.items())),
        "collisions": collisions,
    }


def freeze() -> dict:
    index = collect()
    if index["collisions"]:
        raise SystemExit("Macro name declared twice with different values:\n  "
                         + "\n  ".join(index["collisions"]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def check() -> int:
    """Verify every file named in the index still has the hash recorded there."""
    if not INDEX.exists():
        print("No results/index.json; run the freeze first.", file=sys.stderr)
        return 1
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    bad = []
    for item in index["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            bad.append(f"missing: {item['path']}")
        elif _sha256(path) != item["sha256"]:
            bad.append(f"changed: {item['path']}")
    current = collect()
    if current["sha256_over_all"] != index["sha256_over_all"]:
        bad.append("the set of result files has changed since the freeze")
    for line in bad:
        print(line, file=sys.stderr)
    print(f"{len(index['files'])} files checked, {len(bad)} problems")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify instead of writing")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check())
    index = freeze()
    print(f"froze {index['n_files']} result files, {len(index['macros'])} macros")
    print(f"combined sha256 {index['sha256_over_all']}")


if __name__ == "__main__":
    main()

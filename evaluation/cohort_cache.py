"""Parsed cohorts, committed to the repository, so a machine with no datasets can still run.

The three archives total about 650 MB, CGMacros alone is a 627 MB zip, and parsing them needs
`pandas` plus two Excel engines. None of that belongs on a rented compute node, and none of it needs
to be there: everything the analyses touch is a few thousand numbers per subject. Parsing the
cohorts once and committing the result -- about 3 MB gzipped -- means a pod can `git clone` and run.

What is cached is the OUTPUT of the existing loaders, verbatim, not a reinterpretation of the raw
files. `evaluation/cohort_data.py` then normalizes cache or live parse through the same code path, so
a cached run and a from-archive run cannot diverge.

Integrity is checked, not assumed. `manifest.json` records a SHA-256 of each cache file together
with the subject and meal counts and the sizes of the source archives it was built from. A cache
whose hash does not match is refused rather than used, because a silently corrupted cohort would
produce a complete set of plausible, wrong results.

Run:  python -m evaluation.cohort_cache build --all     # on a machine that has the datasets
      python -m evaluation.cohort_cache verify          # anywhere, including a fresh clone
      python -m evaluation.cohort_cache info
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cohort_cache"
MANIFEST = CACHE_DIR / "manifest.json"

COHORTS = ("cgmacros", "hall", "shanghai")


# --- parsing (only ever runs where the datasets are) ---------------------------------------------

def _parse_cgmacros(min_meals: int = 10) -> dict:
    from evaluation.cgmacros import load_bio, subjects, _profile
    from evaluation.snpe_kfold import _meal_records

    bio = load_bio()
    out = []
    for sid, meals in subjects():
        records = _meal_records(meals)
        entry = bio.get(sid)
        profile = _profile(entry)
        if profile and len(records) >= min_meals:
            # The fasting labs travel with the cache because Phase 5 needs them and cannot rebuild
            # it: the leakage control is a partial correlation against fasting glucose, and the
            # clinical correlations are against HbA1c. Reparsing on a machine without the archive is
            # exactly what the cache exists to avoid.
            clinical = {k: (entry or {}).get(k) for k in
                        ("hba1c", "fasting_glucose", "fasting_insulin", "homa_ir", "bmi", "status")}
            out.append({"subject_id": sid, "records": records, "profile": profile,
                        "clinical": {k: v for k, v in clinical.items() if v is not None}})
    out.sort(key=lambda s: s["subject_id"])
    return {"subjects": out, "sampling_min": 5.0, "min_meals": min_meals,
            "window_min": 180.0, "pre_meal_min": 30.0,
            "carbohydrate_source": "photographed diet log, nutritionist coded"}


def _parse_hall(min_meals: int = 3) -> dict:
    from evaluation.hall_loader import load_hall

    out = sorted(load_hall(min_meals=min_meals), key=lambda s: s["subject_id"])
    return {"subjects": out, "sampling_min": 5.0, "min_meals": min_meals,
            # Hall scores a FIXED 0-145 min window, not the 0-180 min window CGMacros uses; any
            # cross-cohort comparison of iAUC has to say so.
            "window_min": 145.0, "pre_meal_min": 30.0,
            "carbohydrate_source": "standardized meals, known by design"}


def _parse_shanghai(min_meals: int = 3) -> dict:
    from evaluation.shanghai_loader import load_shanghai_t2dm

    out = sorted(load_shanghai_t2dm(min_meals=min_meals), key=lambda s: s["subject_id"])
    return {"subjects": out, "sampling_min": 15.0, "min_meals": min_meals,
            "window_min": 180.0, "pre_meal_min": 30.0,
            # The stored curve is on a 5-min grid, but the SENSOR sampled every 15 min and the grid
            # was filled by linear interpolation. A trace objective on this cohort must not treat
            # the interpolated points as independent observations.
            "grid_is_interpolated": True,
            "carbohydrate_source": "free-text dietary record, carbohydrates estimated"}


_PARSERS = {"cgmacros": _parse_cgmacros, "hall": _parse_hall, "shanghai": _parse_shanghai}

# Which archive each cohort came from, recorded so a changed dataset is visible in the manifest.
_SOURCES = {
    "cgmacros": lambda: [("cgmacros", "1.0.0", "CGMacros_dateshifted365.zip")],
    "hall": lambda: [("hall2018",)],
    "shanghai": lambda: [("diabetes_datasets",)],
}


def _source_fingerprint(cohort: str) -> dict:
    """Size and modification time of the source archive, for provenance only.

    Deliberately NOT part of the cache key: the key would then be unavailable on a machine without
    the archive, which is the whole case this module exists for.
    """
    import data_paths

    out = {}
    for parts in _SOURCES[cohort]():
        try:
            path = data_paths.dataset(*parts)
            if path.is_file():
                stat = path.stat()
                out["/".join(parts)] = {"bytes": stat.st_size, "mtime": int(stat.st_mtime)}
            elif path.is_dir():
                files = [p for p in path.rglob("*") if p.is_file()]
                out["/".join(parts)] = {"files": len(files),
                                        "bytes": sum(p.stat().st_size for p in files)}
            else:
                out["/".join(parts)] = {"missing": True}
        except FileNotFoundError:
            out["/".join(parts)] = {"unresolved": True}
    return out


# --- cache files ---------------------------------------------------------------------------------

def cache_path(cohort: str) -> Path:
    return CACHE_DIR / f"{cohort}.json.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _counts(payload: dict) -> dict:
    subjects = payload["subjects"]
    meals = [len(s.get("records") or s.get("meals") or []) for s in subjects]
    return {"n_subjects": len(subjects), "n_meals": int(sum(meals)),
            "meals_min": min(meals) if meals else 0, "meals_max": max(meals) if meals else 0}


def build(cohort: str) -> dict:
    """Parse `cohort` from its archive and write the cache plus its manifest entry."""
    if cohort not in _PARSERS:
        raise SystemExit(f"unknown cohort {cohort!r}; known: {', '.join(COHORTS)}")
    started = time.perf_counter()
    payload = _PARSERS[cohort]()
    payload["cohort"] = cohort
    payload["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload["source"] = _source_fingerprint(cohort)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(cohort)
    temporary = path.with_suffix(".gz.partial")
    # mtime=0 so the same parse produces byte-identical output on every machine; otherwise the gzip
    # header timestamp would change the SHA-256 on every rebuild and churn the repository.
    with gzip.GzipFile(filename="", mode="wb", fileobj=temporary.open("wb"), mtime=0) as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=float).encode("utf-8"))
    temporary.replace(path)

    entry = {"sha256": _sha256(path), "bytes": path.stat().st_size,
             "built_at": payload["built_at"], "source": payload["source"],
             "parse_seconds": round(time.perf_counter() - started, 1), **_counts(payload)}
    manifest = read_manifest()
    manifest[cohort] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return entry


def verify(cohort: str) -> tuple[bool, str]:
    """Whether the cache file matches its manifest entry, and why not when it does not."""
    manifest = read_manifest()
    if cohort not in manifest:
        return False, "no manifest entry"
    path = cache_path(cohort)
    if not path.exists():
        return False, f"{path.name} is missing"
    actual = _sha256(path)
    if actual != manifest[cohort]["sha256"]:
        return False, (f"{path.name} hash {actual[:12]} does not match the manifest "
                       f"{manifest[cohort]['sha256'][:12]}")
    return True, "ok"


def load(cohort: str, allow_parse: bool = True) -> dict:
    """The parsed cohort: from the verified cache if present, otherwise from its archive.

    Order matters. The cache is preferred because it is the thing that is committed, reviewed and
    hash-checked; re-parsing is the fallback for the machine that builds it. A cache that exists but
    fails its hash is an error either way -- it is never silently bypassed by re-parsing, because
    then a corrupt cache on a pod would quietly produce different numbers from the same commit.
    """
    path = cache_path(cohort)
    if path.exists():
        ok, why = verify(cohort)
        if not ok:
            raise RuntimeError(
                f"cohort cache for {cohort!r} failed verification: {why}. Rebuild it on a machine "
                f"with the datasets (python -m evaluation.cohort_cache build --cohort {cohort}).")
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))

    if not allow_parse:
        raise FileNotFoundError(f"no cohort cache at {path} and parsing was not allowed")

    import data_paths
    if not data_paths.available():
        raise FileNotFoundError(
            f"no cohort cache at {path} and no datasets to parse. On a machine without the "
            f"datasets, the cache must be committed: build it where the data lives with "
            f"`python -m evaluation.cohort_cache build --all` and commit cohort_cache/.")
    return _PARSERS[cohort]()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["build", "verify", "info"])
    ap.add_argument("--cohort", choices=COHORTS, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cohorts = list(COHORTS) if (args.all or not args.cohort) else [args.cohort]

    if args.action == "build":
        for cohort in cohorts:
            try:
                entry = build(cohort)
            except Exception as exc:                       # a missing cohort must not stop the rest
                print(f"  {cohort:10} FAILED: {type(exc).__name__}: {exc}")
                continue
            print(f"  {cohort:10} {entry['n_subjects']:>4} subjects  {entry['n_meals']:>5} meals  "
                  f"{entry['bytes'] / 1024:>7.0f} KB  {entry['parse_seconds']:>6.1f}s parse  "
                  f"sha {entry['sha256'][:12]}")
        return

    if args.action == "verify":
        bad = []
        for cohort in cohorts:
            ok, why = verify(cohort)
            print(f"  {cohort:10} {'ok' if ok else 'FAILED: ' + why}")
            if not ok:
                bad.append(cohort)
        sys.exit(1 if bad else 0)

    manifest = read_manifest()
    if not manifest:
        print("no manifest; nothing built yet")
        return
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

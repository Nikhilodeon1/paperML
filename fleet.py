"""Split the unfinished reruns over N Colab pods and run pod W's share. Lives outside the pod's checkout.

The split is computed only from the saves of the earlier three-pod run (stage2_pod*_results.tgz, stage2_save_*.tgz),
which nothing writes to any more. So every pod computes the same split, whenever it starts. Compare PLAN ID across pods.
Pod saves are named fleet_w<W>of<N>_results.tgz, go to Drive every 10 minutes and after each sweep, and are restored
when the same pod is started again.

Run:  python fleet.py --pod W --of N [--workers k] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(os.environ.get("FLEET_REPO", "/content/paperML"))
DRIVE = Path(os.environ.get("FLEET_DRIVE", "/content/drive/MyDrive"))
SNAPSHOT = Path(os.environ.get("FLEET_SNAPSHOT", "/content/snapshot"))
SAVE_EVERY = 600
# what the workshop paper needs runs first on every pod
FIRST = ["real", "rand0.0_s0", "rand0.0_s1", "rand0.75_s0", "rand0.75_s1", "rand0.25_s0", "rand0.25_s1",
         "fit0.0_s0", "fit0.25_s0", "fit0.75_s0", "iauc_box1", "iauc_box2", "iauc_box4"]


def extract(archive: Path, into: Path) -> None:
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("results/")
                   and not m.name.endswith((".claim", ".heartbeat", ".partial", ".failed.json", "STOP"))]
        tar.extractall(into, members=members, filter="data")


def old_saves() -> list[Path]:
    return sorted(set(DRIVE.glob("stage2_pod*_results.tgz")) | set(DRIVE.glob("stage2_save_*.tgz")))


def make_plan(pods: int):
    """Read the frozen snapshot and return (per-pod sweep names, sweep info, plan id)."""
    from evaluation import stage2_queue as queue
    from evaluation.results_io import git_commit, unit_done
    from evaluation.runner import _overrides, resolve_config

    def key(config, unit) -> str:
        return json.dumps(config, sort_keys=True) + "|" + unit

    order = []                                      # the earlier priority: the three queues interleaved
    for position in range(max(map(len, queue.PODS.values()))):
        order += [queue.PODS[p][position] for p in sorted(queue.PODS) if position < len(queue.PODS[p])]
    sweeps = queue.all_sweeps()
    assert set(order) == set(sweeps), "queue and sweep registry disagree"

    info, recognised = {}, set()
    for name in order:
        module_name, config = sweeps[name]
        module = importlib.import_module(module_name)
        args = queue.runner_args(module_name, config, 1)
        resolved = resolve_config(module, _overrides([args[i + 1] for i, a in enumerate(args) if a == "--set"]))
        units = module.units(resolved)
        done, seconds = [], []
        for unit in units:
            if unit_done(module.ANALYSIS_ID, resolved, unit):
                done.append(unit)
                directory = SNAPSHOT / "results" / module.ANALYSIS_ID
                for path in directory.glob(f"*/{unit}.json"):
                    document = json.loads(path.read_text(encoding="utf-8"))
                    if document["config"] == json.loads(json.dumps(resolved)):
                        seconds.append(document["payload"].get("wall_seconds", 0))
                        break
        recognised.update(key(json.loads(json.dumps(resolved)), unit) for unit in done)
        weight = 2.0 if config.get("objective", "").startswith("iauc") or name.startswith("dm_real") else 1.0
        info[name] = {"units": len(units), "left": len(units) - len(done), "weight": weight,
                      "median": statistics.median(seconds) if seconds else None}

    # check that finished work is recognised: results made under the new protocol must all count as done
    fresh, saved_units = collections.Counter(), set()
    for path in (SNAPSHOT / "results").glob("*/*/*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "payload" in document and document.get("config", {}).get("lbfgsb_restarts") == 2:
            fresh[document.get("commit", "?")] += 1
            saved_units.add(key(document["config"], path.stem))
    lost = saved_units - recognised
    print(f"this pod's code tag: {git_commit()}")
    print(f"saved result files under the new protocol, by code tag: {dict(fresh)}")
    print(f"saved units: {len(saved_units)}; recognised as finished: {len(saved_units & recognised)}")
    if lost:
        raise SystemExit(f"STOP: {len(lost)} saved units are not recognised and would be rerun. Send this output to Claude.")

    rate = [i["median"] / i["weight"] for i in info.values() if i["median"]]
    base = statistics.median(rate) if rate else 600.0
    for item in info.values():
        item["cost"] = item["left"] * (item["median"] or base * item["weight"])

    load = [0.0] * pods
    share: list[list[str]] = [[] for _ in range(pods)]
    for group in ([n for n in order if n in FIRST], [n for n in order if n not in FIRST]):
        for name in sorted((n for n in group if info[n]["left"]), key=lambda n: (-info[n]["cost"], n)):
            target = min(range(pods), key=lambda p: (load[p], p))
            share[target].append(name)
            load[target] += info[name]["cost"]
    rank = {name: (name not in FIRST, i) for i, name in enumerate(order)}
    share = [sorted(names, key=rank.__getitem__) for names in share]
    identity = json.dumps([share, {n: i["left"] for n, i in info.items()}], sort_keys=True)
    return share, info, hashlib.sha256(identity.encode()).hexdigest()[:8]


def save(target_name: str) -> None:
    """Archive this pod's results to Drive. Files that vanish while being read are skipped."""
    local = REPO.parent / target_name
    with tarfile.open(local, "w:gz") as tar:
        for folder in ("results/A5_profile", "results/A12_dalla_man", "logs"):
            for path in sorted((REPO / folder).rglob("*")):
                if path.is_file() and not path.name.endswith((".claim", ".heartbeat", ".partial")):
                    try:
                        tar.add(path, arcname=path.relative_to(REPO).as_posix())
                    except OSError:
                        pass
    try:
        shutil.copy(local, DRIVE / (target_name + ".tmp"))
        os.replace(DRIVE / (target_name + ".tmp"), DRIVE / target_name)
    except OSError as error:
        print(f"!!! could not save to Drive: {error}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.pod <= args.of:
        raise SystemExit("--pod must be between 1 and --of")
    if not old_saves():
        raise SystemExit(f"STOP: no earlier saves in {DRIVE}. Is Drive mounted, and is this the right Google account?")

    shutil.rmtree(SNAPSHOT, ignore_errors=True)
    SNAPSHOT.mkdir(parents=True)
    print("earlier saves used for the split:")
    for path in old_saves():
        print(f"  {path.name}  {path.stat().st_size} bytes")
        extract(path, SNAPSHOT)

    os.environ["HORIZON_RESULTS_DIR"] = str(SNAPSHOT / "results")        # the split reads the snapshot only
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    share, info, plan_id = make_plan(args.of)
    del os.environ["HORIZON_RESULTS_DIR"]

    left = sum(i["left"] for i in info.values())
    total = sum(i["cost"] for i in info.values()) or 1.0
    print(f"\nPLAN ID {plan_id}   <- must be identical on every pod")
    print(f"units left: {left} in {sum(bool(i['left']) for i in info.values())} sweeps")
    for number, names in enumerate(share, start=1):
        part = sum(info[n]["cost"] for n in names)
        print(f"  pod {number}: {len(names):2d} sweeps, {sum(info[n]['left'] for n in names):4d} units, "
              f"{100 * part / total:4.1f}% of estimated work, about {part / args.workers / 3600:.1f} h"
              f"{'   <- this pod' if number == args.pod else ''}")

    mine = f"fleet_w{args.pod}of{args.of}_results.tgz"
    for path in old_saves():
        extract(path, REPO)
    if (DRIVE / mine).exists():
        print(f"restoring this pod's own save {mine}")
        extract(DRIVE / mine, REPO)

    from evaluation import stage2_queue as queue
    sweeps = queue.all_sweeps()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    names, unfinished = share[args.pod - 1], []
    for number, name in enumerate(names, start=1):
        command = [sys.executable, "-m", "evaluation.runner"] + queue.runner_args(*sweeps[name], args.workers)
        print(f"=== {time.strftime('%F %T')} [{number}/{len(names)}] {name} ({info[name]['left']} units left)", flush=True)
        if args.dry_run:
            continue
        process = subprocess.Popen(command, cwd=REPO, env=environment)
        saved = time.time()
        while process.poll() is None:
            time.sleep(30)
            if time.time() - saved >= SAVE_EVERY:
                save(mine)
                saved = time.time()
        if process.returncode:
            unfinished.append(name)
            print(f"!!! {name} did not finish; continuing", flush=True)
        save(mine)
    save(mine)
    print(f"=== FLEET POD {args.pod} of {args.of} DONE; unfinished: {unfinished or 'none'}", flush=True)
    return 1 if unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())

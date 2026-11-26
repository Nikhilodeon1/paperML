"""Run one pod's share of the Stage 2 reruns. The sweep definitions live in `evaluation.manifest.stage2_sweeps`.

Each sweep is run with `--restart`, which clears claim and failure files left by an interrupted run (one runner per pod
at a time). Finished units are kept, so after a disconnect the same command resumes. A sweep that does not finish is
reported at the end and does not stop the queue.

Run:  python -m evaluation.stage2_queue --pod 1 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from evaluation import profile_lik
from evaluation.manifest import PERTURBED, stage2_sweeps
from evaluation.results_io import ROOT

# Three pods with about equal numbers of subject-profiles; within a pod, what the abstract rests on comes first.
PODS = {
    1: ["iauc_box1", "iauc_box4", "iauc_box2", "iauc_shanghai_box1", "iauc_shanghai_box2", "iauc_hall_box1",
        "iauc_hall_box2", "centroid_real", "centroid_rep", "iauc_synth_s0", "iauc_synth_s1"],
    2: [f"iauc_rep_s{seed}" for seed in range(5)] + ["iauc_rep_exact", "iauc_rep_noise0.5", "iauc_rep_noise0.25"]
       + [f"iauc_pert_{c}_{f:g}" for c in ("insulin_secretion", "x_gain") for f in (0.7, 1.3)],
    3: [f"iauc_pert_{c}_{f:g}" for c in PERTURBED[:3] for f in (0.7, 1.3)]
       + ["residual0.1_s0", "residual0_s0", "residual0.05_s0", "residual0.2_s0", "residual0.1_s1"],
}
UNIT_TIMEOUT = 7200


def runner_args(config: dict, workers: int) -> list[str]:
    args = ["evaluation.profile_lik", "--workers", str(workers), "--unit-timeout", str(UNIT_TIMEOUT), "--restart"]
    defaults = profile_lik.default_config()
    for key, value in config.items():
        if value != defaults.get(key):
            args += ["--set", f"{key}={json.dumps(value, sort_keys=True)}"]
    return args


def archive(pod: int, drive: str | None) -> Path:
    path = ROOT / f"stage2_pod{pod}_results.tgz"
    with tarfile.open(path, "w:gz") as tar:
        for name in ("results/A5_profile", "logs"):
            tar.add(ROOT / name, arcname=name)
    if drive and Path(drive).is_dir():
        shutil.copy(path, drive)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod", type=int, choices=sorted(PODS), required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--drive", default="/content/drive/MyDrive", help="copy the archive here if the folder exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    sweeps = stage2_sweeps()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    unfinished = []
    for number, name in enumerate(PODS[args.pod], start=1):
        command = [sys.executable, "-m", "evaluation.runner"] + runner_args(sweeps[name], args.workers)
        print(f"=== {time.strftime('%F %T')} [{number}/{len(PODS[args.pod])}] {name}", flush=True)
        if args.dry_run:
            print(" ".join(command[1:]))
            continue
        if subprocess.run(command, cwd=ROOT, env=environment).returncode:
            unfinished.append(name)
            print(f"!!! {name} did not finish; continuing", flush=True)
        archive(args.pod, args.drive)                      # after every sweep, so a lost pod loses one sweep at most
    if not args.dry_run:
        print(f"wrote {archive(args.pod, args.drive).name}")
    print(f"=== STAGE2 POD {args.pod} DONE; unfinished: {unfinished or 'none'}")
    return 1 if unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())

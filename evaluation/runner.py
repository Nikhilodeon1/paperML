"""Run a long analysis as resumable work units, and survive everything that can go wrong.

These jobs run for hours on rented hardware. The failure modes that matter are not exotic: the
machine is reclaimed, the network drops, one subject hits a solver edge case and hangs, a worker is
killed by the out-of-memory reaper, or somebody presses Ctrl-C at hour six. The design target is that
**none of those costs more than the units that were in flight.**

How it works. An analysis is a module exposing four names (see `CONTRACT` below): an id, a default
config, the list of work units, and a function that computes one unit. The parent process starts
worker subprocesses; every worker claims units from the shared filesystem, computes them, and writes
each result atomically through `evaluation.results_io`. There is no message passing, no queue server
and no shared memory, and that is deliberate -- the filesystem is the coordination mechanism, so:

* **Resume is free and exact.** A finished unit is a file. Re-running skips it. Resuming is just
  running the same command again; it is the default rather than a flag.
* **A crash cannot corrupt state.** Results are written to a temporary file and renamed, so a
  process killed mid-write leaves either the old file or the new one, never half of one.
* **Claims are crash-safe.** A worker claims a unit by creating a lock file with `O_EXCL`, so two
  workers cannot take the same unit even across machines sharing a directory. A claim whose owner
  died is reclaimed once it goes stale, so a killed worker's unit is retried rather than lost.
* **A hung unit cannot stall the run.** Workers touch a heartbeat file. The parent kills a worker
  whose heartbeat has stopped, releases its claim, and starts a replacement.
* **Failures are recorded, not retried forever.** A unit that raises is retried up to
  `--retries` times; after that a failure record is written so the unit is not attempted again, and
  it appears in the final summary and in the phase report instead of silently missing.
* **Ctrl-C is graceful.** The first interrupt stops new units from being dispatched and lets
  in-flight ones finish; the second stops immediately. Only this runner's own children are ever
  signalled -- never a bare `taskkill` on every python process, because the user's own backend runs
  on this machine.

Run:  python -m evaluation.runner evaluation.my_analysis --workers 8
      python -m evaluation.runner evaluation.my_analysis --gate 3      # cheap gate-check first
      python -m evaluation.runner evaluation.my_analysis --status      # what is done, what failed
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from evaluation.results_io import (
    LOGS, analysis_dir, config_hash, git_commit, log, save_result, unit_done, unit_path,
)

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = """
An analysis module must define:

    ANALYSIS_ID: str                      # the results/ subdirectory
    def default_config() -> dict          # everything that changes a number, and nothing that does not
    def units(config: dict) -> list[str]  # deterministic order, unique names
    def run_unit(unit: str, config: dict) -> dict    # the payload for that unit

`run_unit` must be a pure function of `(unit, config)` and must not depend on which worker runs it
or on how many workers there are.
"""

CLAIM_SUFFIX = ".claim"
FAILED_SUFFIX = ".failed.json"
STOP_FILE = "STOP"
DEFAULT_UNIT_TIMEOUT = 1800.0      # seconds a single unit may take before its worker is replaced
HEARTBEAT_GRACE = 120.0            # seconds of silence before a worker is presumed dead


# --- analysis loading ----------------------------------------------------------------------------

def load_analysis(module_name: str):
    module = importlib.import_module(module_name)
    missing = [name for name in ("ANALYSIS_ID", "default_config", "units", "run_unit")
               if not hasattr(module, name)]
    if missing:
        raise SystemExit(f"{module_name} is not a runnable analysis; missing {missing}.{CONTRACT}")
    return module


def resolve_config(module, overrides: dict | None = None) -> dict:
    config = dict(module.default_config())
    for key, value in (overrides or {}).items():
        if key not in config:
            raise SystemExit(f"{key!r} is not a config key of {module.ANALYSIS_ID}; "
                             f"known: {sorted(config)}")
        config[key] = value
    return config


# --- state on disk -------------------------------------------------------------------------------

def _safe(unit: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(unit))
    return keep or "unit"


def claim_path(analysis_id: str, config: dict, unit: str) -> Path:
    return analysis_dir(analysis_id, config) / (_safe(unit) + CLAIM_SUFFIX)


def failed_path(analysis_id: str, config: dict, unit: str) -> Path:
    return analysis_dir(analysis_id, config) / (_safe(unit) + FAILED_SUFFIX)


def stop_path(analysis_id: str, config: dict) -> Path:
    return analysis_dir(analysis_id, config) / STOP_FILE


def unit_failed(analysis_id: str, config: dict, unit: str) -> bool:
    return failed_path(analysis_id, config, unit).exists()


def try_claim(analysis_id: str, config: dict, unit: str, worker_id: int,
              stale_after: float) -> bool:
    """Take exclusive ownership of a unit, or return False.

    `O_EXCL` makes the create-if-absent atomic, which is what prevents two workers doing the same
    unit. A claim older than `stale_after` belonged to a worker that died; it is removed and the
    claim retried once, so a killed worker costs that one unit rather than leaving it undone.
    """
    path = claim_path(analysis_id, config, unit)
    for attempt in (0, 1):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue                       # vanished between the open and the stat: retry
            if attempt == 0 and age > stale_after:
                try:
                    path.unlink()
                except OSError:
                    return False
                continue
            return False
        else:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump({"worker": worker_id, "pid": os.getpid(), "claimed_at": time.time()}, fh)
            return True
    return False


def release_claim(analysis_id: str, config: dict, unit: str) -> None:
    try:
        claim_path(analysis_id, config, unit).unlink()
    except OSError:
        pass


def heartbeat_path(analysis_id: str, config: dict, worker_id: int) -> Path:
    return analysis_dir(analysis_id, config) / f"worker{worker_id}.heartbeat"


def survey(module, config: dict, all_units: list[str], known_done: set[str] | None = None) -> dict:
    """What is done, failed, claimed and pending right now.

    `known_done` is a set the caller keeps between polls. A unit already validated once is not
    re-read: `unit_done` parses the result file to reject a truncated write, which is the right check
    the first time and pure waste every two seconds thereafter. Without this the supervisor would
    re-parse thousands of JSON documents per poll and become the bottleneck it exists to monitor.
    """
    done, failed, claimed, pending = [], [], [], []
    for unit in all_units:
        if known_done is not None and unit in known_done:
            done.append(unit)
            continue
        if unit_done(module.ANALYSIS_ID, config, unit):
            done.append(unit)
            if known_done is not None:
                known_done.add(unit)
        elif unit_failed(module.ANALYSIS_ID, config, unit):
            failed.append(unit)
        elif claim_path(module.ANALYSIS_ID, config, unit).exists():
            claimed.append(unit)
        else:
            pending.append(unit)
    return {"done": done, "failed": failed, "claimed": claimed, "pending": pending,
            "total": len(all_units)}


# --- the worker ----------------------------------------------------------------------------------

def worker_main(module_name: str, config_file: Path, worker_id: int, retries: int,
                stale_after: float) -> int:
    """Claim and compute units until none are left or a stop is requested."""
    module = load_analysis(module_name)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    analysis_id = module.ANALYSIS_ID
    all_units = module.units(config)
    heartbeat = heartbeat_path(analysis_id, config, worker_id)
    stop = stop_path(analysis_id, config)
    completed = 0

    for unit in all_units:
        if stop.exists():
            log(analysis_id, f"worker {worker_id} stopping on request after {completed} units")
            break
        if unit_done(analysis_id, config, unit) or unit_failed(analysis_id, config, unit):
            continue
        if not try_claim(analysis_id, config, unit, worker_id, stale_after):
            continue

        heartbeat.write_text(f"{time.time()} {unit}", encoding="utf-8")
        attempt, last_error = 0, None
        while attempt <= retries:
            started = time.perf_counter()
            try:
                payload = module.run_unit(unit, config)
            except KeyboardInterrupt:
                release_claim(analysis_id, config, unit)
                return 130
            except Exception:
                last_error = traceback.format_exc()
                attempt += 1
                log(analysis_id, f"worker {worker_id} unit {unit} attempt {attempt} failed: "
                                 f"{last_error.strip().splitlines()[-1]}")
                heartbeat.write_text(f"{time.time()} {unit} retry{attempt}", encoding="utf-8")
                continue
            payload.setdefault("wall_seconds", round(time.perf_counter() - started, 2))
            save_result(analysis_id, payload, config, unit=unit, overwrite=True)
            completed += 1
            last_error = None
            log(analysis_id, f"worker {worker_id} finished {unit} in "
                             f"{payload['wall_seconds']:.1f}s")
            break

        if last_error is not None:
            # Recorded rather than retried forever: a unit that fails deterministically would
            # otherwise be picked up by every worker in turn and starve the rest of the run.
            failed_path(analysis_id, config, unit).write_text(
                json.dumps({"unit": unit, "attempts": attempt, "commit": git_commit(),
                            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "traceback": last_error}, indent=2), encoding="utf-8")
            log(analysis_id, f"worker {worker_id} gave up on {unit} after {attempt} attempts")

        release_claim(analysis_id, config, unit)
        heartbeat.write_text(f"{time.time()} idle", encoding="utf-8")

    return 0


# --- the parent ----------------------------------------------------------------------------------

class Supervisor:
    """Starts workers, watches their heartbeats, and reports progress."""

    def __init__(self, module, config: dict, workers: int, retries: int, unit_timeout: float,
                 module_name: str, heartbeat_grace: float = HEARTBEAT_GRACE):
        self.module = module
        self.module_name = module_name
        self.config = config
        self.workers = workers
        self.retries = retries
        self.unit_timeout = unit_timeout
        self.heartbeat_grace = heartbeat_grace
        self.analysis_id = module.ANALYSIS_ID
        self.known_done: set[str] = set()
        self.all_units = module.units(config)
        self.procs: dict[int, subprocess.Popen] = {}
        self.restarts: dict[int, int] = {}
        self.stopping = False
        self.hard_stop = False
        self.started = time.time()
        self.config_file = analysis_dir(self.analysis_id, config) / "config.json"
        self.config_file.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    # -- process management
    def _env(self) -> dict:
        """One thread per worker. Without this each worker spawns as many BLAS and XLA threads as
        the machine has cores, and `n` workers oversubscribe it by a factor of `n`."""
        return {
            **os.environ,
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
            # The amendment specifies this exact string. Recorded finding: XLA parses only tokens
            # beginning with --xla_, and silently ignores the rest, so "intra_op_parallelism_threads=1"
            # has no effect; what actually holds a worker to one thread is OMP_NUM_THREADS below. It is
            # set as specified because it is harmless and the brief asks for it.
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
            "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
        }

    def _spawn(self, worker_id: int) -> None:
        command = [sys.executable, "-m", "evaluation.runner", self.module_name,
                   "--worker", "--worker-id", str(worker_id),
                   "--config-file", str(self.config_file),
                   "--retries", str(self.retries),
                   "--stale-after", str(self.unit_timeout + self.heartbeat_grace)]
        self.procs[worker_id] = subprocess.Popen(command, cwd=ROOT, env=self._env(),
                                                 stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.PIPE, text=True)
        self.restarts[worker_id] = self.restarts.get(worker_id, 0)

    def _kill(self, worker_id: int, why: str) -> None:
        proc = self.procs.get(worker_id)
        if proc is None or proc.poll() is not None:
            return
        log(self.analysis_id, f"killing worker {worker_id}: {why}", echo=True)
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _heartbeat_age(self, worker_id: int) -> float | None:
        path = heartbeat_path(self.analysis_id, self.config, worker_id)
        try:
            return time.time() - path.stat().st_mtime
        except OSError:
            return None

    # -- signals
    def install_signals(self) -> None:
        def handler(signum, _frame):
            if self.stopping:
                self.hard_stop = True
                print("\nsecond interrupt: stopping now", flush=True)
                return
            self.stopping = True
            stop_path(self.analysis_id, self.config).write_text(
                f"requested at {time.strftime('%Y-%m-%dT%H:%M:%S%z')} by signal {signum}",
                encoding="utf-8")
            print("\ninterrupt: finishing the units in flight, then stopping. "
                  "Interrupt again to stop now. Re-run the same command to resume.", flush=True)

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    # -- progress
    def _write_progress(self, state: dict) -> None:
        elapsed = time.time() - self.started
        done = len(state["done"])
        remaining = state["total"] - done - len(state["failed"])
        rate = done / elapsed if elapsed > 0 and done else 0.0
        LOGS.mkdir(parents=True, exist_ok=True)
        payload = {
            "analysis_id": self.analysis_id,
            "config_hash": config_hash(self.config),
            "total": state["total"], "done": done, "failed": len(state["failed"]),
            "claimed": len(state["claimed"]), "pending": len(state["pending"]),
            "elapsed_seconds": round(elapsed, 1),
            "units_per_hour": round(rate * 3600.0, 2),
            "eta_hours": round(remaining / rate / 3600.0, 2) if rate > 0 else None,
            "workers_alive": sum(1 for p in self.procs.values() if p.poll() is None),
            "worker_restarts": sum(self.restarts.values()),
            "stopping": self.stopping,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        temporary = LOGS / f"{self.analysis_id}.progress.json.partial"
        target = LOGS / f"{self.analysis_id}.progress.json"
        # The progress file is bookkeeping. On Windows `os.replace` fails with a permission error if
        # anything (an indexer, a virus scanner, a user reading the file) has it open at that instant,
        # and an unguarded failure here once ended a supervisor in the middle of a run. A few quick
        # retries, then skip this update: the next poll writes a fresh one.
        for attempt in range(5):
            try:
                temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(temporary, target)
                return
            except OSError:
                time.sleep(0.2 * (attempt + 1))

    # -- main loop
    def run(self) -> dict:
        stop_path(self.analysis_id, self.config).unlink(missing_ok=True)
        state = survey(self.module, self.config, self.all_units, self.known_done)
        log(self.analysis_id, f"start: {state['total']} units, {len(state['done'])} already done, "
                              f"{len(state['failed'])} previously failed, {self.workers} workers",
            echo=True)
        if not state["pending"] and not state["claimed"]:
            self._write_progress(state)
            return state

        for worker_id in range(self.workers):
            self._spawn(worker_id)
        self.install_signals()

        last_print = 0.0
        try:
            while True:
                time.sleep(2.0)
                state = survey(self.module, self.config, self.all_units, self.known_done)
                self._write_progress(state)

                if self.hard_stop:
                    break
                for worker_id, proc in list(self.procs.items()):
                    if proc.poll() is not None:
                        continue
                    age = self._heartbeat_age(worker_id)
                    if age is not None and age > self.unit_timeout + self.heartbeat_grace:
                        # The unit is hung or the worker is wedged. Kill it; its claim goes stale and
                        # another worker picks the unit up, so the run continues either way.
                        self._kill(worker_id, f"no heartbeat for {age:.0f}s")

                alive = {w: p for w, p in self.procs.items() if p.poll() is None}
                if not self.stopping:
                    for worker_id, proc in list(self.procs.items()):
                        if proc.poll() is None:
                            continue
                        if proc.returncode != 0 and self.restarts.get(worker_id, 0) < 3:
                            stderr = (proc.stderr.read() if proc.stderr else "") or ""
                            log(self.analysis_id,
                                f"worker {worker_id} exited {proc.returncode}; restarting. "
                                f"{stderr.strip().splitlines()[-1] if stderr.strip() else ''}")
                            self.restarts[worker_id] = self.restarts.get(worker_id, 0) + 1
                            self._spawn(worker_id)
                            alive[worker_id] = self.procs[worker_id]

                if time.time() - last_print > 30.0:
                    last_print = time.time()
                    eta = self._eta_text(state)
                    print(f"  {len(state['done'])}/{state['total']} done, "
                          f"{len(state['failed'])} failed, {len(alive)} workers{eta}", flush=True)

                if not alive:
                    break
                if not state["pending"] and not state["claimed"]:
                    break
        finally:
            if self.hard_stop:
                for worker_id in list(self.procs):
                    self._kill(worker_id, "hard stop")
            for proc in self.procs.values():
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            stop_path(self.analysis_id, self.config).unlink(missing_ok=True)

        state = survey(self.module, self.config, self.all_units, self.known_done)
        self._write_progress(state)
        return state

    def _eta_text(self, state: dict) -> str:
        elapsed = time.time() - self.started
        done = len(state["done"])
        if not done or elapsed <= 0:
            return ""
        remaining = state["total"] - done - len(state["failed"])
        hours = remaining / (done / elapsed) / 3600.0
        return f", eta {hours:.1f}h"


# --- entry point ---------------------------------------------------------------------------------

def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module", help="analysis module, e.g. evaluation.fisher_full")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--unit-timeout", type=float, default=DEFAULT_UNIT_TIMEOUT)
    ap.add_argument("--heartbeat-grace", type=float, default=HEARTBEAT_GRACE,
                    help="extra seconds of worker silence tolerated beyond --unit-timeout")
    ap.add_argument("--limit", type=int, default=None, help="only the first N units")
    ap.add_argument("--gate", type=int, default=None,
                    help="gate check: run only the first N units, then stop")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key (JSON value)")
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--restart", action="store_true",
                    help="clear previous failures and stale claims before starting")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--config-file", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stale-after", type=float, default=DEFAULT_UNIT_TIMEOUT + HEARTBEAT_GRACE,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        sys.exit(worker_main(args.module, args.config_file, args.worker_id, args.retries,
                             args.stale_after))

    module = load_analysis(args.module)
    config = resolve_config(module, _overrides(args.set))
    limit = args.gate or args.limit
    if limit:
        config = {**config, "limit": limit} if "limit" in config else config
    all_units = module.units(config)
    if limit:
        all_units = all_units[:limit]
        # The unit list the workers compute must match the parent's, so a limited run is expressed
        # through the config (which is hashed) rather than by slicing in the parent only.
        if "limit" not in module.default_config():
            raise SystemExit(
                f"{args.module} has no 'limit' config key, so --limit/--gate cannot be applied "
                f"without the workers disagreeing with the parent about the unit list.")

    if args.status:
        state = survey(module, config, all_units)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in state.items()},
                         indent=2))
        if state["failed"]:
            print("failed units:", ", ".join(state["failed"][:20]))
        progress = LOGS / f"{module.ANALYSIS_ID}.progress.json"
        if progress.exists():
            print(progress.read_text(encoding="utf-8"))
        return

    if args.restart:
        directory = analysis_dir(module.ANALYSIS_ID, config)
        removed = 0
        for path in list(directory.glob("*" + FAILED_SUFFIX)) + list(
                directory.glob("*" + CLAIM_SUFFIX)):
            path.unlink(missing_ok=True)
            removed += 1
        print(f"cleared {removed} failure and claim files (results are kept)")

    started = time.time()
    state = Supervisor(module, config, args.workers, args.retries, args.unit_timeout,
                       args.module, heartbeat_grace=args.heartbeat_grace).run()
    elapsed = time.time() - started

    print("=" * 78)
    print(f"{module.ANALYSIS_ID}: {len(state['done'])}/{state['total']} done, "
          f"{len(state['failed'])} failed in {elapsed / 3600.0:.2f}h")
    if state["failed"]:
        print("  FAILED: " + ", ".join(state["failed"][:20]))
        print(f"  tracebacks in {analysis_dir(module.ANALYSIS_ID, config)}")
    if state["pending"]:
        print(f"  {len(state['pending'])} still pending — re-run the same command to resume")
    print("=" * 78)
    sys.exit(1 if state["failed"] else 0)


if __name__ == "__main__":
    main()

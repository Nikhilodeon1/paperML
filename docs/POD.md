# Running the analyses on a rented compute node

Written for a fresh Linux node with a clone of this repository and **no datasets**. That case works
because the parsed cohorts are committed; see `cohort_cache/` and `evaluation/cohort_cache.py`.

## What does and does not need to travel

| | size | needed on the node |
| --- | --- | --- |
| `cohort_cache/*.json.gz` | ~0.4 MB | **yes** — committed, it is the data |
| CGMacros archive | 627 MB | no |
| Hall 2018 | 5.8 MB | no |
| ShanghaiT2DM | 12.6 MB | no |
| `pandas`, `xlrd`, `openpyxl` | — | no (only the Shanghai *parser* needs them) |

The archives and the Excel engines are only required to **rebuild** the cache, which happens on a
machine that has the data. If a dataset's licence restricts where it may be copied, note that the
committed cache is still derived participant data; check the data use agreement before putting either
on a third-party node.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-aistats.txt
python -m evaluation.cohort_cache verify      # must print ok for all three cohorts
python -m pytest tests -q -m "not slow"       # fast suite, about a minute
```

`requirements-aistats.txt` was frozen on Windows and pins a CPU build of JAX. On Linux the same
versions resolve; if `pip` objects to a Windows-only wheel, install `jax[cpu]==0.11.0` first and then
the rest. For a CUDA node, `jax[cuda12]` — but read the note on precision at the bottom first.

## Choose the worker count by measuring, not guessing

Worker processes are independent, so throughput scales with **physical** cores. RunPod quotes vCPUs,
which are threads: a 64-vCPU node has 32 physical cores. On this laptop (8 physical) the measured
speed-up was 4.0x at 4 workers, 6.9x at 8, and only 7.8x at 14 — past the physical core count the
curve flattens hard.

Measure it on the node, it takes about three minutes:

```bash
python -m evaluation.benchmark_scaling --workers 1 8 16 31 --save
```

Then use the knee of that curve. The rule of thumb is **physical cores minus one**.

## Running an analysis

Every long analysis is a module the runner executes as resumable units:

```bash
python -m evaluation.runner evaluation.<analysis> --gate 3            # cheap check first
python -m evaluation.runner evaluation.<analysis> --workers 31        # the real run
```

**Resuming is the default.** Re-running the identical command skips every unit that already has a
result file. Nothing needs to be passed to make that happen, and nothing is recomputed.

Useful flags:

| flag | what it does |
| --- | --- |
| `--gate N` | run only the first N units, then stop. Always do this before a long run. |
| `--status` | print what is done, failed, claimed and pending, and exit without computing |
| `--retries N` | attempts per unit before a failure is recorded (default 1 retry) |
| `--unit-timeout S` | a worker whose unit exceeds this is killed and replaced (default 1800 s) |
| `--restart` | clear recorded failures and stale claims; **results are kept** |
| `--set key=value` | override a config key; changes the config hash, so it writes to a new directory |

## What happens when things go wrong

| event | consequence |
| --- | --- |
| Ctrl-C once | stops dispatching, lets in-flight units finish, exits cleanly. Re-run to resume. |
| Ctrl-C twice | stops immediately. At most the in-flight units are lost. |
| node reclaimed, power cut, OOM kill | every finished unit is already on disk; re-run to resume |
| a worker dies mid-unit | its claim goes stale, another worker takes the unit, a replacement worker starts |
| a unit hangs | the worker is killed after `--unit-timeout`; the rest of the run continues |
| a unit raises | retried, then recorded in `<unit>.failed.json` with its traceback and reported in the summary; it is not retried on later runs unless `--restart` |
| two runs at once | harmless — claims are exclusive filesystem locks |

Results are written to a temporary file and renamed, so a process killed mid-write leaves either the
old file or the new one and never half of one.

Progress is in `logs/<analysis>.progress.json`, updated every two seconds with counts, units per
hour and an ETA. `logs/<analysis>.log` has one line per completed unit.

### Writing results to a mounted volume

A node's root filesystem is often ephemeral. Point the results at a volume:

```bash
export HORIZON_RESULTS_DIR=/workspace/results
export HORIZON_LOGS_DIR=/workspace/logs
```

Resume works across that too — it is just where the unit files live. Copy them back into `results/`
before `python -m evaluation.freeze_results`, which is what the paper is generated from.

## On GPUs

The engine solves one small ODE system per meal, and a fit is 500 *sequential* optimizer steps over
about 36 of them. That batch is far too small to fill a GPU; launch overhead per step would dominate
and a direct port would likely be slower than a CPU node.

A GPU only pays off with two changes:

1. **Batch across the outer loop** — vmap the fit over subject x fold x repeat x initialization
   instead of running one process per fit. Multistart then becomes about 810,000 concurrent solves,
   which is the shape a GPU wants.
2. **Double-precision hardware.** These analyses need float64: the toy tests assert Fisher eigenvalue
   ratios below 1e-8, which single precision cannot represent at all. GeForce and Ada cards run FP64
   at 1/64 rate and are *worse than a CPU* here. It has to be A100 or H100 class.

Until both are true, a CPU node with many physical cores is the better purchase.

#!/usr/bin/env bash
# Phase 14 compute queue (Amendment 10): trace coordinate profiles with 500 inner steps in place of 100, then archive.
# Usage:  N=40 nohup bash scripts/pod_phase14.sh > logs/pod_phase14.out 2>&1 &
# Set one thread per worker first (as in Phase 13):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# The replica truths and noise come from results/A4_fisher (tracked in git), so a fresh clone is enough.
set -u
N="${N:-30}"
if [ "${1:-}" != "archive" ]; then
  python -c "import jax, equinox, diffrax, optax, scipy, sklearn" 2>/dev/null || {
    echo "!!! python cannot import jax/equinox/diffrax/optax/scipy/sklearn. Activate the venv first:"
    echo "    . /tmp/venv/bin/activate     (rebuild it if /tmp/venv is gone: see docs/PHASE9.md)"
    exit 1
  }
fi
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
archive() { tar czf phase14_results.tgz results logs && echo "wrote phase14_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

COMMON="--set objective=\"trace\" --set parameterization=\"coords\" --set bounds_scale=1.0 --set polish=true --set steps=500"

# --- H28 and R8: exact carbohydrate, a quarter of the noise ------------------------------------------------------
run evaluation.profile_lik $COMMON --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
# --- R7: exact carbohydrate at the estimated noise, then carbohydrate error 0.25 -------------------------------
run evaluation.profile_lik $COMMON --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_lik $COMMON --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
# --- H27: the real data ------------------------------------------------------------------------------------------
run evaluation.profile_lik $COMMON

echo "=== $(date '+%F %T') all queued Phase 14 jobs finished"
archive
echo "Download phase14_results.tgz, then STOP THE POD in the console."

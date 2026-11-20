#!/usr/bin/env bash
# Phase 17 compute queue (Amendment 13): the profiles of the paper on a 29-point grid in place of 15 (same Adam inner solver).
# Then archive.
# Usage:  N=8 nohup bash scripts/pod_phase17.sh > logs/pod_phase17.out 2>&1 &
# Set one thread per worker first (as in Phases 13 to 16):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# The replica truths and noise come from results/A4_fisher (colab_seed.tgz or the Drive results folder).
set -u
N="${N:-8}"
if [ "${1:-}" != "archive" ]; then
  python -c "import jax, equinox, diffrax, optax, scipy, sklearn" 2>/dev/null || {
    echo "!!! python cannot import jax/equinox/diffrax/optax/scipy/sklearn. Activate the environment first."
    exit 1
  }
fi
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
archive() { tar czf phase17_results.tgz results logs && echo "wrote phase17_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- 1. H34 and R15: coordinate profiles on 29 grid points (the 15-point sets are already stored) --------------------
COORD=(--set 'objective="trace"' --set 'parameterization="coords"' --set bounds_scale=1.0 --set polish=true --set grid_points=29)
run evaluation.profile_lik "${COORD[@]}"
run evaluation.profile_lik "${COORD[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_lik "${COORD[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'

# --- 2. R16: the positive control on 29 grid points (rate coordinates, as in Phase 15) ---------------------------------
RATES=(--set 'objective="trace"' --set bounds_scale=1.0 --set polish=true --set extension_points=0 --set grid_points=29)
run evaluation.profile_lik "${RATES[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random","residual_gut":0.1}'
run evaluation.profile_lik "${RATES[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random"}'

echo "=== $(date '+%F %T') all queued Phase 17 jobs finished"
archive
echo "Copy phase17_results.tgz to Drive, then stop the runtime."

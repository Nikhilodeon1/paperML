#!/usr/bin/env bash
# Phase 16 compute queue (Amendment 12): the profile at the true value four ways, then the trace coordinate profiles with a
# converged inner solver. Then archive.
# Usage:  N=8 nohup bash scripts/pod_phase16.sh > logs/pod_phase16.out 2>&1 &
# Set one thread per worker first (as in Phases 13 to 15):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# The profile-at-the-truth runs read the stored Phase 14 coordinate profiles (results/A5_profile, 500 inner steps), so
# run this where those are present (see results in colab_seed.tgz).
set -u
N="${N:-8}"
if [ "${1:-}" != "archive" ]; then
  python -c "import jax, equinox, diffrax, optax, scipy, sklearn" 2>/dev/null || {
    echo "!!! python cannot import jax/equinox/diffrax/optax/scipy/sklearn. Activate the environment first."
    exit 1
  }
fi
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
archive() { tar czf phase16_results.tgz results logs && echo "wrote phase16_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- 1. R13 and H31: the profile at the true value, four ways (cheap) -------------------------------------------------
run evaluation.profile_at_truth --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
run evaluation.profile_at_truth --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_at_truth --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'

# --- 2. H33, R14 and H32: the trace coordinate profiles with a converged inner solver (tau1 and p only) ---------------
COMMON=(--set 'objective="trace"' --set 'parameterization="coords"' --set bounds_scale=1.0 --set polish=true
        --set 'inner_solver="lbfgsb"' --set 'profile_only=["log_tau1","log_p"]')
run evaluation.profile_lik "${COMMON[@]}" --set grid_points=29 --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
run evaluation.profile_lik "${COMMON[@]}"
run evaluation.profile_lik "${COMMON[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
run evaluation.profile_lik "${COMMON[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_lik "${COMMON[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'

echo "=== $(date '+%F %T') all queued Phase 16 jobs finished"
archive
echo "Copy phase16_results.tgz to Drive, then stop the runtime."

#!/usr/bin/env bash
# Phase 15 compute queue (Amendment 11): a positive control with the symmetry broken by design, a perturbation of the
# fixed parameters, and the likelihood at the truth. Then archive.
# Usage:  N=48 nohup bash scripts/pod_phase15.sh > logs/pod_phase15.out 2>&1 &
# Set one thread per worker first (as in Phases 13 and 14):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# The likelihood-at-the-truth runs read the stored Phase 14 coordinate profiles (results/A5_profile), so run this on the
# same persistent disk, or restore phase14_results first.
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
archive() { tar czf phase15_results.tgz results logs && echo "wrote phase15_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- 1. H29 and P1: trace profiles in rate coordinates, random truths, exact carbohydrate, residual gut content ----
TRACE="--set objective=\"trace\" --set bounds_scale=1.0 --set polish=true --set steps=500"
run evaluation.profile_lik $TRACE --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random","residual_gut":0.1}'
run evaluation.profile_lik $TRACE --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random"}'
run evaluation.profile_lik $TRACE --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random","residual_gut":0.05}'
run evaluation.profile_lik $TRACE --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"truth":"random","residual_gut":0.2}'
run evaluation.profile_lik $TRACE --set 'replica={"seed":1,"cgm":true,"carb_cv":0.0,"truth":"random","residual_gut":0.1}'

# --- 2. P2: S_I under iAUC when a fixed parameter of the truth is off by 30% --------------------------------------
for perturb in '{"insulin_secretion":1.3}' '{"insulin_secretion":0.7}' '{"x_gain":1.3}' '{"x_gain":0.7}'; do
  run evaluation.profile_lik --set bounds_scale=1.0 --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":0.25,\"perturb\":$perturb}"
done

# --- 3. H30 and P3: the likelihood at the truth (reads the stored Phase 14 coordinate profiles) -------------------
run evaluation.truth_calibration --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'
run evaluation.truth_calibration --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.truth_calibration --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'

echo "=== $(date '+%F %T') all queued Phase 15 jobs finished"
archive
echo "Download phase15_results.tgz, then STOP THE POD in the console."

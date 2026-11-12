#!/usr/bin/env bash
# Phase 13 compute queue (Amendment 9, TMLR revision), in priority order, then archive.
# Usage:  N=30 nohup bash scripts/pod_phase13.sh > logs/pod_phase13.out 2>&1 &
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# Stop it at any point: whatever finished is archived by `bash scripts/pod_phase13.sh archive`.
# The stored Phase 8-12 results (results/A4_fisher, A5_profile, A9_prediction_cv) must be in the checkout: the
# truth-floor and numerics analyses read them. A fresh clone of the committed repository is enough.
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
archive() { tar czf phase13_results.tgz results logs && echo "wrote phase13_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- 1. R1: the floor of the true model on the replicas (cheap: no fitting) -----------------------------------
run evaluation.truth_floor --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
run evaluation.truth_floor --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
for seed in 0 1; do
  run evaluation.truth_floor --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":0.25,\"truth\":\"random\"}"
done

# --- 2. R4: numerical checks (cheap) ----------------------------------------------------------------------------
run evaluation.numerics_check
run evaluation.profile_lik --set steps=500 --set limit=10 --set bounds_scale=1.0

# --- 3. R1: held-out prediction on the random-truth replicas (protocol of A9) ----------------------------------
for seed in 0 1; do
  run evaluation.prediction_cv --set cohort='"cgmacros"' --set min_meals=10 \
      --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":0.25,\"truth\":\"random\"}" \
      --set 'cells=["grad3","grad1","personal_mean"]'
done

# --- 4. R6: carbohydrate-error sweep, trace coordinate profiles (polished), replica seed 0 ---------------------
for cv in 0.1 0.5 0.75; do
  run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
      --set polish=true --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":$cv}"
done

# --- 5. H26: 30 random starts for the polish, real data and replica seed 0 -------------------------------------
run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
    --set polish=true --set polish_starts=30
run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
    --set polish=true --set polish_starts=30 --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'

echo "=== $(date '+%F %T') all queued Phase 13 jobs finished"
archive
echo "Download phase13_results.tgz, then STOP THE POD in the console."

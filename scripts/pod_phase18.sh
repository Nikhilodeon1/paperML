#!/usr/bin/env bash
# Phase 18 compute queue (Amendment 14): the carbohydrate-error result with the noise re-estimated at the fit (H35), a systematic
# carbohydrate error (H37), two more replica seeds (H36), the Shanghai cohort (R18, H38) and chronological folds (R19). Then archive.
# Usage:  N=8 nohup bash scripts/pod_phase18.sh > logs/pod_phase18.out 2>&1 &
# Set one thread per worker first (as in Phases 13 to 17):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
# Replica runs refit every subject (there is no stored fit for a replica), so they are slower than the Phase 14 to 17 runs.
# A unit that fails once is skipped on resume: delete the *.failed.json files and run again to retry it.
set -u
N="${N:-8}"
if [ "${1:-}" != "archive" ]; then
  python -c "import jax, equinox, diffrax, optax, scipy, sklearn" 2>/dev/null || {
    echo "!!! python cannot import jax/equinox/diffrax/optax/scipy/sklearn. Activate the environment first."
    exit 1
  }
fi
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
archive() { tar czf phase18_results.tgz results logs && echo "wrote phase18_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

COORD=(--set 'objective="trace"' --set 'parameterization="coords"' --set bounds_scale=1.0 --set polish=true)

# --- 1. H35: the noise standard deviation and autocorrelation estimated again at the polished estimate -----------------
run evaluation.profile_lik "${COORD[@]}" --set 'noise_source="final"' --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
run evaluation.profile_lik "${COORD[@]}" --set 'noise_source="final"' --set 'replica={"seed":0,"cgm":true,"carb_cv":0.75}'

# --- 2. H37 and R17: a systematic carbohydrate error, one multiplier per subject (trace, then iAUC for the S_I error) ---
run evaluation.profile_lik "${COORD[@]}" --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"carb_bias":0.25}'
run evaluation.profile_lik --set bounds_scale=1.0 --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"carb_bias":0.25}'

# --- 3. H36: replica seeds 1 and 2 at carbohydrate error 0, 0.25 and 0.75 --------------------------------------------------
for seed in 1 2; do
  for cv in 0.0 0.75 0.25; do
    run evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":$cv}"
  done
done

# --- 4. R18: the Shanghai real data (the iAUC profiles of Phase 8 are reused if present, the trace profiles are new) ---
SHANGHAI=(--set 'cohort="shanghai"' --set min_meals=10)
run evaluation.profile_lik "${SHANGHAI[@]}" --set bounds_scale=1.0
run evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}"

# --- 5. H38: the pattern on a replica built from the Shanghai subjects --------------------------------------------------
for cv in 0.0 0.75; do
  run evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}" --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":$cv}"
done

# --- 6. R19: chronological folds for the held-out comparison on CGMacros (one repeat: the folds do not depend on it) -----
for scheme in blocked forward; do
  run evaluation.prediction_cv --set "fold_scheme=\"$scheme\"" --set repeats=1 --set 'cells=["grad3","grad1","personal_mean","population"]'
done

echo "=== $(date '+%F %T') all queued Phase 18 jobs finished"
archive
echo "Copy phase18_results.tgz to Drive, then stop the runtime."

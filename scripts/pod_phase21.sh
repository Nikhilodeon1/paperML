#!/usr/bin/env bash
# Phase 21 compute queue (Amendment 17): every trace coordinate profile of the paper recomputed with the converged inner solver
# (L-BFGS-B at every grid point) on a 29-point grid, with the real-pole projection that respects the fixed coordinate; random-truth
# replicas; a box four times as wide for the iAUC profiles; perturbed fixed constants in the generator. Then archive.
# Usage:  N=8 nohup bash scripts/pod_phase21.sh > logs/pod_phase21.out 2>&1 &
# Set one thread per worker first (as in Phases 13 to 18):
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code version).
set -u
N="${N:-8}"
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
archive() { tar czf phase21_results.tgz results logs && echo "wrote phase21_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

COORD=(--set 'objective="trace"' --set 'parameterization="coords"' --set bounds_scale=1.0 --set polish=true --set grid_points=29 --set 'inner_solver="lbfgsb"')

# --- 1. real CGMacros, and the carbohydrate-error sweep on replica seed 0 with the fitted truths ---------------------------
run evaluation.profile_lik "${COORD[@]}"
for cv in 0.25 0.0 0.75 0.1 0.5; do
  run evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":$cv}"
done

# --- 2. the same sweep with random truths spread through the box (seeds 0 and 1) ---------------------------------------------
for seed in 0 1; do
  for cv in 0.25 0.0 0.75; do
    run evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":$cv,\"truth\":\"random\"}"
  done
done

# --- 3. replica seeds 1 and 2 with fitted truths ------------------------------------------------------------------------------
for seed in 1 2; do
  for cv in 0.0 0.75; do
    run evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":$cv}"
  done
done

# --- 4. Shanghai and the standardized-meal cohort ------------------------------------------------------------------------------
SHANGHAI=(--set 'cohort="shanghai"' --set min_meals=10)
run evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}"
for cv in 0.0 0.75; do
  run evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}" --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":$cv}"
done
run evaluation.profile_lik --set 'cohort="hall"' --set min_meals=5 "${COORD[@]}"

# --- 5. iAUC profiles at a box four times as wide, and the Fisher check at tolerance 1e-8 for every subject --------------------
run evaluation.profile_lik --set 'objective="iauc"' --set bounds_scale=4.0
run evaluation.numerics_check --set n_subjects=45

# --- 6. fixed constants perturbed by 30% in the generator and not in the fit (iAUC profiles of S_I, k_e, k_a) --------------------
for perturb in '{"glucose_effectiveness":1.3}' '{"glucose_effectiveness":0.7}' '{"x_decay":1.3}' '{"x_decay":0.7}' \
               '{"insulin_clearance":1.3}' '{"insulin_clearance":0.7}'; do
  run evaluation.profile_lik --set 'objective="iauc"' --set bounds_scale=1.0 --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":0.25,\"perturb\":$perturb}"
done

echo "=== $(date '+%F %T') all queued Phase 21 jobs finished"
archive
echo "Copy phase21_results.tgz to Drive, then stop the runtime."

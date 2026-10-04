#!/usr/bin/env bash
# Phase 8 compute queue (Amendment 4), in priority order, then archive and stop the pod.
# Usage:  N=30 nohup bash scripts/pod_phase8.sh > logs/pod_phase8.out 2>&1 &
# Every command resumes. Do not commit or pull while it runs (the result folder names depend on the code
# version, and the POD_RUN_MARKER file keeps the dirty flag constant).
set -u
N="${N:-30}"
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" || echo "!!! $* exited non-zero; continuing"; }

# --- T3: H11 and H12 (constructive result) -------------------------------------------------------
for obj in iauc_centroid trace iauc; do
  run evaluation.profile_lik --set objective=$obj --set parameterization=coords --set bounds_scale=1.0
done
for obj in iauc_centroid iauc; do
  run evaluation.profile_lik --set objective=$obj --set parameterization=tied --set bounds_scale=1.0
done
run evaluation.prediction_cv --set 'cells=["grad2_tied","grad3_coords","grad2_tied_trace","grad3_coords_trace"]'

# --- T2: well-specified replica (H13, H14) -------------------------------------------------------
run evaluation.profile_lik --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}' --set bounds_scale=1.0
for setting in '{"seed":0,"cgm":true,"carb_cv":0.25}' '{"seed":0,"cgm":false,"carb_cv":0.0}' \
               '{"seed":0,"cgm":false,"carb_cv":0.25}' '{"seed":0,"cgm":true,"carb_cv":0.0}' \
               '{"seed":0,"cgm":true,"carb_cv":0.5}'; do
  run evaluation.prediction_cv --set "replica=$setting" --set repeats=3 --set 'cells=["grad3","grid3","personal_mean"]'
done

# --- T4: replication on Shanghai and Hall (H15) --------------------------------------------------
run evaluation.fisher_full --set cohort=shanghai --set min_meals=10 --set bounds_scale=1.0
run evaluation.profile_lik --set cohort=shanghai --set min_meals=10 --set bounds_scale=1.0
run evaluation.fisher_full --set cohort=hall --set min_meals=5 --set bounds_scale=1.0
run evaluation.profile_lik --set cohort=hall --set min_meals=5 --set bounds_scale=1.0
run evaluation.fisher_full --set cohort=hall --set min_meals=5 --set objective=trace --set bounds_scale=1.0
run evaluation.fisher_full --set cohort=shanghai --set min_meals=10 --set bounds_scale=2.0
run evaluation.profile_lik --set cohort=shanghai --set min_meals=10 --set bounds_scale=2.0
run evaluation.fisher_full --set cohort=hall --set min_meals=5 --set bounds_scale=2.0
run evaluation.profile_lik --set cohort=hall --set min_meals=5 --set bounds_scale=2.0

# --- T5: diagnosing the S_I result ---------------------------------------------------------------
run evaluation.saturation
for scale in 0.75 1.33; do
  run evaluation.fisher_full --set carb_scale=$scale --set bounds_scale=1.0
done

# --- T6: leakage intervals -----------------------------------------------------------------------
run evaluation.leakage_ci

echo "=== $(date '+%F %T') all queued Phase 8 jobs finished"
tar czf phase8_results.tgz results logs && echo "wrote phase8_results.tgz"
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "Download phase8_results.tgz, then STOP THE POD in the console."
fi

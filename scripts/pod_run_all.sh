#!/usr/bin/env bash
# Runs every queued analysis on a node, in priority order, then stops the pod.
# Usage:  N=15 nohup bash scripts/pod_run_all.sh > logs/pod_run_all.out 2>&1 &
# Every command resumes: re-running this script skips finished units. Do not commit while it runs.
set -u
N="${N:-15}"
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" || echo "!!! $* exited non-zero; continuing"; }

# 1. primary prediction CV (150 steps, 5 repeats, every cell)
run evaluation.prediction_cv
# 2. profile likelihood: pre-registered 1x box first, then the sweep
run evaluation.profile_lik
run evaluation.profile_lik --set bounds_scale=2.0
run evaluation.profile_lik --set bounds_scale=0.5
# 3. the observable ladder: Fisher per objective and box, then the common-reference comparison
for obj in iauc_centroid trace; do for s in 1.0 2.0 0.5; do
  run evaluation.fisher_full --set objective=$obj --set bounds_scale=$s
done; done
for s in 1.0 2.0 0.5; do run evaluation.ladder --set bounds_scale=$s; done
# 4. diagnostics and optimizer checks
for s in 1.0 2.0 0.5; do run evaluation.gradient_diag --set bounds_scale=$s; done
for s in 1.0 2.0 0.5; do run evaluation.optimizer_check --set bounds_scale=$s; done
# 5. 500-step sanity check: grad3 and grad1 iAUC cells only, three repeats (amendment 3)
run evaluation.prediction_cv --set steps=500 --set repeats=3 --set 'cells=["grad3","grad1"]'

echo "=== $(date '+%F %T') all queued jobs finished"
python -m evaluation.runner evaluation.prediction_cv --status || true
tar czf results_return.tgz results logs && echo "wrote results_return.tgz"
# Stop billing. Works on RunPod when the pod id is in the environment; otherwise stop it by hand.
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "Download results_return.tgz, then STOP THE POD in the console."
fi

#!/usr/bin/env bash
# Phase 12 compute queue (Amendment 8): replica with exact carbohydrate (carb_cv 0), then archive.
# Usage:  N=30 nohup bash scripts/pod_phase12.sh > logs/pod_phase12.out 2>&1 &
# Every command resumes. Do not pull or commit while it runs. The replica truths come from results/A4_fisher
# (tracked in git), so a fresh clone is enough.
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
archive() { tar czf phase12_results.tgz results logs && echo "wrote phase12_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- H25: trace coordinate profiles (polished) on the replica with exact carbohydrate -----------------------
run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
    --set polish=true --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
    --set polish=true --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'

# --- H25: iAUC profiles on the same replicas ----------------------------------------------------------------
run evaluation.profile_lik --set bounds_scale=1.0 --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0}'
run evaluation.profile_lik --set bounds_scale=1.0 \
    --set 'replica={"seed":0,"cgm":true,"carb_cv":0.0,"cgm_scale":0.25}'

echo "=== $(date '+%F %T') all queued Phase 12 jobs finished"
archive
echo "Download phase12_results.tgz, then STOP THE POD in the console."

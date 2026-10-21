#!/usr/bin/env bash
# Phase 11 compute queue (Amendment 7), in priority order, then archive.
# Usage:  N=30 nohup bash scripts/pod_phase11.sh > logs/pod_phase11.out 2>&1 &
# Every command resumes. Do not pull or commit while it runs. The queue reads the stored fits in
# results/A4_fisher (tracked in git) for the replica truths, so a fresh clone is enough.
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
archive() { tar czf phase11_results.tgz results logs && echo "wrote phase11_results.tgz"; }
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

# --- H24: trace coordinate profiles (polished) on the replica with a smaller CGM noise scale -------------
for scale in 0.5 0.25; do
  run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
      --set polish=true --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":0.25,\"cgm_scale\":$scale}"
done

# --- H24: iAUC profiles on the same replicas ------------------------------------------------------------
for scale in 0.5 0.25; do
  run evaluation.profile_lik --set bounds_scale=1.0 \
      --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":0.25,\"cgm_scale\":$scale}"
done

# --- report only: more replica seeds for the coordinate calibration ---------------------------------------
for seed in 1 2; do
  run evaluation.profile_lik --set objective='"trace"' --set parameterization='"coords"' --set bounds_scale=1.0 \
      --set polish=true --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":0.25}"
done

echo "=== $(date '+%F %T') all queued Phase 11 jobs finished"
archive
echo "Download phase11_results.tgz, then STOP THE POD in the console."

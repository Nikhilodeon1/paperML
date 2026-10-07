#!/usr/bin/env bash
# Phase 9 compute queue (Amendment 5, TMLR revision), in priority order, then archive.
# Usage:  N=30 nohup bash scripts/pod_phase9.sh > logs/pod_phase9.out 2>&1 &
# Every command resumes. Do not commit or pull while it runs (result folder names depend on the code
# version, and POD_RUN_MARKER keeps the dirty flag constant). Stop it at any point: whatever finished is
# archived by the last block if you run `bash scripts/pod_phase9.sh archive`.
set -u
N="${N:-30}"
# Refuse to start without the environment: with the wrong python every job fails at import in a second
# and the script would still report that everything finished.
if [ "${1:-}" != "archive" ]; then
  python -c "import jax, equinox, diffrax, optax, scipy, sklearn" 2>/dev/null || {
    echo "!!! python cannot import jax/equinox/diffrax/optax/scipy/sklearn. Activate the venv first:"
    echo "    . /tmp/venv/bin/activate     (rebuild it if /tmp/venv is gone: see docs/PHASE9.md)"
    exit 1
  }
fi
# 7200 s per unit: the Phase 8 run killed two 2x Shanghai units at the 1800 s default for taking longer.
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }

archive() {
  tar czf phase9_results.tgz results logs && echo "wrote phase9_results.tgz"
}
if [ "${1:-}" = "archive" ]; then archive; exit 0; fi

RANDOM_TRUTH='{"seed":SEED,"cgm":true,"carb_cv":0.25,"truth":"random"}'

# --- 1. H10: diagnostic robustness sweeps (cheap) -------------------------------------------------
for k in 0 1 2 3 4; do
  run evaluation.gradient_diag --set init_seed=$k
done
run evaluation.gradient_diag --set log_param=true
run evaluation.gradient_diag --set optimizer='"lbfgs"'

# --- 2. H21 and H11 on polished estimates: coordinate and tied profiles -----------------------------
for obj in trace iauc_centroid iauc; do
  run evaluation.profile_lik --set objective=\"$obj\" --set parameterization='"coords"' --set bounds_scale=1.0 --set polish=true
done
for obj in iauc_centroid iauc; do
  run evaluation.profile_lik --set objective=\"$obj\" --set parameterization='"tied"' --set bounds_scale=1.0 --set polish=true
done

# --- 3. H18: coordinate profiles on the true-model replica --------------------------------------------
for obj in trace iauc_centroid iauc; do
  run evaluation.profile_lik --set objective=\"$obj\" --set parameterization='"coords"' --set bounds_scale=1.0 \
      --set polish=true --set 'replica={"seed":0,"cgm":true,"carb_cv":0.25}'
done

# --- 4. H19: synthetic recovery with random truths (profile + diagnostic), seeds 0 and 1 -------------
for seed in 0 1; do
  truth="${RANDOM_TRUTH/SEED/$seed}"
  run evaluation.profile_lik --set bounds_scale=1.0 --set "replica=$truth"
  run evaluation.gradient_diag --set "replica=$truth"
done

# --- 5. H20: prediction on Shanghai (A9 protocol, five repeats of five-fold) --------------------------
run evaluation.prediction_cv --set cohort='"shanghai"' --set min_meals=10 \
    --set 'cells=["grad3","grad1","grid3","grid1","personal_mean","population","persistence"]'

# --- 6. H22: second model class (Dalla Man): Fisher and profiles --------------------------------------
run evaluation.dalla_man_identifiability

# --- 7. H17: replica seeds 1 to 4 (iAUC profile, the H13 configuration) --------------------------------
for seed in 1 2 3 4; do
  run evaluation.profile_lik --set bounds_scale=1.0 --set "replica={\"seed\":$seed,\"cgm\":true,\"carb_cv\":0.25}"
done

# --- 8. Report-only: Hall trace profile; Shanghai 2x profile re-run so it covers all 87 subjects ------
run evaluation.profile_lik --set cohort='"hall"' --set min_meals=5 --set objective='"trace"' --set bounds_scale=1.0
run evaluation.profile_lik --set cohort='"shanghai"' --set min_meals=10 --set bounds_scale=2.0

echo "=== $(date '+%F %T') all queued Phase 9 jobs finished"
archive
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "Download phase9_results.tgz, then STOP THE POD in the console."
fi

#!/usr/bin/env bash
# Shared definitions for the four-pod Phase 21 split (Amendment 17). Source this file; do not run it.
# Needs POD set to a, b, c or d. Optional: N (workers, default all cores), SKIP (space-separated job names to skip).
set -u
: "${POD:?set POD=a|b|c|d}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
N="${N:-$(nproc --all)}"
SKIP="${SKIP:-}"
mkdir -p logs

COORD=(--set 'objective="trace"' --set 'parameterization="coords"' --set bounds_scale=1.0 --set polish=true --set grid_points=29 --set 'inner_solver="lbfgsb"')
SHANGHAI=(--set 'cohort="shanghai"' --set min_meals=10)
HALL=(--set 'cohort="hall"' --set min_meals=5)

want() { case " $SKIP " in *" $1 "*) return 1 ;; esac; return 0; }
run() { echo "=== $(date '+%F %T') $*"; python -m evaluation.runner "$@" --workers "$N" --unit-timeout 7200 || echo "!!! $* exited non-zero; continuing"; }
# job NAME  runner-args...   (skips when NAME is in SKIP; records NAME in logs/pod21_$POD.done when finished)
job() { local name=$1; shift; if want "$name"; then run "$@"; else echo "=== skipped $name"; fi; echo "$name" >> "logs/pod21_${POD}.done"; }
archive() { tar czf "phase21_${POD}_results.tgz" results/A5_profile logs && echo "wrote phase21_${POD}_results.tgz"; }

fit()  { job "fit$1_s$2"  evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":$2,\"cgm\":true,\"carb_cv\":$1}"; }
rand() { job "rand$1_s$2" evaluation.profile_lik "${COORD[@]}" --set "replica={\"seed\":$2,\"cgm\":true,\"carb_cv\":$1,\"truth\":\"random\"}"; }
perturb() { job "perturb_$1_$2" evaluation.profile_lik --set 'objective="iauc"' --set bounds_scale=1.0 \
              --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":0.25,\"perturb\":{\"$1\":$2}}"; }

#!/usr/bin/env bash
# Pod C: random-truth carbohydrate-error sweeps, seeds 0 and 1 (about 600 core-minutes).
export POD=c; . "$(dirname "$0")/pod_phase21_lib.sh"
[ "${1:-}" = "archive" ] && { archive; exit 0; }
rand 0.25 0; rand 0.0 0; rand 0.75 0
rand 0.25 1; rand 0.0 1; rand 0.75 1
perturb glucose_effectiveness 0.7
echo "=== $(date '+%F %T') pod $POD finished"; archive

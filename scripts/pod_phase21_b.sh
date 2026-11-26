#!/usr/bin/env bash
# Pod B: real CGMacros and the fitted-truth carbohydrate-error sweep, seed 0 (about 600 core-minutes).
export POD=b; . "$(dirname "$0")/pod_phase21_lib.sh"
[ "${1:-}" = "archive" ] && { archive; exit 0; }
job real evaluation.profile_lik "${COORD[@]}"
fit 0.25 0; fit 0.0 0; fit 0.75 0; fit 0.5 0; fit 0.1 0
perturb glucose_effectiveness 1.3
echo "=== $(date '+%F %T') pod $POD finished"; archive

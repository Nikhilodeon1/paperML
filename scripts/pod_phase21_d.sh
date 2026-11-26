#!/usr/bin/env bash
# Pod D: fitted-truth seeds 1 and 2, and the fixed-constant perturbations of the generator (about 630 core-minutes).
export POD=d; . "$(dirname "$0")/pod_phase21_lib.sh"
[ "${1:-}" = "archive" ] && { archive; exit 0; }
fit 0.0 1; fit 0.75 1; fit 0.0 2; fit 0.75 2
perturb x_decay 1.3; perturb x_decay 0.7; perturb insulin_clearance 1.3; perturb insulin_clearance 0.7
echo "=== $(date '+%F %T') pod $POD finished"; archive

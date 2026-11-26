#!/usr/bin/env bash
# Pod A: Shanghai and Hall (about 600 core-minutes). Usage: POD=a nohup bash scripts/pod_phase21_a.sh > logs/pod21_a.out 2>&1 &
export POD=a; . "$(dirname "$0")/pod_phase21_lib.sh"
[ "${1:-}" = "archive" ] && { archive; exit 0; }
job hall_real      evaluation.profile_lik "${HALL[@]}" "${COORD[@]}"
job shanghai_real  evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}"
for cv in 0.0 0.75; do
  job "shanghai_rep$cv" evaluation.profile_lik "${SHANGHAI[@]}" "${COORD[@]}" --set "replica={\"seed\":0,\"cgm\":true,\"carb_cv\":$cv}"
done
job box4 evaluation.profile_lik --set 'objective="iauc"' --set bounds_scale=4.0
echo "=== $(date '+%F %T') pod $POD finished"; archive

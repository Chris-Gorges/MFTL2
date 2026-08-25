#!/bin/bash
#
# Submit the whole scaling sweep, one Slurm job per point.
#
#   ./scaling/launch.sh                 # the default sweep
#   ./scaling/launch.sh --dry-run       # print the sbatch lines and submit nothing
#   RANKS="1 2 4 36" ./scaling/launch.sh
#
# Then, once the jobs have landed:
#
#   python3 run_scaling.py --plots-only
#
# which re-renders every plot and the report from whatever is in
# results/scaling/raw/. Nothing needs to be re-run to add a rank count later --
# drop its JSON in and render again.

# ---- EDIT FOR YOUR SITE ------------------------------------------------------
CORES_PER_NODE=36

# Rank counts. Below CORES_PER_NODE these share one node; above it they pack
# whole nodes, which is what makes the 36 -> 72 step measure the INTERCONNECT
# rather than a change of topology.
RANKS="${RANKS:-1 2 4 9 18 36 72 144 288}"

CASES="${CASES:-tiny medium large}"
MODES="${MODES:-strong weak}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-50}"

# Extra sbatch flags for your site, e.g. --qos=normal --constraint=cascadelake
SBATCH_EXTRA="${SBATCH_EXTRA:-}"
# ------------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="$ROOT/scaling/job.sh"
mkdir -p "$ROOT/results/scaling/raw" "$ROOT/results/scaling/slurm"

submit()
{
  local nranks="$1" case="$2" mode="$3" repeat="$4"
  local nodes=$(( (nranks + CORES_PER_NODE - 1) / CORES_PER_NODE ))
  local per_node=$(( nranks < CORES_PER_NODE ? nranks : CORES_PER_NODE ))

  # shellcheck disable=SC2086
  local cmd=(sbatch
             --nodes="$nodes"
             --ntasks="$nranks"
             --ntasks-per-node="$per_node"
             --job-name="mftl-${case}-${mode}-np${nranks}"
             --export="ALL,MFTL_ROOT=${ROOT},NRANKS=${nranks},CASE=${case},MODE=${mode},REPEAT=${repeat},STEPS=${STEPS}"
             $SBATCH_EXTRA
             "$JOB")

  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
}

# Repeats go ROUND ROBIN over the points, not in blocks.
#
# docs/performance-analysis.md sec 7 records the same binary measuring 9.4 ms and
# 6.1 ms in two sessions purely from machine state, and states the rule the whole
# document follows: interleave the variants, report minima, never quote a run
# taken in a block. Submitting all repeats of np=1, then all of np=2, invites the
# scheduler to place each group on one set of nodes at one time of day -- and a
# curve measured that way turns a thermal or placement gradient into a scaling
# law. Interleaving spreads each rank count across the queue.

n=0
for repeat in $(seq 0 $(( REPEATS - 1 ))); do
  echo "--- pass $(( repeat + 1 ))/$REPEATS ---"
  for case in $CASES; do
    for mode in $MODES; do
      for nranks in $RANKS; do
        submit "$nranks" "$case" "$mode" "$repeat"
        n=$(( n + 1 ))
      done
    done
  done
done

echo
echo "$n job(s) $( [ "$DRY_RUN" = "1" ] && echo 'would be submitted' || echo submitted )."
echo "When they have finished:  python3 run_scaling.py --plots-only"

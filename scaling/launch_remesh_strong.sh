#!/bin/bash
# STRONG scaling of the remeshed front-tracking timestep: one problem, varied
# rank count. REGIME picks the remeshing driver and stays fixed along the curve:
#   dispersed  5832 droplets (18^3, refine 3, R/h 0.06 at 1 node .. 0.17 at 30)
#   spanning   8 bubbles (2^3, refine 7, R/h 0.6 at 1 node .. 1.8 at 30); every
#              job runs BOTH spanning modes, once and nested, on the same problem
# One job per rank count, capped at one hour; the soft budget can omit timed
# trajectories, never part of one. The summary compares once with nested and,
# for dispersed, checks that every rank count ends on the same digest.
#
#   REGIME=dispersed scaling/launch_remesh_strong.sh --dry-run
#   REGIME=spanning MAX_NODES=4 scaling/launch_remesh_strong.sh
#   REGIME=spanning SPANNING_MODES=nested scaling/launch_remesh_strong.sh
#
# Build first with the site's modules: python3 build_all.py --comm-profile
# Override settings with environment variables; see QUICKSTART.md.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.sh
source "$HERE/site.sh"
# shellcheck source=sweep_common.sh
source "$HERE/sweep_common.sh"

MODE=remesh_strong
REGIME="${REGIME:-spanning}"
case "$REGIME" in
  dispersed) per_axis=18 refine=3 batch=64 ;;
  spanning) per_axis=2 refine=7 batch=4 ;;
  *) echo "error: REGIME must be dispersed or spanning" >&2; exit 2 ;;
esac
SWEEP_NAME="${SWEEP_NAME:-remesh_${REGIME}_strong}"
RANKS="${RANKS:-}"                    # otherwise derived from NODES x RANKS_PER_NODE
PER_AXIS="${PER_AXIS:-$per_axis}"     # bodies per axis of the periodic box
REFINE="${REFINE:-$refine}"           # 20 * 4^REFINE triangles per body
STEPS="${STEPS:-100}"                 # one trajectory: the flow reverses half-way and is the identity at T
TRAJECTORIES="${TRAJECTORIES:-2}"     # timed trajectories per mode; the fastest is reported
WARMUP="${WARMUP:-1}"                 # untimed trajectories (the first remesh to the band)
SIZING="${SIZING:-uniform}"           # uniform | curvature
PROJECTION="${PROJECTION:-phong}"     # phong | none
SPANNING_MODES="${SPANNING_MODES:-once,nested}"
JOB_REPEATS="${JOB_REPEATS:-1}"
RANKS_PER_NODE="${RANKS_PER_NODE:-$CORES_PER_NODE}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
INIT_BATCH="${INIT_BATCH:-$batch}"
MAX_SECONDS="${MAX_SECONDS:-3000}"    # split evenly over the modes a job runs
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ARGS="--regime $REGIME --per-axis $PER_AXIS --refine $REFINE --steps $STEPS --trajectories $TRAJECTORIES --warmup $WARMUP --sizing $SIZING --projection $PROJECTION --spanning-modes $SPANNING_MODES --init-batch $INIT_BATCH --max-seconds $MAX_SECONDS $EXTRA_ARGS"

sweep_main "$@"

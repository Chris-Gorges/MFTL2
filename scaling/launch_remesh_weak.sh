#!/bin/bash
# WEAK scaling of the remeshed front-tracking timestep: fixed work per rank,
# the problem grows with P. REGIME picks the remeshing driver:
#   dispersed  27 droplets per rank box (PER_AXIS=3, refine 3, R/h 0.083)
#   spanning   one bubble per 2x2x2 rank boxes (BOXES=2, refine 7, R/h 0.6),
#              centred on a rank-box corner; every job runs BOTH spanning modes.
#              Every axis of the grid must have an even number of cuts, which
#              no multiple of 36 ranks allows, so a spanning sweep runs 32 ranks
#              per node unless RANKS_PER_NODE says otherwise.
# Every rank box is a copy of every other at every step, so the measured
# imbalance should be 1 and only P changes along the curve.
#
#   REGIME=dispersed scaling/launch_remesh_weak.sh --dry-run
#   REGIME=spanning MAX_NODES=8 scaling/launch_remesh_weak.sh
#
# Build first with the site's modules: python3 build_all.py --comm-profile
# Override settings with environment variables; see QUICKSTART.md.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
user_rpn="${RANKS_PER_NODE:-}"  # site.sh fills an empty value with CORES_PER_NODE
# shellcheck source=site.sh
source "$HERE/site.sh"
# shellcheck source=sweep_common.sh
source "$HERE/sweep_common.sh"

MODE=remesh_weak
REGIME="${REGIME:-spanning}"
case "$REGIME" in
  dispersed) refine=3 batch=64 rpn="$CORES_PER_NODE" ;;
  spanning) refine=7 batch=4 rpn=$((CORES_PER_NODE < 32 ? CORES_PER_NODE : 32)) ;;
  *) echo "error: REGIME must be dispersed or spanning" >&2; exit 2 ;;
esac
SWEEP_NAME="${SWEEP_NAME:-remesh_${REGIME}_weak}"
RANKS="${RANKS:-}"                    # otherwise derived from NODES x RANKS_PER_NODE
PER_AXIS="${PER_AXIS:-3}"             # dispersed: bodies per rank box per axis
BOXES="${BOXES:-2}"                   # spanning: rank boxes per body per axis
REFINE="${REFINE:-$refine}"           # 20 * 4^REFINE triangles per body
STEPS="${STEPS:-100}"
TRAJECTORIES="${TRAJECTORIES:-2}"
WARMUP="${WARMUP:-1}"
SIZING="${SIZING:-uniform}"
PROJECTION="${PROJECTION:-phong}"
SPANNING_MODES="${SPANNING_MODES:-once,nested}"
JOB_REPEATS="${JOB_REPEATS:-1}"
RANKS_PER_NODE="${user_rpn:-$rpn}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
INIT_BATCH="${INIT_BATCH:-$batch}"
MAX_SECONDS="${MAX_SECONDS:-3000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ARGS="--regime $REGIME --per-axis $PER_AXIS --boxes-per-body $BOXES --refine $REFINE --steps $STEPS --trajectories $TRAJECTORIES --warmup $WARMUP --sizing $SIZING --projection $PROJECTION --spanning-modes $SPANNING_MODES --init-batch $INIT_BATCH --max-seconds $MAX_SECONDS $EXTRA_ARGS"

sweep_main "$@"

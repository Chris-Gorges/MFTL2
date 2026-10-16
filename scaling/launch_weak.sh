#!/bin/bash
# One problem size, varied rank count; no problem-size sweep.
# Defaults use 1, 2, 4, 8, 16 and 30 full 36-core nodes.
# Each point is one job, capped at one hour. Initialization distributes only
# new batches, and a 50-minute soft budget can omit further timed repeats.
#
#   scaling/launch_weak.sh --dry-run
#   scaling/launch_weak.sh
#
# Build first with the site's modules: python3 build_all.py --comm-profile
# Override settings with environment variables; see QUICKSTART.md.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.sh
source "$HERE/site.sh"
# shellcheck source=sweep_common.sh
source "$HERE/sweep_common.sh"

MODE=weak
RANKS="${RANKS:-36 72 144 288 576 1080}"
PER_RANK="${PER_RANK:-1000}" # 10^3 spheres/rank, 1,280,000 triangles/rank
BOX="${BOX:-10}"             # body spacing 1, the strong sweep's density
REFINE="${REFINE:-3}"        # 1280 triangles per body
STEPS="${STEPS:-200}"
CYCLES="${CYCLES:-1}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-3}"
JOB_REPEATS="${JOB_REPEATS:-1}"
RANKS_PER_NODE="${RANKS_PER_NODE:-$CORES_PER_NODE}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
INIT_BATCH="${INIT_BATCH:-512}" # about 107.6 MiB of new mesh per batch at refine=3
MAX_SECONDS="${MAX_SECONDS:-3000}" # soft budget includes setup; Slurm enforces TIME_LIMIT
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ARGS="--spheres-per-rank $PER_RANK --box $BOX --refine $REFINE --steps $STEPS --cycles $CYCLES --warmup $WARMUP --repeat $REPEAT --init-batch $INIT_BATCH --max-seconds $MAX_SECONDS $EXTRA_ARGS"

sweep_main "$@"

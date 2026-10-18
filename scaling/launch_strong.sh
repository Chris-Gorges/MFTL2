#!/bin/bash
# One problem size, varied rank count; no problem-size sweep.
# Defaults use 1, 2, 4 and 8 full 36-core nodes. MAX_NODES caps submissions;
# NODES selects node counts, or RANKS overrides them with explicit rank counts.
# Each point is one job, capped at one hour. Initialization distributes only
# new batches, and a 50-minute soft budget can omit further timed repeats.
#
#   scaling/launch_strong.sh --dry-run
#   MAX_NODES=4 scaling/launch_strong.sh
#   NODES="1 2 4 8" MEMORY_PER_NODE=default scaling/launch_strong.sh
#   scaling/launch_strong.sh
#
# Build first with the site's modules: python3 build_all.py --comm-profile
# Override settings with environment variables; see QUICKSTART.md.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=site.sh
source "$HERE/site.sh"
# shellcheck source=sweep_common.sh
source "$HERE/sweep_common.sh"

MODE=strong
RANKS="${RANKS:-}"          # otherwise derived from NODES x RANKS_PER_NODE
SPHERES="${SPHERES:-110592}" # 48^3 spheres, 141,557,760 triangles; one fixed problem
REFINE="${REFINE:-3}"        # 1280 triangles per body
STEPS="${STEPS:-200}"        # one trajectory advances the lattice (3,2,1) spacings
CYCLES="${CYCLES:-1}"        # trajectories per repeat; increase this to lengthen timing at fixed speed
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-3}"        # timed repeats inside each job; the fastest is reported
JOB_REPEATS="${JOB_REPEATS:-1}"  # separate jobs per rank count (other nodes, other times)
RANKS_PER_NODE="${RANKS_PER_NODE:-$CORES_PER_NODE}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
INIT_BATCH="${INIT_BATCH:-512}" # about 107.6 MiB of new mesh per batch at refine=3
MAX_SECONDS="${MAX_SECONDS:-3000}" # soft budget includes setup; Slurm enforces TIME_LIMIT
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ARGS="--spheres $SPHERES --refine $REFINE --steps $STEPS --cycles $CYCLES --warmup $WARMUP --repeat $REPEAT --init-batch $INIT_BATCH --max-seconds $MAX_SECONDS $EXTRA_ARGS"

sweep_main "$@"

#!/bin/bash
#
# One Slurm job = one point of one scaling curve.
#
# Submitted by launch.sh, which passes NRANKS, CASE, MODE, REPEAT, STEPS and
# EXTRA through --export. Nothing below the site block is machine-specific.
#
# Run it by hand for a single point:
#   sbatch --export=ALL,NRANKS=36,CASE=medium,MODE=strong,REPEAT=0 scaling/job.sh

# ---- EDIT FOR YOUR SITE ------------------------------------------------------
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --time=01:00:00
#SBATCH --job-name=mftl-scaling
#SBATCH --output=results/scaling/slurm/%x-%j.out
#SBATCH --error=results/scaling/slurm/%x-%j.err

# NOTE: --nodes, --ntasks and --ntasks-per-node are NOT here. #SBATCH directives
# are literal comments parsed before the shell runs, so they cannot depend on
# NRANKS. launch.sh passes them on the sbatch command line, where they override
# anything written here.

SITE_MODULES="gcc openmpi cmake python"

# "srun" is right on most clusters; use "mpirun" where srun is not built against
# the MPI in use. The rank count is appended as "-n $NRANKS" either way.
LAUNCHER="srun"

# The compute-node ISA. MFTL_ARCH_NATIVE is OFF by default precisely because
# login nodes routinely differ from compute nodes, which turns into SIGILL at run
# time -- so name the target architecture rather than using -march=native here.
ARCH_FLAGS="-march=native"

CORES_PER_NODE=36
# ------------------------------------------------------------------------------

set -euo pipefail

: "${NRANKS:?set NRANKS, e.g. --export=ALL,NRANKS=36}"
CASE="${CASE:-medium}"
MODE="${MODE:-strong}"
REPEAT="${REPEAT:-0}"
STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-3}"
EXTRA="${EXTRA:-}"

# Slurm may copy the batch script into a spool directory, so BASH_SOURCE is not a
# reliable way back to the repository. launch.sh exports MFTL_ROOT; SLURM_SUBMIT_DIR
# covers a hand-submitted job from the repo root; BASH_SOURCE is the last resort
# for running this script directly.
if [ -n "${MFTL_ROOT:-}" ]; then
  ROOT="$MFTL_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/CMakeLists.txt" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [ ! -f "$ROOT/CMakeLists.txt" ]; then
  echo "cannot locate the MFTL source tree (tried MFTL_ROOT, SLURM_SUBMIT_DIR, \$0). Set MFTL_ROOT." >&2
  exit 2
fi
cd "$ROOT"

RAW="$ROOT/results/scaling/raw"
mkdir -p "$RAW" "$ROOT/results/scaling/slurm"

# shellcheck disable=SC1090
if [ -n "$SITE_MODULES" ] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SITE_MODULES
fi

# ---- Build, once, into a node-architecture-specific tree ---------------------
#
# build_all.py has no pass-through for extra CMake flags, so the Release build
# with an explicit -march is spelled out here rather than pretending it can.
# Release is not optional: the benchmark is meaningless in any other build type,
# and a Debug build carries ASan.

BUILD="$ROOT/build"
BINARY="$BUILD/parallel_advection_scaling"

if [ ! -x "$BINARY" ]; then
  cmake -S "$ROOT" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_SANITIZERS=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_TESTING=OFF \
        -DMFTL_ARCH_FLAGS="$ARCH_FLAGS"
  cmake --build "$BUILD" --target parallel_advection_scaling -j "$CORES_PER_NODE"
fi

# ---- Problem size ------------------------------------------------------------
#
# Kept in step with the CASES table in run_scaling.py. Two tables is one too many,
# but a Slurm job cannot import Python without assuming an interpreter, and the
# Python driver is what owns the definitive version.

case "$CASE" in
  tiny)   REFINE=2; SPHERES=256; PER_RANK=32 ;;
  medium) REFINE=3; SPHERES=256; PER_RANK=32 ;;
  large)  REFINE=4; SPHERES=512; PER_RANK=64 ;;
  iso*)   : "${REFINE:?iso cases must pass REFINE and SPHERES explicitly}"
          : "${SPHERES:?iso cases must pass REFINE and SPHERES explicitly}"
          PER_RANK=0 ;;
  *)      echo "unknown CASE '$CASE' (tiny | medium | large | iso*)" >&2; exit 2 ;;
esac

if [ "$MODE" = "weak" ]; then
  SIZE_ARGS=(--spheres-per-rank "$PER_RANK")
else
  SIZE_ARGS=(--spheres "$SPHERES")
fi

GIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
JSON="$RAW/${CASE}_${MODE}_np${NRANKS}_r${REPEAT}.json"

echo "=== $CASE / $MODE / np=$NRANKS / repeat $REPEAT on ${SLURM_JOB_NUM_NODES:-1} node(s) ==="

# shellcheck disable=SC2086
$LAUNCHER -n "$NRANKS" \
  "$BINARY" \
  --label "$CASE" --mode "$MODE" --refine "$REFINE" \
  "${SIZE_ARGS[@]}" \
  --steps "$STEPS" --warmup "$WARMUP" --repeat 1 \
  --json "$JSON" --git "$GIT" $EXTRA

echo "wrote $JSON"

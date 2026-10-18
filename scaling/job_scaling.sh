#!/bin/bash
#
# One Slurm job = one point (MODE, NRANKS) of a scaling curve.
#
# Submitted by launch_strong.sh / launch_weak.sh, which pass --nodes, --ntasks,
# --ntasks-per-node, --time and --output on the sbatch line (#SBATCH lines cannot
# depend on the rank count) and these variables in the environment:
#
#   MFTL_ROOT  repository root            MODE      strong | weak
#   NRANKS     rank count                 OUT_DIR   the sweep directory
#   TAG        report name (np00064)      RUN_ARGS  driver arguments
#   GIT        commit, recorded in the report
#
# The report lands in $OUT_DIR/reports/$TAG.txt; this job's log, with the node
# list, free memory, the CPU binding and the full report, in $OUT_DIR/slurm/.
# A failed run leaves $OUT_DIR/reports/$TAG.failed, which collect.sh lists.

set -euo pipefail
exec 2>&1

: "${MFTL_ROOT:?set by the launch script}" "${MODE:?}" "${NRANKS:?}" "${OUT_DIR:?}"
TAG="${TAG:-np$(printf %05d "$NRANKS")}"
RUN_ARGS="${RUN_ARGS:-}"
GIT="${GIT:-unknown}"

REPORT="$OUT_DIR/reports/${TAG}.txt"
mkdir -p "$OUT_DIR/reports"
start=$(date +%s)
echo "job ${SLURM_JOB_ID:-local} started; no verified complete report yet" >"$OUT_DIR/reports/${TAG}.failed"
rm -f "$REPORT"
finish()
{
  local rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ] && ! grep -q '^@summary .* ok=1$' "$REPORT" 2>/dev/null; then rc=2; fi
  if [ "$rc" -ne 0 ]; then
    {
      echo "np=$NRANKS exit=$rc job=${SLURM_JOB_ID:-local} elapsed=$(($(date +%s) - start))s"
      echo "log: $OUT_DIR/slurm/ (job ${SLURM_JOB_ID:-local})"
      echo "run did not finish with a verified report; inspect the Slurm log and job state"
    } >"$OUT_DIR/reports/${TAG}.failed"
  else
    rm -f "$OUT_DIR/reports/${TAG}.failed"
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM

SNAPSHOT="$OUT_DIR/snapshot"

# shellcheck source=site.sh
source "$SNAPSHOT/site.sh"
site_modules
site_runtime_env

BIN="$SNAPSHOT/${MODE}_scaling"

echo "=== MFTL $MODE scaling: $NRANKS ranks, job ${SLURM_JOB_ID:-local} ==="
echo "date        $(date)"
echo "nodes       ${SLURM_JOB_NODELIST:-$(hostname)} (${SLURM_NNODES:-1} node(s), tasks/node ${SLURM_TASKS_PER_NODE:-?})"
echo "modules     ${LOADEDMODULES:-none}"
echo "launcher    $(command -v "${LAUNCHER[0]}" || echo "${LAUNCHER[0]} NOT FOUND")"
echo "binary      $BIN"
cksum "$BIN"
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  # Records the actual CPU/memory allocation, including partition defaults when
  # --mem was omitted. Diagnostic only; site restrictions must not kill a run.
  scontrol show job "$SLURM_JOB_ID" || true
fi
command -v lscpu >/dev/null 2>&1 && lscpu

# Free /dev/shm (MPI's shared-memory transport) and free memory, once per node.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
  srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 -l bash -c 'echo "$(uname -n): /dev/shm free $(df -k /dev/shm | tail -1 | awk "{print \$4}") kB, memory available $(free -m 2>/dev/null | awk "/^Mem:/{print \$7}") MB"' |
    sort -k2 | sed 's/^/node        /'
fi

if [ ! -x "$BIN" ]; then
  echo "binary missing: build with  python3 build_all.py --comm-profile"
  echo "missing binary $BIN" >"$OUT_DIR/reports/${TAG}.failed"
  exit 2
fi

cd "$OUT_DIR" || exit 2
echo "command     ${LAUNCHER[*]} $NRANKS $BIN $RUN_ARGS --label $MODE --git $GIT --report $REPORT"
echo

read -r -a args <<< "$RUN_ARGS"  # whitespace-delimited options; no shell evaluation or globbing
rc=0
"${LAUNCHER[@]}" "$NRANKS" "$BIN" "${args[@]}" --label "$MODE" --git "$GIT" --report "$REPORT" || rc=$?
elapsed=$(($(date +%s) - start))

echo
echo "exit code $rc after ${elapsed} s"
exit $rc

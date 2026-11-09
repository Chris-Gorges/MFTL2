# Submission logic shared by the launch_*.sh scripts. Sourced, not run.
#
# The caller sets MODE, RANKS, RUN_ARGS, TIME_LIMIT and JOB_REPEATS (and
# optionally SWEEP_NAME, which names the results directory and the jobs; default
# MODE), sources site.sh, then calls `sweep_main "$@"`. The driver is
# build/${MODE}_scaling. One Slurm job per rank count (x JOB_REPEATS), then one
# small job that waits for all of them (afterany) and runs collect.sh, which
# merges every report into <sweep>/summary.txt (a remeshing sweep through
# collect_remesh.sh).
# shellcheck shell=bash

sweep_usage()
{
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

# Resolve placement once, before making snapshots or submitting anything.
# Keep the requested points for the manifest, but use only eligible points in
# sweep.txt and the collector. A cap exclusion is not a failed measurement.
sweep_resolve_points()
{
  local value n p nodes seen=" " selected=""
  for value in "$CORES_PER_NODE" "$MAX_NODES" "$RANKS_PER_NODE" "$JOB_REPEATS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || [ "${#value}" -gt 7 ]; then
      echo "error: ranks, node counts and repeats must be positive integers (at most 7 digits)." >&2
      return 2
    fi
  done
  if [ "$RANKS_PER_NODE" -gt "$CORES_PER_NODE" ]; then
    echo "error: use RANKS_PER_NODE <= CORES_PER_NODE." >&2
    return 2
  fi
  if [ -n "$RANKS" ]; then
    POINT_SOURCE="RANKS"
    REQUESTED_RANKS="$RANKS"
  else
    POINT_SOURCE="NODES"
    REQUESTED_RANKS=""
    for n in $NODES; do
      if [[ ! "$n" =~ ^[1-9][0-9]*$ ]] || [ "${#n}" -gt 7 ]; then
        echo "error: NODES must contain positive integers (at most 7 digits)." >&2
        return 2
      fi
      REQUESTED_RANKS="$REQUESTED_RANKS $((n * RANKS_PER_NODE))"
    done
  fi
  for p in $REQUESTED_RANKS; do
    if [[ ! "$p" =~ ^[1-9][0-9]*$ ]] || [ "${#p}" -gt 7 ]; then
      echo "error: rank counts must be positive integers (at most 7 digits)." >&2
      return 2
    fi
    if [[ "$seen" == *" $p "* ]]; then echo "error: duplicate rank count $p" >&2; return 2; fi
    seen="$seen$p "
    nodes=$(((p + RANKS_PER_NODE - 1) / RANKS_PER_NODE))
    if [ "$nodes" -le "$MAX_NODES" ]; then selected="$selected $p"; fi
  done
  RANKS="${selected# }"
  return 0
}

# Refuses to queue a sweep whose numbers would be useless: no binary, a
# non-Release build, or a build without the redistribution timers.
sweep_preflight()
{
  local cache="$BUILD_DIR/CMakeCache.txt"
  if [ ! -x "$BIN" ] || [ ! -f "$cache" ]; then
    echo "error: $BIN not found." >&2
    echo "  Build once on the login node, with the modules from scaling/site.sh loaded:" >&2
    echo "    python3 build_all.py --comm-profile" >&2
    return 1
  fi
  if ! grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' "$cache"; then
    echo "error: build/ is not a Release build; timings would be meaningless." >&2
    echo "  python3 build_all.py --comm-profile" >&2
    return 1
  fi
  if ! grep -q '^MFTL_COMM_PROFILE:BOOL=ON' "$cache"; then
    if [ "${ALLOW_NO_PROFILE:-0}" != 1 ]; then
      echo "error: the library was built without -DMFTL_COMM_PROFILE=ON, so the redistribution" >&2
      echo "  profile (the point of this sweep) would be all zeros. Rebuild with" >&2
      echo "    python3 build_all.py --comm-profile" >&2
      echo "  or set ALLOW_NO_PROFILE=1 for a phase-only comparison of profiling overhead." >&2
      return 1
    fi
    echo "note: MFTL_COMM_PROFILE is off -- phase timings only (ALLOW_NO_PROFILE=1)."
  fi
  local src
  for src in $(sweep_sources); do
    if [ "$BIN" -ot "$ROOT/scaling/$src" ]; then
      echo "error: $BIN is older than its sources in scaling/; rebuild it before submitting." >&2
      return 1
    fi
  done
  return 0
}

# The driver's sources in scaling/, snapshotted with every sweep.
sweep_sources()
{
  echo scaling_common.h scaling_init.h "${MODE}_scaling.c"
  case "$MODE" in remesh_*) echo remesh_common.h ;; esac
}

sweep_main()
{
  local dry=0 a src
  for a in "$@"; do
    case "$a" in
      --dry-run | -n) dry=1 ;;
      -h | --help)
        sweep_usage
        return 0
        ;;
      *)
        echo "unknown argument: $a (try --help)" >&2
        return 2
        ;;
    esac
  done

  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
  BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)" || return 1
  BIN="$BUILD_DIR/${MODE}_scaling"
  SWEEP_NAME="${SWEEP_NAME:-$MODE}"
  if [[ ! "$SWEEP_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "error: SWEEP_NAME may contain only letters, digits and _ . -" >&2
    return 2
  fi
  sweep_resolve_points || return $?
  local memory=()
  if [ -n "$MEMORY_PER_NODE" ] && [ "$MEMORY_PER_NODE" != default ]; then
    if [[ ! "$MEMORY_PER_NODE" =~ ^[0-9]+[KMGTkmgt]?$ ]]; then
      echo "error: MEMORY_PER_NODE must be default, empty, 0, or an integer size such as 100G." >&2
      return 2
    fi
    memory=(--mem="$MEMORY_PER_NODE")
  fi
  sweep_preflight || return 1
  if [ $dry = 0 ] && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch not found (use --dry-run to only print the commands)." >&2
    return 1
  fi

  local stamp git
  stamp="$(date +%Y%m%d-%H%M%S)"
  git="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [ "$git" != unknown ] && [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    git="${git}-dirty"
  fi
  mkdir -p "$ROOT/results/scaling" || return 1
  if [ -n "${OUT_DIR:-}" ]; then
    mkdir -p "$(dirname "$OUT_DIR")" || return 1
    mkdir "$OUT_DIR" || { echo "error: OUT_DIR must be a new directory (prevents overwriting a sweep)." >&2; return 1; }
  else
    OUT_DIR="$(mktemp -d "$ROOT/results/scaling/${SWEEP_NAME}_${stamp}_XXXXXX")" || return 1
  fi
  OUT_DIR="$(cd "$OUT_DIR" && pwd -P)" || return 1
  mkdir -p "$OUT_DIR/reports" "$OUT_DIR/slurm" "$OUT_DIR/snapshot" || return 1
  # Queued jobs must not change if the checkout is rebuilt or site.sh is edited.
  cp "$BIN" "$OUT_DIR/snapshot/${MODE}_scaling" || return 1
  cp "$ROOT/scaling/site.sh" "$ROOT/scaling/job_scaling.sh" "$ROOT/scaling/collect.sh" "$ROOT/scaling/collect_remesh.sh" "$OUT_DIR/snapshot/" || return 1
  cp "$BUILD_DIR/CMakeCache.txt" "$OUT_DIR/snapshot/" || return 1
  git -C "$ROOT" diff HEAD -- src include CMakeLists.txt >"$OUT_DIR/snapshot/source.diff"
  for src in $(sweep_sources); do cp "$ROOT/scaling/$src" "$OUT_DIR/snapshot/" || return 1; done
  printf 'tag\tnp\tnodes\trpn\tstatus\tjob_id\n' >"$OUT_DIR/points.tsv"

  {
    echo "MFTL $SWEEP_NAME scaling sweep (driver ${MODE}_scaling)"
    echo "  created       $(date)"
    echo "  commit        $git"
    echo "  binary        $BIN"
    echo "  ranks         $RANKS"
    echo "  point source  $POINT_SOURCE   candidate nodes $NODES   candidate ranks $REQUESTED_RANKS"
    echo "  driver args   $RUN_ARGS"
    echo "  job repeats   $JOB_REPEATS   time limit $TIME_LIMIT   cores/node $CORES_PER_NODE   ranks/node ${RANKS_PER_NODE:-$CORES_PER_NODE}   max nodes $MAX_NODES"
    echo "  sbatch extra  ${SBATCH_EXTRA:-none}   exclusive $EXCLUSIVE"
    echo "  memory/node   ${MEMORY_PER_NODE:-default}   (default = no --mem; 0 = all configured node memory)"
    echo "  cores/node    $CORES_PER_NODE"
    echo "  snapshot      $OUT_DIR/snapshot"
    echo "  binary cksum  $(cksum "$OUT_DIR/snapshot/${MODE}_scaling")"
  } >"$OUT_DIR/sweep.txt"

  local excl=()
  [ "$EXCLUSIVE" = 1 ] && excl=(--exclusive)
  local extra=()
  # SBATCH_EXTRA is a whitespace-delimited word list.
  [ -n "$SBATCH_EXTRA" ] && read -r -a extra <<< "$SBATCH_EXTRA"

  local rpn="${RANKS_PER_NODE:-$CORES_PER_NODE}"
  echo "campaign: $SWEEP_NAME; eligible ranks: ${RANKS:-none}; ranks/node=$rpn; MAX_NODES=$MAX_NODES; memory/node=${MEMORY_PER_NODE:-default}"
  local ids=() rep nodes tpn tag id submit_errors=0
  # Submit repeats round robin. Slurm still chooses execution order and nodes.
  for rep in $(seq 1 "$JOB_REPEATS"); do
    for p in $REQUESTED_RANKS; do
      tag="$(printf 'np%05d' "$p")"
      [ "$JOB_REPEATS" -gt 1 ] && tag="${tag}_r${rep}"
      # Keep the requested rank density. Never silently repack a large point.
      nodes=$(((p + rpn - 1) / rpn))
      if [ "$nodes" -gt "$MAX_NODES" ]; then
        echo "skip np=$p: needs $nodes nodes > MAX_NODES=$MAX_NODES"
        printf '%s\t%s\t%s\t-\tskipped\t-\n' "$tag" "$p" "$nodes" >>"$OUT_DIR/points.tsv"
        continue
      fi
      tpn=$(((p + nodes - 1) / nodes))

      local cmd=(sbatch --parsable
        --job-name="mftl-${SWEEP_NAME}-np${p}"
        --nodes="$nodes" --ntasks="$p" --ntasks-per-node="$tpn" --cpus-per-task=1
        ${memory[@]+"${memory[@]}"}
        --time="$TIME_LIMIT"
        --output="$OUT_DIR/slurm/${tag}-%j.out"
        ${excl[@]+"${excl[@]}"} ${extra[@]+"${extra[@]}"}
        --export=ALL
        "$OUT_DIR/snapshot/job_scaling.sh")
      local vars=(MFTL_ROOT="$ROOT" MODE="$MODE" NRANKS="$p" OUT_DIR="$OUT_DIR" TAG="$tag" GIT="$git" RUN_ARGS="$RUN_ARGS"
        MFTL_EXPECT_NODES="$nodes" MFTL_EXPECT_RPN="$tpn")

      if [ $dry = 1 ]; then
        printf 'env'
        printf ' %q' "${vars[@]}" "${cmd[@]}"
        printf '\n'
        ids+=("DRY$p")
        printf '%s\t%s\t%s\t%s\tdry-run\t-\n' "$tag" "$p" "$nodes" "$tpn" >>"$OUT_DIR/points.tsv"
      else
        if ! id="$(env "${vars[@]}" "${cmd[@]}")"; then
          echo "error: sbatch failed for np=$p" >&2
          printf '  request: nodes=%s tasks=%s tasks/node=%s memory/node=%s; sbatch extras: %s\n' \
            "$nodes" "$p" "$tpn" "${MEMORY_PER_NODE:-default}" "${SBATCH_EXTRA:-none}" >&2
          echo "  Check sinfo -N and scontrol show partition for CPU/memory limits; MEMORY_PER_NODE=default omits --mem." >&2
          echo "sbatch submission failed" >"$OUT_DIR/reports/${tag}.failed"
          printf '%s\t%s\t%s\t%s\tsubmit-failed\t-\n' "$tag" "$p" "$nodes" "$tpn" >>"$OUT_DIR/points.tsv"
          submit_errors=1
          continue
        fi
        id="${id%%;*}"
        ids+=("$id")
        printf '%s\t%s\t%s\t%s\tsubmitted\t%s\n' "$tag" "$p" "$nodes" "$tpn" "$id" >>"$OUT_DIR/points.tsv"
        printf 'submitted np=%-5s nodes=%-3s tasks/node=%-3s job %s\n' "$p" "$nodes" "$tpn" "$id"
      fi
    done
  done

  if [ ${#ids[@]} -eq 0 ]; then
    echo "nothing submitted: check MAX_NODES/NODES/RANKS and any sbatch errors above." >&2
    return 1
  fi

  # The collector: runs whatever the points' fate (afterany), so a failed or
  # timed-out point still yields a summary that names it.
  local dep
  dep="afterany:$(
    IFS=:
    echo "${ids[*]}"
  )"
  local ccmd=(sbatch --parsable --job-name="mftl-${SWEEP_NAME}-collect" --dependency="$dep"
    --nodes=1 --ntasks=1 --time=00:15:00 --output="$OUT_DIR/slurm/collect-%j.out"
    ${extra[@]+"${extra[@]}"} "$OUT_DIR/snapshot/collect.sh" "$OUT_DIR")
  if [ $dry = 1 ]; then
    printf '%q ' "${ccmd[@]}"
    printf '\n\n(dry run: nothing submitted; configuration and snapshot saved in %s)\n' "$OUT_DIR"
  else
    if id="$("${ccmd[@]}")"; then
      echo "submitted collector job ${id%%;*} (runs after all points)"
    else
      echo "warning: collector job not submitted; run scaling/collect.sh $OUT_DIR by hand later." >&2
      submit_errors=1
    fi
    ln -sfn "$OUT_DIR" "$ROOT/results/scaling/${SWEEP_NAME}_latest"
    echo
    echo "results:   $OUT_DIR"
    echo "           (also $ROOT/results/scaling/${SWEEP_NAME}_latest)"
    echo "summary:   $OUT_DIR/summary.txt  -- written when the last point has finished"
    echo "watch:     squeue -u \$USER -o '%.18i %.24j %.9P %.2t %.6D %R'"
    echo "queue:     Resources = waiting for nodes; Dependency = collector waiting for benchmark jobs."
    echo "by hand:   scaling/collect.sh $OUT_DIR   (any time; partial sweeps are fine)"
  fi
  return "$submit_errors"
}

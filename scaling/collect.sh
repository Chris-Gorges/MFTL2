#!/bin/bash
#
# Merges every report of ONE sweep into ONE text file, <sweep>/summary.txt:
# the scaling table, the redistribution profile per rank count, which part of the
# redistribution stops scaling, traffic and partition per rank count, the points
# that failed or are missing, and then every full per-job report.
#
#   scaling/collect.sh results/scaling/strong_20261013-120000
#   scaling/collect.sh results/scaling/weak_latest
#
# Runs automatically as the last job of a sweep; safe to run by hand at any time,
# also on a partial sweep. Plain bash + POSIX awk: nothing to install.

set -uo pipefail
# The reports are written in the C locale ('.' decimal point). awk and sort honour
# LC_NUMERIC, and under e.g. de_DE "0.19" + 0 is 0 -- so pin the locale.
export LC_ALL=C

DIR="${1:?usage: collect.sh <sweep directory>}"
DIR="$(cd "$DIR" && pwd -P)" || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORES_PER_NODE=36
# Read historical settings, not today's site.sh.
if [ -f "$DIR/sweep.txt" ]; then
  saved_cores="$(sed -n 's/^  cores\/node *//p' "$DIR/sweep.txt")"
  [ -n "$saved_cores" ] && CORES_PER_NODE="$saved_cores"
fi
OUT="$DIR/summary.txt"

shopt -s nullglob
reports=("$DIR"/reports/*.txt)
failed=("$DIR"/reports/*.failed)
valid_reports=()
rejected=()
for f in ${reports[@]+"${reports[@]}"}; do
  if [ ! -f "${f%.txt}.failed" ] && awk '
    /^@summary / {
      n++
      for (i=2; i<=NF; i++) { split($i, a, "="); v[a[1]]=a[2] }
    }
    END { exit !(n == 1 && v["ok"] == 1 && v["np"] > 0 && v["ms_step"] > 0) }
  ' "$f"; then
    valid_reports+=("$f")
  else
    rejected+=("$(basename "$f" .txt)")
  fi
done

# Every successful job, before fastest-job selection: easy to plot or analyse
# statistically. Full profiles remain in the individual reports and summary.
samples_tmp="$DIR/samples.tsv.tmp.$$"
summary_tmp="$OUT.tmp.$$"
trap 'rm -f "$samples_tmp" "$summary_tmp"' EXIT
{
  printf 'report\tmode\tnp\tnodes\trpn\ttris\ttris_rank\tms_min\tms_median\tms_max\trep_spread\tprofiled\tsync_phases\timbalance\trepeats\trequested_repeats\tbudget_stop\tinit_s\tinit_batches\tinit_batch_pool_mb\n'
  for f in ${valid_reports[@]+"${valid_reports[@]}"}; do
    awk -v report="$(basename "$f")" '/^@summary / {
      for (i=2; i<=NF; i++) { split($i,a,"="); v[a[1]]=a[2] }
      printf "%s", report
      n=split("mode np nodes rpn tris tris_rank ms_step ms_median ms_max rep_spread profiled sync_phases imbalance repeats requested_repeats budget_stop init_s init_batches init_batch_pool_mb", keys, " ")
      for (i=1;i<=n;i++) printf "\t%s", (keys[i] in v) ? v[keys[i]] : "NA"
      printf "\n"
    }' "$f"
  done
} >"$samples_tmp"
mv "$samples_tmp" "$DIR/samples.tsv"

read -r -d '' AWK_PROG <<'AWK'
function num(i, k) { return R[i, k] + 0 }
function has(i, k) { return ((i, k) in R) }
function f3(x) { return sprintf("%.3f", x) }
function cell(i, k, fmt) { return has(i, k) ? sprintf(fmt, num(i, k)) : "-" }

{
  split("", F)
  for (j = 2; j <= NF; j++) { e = index($j, "="); if (e > 1) F[substr($j, 1, e - 1)] = substr($j, e + 1) }
  p = F["np"] + 0
  if (n > 0 && P[n] == p) {                       # a job repeat of the same rank count
    cnt[n]++
    t = F["ms_step"] + 0
    J[n, cnt[n]] = t
    if (t > jmax[n]) jmax[n] = t
    if (t >= num(n, "ms_step")) next              # keep the fastest job
    jmin[n] = t
  } else {
    n++; P[n] = p; cnt[n] = 1; J[n, 1] = jmin[n] = jmax[n] = F["ms_step"] + 0
  }
  for (k in F) R[n, k] = F[k]
}

END {
  if (n == 0) { print "  (no completed reports yet)"; exit }
  mode = R[1, "mode"]; strong = (mode == "strong")

  nb = 0; rb = 0                                   # one-node baseline; first P >= 2
  for (i = 1; i <= n; i++) { if (num(i, "nodes") == 1) nb = i; if (!rb && P[i] >= 2) rb = i }

  printf "SCALING  (%s, motion %s, %s bodies x refine %s; times are ms per timestep)\n", toupper(mode), R[1, "motion"], R[n, "bodies"], R[1, "refine"]
  if (strong) {
    printf "  speedup and eff are against P = %d; eff/node against one node (P = %d)\n", P[1], (nb ? P[nb] : 0)
  } else {
    printf "  weak efficiency = T(P0)/T(P): against P = %d, and eff/node against one node (P = %d)\n", P[1], (nb ? P[nb] : 0)
  }
  printf "\n%6s %5s %4s %-9s %9s %10s %7s %8s %6s %8s %9s %6s %8s %8s %8s %8s %7s %6s %6s %3s\n", \
    "P", "nodes", "rpn", "cuts", "tri/rank", "ms/step", "jobs", strong ? "speedup" : "-", "eff", "eff/node", \
    "redist", "red%", "ghost", "geom", "post", "normals", "advect", "comm%", "imbal", "ok"
  for (i = 1; i <= n; i++) {
    T = num(i, "ms_step")
    if (strong) { sp = num(1, "ms_step") / T; ef = sp * P[1] / P[i]; en = (nb && i >= nb) ? num(nb, "ms_step") * P[nb] / (T * P[i]) : -1 }
    else        { sp = -1; ef = num(1, "ms_step") / T; en = (nb && i >= nb) ? num(nb, "ms_step") / T : -1 }
    jobs = (cnt[i] > 1) ? sprintf("%dx%.0f%%", cnt[i], 100 * (jmax[i] - jmin[i]) / jmin[i]) : "1"
    printf "%6d %5s %4s %-9s %9s %10.3f %7s %8s %6.3f %8s %9.3f %6.1f %8.3f %8.3f %8.3f %8.3f %7.3f %6s %6.3f %3s\n", \
      P[i], R[i, "nodes"], R[i, "rpn"], R[i, "cuts"], R[i, "tris_rank"], T, jobs, (sp < 0) ? "-" : sprintf("%.2f", sp), ef, \
      (en < 0) ? "-" : sprintf("%.3f", en), num(i, "redist"), 100 * num(i, "redist") / T, num(i, "ghost"), \
      num(i, "geom"), num(i, "postgeom"), num(i, "normals"), num(i, "advect"), \
      (num(i, "comm_pct") < 0) ? "n/a" : sprintf("%.1f", num(i, "comm_pct")), num(i, "imbalance"), num(i, "ok") ? "ok" : "BAD"
  }

  printf "\nREPEATS AND COST  (main tables use the fastest valid repeat of the fastest valid job)\n"
  printf "  Job median below is the median of job minima; samples.tsv also retains each job's repeat median.\n"
  printf "  core-ms = P*T; node-ms = allocated nodes*T (captures reserved idle cores).\n\n"
  printf "%6s %8s %10s %10s %10s %10s %10s %10s\n", "P", "jobs", "job min", "job median", "job max", "repeat med", "core-ms", "node-ms"
  for (i = 1; i <= n; i++) {
    for (j=2; j<=cnt[i]; j++) {
      x=J[i,j]; k=j-1
      while (k>=1 && J[i,k]>x) { J[i,k+1]=J[i,k]; k-- }
      J[i,k+1]=x
    }
    mid=int((cnt[i]+1)/2); med=J[i,mid]
    if (cnt[i]%2 == 0) med=(med+J[i,mid+1])/2
    printf "%6d %8d %10.4f %10.4f %10.4f %10s %10.3f %10.3f\n", P[i], cnt[i], jmin[i], med, jmax[i], cell(i,"ms_median","%.4f"), num(i,"ms_step")*P[i], num(i,"ms_step")*num(i,"nodes")
  }

  split("setup owners masks dest pack group reserve kgather kplace relink arrivals twins teardown", W, " "); nw = 13
  split("route skipvote xcounts xpost wait", C, " "); nc = 5

  printf "\nREDISTRIBUTION, LOCAL WORK  (mean over ranks, ms per step; should fall like 1/P strong, stay flat weak)\n\n%6s", "P"
  for (j = 1; j <= nw; j++) printf " %8s", W[j]
  printf " %8s\n", "WORK"
  for (i = 1; i <= n; i++) {
    printf "%6d", P[i]
    for (j = 1; j <= nw; j++) printf " %8s", cell(i, "rb_" W[j], "%.4f")
    printf " %8s\n", cell(i, "redist_work", "%.4f")
  }

  printf "\nREDISTRIBUTION, COMMUNICATION + SYNC  (mean over ranks, ms per step; route/skipvote/xcounts/wait also absorb imbalance)\n\n%6s", "P"
  for (j = 1; j <= nc; j++) printf " %8s", C[j]
  printf " %8s %8s %9s %9s %8s %9s\n", "COMM", "TOTAL", "phase", "max rank", "comm%", "residue"
  for (i = 1; i <= n; i++) {
    printf "%6d", P[i]
    for (j = 1; j <= nc; j++) printf " %8s", cell(i, "rb_" C[j], "%.4f")
    tot = num(i, "redist_lib")
    printf " %8.4f %8.4f %9.4f %9.4f %8s %9.4f\n", num(i, "redist_comm"), tot, num(i, "redist"), num(i, "redist_max"), \
      (tot > 0) ? sprintf("%.1f", 100 * num(i, "redist_comm") / tot) : "-", tot - num(i, "redist_work") - num(i, "redist_comm")
  }

  if (rb && rb < n) {
    printf "\nWHICH PART STOPS SCALING  (efficiency of each redistribution bucket against P = %d, the first point with\n", P[rb]
    printf "a real exchange; strong: t0*P0/(t*P), weak: t0/t. 1.00 = ideal, <1 = scales worse than the work)\n\n%6s", "P"
    split("TOTAL WORK COMM owners dest kgather relink arrivals twins setup skipvote wait xcounts", E, " "); ne = 13
    split("redist_lib redist_work redist_comm rb_owners rb_dest rb_kgather rb_relink rb_arrivals rb_twins rb_setup rb_skipvote rb_wait rb_xcounts", EK, " ")
    for (j = 1; j <= ne; j++) printf " %8s", E[j]
    printf "\n"
    for (i = rb; i <= n; i++) {
      printf "%6d", P[i]
      for (j = 1; j <= ne; j++) {
        t0 = num(rb, EK[j]); t = num(i, EK[j])
        if (t0 <= 0 || t <= 0) { printf " %8s", "-"; continue }
        printf " %8.3f", strong ? (t0 * P[rb]) / (t * P[i]) : t0 / t
      }
      printf "\n"
    }
  }

  printf "\nTRAFFIC, PARTITION, MEMORY, SETUP  (per step; MB = MiB)\n\n"
  printf "%6s %9s %9s %7s %7s %7s %6s %8s %6s %6s %8s %9s %9s %8s %8s %8s %9s\n", "P", "MBsent/r", "MBsentMax", "msgs/r", \
    "owner%", "skip%", "halo%", "bodies/r", "degree", "global", "ws/rank", "rss/node", "rss/rank", "step p90", "gen s", "scatter", "perturb"
  for (i = 1; i <= n; i++) {
    printf "%6d %9.4f %9.4f %7.2f %7.3f %7.1f %6.2f %8.1f %6s %6s %6.1fkB %7.0fMB %7.0fMB %8.3f %8.2f %8.2f %9.4f\n", P[i], \
      num(i, "mb_sent_rank"), num(i, "mb_sent_max"), num(i, "msgs_rank"), num(i, "owner_pct"), num(i, "skip_pct"), \
      num(i, "halo_pct"), num(i, "bodies_rank"), R[i, "degree"], R[i, "route_global"], num(i, "ws_rank_kb"), \
      num(i, "rss_node_mb"), num(i, "rss_rank_mb"), num(i, "step_p90"), num(i, "gen_s"), num(i, "scatter_s"), num(i, "perturb_ms")
  }

  bad = 0
  for (i = 1; i <= n; i++) { if (!num(i, "profiled")) bad = bad " " P[i]; }
  if (bad) printf "\n!! built WITHOUT MFTL_COMM_PROFILE at P =%s: their redistribution buckets are zero.\n", bad
  bad = 0
  for (i = 1; i <= n; i++) { if (!num(i, "optimised")) bad = bad " " P[i]; }
  if (bad) printf "\n!! NOT an optimised NDEBUG build at P =%s: do not quote those timings.\n", bad
  bad = ""
  for (i = 1; i <= n; i++) { if (num(i, "sync_phases")) bad = bad " " P[i] }
  if (bad != "") printf "\n!! Diagnostic phase barriers enabled at P =%s: do not quote these wall clocks.\n", bad
  bad = ""
  for (i = 1; i <= n; i++) { if (num(i, "budget_stop")) bad = bad " " P[i] }
  if (bad != "") printf "\n!! Soft runtime budget reduced the number of complete repeats at P =%s; see samples.tsv.\n", bad
}
AWK

sweep_ranks=""
[ -f "$DIR/sweep.txt" ] && sweep_ranks="$(sed -n 's/^  ranks *//p' "$DIR/sweep.txt")"

{
  echo "##############################################################################"
  echo "# MFTL scaling summary -- $(basename "$DIR")"
  echo "# collected $(date) from ${#reports[@]} report(s), ${#valid_reports[@]} valid"
  echo "##############################################################################"
  echo
  [ -f "$DIR/sweep.txt" ] && cat "$DIR/sweep.txt" && echo

  if [ ${#valid_reports[@]} -gt 0 ]; then
    grep -h '^@summary' "${valid_reports[@]}" |
      awk '{ for (i = 2; i <= NF; i++) if (substr($i, 1, 3) == "np=") print substr($i, 4) "\t" $0 }' |
      sort -n -k1,1 | cut -f2- |
      awk -v cores="$CORES_PER_NODE" "$AWK_PROG"
  else
    echo "(no reports yet)"
  fi

  echo
  echo "HOW TO READ"
  echo "  ms/step      the barriered wall clock of the timed loop per step: THE scaling number."
  echo "  redist..     the MFTLRedistribute phase; red% its share of the step. 'phase' is what the"
  echo "               harness timed, TOTAL the library's own timer, 'max rank' the busiest rank."
  echo "  WORK / COMM  local rebuild vs exchange + synchronisation inside the redistribution. Strong:"
  echo "               WORK should fall like 1/P; COMM falls slower (surface/volume, latency) and"
  echo "               a rising communication share signals overhead; choose useful P from measured T(P),"
  echo "               efficiency and core/node cost, not where WORK equals COMM. Weak: aim for flat times;"
  echo "               'setup' and ws/rank are O(nproc) by design and grow slowly with P."
  echo "  wait, skipvote, route, xcounts  the first collectives after local work: they absorb"
  echo "               load imbalance. Equal owned-triangle counts do not exclude unequal ghost/body"
  echo "               work, NUMA effects or contention; MPI profiling is needed to distinguish causes."
  echo "  comm%        exchange + sync buckets of redistribution AND ghost exchanges / wall clock."
  echo "  degree       distinct neighbour ranks; 26 only once every axis has >= 3 cuts (P >= 27)."
  echo "  global       redistributions that fell back to the global route (must be 0)."
  echo "  perturb      clock-read cost estimate only; compare a profiling-OFF build for total perturbation."
  echo "  rss/node     sum of per-rank RSS peaks: upper bound, not a simultaneous node high-water mark."
  echo "  samples.tsv  all successful jobs, including repeat medians; use them for error bars."
  echo "  P <= ${CORES_PER_NODE}      one node: shared memory bandwidth, not the interconnect."

  missing=""
  if [ -f "$DIR/points.tsv" ]; then
    while read -r tag p nodes rpn status job; do
      [ "$tag" = tag ] && continue
      [ "$status" = skipped ] && { echo "  skipped $tag: needs $nodes nodes"; continue; }
      if [ -f "$DIR/reports/$tag.failed" ] || ! grep -q '^@summary .* ok=1$' "$DIR/reports/$tag.txt" 2>/dev/null; then
        missing="$missing $tag(job=$job,status=$status)"
      fi
    done <"$DIR/points.tsv"
  else
    for p in $sweep_ranks; do
      tag="$(printf 'np%05d' "$p")"
      compgen -G "$DIR/reports/${tag}*.txt" >/dev/null || missing="$missing $p"
    done
  fi
  if [ ${#rejected[@]} -gt 0 ]; then
    echo
    echo "EXCLUDED FROM TIMING TABLES (failed or incomplete reports): ${rejected[*]}"
  fi
  if [ ${#failed[@]} -gt 0 ] || [ -n "$missing" ]; then
    echo
    echo "FAILED OR MISSING POINTS"
    for f in ${failed[@]+"${failed[@]}"}; do
      echo "  $(basename "$f" .failed):"
      sed 's/^/    /' "$f"
    done
    [ -n "$missing" ] && echo "  no verified report (queued, running, submission failed, or died):$missing"
  fi

  for f in ${reports[@]+"${reports[@]}"}; do
    echo
    echo
    echo "################ $(basename "$f") ################"
    cat "$f"
  done
} >"$summary_tmp"
mv "$summary_tmp" "$OUT"

echo "wrote $OUT (${#reports[@]} report(s))"

#!/bin/bash
#
# Merges every report of ONE remeshing sweep into <sweep>/summary.txt: per
# driver mode (dispersed, or the spanning modes once and nested) the scaling
# table, the remesh call's own profile, the accuracy at the end of the last
# trajectory, the once/nested comparison, the dispersed digest check, the points
# that failed or are missing, and then every full per-job report.
#
#   scaling/collect_remesh.sh results/scaling/remesh_spanning_strong_latest
#
# collect.sh hands a remeshing sweep to this script, so either name works. Runs
# automatically as the last job of a sweep; safe by hand at any time, also on a
# partial sweep. Plain bash + POSIX awk.

set -uo pipefail
export LC_ALL=C  # the reports use '.' decimals; awk honours LC_NUMERIC

DIR="${1:?usage: collect_remesh.sh <sweep directory>}"
DIR="$(cd "$DIR" && pwd -P)" || exit 2
CORES_PER_NODE=36
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
  # Valid: at least one @summary line, every one of them ok=1.
  if [ ! -f "${f%.txt}.failed" ] && awk '
    /^@summary / { n++; if ($NF != "ok=1") bad = 1 }
    END { exit !(n >= 1 && !bad) }
  ' "$f"; then
    valid_reports+=("$f")
  else
    rejected+=("$(basename "$f" .txt)")
  fi
done

samples_tmp="$DIR/samples.tsv.tmp.$$"
summary_tmp="$OUT.tmp.$$"
trap 'rm -f "$samples_tmp" "$summary_tmp"' EXIT
KEYS="mode regime spanning np nodes rpn cuts bodies refine R_over_h tris tris_rank ms_step remesh migrate sweep reference projection ids phases migrations frozen_pct imbalance dv_pct dv_worst_pct dist_rms_pct dist_max_pct qmin qmean oob_pct not_converged missed digest"
{
  printf 'report'
  for k in $KEYS; do printf '\t%s' "$k"; done
  printf '\n'
  for f in ${valid_reports[@]+"${valid_reports[@]}"}; do
    awk -v report="$(basename "$f")" -v keys="$KEYS" '/^@summary / {
      split("", v)
      for (i = 2; i <= NF; i++) { e = index($i, "="); if (e > 1) v[substr($i, 1, e - 1)] = substr($i, e + 1) }
      n = split(keys, K, " ")
      printf "%s", report
      for (i = 1; i <= n; i++) printf "\t%s", (K[i] in v) ? v[K[i]] : "NA"
      printf "\n"
    }' "$f"
  done
} >"$samples_tmp"
mv "$samples_tmp" "$DIR/samples.tsv"

read -r -d '' AWK_PROG <<'AWK'
function num(m, i, k) { return R[m, i, k] + 0 }
function val(m, i, k) { return R[m, i, k] }

{
  split("", F)
  for (j = 2; j <= NF; j++) { e = index($j, "="); if (e > 1) F[substr($j, 1, e - 1)] = substr($j, e + 1) }
  m = F["spanning"]; p = F["np"] + 0
  if (!(m in nm)) { modes[++nmodes] = m; nm[m] = 0 }
  i = nm[m]
  if (i > 0 && P[m, i] == p) {                    # a job repeat of the same rank count: keep the fastest
    jobs[m, i]++
    if (F["ms_step"] + 0 >= num(m, i, "ms_step")) next
  } else {
    i = ++nm[m]; P[m, i] = p; jobs[m, i] = 1
  }
  for (k in F) R[m, i, k] = F[k]
}

END {
  if (nmodes == 0) { print "  (no completed reports yet)"; exit }
  strong = (R[modes[1], 1, "mode"] == "remesh_strong")
  printf "REMESHING SCALING  (%s, %s regime, %s bodies x refine %s, %s sizing, projection %s; ms per timestep, max over ranks)\n", \
    strong ? "STRONG" : "WEAK", R[modes[1], 1, "regime"], R[modes[1], nm[modes[1]], "bodies"], R[modes[1], 1, "refine"], \
    R[modes[1], 1, "sizing"], R[modes[1], 1, "proj_mode"]
  printf "  %s against the first point of each mode; eff/node against one full node (P = %d) where present\n", \
    strong ? "speedup and efficiency" : "weak efficiency T(P0)/T(P)", cores

  for (a = 1; a <= nmodes; a++) {
    m = modes[a]; n = nm[m]; nb = 0
    for (i = 1; i <= n; i++) if (num(m, i, "nodes") == 1 && P[m, i] == cores) nb = i
    printf "\n== %s ==\n", m
    printf "%6s %5s %-9s %9s %6s %10s %8s %6s %8s %9s %8s %8s %8s %8s %8s %7s %6s %7s %6s %4s\n", \
      "P", "nodes", "cuts", "tri/rank", "R/h", "ms/step", strong ? "speedup" : "-", "eff", "eff/node", \
      "remesh", "migrate", "sweep", "ref", "proj", "ids", "places", "migr", "frozen%", "imbal", "jobs"
    for (i = 1; i <= n; i++) {
      T = num(m, i, "ms_step")
      if (strong) { sp = num(m, 1, "ms_step") / T; ef = sp * P[m, 1] / P[m, i]; en = (nb && i >= nb) ? num(m, nb, "ms_step") * P[m, nb] / (T * P[m, i]) : -1 }
      else        { sp = -1; ef = num(m, 1, "ms_step") / T; en = (nb && i >= nb) ? num(m, nb, "ms_step") / T : -1 }
      printf "%6d %5s %-9s %9s %6.3f %10.3f %8s %6.3f %8s %9.3f %8.3f %8.3f %8.3f %8.3f %8.3f %7.2f %6.2f %7.1f %6.3f %4d\n", \
        P[m, i], val(m, i, "nodes"), val(m, i, "cuts"), val(m, i, "tris_rank"), num(m, i, "R_over_h"), T, (sp < 0) ? "-" : sprintf("%.2f", sp), ef, \
        (en < 0) ? "-" : sprintf("%.3f", en), num(m, i, "remesh"), num(m, i, "migrate"), num(m, i, "sweep"), num(m, i, "reference"), \
        num(m, i, "projection"), num(m, i, "ids"), num(m, i, "phases"), num(m, i, "migrations"), num(m, i, "frozen_pct"), num(m, i, "imbalance"), jobs[m, i]
    }
    printf "\n  accuracy at T (the flow is the identity at T: every body should be its initial sphere again)\n"
    printf "%6s %11s %10s %10s %10s %10s %8s %9s %8s %7s %7s %18s\n", \
      "P", "triangles", "dV %", "worst %", "rms %R", "max %R", "q min", "q mean", "oob %", "!conv", "missed", "digest"
    for (i = 1; i <= n; i++)
      printf "%6d %11s %10.4f %10.4f %10.4f %10.3f %8.4f %9.5f %8.3f %7s %7s %18s\n", P[m, i], val(m, i, "tris"), num(m, i, "dv_pct"), \
        num(m, i, "dv_worst_pct"), num(m, i, "dist_rms_pct"), num(m, i, "dist_max_pct"), num(m, i, "qmin"), num(m, i, "qmean"), \
        num(m, i, "oob_pct"), val(m, i, "not_converged"), val(m, i, "missed"), val(m, i, "digest")
    if (m == "dispersed" && strong && n > 1) {
      same = 1
      for (i = 2; i <= n; i++) if (val(m, i, "digest") != val(m, 1, "digest")) same = 0
      printf "\n  DISPERSED DIGEST %s: the dispersed answer is the serial answer to the bit and the rest of the\n", same ? "IDENTICAL at every P" : "DIFFERS between rank counts -- a bug"
      printf "  step is partition-independent, so every point of a strong sweep must end on the same digest.\n"
    }
  }

  if (("once" in nm) && ("nested" in nm)) {
    printf "\n== once against nested, per P (ratio nested/once; accuracy differences nested - once) ==\n"
    printf "%6s %10s %10s %10s %10s %10s %10s %10s %10s\n", "P", "ms once", "ms nested", "ratio", "migr o/n", "d dV %", "d rms %R", "d q min", "d tris"
    for (i = 1; i <= nm["once"]; i++) {
      for (j = 1; j <= nm["nested"]; j++) if (P["nested", j] == P["once", i]) break
      if (j > nm["nested"]) continue
      printf "%6d %10.3f %10.3f %10.3f %4.2f/%-5.2f %10.4f %10.4f %10.4f %10.0f\n", P["once", i], num("once", i, "ms_step"), num("nested", j, "ms_step"), \
        num("nested", j, "ms_step") / num("once", i, "ms_step"), num("once", i, "migrations"), num("nested", j, "migrations"), \
        num("nested", j, "dv_pct") - num("once", i, "dv_pct"), num("nested", j, "dist_rms_pct") - num("once", i, "dist_rms_pct"), \
        num("nested", j, "qmin") - num("once", i, "qmin"), num("nested", j, "tris") - num("once", i, "tris")
    }
  }
}
AWK

{
  echo "##############################################################################"
  echo "# MFTL remeshing scaling summary -- $(basename "$DIR")"
  echo "# collected $(date) from ${#reports[@]} report(s), ${#valid_reports[@]} valid"
  echo "##############################################################################"
  echo
  [ -f "$DIR/sweep.txt" ] && cat "$DIR/sweep.txt" && echo

  if [ ${#valid_reports[@]} -gt 0 ]; then
    grep -h '^@summary' "${valid_reports[@]}" |
      awk '{ for (i = 2; i <= NF; i++) if (substr($i, 1, 3) == "np=") print substr($i, 4) "\t" $0 }' |
      sort -s -n -k1,1 | cut -f2- |
      awk -v cores="$CORES_PER_NODE" "$AWK_PROG"
  else
    echo "(no reports yet)"
  fi

  echo
  echo "HOW TO READ"
  echo "  ms/step     barriered wall clock of one remeshed timestep (advect, ghost sync, geometry, remesh"
  echo "              with its closing redistribution): THE scaling number."
  echo "  remesh      MFTLRemeshEnsemble; migrate/sweep/ref/proj/ids are its own timers (maximum over ranks"
  echo "              per call): migrations incl. the closing redistribution, the remeshing work, building"
  echo "              the projection reference (with its halo), the projection, the id collectives."
  echo "              'ids' absorbs the wait for the slowest rank."
  echo "  places      cut-plane placements per call; migr migrations per call; frozen% the band share that"
  echo "              travels without being remeshed (spanning). The dispersed driver has one placement."
  echo "  accuracy    after the last trajectory: volume against the generated mesh, vertex distance to the"
  echo "              exact sphere, triangle quality, edges outside the length band. A spanning mode should"
  echo "              match the dispersed / serial answer within the remesher's order noise, not to the bit."
  echo "  !conv       remesh calls that did not converge in the budget; missed: vertices the projection"
  echo "              could not place (both should be 0)."
  echo "  samples.tsv every valid report, one row per mode."
  echo "  P <= ${CORES_PER_NODE}      one node: shared memory bandwidth, not the interconnect."

  missing=""
  if [ -f "$DIR/points.tsv" ]; then
    while read -r tag p nodes rpn status job; do
      [ "$tag" = tag ] && continue
      [ "$status" = skipped ] && { echo "  skipped $tag: needs $nodes nodes"; continue; }
      if [ -f "$DIR/reports/$tag.failed" ] || ! grep -q '^@summary .* ok=1$' "$DIR/reports/$tag.txt" 2>/dev/null ||
         grep -q '^@summary .* ok=0$' "$DIR/reports/$tag.txt" 2>/dev/null; then
        missing="$missing $tag(job=$job,status=$status)"
      fi
    done <"$DIR/points.tsv"
  fi
  if [ ${#rejected[@]} -gt 0 ]; then
    echo
    echo "EXCLUDED FROM THE TABLES (failed or incomplete reports): ${rejected[*]}"
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

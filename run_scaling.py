#!/usr/bin/env python3
"""
Strong, weak and granularity scaling of the MFTL advection timestep
===================================================================

Sweeps ``examples/parallel_advection_scaling.c`` over a set of rank counts,
accumulates one JSON file per run, and re-renders every plot and the report from
the whole accumulated store.

Three axes, and only two of them are scaling laws:

  STRONG       total work fixed, P grows.            Ideal: T proportional to 1/P.
  WEAK         work PER RANK fixed, P and total work grow together. The domain is
               the per-rank box times the process grid, so every rank owns a
               constant physical volume holding a constant number of bodies.
               Ideal: a flat line.
  GRANULARITY  the same triangle count, split differently between bodies and
               triangles-per-body. NOT a scaling law -- it changes the constant in
               both of the above. Swept as an ISO-TRIANGLE ladder: 4096 bodies at
               refine 1 and 16 bodies at refine 5 are both 327 680 triangles, and
               they do not cost the same.

Two properties of the accumulation are deliberate:

  * Every invocation regenerates EVERY plot from the whole raw store. Adding a
    rank count means running it (or dropping its JSON in from the cluster) and
    re-rendering. That is idempotent, order-independent, and survives a partly
    failed sweep; editing plots in place is none of those.

  * Repeats are taken ROUND ROBIN over the rank counts, not in blocks.
    docs/performance-analysis.md sec 7 records the same binary measuring 9.4 ms and
    6.1 ms in two sessions purely from machine state, and states the rule the whole
    document follows: interleave the variants, report minima, never quote a run
    taken in a block. A scaling curve measured in blocks is exactly the shape that
    trap produces -- a thermal ramp that looks like a scaling law.

Usage:
    python3 run_scaling.py --preset laptop --all
    python3 run_scaling.py --preset laptop --case medium --mode strong --ranks 1,2,4,8
    python3 run_scaling.py --plots-only
    python3 run_scaling.py --smoke

Options:
    --preset laptop|cluster   rank set, repeat count and default cases
    --case NAME               one case, repeatable; default is the preset's list
    --mode strong|weak|both   default both
    --ranks A,B,C             override the preset's rank set
    --repeats N               override the preset's repeat count
    --steps N                 override the preset's timed step count
    --iso                     also run the iso-triangle granularity ladder
    --io                      also run the two I/O configurations
    --no-build                skip the Release rebuild
    --plots-only              render from the existing store, run nothing
    --smoke                   one tiny run at np=2, to check the wiring
    --results DIR             store location (default results/scaling)

Examples:
    python3 run_scaling.py --preset laptop --all
    python3 run_scaling.py --preset cluster --plots-only
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
BINARY = os.path.join(BUILD_DIR, "parallel_advection_scaling")
DEFAULT_RESULTS = os.path.join(PROJECT_ROOT, "results", "scaling")

#: One sphere at refinement r has 20*4^r triangles.
TRIS_PER_BODY = {r: 20 * (4 ** r) for r in range(0, 8)}

#: The matrix. Each case is one strong curve and one weak curve; sweeping several
#: is what separates "scaling holds" from "scaling holds at this size".
CASES = {
    "tiny": {"spheres": 100, "per_rank": 100, "refine": 5},
    "medium": {"spheres": 1000, "per_rank": 1000, "refine": 5},
    "large": {"spheres": 2000, "per_rank": 2000, "refine": 6},
}

#: The iso-triangle ladder: (bodies, refine), all exactly 327 680 triangles.
#: Exact powers of four, so the ladder is exact rather than approximate. With the
#: radius derived from a fixed volume fraction, the 16-body rung has bodies
#: comparable to a rank box and the 4096-body rung has bodies far smaller than one.
ISO_LADDER = [(4096, 1), (1024, 2), (256, 3), (64, 4), (16, 5)]

PRESETS = {
    "laptop": {
        "ranks": [1, 2, 4, 8],
        "repeats": 3,
        "steps": 30,
        "warmup": 3,
        "cases": ["tiny", "medium", "large"],
        "iso_ranks": [4],
        "cores": os.cpu_count() or 4,
    },
    "cluster": {
        "ranks": [1, 2, 4, 9, 18, 36, 72, 144, 288],
        "repeats": 3,
        "steps": 50,
        "warmup": 3,
        "cases": ["tiny", "medium", "large"],
        "iso_ranks": [36],
        "cores": 36,
    },
}

# ── Palette ────────────────────────────────────────────────────────────────
#
# The categorical slots are taken in FIXED ORDER and never cycled; the ordering
# is the colour-blind-safety mechanism, not a preference. Rank count is ORDERED,
# so anything keyed by rank count uses the ordinal blue ramp instead of
# categorical hues -- an ordered quantity encoded categorically is the wrong
# mark, and it also sidesteps the eight-slot cap.

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Ordinal blue, light to dark. Five steps, not nine: the full ramp fails the
#: adjacent-lightness gate (gaps of 0.047 against a 0.06 floor), so neighbouring
#: steps would be indistinguishable. Starts at step 250, the lightest that still
#: clears 2:1 against a light surface. Validated with
#: `validate_palette.js "<ramp>" --mode light --ordinal`.
ORDINAL_BLUE = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

#: Marker shapes, so a rank-count series is never identified by colour alone.
ORDINAL_MARKERS = ["o", "s", "^", "D", "v"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"


def ordinal_steps(n):
    """Pick n evenly spaced steps from the ordinal blue ramp, with marker shapes.

    The ramp holds five validated steps. Beyond that, colour alone can no longer
    separate the series, so the marker carries the identity and the caller should
    facet instead -- hence the paired return rather than a colour list.

    :param n: How many steps are needed.
    :returns: A list of n (colour, marker) pairs, light to dark.
    """
    ramp = ORDINAL_BLUE
    if n <= 1:
        return [(ramp[len(ramp) // 2], ORDINAL_MARKERS[0])]
    last = len(ramp) - 1
    return [(ramp[int(round(i * last / (n - 1.0)))], ORDINAL_MARKERS[i % len(ORDINAL_MARKERS)]) for i in range(n)]


# ── Running ────────────────────────────────────────────────────────────────


def fail(message):
    """Print an error and stop.

    :param message: What went wrong.
    """
    print("Error: %s" % message, file=sys.stderr)
    sys.exit(1)


def git_hash():
    """The short commit of the working tree, or 'unknown'.

    :returns: A string recorded verbatim in every result, so a number can be traced.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mpiexec_command():
    """Locate the MPI launcher.

    :returns: Its path.
    """
    for name in ("mpiexec", "mpirun", "srun"):
        found = shutil.which(name)
        if found:
            return found
    fail("no mpiexec, mpirun or srun on PATH")


def wants_oversubscribe(launcher):
    """Whether this launcher needs --oversubscribe above the core count.

    Open MPI refuses to place more ranks than slots without it; MPICH and Intel
    MPI oversubscribe silently and REJECT the flag. Same version sniff as
    ``mftl_add_mpi_test_oversub`` in tests/CMakeLists.txt.

    :param launcher: Path to the launcher.
    :returns: True if the flag should be passed.
    """
    try:
        out = subprocess.run([launcher, "--version"], capture_output=True, text=True)
        text = (out.stdout or "") + (out.stderr or "")
        return "Open MPI" in text or "OpenRTE" in text
    except OSError:
        return False


def build_release():
    """Rebuild in Release. The benchmark is meaningless in any other configuration."""
    print("Building Release ...")
    try:
        subprocess.check_call([sys.executable, "build_all.py"], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        fail("build_all.py failed")


def case_args(label, mode, spheres, per_rank, refine, steps, warmup):
    """The benchmark arguments for one point of one curve.

    :param label:    Tag recorded in the result.
    :param mode:     'strong' or 'weak'.
    :param spheres:  Body count for strong mode.
    :param per_rank: Bodies per rank for weak mode.
    :param refine:   Subdivision level.
    :param steps:    Timed steps.
    :param warmup:   Untimed steps first.
    :returns: A list of command-line arguments.
    """
    args = ["--label", label, "--mode", mode, "--refine", str(refine), "--steps", str(steps), "--warmup", str(warmup), "--repeat", "1"]
    if mode == "weak":
        args += ["--spheres-per-rank", str(per_rank)]
    else:
        args += ["--spheres", str(spheres)]
    return args


def build_jobs(opts, preset):
    """Every (label, mode, ranks, args) point the sweep will visit.

    :param opts:   Parsed command line.
    :param preset: The active preset dict.
    :returns: A list of job dicts.
    """
    jobs = []
    modes = ["strong", "weak"] if opts.mode == "both" else [opts.mode]
    ranks = opts.ranks or preset["ranks"]
    steps = opts.steps or preset["steps"]
    warmup = preset["warmup"]

    for name in opts.case or preset["cases"]:
        if name not in CASES:
            fail("unknown case '%s' (have: %s)" % (name, ", ".join(sorted(CASES))))
        c = CASES[name]
        for mode in modes:
            for p in ranks:
                jobs.append({"label": name, "mode": mode, "ranks": p,
                             "args": case_args(name, mode, c["spheres"], c["per_rank"], c["refine"], steps, warmup)})

    if opts.iso:
        # The granularity ladder is a STRONG-mode family: the total triangle count
        # is held fixed and only the body/refinement split moves.
        for p in preset["iso_ranks"]:
            for bodies, refine in ISO_LADDER:
                label = "iso%dr%d" % (bodies, refine)
                jobs.append({"label": label, "mode": "strong", "ranks": p,
                             "args": case_args(label, "strong", bodies, 0, refine, steps, warmup)})

    if opts.io:
        for p in ranks:
            for vtk_mode in ("binary", "shared"):
                label = "io_%s" % vtk_mode
                c = CASES["medium"]
                args = case_args(label, "strong", c["spheres"], c["per_rank"], c["refine"], steps, warmup)
                args += ["--vtk", "10", "--vtk-mode", vtk_mode, "--vtk-dir", opts.vtk_dir]
                jobs.append({"label": label, "mode": "strong", "ranks": p, "args": args})

    return jobs


def run_job(job, rep, raw_dir, launcher, oversub_above, commit):
    """Run one point once, writing its JSON into the store.

    :param job:           The job dict from build_jobs.
    :param rep:           Repeat index, part of the filename.
    :param raw_dir:       Where results land.
    :param launcher:      MPI launcher path.
    :param oversub_above: Add --oversubscribe above this many ranks; 0 disables.
    :param commit:        Git hash recorded in the result.
    :returns: True if the run succeeded.
    """
    out = os.path.join(raw_dir, "%s_%s_np%d_r%d.json" % (job["label"], job["mode"], job["ranks"], rep))
    cmd = [launcher, "-n", str(job["ranks"])]
    if oversub_above and job["ranks"] > oversub_above:
        cmd.append("--oversubscribe")
    cmd += [BINARY] + job["args"] + ["--json", out, "--git", commit]

    print("  np=%-4d %-12s %-6s rep %d" % (job["ranks"], job["label"], job["mode"], rep), end="", flush=True)
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    except OSError as exc:
        print("   LAUNCH FAILED: %s" % exc)
        return False

    if res.returncode != 0:
        print("   FAILED (exit %d)" % res.returncode)
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-4:]
        for line in tail:
            print("      %s" % line)
        return False

    for line in (res.stdout or "").splitlines():
        if "wall clock (barriered)" in line:
            print("   %s" % line.split("s   ")[0].split()[-1] + " s")
            break
    else:
        print("   done")
    return True


# ── Loading and aggregation ────────────────────────────────────────────────

PHASE_ORDER = ["advect", "ghost_sync", "geometry", "redistribute", "post_geom", "quality", "normals", "vtk", "instrument"]
PHASE_LABEL = {
    "advect": "advection",
    "ghost_sync": "ghost sync",
    "geometry": "geometry",
    "redistribute": "redistribute",
    "post_geom": "post-redist geom",
    "quality": "quality",
    "normals": "normals",
    "vtk": "vtk",
    "instrument": "instrumentation",
}
#: The phases that are communication. Everything else is arithmetic on local data.
COMM_PHASES = ("ghost_sync", "redistribute")


def load_runs(raw_dir):
    """Read every result in the store.

    :param raw_dir: Directory of JSON results.
    :returns: A list of result dicts, skipping anything unreadable or of a
              schema this script does not know.
    """
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        try:
            with open(path) as fp:
                d = json.load(fp)
        except (OSError, ValueError) as exc:
            print("  skipping %s: %s" % (os.path.basename(path), exc), file=sys.stderr)
            continue
        if d.get("schema") != 1:
            print("  skipping %s: unknown schema %r" % (os.path.basename(path), d.get("schema")), file=sys.stderr)
            continue
        d["_file"] = os.path.basename(path)
        runs.append(d)
    return runs


def best_by_rank(runs):
    """Collapse repeats to the fastest run at each rank count.

    Minimum over repeats, not mean: the repeats exist to cancel machine state
    (thermal, background load, allocator warmth), and the minimum is the estimate
    least contaminated by it. Within a run the headline is still a MEAN over
    steps, because only means add up to the wall clock.

    :param runs: Results sharing one (label, mode).
    :returns: A dict {ranks: run}, keeping the smallest wall_s at each rank count.
    """
    best = {}
    for r in runs:
        p = r["nranks"]
        if p not in best or r["wall_s"] < best[p]["wall_s"]:
            best[p] = r
    return best


def group_runs(runs):
    """Index the store by (label, mode).

    :param runs: All results.
    :returns: A dict {(label, mode): {ranks: run}}.
    """
    buckets = {}
    for r in runs:
        buckets.setdefault((r["label"], r["mode"]), []).append(r)
    return {k: best_by_rank(v) for k, v in buckets.items()}


def ms_per_step(run):
    """Wall clock per timed step, in milliseconds.

    :param run: One result.
    :returns: Milliseconds.
    """
    return 1000.0 * run["wall_s"] / float(run["steps"])


def comm_fraction(run):
    """Share of the phase-mean budget spent in the two communicating phases.

    :param run: One result.
    :returns: A fraction in [0, 1], or 0 when nothing was measured.
    """
    total = sum(run["phases"][p]["mean_s"] for p in PHASE_ORDER)
    if total <= 0.0:
        return 0.0
    return sum(run["phases"][p]["mean_s"] for p in COMM_PHASES) / total


def karp_flatt(speedup, p_ratio):
    """The experimentally determined serial fraction.

    A RISING f across the sweep means overhead that grows with P; a FLAT f means a
    genuine fixed serial part. A speedup plot alone cannot tell those apart, and
    they call for completely different responses.

    :param speedup: T(baseline) / T(P).
    :param p_ratio: P / P_baseline.
    :returns: The serial fraction, or None where it is undefined.
    """
    if p_ratio <= 1.0 or speedup <= 0.0:
        return None
    return (1.0 / speedup - 1.0 / p_ratio) / (1.0 - 1.0 / p_ratio)


def timer_limited(run, phase):
    """Whether a phase's per-step mean sits below the timer's trustworthy floor.

    At high P the local phases shrink toward MPI_Wtick, and a curve that flattens
    because the timer ran out of bits is indistinguishable from one that flattened
    physically. Points that hit this are drawn hollow and footnoted.

    :param run:   One result.
    :param phase: Phase key.
    :returns: True if the value is below 100 * MPI_Wtick per step.
    """
    per_step = run["phases"][phase]["mean_s"] / float(run["steps"])
    return per_step < run.get("timer_floor_s", 0.0)


def curve(by_rank):
    """One curve, as parallel lists sorted by rank count.

    :param by_rank: {ranks: run}.
    :returns: (ranks list, runs list).
    """
    ps = sorted(by_rank)
    return ps, [by_rank[p] for p in ps]


# ── Text outputs ───────────────────────────────────────────────────────────

SUMMARY_COLUMNS = [
    "label", "mode", "nranks", "cuts", "n_bodies", "refinements", "tris_per_body", "global_triangles",
    "steps", "ms_per_step", "wall_s", "t_setup_s", "throughput_tri_per_s",
    "structural_imbalance", "halo_fraction", "bodies_touched_min", "bodies_touched_max",
    "comm_fraction", "ghost_bytes_per_step", "churn_tris_per_step", "wire_tris_per_step",
    "redist_p50_ms", "redist_p95_ms", "maxrss_mb_max", "io_dumps", "io_bytes", "git", "hostname", "file",
]


def write_summary_csv(runs, path):
    """One row per run, for anything that wants the numbers without the plots.

    :param runs: All results.
    :param path: Destination CSV.
    """
    with open(path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(SUMMARY_COLUMNS + [PHASE_LABEL[p] + "_ms_per_step" for p in PHASE_ORDER])
        for r in sorted(runs, key=lambda x: (x["label"], x["mode"], x["nranks"])):
            row = [
                r["label"], r["mode"], r["nranks"], "x".join(str(c) for c in r["cuts"]),
                r["n_bodies"], r["refinements"], r["tris_per_body"], r["global_triangles"],
                r["steps"], "%.6f" % ms_per_step(r), "%.6f" % r["wall_s"], "%.6f" % r["t_setup_s"],
                "%.6g" % r["throughput_tri_per_s"], "%.6f" % r["structural_imbalance"], "%.6f" % r["halo_fraction"],
                r["bodies_touched"]["min"], r["bodies_touched"]["max"], "%.6f" % comm_fraction(r),
                "%.1f" % r["ghost_bytes_per_step"], "%.1f" % r["churn_tris_per_step"], "%.1f" % r["wire_tris_per_step"],
                "%.6f" % (1000.0 * r["redist_p50_s"]), "%.6f" % (1000.0 * r["redist_p95_s"]),
                "%.2f" % (r["maxrss_bytes_max"] / (1024.0 * 1024.0)), r["io"]["dumps"], r["io"]["bytes"],
                r["git"], r["hostname"], r["_file"],
            ]
            row += ["%.6f" % (1000.0 * r["phases"][p]["mean_s"] / float(r["steps"])) for p in PHASE_ORDER]
            w.writerow(row)


def write_report(groups, runs, path):
    """The measurement document: the same content as the plots, as tables.

    :param groups: Output of group_runs.
    :param runs:   All results, for the provenance header.
    :param path:   Destination Markdown file.
    """
    hosts = sorted({r["hostname"] for r in runs})
    commits = sorted({r["git"] for r in runs})
    mpis = sorted({r["mpi"] for r in runs})

    out = []
    out.append("# Scaling report\n")
    out.append("Generated by `run_scaling.py` from %d runs in the raw store.\n" % len(runs))
    out.append("| | |\n|---|---|\n")
    out.append("| hosts | %s |\n" % ", ".join(hosts))
    out.append("| commits | %s |\n" % ", ".join(commits))
    out.append("| MPI | %s |\n" % "; ".join(mpis))
    out.append("\nEach row is the **minimum over repeats** at that rank count; within a run the\n")
    out.append("headline is the **mean over timed steps**, because only means add up to the wall\n")
    out.append("clock. Repeats were taken round robin over the rank counts, never in blocks.\n")

    strong = sorted(k for k in groups if k[1] == "strong" and not k[0].startswith("iso") and not k[0].startswith("io_"))
    weak = sorted(k for k in groups if k[1] == "weak")

    if strong:
        out.append("\n## Strong scaling\n")
        out.append("\nTotal work fixed, P grows. `f` is the Karp-Flatt serial fraction: RISING means\n")
        out.append("overhead growing with P, FLAT means a genuine fixed serial part. **`f` above 1 means\n")
        out.append("the parallel run is slower than the baseline** -- the added ranks bought negative\n")
        out.append("time, so the model's assumption of a serial part plus perfectly divisible work no\n")
        out.append("longer holds and only the sign is informative.\n")
        for key in strong:
            ps, rs = curve(groups[key])
            if not ps:
                continue
            base_p, base = ps[0], rs[0]
            out.append("\n### %s -- %d bodies x refine %d = %s triangles\n\n" % (key[0], base["n_bodies"], base["refinements"], "{:,}".format(base["global_triangles"])))
            out.append("| P | grid | ms/step | speedup | efficiency | f | comm % | imbalance | halo % | setup s |\n")
            out.append("|---|---|---|---|---|---|---|---|---|---|\n")
            for p, r in zip(ps, rs):
                s = base["wall_s"] / r["wall_s"]
                ratio = float(p) / float(base_p)
                f = karp_flatt(s, ratio)
                out.append("| %d | %s | %.3f | %.2f | %.2f | %s | %.1f | %.3f | %.2f | %.3f |\n" % (
                    p, "x".join(str(c) for c in r["cuts"]), ms_per_step(r), s, s / ratio,
                    "-" if f is None else "%.3f" % f, 100.0 * comm_fraction(r),
                    r["structural_imbalance"], 100.0 * r["halo_fraction"], r["t_setup_s"]))

    if weak:
        out.append("\n## Weak scaling\n")
        out.append("\nWork per rank fixed; the domain grows with the process grid, so bodies per rank,\n")
        out.append("halo fraction and neighbour count are all invariant. Ideal is a FLAT line.\n")
        out.append("\nThe `E(P0)` and `E(P2)` columns differ on purpose. P=1 is a different code path --\n")
        out.append("no ghosts exist and the redistribution communicates with nobody -- so the first\n")
        out.append("step is the cost of being parallel at all rather than a scaling trend, and\n")
        out.append("normalising only to P=1 charges that one-off to the trend.\n")
        for key in weak:
            ps, rs = curve(groups[key])
            if not ps:
                continue
            base = rs[0]
            second = rs[1] if len(rs) > 1 else None
            out.append("\n### %s -- %d bodies/rank x refine %d\n\n" % (key[0], base["n_bodies"] // base["nranks"], base["refinements"]))
            out.append("| P | grid | bodies | triangles | ms/step | E(P0) | E(P2) | comm % | imbalance | halo % | setup s |\n")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for p, r in zip(ps, rs):
                e0 = base["wall_s"] / r["wall_s"]
                e2 = (second["wall_s"] / r["wall_s"]) if second else float("nan")
                out.append("| %d | %s | %d | %s | %.3f | %.2f | %s | %.1f | %.3f | %.2f | %.3f |\n" % (
                    p, "x".join(str(c) for c in r["cuts"]), r["n_bodies"], "{:,}".format(r["global_triangles"]),
                    ms_per_step(r), e0, "-" if second is None else "%.2f" % e2,
                    100.0 * comm_fraction(r), r["structural_imbalance"], 100.0 * r["halo_fraction"], r["t_setup_s"]))
        out.append("\n> **Setup is the harness's own Amdahl ceiling, not the library's.** `MFTLGenerateSphere`\n")
        out.append("> builds every body on rank 0, so in weak mode the setup column grows linearly with P\n")
        out.append("> on one rank while the timed loop does not. It is excluded from every scaling total\n")
        out.append("> above and must never be folded back in.\n")

    iso = sorted(k for k in groups if k[0].startswith("iso"))
    if iso:
        out.append("\n## Granularity: the iso-triangle ladder\n")
        out.append("\nThe same triangle count on every rung, split differently between bodies and\n")
        out.append("triangles-per-body. This is not a scaling law -- it is what sets the constant in\n")
        out.append("both of the above.\n\n")
        out.append("| bodies | refine | tri/body | P | triangles | ms/step | ns/triangle | comm % | halo % | imbalance | bodies/rank |\n")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        rows = []
        for key in iso:
            for p, r in zip(*curve(groups[key])):
                rows.append(r)
        for r in sorted(rows, key=lambda x: (x["nranks"], -x["n_bodies"])):
            ns = 1e6 * ms_per_step(r) / float(r["global_triangles"])
            out.append("| %d | %d | %d | %d | %s | %.3f | %.2f | %.1f | %.2f | %.3f | %d..%d |\n" % (
                r["n_bodies"], r["refinements"], r["tris_per_body"], r["nranks"], "{:,}".format(r["global_triangles"]),
                ms_per_step(r), ns, 100.0 * comm_fraction(r), 100.0 * r["halo_fraction"], r["structural_imbalance"],
                r["bodies_touched"]["min"], r["bodies_touched"]["max"]))

    io_keys = sorted(k for k in groups if k[0].startswith("io_"))
    if io_keys:
        out.append("\n## I/O\n\n")
        out.append("| layout | P | dumps | MB | vtk ms/step | MB/s |\n|---|---|---|---|---|---|\n")
        for key in io_keys:
            for p, r in zip(*curve(groups[key])):
                mb = r["io"]["bytes"] / (1024.0 * 1024.0)
                vtk_s = r["phases"]["vtk"]["mean_s"]
                out.append("| %s | %d | %d | %.1f | %.3f | %.1f |\n" % (
                    key[0].replace("io_", ""), p, r["io"]["dumps"], mb,
                    1000.0 * vtk_s / float(r["steps"]), (mb / vtk_s) if vtk_s > 0 else 0.0))

    out.append("\n## Phase breakdown, ms/step\n")
    for key in sorted(groups):
        ps, rs = curve(groups[key])
        if not ps:
            continue
        shown = [p for p in PHASE_ORDER if any(r["phases"][p]["mean_s"] > 0 for r in rs)]
        out.append("\n### %s / %s\n\n" % (key[0], key[1]))
        out.append("| P | " + " | ".join(PHASE_LABEL[p] for p in shown) + " |\n")
        out.append("|---" * (len(shown) + 1) + "|\n")
        for p, r in zip(ps, rs):
            cells = []
            for ph in shown:
                v = 1000.0 * r["phases"][ph]["mean_s"] / float(r["steps"])
                cells.append("%.3f%s" % (v, " !" if timer_limited(r, ph) else ""))
            out.append("| %d | %s |\n" % (p, " | ".join(cells)))
    out.append("\n`!` marks a phase whose per-step mean is below 100 x MPI_Wtick: not trustworthy,\n")
    out.append("and the reason a flattening curve must not be read as a scaling limit there.\n")

    with open(path, "w") as fp:
        fp.write("".join(out))


# ── Plots ──────────────────────────────────────────────────────────────────


def import_matplotlib():
    """Import matplotlib in a headless-safe way.

    :returns: The pyplot module, or None if it is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.size": 9,
        "figure.dpi": 140,
    })
    return plt


def finish(ax, title, xlabel, ylabel, note=None):
    """Apply the shared axis furniture.

    :param ax:     The axes.
    :param title:  Axis title.
    :param xlabel: X label.
    :param ylabel: Y label.
    :param note:   Optional footnote drawn under the axes.
    """
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if note:
        ax.annotate(note, xy=(0, -0.20), xycoords="axes fraction", fontsize=7, color=INK_MUTED, va="top")


def rank_axis(ax, ranks):
    """Label a rank-count axis with the rank counts themselves, not powers.

    :param ax:    The axes.
    :param ranks: The rank counts present.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(p) for p in ranks])
    ax.minorticks_off()


def save(plt, fig, plots_dir, name):
    """Write one figure and close it.

    :param plt:       pyplot.
    :param fig:       The figure.
    :param plots_dir: Destination directory.
    :param name:      File name.
    """
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, name), bbox_inches="tight")
    plt.close(fig)


def matrix_keys(groups, mode):
    """The non-iso, non-io cases of one mode, in a stable order.

    :param groups: Output of group_runs.
    :param mode:   'strong' or 'weak'.
    :returns: A sorted list of keys.
    """
    order = list(CASES)
    keys = [k for k in groups if k[1] == mode and not k[0].startswith("iso") and not k[0].startswith("io_")]
    return sorted(keys, key=lambda k: order.index(k[0]) if k[0] in order else 99)


def plot_strong(plt, groups, plots_dir):
    """Wall clock, speedup and parallel efficiency against rank count.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = matrix_keys(groups, "strong")
    if not keys:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_ranks = set()
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        all_ranks.update(ps)
        ax.plot(ps, [ms_per_step(r) for r in rs], marker="o", color=CATEGORICAL[i % len(CATEGORICAL)], label=key[0])
        ideal = [ms_per_step(rs[0]) * ps[0] / float(p) for p in ps]
        ax.plot(ps, ideal, linestyle="--", linewidth=1.2, color=INK_MUTED, zorder=0, label="ideal 1/P" if i == 0 else None)
    ax.set_yscale("log")
    rank_axis(ax, sorted(all_ranks))
    ax.legend()
    finish(ax, "Strong scaling: total work fixed", "MPI ranks", "ms per timestep",
           "Dashed: ideal 1/P from each case's own baseline. Log-log, so ideal is a straight line.")
    save(plt, fig, plots_dir, "strong_walltime.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        base_p, base = ps[0], rs[0]
        sp = [base["wall_s"] / r["wall_s"] for r in rs]
        eff = [s / (float(p) / base_p) for s, p in zip(sp, ps)]
        c = CATEGORICAL[i % len(CATEGORICAL)]
        axes[0].plot(ps, sp, marker="o", color=c, label=key[0])
        axes[1].plot(ps, eff, marker="o", color=c, label=key[0])
    lim = sorted(set().union(*[set(curve(groups[k])[0]) for k in keys]))
    axes[0].plot(lim, [p / float(lim[0]) for p in lim], "--", linewidth=1.2, color=INK_MUTED, zorder=0, label="ideal")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.2, color=INK_MUTED, zorder=0)
    for ax in axes:
        rank_axis(ax, lim)
    axes[0].set_yscale("log", base=2)
    axes[0].legend()
    finish(axes[0], "Speedup", "MPI ranks", "T(baseline) / T(P)")
    finish(axes[1], "Parallel efficiency", "MPI ranks", "speedup / (P / P_baseline)",
           "1.0 is ideal. Below 1 is the fraction of added ranks that bought nothing.")
    save(plt, fig, plots_dir, "strong_speedup_efficiency.png")


def plot_weak(plt, groups, plots_dir):
    """Time per step and weak efficiency, normalised two ways.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = matrix_keys(groups, "weak")
    if not keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    all_ranks = set()
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        all_ranks.update(ps)
        c = CATEGORICAL[i % len(CATEGORICAL)]
        axes[0].plot(ps, [ms_per_step(r) for r in rs], marker="o", color=c, label=key[0])
        base = rs[0]
        axes[1].plot(ps, [base["wall_s"] / r["wall_s"] for r in rs], marker="o", color=c, label="%s (vs P=%d)" % (key[0], ps[0]))
        if len(rs) > 1:
            second = rs[1]
            axes[1].plot(ps[1:], [second["wall_s"] / r["wall_s"] for r in rs[1:]], marker="s", markerfacecolor=SURFACE,
                         linestyle=":", color=c, label="%s (vs P=%d)" % (key[0], ps[1]))
    ranks = sorted(all_ranks)
    axes[1].axhline(1.0, linestyle="--", linewidth=1.2, color=INK_MUTED, zorder=0)
    for ax in axes:
        rank_axis(ax, ranks)
    axes[0].legend()
    axes[1].legend()
    finish(axes[0], "Weak scaling: work per rank fixed", "MPI ranks", "ms per timestep",
           "Ideal is a FLAT line: the domain grows with the grid, so each rank keeps a constant volume.")
    finish(axes[1], "Weak efficiency", "MPI ranks", "T(baseline) / T(P)",
           "Two baselines on purpose: P=1 has no ghosts and no exchange, so the first step is the\n"
           "cost of being parallel at all, not a trend. The dotted series removes it.")
    save(plt, fig, plots_dir, "weak_scaling.png")


def plot_phases(plt, groups, plots_dir):
    """Stacked per-phase breakdown, one figure per case and mode.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    for key in sorted(groups):
        if key[0].startswith("io_"):
            continue
        ps, rs = curve(groups[key])
        if len(ps) < 2:
            continue
        shown = [p for p in PHASE_ORDER if any(r["phases"][p]["mean_s"] > 0 for r in rs)]
        if not shown:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = list(range(len(ps)))
        bottom = [0.0] * len(ps)
        for i, ph in enumerate(shown):
            vals = [1000.0 * r["phases"][ph]["mean_s"] / float(r["steps"]) for r in rs]
            # A 2px surface-coloured edge is the spacer between stacked segments.
            ax.bar(x, vals, bottom=bottom, width=0.62, color=CATEGORICAL[i % len(CATEGORICAL)],
                   edgecolor=SURFACE, linewidth=2, label=PHASE_LABEL[ph])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in ps])
        ax.legend(ncol=2)
        finish(ax, "Where the step goes: %s / %s" % key, "MPI ranks", "ms per timestep",
               "Phase MEANS over ranks, which is the breakdown that adds up to the wall clock.")
        save(plt, fig, plots_dir, "phases_%s_%s.png" % key)


def plot_phase_efficiency(plt, groups, plots_dir):
    """Per-phase strong efficiency, so the phase that stops scaling is named.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    for key in matrix_keys(groups, "strong"):
        ps, rs = curve(groups[key])
        if len(ps) < 2:
            continue
        shown = [p for p in PHASE_ORDER if p != "instrument" and all(r["phases"][p]["mean_s"] > 0 for r in rs)]
        if not shown:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        base_p = ps[0]
        for i, ph in enumerate(shown):
            b = rs[0]["phases"][ph]["mean_s"]
            eff = [(b / r["phases"][ph]["mean_s"]) / (float(p) / base_p) for p, r in zip(ps, rs)]
            c = CATEGORICAL[i % len(CATEGORICAL)]
            ax.plot(ps, eff, marker="o", color=c, label=PHASE_LABEL[ph])
            # Hollow markers where the timer cannot be trusted at that point.
            hx = [p for p, r in zip(ps, rs) if timer_limited(r, ph)]
            hy = [e for e, r in zip(eff, rs) if timer_limited(r, ph)]
            if hx:
                ax.plot(hx, hy, linestyle="none", marker="o", markerfacecolor=SURFACE, markeredgecolor=c, markeredgewidth=2)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, color=INK_MUTED, zorder=0)
        ax.set_yscale("log", base=10)
        rank_axis(ax, ps)
        ax.legend(ncol=2)
        finish(ax, "Per-phase strong efficiency: %s" % key[0], "MPI ranks", "phase efficiency",
               "Hollow markers: per-step mean below 100 x MPI_Wtick, so the value is timer-limited\n"
               "and a flattening there is not a scaling result.")
        save(plt, fig, plots_dir, "phase_efficiency_%s.png" % key[0])


def plot_partition(plt, groups, plots_dir):
    """Load imbalance and halo fraction against rank count.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = matrix_keys(groups, "strong") + matrix_keys(groups, "weak")
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    all_ranks = set()
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        all_ranks.update(ps)
        c = CATEGORICAL[i % len(CATEGORICAL)]
        style = "-" if key[1] == "strong" else ":"
        label = "%s / %s" % key
        axes[0].plot(ps, [r["structural_imbalance"] for r in rs], marker="o", linestyle=style, color=c, label=label)
        axes[1].plot(ps, [100.0 * r["halo_fraction"] for r in rs], marker="o", linestyle=style, color=c, label=label)
    axes[0].axhline(1.0, linestyle="--", linewidth=1.2, color=INK_MUTED, zorder=0)
    for ax in axes:
        rank_axis(ax, sorted(all_ranks))
    axes[0].legend(ncol=2)
    finish(axes[0], "Structural load imbalance", "MPI ranks", "busiest rank / mean owned triangles",
           "Max over mean of the TIME would read 1.00 by construction: the loop is barrier-bounded.")
    finish(axes[1], "Halo fraction", "MPI ranks", "ghost triangles, % of owned",
           "Solid: strong. Dotted: weak, where it should stay flat by construction.")
    save(plt, fig, plots_dir, "partition.png")


def plot_granularity(plt, groups, plots_dir):
    """The iso-triangle ladder: same work, different body/refinement split.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    rows = []
    for key in groups:
        if not key[0].startswith("iso"):
            continue
        for p, r in zip(*curve(groups[key])):
            rows.append(r)
    if not rows:
        return

    by_rank = {}
    for r in rows:
        by_rank.setdefault(r["nranks"], []).append(r)
    ranks = sorted(by_rank)
    steps = ordinal_steps(len(ranks))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for (c, mk), p in zip(steps, ranks):
        pts = sorted(by_rank[p], key=lambda x: x["tris_per_body"])
        xs = [r["tris_per_body"] for r in pts]
        axes[0].plot(xs, [1e6 * ms_per_step(r) / float(r["global_triangles"]) for r in pts], marker=mk, color=c, label="P = %d" % p)
        axes[1].plot(xs, [100.0 * r["halo_fraction"] for r in pts], marker=mk, color=c, label="P = %d" % p)
    for ax in axes:
        ax.set_xscale("log", base=4)
        ax.set_xticks(sorted({r["tris_per_body"] for r in rows}))
        ax.set_xticklabels([str(t) for t in sorted({r["tris_per_body"] for r in rows})])
        ax.minorticks_off()
    axes[0].legend()
    axes[1].legend()
    finish(axes[0], "Granularity at constant total work", "triangles per body", "ns per triangle per step",
           "Every point is the same triangle count. Left: many tiny bodies. Right: few large ones.")
    finish(axes[1], "Halo fraction along the ladder", "triangles per body", "ghost triangles, % of owned")
    save(plt, fig, plots_dir, "granularity.png")


def plot_karp_flatt(plt, groups, plots_dir):
    """The experimentally determined serial fraction against rank count.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = matrix_keys(groups, "strong")
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_ranks = set()
    drew = False
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        base_p, base = ps[0], rs[0]
        xs, ys = [], []
        for p, r in zip(ps, rs):
            f = karp_flatt(base["wall_s"] / r["wall_s"], float(p) / base_p)
            if f is not None:
                xs.append(p)
                ys.append(f)
        if xs:
            ax.plot(xs, ys, marker="o", color=CATEGORICAL[i % len(CATEGORICAL)], label=key[0])
            all_ranks.update(xs)
            drew = True
    if not drew:
        plt.close(fig)
        return
    rank_axis(ax, sorted(all_ranks))
    ax.legend()
    finish(ax, "Karp-Flatt serial fraction", "MPI ranks", "f",
           "RISING f means overhead growing with P. FLAT f means a genuine fixed serial part.\n"
           "A speedup plot cannot tell those apart, and they call for different fixes.")
    save(plt, fig, plots_dir, "karp_flatt.png")


def plot_memory_setup(plt, groups, plots_dir):
    """Peak RSS per rank, and the rank-0-serial setup cost.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = matrix_keys(groups, "strong") + matrix_keys(groups, "weak")
    if not keys:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    all_ranks = set()
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        all_ranks.update(ps)
        c = CATEGORICAL[i % len(CATEGORICAL)]
        style = "-" if key[1] == "strong" else ":"
        label = "%s / %s" % key
        axes[0].plot(ps, [r["maxrss_bytes_max"] / (1024.0 * 1024.0) for r in rs], marker="o", linestyle=style, color=c, label=label)
        axes[1].plot(ps, [r["t_setup_s"] for r in rs], marker="o", linestyle=style, color=c, label=label)
    for ax in axes:
        rank_axis(ax, sorted(all_ranks))
        ax.set_yscale("log")
    axes[0].legend(ncol=2)
    finish(axes[0], "Peak RSS, busiest rank", "MPI ranks", "MB")
    finish(axes[1], "Setup: generation plus the initial scatter", "MPI ranks", "seconds",
           "Rank-0 serial by construction: MFTLGenerateSphere builds every body on rank 0. In WEAK\n"
           "mode (dotted) this grows linearly with P and is the harness's ceiling, not the library's.\n"
           "It is excluded from every scaling total.")
    save(plt, fig, plots_dir, "memory_setup.png")


def plot_io(plt, groups, plots_dir):
    """VTU write throughput, both file layouts.

    :param plt:       pyplot.
    :param groups:    Output of group_runs.
    :param plots_dir: Destination directory.
    """
    keys = sorted(k for k in groups if k[0].startswith("io_"))
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    all_ranks = set()
    for i, key in enumerate(keys):
        ps, rs = curve(groups[key])
        all_ranks.update(ps)
        rate = []
        for r in rs:
            vtk_s = r["phases"]["vtk"]["mean_s"]
            mb = r["io"]["bytes"] / (1024.0 * 1024.0)
            rate.append((mb / vtk_s) if vtk_s > 0 else 0.0)
        ax.plot(ps, rate, marker="o", color=CATEGORICAL[i % len(CATEGORICAL)], label=key[0].replace("io_", ""))
    rank_axis(ax, sorted(all_ranks))
    ax.legend()
    finish(ax, "VTU write throughput", "MPI ranks", "MB/s",
           "binary: one .vtu per rank plus a .pvtu index. shared: one .vtu written collectively with MPI-IO.")
    save(plt, fig, plots_dir, "io.png")


def render(results_dir):
    """Rebuild every text and graphical output from the whole raw store.

    :param results_dir: The store root.
    :returns: The number of runs found.
    """
    raw_dir = os.path.join(results_dir, "raw")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    runs = load_runs(raw_dir)
    if not runs:
        print("No results in %s -- nothing to render." % raw_dir)
        return 0

    groups = group_runs(runs)
    write_summary_csv(runs, os.path.join(results_dir, "summary.csv"))
    write_report(groups, runs, os.path.join(results_dir, "report.md"))

    plt = import_matplotlib()
    if plt is None:
        print("matplotlib not available: wrote summary.csv and report.md, skipped the plots.", file=sys.stderr)
        return len(runs)

    for fn in (plot_strong, plot_weak, plot_phases, plot_phase_efficiency, plot_partition,
               plot_granularity, plot_karp_flatt, plot_memory_setup, plot_io):
        fn(plt, groups, plots_dir)

    n_plots = len(glob.glob(os.path.join(plots_dir, "*.png")))
    print("-" * 49)
    print("%-17s: %d" % ("Runs in store", len(runs)))
    print("%-17s: %d" % ("Curves", len(groups)))
    print("%-17s: %d in %s" % ("Plots", n_plots, os.path.relpath(plots_dir, PROJECT_ROOT)))
    print("%-17s: %s" % ("Report", os.path.relpath(os.path.join(results_dir, "report.md"), PROJECT_ROOT)))
    print("-" * 49)
    return len(runs)


# ── Entry point ────────────────────────────────────────────────────────────


def main():
    """Parse the command line, run what was asked for, and re-render everything."""
    parser = argparse.ArgumentParser(description="Strong, weak and granularity scaling of the MFTL advection timestep.",
                                     epilog="Example: python3 run_scaling.py --preset laptop --all")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="laptop", help="rank set, repeats and default cases")
    parser.add_argument("--case", action="append", help="one case name; repeatable (default: the preset's list)")
    parser.add_argument("--mode", choices=["strong", "weak", "both"], default="both", help="which curves to run")
    parser.add_argument("--ranks", help="comma-separated rank counts, overriding the preset")
    parser.add_argument("--repeats", type=int, help="repeats per point, overriding the preset")
    parser.add_argument("--steps", type=int, help="timed steps per run, overriding the preset")
    parser.add_argument("--iso", action="store_true", help="also run the iso-triangle granularity ladder")
    parser.add_argument("--io", action="store_true", help="also run the two I/O configurations")
    parser.add_argument("--all", action="store_true", help="the preset's cases plus the ladder and the I/O configurations")
    parser.add_argument("--no-build", dest="build", action="store_false", help="skip the Release rebuild")
    parser.add_argument("--plots-only", action="store_true", help="render from the existing store and run nothing")
    parser.add_argument("--smoke", action="store_true", help="one tiny run at np=2, to check the wiring")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="store location")
    parser.add_argument("--vtk-dir", default=os.path.join(DEFAULT_RESULTS, "vtk"), help="where the I/O configurations write their dumps")
    opts = parser.parse_args()

    if opts.ranks:
        opts.ranks = [int(x) for x in opts.ranks.split(",") if x.strip()]
    if opts.all:
        opts.iso = True
        opts.io = True

    results_dir = os.path.abspath(opts.results)
    raw_dir = os.path.join(results_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if opts.plots_only:
        render(results_dir)
        return

    preset = dict(PRESETS[opts.preset])
    if opts.repeats:
        preset["repeats"] = opts.repeats

    if opts.build:
        build_release()
    if not os.path.exists(BINARY):
        fail("%s not found -- run python3 build_all.py (without --lib-only)" % os.path.relpath(BINARY, PROJECT_ROOT))

    launcher = mpiexec_command()
    oversub_above = preset["cores"] if wants_oversubscribe(launcher) else 0
    commit = git_hash()

    if opts.smoke:
        jobs = [{"label": "smoke", "mode": "strong", "ranks": 2,
                 "args": case_args("smoke", "strong", 8, 0, 1, 3, 1) + ["--verify"]}]
        repeats = 1
    else:
        jobs = build_jobs(opts, preset)
        repeats = preset["repeats"]

    if not jobs:
        fail("nothing to run -- check --case / --mode / --ranks")

    if opts.io:
        os.makedirs(opts.vtk_dir, exist_ok=True)

    print("-" * 49)
    print("%-17s: %s" % ("Preset", opts.preset))
    print("%-17s: %s" % ("Launcher", launcher))
    print("%-17s: %s" % ("Oversubscribe", "above %d ranks" % oversub_above if oversub_above else "not needed"))
    print("%-17s: %d points x %d repeats = %d runs" % ("Sweep", len(jobs), repeats, len(jobs) * repeats))
    print("%-17s: %s" % ("Store", os.path.relpath(results_dir, PROJECT_ROOT)))
    print("-" * 49)

    # Round robin over the points, NOT repeats in blocks. See the module docstring:
    # a curve measured in blocks turns a thermal ramp into a scaling law.
    failed = 0
    for rep in range(repeats):
        print("pass %d/%d" % (rep + 1, repeats))
        for job in jobs:
            if not run_job(job, rep, raw_dir, launcher, oversub_above, commit):
                failed += 1

    if failed:
        print("\n%d run(s) failed; rendering what did land." % failed, file=sys.stderr)

    render(results_dir)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

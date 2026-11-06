#!/usr/bin/env python3
"""Paraboloid placement of split/collapse vertices and Phong back-projection of
smoothed vertices, in the library (MFTL_RemeshParameters::placement, ::projection),
over the LeVeque period.

    python3 placement_study.py run [--jobs 4] [--exe B]     # results/leveque_remeshing_placement/runs/
    python3 evaluate.py --out ../../results/leveque_remeshing_placement
    python3 placement_study.py timing [--exe B]              # each configuration alone, best of 2 -> timing.json
    python3 placement_study.py tables                        # markdown tables

Every configuration runs on the base sphere and three rotated replicas at
refinements 4 and 5, for uniform and curvature sizing. The timing pass runs the
base orientation of each alone, without snapshots, and reads the library's own
timers (reference copies, projections, fits) from the example's summary.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "leveque_remeshing_placement")
DT = "0.015"
ROT = (None, 7.0, 13.0, 29.0)
SIZINGS = ("uniform", "curvature")
REFINE = (4, 5)

P = lambda r: ["-placement", "paraboloid", "-fitradius", str(r)]
N = lambda r: ["-placement", "paraboloid_normals", "-fitradius", str(r)]
J = ["-projection", "phong"]
# (name, example arguments, environment). The library projects once per remesh
# call, after the last smoothing sweep ("proj"). The projEvery* and projS1 rows
# are the schedule experiment of PAPER_NOTES.md section 9.5 and need
# archive/projection_schedule_experiment.patch: without MFTL_EXP_PROJSWEEP that
# build projects after EVERY sweep (the library's first version), with 1 after
# the last sweep of each iteration, with 2 after the last sweep of the call
# (bit-identical to the library). Run them with --exe <that build>.
EXPERIMENT = ("projEvery", "projS1")
VARIANTS = [
    ("base", [], {}),
    ("fitP3", P(3), {}),
    ("fitP4", P(4), {}),
    ("fitN3", N(3), {}),
    ("fitN4", N(4), {}),
    ("proj", J, {}),
    ("proj+fitP3", J + P(3), {}),
    ("proj+fitN3", J + N(3), {}),
    ("projEvery", J, {}),
    ("projEvery+fitP3", J + P(3), {}),
    ("projEvery+fitN3", J + N(3), {}),
    ("projS1", J, {"MFTL_EXP_PROJSWEEP": "1"}),
]
LABEL = {
    "base": "midpoint, no projection",
    "fitP3": "paraboloid (positions), radius 3",
    "fitP4": "paraboloid (positions), radius 4",
    "fitN3": "paraboloid (+ normals), radius 3",
    "fitN4": "paraboloid (+ normals), radius 4",
    "proj": "Phong projection (once per call, the library)",
    "proj+fitP3": "Phong projection + paraboloid (positions)",
    "proj+fitN3": "Phong projection + paraboloid (+ normals)",
    "projEvery": "Phong projection after every sweep (experiment)",
    "projEvery+fitP3": "projection every sweep + paraboloid (positions)",
    "projEvery+fitN3": "projection every sweep + paraboloid (+ normals)",
    "projS1": "Phong projection after each iteration (experiment)",
}


def runs():
    out = []
    for s in SIZINGS:
        for r in REFINE:
            for name, args, env in VARIANTS:
                for i, rot in enumerate(ROT):
                    tag = f"{s}_{name}_r{r}" + (f"@rot{i}" if rot else "")
                    a = ["-refine", str(r), "-sizing", s] + args + (["-rotate", repr(rot)] if rot else [])
                    out.append((tag, a, env))
    return out


def done(tag):
    log = os.path.join(OUT, "runs", tag, "log.txt")
    return os.path.exists(log) and "wall clock" in open(log).read()


def run_one(exe, tag, args, env):
    if done(tag):
        return 0
    d = os.path.join(OUT, "runs", tag)
    os.makedirs(d, exist_ok=True)
    cmd = [exe, "-dt", DT, "-quiet"] + args + ["-csv", os.path.join(d, "series.csv"), "-snapshots", os.path.join(d, "snap")]
    t0 = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write("# " + " ".join(f"{k}={v}" for k, v in env.items()) + " " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d, env=dict(os.environ, **env)).returncode
    print(f"{tag:44s} rc={rc} {time.time() - t0:7.1f} s", flush=True)
    return rc


def parse_summary(text):
    """The example's summary lines -> numbers."""
    v = {}
    m = re.search(r"wall clock\s+([\d.]+) s\s+\(([\d.]+) ms/step\)", text)
    if m:
        v["wall_s"], v["ms_per_step"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^\s+remesh\s+([\d.]+) s", text, re.M)
    if m:
        v["remesh_s"] = float(m.group(1))
    m = re.search(r"=== LeVeque, .* ([\d]+) -> ([\d]+) triangles \(peak ([\d]+)\)", text)
    if m:
        v["tri0"], v["tri_T"], v["tri_peak"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"remesh per triangle\s+([\d.]+) ns", text)
    if m:
        v["ns_per_tri_step"] = float(m.group(1))
    m = re.search(r"reference copies\s+([\d.]+) s .*\(largest ([\d.]+) kB\)", text)
    if m:
        v["t_reference"], v["reference_kB"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"projections\s+([\d.]+) s .*\((\d+) moves, ([\d.]+) ns each, (\d+) missed\)", text)
    if m:
        v["t_projection"], v["n_projected"], v["ns_per_projection"], v["n_missed"] = float(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))
    m = re.search(r"fits\s+([\d.]+) s .*\((\d+) fits, ([\d.]+) ns each, (\d+) rank-deficient, (\d+) fell back", text)
    if m:
        v["t_fit"], v["n_fits"], v["ns_per_fit"], v["n_rank_deficient"], v["n_fallbacks"] = (float(m.group(1)), int(m.group(2)), float(m.group(3)),
                                                                                         int(m.group(4)), int(m.group(5)))
    return v


def timing(exe):
    path = os.path.join(OUT, "timing.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for s in SIZINGS:
        for r in REFINE:
            for name, args, env in VARIANTS:
                tag = f"{s}_{name}_r{r}"
                if tag in res:
                    continue
                best = None
                for _ in range(2):
                    cmd = [exe, "-dt", DT, "-quiet", "-refine", str(r), "-sizing", s] + args
                    text = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **env)).stdout
                    v = parse_summary(text)
                    if best is None or v["wall_s"] < best["wall_s"]:
                        best = v
                res[tag] = best
                print(f"{tag:30s} {best['ms_per_step']:8.2f} ms/step", flush=True)
                json.dump(res, open(path, "w"), indent=1)


def tables():
    F = json.load(open(os.path.join(OUT, "metrics.json")))
    Tm = json.load(open(os.path.join(OUT, "timing.json"))) if os.path.exists(os.path.join(OUT, "timing.json")) else {}

    def entries(tag):
        return [F[k] for k in [tag] + sorted(k for k in F if k.startswith(tag + "@"))] if tag in F else []

    def ms(vals, fmt):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        if len(vals) == 1:
            return fmt.format(vals[0])
        return fmt.format(np.mean(vals)) + " ± " + fmt.replace("+", "").format(np.std(vals, ddof=1))

    print("### Accuracy and smoothness at T (mean ± sd over 4 orientations)\n")
    print("| sizing | r | variant | V/V(0) − 1 [%] | d_rms [% R0] | E_H | N_H | E_H at T/2 | triangles at T |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SIZINGS:
        for r in REFINE:
            for name, _, _ in VARIANTS:
                E = entries(f"{s}_{name}_r{r}")
                if not E:
                    continue
                q4 = [e["q4"] for e in E]
                q2 = [e["q2"] for e in E]
                print(f"| {s} | {r} | {LABEL[name]} | {ms([100 * q['dV'] for q in q4], '{:+.3f}')} | {ms([100 * q['d_rms'] for q in q4], '{:.3f}')} | "
                      f"{ms([q['E_H'] for q in q4], '{:.2f}')} | {ms([q['N_H'] for q in q4], '{:.3f}')} | {ms([q['E_H'] for q in q2], '{:.2f}')} | "
                      f"{int(np.mean([q['n_tri'] for q in q4]))} |")

    if not Tm:
        return
    print("\n### Cost (base orientation, each configuration alone, best of 2)\n")
    print("| sizing | r | variant | ms/step | remesh ns per triangle-step | fit: share of remesh, ns per fit, fits per step | "
          "projection: share, ns per move | reference: share, largest kB |")
    print("|---|---|---|---|---|---|---|---|")
    for s in SIZINGS:
        for r in REFINE:
            for name, _, _ in VARIANTS:
                v = Tm.get(f"{s}_{name}_r{r}")
                if not v:
                    continue
                rem = v.get("remesh_s", float("nan"))
                fit = (f"{100 * v['t_fit'] / rem:.1f} %, {v['ns_per_fit']:.0f}, {v['n_fits'] / 200:.0f}" if "t_fit" in v else "-")
                prj = (f"{100 * v['t_projection'] / rem:.1f} %, {v['ns_per_projection']:.0f}" if "t_projection" in v else "-")
                ref = (f"{100 * v['t_reference'] / rem:.1f} %, {v['reference_kB']:.0f}" if "t_reference" in v else "-")
                print(f"| {s} | {r} | {LABEL[name]} | {v['ms_per_step']:.1f} | {v.get('ns_per_tri_step', float('nan')):.0f} | {fit} | {prj} | {ref} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "timing", "tables"])
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "leveque_remeshing_parallel"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="run only variants whose tag contains this")
    a = ap.parse_args()
    if a.what == "tables":
        return tables()
    if a.what == "timing":
        return timing(a.exe)
    todo = [x for x in runs() if not a.only or a.only in x[0]]
    todo.sort(key=lambda x: (-int(x[1][1]), x[0]))  # refinement 5 first
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda x: run_one(a.exe, *x), todo))
    sys.exit(1 if any(rcs) else 0)


if __name__ == "__main__":
    main()

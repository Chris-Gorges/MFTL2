#!/usr/bin/env python3
"""Where the remesher loses volume, and three local fixes, over the LeVeque period.

    python3 volume_ops_study.py run --exe E [--jobs 4] [--only S]
    python3 volume_ops_study.py run --exe B --only vpc_lib     # the library option
    python3 evaluate.py --out ../../results/leveque_remeshing_volume_ops
    python3 volume_ops_study.py tables

E is a build with archive/volume_budget_experiment.patch, B the current tree. The
volume-preserving collapse (Lindstrom & Turk: the chosen point, midpoint or paraboloid,
moved along the gradient of the collapse's volume change until it is zero) is the
library's MFTL_RemeshParameters::collapse_preserves_volume, -vpcollapse 1; vpc runs it
on E (with the budget), vpc_lib on B (no budget columns). The stored vpc runs were made
with an earlier form of the patch that switched it by MFTL_EXP_VPCOLLAPSE=1; the
library option reproduces them in every metric and every printed budget digit.
Switches of E (environment):
  MFTL_EXP_VBUDGET=1     per-sweep budget and exact per-operation volume changes, binned
                         by the local bending angle, printed to stderr at exit (always on here)
  MFTL_EXP_FLIPGUARD=t   refuse a flip whose volume change exceeds t * A_quad * |edge| / 3
                         (t = 0.05 is a fold of about 11 degrees across the edge)
  MFTL_EXP_FLIPCOMP=1    restore a flip's volume exactly by shifting its four vertices by one
                         vector along the quad normal
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
OUT = os.path.join(ROOT, "results", "leveque_remeshing_volume_ops")
DT = "0.015"
ROT = (None, 7.0, 13.0, 29.0)
REFINE = (4, 5)

CONFIGS = [("mid", []), ("proj", ["-projection", "phong"]), ("proj+fitP3", ["-projection", "phong", "-placement", "paraboloid", "-fitradius", "3"])]
FIXES = [
    ("none", {}, []),
    ("vpc", {}, ["-vpcollapse", "1"]),
    ("vpc_lib", {}, ["-vpcollapse", "1"]),
    ("fg05", {"MFTL_EXP_FLIPGUARD": "0.05"}, []),
    ("fc", {"MFTL_EXP_FLIPCOMP": "1"}, []),
    ("vpc+fc", {"MFTL_EXP_FLIPCOMP": "1"}, ["-vpcollapse", "1"]),
]
LABEL = {"none": "as is", "vpc": "volume-preserving collapse", "vpc_lib": "volume-preserving collapse (library)", "fg05": "flip guard 0.05",
         "fc": "compensated flip", "vpc+fc": "vp collapse + compensated flip"}


def runs():
    out = []
    for r in REFINE:
        for cname, cargs in CONFIGS:
            for fname, env, fargs in FIXES:
                for i, rot in enumerate(ROT):
                    tag = f"uniform_{cname}_{fname}_r{r}" + (f"@rot{i}" if rot else "")
                    a = ["-refine", str(r)] + cargs + fargs + (["-rotate", repr(rot)] if rot else [])
                    out.append((tag, a, dict(env, MFTL_EXP_VBUDGET="1")))
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
    print(f"{tag:48s} rc={rc} {time.time() - t0:7.1f} s", flush=True)
    return rc


def parse(text):
    v = {}
    m = re.search(r"EXPVOL sweeps \(% of V0\): split ([-+\d.]+) collapse ([-+\d.]+) flip ([-+\d.]+) smooth ([-+\d.]+)", text)
    if m:
        v["split"], v["collapse"], v["flip"], v["smooth"] = (float(m.group(i)) for i in range(1, 5))
    m = re.search(r"V/V0 at t=T\s+([\d.]+)", text)
    if m:
        v["dV"] = 100.0 * (float(m.group(1)) - 1.0)
    m = re.search(r"off the exact sphere\s+rms ([\d.]+) %", text)
    if m:
        v["sph_rms"] = float(m.group(1))
    m = re.search(r"min / avg quality\s+([\d.]+) / ([\d.]+)", text)
    if m:
        v["min_q"] = float(m.group(1))
    m = re.search(r"flips refused by the guard (\d+), compensated (\d+) \(largest shift ([\d.e+-]+)", text)
    if m:
        v["refused"], v["comp"], v["comp_max"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    m = re.search(r"=== LeVeque, .* ([\d]+) -> ([\d]+) triangles", text)
    if m:
        v["tri_T"] = int(m.group(2))
    m = re.search(r"wall clock\s+([\d.]+) s\s+\(([\d.]+) ms/step\)", text)
    if m:
        v["ms"] = float(m.group(2))
    bins = re.findall(r"EXPVOL bin (\S+)\s+collapses\s+(\d+) dV ([-+\d.]+) %\s+flips\s+(\d+) dV ([-+\d.]+) %", text)
    v["bins"] = [(b, int(nc), float(dc), int(nf), float(df)) for b, nc, dc, nf, df in bins]
    return v


def tables():
    F = json.load(open(os.path.join(OUT, "metrics.json"))) if os.path.exists(os.path.join(OUT, "metrics.json")) else {}

    def entries(tag):
        tags = [t for t in [tag] + [f"{tag}@rot{i}" for i in range(1, 4)] if os.path.exists(os.path.join(OUT, "runs", t, "log.txt"))]
        return [(parse(open(os.path.join(OUT, "runs", t, "log.txt")).read()), F.get(t)) for t in tags]

    def ms(vals, fmt):
        vals = [x for x in vals if x is not None]
        if not vals:
            return "-"
        if len(vals) == 1:
            return fmt.format(vals[0])
        return fmt.format(np.mean(vals)) + " ± " + fmt.replace("+", "").format(np.std(vals, ddof=1))

    print("### Budget over the period (% of V0, mean of 4 orientations) and accuracy at T\n")
    print("| r | config | fix | split | collapse | flip | smooth | V/V(0) − 1 [%] | d_rms [% R0] | E_H | N_H | min q | triangles | ms/step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in REFINE:
        for cname, _ in CONFIGS:
            for fname, _, _ in FIXES:
                E = entries(f"uniform_{cname}_{fname}_r{r}")
                if not E:
                    continue
                S = [e[0] for e in E]
                Q = [e[1]["q4"] if e[1] else None for e in E]
                mean = lambda k: f"{np.mean([s[k] for s in S if k in s]):+.3f}" if any(k in s for s in S) else "-"
                print(f"| {r} | {cname} | {LABEL[fname]} | {mean('split')} | {mean('collapse')} | {mean('flip')} | {mean('smooth')} | "
                      f"{ms([s.get('dV') for s in S], '{:+.3f}')} | {ms([100 * q['d_rms'] if q else None for q in Q], '{:.3f}')} | "
                      f"{ms([q['E_H'] if q else None for q in Q], '{:.2f}')} | {ms([q['N_H'] if q else None for q in Q], '{:.3f}')} | "
                      f"{ms([s.get('min_q') for s in S], '{:.3f}')} | {int(np.mean([s.get('tri_T', 0) for s in S]))} | "
                      f"{ms([s.get('ms') for s in S], '{:.1f}')} |")

    print("\n### Where collapses and flips lose volume (base orientation, % of V0, by the local bending angle)\n")
    print("| r | config | angle | collapses | collapse dV | flips | flip dV |")
    print("|---|---|---|---|---|---|---|")
    for r in REFINE:
        for cname, _ in CONFIGS:
            E = entries(f"uniform_{cname}_none_r{r}")
            if not E:
                continue
            for b, nc, dc, nf, df in E[0][0]["bins"]:
                print(f"| {r} | {cname} | {b} | {nc} | {dc:+.3f} | {nf} | {df:+.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "tables"])
    ap.add_argument("--exe", required=False)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only")
    a = ap.parse_args()
    if a.what == "tables":
        return tables()
    todo = [x for x in runs() if not a.only or a.only in x[0]]
    todo.sort(key=lambda x: (-int(x[1][1]), x[0]))
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda x: run_one(a.exe, *x), todo))
    sys.exit(1 if any(rcs) else 0)


if __name__ == "__main__":
    main()

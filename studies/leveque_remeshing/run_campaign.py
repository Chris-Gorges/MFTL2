#!/usr/bin/env python3
"""Runs the LeVeque remeshing study: every configuration once, serially per
run (np = 1), several runs at a time.

    python3 run_campaign.py [--exe build/leveque_remeshing_parallel] [--out DIR] [--jobs 4] [--only NAME...]

Each run writes DIR/runs/NAME/{log.txt, series.csv, snap_q0..q4.vtk}.

The report's sweep-order rows (its sections 5 and 6: NAME_r4_jiao*, *_mftl-sorted,
*-smooth1/2, *_r5_jiao-sorted-every4) needed the library's `sweep_order` and
`sorted_sweeps` and the example's -order, -sorted and -smoothevery. All of these
were removed after the study, so those runs are no longer configured here. Their
data stay in results/leveque_remeshing/, and make_tables.py and make_figures.py
still read them from metrics.json.

The anisotropic mode was removed later too. Its runs below, and the sweep-order
runs, need the library of archive/ (archive/README.md): build that tree in its own
build directory and pass it with --exe. On the current tree pass --only to run the
uniform and curvature rows.
"""
import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODES = ["uniform", "curvature", "anisotropic"]
DT = "0.015"  # Jiao et al.'s step; divides T/4 = 0.75


def configurations():
    runs = []
    # (A) refinement study ("_mftl": MFTL's sweep order, the only one left)
    for r in (3, 4, 5):
        for m in MODES:
            runs.append((f"{m}_r{r}_mftl", ["-refine", str(r), "-sizing", m]))
    # (C) no remeshing: what the time integration alone does
    for r in (3, 4, 5):
        runs.append((f"none_r{r}", ["-refine", str(r), "-noremesh"]))
    return runs


ROTATIONS = (7.0, 13.0, 29.0)


def replicas():
    """The same configurations on the initial sphere rotated about its centre:
    the exact solution is identical, the mesh's alignment with the flow is not,
    so the spread over these is the run-to-run noise of the remeshing decisions."""
    base = [c for c in configurations() if not c[0].startswith("none")]
    out = []
    for i, rot in enumerate(ROTATIONS, start=1):
        for name, args in base:
            out.append((f"{name}@rot{i}", args + ["-rotate", repr(rot)]))
    return out


def cost_rank(name):
    r = 5 if "_r5" in name else 4 if "_r4" in name else 3
    w = {"none": 0, "uniform": 1, "curvature": 3, "anisotropic": 4}[name.split("_")[0]]
    name = name.split("@")[0]
    return -(r * 10 + w)


def run_one(exe, out, name, args):
    d = os.path.join(out, "runs", name)
    os.makedirs(d, exist_ok=True)
    cmd = [exe, "-dt", DT] + args + ["-snapshots", os.path.join(d, "snap"), "-csv", os.path.join(d, "series.csv")]
    t0 = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d).returncode
    dt = time.time() - t0
    print(f"{name:34s} rc={rc}  {dt:7.1f} s", flush=True)
    return name, rc, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "leveque_remeshing_parallel"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "leveque_remeshing"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--replicas", action="store_true", help="run the rotated replicas instead")
    a = ap.parse_args()
    runs = replicas() if a.replicas else configurations()
    if a.only:
        runs = [r for r in runs if r[0] in a.only]
    runs.sort(key=lambda r: cost_rank(r[0]))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda r: run_one(a.exe, a.out, *r), runs))
    bad = [n for n, rc, _ in res if rc != 0]
    print("failed:", bad if bad else "none")


if __name__ == "__main__":
    main()

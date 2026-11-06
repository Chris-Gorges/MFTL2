#!/usr/bin/env python3
"""Evaluates every snapshot of every run against the exact surface.

    python3 evaluate.py [--out results/leveque_remeshing]

Writes OUT/metrics.json: {run: {"q0".."q4": metrics, "series": per-step csv columns}}.
"""
import argparse
import csv
import json
import os
import subprocess
import time

import numpy as np

import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def build_reference():
    exe = os.path.join(HERE, "leveque_reference")
    src = os.path.join(HERE, "leveque_reference.c")
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.run(["cc", "-O2", "-std=c11", "-o", exe, src, "-lm"], check=True)
    return exe


def read_series(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0].keys()} if rows else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "leveque_remeshing"))
    ap.add_argument("--force", action="store_true", help="re-evaluate runs already in metrics.json")
    a = ap.parse_args()
    os.environ["LEVEQUE_REFERENCE"] = build_reference()
    runs_dir = os.path.join(a.out, "runs")
    path = os.path.join(a.out, "metrics.json")
    result = json.load(open(path)) if (os.path.exists(path) and not a.force) else {}
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        q0 = os.path.join(d, "snap_q0.vtk")
        if not os.path.exists(q0) or not os.path.exists(os.path.join(d, "snap_q4.vtk")):
            continue  # not run, or still running
        if name in result and os.path.getmtime(path) > os.path.getmtime(os.path.join(d, "snap_q4.vtk")):
            continue
        t0 = time.time()
        P, F, _ = M.read_vtk(q0)
        V0 = M.volume(P, F)
        entry = {"series": read_series(os.path.join(d, "series.csv"))}
        log = open(os.path.join(d, "log.txt")).read()
        for line in log.splitlines():
            if "wall clock" in line:
                entry["ms_per_step"] = float(line.split("(")[1].split()[0])
            if line.strip().startswith("remesh  ") and " s " in line:
                entry["remesh_s"] = float(line.split()[1])
        for q in range(5):
            entry[f"q{q}"] = M.evaluate(os.path.join(d, f"snap_q{q}.vtk"), V0)
        result[name] = entry
        print(f"{name:36s} {time.time() - t0:6.1f} s", flush=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=float)


if __name__ == "__main__":
    main()

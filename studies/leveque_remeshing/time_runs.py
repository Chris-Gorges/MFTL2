#!/usr/bin/env python3
"""Clean timings: each configuration alone on the machine, no snapshots, no CSV.

    python3 time_runs.py [--out results/leveque_remeshing] [--repeat 2]

Writes OUT/timing.json {run: ms/step (best of --repeat)}.
"""
import argparse
import json
import os
import subprocess

import run_campaign as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(C.ROOT, "build", "leveque_remeshing_parallel"))
    ap.add_argument("--out", default=os.path.join(C.ROOT, "results", "leveque_remeshing"))
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    res = {}
    for name, args in C.configurations():
        best = float("inf")
        for _ in range(a.repeat):
            out = subprocess.run([a.exe, "-dt", C.DT, "-quiet"] + args, capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                if "wall clock" in line:
                    best = min(best, float(line.split("(")[1].split()[0]))
        res[name] = best
        print(f"{name:36s} {best:8.2f} ms/step", flush=True)
    json.dump(res, open(os.path.join(a.out, "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Follow-up to REPORT.md: what the anisotropic mode owes its accuracy to, and
what a back-projection after tangential smoothing does.

    python3 followup.py run [--jobs 4]      # results/leveque_remeshing_followup/runs/
    python3 evaluate.py --out ../../results/leveque_remeshing_followup
    python3 followup.py tables              # markdown tables (base runs from metrics.json of the study)

Every run also logs the enclosed volume change of each sweep (split, collapse,
flip, smooth) per remesh call. The volume budget and the projections are NOT
library features: they need followup_experiment.patch applied to the tree
(MFTL_EXP_BUDGET=1; MFTL_EXP_PROJECT=1 onto the flat input surface, 2 onto its
Phong surface, 3 onto the Phong patch of the current fan; MFTL_EXP_PHONG=1 Phong
midpoints for split and collapse; MFTL_EXP_FLIP=1 Delaunay flips in the quad's
tangent plane, 2 with 3D angles, in uniform and curvature sizing). QEM placement was the library's own
`qem_placement` (the example's -qem). Both the anisotropic mode and QEM placement are removed
from the current tree: apply archive/sizing_with_anisotropic_and_qem.patch to commit bb3c785,
then followup_experiment.patch, and pass that build with --exe.

Superseded: the first "localphong" runs (results/.../runs/*+localphong-flatpre_r4)
were made with a patch whose MFTL_EXP_PROJECT=3 ALSO built the stored input and
projected onto its flat triangles before the fan's Phong lift. The patch now
builds the reference only for PROJECT=1 and 2, so PROJECT=3 is purely local.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "leveque_remeshing_followup")
BASE = os.path.join(ROOT, "results", "leveque_remeshing")
DT = "0.015"
ROT = (None, 7.0, 13.0, 29.0)

P1, P2, DEL, DEL3 = {"MFTL_EXP_PROJECT": "1"}, {"MFTL_EXP_PROJECT": "2"}, {"MFTL_EXP_FLIP": "1"}, {"MFTL_EXP_FLIP": "2"}
FAN, MID = {"MFTL_EXP_PROJECT": "3"}, {"MFTL_EXP_PHONG": "1"}
LOCAL = {**MID, **FAN}
# (name, sizing, example args, experiment environment)
VARIANTS = [
    ("uniform", "uniform", [], {}),
    ("curvature", "curvature", [], {}),
    ("anisotropic", "anisotropic", [], {}),
    ("uniform+qem", "uniform", ["-qem", "on"], {}),
    ("curvature+qem", "curvature", ["-qem", "on"], {}),
    ("anisotropic-qem", "anisotropic", ["-qem", "off"], {}),
    ("uniform+flatproj", "uniform", [], P1),
    ("curvature+flatproj", "curvature", [], P1),
    ("uniform+phongproj", "uniform", [], P2),
    ("curvature+phongproj", "curvature", [], P2),
    ("uniform+qem+phongproj", "uniform", ["-qem", "on"], P2),
    ("curvature+qem+phongproj", "curvature", ["-qem", "on"], P2),
    ("uniform+delaunay", "uniform", [], DEL),
    ("curvature+delaunay", "curvature", [], DEL),
    ("curvature+delaunay3d", "curvature", [], DEL3),
    ("curvature+delaunay+qem", "curvature", ["-qem", "on"], DEL),
    ("curvature+delaunay+qem+phongproj", "curvature", ["-qem", "on"], {**DEL, **P2}),
    # no stored input (base run only): Phong midpoints for split and collapse,
    # projection onto the Phong patch of the current fan after smoothing, and both
    ("uniform+phongmid", "uniform", [], MID),
    ("curvature+phongmid", "curvature", [], MID),
    ("uniform+fanproj", "uniform", [], FAN),
    ("curvature+fanproj", "curvature", [], FAN),
    ("uniform+localphong", "uniform", [], LOCAL),
    ("curvature+localphong", "curvature", [], LOCAL),
]
BASE_ONLY = ("uniform+phongmid", "curvature+phongmid", "uniform+fanproj", "curvature+fanproj", "uniform+localphong", "curvature+localphong")


def runs():
    out = []
    for r, rots in ((4, ROT), (5, ROT[:1])):
        for name, sizing, args, xenv in VARIANTS:
            for i, rot in enumerate(rots if name not in BASE_ONLY else rots[:1]):
                tag = f"{name}_r{r}" + (f"@rot{i}" if rot else "")
                a = ["-refine", str(r), "-sizing", sizing] + args + (["-rotate", repr(rot)] if rot else [])
                # the three plain modes are re-run only for their volume budget (no snapshots)
                out.append((tag, a, xenv, name in ("uniform", "curvature", "anisotropic")))
    return out


def done(tag, budget_only):
    d = os.path.join(OUT, "runs" if not budget_only else "budget", tag)
    log = os.path.join(d, "log.txt")
    return os.path.exists(log) and "wall clock" in open(log).read()


def run_one(exe, tag, args, xenv, budget_only):
    if done(tag, budget_only):
        return 0
    d = os.path.join(OUT, "runs" if not budget_only else "budget", tag)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, MFTL_EXP_BUDGET="1", **xenv)
    cmd = [exe, "-dt", DT, "-quiet"] + args + ["-csv", os.path.join(d, "series.csv")]
    if not budget_only:
        cmd += ["-snapshots", os.path.join(d, "snap")]
    t0 = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write("# " + " ".join(f"{k}={v}" for k, v in xenv.items()) + " " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d, env=env).returncode
    print(f"{tag:40s} rc={rc} {time.time() - t0:7.1f} s", flush=True)
    return rc


def budget(tag):
    for sub in ("runs", "budget"):
        p = os.path.join(OUT, sub, tag, "log.txt")
        if os.path.exists(p):
            rows = [list(map(float, l.split()[1:5])) for l in open(p) if l.startswith("#VB")]
            return np.array(rows).sum(axis=0) if rows else None
    return None


def series_v(tag):
    for sub in ("runs", "budget"):
        p = os.path.join(OUT, sub, tag, "series.csv")
        if os.path.exists(p):
            last = open(p).read().strip().splitlines()[-1].split(",")
            return float(last[8])
    return None


def tables():
    F = json.load(open(os.path.join(OUT, "metrics.json")))
    B = json.load(open(os.path.join(BASE, "metrics.json")))
    base_name = {"uniform": "uniform", "curvature": "curvature", "anisotropic": "anisotropic"}

    def entries(name, r):
        """Metric dicts of the run and its replicas (base modes from the study)."""
        if name in base_name:
            k = f"{name}_r{r}_mftl"
            src = B
        else:
            k = f"{name}_r{r}"
            src = F
        return [src[x] for x in [k] + sorted(x for x in src if x.startswith(k + "@"))] if k in src else []

    def ms(vals, fmt):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        if len(vals) == 1:
            return fmt.format(vals[0])
        return fmt.format(np.mean(vals)) + " ± " + fmt.format(np.std(vals, ddof=1))

    print("| variant | r | V/V(0) - 1 at T [%] | d_rms at T [% R0] | E_H at T | N_H at T | E_H at T/2 | triangles at T | ms/step |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in (4, 5):
        for name, *_ in VARIANTS:
            E = entries(name, r)
            if not E:
                continue
            q4 = [e["q4"] for e in E]
            q2 = [e["q2"] for e in E]
            print(f"| {name} | {r} | {ms([100 * q['dV'] for q in q4], '{:+.2f}')} | {ms([100 * q['d_rms'] for q in q4], '{:.3f}')} | "
                  f"{ms([q['E_H'] for q in q4], '{:.2f}')} | {ms([q['N_H'] for q in q4], '{:.2f}')} | {ms([q['E_H'] for q in q2], '{:.2f}')} | "
                  f"{int(np.mean([q['n_tri'] for q in q4]))} | {E[0].get('ms_per_step', float('nan')):.0f} |")

    print("\n### Volume budget over the period, in % of V(0) (base run of each variant)\n")
    print("| variant | r | split | collapse | flip | smooth | advection | total |")
    print("|---|---|---|---|---|---|---|---|")
    for r in (4, 5):
        for name, *_ in VARIANTS:
            tag = f"{name}_r{r}"
            b = budget(tag)
            v = series_v(tag)
            if b is None or v is None:
                continue
            E = entries(name, r)
            V0 = E[0]["q0"]["V"] if E and "V" in E[0]["q0"] else None
            if V0 is None:
                continue
            p = 100 * b / V0
            total = 100 * (v - 1)
            adv = total - p.sum()
            print(f"| {name} | {r} | {p[0]:+.3f} | {p[1]:+.3f} | {p[2]:+.3f} | {p[3]:+.3f} | {adv:+.3f} | {total:+.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "tables"])
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "leveque_remeshing_parallel"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="run only variants whose name contains this")
    a = ap.parse_args()
    if a.what == "tables":
        return tables()
    todo = [x for x in runs() if not a.only or a.only in x[0]]
    # biggest first
    todo.sort(key=lambda x: (-int(x[1][1]), not x[2], "anisotropic" not in x[0]))
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda x: run_one(a.exe, *x), todo))
    sys.exit(1 if any(rcs) else 0)


if __name__ == "__main__":
    main()

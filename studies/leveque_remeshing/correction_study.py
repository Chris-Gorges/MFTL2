#!/usr/bin/env python3
"""Volume correction and TSUR3D undulation removal in an advection loop, outside the
remesher (MFTLEnsembleCorrectVolumes, MFTLEnsembleRemoveUndulations).

    python3 correction_study.py run [--jobs 4] [--exe B] [--exp-exe E] [--only S]
    python3 evaluate.py --out ../../results/leveque_remeshing_correction   # LeVeque runs only
    python3 correction_study.py tables

Questions:
  1. volume correction: Newton (3 steps, 1 step) against mftl's VolumeRecovery (one step
     dividing by the AREA; needs --exp-exe built with archive/volume_area_step_experiment.patch,
     switched on by MFTL_EXP_VOLAREA=1), and once per step (after the remesh) against twice
     (also right after the advection);
  2. TSUR3D on the LeVeque deformation (filaments the mesh under-resolves), with and without
     a white-noise velocity, every edge or only kinks above 15 degrees;
  3. TSUR3D where it is meant to work: a rigid rotation of the sphere about its own centre
     (exact surface = the sphere at all times) with a white-noise velocity.

LeVeque runs write snapshots for evaluate.py; every run's summary carries the distance from
the exact sphere at T (both fields return the initial sphere at T).
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
OUT = os.path.join(ROOT, "results", "leveque_remeshing_correction")
DT = "0.015"
ROT = (None, 7.0, 13.0, 29.0)
REFINE = (4, 5)

VE = ["-volcorrect", "end"]
VB = ["-volcorrect", "both"]
J = ["-projection", "phong"]
FP = ["-placement", "paraboloid", "-fitradius", "3"]
T0 = ["-tsur3d", "1", "-tsur3d_angle", "0"]  # explicit: the library default is now 15
T15 = ["-tsur3d", "1", "-tsur3d_angle", "15"]
T5 = ["-tsur3d", "1", "-tsur3d_angle", "5"]
N05 = ["-noise", "0.05"]
ROTF = ["-field", "rotation"]
AREA = {"MFTL_EXP_VOLAREA": "1"}

# (name, arguments, environment, needs the experiment build)
VARIANTS = [
    ("base", [], {}, False),
    ("vol_end", VE, {}, False),
    ("vol_both", VB, {}, False),
    ("vol_end_newton1", VE + ["-volsteps", "1"], {}, False),
    ("vol_end_area1", VE + ["-volsteps", "1"], AREA, True),
    ("proj", J, {}, False),
    ("proj+vol_end", J + VE, {}, False),
    ("proj+vol_both", J + VB, {}, False),
    ("proj+fitP3+vol_end", J + FP + VE, {}, False),
    ("tsur0", T0, {}, False),
    ("tsur15", T15, {}, False),
    ("n05_base", N05, {}, False),
    ("n05_tsur0", N05 + T0, {}, False),
    ("n05_tsur15", N05 + T15, {}, False),
    ("n05_tsur15+vol_end", N05 + T15 + VE, {}, False),
    ("rot_base", ROTF, {}, False),
    ("rot_tsur0", ROTF + T0, {}, False),
    ("rot_n05_base", ROTF + N05, {}, False),
    ("rot_n05_tsur0", ROTF + N05 + T0, {}, False),
    ("rot_n05_tsur5", ROTF + N05 + T5, {}, False),
    ("rot_n05_tsur15", ROTF + N05 + T15, {}, False),
    ("rot_n05_tsur0+vol_end", ROTF + N05 + T0 + VE, {}, False),
]
LABEL = {
    "base": "no correction",
    "vol_end": "volume, Newton x3, after the remesh",
    "vol_both": "volume, Newton x3, after advection AND after the remesh",
    "vol_end_newton1": "volume, Newton x1, after the remesh",
    "vol_end_area1": "volume, mftl area step x1, after the remesh",
    "proj": "Phong projection",
    "proj+vol_end": "projection + volume after the remesh",
    "proj+vol_both": "projection + volume twice",
    "proj+fitP3+vol_end": "projection + paraboloid + volume after the remesh",
    "tsur0": "TSUR3D every edge",
    "tsur15": "TSUR3D kinks > 15 deg",
    "n05_base": "noise 0.05, no correction",
    "n05_tsur0": "noise 0.05, TSUR3D every edge",
    "n05_tsur15": "noise 0.05, TSUR3D kinks > 15 deg",
    "n05_tsur15+vol_end": "noise 0.05, TSUR3D > 15 deg + volume",
    "rot_base": "rotation, no correction",
    "rot_tsur0": "rotation, TSUR3D every edge",
    "rot_n05_base": "rotation + noise 0.05, no correction",
    "rot_n05_tsur0": "rotation + noise 0.05, TSUR3D every edge",
    "rot_n05_tsur5": "rotation + noise 0.05, TSUR3D kinks > 5 deg",
    "rot_n05_tsur15": "rotation + noise 0.05, TSUR3D kinks > 15 deg",
    "rot_n05_tsur0+vol_end": "rotation + noise 0.05, TSUR3D every edge + volume",
}


def runs():
    out = []
    for r in REFINE:
        for name, args, env, exp in VARIANTS:
            for i, rot in enumerate(ROT):
                tag = f"uniform_{name}_r{r}" + (f"@rot{i}" if rot else "")
                a = ["-refine", str(r)] + args + (["-rotate", repr(rot)] if rot else [])
                out.append((tag, a, env, exp, not name.startswith("rot_")))
    return out


def done(tag):
    log = os.path.join(OUT, "runs", tag, "log.txt")
    return os.path.exists(log) and "wall clock" in open(log).read()


def run_one(exe, exp_exe, tag, args, env, exp, snapshots):
    if done(tag):
        return 0
    d = os.path.join(OUT, "runs", tag)
    os.makedirs(d, exist_ok=True)
    cmd = [exp_exe if exp else exe, "-dt", DT, "-quiet"] + args + ["-csv", os.path.join(d, "series.csv")]
    if snapshots:
        cmd += ["-snapshots", os.path.join(d, "snap")]
    t0 = time.time()
    with open(os.path.join(d, "log.txt"), "w") as log:
        log.write("# " + " ".join(f"{k}={v}" for k, v in env.items()) + " " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=d, env=dict(os.environ, **env)).returncode
    print(f"{tag:48s} rc={rc} {time.time() - t0:7.1f} s", flush=True)
    return rc


def parse_summary(text):
    v = {}
    m = re.search(r"V/V0 at t=T\s+([\d.]+)", text)
    if m:
        v["dV"] = float(m.group(1)) - 1.0
    m = re.search(r"off the exact sphere\s+rms ([\d.]+) %, mean ([-+\d.]+) %, max ([\d.]+) %", text)
    if m:
        v["sph_rms"], v["sph_mean"], v["sph_max"] = (float(m.group(i)) / 100.0 for i in (1, 2, 3))
    m = re.search(r"min / avg quality\s+([\d.]+) / ([\d.]+)", text)
    if m:
        v["min_q"], v["avg_q"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"wall clock\s+([\d.]+) s\s+\(([\d.]+) ms/step\)", text)
    if m:
        v["ms_per_step"] = float(m.group(2))
    m = re.search(r"corrections\s+([\d.]+) s\s+([\d.]+) %.*largest volume error corrected ([\d.e+-]+)\)", text)
    if m:
        v["t_corr"], v["corr_share"], v["worst_corrected"] = float(m.group(1)), float(m.group(2)) / 100.0, float(m.group(3))
    m = re.search(r"=== LeVeque, .* ([\d]+) -> ([\d]+) triangles", text)
    if m:
        v["tri_T"] = int(m.group(2))
    return v


def tables():
    F = json.load(open(os.path.join(OUT, "metrics.json"))) if os.path.exists(os.path.join(OUT, "metrics.json")) else {}

    def entries(tag):
        tags = [t for t in [tag] + [f"{tag}@rot{i}" for i in range(1, 4)] if os.path.exists(os.path.join(OUT, "runs", t, "log.txt"))]
        return [(t, parse_summary(open(os.path.join(OUT, "runs", t, "log.txt")).read()), F.get(t)) for t in tags]

    def ms(vals, fmt):
        vals = [x for x in vals if x is not None]
        if not vals:
            return "-"
        if len(vals) == 1:
            return fmt.format(vals[0])
        return fmt.format(np.mean(vals)) + " ± " + fmt.replace("+", "").format(np.std(vals, ddof=1))

    print("### At T (mean ± sd over 4 orientations), uniform sizing\n")
    print("| r | variant | V/V(0) − 1 [%] | rms off the sphere [% R0] | max off [% R0] | E_H | N_H | E_H at T/2 | min q | ms/step | corrections share |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in REFINE:
        for name, _, _, _ in VARIANTS:
            E = entries(f"uniform_{name}_r{r}")
            if not E:
                continue
            S = [e[1] for e in E]
            Q4 = [e[2]["q4"] if e[2] else None for e in E]
            Q2 = [e[2]["q2"] if e[2] else None for e in E]
            print(f"| {r} | {LABEL[name]} | {ms([100 * s.get('dV', np.nan) for s in S], '{:+.4f}')} | "
                  f"{ms([100 * s.get('sph_rms', np.nan) for s in S], '{:.3f}')} | {ms([100 * s.get('sph_max', np.nan) for s in S], '{:.2f}')} | "
                  f"{ms([q['E_H'] if q else None for q in Q4], '{:.2f}')} | {ms([q['N_H'] if q else None for q in Q4], '{:.3f}')} | "
                  f"{ms([q['E_H'] if q else None for q in Q2], '{:.2f}')} | {ms([s.get('min_q') for s in S], '{:.3f}')} | "
                  f"{ms([s.get('ms_per_step') for s in S], '{:.1f}')} | {ms([100 * s['corr_share'] if 'corr_share' in s else None for s in S], '{:.1f}')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "tables"])
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "leveque_remeshing_parallel"))
    ap.add_argument("--exp-exe", help="build with archive/volume_area_step_experiment.patch (vol_end_area1)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", help="run only variants whose tag contains this")
    a = ap.parse_args()
    if a.what == "tables":
        return tables()
    todo = [x for x in runs() if (not a.only or a.only in x[0]) and (a.exp_exe or not x[3])]
    todo.sort(key=lambda x: (-int(x[1][1]), x[0]))  # refinement 5 first
    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(lambda x: run_one(a.exe, a.exp_exe, *x), todo))
    sys.exit(1 if any(rcs) else 0)


if __name__ == "__main__":
    main()

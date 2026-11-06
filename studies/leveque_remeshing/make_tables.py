#!/usr/bin/env python3
"""Prints the study's markdown tables from metrics.json (replicas folded in).

    python3 make_tables.py [--out results/leveque_remeshing] > tables.md
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODES = ["uniform", "curvature", "anisotropic"]
LABEL = {"uniform": "uniform", "curvature": "curvature", "anisotropic": "anisotropic"}


def group(R, base):
    """The base run (first) and its rotated replicas."""
    out = []
    for k in [base] + sorted(k for k in R if k.startswith(base + "@")):
        if k in R:
            R[k]["_name"] = k
            out.append(R[k])
    return out


def stat(runs, q, key, scale=1.0):
    v = np.array([r[f"q{q}"][key] for r in runs]) * scale
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def pm(runs, q, key, scale=1.0, fmt="{:+.2f}", sdfmt="{:.2f}"):
    m, s, n = stat(runs, q, key, scale)
    return (fmt.format(m) + " ± " + sdfmt.format(s)) if n > 1 else fmt.format(m)


def val(runs, q, key, scale=1.0, fmt="{:.3g}"):
    m, _, _ = stat(runs, q, key, scale)
    return fmt.format(m)


TIMING = {}


def ms(runs):
    """ms/step of the BASE run (the first of a group), from the sequential
    timing pass when there is one: replicas ran four at a time."""
    name = runs[0].get("_name")
    if name in TIMING:
        return "{:.0f}".format(TIMING[name])
    return "{:.0f}".format(runs[0].get("ms_per_step", np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "leveque_remeshing"))
    ap.add_argument("--template", help="markdown with <!-- table:N --> markers, N the 0-based table section")
    ap.add_argument("--report", help="where the filled template goes")
    a = ap.parse_args()
    if a.template:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tables(a.out)
        sections = [sec.split("\n", 1)[1].strip() for sec in buf.getvalue().split("### ")[1:]]
        text = open(a.template).read()
        for i, body in enumerate(sections):
            text = text.replace(f"<!-- table:{i} -->", body)
        open(a.report, "w").write(text)
        return
    tables(a.out)


def tables(out):
    class A:
        pass

    a = A()
    a.out = out
    R = json.load(open(os.path.join(a.out, "metrics.json")))
    tpath = os.path.join(a.out, "timing.json")
    if os.path.exists(tpath):
        TIMING.update(json.load(open(tpath)))

    print("### Volume, V/V(0) - 1 in % (mean ± sd over the run and its three rotated replicas)\n")
    print("| mode | refine | T/4 | T/2 | 3T/4 | **T** | area A/A(0) - 1 at T [%] |")
    print("|---|---|---|---|---|---|---|")

    def area_T(g):
        v = np.array([100.0 * (r_["series"]["A_over_A0"][-1] - 1.0) for r_ in g])
        return "{:+.2f}".format(v.mean()) + (" ± {:.2f}".format(v.std(ddof=1)) if len(v) > 1 else "")

    for m in MODES:
        for r in (3, 4, 5):
            g = group(R, f"{m}_r{r}_mftl")
            cells = [pm(g, q, "dV", 100.0) for q in (1, 2, 3)] + ["**" + pm(g, 4, "dV", 100.0) + "**"]
            print(f"| {LABEL[m]} | {r} | " + " | ".join(cells) + f" | {area_T(g)} |")
    for r in (3, 4, 5):
        g = group(R, f"none_r{r}")
        print(f"| no remeshing | {r} | " + " | ".join(pm(g, q, "dV", 100.0) for q in (1, 2, 3, 4)) + f" | {area_T(g)} |")

    for q, title in ((2, "t = T/2 (largest deformation)"), (4, "t = T (the exact surface is the initial sphere)")):
        print(f"\n### Accuracy and smoothness at {title}\n")
        print("| mode | refine | triangles | d_rms [% R0] | d_mean [% R0] | normal err. RMS [°] | E_H | N_H | E_H on projected mesh |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in MODES:
            for r in (3, 4, 5):
                g = group(R, f"{m}_r{r}_mftl")
                print(f"| {LABEL[m]} | {r} | {val(g, q, 'n_tri', fmt='{:.0f}')} | {val(g, q, 'd_rms', 100, '{:.3f}')} | "
                      f"{val(g, q, 'd_mean', 100, '{:+.3f}')} | {val(g, q, 'nerr_rms', 1, '{:.2f}')} | {pm(g, q, 'E_H', 1, '{:.2f}', '{:.2f}')} | "
                      f"{pm(g, q, 'N_H', 1, '{:.2f}', '{:.2f}')} | {val(g, q, 'E_H_proj', 1, '{:.3f}')} |")

    print("\n### Mesh and cost\n")
    print("| mode | refine | peak triangles | triangles at T | min q (worst of T/4..T) | mean q at T | faces against the exact normal (max over T/4..T) | ms/step |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for r in (3, 4, 5):
            g = group(R, f"{m}_r{r}_mftl")
            peak = np.mean([max(r_["series"]["ntri"]) for r_ in g])
            qmin = np.mean([min(r_[f"q{q}"]["q_min"] for q in (1, 2, 3, 4)) for r_ in g])
            opp = np.mean([max(r_[f"q{q}"]["inverted"] for q in (1, 2, 3, 4)) for r_ in g])
            print(f"| {LABEL[m]} | {r} | {peak:.0f} | {val(g, 4, 'n_tri', fmt='{:.0f}')} | {qmin:.3f} | {val(g, 4, 'q_mean', fmt='{:.3f}')} | {opp:.1f} | {ms(g)} |")

    print("\n### Sweep order at refinement 4\n")
    orders = [("mftl", "MFTL (default)"), ("mftl-sorted", "MFTL, length-sorted"), ("jiao", "Jiao"), ("jiao-sorted", "Jiao, length-sorted"),
              ("mftl-smooth2", "MFTL, 2 smoothing sweeps"), ("jiao-smooth1", "Jiao, 2 x 1 smoothing sweeps")]
    print("| mode | order | V/V(0) - 1 at T/2 [%] | V/V(0) - 1 at T [%] | d_rms at T [% R0] | E_H at T | N_H at T | min q | triangles at T/2 / T | ms/step |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for o, lab in orders:
            g = group(R, f"{m}_r4_{o}")
            if not g:
                continue
            qmin = np.mean([min(r_[f"q{q}"]["q_min"] for q in (1, 2, 3, 4)) for r_ in g])
            print(f"| {LABEL[m]} | {lab} | {pm(g, 2, 'dV', 100.0)} | {pm(g, 4, 'dV', 100.0)} | {val(g, 4, 'd_rms', 100, '{:.2f}')} | "
                  f"{pm(g, 4, 'E_H', 1, '{:.2f}', '{:.2f}')} | {pm(g, 4, 'N_H', 1, '{:.2f}', '{:.2f}')} | {qmin:.3f} | "
                  f"{val(g, 2, 'n_tri', fmt='{:.0f}')} / {val(g, 4, 'n_tri', fmt='{:.0f}')} | {ms(g)} |")

    print("\n### Jiao et al.'s protocol at refinement 5 (full adaptation every 4 steps, Jiao order, sorted; smoothing every step)\n")
    print("| mode | protocol | V/V(0) - 1 at T/2 [%] | V/V(0) - 1 at T [%] | d_rms at T [% R0] | E_H at T | N_H at T | triangles at T | ms/step |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MODES:
        for o, lab in (("mftl", "every step, MFTL order"), ("jiao-sorted-every4", "Jiao's protocol")):
            g = group(R, f"{m}_r5_{o}")
            print(f"| {LABEL[m]} | {lab} | {val(g, 2, 'dV', 100, '{:+.3f}')} | {val(g, 4, 'dV', 100, '{:+.3f}')} | {val(g, 4, 'd_rms', 100, '{:.3f}')} | "
                  f"{val(g, 4, 'E_H', 1, '{:.2f}')} | {val(g, 4, 'N_H', 1, '{:.2f}')} | {val(g, 4, 'n_tri', fmt='{:.0f}')} | {ms(g)} |")


if __name__ == "__main__":
    main()

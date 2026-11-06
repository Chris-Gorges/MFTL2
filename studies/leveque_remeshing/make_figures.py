#!/usr/bin/env python3
"""Renders the study's figures from the snapshots and metrics.json.

    python3 make_figures.py [--out results/leveque_remeshing] [--fig studies/leveque_remeshing/figures]
"""
import argparse
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PVBATCH = "/Applications/ParaView-6.0.1.app/Contents/bin/pvbatch"
MODES = ["uniform", "curvature", "anisotropic"]
MODE_LABEL = {"uniform": "uniform", "curvature": "curvature (Dunyach)", "anisotropic": "anisotropic (Jiao)"}
QLABEL = {1: "t = T/4", 2: "t = T/2", 3: "t = 3T/4", 4: "t = T"}
VIEW = np.array([1.0, -1.0, 0.8]) / np.linalg.norm([1.0, -1.0, 0.8])


def write_vtk_with_field(path, P, F, name, values):
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nleveque\nBINARY\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(P)} double\n".encode())
        f.write(np.ascontiguousarray(P, dtype=">f8").tobytes())
        f.write(f"\nPOLYGONS {len(F)} {4 * len(F)}\n".encode())
        rec = np.concatenate([np.full((len(F), 1), 3), F], axis=1).astype(">i4")
        f.write(rec.tobytes())
        f.write(f"\nPOINT_DATA {len(P)}\nSCALARS {name} double 1\nLOOKUP_TABLE default\n".encode())
        f.write(np.ascontiguousarray(values, dtype=">f8").tobytes())
        f.write(b"\n")


def camera_for(path, margin=1.04):
    """A parallel camera along VIEW framing the snapshot's PROJECTED extent."""
    P, _, _ = M.read_vtk(path)
    up = np.array([0.0, 0.0, 1.0])
    up = up - VIEW * (up @ VIEW)
    up /= np.linalg.norm(up)
    right = np.cross(up, VIEW)
    x, y, z = P @ right, P @ up, P @ VIEW
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    c = cx * right + cy * up + 0.5 * (z.min() + z.max()) * VIEW
    half = 0.5 * max(x.max() - x.min(), y.max() - y.min())
    return {"focal": c.tolist(), "position": (c + 3.0 * VIEW).tolist(), "up": up.tolist(), "parallel_scale": margin * float(half)}


def run_pv(jobs, tmp):
    path = os.path.join(tmp, "jobs.json")
    json.dump(jobs, open(path, "w"))
    subprocess.run([PVBATCH, "--force-offscreen-rendering", os.path.join(HERE, "render_pv.py"), path], check=True, stdout=subprocess.DEVNULL)


def grid(images, row_labels, col_labels, out, cell=3.0, title=None, cbar=None):
    nr, nc = len(images), len(images[0])
    fig, ax = plt.subplots(nr, nc, figsize=(cell * nc + 0.9, cell * nr + (0.5 if title else 0.2)), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            a = ax[i][j]
            if images[i][j] is not None:
                a.imshow(mpimg.imread(images[i][j]))
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if i == 0:
                a.set_title(col_labels[j], fontsize=11)
            if j == 0:
                a.set_ylabel(row_labels[i], fontsize=11)
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if cbar:
        cmap, lo, hi, label = cbar
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        fig.colorbar(sm, cax=cax, label=label)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "leveque_remeshing"))
    ap.add_argument("--fig", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    from evaluate import build_reference

    os.environ["LEVEQUE_REFERENCE"] = build_reference()
    runs = os.path.join(a.out, "runs")
    tmp = os.path.join(a.out, "render")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(a.fig, exist_ok=True)
    R = json.load(open(os.path.join(a.out, "metrics.json")))

    cams = {q: camera_for(os.path.join(runs, "uniform_r5_mftl", f"snap_q{q}.vtk")) for q in (1, 2, 3, 4)}
    jobs = []

    def panel(run, q, tag, **kw):
        out = os.path.join(tmp, f"{run}_q{q}_{tag}.png")
        job = {"in": os.path.join(runs, run, f"snap_q{q}.vtk"), "out": out, "size": [700, 700]}
        job.update(cams[q])
        job.update(kw)
        jobs.append(job)
        return out

    # 1. shapes and meshes, one figure per refinement
    shape_imgs = {}
    for r in (3, 4, 5):
        ew = {3: 1.2, 4: 0.8, 5: 0.45}[r]
        shape_imgs[r] = [[panel(f"{m}_r{r}_mftl", q, "mesh", edges=True, edge_width=ew) for q in (1, 2, 3, 4)] for m in MODES]

    # 2. close-up of the triangulation at T/2 where the exact curvature peaks
    P, F, t = M.read_vtk(os.path.join(runs, "uniform_r5_mftl", "snap_q2.vtk"))
    _, _, Hex, _, _ = M.reference(P, t)
    spot = P[np.argmax(np.abs(Hex))]
    zoom_cam = {"focal": spot.tolist(), "position": (spot + 3.0 * VIEW).tolist(), "up": [0.0, 0.0, 1.0], "parallel_scale": 0.045}
    zoom_imgs = []
    for r in (4, 5):
        row = []
        for m in MODES:
            out = os.path.join(tmp, f"zoom_{m}_r{r}.png")
            job = {"in": os.path.join(runs, f"{m}_r{r}_mftl", "snap_q2.vtk"), "out": out, "size": [700, 700], "edges": True, "edge_width": 0.9}
            job.update(zoom_cam)
            jobs.append(job)
            row.append(out)
        zoom_imgs.append(row)

    # 3. the returned sphere coloured by its discrete mean curvature, H R0
    def curvature_panel(run):
        src = os.path.join(runs, run, "snap_q4.vtk")
        P, F, _ = M.read_vtk(src)
        H, _, _, _, _, _ = M.cotan_curvature(P, F)
        vtk = os.path.join(tmp, f"{run}_q4_H.vtk")
        write_vtk_with_field(vtk, P, F, "HR", H * M.R0)
        out = os.path.join(tmp, f"{run}_q4_H.png")
        job = {"in": vtk, "out": out, "size": [600, 600], "edges": False, "array": "HR", "range": [0.0, 2.0], "cmap": "Cool to Warm"}
        job.update(cams[4])
        jobs.append(job)
        return out

    curv_imgs = [[curvature_panel(f"{m}_r{r}_mftl") for r in (3, 4, 5)] for m in MODES]
    ORDERS = [("mftl", "MFTL order"), ("mftl-sorted", "MFTL, length-sorted"), ("jiao", "Jiao order"), ("jiao-sorted", "Jiao, length-sorted")]
    order_curv = [[curvature_panel(f"{m}_r4_{o}") for o, _ in ORDERS] for m in MODES]
    order_mesh = [[panel(f"{m}_r4_{o}", 4, "mesh", edges=True, edge_width=0.8) for o, _ in ORDERS] for m in MODES]

    run_pv(jobs, tmp)

    rows = [MODE_LABEL[m] for m in MODES]
    cols = [QLABEL[q] for q in (1, 2, 3, 4)]
    for r in (3, 4, 5):
        grid(shape_imgs[r], rows, cols, os.path.join(a.fig, f"shapes_r{r}.png"), title=f"refinement {r} ({20 * 4 ** r} initial triangles), MFTL sweep order")
    grid(zoom_imgs, ["refinement 4", "refinement 5"], [MODE_LABEL[m] for m in MODES], os.path.join(a.fig, "zoom_T2.png"),
         title="t = T/2, close-up where the exact curvature is largest")
    grid(curv_imgs, rows, [f"refinement {r}" for r in (3, 4, 5)], os.path.join(a.fig, "curvature_T.png"),
         title="t = T: discrete mean curvature H R0 (exact: 1)", cbar=("coolwarm", 0.0, 2.0, "H R0"))
    grid(order_curv, rows, [l for _, l in ORDERS], os.path.join(a.fig, "orders_curvature_T_r4.png"),
         title="refinement 4, t = T: discrete mean curvature H R0 by sweep order", cbar=("coolwarm", 0.0, 2.0, "H R0"))
    grid(order_mesh, rows, [l for _, l in ORDERS], os.path.join(a.fig, "orders_mesh_T_r4.png"), title="refinement 4, t = T: mesh by sweep order")

    # 4. time series
    colors = {"uniform": "tab:blue", "curvature": "tab:orange", "anisotropic": "tab:green"}
    styles = {3: ":", 4: "--", 5: "-"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in MODES:
        for r in (3, 4, 5):
            s = R[f"{m}_r{r}_mftl"]["series"]
            t = np.array(s["t"]) / 3.0
            ax[0].plot(t, 100 * (np.array(s["V_over_V0"]) - 1), styles[r], color=colors[m], label=f"{m} r{r}")
            ax[1].semilogy(t, s["ntri"], styles[r], color=colors[m])
    for r in (3, 4, 5):
        s = R[f"none_r{r}"]["series"]
        ax[0].plot(np.array(s["t"]) / 3.0, 100 * (np.array(s["V_over_V0"]) - 1), styles[r], color="0.6", lw=0.8, label=f"no remeshing r{r}" if r == 5 else None)
    ax[0].set_ylim(-26, 3)
    ax[0].set_xlabel("t / T")
    ax[0].set_ylabel("V / V(0) - 1  [%]")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title("enclosed volume")
    ax[1].set_xlabel("t / T")
    ax[1].set_ylabel("triangles")
    ax[1].set_title("triangle count")
    for x in (0.25, 0.5, 0.75):
        ax[0].axvline(x, color="0.85", lw=0.5)
        ax[1].axvline(x, color="0.85", lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(a.fig, "volume_and_count.png"), dpi=110)
    plt.close(fig)

    # zoomed volume of the r5 runs
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODES:
        s = R[f"{m}_r5_mftl"]["series"]
        ax.plot(np.array(s["t"]) / 3.0, 100 * (np.array(s["V_over_V0"]) - 1), color=colors[m], label=m)
        s = R[f"{m}_r5_jiao-sorted-every4"]["series"]
        ax.plot(np.array(s["t"]) / 3.0, 100 * (np.array(s["V_over_V0"]) - 1), "--", color=colors[m], lw=1, label=f"{m}, Jiao protocol")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-0.1, 0.1, color="0.9", zorder=0, label="Jiao et al.: < 0.1 %")
    ax.set_xlabel("t / T")
    ax.set_ylabel("V / V(0) - 1  [%]")
    ax.set_title("refinement 5")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(a.fig, "volume_r5.png"), dpi=110)
    plt.close(fig)

    # sweep orders at r4
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ostyles = {"mftl": ("-", "MFTL"), "mftl-sorted": ("--", "MFTL, sorted"), "jiao": ("-", "Jiao"), "jiao-sorted": ("--", "Jiao, sorted"),
               "mftl-smooth2": (":", "MFTL, 2 smoothing sweeps"), "jiao-smooth1": (":", "Jiao, 2 x 1 smoothing sweeps")}
    ocolor = {"mftl": "tab:blue", "mftl-sorted": "tab:blue", "mftl-smooth2": "tab:blue", "jiao": "tab:red", "jiao-sorted": "tab:red", "jiao-smooth1": "tab:red"}
    for i, m in enumerate(MODES):
        for o, (ls, lab) in ostyles.items():
            s = R[f"{m}_r4_{o}"]["series"]
            ax[i].plot(np.array(s["t"]) / 3.0, 100 * (np.array(s["V_over_V0"]) - 1), ls, color=ocolor[o], label=lab)
        ax[i].axhline(0, color="k", lw=0.5)
        ax[i].set_title(MODE_LABEL[m] + ", refinement 4")
        ax[i].set_xlabel("t / T")
        ax[i].set_ylabel("V / V(0) - 1  [%]")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(a.fig, "orders_volume_r4.png"), dpi=110)
    plt.close(fig)

    # curvature error and noise at the quarter periods
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for m in MODES:
        for r in (3, 4, 5):
            e = R[f"{m}_r{r}_mftl"]
            qs = [0.25, 0.5, 0.75, 1.0]
            ax[0].semilogy(qs, [100 * e[f"q{q}"]["d_rms"] for q in (1, 2, 3, 4)], styles[r], marker="o", ms=3, color=colors[m], label=f"{m} r{r}")
            ax[1].semilogy(qs, [e[f"q{q}"]["E_H"] for q in (1, 2, 3, 4)], styles[r], marker="o", ms=3, color=colors[m])
            ax[2].semilogy(qs, [e[f"q{q}"]["N_H"] for q in (1, 2, 3, 4)], styles[r], marker="o", ms=3, color=colors[m])
    ax[0].set_title("vertex distance to the exact surface, RMS [% R0]")
    ax[1].set_title("curvature error E_H")
    ax[2].set_title("curvature noise N_H (smoothness)")
    for x in ax:
        x.set_xlabel("t / T")
        x.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax[0].legend(ncol=3, fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(a.fig, "errors_quarters.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

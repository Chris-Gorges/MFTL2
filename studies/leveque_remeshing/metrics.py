"""Error and smoothness metrics of one LeVeque snapshot against the exact surface.

Everything here is measured against `leveque_reference` (the exact level set
traced back to the initial sphere), never against another mesh:

  volume      V / V(0) - 1, V(0) the initial discrete volume
  position    signed distance d of every vertex to the exact surface:
              area-weighted mean (bias), RMS and max, in units of R0 = 0.15
  normal      angle between each face normal and the exact normal at its
              centroid: area-weighted RMS and max (degrees); a face whose normal
              points against the exact one is counted as INVERTED
  curvature   discrete mean curvature H_h (cotangent Laplacian, Meyer mixed
              area: what a front-tracking solver computes for surface tension)
              against the exact H at the vertex's foot point:
                E_H = ||H_h - H||_A / ||H||_A          (accuracy)
                N_H = ||H_h - <H_h>_ring||_A / ||H||_A (noise: smoothness; no
                      reference needed except for the scale)
              with ||f||_A^2 = sum_v A_v f_v^2 over the mixed areas.
  mesh        triangle/vertex counts, min/mean quality 4 sqrt3 A / sum l^2,
              share of triangles with quality < 0.3, max aspect (longest
              edge over smallest height)
"""
import os
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp

R0 = 0.15
HERE = os.path.dirname(os.path.abspath(__file__))


def read_vtk(path):
    data = open(path, "rb").read()
    head = data[: data.index(b"POINTS")].decode()
    t = float(head.split("t=")[1].split()[0])
    i = data.index(b"POINTS")
    e = data.index(b"\n", i)
    n = int(data[i:e].split()[1])
    P = np.frombuffer(data, dtype=">f8", count=3 * n, offset=e + 1).reshape(n, 3).astype(np.float64)
    j = data.index(b"POLYGONS", e + 1 + 24 * n)
    e2 = data.index(b"\n", j)
    m = int(data[j:e2].split()[1])
    F = np.frombuffer(data, dtype=">i4", count=4 * m, offset=e2 + 1).reshape(m, 4)[:, 1:].astype(np.int64)
    return P, F, t


def reference(P, t, T=3.0, per_unit=1000.0, exe=None):
    """d, n (n,3), H, K, |grad phi| at every point (exact surface at time t)."""
    exe = exe or os.environ.get("LEVEQUE_REFERENCE", os.path.join(HERE, "leveque_reference"))
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        with open(fin, "wb") as f:
            np.array([len(P)], dtype=np.int64).tofile(f)
            np.ascontiguousarray(P, dtype=np.float64).tofile(f)
        subprocess.run([exe, repr(T), repr(t), repr(per_unit), fin, fout], check=True)
        out = np.fromfile(fout, dtype=np.float64).reshape(-1, 7)
    return out[:, 0], out[:, 1:4], out[:, 4], out[:, 5], out[:, 6]


def reference_parallel(P, t, T=3.0, per_unit=1000.0, chunks=4):
    """reference() split over `chunks` processes."""
    if len(P) < 20000 or chunks <= 1:
        return reference(P, t, T, per_unit)
    from concurrent.futures import ThreadPoolExecutor
    parts = np.array_split(np.arange(len(P)), chunks)
    with ThreadPoolExecutor(chunks) as ex:
        res = list(ex.map(lambda idx: reference(P[idx], t, T, per_unit), parts))
    return tuple(np.concatenate([r[k] for r in res]) for k in range(5))


def mesh_geometry(P, F):
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    area = 0.5 * area2
    with np.errstate(invalid="ignore", divide="ignore"):
        nf = cr / area2[:, None]
    return a, b, c, area, nf


def volume(P, F):
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def cotan_curvature(P, F):
    """Mean curvature (cotangent Laplacian / Meyer mixed area), area-weighted
    vertex normals, mixed areas and the vertex adjacency."""
    n = len(P)
    a, b, c, area, nf = mesh_geometry(P, F)
    corners = [a, b, c]
    # edge vectors opposite each corner: e0 = c - b (opposite a), ...
    e = [corners[2] - corners[1], corners[0] - corners[2], corners[1] - corners[0]]
    l2 = [np.einsum("ij,ij->i", x, x) for x in e]
    area2 = 2.0 * area
    safe = np.where(area2 > 0, area2, np.inf)
    # cot at corner k = (u . v) / |u x v| with u, v the two edges leaving it
    cot = []
    for k in range(3):
        u = corners[(k + 1) % 3] - corners[k]
        v = corners[(k + 2) % 3] - corners[k]
        cot.append(np.einsum("ij,ij->i", u, v) / safe)
    lap = np.zeros((n, 3))
    for k in range(3):
        i, j = F[:, (k + 1) % 3], F[:, (k + 2) % 3]  # the edge opposite corner k
        w = cot[k][:, None]
        d = P[i] - P[j]
        np.add.at(lap, i, w * d)
        np.add.at(lap, j, -w * d)
    # Meyer mixed area
    amix = np.zeros(n)
    obtuse = [cot[k] < 0 for k in range(3)]
    any_obtuse = obtuse[0] | obtuse[1] | obtuse[2]
    for k in range(3):
        vk = F[:, k]
        # Voronoi part at corner k: (|e_{k+1}|^2 cot_{k+1} + |e_{k+2}|^2 cot_{k+2}) / 8
        vor = (l2[(k + 1) % 3] * cot[(k + 1) % 3] + l2[(k + 2) % 3] * cot[(k + 2) % 3]) / 8.0
        val = np.where(any_obtuse, np.where(obtuse[k], area / 2.0, area / 4.0), vor)
        np.add.at(amix, vk, val)
    vn = np.zeros((n, 3))
    for k in range(3):
        np.add.at(vn, F[:, k], nf * area[:, None])
    vn /= np.linalg.norm(vn, axis=1)[:, None]
    Kn = lap / (2.0 * amix[:, None])  # = 2 H n
    H = 0.5 * np.einsum("ij,ij->i", Kn, vn)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adj.data[:] = 1.0
    return H, vn, amix, adj, area, nf


def quality(P, F):
    a, b, c, area, _ = mesh_geometry(P, F)
    l = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=1)
    q = 4.0 * np.sqrt(3.0) * area / np.maximum((l * l).sum(axis=1), 1e-300)
    lmax = l.max(axis=1)
    with np.errstate(divide="ignore"):
        aspect = lmax * lmax / np.maximum(2.0 * area, 1e-300)  # longest edge over smallest height
    return q, aspect


def evaluate(path, V0, T=3.0, per_unit=1000.0, chunks=4):
    P, F, t = read_vtk(path)
    d, n_ex, H_ex, K_ex, gphi = reference_parallel(P, t, T, per_unit, chunks)
    H, vn, amix, adj, area, nf = cotan_curvature(P, F)
    q, aspect = quality(P, F)

    A = amix
    Atot = A.sum()
    w = A / Atot
    out = {"t": t, "n_tri": len(F), "n_vert": len(P)}
    V = volume(P, F)
    out["V"] = V
    out["dV"] = V / V0 - 1.0 if V0 else 0.0
    out["area"] = area.sum()
    out["d_mean"] = float((w * d).sum()) / R0
    out["d_rms"] = float(np.sqrt((w * d * d).sum())) / R0
    out["d_max"] = float(np.abs(d).max()) / R0

    # exact normal at a face centroid from its three vertices (O(h^2) off; the
    # face normal's own error is O(h))
    ne = n_ex[F[:, 0]] + n_ex[F[:, 1]] + n_ex[F[:, 2]]
    ne /= np.linalg.norm(ne, axis=1)[:, None]
    cosang = np.clip(np.einsum("ij,ij->i", nf, ne), -1.0, 1.0)
    # the chord |n_f - n_e| -> angle, robust near 0 (CLAUDE.md: never acos(a.b) near 1)
    chord = np.linalg.norm(nf - ne, axis=1)
    ang = 2.0 * np.degrees(np.arcsin(np.clip(0.5 * chord, 0.0, 1.0)))
    wa = area / area.sum()
    out["nerr_rms"] = float(np.sqrt((wa * ang * ang).sum()))
    out["nerr_max"] = float(ang.max())
    out["inverted"] = int((cosang < 0.0).sum())

    Hn = float(np.sqrt((w * H_ex * H_ex).sum()))
    out["H_scale"] = Hn
    out["E_H"] = float(np.sqrt((w * (H - H_ex) ** 2).sum())) / Hn
    out["Einf_H"] = float(np.abs(H - H_ex).max()) / Hn
    # one-ring area-weighted mean of the discrete H (neighbours only)
    num = adj @ (A * H)
    den = adj @ A
    Hbar = num / den
    out["N_H"] = float(np.sqrt((w * (H - Hbar) ** 2).sum())) / Hn
    out["Hex_max"] = float(np.abs(H_ex).max())

    # The same curvature and normals on the mesh with every vertex moved onto
    # the exact surface (one Newton step, x - d n): what is left there is the
    # TRIANGULATION's share of the error (the cotangent Laplacian is not
    # pointwise consistent on irregular meshes); the difference to the values
    # above is the share of the vertices being off the surface (roughness).
    Ps = P - d[:, None] * n_ex
    Hs, _, As, _, area_s, nf_s = cotan_curvature(Ps, F)
    ws = As / As.sum()
    out["E_H_proj"] = float(np.sqrt((ws * (Hs - H_ex) ** 2).sum())) / Hn
    Hbar_s = (adj @ (As * Hs)) / (adj @ As)
    out["N_H_proj"] = float(np.sqrt((ws * (Hs - Hbar_s) ** 2).sum())) / Hn
    chord_s = np.linalg.norm(nf_s - ne, axis=1)
    ang_s = 2.0 * np.degrees(np.arcsin(np.clip(0.5 * chord_s, 0.0, 1.0)))
    was = area_s / area_s.sum()
    out["nerr_rms_proj"] = float(np.sqrt((was * ang_s * ang_s).sum()))
    # mean edge length, for scale
    a_, b_, c_, _, _ = mesh_geometry(P, F)
    out["h_mean"] = float(np.mean(np.concatenate([np.linalg.norm(b_ - a_, axis=1), np.linalg.norm(c_ - b_, axis=1), np.linalg.norm(a_ - c_, axis=1)])))

    out["q_min"] = float(q.min())
    out["q_mean"] = float(q.mean())
    out["q_lt03"] = float((q < 0.3).mean())
    out["aspect_max"] = float(aspect.max())
    edges = adj.nnz // 2
    out["euler"] = len(P) - edges + len(F)
    return out

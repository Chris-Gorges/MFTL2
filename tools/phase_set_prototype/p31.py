"""Revision 3.1 of plans/16 §3.6 (SUPERSEDED by revision 3.2, p36.py; kept for the review record of 2026-09-24, §18.2).
admissible axis k = z. Implements the text LITERALLY (mode='literal') and with the reviewer's
corrections (mode='fixed'):
  F1 local rays stay through the member (small deterministic perturbation inside one of its
     own triangles) and placement uses the member's own crossings on that ray;
  F2 the window start W0 is taken at a gap between layer crossing spans, not at the smallest
     reduced entering crossing;
  F3 global orientation pinning: members whose local ray meets no other member must all have the
     same exterior level (no-layer case), and V = V_raw mod V_cell for periodic k.
Exact per-triangle integrals: int_T z n_z dA = A_xy * mean(z); int_T z^2 n_z dA = A_xy*(sum z_i^2 + sum z_i z_j)/6.
"""
import math, sys
import numpy as np

L = np.array([1.0, 1.0, 1.0])
A = L[0] * L[1]
VC = A * L[2]
PHI1 = (math.sqrt(5) - 1) / 2
PHI2 = math.sqrt(2) - 1
ETA_B, ETA_N, ETA_Z = 1e-6, 1e-6, 1e-9


class Member:
    def __init__(self, V, T, sigma, name):
        self.V = np.asarray(V, float)  # continuous coordinates (own gauge)
        self.T = np.asarray(T, int)
        self.sigma = sigma
        self.name = name
        self.shift = 0.0  # placed gauge offset along z

    def tri(self, t):
        return self.V[self.T[t]]


def sphere(c, r, n=24, sigma=1, name='sph'):
    V, T = [], []
    V.append([c[0], c[1], c[2] + r])
    for i in range(1, n):
        th = math.pi * i / n
        for j in range(2 * n):
            ph = 2 * math.pi * j / (2 * n)
            V.append([c[0] + r * math.sin(th) * math.cos(ph), c[1] + r * math.sin(th) * math.sin(ph), c[2] + r * math.cos(th)])
    V.append([c[0], c[1], c[2] - r])
    m = 2 * n
    idx = lambda i, j: 1 + (i - 1) * m + (j % m)
    for j in range(m):
        T.append([0, idx(1, j), idx(1, j + 1)])
    for i in range(1, n - 1):
        for j in range(m):
            a, b, cc, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            T.append([a, d, cc]); T.append([a, cc, b])
    last = len(V) - 1
    for j in range(m):
        T.append([idx(n - 1, j), last, idx(n - 1, j + 1)])
    return Member(V, T, sigma, name)


def sheet(f, N=48, up=True, sigma=1, name='sheet', zoff=0.0):
    """z = f(x,y) over the periodic unit square, continuous lateral coordinates; normal +z if up."""
    V, T = [], []
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = i / N, j / N
            V.append([x, y, f(x, y) + zoff])
    idx = lambda i, j: j * (N + 1) + i
    for j in range(N):
        for i in range(N):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if up:
                T.append([a, b, c]); T.append([a, c, d])
            else:
                T.append([a, c, b]); T.append([a, d, c])
    return Member(V, T, sigma, name)


def curve_sheet(xz, N=400, M=8, up=True, sigma=1, name='fold'):
    """(x(u), y, z(u)), u in [0,1], x(u+1)=x(u)+1: a folded (overhanging) layer."""
    V, T = [], []
    for j in range(M + 1):
        for i in range(N + 1):
            u = i / N
            x, z = xz(u)
            V.append([x, j / M, z])
    idx = lambda i, j: j * (N + 1) + i
    for j in range(M):
        for i in range(N):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if up:
                T.append([a, b, c]); T.append([a, c, d])
            else:
                T.append([a, c, b]); T.append([a, d, c])
    return Member(V, T, sigma, name)


def integrals(m):
    P = m.V[m.T]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    n = 0.5 * np.cross(e1, e2)  # area vectors
    z = P[:, :, 2] + m.shift
    zs = z.sum(1)
    z2 = (z ** 2).sum(1) + z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 0] * z[:, 2]
    phi = m.sigma * n.sum(0)
    v = m.sigma * (n[:, 2] * zs / 3).sum()
    c2 = m.sigma * (n[:, 2] * z2 / 6).sum()
    return phi, v, c2


def crossings(m, x0, y0):
    """all crossings of the vertical line (x0,y0) (lateral periodic) with m: (z_own, sign s, generic)."""
    out, generic = [], True
    P = m.V[m.T]
    for t in range(len(m.T)):
        p = P[t]
        lo, hi = p[:, :2].min(0), p[:, :2].max(0)
        for a in (-2, -1, 0, 1, 2):
            for b in (-2, -1, 0, 1, 2):
                X, Y = x0 + a * L[0], y0 + b * L[1]
                if X < lo[0] - 1e-12 or X > hi[0] + 1e-12 or Y < lo[1] - 1e-12 or Y > hi[1] + 1e-12:
                    continue
                (x1, y1), (x2, y2), (x3, y3) = p[0, :2], p[1, :2], p[2, :2]
                det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if abs(det) < 1e-18:
                    continue
                l2 = ((X - x1) * (y3 - y1) - (x3 - x1) * (Y - y1)) / det
                l3 = ((x2 - x1) * (Y - y1) - (X - x1) * (y2 - y1)) / det
                l1 = 1 - l2 - l3
                if min(l1, l2, l3) < -1e-12:
                    continue
                if min(l1, l2, l3) <= ETA_B:
                    generic = False
                nz = det / 2
                area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
                if abs(nz) / area <= ETA_N:
                    generic = False
                z = l1 * p[0, 2] + l2 * p[1, 2] + l3 * p[2, 2]
                out.append((z, m.sigma * (1 if nz > 0 else -1), t))
    # dedupe hits of shared edges (non-generic anyway)
    return out, generic


def ray_candidates(x0, y0, mode, member=None, i_max=32):
    if mode == 'literal' or member is None:  # F1 applies to every non-literal mode
        for i in range(i_max):
            yield (x0 + (i * PHI1 % 1.0) * L[0]) % L[0], (y0 + (i * PHI2 % 1.0) * L[1]) % L[1]
    else:  # F1: stay inside the member's own triangles (deterministic walk over its triangles)
        for i in range(i_max):
            t = (i * 7919) % len(member.T)
            p = member.V[member.T[t]]
            w = np.array([1 + (i * PHI1 % 1.0), 1 + (i * PHI2 % 1.0), 1.0])
            w /= w.sum()
            q = w @ p
            yield q[0] % L[0], q[1] % L[1]


def all_crossings(members, x, y, placed=None):
    recs, gen = [], True
    for mi, m in enumerate(members):
        if placed is not None and mi not in placed:
            continue
        cr, g = crossings(m, x, y)
        gen = gen and g
        for z, s, t in cr:
            recs.append([z, s, mi])
    zs = sorted(r[0] % L[2] for r in recs)
    for a, b in zip(zs, zs[1:]):
        if b - a < ETA_Z * L[2]:
            gen = False
    return recs, gen


def alternates(recs):
    if not recs:
        return True
    seq = sorted(recs, key=lambda r: r[0] % L[2])
    if len(seq) % 2:
        return False
    return all(seq[i][1] != seq[(i + 1) % len(seq)][1] for i in range(len(seq)))


def classify(members, mode='literal', verbose=False):
    rep = {}
    for m in members:
        m.shift = 0.0
    phis = [integrals(m)[0] for m in members]
    Phi = np.sum(phis, axis=0)
    tol = 1e-9
    if np.any(np.abs(Phi) > tol):
        return dict(status='UNDEFINED', why='Phi != 0', Phi=Phi)
    layers = [i for i, p in enumerate(phis) if abs(p[2]) > tol]
    for i in layers:
        if abs(abs(phis[i][2]) - A) > tol:
            return dict(status='UNSUPPORTED', why='|Phi_k| != A')
    compact = [i for i in range(len(members)) if i not in layers]
    Vraw = sum(integrals(m)[1] for m in members)
    # canonical ray
    x0, y0 = members[0].V[0, 0], members[0].V[0, 1]
    canon = None
    for (x, y) in ray_candidates(x0, y0, 'literal'):
        recs, gen = all_crossings(members, x, y)
        if gen:
            canon = (x, y, recs)
            break
    if canon is None:
        return dict(status='UNSUPPORTED', why='no generic canonical ray')
    x, y, recs = canon
    if not alternates(recs):
        return dict(status='UNDEFINED', why='alternation fails on canonical ray')
    hit = set(r[2] for r in recs)
    # layer placement
    ent = [r[0] % L[2] for r in recs if r[1] < 0]
    if layers:
        if mode == 'literal':
            W0 = min(ent)
        else:  # F2: smallest entering crossing NOT strictly inside any layer's crossing span
            def span(li):
                pts = sorted(r[0] % L[2] for r in recs if r[2] == li)
                if len(pts) == 1:
                    return pts[0], 0.0
                gaps = [((pts[(i + 1) % len(pts)] - pts[i]) % L[2], i) for i in range(len(pts))]
                g, i = max(gaps)
                start = pts[(i + 1) % len(pts)]
                return start, L[2] - g
            spans = [span(li) for li in layers]
            def inside(e):
                return any(0 < (e - st) % L[2] < ln for st, ln in spans)
            cand = sorted(e for e in ent if not inside(e))
            if not cand:
                return dict(status='UNSUPPORTED', why='no admissible window start')
            W0 = cand[0]
        for li in layers:
            zs = [r[0] for r in recs if r[2] == li]
            ok = False
            for p in range(-5, 6):
                if all(W0 - 1e-15 <= z + p * L[2] < W0 + L[2] for z in zs):
                    members[li].shift = p * L[2]
                    ok = True
                    break
            if not ok:
                return dict(status='NO-SHIFT', why=f'layer {members[li].name}: no integer shift puts all crossings in [W0,W0+L), W0={W0:.4f}')
    # compact placement / consistency
    exterior_levels = []
    order = compact if layers else sorted(compact, key=lambda i: -abs(integrals(members[i])[1]))
    placed = set(layers)
    for ci in order:
        c = members[ci]
        ref = c.V[0]
        found = None
        for (x, y) in ray_candidates(ref[0], ref[1], mode, member=c):
            recs_c, gen = all_crossings(members, x, y, placed=placed | {ci})
            if gen:
                found = (x, y, recs_c)
                break
        if found is None:
            return dict(status='UNSUPPORTED', why=f'no generic local ray for {c.name}')
        x, y, recs_c = found
        own = [r for r in recs_c if r[2] == ci]
        if not own:
            if ci not in hit:
                return dict(status='UNSUPPORTED', why=f'{c.name} appears on no generic ray')
        if not alternates(recs_c):
            return dict(status='UNDEFINED', why=f'alternation fails on local ray of {c.name}')
        others = [r for r in recs_c if r[2] != ci]
        if layers:
            placed_z = [r[0] + members[r[2]].shift for r in others if r[2] in layers]
            cmin = min(placed_z)
            if mode == 'literal':
                zref = ref[2]
            else:  # F1: use the member's own crossing on the ray
                zref = own[0][0]
            p = math.floor((zref - cmin) / L[2])
            c.shift = -p * L[2]
        else:
            if mode == 'fixed' and not others and own:  # F3 only in 'fixed'
                # F3: exterior level of an outermost member: sign of the first crossing = entering(-)
                # means exterior is non-phase
                exterior_levels.append(min(own, key=lambda r: r[0])[1])
            # nesting placement (only needed for centre): place inside the interval of placed crossings
            if others and own:
                zc = own[0][0]
                pz = sorted(r[0] + members[r[2]].shift for r in others)
                # shift so zc lies above the lowest placed crossing within one period
                p = math.floor((zc - pz[0]) / L[2])
                c.shift = -p * L[2]
        placed.add(ci)
    if mode == 'fixed' and len(set(exterior_levels)) > 1:
        return dict(status='UNDEFINED', why='outermost compact members disagree on the exterior (inconsistent orientation)')
    # volume
    if layers or mode == 'fixed':  # F3
        V = Vraw % VC
    else:
        V = Vraw
        if not (0 < V < VC):
            return dict(status='UNDEFINED', why=f'V_raw={Vraw:.6g} outside (0,V_cell)')
    Cz = sum(integrals(m)[2] for m in members) / (2 * V)
    return dict(status='DEFINED', V=V, Cz=Cz, Vraw=Vraw)


def report(title, res, expect=None):
    s = res['status']
    extra = ''
    if s == 'DEFINED':
        extra = f"V={res['V']:.12f} Cz={res['Cz']:.12f}"
    else:
        extra = res.get('why', '')
    flag = ''
    if expect is not None:
        flag = '  <-- ' + expect
    print(f"  {title:62s} {s:11s} {extra}{flag}")

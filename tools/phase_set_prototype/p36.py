"""Executable model of plans/16 §3.6, REVISION 3.2 (phase-boundary sets).

Scope of the model: unit cell L = (1,1,1), lateral axes x, y periodic, admissible axis k = z,
which is either periodic (dom['per_z'] = True) or walled at z = 0 / z = 1 with optional wall
closures (dom['closures'] subset of {'zmin','zmax'}). Members are connected, oriented, embedded
triangulated surfaces given in their own continuous gauge (numbers in comments refer to the
steps of §3.6 revision 3.2):

  R1  closure of open members by wall patches (multiplicity mu = eps_W*wind + n_W), flux choice
  R2  Phi(S) = 0 on every axis                                   -> else UNDEFINED
  R3  admissible axis                                            -> else UNSUPPORTED
  R4  component classes: layer |Phi_k| = A_k, compact Phi_k = 0  -> else UNSUPPORTED
  R5  lines: generic, member lines through interior points of the member's own triangles;
      point-query lines inside the clearance ball of the query point
  R6  layer consistency: all crossings of one generic line alternate (cyclic / linear)
  R7  compact tree: enclosers by crossing parity, parent, depth, surrounding level;
      orientation rule e_m = +1 <=> surrounding level 0; exterior level l_E common
  R8  volume: V_raw mod V_cell (layers), V_raw + l_E V_cell (no layers), V_raw (walled)
  R9  placement: layers from the sheet index a_Y(z0) at a level-0 point z0; compacts relative to
      their parent image / adjacent layer lift; centre in the placed gauge
  R10 point query: level at the query point from the nearest crossing below

Exact per-triangle integrals: int_T z n_z dA = A_xy * mean(z);
int_T z^2 n_z dA = A_xy * (sum z_i^2 + sum_{i<j} z_i z_j) / 6.
"""
import math
import numpy as np

L = np.array([1.0, 1.0, 1.0])
A = L[0] * L[1]
VC = A * L[2]
PHI1 = (math.sqrt(5) - 1) / 2
PHI2 = math.sqrt(2) - 1
ETA_B, ETA_N = 1e-7, 1e-7
TAU_PHI = 1e-9
TAU_V = 1e-10
ETA_Z = 0.1 * TAU_V / (A * L[2])  # crossing separation tied to the volume tolerance (R5)
N_TRY = 64
BARY = [(1 / 3, 1 / 3, 1 / 3)] + [(1 + (i * PHI1) % 1.0, 1 + (i * PHI2) % 1.0, 1.0) for i in range(1, 8)]


class Member:
    def __init__(self, V, T, sigma, name, wrap=(0, 0, 0), closure=None):
        self.V = np.asarray(V, float)
        self.T = np.asarray(T, int)
        self.sigma = sigma
        self.name = name
        self.wrap = tuple(wrap)  # cached WRAP flags (16B canonicalisation); input to R3
        self.closure = closure  # wall on which this member's boundary loops lie ('zmin'/'zmax')


class Res(dict):
    pass


def fail(status, why, **kw):
    r = Res(status=status, why=why)
    r.update(kw)
    return r


# ------------------------------------------------------------------------------------------
# geometry helpers
# ------------------------------------------------------------------------------------------
def tri_data(m):
    P = m.V[m.T]
    n = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area = np.linalg.norm(n, axis=1)
    return P, n, area


def own_integrals(m, shift=0.0):
    """unsigned-by-sigma (stored orientation): area vector, int z n_z, int z^2 n_z"""
    P, n, _ = tri_data(m)
    z = P[:, :, 2] + shift
    zs = z.sum(1)
    z2 = (z ** 2).sum(1) + z[:, 0] * z[:, 1] + z[:, 1] * z[:, 2] + z[:, 0] * z[:, 2]
    return n.sum(0), (n[:, 2] * zs / 3).sum(), (n[:, 2] * z2 / 6).sum()


def line_hits(m, x0, y0):
    """crossings of the vertical line (x0,y0) with member triangles, all lateral images.
    returns list of (z_own, sgeom, generic). Each triangle is first moved laterally (by whole
    periods) so that its first vertex lies within half a period of the line (per-triangle
    minimum image), so any member gauge works; the images -1..1 then cover every edge that is
    shorter than half a period (§3.5)."""
    P0, n, area = tri_data(m)
    out = []
    sh = np.zeros((len(P0), 3))
    sh[:, 0] = -np.round(P0[:, 0, 0] - x0)
    sh[:, 1] = -np.round(P0[:, 0, 1] - y0)
    P = P0 + sh[:, None, :]
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            X, Y = x0 + a * L[0], y0 + b * L[1]
            x1, y1 = P[:, 0, 0], P[:, 0, 1]
            x2, y2 = P[:, 1, 0], P[:, 1, 1]
            x3, y3 = P[:, 2, 0], P[:, 2, 1]
            det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            ok = np.abs(det) > 1e-18
            with np.errstate(divide='ignore', invalid='ignore'):
                l2 = ((X - x1) * (y3 - y1) - (x3 - x1) * (Y - y1)) / det
                l3 = ((x2 - x1) * (Y - y1) - (X - x1) * (y2 - y1)) / det
            l1 = 1 - l2 - l3
            lm = np.minimum(np.minimum(l1, l2), l3)
            hit = ok & (lm >= -1e-12)
            for t in np.nonzero(hit)[0]:
                gen = lm[t] > ETA_B and abs(n[t, 2]) / area[t] > ETA_N
                z = l1[t] * P[t, 0, 2] + l2[t] * P[t, 1, 2] + l3[t] * P[t, 2, 2]
                out.append((z, 1 if det[t] > 0 else -1, gen))
            # vertical triangles (det == 0) that contain the line: non-generic
            vert = (~ok)
            if vert.any():
                for t in np.nonzero(vert)[0]:
                    p = P[t]
                    if min(p[:, 0]) - 1e-12 <= X <= max(p[:, 0]) + 1e-12 and min(p[:, 1]) - 1e-12 <= Y <= max(p[:, 1]) + 1e-12:
                        d = np.cross(np.append(p[1, :2] - p[0, :2], 0), np.append([X, Y] - p[0, :2], 0))[2]
                        if abs(d) < 1e-12:
                            out.append((p[0, 2], 0, False))
    return out


# ------------------------------------------------------------------------------------------
# R1: boundary loops and wall patches
# ------------------------------------------------------------------------------------------
def boundary_loops(m, dom):
    def red(a, P):
        r = round(a % P, 8)
        return 0.0 if r >= P else r

    def key(v):
        x, y, z = v
        kz = red(z, L[2]) if dom['per_z'] else round(z, 8)
        return (red(x, L[0]), red(y, L[1]), kz)
    keys = [key(v) for v in m.V]
    ukey = {}
    kid = [ukey.setdefault(k, len(ukey)) for k in keys]
    cnt = {}
    dedges = []
    for t in m.T:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            ka, kb = kid[a], kid[b]
            cnt[frozenset((ka, kb))] = cnt.get(frozenset((ka, kb)), 0) + 1
            dedges.append((ka, kb, a, b))
    bnd = [e for e in dedges if cnt[frozenset((e[0], e[1]))] == 1]
    if not bnd:
        return []
    nxt = {}
    for e in bnd:
        if e[0] in nxt:
            raise ValueError('non-manifold boundary')
        nxt[e[0]] = e
    loops, used = [], set()
    for e in bnd:
        if e[0] in used:
            continue
        pts, disp, cur = [], np.zeros(3), e
        start = e[0]
        pos = m.V[e[2]].copy()
        while True:
            used.add(cur[0])
            pts.append(pos.copy())
            dv = m.V[cur[3]] - m.V[cur[2]]
            pos = pos + dv
            disp += dv
            if cur[1] == start:
                break
            cur = nxt[cur[1]]
        winding = np.round(disp / L).astype(int)
        loops.append(dict(pts=np.array(pts), winding=winding))
    return loops


def wind_xy(poly, qx, qy):
    """winding number of the closed lateral polygon (continuous coords) around (qx,qy), summed
    over lateral images (the patch multiplicity on the periodic wall plane)."""
    tot = 0.0
    c = poly[:, :2].mean(0)  # lateral minimum image of the (closed, non-winding) loop w.r.t. q
    q0 = np.array([qx, qy]) + np.round(c - np.array([qx, qy])) * L[:2]
    for a in (-2, -1, 0, 1, 2):
        for b in (-2, -1, 0, 1, 2):
            d = poly[:, :2] - np.array([q0[0] + a * L[0], q0[1] + b * L[1]])
            ang = np.arctan2(d[:, 1], d[:, 0])
            da = np.diff(np.append(ang, ang[0]))
            da = (da + np.pi) % (2 * np.pi) - np.pi
            tot += da.sum()
    return int(round(tot / (2 * np.pi)))


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def dist_to_polyline(poly, qx, qy):
    best = 1e300
    c = poly[:, :2].mean(0)
    q0 = np.array([qx, qy]) + np.round(c - np.array([qx, qy])) * L[:2]
    for a in (-2, -1, 0, 1, 2):
        for b in (-2, -1, 0, 1, 2):
            p = poly[:, :2]
            q = np.array([q0[0] + a * L[0], q0[1] + b * L[1]])
            p2 = np.roll(p, -1, axis=0)
            d = p2 - p
            t = np.clip(np.einsum('ij,ij->i', q - p, d) / np.maximum(np.einsum('ij,ij->i', d, d), 1e-300), 0, 1)
            c = p + t[:, None] * d
            best = min(best, np.min(np.linalg.norm(c - q, axis=1)))
    return best


WALL = {'zmin': dict(z=0.0, s_out=-1, eps=+1), 'zmax': dict(z=1.0, s_out=+1, eps=-1)}


# ------------------------------------------------------------------------------------------
# components: members (+ their patch parts) and full-wall layers
# ------------------------------------------------------------------------------------------
class Comp:
    """a connected closed component of S-hat: members glued, along their boundary loops, to the
    wall patches of the walls those loops lie on (one patch piece per wall in this model), or a
    wall patch alone (a full wall, n_W = 1, no loops)"""

    def __init__(self, name, members=(), pieces=()):
        self.name = name
        self.members = list(members)
        # connected pieces of {mu_W = 1} glued to this component: (W, contains(qx,qy), area)
        self.pieces = list(pieces)
        self.walls = [p[0] for p in self.pieces]
        self.shift = 0.0

    def flux(self):
        F = np.zeros(3)
        for m in self.members:
            F += m.sigma * own_integrals(m)[0]
        for (W, _, ar) in self.pieces:
            F[2] += WALL[W]['s_out'] * ar
        return F

    def vol_terms(self, shift=0.0):
        """sigma-weighted int Y n_z dA and int Y^2 n_z dA (reference z = 0), members shifted"""
        v = c2 = 0.0
        for m in self.members:
            _, a, b = own_integrals(m, shift)
            v += m.sigma * a
            c2 += m.sigma * b
        for (W, _, pa) in self.pieces:
            ar = WALL[W]['s_out'] * pa
            v += WALL[W]['z'] * ar
            c2 += WALL[W]['z'] ** 2 * ar
        return v, c2

    def crossings(self, qx, qy, dom):
        """list of (z_own, s, generic): s = -1 entering the phase along +z"""
        out = []
        for m in self.members:
            for z, sg, gen in line_hits(m, qx, qy):
                out.append((z, m.sigma * sg, gen))
        for (W, contains, _) in self.pieces:
            gen = self.loop_clear(W, qx, qy)
            if contains(qx, qy):
                out.append((WALL[W]['z'], -1 if W == 'zmin' else 1, gen))
            elif not gen:
                out.append((WALL[W]['z'], 0, False))
        return out

    def loop_clear(self, W, qx, qy):
        polys = [p for (w, p) in self.loops if w == W]
        return (not polys) or min(dist_to_polyline(p, qx, qy) for p in polys) > 1e-7


def all_crossings(comps, qx, qy, dom):
    recs, gen = [], True
    for ci, c in enumerate(comps):
        for z, s, g in c.crossings(qx, qy, dom):
            recs.append([z, s, ci])
            gen = gen and g and s != 0
    red = sorted((r[0] % L[2]) if dom['per_z'] else r[0] for r in recs)
    for a, b in zip(red, red[1:]):
        if b - a < ETA_Z * L[2]:
            gen = False
    if dom['per_z'] and len(red) > 1 and (red[0] + L[2] - red[-1]) < ETA_Z * L[2]:
        gen = False
    return recs, gen


def alternation_ok(recs, dom):
    """R6: cyclic strict alternation (periodic k) / linear levels in {0,1} from level 0 (walled)"""
    if dom['per_z']:
        seq = sorted(recs, key=lambda r: r[0] % L[2])
        if len(seq) % 2:
            return False
        return all(seq[i][1] != seq[(i + 1) % len(seq)][1] for i in range(len(seq)))
    lev = 0
    for r in sorted(recs, key=lambda r: r[0]):
        lev += -r[1]
        if lev not in (0, 1):
            return False
    return lev == 0


def level_below(recs, z, dom, only=None):
    """level just above the nearest crossing strictly below height z (own gauges of each
    crossing, all periodic images for periodic k). Returns (level, rec, q) where the crossing
    lift is rec.z + q*L; (0, None, 0) if there is none (walled: bottom of the domain)."""
    best, br, bq = None, None, 0
    for r in recs:
        if only is not None and r[2] not in only:
            continue
        if dom['per_z']:
            d = (z - r[0]) % L[2]
            q = int(round((z - d - r[0]) / L[2]))
        else:
            d = z - r[0]
            q = 0
            if d <= 0:
                continue
        if best is None or d < best:
            best, br, bq = d, r, q
    if br is None:
        return 0, None, 0
    return (1 if br[1] < 0 else 0), br, bq


# ------------------------------------------------------------------------------------------
# the classifier
# ------------------------------------------------------------------------------------------
def split_components(members, dom):
    """§3.6.1: members are connected components; a body with several components contributes each
    component separately with the body's sigma (review erratum E8: the model used to keep a
    disconnected body as one member). Connectivity through vertices identified modulo the periods."""
    out = []
    for m in members:
        def key(v):
            r = [round(v[0] % 1.0, 8) % 1.0, round(v[1] % 1.0, 8) % 1.0, (round(v[2] % 1.0, 8) % 1.0) if dom['per_z'] else round(v[2], 8)]
            return tuple(r)
        kid = {}
        vk = [kid.setdefault(key(v), len(kid)) for v in m.V]
        par = list(range(len(kid)))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for t in m.T:
            a, b, c = find(vk[t[0]]), find(vk[t[1]]), find(vk[t[2]])
            for y in (b, c):
                ra, ry = find(a), find(y)
                if ra != ry:
                    par[max(ra, ry)] = min(ra, ry)
        groups = {}
        for ti, t in enumerate(m.T):
            groups.setdefault(find(vk[t[0]]), []).append(ti)
        if len(groups) == 1:
            out.append(m)
            continue
        for gi, tris in enumerate(sorted(groups.values(), key=lambda g: g[0])):
            T = m.T[tris]
            used = sorted(set(T.flatten()))
            remap = {v: i for i, v in enumerate(used)}
            mm = Member(m.V[used], [[remap[a] for a in t] for t in T], m.sigma, f'{m.name}#{gi}', wrap=m.wrap, closure=m.closure)
            out.append(mm)
    return out


def classify(members, dom=None, want_placement=True):
    dom = dom or dict(per_z=True)
    dom.setdefault('closures', [])
    per = dom['per_z']
    members = split_components(members, dom)
    # ---- R1: boundary loops, closure by wall patches -------------------------------------
    loops_of = {}
    for mi, m in enumerate(members):
        loops = boundary_loops(m, dom)
        if loops:
            if m.closure is None or m.closure not in dom['closures'] or per:
                return fail('UNDEFINED', f'{m.name}: open member whose boundary is not on a declared wall closure')
            if any(np.any(l['winding'] != 0) for l in loops):
                return fail('UNSUPPORTED', f'{m.name}: boundary loop winds along the wall plane')
            zw = WALL[m.closure]['z']
            if any(np.max(np.abs(l['pts'][:, 2] - zw)) > 1e-9 for l in loops):
                return fail('UNDEFINED', f'{m.name}: boundary loop not on its wall closure')
            # Gamma = sigma * boundary(m), as lateral polygons
            loops_of[mi] = [(m.closure, l['pts'] if m.sigma > 0 else l['pts'][::-1]) for l in loops]
    walls = list(dom['closures'])
    gam = {W: [p for mi in loops_of for (w, p) in loops_of[mi] if w == W] for W in walls}
    wind = {W: (lambda qx, qy, W=W: WALL[W]['eps'] * sum(wind_xy(p, qx, qy) for p in gam[W])) for W in walls}
    cand = {}
    for W in walls:
        vals = set()
        for i in range(48):
            for j in range(48):
                vals.add(wind[W]((i + 0.5 + 0.1 * PHI1) / 48, (j + 0.5 + 0.1 * PHI2) / 48))
        cand[W] = [n for n in (0, 1) if all(v + n in (0, 1) for v in vals)]
        if not cand[W]:
            return fail('UNDEFINED', f'wall {W}: loop winding numbers outside a single {{0,1}} pair (inconsistent loop orientation)')
    wind_area = {W: WALL[W]['eps'] * sum(shoelace(p) for p in gam[W]) for W in walls}
    Fm = sum((m.sigma * own_integrals(m)[0] for m in members), np.zeros(3))
    sols = []
    import itertools
    for ns in itertools.product(*[cand[W] for W in walls]):
        F = Fm.copy()
        for W, n in zip(walls, ns):
            F[2] += WALL[W]['s_out'] * (wind_area[W] + n * A)
        if np.all(np.abs(F) < TAU_PHI):
            sols.append(ns)
    # ---- R2: Phi(S) = 0 -------------------------------------------------------------------
    if not sols:
        return fail('UNDEFINED', 'Phi(S) != 0 for every admissible wall patch', Phi=Fm)
    if len(sols) > 1:
        return fail('UNSUPPORTED', 'ambiguous wall closure (more than one patch choice closes S)')
    nW = dict(zip(walls, sols[0]))
    # components (§3.6.2 step 7): members glued, along their loops, to the CONNECTED pieces of
    # {mu_W = 1} those loops bound. The loops of one wall are disjoint simple closed curves
    # (non-winding); their nesting tree gives the regions: the outer region (outside every
    # top-level loop; connected on the doubly periodic wall) and, per loop b, "inside b minus its
    # children". mu is constant on a region: mu = eps_W * (sum of the orientation signs of the
    # loops containing it) + n_W.
    parent_uf = list(range(len(members)))
    def find(x):
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x
    pieces = []  # (W, member indices, contains, area)
    for W in walls:
        Ls = [(mi, p) for mi in loops_of for (w, p) in loops_of[mi] if w == W]
        n = len(Ls)
        ins = lambda p, qx, qy: wind_xy(p, qx, qy) != 0
        cont = [[j != i and ins(Ls[j][1], Ls[i][1][0, 0], Ls[i][1][0, 1]) for j in range(n)] for i in range(n)]
        dep = [sum(cont[i]) for i in range(n)]
        par = [max((j for j in range(n) if cont[i][j]), key=lambda j: dep[j], default=None) for i in range(n)]
        sgn = [1 if shoelace(Ls[i][1]) > 0 else -1 for i in range(n)]
        def anc(b):
            out = []
            while b is not None:
                out.append(b)
                b = par[b]
            return out
        for b in [None] + list(range(n)):
            kids = [i for i in range(n) if par[i] == b]
            mu = WALL[W]['eps'] * sum(sgn[a] for a in (anc(b) if b is not None else [])) + nW[W]
            if mu != 1:
                continue
            def contains(qx, qy, b=b, kids=kids, Ls=Ls, ins=ins):
                if b is not None and not ins(Ls[b][1], qx, qy):
                    return False
                return not any(ins(Ls[c][1], qx, qy) for c in kids)
            area = (A if b is None else abs(shoelace(Ls[b][1]))) - sum(abs(shoelace(Ls[c][1])) for c in kids)
            mis = sorted({Ls[i][0] for i in ([b] if b is not None else []) + kids})
            pieces.append((W, mis, contains, area))
            for a_ in mis[1:]:
                parent_uf[find(a_)] = find(mis[0])
    groups = {}
    for mi in range(len(members)):
        groups.setdefault(find(mi), []).append(mi)
    comps = []
    for g in groups.values():
        pcs = [(W, cf, ar) for (W, mis, cf, ar) in pieces if mis and find(mis[0]) == find(g[0])]
        c = Comp('+'.join(members[mi].name for mi in g) + ''.join(f'+patch({W})' for (W, _, _) in pcs), [members[mi] for mi in g], pcs)
        c.loops = [lp for mi in g if mi in loops_of for lp in loops_of[mi]]
        comps.append(c)
    for (W, mis, cf, ar) in pieces:
        if not mis:  # a piece without loops: a full wall, a component of its own
            c = Comp(f'wall {W}', [], [(W, cf, ar)])
            c.loops = []
            comps.append(c)
    # ---- R3: admissible axis --------------------------------------------------------------
    if per and any(m.wrap[2] for m in members):
        if all(any(m.wrap[j] for m in members) for j in (0, 1)):
            return fail('UNSUPPORTED', 'no admissible axis (every periodic axis wrapped by some member)')
        return fail('NOT-IN-MODEL', 'admissible axis is x or y (model evaluates k = z only)')
    # ---- R4: component classes --------------------------------------------------------------
    layers, compact = [], []
    for i, c in enumerate(comps):
        fz = c.flux()[2]
        if abs(fz) < TAU_PHI:
            compact.append(i)
        elif abs(abs(fz) - A) < TAU_PHI:
            layers.append(i)
        else:
            return fail('UNSUPPORTED', f'{c.name}: |Phi_k| = {abs(fz):.6g} not in {{0, A_k}}')
    vown = {}
    for i in compact:
        v, _ = comps[i].vol_terms()
        if abs(v) <= TAU_V:
            return fail('UNSUPPORTED', f'{comps[i].name}: degenerate compact component (zero enclosed volume)')
        vown[i] = v  # sigma-weighted own volume
    e = {i: (1 if vown[i] > 0 else -1) for i in compact}  # e = sigma * o
    # ---- R8a: degenerate volume (before any line is cast) --------------------------------------
    Vraw = sum(c.vol_terms()[0] for c in comps)
    Vrep = (Vraw % VC) if per else Vraw
    if Vrep <= TAU_V or Vrep >= VC - TAU_V:
        return fail('UNDEFINED', f'degenerate volume (V_raw representative {Vrep:.3g}: empty or full region)')
    # ---- R5/R6: canonical line, layer consistency -----------------------------------------
    canon = None
    if layers:
        m0 = members[0]
        x0, y0 = m0.V[0, 0], m0.V[0, 1]
        for i in range(N_TRY):
            qx, qy = (x0 + (i * PHI1 % 1.0)) % 1.0, (y0 + (i * PHI2 % 1.0)) % 1.0
            recs, gen = all_crossings(comps, qx, qy, dom)
            if gen:
                canon = (qx, qy, recs)
                break
        if canon is None:
            return fail('UNSUPPORTED', 'no generic canonical line')
        if not alternation_ok(canon[2], dom):
            return fail('UNDEFINED', 'crossings of the canonical line do not alternate (inconsistent orientation)')
    # ---- R7: compact tree -----------------------------------------------------------------
    info = {}
    for i in compact:
        c = comps[i]
        m = c.members[0]
        P, n, area = tri_data(m)
        found = None
        tries = 0
        for t in range(len(m.T)):
            if abs(n[t, 2]) / area[t] <= ETA_N:
                continue
            for w in BARY:
                w = np.array(w) / sum(w)
                p = w @ P[t]
                recs, gen = all_crossings(comps, p[0] % 1.0, p[1] % 1.0, dom)
                tries += 1
                if gen:
                    found = (p, recs)
                    break
                if tries >= N_TRY:
                    break
            if found or tries >= N_TRY:
                break
        if found is None:
            return fail('UNSUPPORTED', f'{c.name}: no generic line through the member')
        p, recs = found
        zP = p[2]
        enc = []
        for j in compact:
            if j == i:
                continue
            xs = [r[0] for r in recs if r[2] == j]
            if not xs:
                continue
            prange = range(int(math.floor((zP - max(xs)) / L[2])) - 1, int(math.ceil((zP - min(xs)) / L[2])) + 2) if per else [0]
            for q in prange:
                if sum(1 for x in xs if x + q * L[2] < zP) % 2 == 1:
                    enc.append((j, q))
        lay = level_below(recs, zP, dom, only=set(layers)) if layers else (None, None, 0)
        info[i] = dict(enc=enc, zP=zP, lay=lay, tries=tries)
    depth = {i: len(info[i]['enc']) for i in compact}
    parent = {}
    for i in compact:
        par = [(j, q) for (j, q) in info[i]['enc'] if depth[j] == depth[i] - 1]
        if depth[i] > 0 and len(par) != 1:
            return fail('UNSUPPORTED', f'{comps[i].name}: nesting not resolvable')
        parent[i] = par[0] if par else None
    # surrounding level, exterior level
    lE = None
    if not layers:
        outer = [i for i in compact if parent[i] is None]
        if not outer:
            return fail('UNDEFINED', 'empty phase-boundary set')
        vals = {0 if e[i] > 0 else 1 for i in outer}
        if len(vals) > 1:
            return fail('UNDEFINED', 'outermost compact components disagree on the exterior level (inconsistent orientation)')
        lE = vals.pop()
        if lE == 1 and not per:
            return fail('UNDEFINED', 'exterior region would be phase but touches an undeclared domain wall')
    for i in sorted(compact, key=lambda i: depth[i]):
        if parent[i] is not None:
            j = parent[i][0]
            lout = 1 if e[j] > 0 else 0
        elif layers:
            lout = info[i]['lay'][0]
        else:
            lout = lE
        if (e[i] > 0) != (lout == 0):
            return fail('UNDEFINED', f'{comps[i].name}: orientation inconsistent with its surroundings (level {lout} outside)')
    # ---- R8: volume -------------------------------------------------------------------------
    Vraw = sum(c.vol_terms()[0] for c in comps)
    if per and layers:
        V = Vraw % VC
    elif per:
        V = Vraw + lE * VC
    else:
        V = Vraw
    Vmax = VC
    if V <= TAU_V or V >= Vmax - TAU_V:
        return fail('UNDEFINED', f'degenerate volume V = {V:.3g} (empty or full region)')
    res = Res(status='DEFINED', V=V, Vraw=Vraw, lE=lE, n_layers=len(layers))
    res['member_line_tries'] = {comps[i].name: info[i]['tries'] for i in compact}
    # phase components
    ncomp = sum(1 for i in compact if e[i] > 0) + (len(layers) // 2) + (1 if lE == 1 else 0)
    res['n_phase_components'] = ncomp
    # ---- R9: placement and centre -------------------------------------------------------------
    if not want_placement:
        return res
    if per and layers:
        lrecs = [r for r in canon[2] if r[2] in layers]
        red = sorted(lrecs, key=lambda r: r[0] % L[2])
        z0 = None
        for a, r in enumerate(red):
            nx = red[(a + 1) % len(red)]
            if r[1] > 0:  # leaving crossing -> level-0 segment above it
                gap = (nx[0] - r[0]) % L[2]
                z0 = (r[0] % L[2]) + 0.5 * gap
                break
        for li in layers:
            ys = [r[0] for r in canon[2] if r[2] == li]
            best = None
            for q in range(int(math.floor((z0 - max(ys)) / L[2])) - 2, int(math.ceil((z0 - min(ys)) / L[2])) + 3):
                if sum(1 for y in ys if y + q * L[2] < z0) % 2 == 1:
                    best = q if best is None else max(best, q)
            comps[li].shift = best * L[2]
    for i in sorted(compact, key=lambda i: depth[i]):
        if not per:
            comps[i].shift = 0.0
        elif parent[i] is not None:
            j, q = parent[i]
            comps[i].shift = comps[j].shift - q * L[2]
        elif layers:
            _, r, q = info[i]['lay']
            comps[i].shift = comps[r[2]].shift - q * L[2]
        else:
            comps[i].shift = 0.0
    Vp = sum(c.vol_terms(c.shift)[0] for c in comps)
    C2 = sum(c.vol_terms(c.shift)[1] for c in comps)
    if lE == 1:
        res['centre'] = 'UNDEFINED (phase region wraps every periodic axis)'
    elif ncomp != 1 and per:
        res['centre'] = 'UNSUPPORTED (disconnected phase region, periodic axis)'
    else:
        if abs(Vp - V) > 1e-9:
            return fail('MODEL-ERROR', f'placed volume {Vp} != V {V}')
        res['Cz'] = C2 / (2 * V)
        res['centre'] = 'DEFINED'
    res['placed'] = {c.name: c.shift for c in comps}
    res['_comps'] = comps
    res['_dom'] = dom
    return res


# ------------------------------------------------------------------------------------------
# R10: point query
# ------------------------------------------------------------------------------------------
def point_tri_dist(P, q):
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    nu = n / np.maximum(nn, 1e-300)[:, None]
    dpl = np.einsum('ij,ij->i', q - a, nu)
    proj = q - dpl[:, None] * nu

    def inside(p):
        c1 = np.einsum('ij,ij->i', np.cross(b - a, p - a), n)
        c2 = np.einsum('ij,ij->i', np.cross(c - b, p - b), n)
        c3 = np.einsum('ij,ij->i', np.cross(a - c, p - c), n)
        return (c1 >= 0) & (c2 >= 0) & (c3 >= 0)

    def seg(p0, p1):
        d = p1 - p0
        t = np.clip(np.einsum('ij,ij->i', q - p0, d) / np.maximum(np.einsum('ij,ij->i', d, d), 1e-300), 0, 1)
        return np.linalg.norm(p0 + t[:, None] * d - q, axis=1)

    de = np.minimum(np.minimum(seg(a, b), seg(b, c)), seg(c, a))
    return np.where(inside(proj), np.abs(dpl), de)


def inside(res, Q):
    """R10: returns (status, inside?)"""
    if res['status'] != 'DEFINED':
        return res['status'], None
    comps, dom = res['_comps'], res['_dom']
    Q = np.asarray(Q, float)
    d = 1e300
    for c in comps:
        for mm in c.members:
            P = mm.V[mm.T].copy()
            # per-triangle minimum image relative to Q (every periodic axis), then the 27 images
            per = np.array([1.0, 1.0, 1.0 if dom['per_z'] else 0.0])
            P -= (np.round(P[:, 0, :] - Q) * per)[:, None, :]
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for cz in ((-1, 0, 1) if dom['per_z'] else (0,)):
                        d = min(d, float(np.min(point_tri_dist(P + np.array([a, b, cz]) * L, Q))))
        for W in c.walls:
            d = min(d, abs(Q[2] - WALL[W]['z']))
    if d < 1e-9:
        return 'DEFINED', 'ON_BOUNDARY'
    for i in range(N_TRY):
        r = 0.0 if i == 0 else 0.5 * d * ((i * PHI1) % 1.0)
        th = 2 * math.pi * ((i * PHI2) % 1.0)
        qx, qy = (Q[0] + r * math.cos(th)) % 1.0, (Q[1] + r * math.sin(th)) % 1.0
        recs, gen = all_crossings(comps, qx, qy, dom)
        if gen:
            lev, r, _ = level_below(recs, Q[2], dom)
            if r is None and dom['per_z']:
                lev = res['lE']  # no crossing on the whole line: exterior level (no layers)
            return 'DEFINED', bool(lev)
    return 'UNSUPPORTED', None

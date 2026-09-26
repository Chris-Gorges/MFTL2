"""Validation cases for plans/16 §3.6 revision 3.2 (tests B11/B14/B15/B16 of §10), run on the
executable model p36.py. Each case states the expected status; DEFINED values are compared with
the value of the same mesh in an explicitly given physical placement ("truth"), which is
independent of the model's own placement rules. Run: python3 cases.py (numpy only).
The superseded revision 3.1 and its review record: p31.py / cases31.py."""
import math
import sys
import numpy as np
from p36 import Member, classify, inside, own_integrals, line_hits, L, A

NPASS = NFAIL = 0


# ------------------------------------------------------------------------------------------
# meshes (continuous coordinates, outward = stored orientation)
# ------------------------------------------------------------------------------------------
def _orient_out(V, T, inside_pt=None):
    m = Member(V, T, 1, 'tmp')
    if own_integrals(m)[1] < 0:
        T = [[a, c, b] for a, b, c in T]
    return T


def sphere(c, r, n=24, sigma=1, name='sphere', cap=False):
    """closed sphere, or (cap=True) the upper hemisphere z >= c_z with its equator as boundary"""
    V, T = [[c[0], c[1], c[2] + r]], []
    m = 2 * n
    rings = n // 2 if cap else n - 1
    for i in range(1, rings + 1):
        th = math.pi * i / n
        for j in range(m):
            ph = 2 * math.pi * j / m
            V.append([c[0] + r * math.sin(th) * math.cos(ph), c[1] + r * math.sin(th) * math.sin(ph), c[2] + r * math.cos(th)])
    idx = lambda i, j: 1 + (i - 1) * m + (j % m)
    for j in range(m):
        T.append([0, idx(1, j), idx(1, j + 1)])
    for i in range(1, rings):
        for j in range(m):
            a, b, cc, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            T.append([a, d, cc]); T.append([a, cc, b])
    if not cap:
        V.append([c[0], c[1], c[2] - r])
        last = len(V) - 1
        for j in range(m):
            T.append([idx(n - 1, j), last, idx(n - 1, j + 1)])
        T = _orient_out(V, T)
    else:
        # orient outward: the normal at the pole must point +z
        mm = Member(V, T, 1, 't')
        P = mm.V[mm.T[0]]
        nz = np.cross(P[1] - P[0], P[2] - P[0])[2]
        if nz < 0:
            T = [[a, cc, b] for a, b, cc in T]
    return Member(V, T, sigma, name, wrap=(0, 0, 0))


def sheet(f, N=48, up=True, sigma=1, name='sheet', zoff=0.0):
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
    return Member(V, T, sigma, name, wrap=(1, 1, 0))


def curve_sheet(xz, N=400, M=8, up=True, sigma=1, name='fold'):
    """(x(u), y, z(u)), x(u+1) = x(u) + 1: an overhanging layer when x(u) is not monotone"""
    V, T = [], []
    for j in range(M + 1):
        for i in range(N + 1):
            x, z = xz(i / N)
            V.append([x, j / M, z])
    idx = lambda i, j: j * (N + 1) + i
    for j in range(M):
        for i in range(N):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if up:
                T.append([a, b, c]); T.append([a, c, d])
            else:
                T.append([a, c, b]); T.append([a, d, c])
    return Member(V, T, sigma, name, wrap=(1, 1, 0))


def tube(yc=0.5, zc=0.5, r=0.15, N=40, M=48, sigma=1, zoff=0.0, name='x-tube', ref_bottom=False, half=False):
    """closed tube around the x direction (wraps x); half=True: half tube z >= zc standing on a
    wall (boundary loops = two lines along x)"""
    V, T = [], []
    th0 = -math.pi / 2 if ref_bottom else 0.0
    Mc = M // 2 + 1 if half else M
    for i in range(N + 1):
        for j in range(Mc):
            th = (math.pi * j / (Mc - 1)) if half else th0 + 2 * math.pi * j / M
            V.append([i / N, yc + r * math.cos(th), zc + r * math.sin(th) + zoff])
    idx = lambda i, j: i * Mc + (j if half else j % M)
    for i in range(N):
        for j in range(Mc - 1 if half else M):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T.append([a, b, c]); T.append([a, c, d])
    if not half:
        T = _orient_out(V, T)
    else:
        mm = Member(V, T, 1, 't')
        P = mm.V[mm.T[len(T) // 2]]
        if np.cross(P[1] - P[0], P[2] - P[0])[2] < 0:
            T = [[a, c, b] for a, b, c in T]
    return Member(V, T, sigma, name, wrap=(1, 0, 0))


def tilted_plane(c, up=True, sigma=1, N=32, name='tilted'):
    """the plane x + z = c in the triply periodic cell (winds x, y and z)"""
    V, T = [], []
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = i / N, j / N
            V.append([x, y, c - x])
    idx = lambda i, j: j * (N + 1) + i
    for j in range(N):
        for i in range(N):
            a, b, cc, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if up:
                T.append([a, b, cc]); T.append([a, cc, d])
            else:
                T.append([a, cc, b]); T.append([a, d, cc])
    return Member(V, T, sigma, name, wrap=(1, 1, 1))


def merge(ms, name, sigma=1):
    V, T, off = [], [], 0
    for m in ms:
        V += list(m.V)
        T += [list(t + off) for t in m.T]
        off += len(m.V)
    w = tuple(max(m.wrap[k] for m in ms) for k in range(3))
    return Member(V, T, sigma, name, wrap=w)


# ------------------------------------------------------------------------------------------
# harness
# ------------------------------------------------------------------------------------------
def truth(members, shifts, extra_V=0.0, extra_C2=0.0):
    """V and centre_z of the same mesh in an explicitly given physical placement"""
    V = extra_V
    C2 = extra_C2
    for m, s in zip(members, shifts):
        _, a, b = own_integrals(m, s)
        V += m.sigma * a
        C2 += m.sigma * b
    return V, C2 / (2 * V)


def case(tag, title, build, expect, dom=None, truth_shifts=None, truth_fn=None, centre=None, queries=(), tol=1e-10):
    global NPASS, NFAIL
    ms = build()
    res = classify(ms, dict(dom) if dom else None)
    st = res['status']
    ok = (st == expect)
    msg = [f"{st:11s}"]
    if st == 'DEFINED':
        msg.append(f"V={res['V']:.12f}")
        if 'Cz' in res:
            msg.append(f"Cz={res['Cz']:.12f}")
        msg.append(f"centre {res['centre'].split(' ')[0]}")
        tv = None
        if truth_shifts is not None:
            tv = truth(build(), truth_shifts)
        if truth_fn is not None:
            tv = truth_fn(build())
        if tv is not None:
            dv = abs(res['V'] - tv[0])
            ok = ok and dv < tol
            msg.append(f"|dV|={dv:.1e}")
            if centre == 'DEFINED' and 'Cz' in res:
                dc = abs(res['Cz'] - tv[1])
                if dom is None or dom.get('per_z', True):
                    dc = abs(((res['Cz'] - tv[1]) + 0.5) % L[2] - 0.5)  # centre is defined modulo the period
                ok = ok and dc < tol
                msg.append(f"|dCz|={dc:.1e}")
        if centre is not None:
            ok = ok and res['centre'].startswith(centre)
    else:
        msg.append(res['why'])
    for (Q, want) in queries:
        s, v = inside(res, Q)
        good = (v == want)
        ok = ok and good
        msg.append(f"inside{tuple(round(q, 3) for q in Q)}={v}{'' if good else '(!=' + str(want) + ')'}")
    if ok:
        NPASS += 1
    else:
        NFAIL += 1
    print(f"{'PASS' if ok else 'FAIL'} {tag:4s} {title:58s} expect {expect:11s} got " + '  '.join(msg))
    sys.stdout.flush()
    return res


WALLED = dict(per_z=False, closures=[])


def main():
    # ---------------------------------------------------------------- compact components
    case('C1', 'one compact drop', lambda: [sphere([0.4, 0.5, 0.45], 0.2)], 'DEFINED', truth_shifts=[0.0], centre='DEFINED',
         queries=[((0.4, 0.5, 0.45), True), ((0.9, 0.9, 0.9), False)])
    case('C2', 'one compact bubble (sigma=-1): matrix phase, fully periodic', lambda: [sphere([0.4, 0.5, 0.45], 0.2, sigma=-1)], 'DEFINED',
         truth_fn=lambda ms: (1.0 + truth(ms, [0.0])[0], 0.0), centre='UNDEFINED', queries=[((0.4, 0.5, 0.45), False), ((0.9, 0.9, 0.9), True)])
    case('C3', 'seam-crossing sphere (x and z seams, gauge z+0)', lambda: [sphere([0.95, 0.5, 0.97], 0.2)], 'DEFINED', truth_shifts=[0.0],
         centre='DEFINED', queries=[((0.05, 0.5, 0.05), True), ((0.5, 0.5, 0.5), False)])
    case('C4', 'two disjoint drops, same phase', lambda: [sphere([0.25, 0.25, 0.5], 0.2), sphere([0.75, 0.75, 0.5], 0.1)], 'DEFINED',
         truth_shifts=[0.0, 0.0], centre='UNSUPPORTED')
    case('C5', 'two disjoint drops, opposite sigma (review case 8c)', lambda: [sphere([0.25, 0.25, 0.5], 0.2), sphere([0.75, 0.75, 0.5], 0.1, sigma=-1)],
         'UNDEFINED')
    case('C6', 'drop with a cavity (cavity stored one period up)', lambda: [sphere([0.5, 0.5, 0.5], 0.3), sphere([0.55, 0.45, 1.5], 0.1, sigma=-1)],
         'DEFINED', truth_shifts=[0.0, -1.0], centre='DEFINED', queries=[((0.55, 0.45, 0.5), False), ((0.5, 0.5, 0.25), True)])
    case('C7', 'drop > cavity > drop (three levels)', lambda: [sphere([0.5, 0.5, 0.5], 0.35), sphere([0.5, 0.5, 0.5], 0.2, sigma=-1),
                                                            sphere([0.52, 0.5, 0.5], 0.08)], 'DEFINED', truth_shifts=[0, 0, 0], centre='UNSUPPORTED',
         queries=[((0.52, 0.5, 0.5), True), ((0.5, 0.5, 0.35), False), ((0.5, 0.5, 0.2), True)])
    case('C8', 'nested drops, same orientation (review case 8b)', lambda: [sphere([0.5, 0.5, 0.5], 0.3), sphere([0.52, 0.48, 0.5], 0.1)], 'UNDEFINED')
    case('C9', 'cavity outside any drop, sigma=-1, second drop sigma=+1 nested in it',
         lambda: [sphere([0.5, 0.5, 0.5], 0.3, sigma=-1), sphere([0.5, 0.5, 0.5], 0.1)], 'DEFINED',
         truth_fn=lambda ms: (1.0 + truth(ms, [0.0, 0.0])[0], 0.0), centre='UNDEFINED')
    case('C10', 'closed tube wrapping x (compact along z)', lambda: [tube()], 'DEFINED', truth_shifts=[0.0], centre='DEFINED')
    # ---------------------------------------------------------------- layers
    case('L1', 'one wrapping sheet', lambda: [sheet(lambda x, y: 0.3)], 'UNDEFINED')
    slab = lambda: [sheet(lambda x, y: 0.2, up=False, name='floor'), sheet(lambda x, y: 0.7, up=True, name='ceiling', zoff=1.0)]
    case('L2', 'two-sheet slab (ceiling stored one period up)', slab, 'DEFINED', truth_shifts=[0.0, -1.0], centre='DEFINED',
         queries=[((0.3, 0.3, 0.5), True), ((0.3, 0.3, 0.9), False), ((0.5, 0.5, 0.2 + 1e-5), True)])
    case('L3', 'complement of the slab', lambda: [sheet(lambda x, y: 0.2, up=True), sheet(lambda x, y: 0.7, up=False)], 'DEFINED',
         truth_shifts=[1.0, 0.0], centre='DEFINED')
    case('L4', 'three sheets', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3), sheet(lambda x, y: 0.5, up=False)], 'UNDEFINED')
    case('L5', 'four sheets (-,+,-,+): two slabs', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3), sheet(lambda x, y: 0.5, up=False),
                                                           sheet(lambda x, y: 0.8)], 'DEFINED', truth_shifts=[0, 0, 0, 0], centre='UNSUPPORTED')
    case('L6', 'four sheets (-,-,+,+): inconsistent', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3, up=False), sheet(lambda x, y: 0.5),
                                                              sheet(lambda x, y: 0.8)], 'UNDEFINED')
    six = lambda: [sheet(lambda x, y: 0.05, up=False), sheet(lambda x, y: 0.15, zoff=2), sheet(lambda x, y: 0.4, up=False, zoff=-1),
                   sheet(lambda x, y: 0.5), sheet(lambda x, y: 0.7, up=False, zoff=3), sheet(lambda x, y: 0.95)]
    case('L7', 'six alternating sheets in mixed images', six, 'DEFINED', truth_shifts=[0, -2, 1, 0, -3, 0], centre='UNSUPPORTED')
    case('L8', 'thin slab (h = 1e-6): resolved', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.2 + 1e-6)], 'DEFINED',
         truth_shifts=[0, 0], centre='DEFINED', tol=1e-12)
    case('L9', 'near-degenerate slab (h = 1e-11)', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.2 + 1e-11)], 'UNDEFINED')
    # L10 (review erratum E8): each body is two disconnected sheets. By §3.6.1 every connected
    # component is a member, so this is four layers (-,-,+,+) -> UNDEFINED as L6. (Revision 3.2's
    # model kept the body as one member with |Phi_k| = 2A -> UNSUPPORTED; that branch is unreachable
    # for connected embedded members and stays as a safety net.)
    case('L10', 'two bodies of two disconnected sheets each (components are members)',
         lambda: [merge([sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3, up=False)], 'double-'),
                  merge([sheet(lambda x, y: 0.5), sheet(lambda x, y: 0.7)], 'double+')], 'UNDEFINED')
    # ---------------------------------------------------------------- layers + compact
    b15 = lambda: [sheet(lambda x, y: 0.2 + 0.15 * math.sin(2 * math.pi * x), N=64, up=False, name='wavy floor'),
                   sheet(lambda x, y: 0.7, N=16, name='ceiling'), sphere([0.75, 0.5, 1.1], 0.04, sigma=-1, name='bubble')]
    case('M1', 'B15: wavy floor + ceiling + bubble in the dip (bubble at +1)', b15, 'DEFINED', truth_shifts=[0, 0, -1], centre='DEFINED',
         queries=[((0.75, 0.5, 0.1), False), ((0.75, 0.5, 0.155), True), ((0.25, 0.5, 0.3), False), ((0.25, 0.5, 0.36), True),
                  ((0.75 + 1 / 64, 0.5, 0.03), False)])
    b15b = lambda: [sheet(lambda x, y: 0.10 - 0.08 * math.sin(2 * math.pi * y), N=64, up=False, name='wavy floor'),
                    sheet(lambda x, y: 0.99, N=16, name='ceiling'), tube(0.5, 0.54, 0.42, M=64, sigma=-1, zoff=1.0, ref_bottom=True, name='wide x-tube')]
    case('M2', 'review case 4b: wide x-tube near a wavy floor (tube at +1)', b15b, 'DEFINED', truth_shifts=[0, 0, -1], centre='DEFINED')
    case('M3', 'slab + tube (other phase) inside it', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.8), tube(sigma=-1)],
         'DEFINED', truth_shifts=[0, 0, 0], centre='DEFINED')
    case('M4', 'slab + drop sigma=-1 in the non-phase region (review 8d)', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.5),
                                                                                  sphere([0.5, 0.5, 0.75], 0.1, sigma=-1)], 'UNDEFINED')
    case('M5', 'slab + phase drop in the non-phase region', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.5),
                                                                    sphere([0.5, 0.5, 1.75], 0.1)], 'DEFINED', truth_shifts=[0, 0, -1], centre='UNSUPPORTED')
    case('M6', 'slab + bubble whose reference vertex sits exactly above a floor vertex',
         lambda: [sheet(lambda x, y: 0.2, up=False, N=32), sheet(lambda x, y: 0.7, N=32), sphere([0.5, 0.5, 0.4], 0.1, sigma=-1)], 'DEFINED',
         truth_shifts=[0, 0, 0], centre='DEFINED')

    def m7():
        sp = sphere([0.5, 0.5, 0.4], 0.1, sigma=-1, name='bubble')
        c0 = sp.V[sp.T[0]].mean(0)  # first candidate point of the bubble ...
        sp.V[:, 0] += 0.5 - c0[0]  # ... placed exactly above a vertex of both layers (N = 32)
        sp.V[:, 1] += 0.5 - c0[1]
        return [sheet(lambda x, y: 0.2, up=False, N=32), sheet(lambda x, y: 0.7, N=32), sp]
    r = case('M7', 'first candidate line of the bubble passes through layer vertices', m7, 'DEFINED', truth_shifts=[0, 0, 0], centre='DEFINED')
    print(f"     (M7 candidate lines used per compact member: {r.get('member_line_tries')})")
    # ---------------------------------------------------------------- overhanging layers
    def fold(z0, delta, b=0.12, c=2.0):
        return lambda u: (u + delta + c * b * math.cos(2 * math.pi * u), z0 + b * math.cos(2 * math.pi * u))

    def ov(z0, delta, ceil_gap=0.3):
        top = z0 + 0.12 + ceil_gap
        zc = top - (1.0 if top >= 1.0 else 0.0)
        return lambda: [sheet(lambda x, y: zc, N=16, name='ceiling'), curve_sheet(fold(z0, delta), up=False, name='overhanging floor')], (1.0 if top >= 1.0 else 0.0)

    for tag, z0 in (('O1', 0.3), ('O2', 0.9)):
        b, sh = ov(z0, 0.0)
        case(tag, f'overhanging floor, z0 = {z0}', b, 'DEFINED', truth_shifts=[sh, 0.0], centre='DEFINED',
             queries=[((0.5, 0.5, z0 + 0.12 + 0.15), True)])
    # S-36-3: the canonical line through the overhang with the fold straddling the z seam
    f0 = curve_sheet(fold(0.0, 0.0), up=False)
    best = None
    for x in np.linspace(0, 1, 401):
        hs = sorted(h[0] for h in line_hits(f0, x, 0.43) if h[2])
        if len(hs) == 3 and (best is None or hs[2] - hs[0] > best[1]):
            best = (x, hs[2] - hs[0], hs)
    x3, _, hs = best
    delta = (PHI := (math.sqrt(5) - 1) / 2) - x3
    z0 = 1.0 - 0.5 * (hs[0] + hs[2])
    b, sh = ov(z0, delta)
    ms = b()
    hits = sorted(round(h[0], 4) for h in line_hits(ms[1], PHI % 1.0, (math.sqrt(2) - 1) % 1.0))
    print(f"     (O3 setup: overhang crossings on the canonical line at z = {hits})")
    case('O3', 'overhang straddling the z seam on the canonical line (S-36-3)', b, 'DEFINED', truth_shifts=[sh, 0.0], centre='DEFINED')
    # ---------------------------------------------------------------- unsupported / undefined topology
    case('U1', 'one tilted triply-wrapping plane x+z=c', lambda: [tilted_plane(0.3)], 'UNDEFINED')
    case('U2', 'two opposite tilted planes (valid, no admissible axis)', lambda: [tilted_plane(0.3), tilted_plane(0.7, up=False)], 'UNSUPPORTED')
    # ---------------------------------------------------------------- walls (z walled at 0 and 1)
    film = lambda up: (lambda: [sheet(lambda x, y: 0.6, up=up, name='film')])
    case('W1', 'B6: film over the z-min wall (full-wall patch)', film(True), 'DEFINED', dom=dict(WALLED, closures=['zmin']),
         truth_fn=lambda ms: (0.6, 0.3), centre='DEFINED', queries=[((0.5, 0.5, 0.3), True), ((0.5, 0.5, 0.8), False)])
    case('W2', 'film with the wrong orientation for a z-min closure', film(False), 'UNDEFINED', dom=dict(WALLED, closures=['zmin']))
    case('W3', 'film, no wall closure', film(True), 'UNDEFINED', dom=WALLED)
    case('W4', 'film, closures on both walls, film normal up', film(True), 'DEFINED', dom=dict(WALLED, closures=['zmin', 'zmax']),
         truth_fn=lambda ms: (0.6, 0.3), centre='DEFINED')
    capm = lambda sigma: (lambda: [sphere([0.5, 0.5, 0.0], 0.3, n=32, sigma=sigma, cap=True, name='cap')])
    def cap_with(sigma):
        def b():
            ms = capm(sigma)()
            ms[0].closure = 'zmin'
            return ms
        return b
    capV = lambda ms: truth(ms, [0.0])
    case('W5', 'B14: cap on a z-min wall orifice', cap_with(1), 'DEFINED', dom=dict(WALLED, closures=['zmin']), truth_fn=capV, centre='DEFINED',
         queries=[((0.5, 0.5, 0.1), True), ((0.1, 0.1, 0.1), False)])
    case('W6', 'cap, closures on both walls (flux selects the empty top patch)', cap_with(1), 'DEFINED', dom=dict(WALLED, closures=['zmin', 'zmax']),
         truth_fn=capV, centre='DEFINED')
    case('W7', 'inverted cap, z-min closure only', cap_with(-1), 'UNDEFINED', dom=dict(WALLED, closures=['zmin']))
    case('W8', 'inverted cap, both closures: complement region', cap_with(-1), 'DEFINED', dom=dict(WALLED, closures=['zmin', 'zmax']),
         truth_fn=lambda ms: truth(ms, [0.0], extra_V=1.0, extra_C2=1.0),
         centre='DEFINED', queries=[((0.5, 0.5, 0.1), False), ((0.1, 0.1, 0.1), True)])
    case('W9', 'open member without a declared closure', cap_with(1), 'UNDEFINED', dom=WALLED)

    def tunnel():
        ms = [tube(0.5, 0.0, 0.25, half=True, name='half-tube along x')]
        ms[0].closure = 'zmin'
        return ms
    case('W10', 'half tube along x on the wall: loops wind (valid, unsupported)', tunnel, 'UNSUPPORTED', dom=dict(WALLED, closures=['zmin']))
    case('W11', 'drop in a walled domain', lambda: [sphere([0.5, 0.5, 0.5], 0.2)], 'DEFINED', dom=WALLED, truth_shifts=[0.0], centre='DEFINED')
    case('W12', 'bubble sigma=-1 alone in a walled domain', lambda: [sphere([0.5, 0.5, 0.5], 0.2, sigma=-1)], 'UNDEFINED', dom=WALLED)
    case('W13', 'B6 film + cavity below it', lambda: [sheet(lambda x, y: 0.6, name='film'), sphere([0.5, 0.5, 0.3], 0.15, sigma=-1)], 'DEFINED',
         dom=dict(WALLED, closures=['zmin']), truth_fn=lambda ms: truth(ms[1:], [0.0], extra_V=0.6, extra_C2=0.36), centre='DEFINED', tol=1e-10)
    case('W14', 'two drops in a walled domain (centre along a walled axis)', lambda: [sphere([0.3, 0.3, 0.3], 0.15), sphere([0.7, 0.7, 0.6], 0.1)],
         'DEFINED', dom=WALLED, truth_shifts=[0.0, 0.0], centre='DEFINED')
    case('W15', 'drop with closures on both walls: empty or full-box complement', lambda: [sphere([0.5, 0.5, 0.5], 0.2)], 'UNSUPPORTED',
         dom=dict(WALLED, closures=['zmin', 'zmax']))
    print(f"\n{NPASS} passed, {NFAIL} failed")
    return NFAIL


if __name__ == '__main__':
    sys.exit(1 if main() else 0)

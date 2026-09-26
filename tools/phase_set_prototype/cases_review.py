"""Additional §3.6 revision-3.2 review cases (independent review 2026-09-24): interlocking
overhangs, overhang stacks, seam-straddling wavy layers, compacts near seams / in overhang
pockets. Same harness and truth convention as cases.py."""
import math, sys
import numpy as np
import cases as K
from cases import case, sheet, sphere, tube, merge
from p36 import line_hits, Member
from interlock import offset_layer, base

X_LINE = 0.26  # a canonical line through the interlocking zone


def u_at_x(x_target, z0=0.3):
    best = None
    for i in range(20001):
        u = i / 20000
        x = base(u, z0)[0] % 1.0
        if best is None or abs(x - x_target) < best[1]:
            best = (u, abs(x - x_target))
    return best[0]


PHI1 = (math.sqrt(5) - 1) / 2
U0 = u_at_x((X_LINE - PHI1) % 1.0)  # the model's first generic canonical candidate is n = 1 (n = 0 hits the reference vertex)


def L(off, up, zoff=0.0, z0=0.3, name='fold', sigma=1):
    return offset_layer(z0, off, up, name=name, zoff=zoff, u0=U0, sigma=sigma)


def show_line(ms):
    """crossing order (reduced heights) on the canonical line the model actually uses"""
    import p36
    x0, y0 = ms[0].V[0, 0], ms[0].V[0, 1]
    for i in range(p36.N_TRY):
        qx, qy = (x0 + (i * p36.PHI1 % 1.0)) % 1.0, (y0 + (i * p36.PHI2 % 1.0)) % 1.0
        hs = [(h[0] % 1.0, mi, h[2]) for mi, m in enumerate(ms) for h in line_hits(m, qx, qy)]
        if all(g for _, _, g in hs):
            return f"candidate {i} at x={qx:.4f}: " + ''.join('ABCDEFGH'[mi] for _, mi, _ in sorted(hs))
    return 'no generic candidate' 


def main():
    d = 0.02
    # ---------------------------------------------------------------- interlocking overhangs
    i1 = lambda: [L(0.0, False, name='Y1 floor'), L(d, True, zoff=1.0, name='Y2 ceiling (+1)')]
    print('     (I1 canonical line crossing order (A=Y1, B=Y2):', show_line(i1()), ')')
    case('I1', 'interlocking: thin folded slab, ceiling in the floor pocket', i1, 'DEFINED', truth_shifts=[0.0, -1.0], centre='DEFINED')
    i2 = lambda: [L(0.0, True, zoff=-2.0, name='Y1 (-2)'), L(d, False, name='Y2')]
    case('I2', 'interlocking: complement of the folded slab', i2, 'DEFINED', truth_shifts=[3.0, 0.0], centre='DEFINED')
    i3 = lambda: [L(0.0, False, name='Y1'), L(0.012, True, zoff=1.0, name='Y2'), L(0.024, False, zoff=-1.0, name='Y3'), L(0.036, True, name='Y4')]
    print('     (I3 canonical line crossing order:', show_line(i3()), ')')
    case('I3', 'interlocking 4-layer stack (-,+,-,+) in mixed images', i3, 'DEFINED', truth_shifts=[0, -1, 1, 0], centre='UNSUPPORTED')
    case('I4', 'interlocking 3-layer stack', lambda: [L(0.0, False), L(0.012, True), L(0.024, False)], 'UNDEFINED')
    case('I5', 'interlocking 4 layers (-,-,+,+)', lambda: [L(0.0, False), L(0.012, False), L(0.024, True), L(0.036, True)], 'UNDEFINED')
    i6 = lambda: [L(0.0, True, name='Y1 up'), L(0.012, False, name='Y2 down'), L(0.024, True, zoff=1.0, name='Y3 up'), L(0.036, False, name='Y4 down')]
    case('I6', 'interlocking 4-layer stack (+,-,+,-) complement', i6, 'DEFINED', truth_shifts=[1, 0, -1, 0], centre='UNSUPPORTED')

    # bubble inside the folded slab, in the overhang pocket (centre on the mid-surface)
    def i7():
        dd = 0.04
        u = U0 + 0.012  # a point on the fold near the canonical line
        p = base(u, 0.3) + 0.5 * dd * np.array(K_normal(u))
        return [L(0.0, False, name='Y1 floor'), L(dd, True, name='Y2 ceiling'),
                sphere([p[0], 0.5, p[1] + 1.0], 0.009, n=10, sigma=-1, name='bubble (+1)')]
    case('I7', 'bubble in the overhang pocket of an interlocking slab (stored +1)', i7, 'DEFINED', truth_shifts=[0, 0, -1.0], centre='DEFINED')
    i8 = lambda: [L(0.0, False, z0=0.9, name='Y1 floor'), L(d, True, z0=0.9, zoff=-1.0, name='Y2 ceiling (-1)')]
    case('I8', 'interlocking slab whose fold straddles the z seam', i8, 'DEFINED', truth_shifts=[0.0, 1.0], centre='DEFINED')
    # point queries in the interlocking zone: along the canonical line
    def i1q():
        return i1()
    ms = i1()
    x, y = ms[0].V[0, 0] % 1.0, ms[0].V[0, 1] % 1.0
    hs = sorted((h[0] % 1.0, mi, h[1]) for mi, m in enumerate(ms) for h in line_hits(m, x, y) if h[2])
    qs = []
    for (z1, m1, _), (z2, m2, _) in zip(hs, hs[1:]):
        zm = 0.5 * (z1 + z2)
        # truth: inside the slab <=> between a Y1 crossing (below) and a Y2 crossing (above) at level 1:
        # computed from the physical placement: count signed crossings below zm (s=-1 enters)
        lev = 0
        for (zz, mi, sg) in hs:
            if zz < zm:
                s = ms[mi].sigma * sg
                lev += -s
        qs.append(((x + 1e-4, y + 1e-4, zm), bool(lev % 2)))  # nearby point, same region (segments are wide)
    case('I1q', 'I1 point queries between consecutive crossings of the fold line', i1, 'DEFINED', queries=qs)
    # ---------------------------------------------------------------- wavy layers across the seam, stacks, overhang stacks
    w1 = lambda: [sheet(lambda x, y: 0.02 + 0.1 * math.sin(2 * math.pi * x), N=64, up=False, name='wavy floor across seam'),
                  sheet(lambda x, y: 0.5 + 0.05 * math.cos(2 * math.pi * y), N=32, zoff=5.0, name='ceiling (+5)')]
    case('S1', 'wavy floor crossing the z seam, ceiling stored +5 periods', w1, 'DEFINED', truth_shifts=[0.0, -5.0], centre='DEFINED',
         queries=[((0.75, 0.3, 0.95), True), ((0.75, 0.3, 0.85), False), ((0.75, 0.3, 0.0), True), ((0.25, 0.3, 0.1), False), ((0.25, 0.3, 0.2), True)])
    w2 = lambda: [sheet(lambda x, y: 0.02 + 0.1 * math.sin(2 * math.pi * x), N=64, up=False), sheet(lambda x, y: 0.3, N=16, zoff=1.0),
                  sheet(lambda x, y: 0.45 + 0.1 * math.sin(2 * math.pi * (x + y)), N=48, up=False, zoff=-1.0), sheet(lambda x, y: 0.8, N=16)]
    case('S2', '4-layer stack of wavy layers, one across the seam', w2, 'DEFINED', truth_shifts=[0, -1, 1, 0], centre='UNSUPPORTED')
    def w3():
        b, sh = K_ov(0.3, 0.0)
        ms = b()
        return ms + [sheet(lambda x, y: 0.85, N=16, up=False, name='third'), sheet(lambda x, y: 0.95, N=16, name='fourth')]
    case('S3', 'overhanging floor in a 4-layer stack', w3, 'DEFINED', truth_shifts=[0, 0, 0, 0], centre='UNSUPPORTED')
    # ---------------------------------------------------------------- compacts near seams / overhangs
    c1 = lambda: [sheet(lambda x, y: 0.9, up=False, name='floor'), sheet(lambda x, y: 0.3, zoff=1.0, name='ceiling (+1)'),
                  sphere([0.5, 0.5, 0.0], 0.1, sigma=-1, name='bubble across the seam')]
    case('S4', 'bubble crossing the z seam inside a slab crossing the seam', c1, 'DEFINED', truth_shifts=[0.0, 0.0, 1.0], centre='DEFINED',
         queries=[((0.5, 0.5, 0.0), False), ((0.5, 0.5, 0.15), True), ((0.1, 0.1, 0.99), True)])
    c2 = lambda: [sheet(lambda x, y: 0.9, up=False, name='floor'), sheet(lambda x, y: 0.3, zoff=1.0, name='ceiling (+1)'),
                  sphere([0.5, 0.5, 3.0], 0.1, sigma=-1, name='bubble stored +3')]
    case('S5', 'same, bubble stored three periods up', c2, 'DEFINED', truth_shifts=[0.0, 0.0, -2.0], centre='DEFINED')
    # ---------------------------------------------------------------- wall pieces (§3.6.2 step 7)
    WALLED = dict(per_z=False, closures=['zmin'])
    def caps(*spec):
        def b():
            ms = []
            for (cx, cy, r, sg, nm) in spec:
                m = sphere([cx, cy, 0.0], r, n=24, sigma=sg, cap=True, name=nm)
                m.closure = 'zmin'
                ms.append(m)
            return ms
        return b
    capV = lambda ms: K.truth(ms, [0.0] * len(ms))
    r = case('W16', 'two separate caps on the z-min wall (two drops)', caps((0.3, 0.3, 0.15, 1, 'cap1'), (0.7, 0.7, 0.1, 1, 'cap2')), 'DEFINED',
             dom=WALLED, truth_fn=capV, centre='DEFINED', queries=[((0.3, 0.3, 0.05), True), ((0.7, 0.7, 0.05), True), ((0.5, 0.5, 0.05), False)])
    extra_check('W16', r.get('n_phase_components') == 2, f"phase components {r.get('n_phase_components')} (expect 2: two wall pieces)")
    r = case('W17', 'nested caps on the wall: drop cap with a cavity cap', caps((0.5, 0.5, 0.3, 1, 'outer cap'), (0.5, 0.5, 0.1, -1, 'inner cap')), 'DEFINED',
             dom=WALLED, truth_fn=capV, centre='DEFINED', queries=[((0.5, 0.5, 0.05), False), ((0.5, 0.5, 0.2), True), ((0.1, 0.1, 0.05), False)])
    extra_check('W17', r.get('n_phase_components') == 1, f"phase components {r.get('n_phase_components')} (expect 1: one annulus piece)")
    def caps_cav():
        ms = caps((0.3, 0.3, 0.15, 1, 'cap1'), (0.7, 0.7, 0.25, 1, 'cap2'))()
        return ms + [sphere([0.7, 0.7, 0.12], 0.05, n=12, sigma=-1, name='cavity in cap2')]
    r = case('W18', 'two caps, a cavity inside the second', caps_cav, 'DEFINED', dom=WALLED, truth_fn=lambda ms: K.truth(ms, [0, 0, 0]), centre='DEFINED',
             queries=[((0.7, 0.7, 0.12), False), ((0.7, 0.7, 0.2), True)])
    extra_check('W18', r.get('n_phase_components') == 2, f"phase components {r.get('n_phase_components')}")
    case('W19', 'two caps with opposite sigma on one wall', caps((0.3, 0.3, 0.15, 1, 'cap1'), (0.7, 0.7, 0.1, -1, 'cap2')), 'UNDEFINED', dom=WALLED)

    def bridge():
        V, T = [], []
        Nz, M, r = 16, 24, 0.15
        for i in range(Nz + 1):
            for j in range(M):
                th = 2 * math.pi * j / M
                V.append([0.5 + r * math.cos(th), 0.5 + r * math.sin(th), i / Nz])
        idx = lambda i, j: i * M + j % M
        for i in range(Nz):
            for j in range(M):
                a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
                T.append([a, b, c]); T.append([a, c, d])
        m = Member(V, T, 1, 'bridge')
        P = m.V[m.T[0]]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        if np.dot(nrm, P.mean(0) - np.array([0.5, 0.5, P.mean(0)[2]])) < 0:
            m = Member(V, [[a, c, b] for a, b, c in T], 1, 'bridge')
        m.closure = 'zmin'
        return m
    def bridge_case():
        m = bridge()
        # the loops lie on two walls: the model keeps one closure per member, so the bridge is
        # split into two half columns glued at mid height is not possible; use the both-walls
        # declaration with the member's loops checked against each wall
        return [m]
    # (the model supports one closure wall per member; the both-wall bridge is documented as
    # UNSUPPORTED "no generic line": every member triangle is vertical, S-review item)
    # ---------------------------------------------------------------- degeneracy (D)
    case('D1', 'single tiny drop (V < tau_V)', lambda: [sphere([0.5, 0.5, 0.5], 2.5e-4, n=8)], 'UNSUPPORTED')
    case('D2', 'tiny drop next to a normal drop', lambda: [sphere([0.3, 0.3, 0.5], 0.2), sphere([0.8, 0.8, 0.5], 2.5e-4, n=8)], 'UNSUPPORTED')
    case('D3', 'matrix phase with a tiny bubble (V > V_cell - tau_V)', lambda: [sphere([0.5, 0.5, 0.5], 2.5e-4, n=8, sigma=-1)], 'UNSUPPORTED')
    case('D4', 'slab of the full cell minus 1e-11 (complement of L9)', lambda: [sheet(lambda x, y: 0.2, up=True), sheet(lambda x, y: 0.2 + 1e-11, up=False)],
         'UNDEFINED')
    print(f"\n{K.NPASS} passed, {K.NFAIL} failed")
    return K.NFAIL


def extra_check(tag, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'} {tag}+  {msg}")
    if ok:
        K.NPASS += 1
    else:
        K.NFAIL += 1


def K_normal(u):
    from interlock import normal
    return normal(u)


def K_ov(z0, delta, ceil_gap=0.3):
    def fold(z0, delta, b=0.12, c=2.0):
        return lambda u: (u + delta + c * b * math.cos(2 * math.pi * u), z0 + b * math.cos(2 * math.pi * u))
    top = z0 + 0.12 + ceil_gap
    zc = top - (1.0 if top >= 1.0 else 0.0)
    return (lambda: [sheet(lambda x, y: zc, N=16, name='ceiling'), K.curve_sheet(fold(z0, delta), up=False, name='overhanging floor')]), 0.0


if __name__ == '__main__':
    sys.exit(1 if main() else 0)

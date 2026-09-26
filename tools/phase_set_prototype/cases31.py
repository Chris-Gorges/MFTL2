"""Reviewer prototype for plans/16 §3.6 (independent review 2026-09-24, §18.2). Not part of the build."""
import math, sys
import numpy as np
from p31 import *


def truth(members, shifts):
    """reference V, Cz for a manually (physically) placed configuration"""
    for m, s in zip(members, shifts):
        m.shift = s
    V = sum(integrals(m)[1] for m in members)
    c2 = sum(integrals(m)[2] for m in members)
    for m in members:
        m.shift = 0.0
    return V, c2 / (2 * V)


def simple_curve(xz, N=400):
    P = np.array([xz(i / N) for i in range(N)])
    Q = np.vstack([P, P[:1] + [1, 0]])
    segs = [(Q[i], Q[i + 1]) for i in range(N)]
    def inter(a, b, c, d):
        def orient(p, q, r):
            return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0
    bad = 0
    for i in range(N):
        for j in range(i + 2, N):
            for sh in (-1, 0, 1):
                c, d = segs[j][0] + [sh, 0], segs[j][1] + [sh, 0]
                if (i == 0 and j == N - 1):
                    continue
                if inter(segs[i][0], segs[i][1], c, d):
                    bad += 1
    return bad


MODES = ['literal', 'F12', 'fixed']


def run(title, build, expect_truth=None, expect=None):
    print(f"-- {title}" + (f"   [expected: {expect}]" if expect else ''))
    for mode in MODES:
        members = build()
        res = classify(members, mode)
        tag = ''
        if expect_truth is not None and res['status'] == 'DEFINED':
            Vt, Ct = expect_truth(build())
            dv, dc = abs(res['V'] - Vt), abs(res['Cz'] - Ct)
            tag = f"  |dV|={dv:.1e} |dCz|={dc:.1e}" + ("   WRONG VALUE" if max(dv, dc) > 1e-10 else "   ok")
        report(mode, res)
        if tag:
            print(' ' * 4 + tag)


# 1 seam-crossing sphere
run('1 seam-crossing sphere (x,z seams)', lambda: [sphere([0.95, 0.5, 0.97], 0.2, sigma=1)],
    expect_truth=lambda ms: truth(ms, [0.0]), expect='DEFINED')

# 2 single wrapping sheet
run('2 one wrapping xy sheet', lambda: [sheet(lambda x, y: 0.3, up=True)], expect='UNDEFINED')

# 3 slab, ceiling stored one period up
def slab():
    return [sheet(lambda x, y: 0.2, up=False, name='floor'), sheet(lambda x, y: 0.7, up=True, name='ceil', zoff=1.0)]
run('3 two opposite sheets (slab, ceiling in image +1)', slab, expect_truth=lambda ms: truth(ms, [0.0, -1.0]), expect='DEFINED V=0.5 Cz=0.45')

# 3b complement slab (orientation flipped)
def slab_c():
    return [sheet(lambda x, y: 0.2, up=True, name='floor'), sheet(lambda x, y: 0.7, up=False, name='ceil')]
run('3b complement of the slab', slab_c, expect_truth=lambda ms: truth(ms, [1.0, 0.0]), expect='DEFINED V=0.5 Cz=0.95')

# 4 wavy floor + flat ceiling + bubble in the dip (B15), bubble stored one period up
def b15():
    fl = sheet(lambda x, y: 0.2 + 0.15 * math.sin(2 * math.pi * x), N=64, up=False, name='wavy floor')
    ce = sheet(lambda x, y: 0.7, N=16, up=True, name='ceiling')
    bu = sphere([0.75, 0.5, 1.1], 0.04, sigma=-1, name='bubble')
    return [fl, ce, bu]
run('4 wavy floor + ceiling + bubble in dip (bubble gauge +1)', b15, expect_truth=lambda ms: truth(ms, [0.0, 0.0, -1.0]), expect='DEFINED, bubble at z=0.1')

# 5 overhanging floor (breaking-wave cap), two vertical placements
def fold(z0):
    return lambda u: (u - 0.25 * math.sin(2 * math.pi * u), z0 + 0.12 * math.cos(2 * math.pi * u))
for z0 in (0.3, 0.9):
    print(f"   (fold curve self-intersections: {simple_curve(fold(z0))})")
    def ov(z0=z0):
        f = curve_sheet(fold(z0), up=False, name='fold floor')
        c = sheet(lambda x, y: z0 + 0.45, N=16, up=True, name='ceiling', zoff=(-1.0 if z0 + 0.45 >= 1 else 0.0))
        return [f, c]
    sh = 1.0 if z0 + 0.45 >= 1 else 0.0
    run(f'5 overhanging floor z0={z0} (3 crossings on some lines)', ov, expect_truth=lambda ms, sh=sh: truth(ms, [0.0, sh]), expect='DEFINED')
    # verify 3 crossings exist
    f = curve_sheet(fold(z0), up=False)
    cnt = max(len(crossings(f, x, 0.43)[0]) for x in np.linspace(0, 1, 101))
    print(f"   (max crossings of the fold floor on a vertical line: {cnt})")

# 6 layer stacks
run('6a four sheets (-,+,-,+): two slabs', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3, up=True), sheet(lambda x, y: 0.5, up=False),
                                                     sheet(lambda x, y: 0.8, up=True)], expect='DEFINED V=0.5 (set centre UNSUPPORTED per plan)')
run('6b three sheets', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3, up=True), sheet(lambda x, y: 0.5, up=False)], expect='UNDEFINED')
run('6c six sheets, alternating, mixed images', lambda: [sheet(lambda x, y: 0.05, up=False), sheet(lambda x, y: 0.15, up=True, zoff=2), sheet(lambda x, y: 0.4, up=False, zoff=-1),
                                                          sheet(lambda x, y: 0.5, up=True), sheet(lambda x, y: 0.7, up=False, zoff=3), sheet(lambda x, y: 0.95, up=True)],
    expect='DEFINED V=0.1+0.1+0.25=0.45')

# 7 homology cancelling / laterally wrapping compact member: closed tube along x
def tube(yc=0.5, zc=0.5, r=0.15, N=40, M=48, sigma=1):
    V, T = [], []
    for i in range(N + 1):
        for j in range(M):
            th = 2 * math.pi * j / M
            V.append([i / N, yc + r * math.cos(th), zc + r * math.sin(th)])
    idx = lambda i, j: i * M + (j % M)
    for i in range(N):
        for j in range(M):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T.append([a, c, b]); T.append([a, d, c])
    return Member(V, T, sigma, 'tube')
run('7 closed tube wrapping x (Phi=0, compact along z)', lambda: [tube()], expect_truth=lambda ms: truth(ms, [0.0]), expect='DEFINED V~pi r^2')
run('7b slab + tube inside it (tube = other phase)', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.8, up=True), tube(sigma=-1)],
    expect_truth=lambda ms: truth(ms, [0.0, 0.0, 0.0]), expect='DEFINED')

# 8 inconsistent orientation
run('8a four sheets (-,-,+,+)', lambda: [sheet(lambda x, y: 0.1, up=False), sheet(lambda x, y: 0.3, up=False), sheet(lambda x, y: 0.5, up=True),
                                         sheet(lambda x, y: 0.8, up=True)], expect='UNDEFINED')
run('8b nested drops, same orientation (inner drop sigma=+1)', lambda: [sphere([0.5, 0.5, 0.5], 0.3, sigma=1), sphere([0.52, 0.48, 0.5], 0.1, sigma=1)], expect='UNDEFINED')
run('8c laterally disjoint drops, sigma=+1 and sigma=-1', lambda: [sphere([0.25, 0.25, 0.5], 0.2, sigma=1), sphere([0.75, 0.75, 0.5], 0.1, sigma=-1)],
    expect='UNDEFINED (levels -1,0,+1)')
run('8d slab + drop in the non-phase region with sigma=-1', lambda: [sheet(lambda x, y: 0.2, up=False), sheet(lambda x, y: 0.5, up=True), sphere([0.5, 0.5, 0.75], 0.1, sigma=-1)],
    expect='UNDEFINED')
run('8e two disjoint drops, both sigma=+1 (consistent)', lambda: [sphere([0.25, 0.25, 0.5], 0.2, sigma=1), sphere([0.75, 0.75, 0.5], 0.1, sigma=1)],
    expect_truth=lambda ms: truth(ms, [0.0, 0.0]), expect='DEFINED V=V1+V2 (centre UNSUPPORTED per plan)')
run('8f matrix: one bubble sigma=-1 in the triply periodic box', lambda: [sphere([0.5, 0.5, 0.5], 0.2, sigma=-1)], expect='DEFINED V=1-Vb (valid region)')

# 4b literal local-ray rule gives a WRONG value: wavy floor (along y), ceiling, wide tube along x
def tube_bottom_ref(yc, zc, r, N=40, M=64, sigma=-1, zoff=0.0):
    V, T = [], []
    for i in range(N + 1):
        for j in range(M):
            th = -math.pi / 2 + 2 * math.pi * j / M  # j=0: bottom point (reference vertex)
            V.append([i / N, yc + r * math.cos(th), zc + r * math.sin(th) + zoff])
    idx = lambda i, j: i * M + (j % M)
    for i in range(N):
        for j in range(M):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T.append([a, b, c]); T.append([a, c, d])
    m = Member(V, T, sigma, 'wide tube')
    # orientation: outward normals (check sign of own volume)
    if integrals(m)[1] * sigma < 0:
        m.T = m.T[:, [0, 2, 1]]
    return m
def b15b():
    fl = sheet(lambda x, y: 0.10 - 0.08 * math.sin(2 * math.pi * y), N=64, up=False, name='wavy floor')
    ce = sheet(lambda x, y: 0.99, N=16, up=True, name='ceiling')
    tb = tube_bottom_ref(0.5, 0.54, 0.42, zoff=1.0)
    return [fl, ce, tb]
run('4b wavy floor + ceiling + wide x-tube near the floor (tube gauge +1)', b15b, expect_truth=lambda ms: truth(ms, [0.0, 0.0, -1.0]), expect='DEFINED, tube at z=0.54')


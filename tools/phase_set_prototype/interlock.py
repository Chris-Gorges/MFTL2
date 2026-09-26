"""Interlocking overhang layers for the §3.6 review: offset curves of a folded curve, extruded
along y. Y_j = Y_1 + j*d*n (normal offsets) are embedded and disjoint for j*d < min radius; on
lines through the fold their crossings interleave (a layer reaches into the overhang pocket of
the layer below)."""
import math, numpy as np
from p36 import Member, line_hits

B, C = 0.3, 0.7


def base(u, z0):
    return np.array([u + C * B * math.cos(2 * math.pi * u), z0 + B * math.cos(2 * math.pi * u)])


def normal(u):
    t = np.array([1 - 2 * math.pi * C * B * math.sin(2 * math.pi * u), -2 * math.pi * B * math.sin(2 * math.pi * u)])
    t /= np.linalg.norm(t)
    return np.array([-t[1], t[0]])


def offset_layer(z0, off, up, sigma=1, N=600, M=6, name='fold', zoff=0.0, u0=0.0):
    V, T = [], []
    for j in range(M + 1):
        for i in range(N + 1):
            u = u0 + i / N
            p = base(u, z0) + off * normal(u)
            V.append([p[0], j / M + 0.37 / M, p[1] + zoff])
    idx = lambda i, j: j * (N + 1) + i
    for j in range(M):
        for i in range(N):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            if up:
                T.append([a, b, c]); T.append([a, c, d])
            else:
                T.append([a, c, b]); T.append([a, d, c])
    return Member(V, T, sigma, name, wrap=(1, 1, 0))


if __name__ == '__main__':
    z0, d = 0.3, 0.02
    ms = [offset_layer(z0, 0.0, False, name='Y1'), offset_layer(z0, d, True, name='Y2')]
    for x in np.linspace(0, 1, 201):
        hs = sorted((h[0], mi) for mi, m in enumerate(ms) for h in line_hits(m, x, 0.37) if h[2])
        seq = ''.join('ab'[mi] for _, mi in hs)
        if 'ab' in seq and seq not in ('ab',):
            print(round(x, 3), seq, [round(h, 4) for h, _ in hs])

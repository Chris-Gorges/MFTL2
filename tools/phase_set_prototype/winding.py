"""§3.3 winding in the prototype: from a member's continuous coordinates, build the WRAPPED
storage (pos in [0,L), img = floor), identify vertices (same wrapped position), compute the
edge lattice shifts s(u,v) = round((X_v - X_u - d(u,v))/L) of the stored representatives,
a deterministic BFS spanning tree (root = lowest id), potentials phi and the cycle residuals
r(e) = s(e) - (phi(v) - phi(u)). WRAP_k <=> some r_k != 0. Returns (flags, lattice generators)."""
import numpy as np
from collections import deque

L = np.array([1.0, 1.0, 1.0])


def winding(m, per=(True, True, True)):
    X = m.V
    per = np.array(per)
    key = {}
    vid = []
    rep = []
    for x in X:
        w = np.where(per, np.mod(x, L), x)
        k = tuple(np.round(w, 7) % np.where(per, 1.0, 1e300))
        if k not in key:
            key[k] = len(rep)
            rep.append(np.where(per, w, x))  # stored position: wrapped; img = 0 for every representative
        vid.append(key[k])
    rep = np.array(rep)
    edges = {}
    for t in m.T:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            u, v = vid[a], vid[b]
            D = X[b] - X[a]  # true (short) displacement of the edge
            s = np.round((rep[v] - rep[u] - D) / L).astype(int) * per  # lattice shift in the stored gauge
            if (v, u) in edges:
                continue
            edges[(u, v)] = s
    n = len(rep)
    adj = [[] for _ in range(n)]
    for (u, v), s in edges.items():
        adj[u].append((v, s))
        adj[v].append((u, -s))
    phi = [None] * n
    gens = []
    for r0 in range(n):
        if phi[r0] is not None:
            continue
        phi[r0] = np.zeros(3, int)
        q = deque([r0])
        while q:
            u = q.popleft()
            for v, s in sorted(adj[u], key=lambda e: e[0]):
                if phi[v] is None:
                    phi[v] = phi[u] + s
                    q.append(v)
    for (u, v), s in edges.items():
        r = s - (phi[v] - phi[u])
        if np.any(r != 0):
            gens.append(tuple(int(c) for c in r))
    flags = tuple(int(any(g[k] != 0 for g in gens)) for k in range(3))
    return flags, sorted(set(gens))

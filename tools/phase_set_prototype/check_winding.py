"""F: the wrap flags used by §3.6 are exactly the §3.3 flags derived from the mesh winding."""
import math, sys
import numpy as np
import cases as K
from winding import winding
from p36 import classify, Member

bad = 0
def chk(name, m, expect=None):
    global bad
    f, gens = winding(m)
    exp = expect if expect is not None else m.wrap
    ok = tuple(f) == tuple(exp)
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name:40s} declared {tuple(exp)} derived {f} lattice {gens[:4]}{'...' if len(gens) > 4 else ''}")
    return f

ms = [('sphere', K.sphere([0.4, 0.5, 0.45], 0.2)), ('seam sphere', K.sphere([0.95, 0.5, 0.97], 0.2)),
      ('flat sheet', K.sheet(lambda x, y: 0.3)), ('wavy sheet', K.sheet(lambda x, y: 0.02 + 0.1 * math.sin(2 * math.pi * x), N=32)),
      ('x-tube', K.tube()), ('tilted plane', K.tilted_plane(0.3)),
      ('fold', K.curve_sheet(lambda u: (u + 0.24 * math.cos(2 * math.pi * u), 0.3 + 0.12 * math.cos(2 * math.pi * u))))]
for n, m in ms:
    chk(n, m)

# a diagonal tube along (1,1,0): winding lattice (1,1,0) -> WRAP_x and WRAP_y (conservative, §3.4)
def diag_tube(r=0.1, N=64, M=16):
    V, T = [], []
    d = np.array([1.0, 1.0, 0.0]); e1 = np.array([1.0, -1.0, 0.0]) / math.sqrt(2); e2 = np.array([0.0, 0.0, 1.0])
    for i in range(N + 1):
        for j in range(M):
            th = 2 * math.pi * j / M
            V.append(list(np.array([0.2, 0.3, 0.5]) + d * i / N + r * (math.cos(th) * e1 + math.sin(th) * e2)))
    idx = lambda i, j: i * M + j % M
    for i in range(N):
        for j in range(M):
            a, b, c, dd = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T.append([a, b, c]); T.append([a, c, dd])
    T = K._orient_out(V, T)
    return Member(V, T, 1, 'diagonal tube', wrap=(1, 1, 0))
dt = diag_tube()
f = chk('diagonal tube (lattice (1,1,0))', dt)
# explicit winding data -> §3.6 classification: flags from the winding, not declared
dt.wrap = f
r = classify([dt])
print(f"     diagonal tube classified with derived flags: {r['status']} V={r.get('V')}")
bad += r['status'] != 'DEFINED'
tp = K.tilted_plane(0.3); tp2 = K.tilted_plane(0.7, up=False)
tp.wrap = winding(tp)[0]; tp2.wrap = winding(tp2)[0]
r = classify([tp, tp2])
print(f"     two opposite tilted planes, derived flags {tp.wrap}: {r['status']} ({r['why']})")
bad += r['status'] != 'UNSUPPORTED'
print('winding checks:', 'OK' if not bad else f'{bad} FAILED')
sys.exit(1 if bad else 0)

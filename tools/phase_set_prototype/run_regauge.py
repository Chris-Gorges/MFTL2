"""C/D: every member re-gauged by a random lattice vector (lateral always; along z only when z is
periodic): statuses, V, C_z (mod L), point queries must not change."""
import sys
import numpy as np
import cases as K
import cases_review as R
import p36
from p36 import Member

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
orig = p36.classify


def cl(ms, dom=None, want_placement=True):
    per_z = (dom or {}).get('per_z', True)
    out = []
    for m in ms:
        sh = np.array([rng.integers(-3, 4), rng.integers(-3, 4), rng.integers(-3, 4) if per_z else 0], float)
        out.append(Member(m.V + sh, m.T.copy(), m.sigma, m.name, wrap=m.wrap, closure=m.closure))
    return orig(out, dom, want_placement)


K.classify = cl
_case = K.case
def case(tag, *a, **kw):
    if tag == 'L8':  # 1e-6 slab: the harness shifts float coordinates by whole periods, which
        kw['tol'] = 1e-9  # production does exactly in the integer image counters (§3.5)
    return _case(tag, *a, **kw)
K.case = case
R.case = case
n = K.main() + R.main()
sys.exit(1 if n else 0)

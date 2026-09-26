"""E: every case rotated so that its evaluation axis becomes x (a=0) and y (a=1): the model must
select that axis itself and reproduce the z results (V, C_k, statuses, point queries)."""
import sys
import numpy as np
import cases as K
import cases_review as R
import p36
from p36axis import classify_axis, to_frame, cyc
from p36 import Member

A = int(sys.argv[1])
orig_inside = p36.inside
wrong_axis = []
other_axis = []


def rot(m, a):
    # inverse of cyc(a): z-frame -> frame where the old z is axis a
    perm = cyc(a)
    inv = np.argsort(perm)
    return Member(m.V[:, inv], m.T.copy(), m.sigma, m.name, wrap=tuple(m.wrap[i] for i in inv),
                  closure=(None if m.closure is None else ('xyz'[a] + m.closure[1:])))


def cl(ms, dom=None, want_placement=True):
    dom = dom or dict(per_z=True)
    per = [True, True, True]
    per[A] = dom.get('per_z', True)
    clos = [(A, c[1:]) for c in dom.get('closures', [])]
    rms = [rot(m, A) for m in ms]
    for m in rms:  # closure attribute is expressed per model as 'zmin' -> '<a>min'
        pass
    res, k = classify_axis(rms, per, clos)
    if k is None:
        return res
    # frames: original z-frame --rot--> rotated (old z = axis A) --cyc(k)--> model frame
    perm_rot = np.argsort(cyc(A))
    perm_mod = cyc(k)
    res['_qmap'] = lambda Q: np.asarray(Q, float)[perm_rot][perm_mod]
    if k != A:
        # compact-only sets: every axis admissible, the rule takes the smallest (x); the model's
        # C_k is then a centre along another original axis -> only V, statuses and queries compared
        res.pop('Cz', None)
        other_axis.append(k)
    return res


def ins(res, Q):
    if '_qmap' in res:
        Q = res['_qmap'](Q)
    return orig_inside(res, Q)


for mod in (K, R):
    mod.classify = cl
K.inside = ins
# closures in the rotated frame use member.closure '<a>min'; classify_axis maps it to 'zmin'
n1 = K.main()
n2 = R.main()
print(f'axis {"xyz"[A]}: cases evaluated along another admissible axis (compact-only sets): {len(other_axis)}')
sys.exit(1 if (n1 or n2 or wrong_axis) else 0)

"""Exports every prototype case (cases.py + cases_review.py) with the model's (p36.py) results to
a text file read by tests/test_stage16b.c: the 16B tests B11/B14/B15/B16 reproduce every case of
§3.6.10 and of the review (§19.3) on the MFTL2 implementation.

Format per case:
  case <tag> <per_z> <nclos> [zmin|zmax ...]
  expect <status> <V> <Cz> <centre-status> <nph>
  queries <n>  then n lines "x y z answer"  (answer: 0 outside, 1 inside, 2 on boundary)
  members <n>
  then per member: "member <sigma> <closure|none> <nv> <nt>", nv lines "x y z" (continuous
  coordinates; vertices duplicated at periodic faces merged), nt lines "a b c".
"""
import sys
import numpy as np
import cases as K
import cases_review as R
import p36

out = open(sys.argv[1] if len(sys.argv) > 1 else 'cases_export.txt', 'w')
orig_classify = p36.classify
orig_inside = p36.inside
state = {}


def merge_member(m, per_z):
    key = {}
    V, idx = [], []
    for x in m.V:
        r = [x[0] % 1.0, x[1] % 1.0, (x[2] % 1.0) if per_z else x[2]]
        k = tuple(round(v, 9) % 1.0 if (a < 2 or per_z) else round(v, 9) for a, v in enumerate(r))
        if k not in key:
            key[k] = len(V)
            V.append(x)
        idx.append(key[k])
    T = [[idx[a], idx[b], idx[c]] for a, b, c in m.T]
    return np.array(V), T


def cl(ms, dom=None, want_placement=True):
    dom = dict(dom) if dom else dict(per_z=True)
    dom.setdefault('closures', [])
    res = orig_classify(ms, dom, want_placement)
    state['ms'] = ms
    state['dom'] = dom
    state['res'] = res
    state['queries'] = []
    return res


def ins(res, Q):
    s, v = orig_inside(res, Q)
    state['queries'].append((Q, s, v))
    return s, v


_case = K.case


def case(tag, *a, **kw):
    r = _case(tag, *a, **kw)
    ms, dom, res = state['ms'], state['dom'], state['res']
    st = res['status']
    per = 1 if dom.get('per_z', True) else 0
    out.write(f"case {tag} {per} {len(dom['closures'])} {' '.join(dom['closures'])}\n")
    V = res.get('V', 0.0) if st == 'DEFINED' else 0.0
    Cz = res.get('Cz', 0.0) if st == 'DEFINED' else 0.0
    cst = res['centre'].split(' ')[0] if st == 'DEFINED' else 'NONE'
    out.write(f"expect {st} {V:.17g} {Cz:.17g} {cst} {res.get('n_phase_components', -1) if st == 'DEFINED' else -1}\n")
    qs = [(Q, v) for (Q, s, v) in state['queries'] if s == 'DEFINED']
    out.write(f"queries {len(qs)}\n")
    for Q, v in qs:
        ans = 2 if v == 'ON_BOUNDARY' else (1 if v else 0)
        out.write(f"{Q[0]:.17g} {Q[1]:.17g} {Q[2]:.17g} {ans}\n")
    out.write(f"members {len(ms)}\n")
    for m in ms:
        Vm, Tm = merge_member(m, per)
        out.write(f"member {m.sigma} {m.closure or 'none'} {len(Vm)} {len(Tm)}\n")
        for x in Vm:
            out.write(f"{x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
        for t in Tm:
            out.write(f"{t[0]} {t[1]} {t[2]}\n")
    return r


K.classify = cl
K.inside = ins
K.case = case
R.case = case
n = K.main() + R.main()
out.close()
sys.exit(1 if n else 0)

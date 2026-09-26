"""Run every case with the wrap flags DERIVED from the §3.3 winding (not declared)."""
import sys
import cases as K
import cases_review as R
import p36
from winding import winding
orig = p36.classify
mism = []
def cl(ms, dom=None, want_placement=True):
    per = (True, True, (dom or {}).get('per_z', True))
    for m in ms:
        f, _ = winding(m, per)
        if tuple(f) != tuple(m.wrap):
            mism.append((m.name, m.wrap, f))
        m.wrap = f
    return orig(ms, dom, want_placement)
K.classify = cl
n1 = K.main()
n2 = R.main()
print('flag mismatches declared vs derived:', mism)
sys.exit(1 if (n1 or n2 or mism) else 0)

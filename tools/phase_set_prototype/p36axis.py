"""§3.6.3 evaluation-axis selection for the model: closures -> k = w; else the smallest
non-periodic axis; else the smallest periodic axis that no member wraps; none -> UNSUPPORTED.
The chosen axis k is mapped onto the model's z by the cyclic (orientation-preserving)
permutation (i, j, k) = (k+1, k+2, k); lateral axes must be periodic in the model."""
import numpy as np
import p36
from p36 import Member, fail


def cyc(k):
    return [(k + 1) % 3, (k + 2) % 3, k]


def to_frame(m, perm):
    V = m.V[:, perm]
    mm = Member(V, m.T.copy(), m.sigma, m.name, wrap=tuple(m.wrap[p] for p in perm), closure=m.closure)
    return mm


def classify_axis(members, per3, closures=()):
    # precedence (§3.6.3): Phi(S) != 0 -> UNDEFINED is decided before the axis choice; without
    # wall closures Phi is the sigma-weighted area vector of the members
    if not closures:
        F = sum((m.sigma * p36.own_integrals(m)[0] for m in members), np.zeros(3))
        if np.any(np.abs(F) >= p36.TAU_PHI):
            return fail('UNDEFINED', 'Phi(S) != 0', Phi=F), None
    axes = {a for a, _ in closures}
    if len(axes) > 1:
        return fail('UNSUPPORTED', 'wall closures on two axes'), None
    if axes:
        k = axes.pop()
    else:
        nonper = [a for a in range(3) if not per3[a]]
        if nonper:
            k = nonper[0]
        else:
            cand = [a for a in range(3) if not any(m.wrap[a] for m in members)]
            if not cand:
                return fail('UNSUPPORTED', 'no admissible axis (every periodic axis wrapped by some member)'), None
            k = cand[0]
    perm = cyc(k)
    if not (per3[perm[0]] and per3[perm[1]]):
        return fail('NOT-IN-MODEL', 'walled lateral axis'), k
    ms = [to_frame(m, perm) for m in members]
    for m in ms:
        if m.closure is not None:
            m.closure = 'z' + m.closure[1:]
    dom = dict(per_z=per3[k], closures=['z' + s for (_, s) in closures])
    return p36.classify(ms, dom), k

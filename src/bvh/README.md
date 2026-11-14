`src/bvh/` — LBVH and contact detection {#mftl_mod_bvh}
======================================

**Public header:** `include/mftl_bvh.h` · **Layer:** 3 ·
**Module group:** @ref mftl_bvh

| File | Job |
|---|---|
| `bvh_build.c` | primitive access, Morton keys, radix split, bottom-up fit, refit-only path |
| `bvh_traverse.c` | dual-tree traversal over an explicit stack, batched leaf phase, the ensemble pipeline |
| `bvh_predicates.c` | Shewchuk-adaptive `orient3d`, exact triangle–triangle intersection (Guigue–Devillers, coplanar, zero-area), the distance terms' entry points |
| `bvh_internal.h` | outward float rounding, box overlap, the two module-private build entry points |

@warning `bvh_predicates.c` **must** be compiled with `-ffp-contract=off` (set in
         `CMakeLists.txt`); see @ref bvh_fma.

Contact detection is a **broad phase** (a tree of boxes: which triangle pairs
deserve a closer look), a **narrow phase** (do they intersect — decided by exact
signs — or is their distance, accurate to rounding, within the envelope), and, on a distributed mesh, the **spatial halo**
(@ref mftl_mod_communication) that brings the other side of a cross-rank contact to the
rank that must report it.

[TOC]

---

## The ensemble pipeline {#bvh_overview}

```
MFTLEnsembleIntersectObjects(ens, params, callback, user_data)      COLLECTIVE
├─ n_bodies_global < (self_pairs ? 1 : 2) ?  ─▶ return   (GLOBAL count: all
│                                                          ranks must agree)
├─ 1  MFTL_GenerateLBVH per body THIS RANK HOLDS                     local
├─ 2  MFTLSpatialHaloResolveRadii: per-body reduction over the
│     holders + (gid, radius) to the 26 neighbours              collective
│     MFTL_BVHSetGlobalRadius per tree (refit only)                  local
├─ 3  MFTLExtendSpatialHaloNearBodies: extents + envelopes out,
│     patches and candidate pairs back                           collective
├─ 4  per directory entry: MFTL_BVHSetHalo (rebuild), radius last    local
├─ 5  per candidate pair: MFTL_IntersectObjects                      local
└─ 6  destroy trees and halo
```

- **The radii come first.** The envelope is a fraction of the body, and a body's
  size is only known collectively; this rank's share would make the detection
  distance depend on the process count (`MFTL_BVH::radius_hint`).
- **One exchange for the whole ensemble.** Its collectives are two
  neighbourhood allgathers whose number does not depend on the pairs.
- **Pairs are derived, not enumerated.** The envelopes travel with the extents,
  and a uniform grid over the arrivals yields the candidates. Six well-separated
  touching couples (6 of 66 pairs real): 6 / 6 / 9 / 18 / 6 candidates over all
  ranks at np = 1 / 2 / 4 / 8 / 27, against 66 / 132 / 264 / 528 / 1782 for all
  pairs (`test_the_broad_phase_finds_the_pairs_without_enumerating_them`).
- **Trees are `O(local bodies)`.** A body this rank holds nothing of but received
  a patch for gets a tree over a shared empty object (all primitives resolve
  through the patch) — which is why `MFTL_GenerateLBVH` returns an *empty*
  hierarchy rather than `MFTL_NULL` for an empty partition.
- **Self-proximity** (`self_pairs`) tests a body's owned triangles (A side, zero
  envelope) against its tree *with* patch (B side). A zero A-side envelope makes
  the broad phase admit $\tau_B$ instead of $\tau_A+\tau_B$; with both inflated
  a 17k-triangle tube produced 129 candidates per triangle.

---

## Construction {#bvh_build}

### Primitives {#bvh_prims}

The BVH stores no geometry; it indexes a primitive set

$$
p \in [0, n_\text{local}) \mapsto \text{triangle } p \text{ of } \texttt{obj},
\qquad
p \in [n_\text{local}, n) \mapsto \text{triangle } p - n_\text{local} \text{ of } \texttt{halo}.
$$

Both kinds are read through `MFTL_BVHPrimVerts`, which gathers the current
corners with minimum-image edges, so the three corners are always mutually
unwrapped and "local or remote" is one index comparison.

### Morton key {#bvh_morton}

The key of a primitive is the Morton code of the centroid of its
**seam-consistent** corners, deliberately *not* wrapped back into the domain:
a wrapped centroid is right for "where is this triangle" (redistribution routes
by it) but as a sort key puts two neighbours across a seam at opposite ends of
the key range. With $[\ell_k, h_k]$ the centroid box and **one** scale
$s = \max_k (h_k-\ell_k)$ for all axes (cubic cells, as in the pool
renumbering),

$$
q_{ik} = \Bigl\lfloor \frac{c_{ik}-\ell_k}{s}\,(2^{21}-1) \Bigr\rfloor,
\qquad
Z_i = \sum_{b=0}^{20} \bigl( q^x_{i,b}\,2^{3b} + q^y_{i,b}\,2^{3b+1} + q^z_{i,b}\,2^{3b+2} \bigr),
$$

the same 63-bit `MFTLMorton3D` key as `MFTLObjectReorderMorton`. Any finite
$s > 0$ is valid; there is no absolute floor. Keys and indices are sorted
together by `MFTLSortPairs64`, a **stable** radix sort, so the tree is a
deterministic function of the input (`test_refit_only_matches_full_rebuild`
compares trees node by node).

### Radix split {#bvh_radix}

The tree is a binary radix tree over the sorted keys (Lauterbach et al. 2009,
with Karras' 2012 uniqueness augmentation). A node covering sorted slots
$[f, l]$ splits where the highest differing key bit changes. With the
common-prefix length

$$
\delta(i,j) =
\begin{cases}
\operatorname{clz}_{64}(Z_i \oplus Z_j), & Z_i \ne Z_j \\
64 + \operatorname{clz}_{32}(i \oplus j), & Z_i = Z_j
\end{cases}
$$

the split is $\gamma = \max\{m \in [f,l) : \delta(f,m) > \delta(f,l)\}$, found by
binary search because $\delta(f,\cdot)$ is non-increasing over sorted keys;
the children are $[f,\gamma]$ and $[\gamma+1,l]$.

- **Why the index is part of the key.** Two triangles in one Morton cell share a
  code — common for a finely resolved interface — and equal keys never split.
  Comparing $(Z_i, i)$ lexicographically makes all keys unique without another
  sort and bounds the depth by **95**.
- **The split is a plane.** The first differing bit at position
  $\pi = 63-\delta$ belongs to axis $\pi \bmod 3$ at level $\lfloor \pi/3 \rfloor$,
  so the children are separated by a dyadic plane of the Morton grid — no
  partitioning pass is needed.
- **Node budget.** $n$ leaves give at most $2n-1$ nodes; the array is allocated
  at $2n$ and never grown. Leaves hold at most `leaf_size` (default 4)
  primitives.

### Fit and refit {#bvh_refit}

<!-- figure: lbvh
The LBVH over ten primitives, in 2D for legibility: Morton keys of 3 bits per
axis, sorted stably. (a) Every node's box, coloured by depth. A split is taken
at the highest differing key bit, so the children are separated by a dyadic
plane of the Morton grid. (b) The radix tree over the sorted keys, each node
labelled with the axis of its splitting bit. Two primitives share a key; the
index in the extended key splits them, so every node splits.
-->

`MFTLBVHRefit` is three linear passes, shared by the build and by
`MFTL_RefitLBVH` so they cannot disagree:

1. **Reverse scan** of the node array: leaves fit their primitives, internal
   nodes merge their children. Valid because a child pair is allocated after its
   parent (`left(i) > i`, `right(i) = left(i)+1`), so array order is a reverse
   topological order — no stack, no recursion.
2. The root box gives `bounding_radius`, then `tolerance` (@ref bvh_envelope).
3. **Forward scan** inflating every node by $\tau$. With
   $G_\tau(B) = [B^{\min}-\tau, B^{\max}+\tau]$ and $\oplus$ the min/max merge,
   $G_\tau(B_1 \oplus B_2) = G_\tau(B_1) \oplus G_\tau(B_2)$, so inflating the
   nodes afterwards equals inflating the leaves first — without knowing the
   radius in pass 1.

`MFTL_BVHSetGlobalRadius` and `MFTL_RefitLBVH` re-enter at the fit: neither the
order nor the shape depends on the tolerance or on vertex motion.

### Refit or rebuild {#bvh_refitguard}

`MFTL_RefitLBVH` returns `MFTL_ERROR` (not an assertion — "refit if you can,
otherwise rebuild" is supported) when a primitive count changed. Leaves address
primitives by index; after a re-triangulation the indices name different
triangles and a refit would report **wrong** contacts, not missing ones.

### Attaching a halo rebuilds {#bvh_setphalo}

`MFTL_BVHSetHalo` rebuilds the whole tree over local plus received primitives.
The Morton order is a function of the whole set, so leaves cannot be appended to
a tree over the local part: the result would still terminate and still be
fitted, and would silently miss contacts.

### Node layout {#bvh_node}

32 bytes: six `float` box bounds and a `uint64_t` payload.

| | bit 0 | bits 1–31 | bits 32–63 |
|---|---|---|---|
| leaf | 1 | primitive count | first slot in `prim_indices` |
| internal | 0 | left child (right = left + 1) | — |

Boxes are rounded **outward** to float: a round-to-nearest cast can shrink a box
by an ulp and exclude a vertex lying on its own leaf's face. With
$\epsilon_s = 2.4\cdot10^{-7} \approx 2^{-22}$ and $\eta = 10^{-30}$,

$$
\underline f(v) = \mathrm{fl}(v)(1 \mp \epsilon_s) - \eta \le v \le
\overline f(v) = \mathrm{fl}(v)(1 \pm \epsilon_s) + \eta
$$

(upper sign for $v \ge 0$), which leaves at least one float ulp of margin after
both roundings; $\eta$ keeps the rounding outward through zero and the
denormals. A float box is thus a conservative superset of the double one — the
only direction in which a broad phase may err.

---

## Periodicity is two problems {#bvh_periodic}

**Within an object**, a triangle straddling a seam has corners on both sides of
the box, so raw coordinates give a centroid in the middle of the domain and a
domain-spanning box. Fixed by `MFTL_BVHPrimVerts` (minimum-image edges) and the
unwrapped centroid (`test_periodic_morton_of_a_straddling_object`). A straddling
object's *root* box legitimately spans the domain.

**Between objects**, two bodies at opposite faces are neighbours although a
domain length apart in stored coordinates; no per-triangle arithmetic can see
it. The pair is traversed once per image of B,

$$
\mathcal S = \prod_{k} S_k,\qquad
S_k = \begin{cases}\{0, -L_k, +L_k\} & \text{axis } k \text{ periodic} \\ \{0\} & \text{otherwise}\end{cases}
\qquad |\mathcal S| \le 27,
$$

each gated by one root-box test (`test_contact_across_a_periodic_face`). The
shift is added to the B side inside `flush_batch`, because it belongs to the
image being traversed, not to the primitive.

<!-- figure: periodic-images
(a) Between bodies, periodicity is handled by traversing the pair once per
image of $B$, each image gated by one root-box test; only the image adjacent to
$A$ across the seam is traversed. (b) Each tree inflates its leaves by its own
$\tau$, so the broad phase admits gaps up to $\tau_A+\tau_B$ while the narrow
phase judges the pair at $\max(\tau_A,\tau_B)$. No contact the narrow phase
accepts can be culled; the pair drawn is admitted by the boxes and rejected by
the narrow phase.
-->

---

## The contact envelope {#bvh_envelope}

$$
\tau = \max\bigl(\texttt{proximity\_floor},\ \texttt{proximity\_factor}\cdot R\bigr),
\qquad
R = \tfrac12 \max_k \bigl(\text{root}^{\max}_k - \text{root}^{\min}_k\bigr).
$$

- **Relative, not absolute.** An absolute gap is correct at one length scale
  only; a run with millimetre droplets and a metre-scale body has no single
  right number (`test_detection_at_two_disparate_length_scales`).
- **$R$ is half the longest side**, not half the diagonal, so $R$ equals the
  radius of a sphere and `proximity_factor` means what it says (the diagonal
  would inflate every envelope by $\sqrt3$).
- **$R$ is global.** The root box covers one rank's share; the pipeline replaces
  it with the body's global radius (`MFTL_BVHSetGlobalRadius`, from the halo's
  per-body holder reduction), so the detection distance does not depend on the
  partition (`test_tolerance_is_independent_of_the_partition`).
- **One $\tau$ inflates the leaves and is the narrow-phase gap.**

A pair is judged at $\tau_{AB} = \max(\tau_A, \tau_B)$. The broad phase admits
every pair with gap $g \le \tau_A + \tau_B$ (each tree carries its own
inflation), and $\max(\tau_A,\tau_B) \le \tau_A + \tau_B$, so **no contact the
narrow phase accepts can be culled**. The price: a µm droplet against a metre
body is judged at the metre scale, and the envelope can exceed the droplet's
diameter. For a physical gap set `proximity_factor = 0` and
`proximity_floor = gap` (`test_two_length_scales_share_the_larger_envelope`).

---

## The traversal {#bvh_traverse}

`MFTL_IntersectObjects` walks the two trees with an **explicit stack** of node
pairs (initial depth 256, doubling): the loop-invariant state stays in
registers, a pathological pair cannot overflow the call stack (a dual
traversal is not tail-recursive), and leaf pairs can be batched.

```
pop (a, b) ─▶ boxes overlap (with the image shift)?  no ─▶ next
            ├─ both internal: split the one with the LARGER box volume
            ├─ one leaf: split the other
            └─ both leaves: A-side primitive owned here? no ─▶ next
                            append (pA, pB) to the batch; flush at 64
```

### Descent {#bvh_descent}

Splitting the node with the larger volume $V = \prod_k (n^{\max}_k - n^{\min}_k)$
makes the pair of boxes converge fastest.

### Who reports a contact {#bvh_ownership}

A distributed contact is reported by the **owner of the A-side primitive**, and
only by it:

- trees are built over **owned** triangles (`include_ghosts = false`); with the
  redistribution halo included, two ranks would report the same pair;
- only B-side triangles travel in the spatial halo, and a leaf pair whose A side
  is a halo primitive is skipped (one compare);
- the candidate list is ordered (`body_a < body_b`), only the larger id travels,
  and duplicates are removed, so every unordered pair is traversed once.

Wrong in one direction, cross-rank contacts are missed; in the other, every
cross-rank contact force is applied twice.
`test_distributed_contacts_match_the_serial_answer` and
`tests/test_bvh_ghosting.c` compare the gathered **set** against the answer on
the unpartitioned mesh, which rules out both.

### The batch {#bvh_batch}

`flush_batch` receives up to 64 independent $(p_A, p_B)$ tests — the seam for a
SIMD narrow phase, which would change that one function. It is scalar today.

---

## The narrow phase {#bvh_narrow}

`MFTL_TriTriProximity(A, B, τ)` returns 1 if the triangles intersect or their
distance is at most $\tau$:

```
MFTL_TriTriIntersect ─▶ 1 ?  yes ─▶ contact                      (exact)
          │ no
          └─ τ > 0 and  min over 9 edge–edge and 6 point–face  d² ≤ τ² ─▶ contact   (to rounding)
```

The intersection test must come first: an edge piercing the interior of the
other face has positive distance to every edge and vertex, so the fifteen
distance terms cannot see it. Measured: 31 ns/pair for the intersection test,
244 ns/pair for the full proximity test (176 ns before the distance terms were
made accurate to rounding, @ref bvh_distance).

### The orientation predicate {#bvh_orient}

$$
\operatorname{orient3d}(\mathbf a,\mathbf b,\mathbf c,\mathbf d)
= (\mathbf a-\mathbf d)\cdot\bigl[(\mathbf b-\mathbf d)\times(\mathbf c-\mathbf d)\bigr],
$$

six times the signed volume of the tetrahedron; its sign says on which side of
the oriented plane through $\mathbf b,\mathbf c,\mathbf d$ the point
$\mathbf a$ lies. The **query point comes first**. Plane-first order is a cyclic
4-permutation — odd — and negates every sign; the early rejection survives that
(it compares products) but the dispatch takes the mirror branch and reports a
clean miss for crossing triangles (`test_tri_tri_intersect_basic_cases`).

### Guigue–Devillers {#bvh_gd}

(Guigue & Devillers, J. Graphics Tools 8(1), 2003.) With $d_{p_1}, d_{q_1},
d_{r_1}$ the orientations of A's corners against B's plane and $d_{p_2}, d_{q_2},
d_{r_2}$ of B's against A's:

1. **Early rejection:** $d_{p_1}d_{q_1} > 0 \wedge d_{p_1}d_{r_1} > 0$ (all of A
   strictly on one side) → disjoint after **3** predicates; likewise for B after 6.
2. Otherwise the planes meet in a line $L$, each triangle cuts $L$ in a closed
   interval, and the triangles intersect iff the intervals overlap. The dispatch
   permutes A so that $p_1$ is its isolated corner and B so that its
   orientation relative to A is fixed — branching on signs already computed —
   after which (`check_min_max`)

$$
T_1 \cap T_2 \ne \emptyset \iff
\operatorname{orient3d}(q_2,p_2,p_1,q_1) \le 0 \ \wedge\
\operatorname{orient3d}(r_2,p_2,r_1,p_1) \le 0 .
$$

At most **eight** predicates per pair, no division and no constructed point.
Normalising only which vertex is isolated, and not B's orientation, gets the
final test wrong on half the inputs.

<!-- figure: guigue-devillers
Guigue–Devillers. (a) The sign of `orient3d` for each corner against the other
triangle's plane (coral) rejects most pairs at once. Otherwise the two planes
meet in a line $L$ and each triangle cuts a closed interval from it. (b) The
triangles intersect iff the intervals overlap, which two more orientation signs
decide: no division, no constructed point.
-->

### Pairs Guigue–Devillers cannot take {#bvh_coplanar}

Guigue–Devillers assumes two proper triangles in distinct planes. Three exact
zeros among $d_{p_1}, d_{q_1}, d_{r_1}$ (or among B's) mean that is not the
case: A lies in B's plane — or B has zero area, because every point is "in" the
plane of a degenerate triangle. Those pairs are recognised before the dispatch
and decided by exact signs as well:

| Pair | Test |
|---|---|
| two proper triangles, coplanar | 2D, in a coordinate projection |
| a zero-area triangle and a proper one | the segment it spans crossing the triangle; in its plane, 2D |
| two zero-area triangles | two segments: disjoint unless coplanar, then 2D |

**Zero area is decided exactly**: three points are collinear iff all three
coordinate projections have zero signed area. The hull of collinear corners is
the segment between the extremes along any axis on which they differ. A segment
$[\mathbf s_0, \mathbf s_1]$ that is not in a proper triangle's plane and does
not have both ends strictly on one side of it meets the plane in one point $X$;
$\operatorname{orient3d}(\mathbf t_i, \mathbf t_{i+1}, \mathbf s_1, \mathbf s_0)$
does not depend on where along the segment's line it is evaluated, equals
$(\mathbf s_1-\mathbf s_0)\cdot\mathbf n$ times the in-plane orientation of $X$
against edge $i$, and $X$ is in the closed triangle iff the three signs are not
mixed.

**The 2D test** covers hulls of any shape (a triangle, a segment, a point): two
closed convex sets meet iff their boundaries do — nine segment pairs, each by
strict crossing or an endpoint on the other segment (a zero orientation and its
bounding box) — or one contains the other, possible only in a hull of non-zero
area and tested with one corner. The 18 edge orientations serve both. **The 2D
orientation is exact**: $\operatorname{orient2d}(a,b,c)$ is
$-\operatorname{orient3d}$ of $(a,0)$, $(b,0)$, $(c,0)$ against $(c,1)$, the same
determinant, so its sign is Shewchuk's. A plain 2D determinant rounded a small
orientation to zero, took a corner a rounding error off an edge to be on it, and
reported two disjoint triangles as touching, at distance zero
(`test_coplanar_triangles_are_decided_exactly`).

**Which projection.** Dropping an axis preserves intersection, and preserves
disjointness when it is one-to-one on the common plane — certified exactly by a
projected triangle of non-zero area. Some axis is one-to-one on the plane (on the
line, if everything is collinear), so: a disjoint projection proves disjoint, a
certified one proves intersecting, and three intersecting uncertified projections
prove intersecting. The axis the larger normal is most aligned with goes first,
so a proper coplanar pair costs one projection.

Before, a zero-area triangle fell through to a 2D test in an arbitrary projection
with its zero normal: a point five units away was reported as intersecting, and
two collinear triangles a unit apart as touching
(`test_zero_area_triangles_are_segments`).

### Distance between disjoint triangles, to rounding {#bvh_distance}

The distance between two disjoint closed triangles is attained between two edges
or between a vertex and the interior of the other face, so

$$
d^2(T_A,T_B) = \min\Bigl( \min_{i,j} d^2_\text{ee}(e^A_i, e^B_j),\
\min_i d^2_\text{pf}(a_i, T_B),\ \min_i d^2_\text{pf}(b_i, T_A) \Bigr)
$$

is a complete cover. The terms are `MFTLClosestPointsBetweenTriangles`
(src/geometry/, @ref geom_closest_accuracy), the routine topology detection reads
its witnesses from, so contact and topology change agree on what a gap is.
Each term is a small least-squares problem, and it is NOT solved through its
$2\times2$ normal equations: the Gram determinant $D = ac-b^2$ of two edges at an
angle $\theta$ cancels to a relative error $\varepsilon/\sin^2\theta$, and a
"parallel" threshold $D \le 10^{-15}ac$ then pinned the parameter to an end. At
an angle of $2\cdot10^{-8}$ a gap of $10^{-9}$ read as $10^{-8}$, and a query at
tolerance $2\cdot10^{-9}$ reported no contact; a corner over a 1:1000 sliver read
$6\cdot10^{-5}$ too far, over a 1:$10^5$ one 26 times
(`test_nearly_parallel_edges_are_in_contact`; in `tests/test_geometry.c`
`test_point_face_distance_is_accurate_on_slivers` and the double-double
reference). With the determinant written as
$|\mathbf u\times\mathbf v|^2$, triple-product numerators and a frame
orthogonalised on a triangle's longest edge, the error is at most
$1.5\,\varepsilon L$ over $2\times10^5$ random near-parallel and sliver pairs in
general position ($L$ the pair's diameter), against $10^{10}\varepsilon L$
before. The intersection verdict is exact; the distance is a few ulps of $L$, so
only a gap within rounding of the tolerance can go either way.

#### No absolute thresholds {#bvh_dimensionless}

$D$ has units of length⁴, so an absolute guard (`D < 1e-12`) fired for every
triangle of a mesh at 20 µm and silently removed all point–face terms — exactly
the vertex-into-face configuration of a resolved contact. The terms now have no
threshold at all: a segment is a point only at exactly zero length, two segments
are parallel only at $\mathbf u\times\mathbf v = \mathbf 0$, and a triangle loses
its face term only when its corners are numerically collinear.

### Why only orient3d is adaptive {#bvh_adaptive}

It is the only predicate whose **sign** decides the answer (the 2D orientations
are this predicate on lifted points); a wrong sign takes a branch for a
configuration that does not exist and typically reports a clean miss for
crossing triangles. Distances degrade continuously and only move decisions
sitting on the tolerance — once they are computed without the normal equations
(@ref bvh_distance). The determinant
is ill-conditioned exactly where this library works: thin films (four points
coplanar to within the gap) and small features at large coordinates (the
differences cancel the offset).

#### The filter {#bvh_escalation}

Shewchuk's adaptive predicate (Discrete Comput. Geom. 18, 1997) evaluates the
plain determinant and its **permanent** (with $\mathbf a' = \mathbf a-\mathbf d$
etc.)

$$
\text{perm} =
\bigl(|b'_xc'_y| + |c'_xb'_y|\bigr)|a'_z| +
\bigl(|c'_xa'_y| + |a'_xc'_y|\bigr)|b'_z| +
\bigl(|a'_xb'_y| + |b'_xa'_y|\bigr)|c'_z| .
$$

If $|\det| > \varepsilon_A\,\text{perm}$ the sign is certain — the path nearly
every pair takes. Otherwise two correction stages with bounds $\varepsilon_B$,
$\varepsilon_C$ follow, and finally exact expansion arithmetic:

$$
\varepsilon_A = (7+56\varepsilon)\varepsilon,\quad
\varepsilon_B = (3+28\varepsilon)\varepsilon,\quad
\varepsilon_C = (26+288\varepsilon)\varepsilon^2,
$$

with the machine $\varepsilon$ **measured** once by `exactinit()` (halving until
$1+\varepsilon = 1$), so the bounds are true for the actual arithmetic.

#### Error-free transformations {#bvh_eft}

The exact stages use operations returning a rounded result and its exact
residue, $x + y = a \circ b$ exactly: `Two_Sum` ($y = (a+b)-x$ by Knuth's
branch-free formula) and `Two_Product`, whose residue is assembled from
Dekker's split $a = a_\text{hi} + a_\text{lo}$ (at most 26 significant bits
each, so every partial product is exact). An expansion is a
magnitude-ordered, non-overlapping list of such components whose sum is the
exact value.

#### `-ffp-contract=off` is correctness {#bvh_fma}

`Two_Product` relies on $x - a_\text{hi}b_\text{hi}$ being rounded separately.
A fused multiply-add contracts it into one rounding, the residue comes out zero,
and every guarantee silently disappears while the results still look plausible.
The file carries `#pragma STDC FP_CONTRACT OFF`, and `CMakeLists.txt` sets the
flag for compilers that ignore the pragma. **Do not remove either.**

#### Cost and benefit {#bvh_orientcost}

Measured by `examples/orient3d_predicate_profile.c` (20 000 samples per row, the
adaptive result as exact reference):

| gap / triangle size | plain determinant misclassified |
|---|---|
| $\ge 10^{-12}$ | 0 % |
| $10^{-14}$ | ≤ 0.035 % |
| $10^{-15}$ | ≤ 0.125 % |
| $10^{-16}$ | ≤ 4.7 % |

The adaptive form costs 3.66 ns against 1.83 ns in general position (40 ns only
for exactly coplanar input). A thin-film model works at gaps of $10^{-6}$ to
$10^{-9}$ of the body size, where the plain form would almost surely be right;
the adaptive one is kept because it costs at most 4.5 % of a
`MFTL_TriTriProximity` call and removes a **silent** failure.
`MFTL_Orient3DFast` is the plain form.

---

## Cost {#bvh_cost}

Only the narrow-phase numbers are measured; the rest is argued from structure,
and there is no BVH section in `docs/analysis.md` yet.

| Stage | Work |
|---|---|
| keys + stable radix sort | $O(n)$ |
| radix split | $O(n)$ nodes, binary search per split, depth ≤ 95 |
| fit / refit | $O(\text{nodes} + n)$, three linear scans |
| periodic images | ≤ 27 root-box tests |
| `MFTL_TriTriIntersect` | 3–8 `orient3d`; **31 ns/pair** |
| `MFTL_TriTriProximity` | the above + 15 distance terms; **244 ns/pair** (176 with the normal-equation terms) |

**Memory:** 32 B per node with $2n$ nodes reserved (64 B/primitive, about
16 B/primitive used at `leaf_size` 4), plus 4 B/primitive of `prim_indices`.
Both objects are borrowed `const`.

**Defaults** (`MFTL_BVHDefaultParams()`): `proximity_factor` $10^{-3}$,
`proximity_floor` 0, `leaf_size` 4, `include_ghosts` false, `self_pairs` false,
`pair_filter` NULL.

---

## Open issues {#bvh_deferred}

1. **The batched narrow phase is not vectorised.** `MFTL_TriTriIntersect` is a
   decision tree with early exits, so the useful form is a gather of the first
   orientation test over the batch; start with the fifteen distance terms, where
   the time is.
2. **Build, refit and traversal have not been profiled.**
3. **Detection is discrete in time.** Two bodies whose relative displacement in
   one step exceeds the sum of their sizes can pass through each other
   unobserved; continuous detection would need swept primitives and a velocity.
   `test_tunnelling_between_two_timesteps_is_a_known_miss` asserts the miss.

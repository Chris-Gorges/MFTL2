`src/geometry/` — Periodic-aware geometry {#mftl_mod_geometry}
=========================================

**Public header:** `include/mftl_geometry.h` · **Layer:** 2 ·
**Module group:** @ref mftl_geometry

Everything in this directory is **local** — no MPI:

| File | Produces |
|---|---|
| `triangle_geometry.c` | `tris.area`, `tris.normal`, `tris.center` for a mesh; closest points on a triangle, between two segments and between two triangles |
| `vertex_geometry.c` | area-weighted, angle-weighted or fitted vertex normals, for the whole pool (`verts.normal`) or one vertex |
| `signed_distance.c` | signed distance to a mesh on a (periodic) lattice window, at one cutoff or two |
| `periodic_extent.c` | a body's extent on a periodic axis, and whether it covers the axis |
| `body_integrals.c` | a body's flux integrals (volume, first moments) and where its rims lie, per rank |
| `vertex_curvature.c` | the SIGNED curvature per vertex with the Gaussian, the principal curvatures and direction, by Meyer et al.'s operators or a one-ring fit (the remesher's sizing reads the larger principal curvature magnitude off it) |
| `surface_tension.c` | the integrated surface-tension force per vertex or per triangle (Tryggvason's Frenet forms) |
| `surface_fit.c` | small dense least squares (QR with grouped column pivoting) and the weighted height-paraboloid fit |

The collective wrappers (`MFTLEnsembleUpdate`, `MFTLEnsembleUpdateVertexNormals`,
`MFTLEnsembleComputeCurvature`, `MFTLEnsembleComputeSurfaceTension`) live in
@ref mftl_mod_ensemble; the split is by *scope*, not by entity
(@ref arch_geometry_split). Small queries (`MFTLGetMinimumImageVector`,
`MFTLGetTriangleProperties`, `MFTLGetTriangleCorners`, `MFTLGetOwnerProc`) are
`static inline` in the header.

---

## The minimum-image convention {#geom_minimum_image}

On a periodic axis the domain is a torus of length $L_k$, and the difference of
two stored coordinates is not the vector between the points. The shortest
representative is

$$
  \Delta_k = (p_2 - p_1)_k - L_k\,\mathrm{nearbyint}\!\left(\frac{(p_2-p_1)_k}{L_k}\right),
  \qquad |\Delta_k| \le L_k/2 ,
$$

evaluated with the precomputed reciprocal `inv_domain_size` — one multiply, one
round, one fused multiply-subtract, valid for any displacement (the comparison
form is valid only for $|d| < 3L/2$). Every edge vector in the library is formed
this way, so a triangle straddling a periodic face has correct edges although its
corners are stored on opposite sides of the box. This requires every edge to be
shorter than $L_k/2$, which any resolved surface satisfies.

<!-- figure: minimum-image
A triangle straddling a periodic seam. (a) Its corners are stored wrapped, so
raw differences span the domain and the centroid of the stored corners lies in
its middle. (b) Every edge is a minimum-image vector, so the triangle has its
true shape across the seam; the centroid is built from those offsets and then
wrapped, which makes it both the true centroid and a position
`MFTLGetOwnerProc` accepts.
-->

`MFTLApplyPeriodicity` wraps a position into $[\min_k, \max_k)$ by at most one
period — correct for positions wrapped every step. Its perfectly predicted
compare-and-shift form beat a branchless variant by 1.34–1.37×.

`MFTLGetOwnerProc` maps a *wrapped* position to the rank owning its Cartesian
cell, $i_k = \lfloor (x_k-\min_k)\,\mathrm{cuts}_k/L_k \rfloor$ clamped to the
grid, through the precomputed `grid_factors` and `rank_map`.

---

## Triangle area, normal and centroid {#geom_triangle}

With minimum-image edges $\mathbf{e}_1 = \mathbf{p}_1 - \mathbf{p}_0$,
$\mathbf{e}_2 = \mathbf{p}_2 - \mathbf{p}_0$:

$$
  \mathbf{n} = \mathbf{e}_1 \times \mathbf{e}_2, \qquad
  A = \tfrac{1}{2}\lVert \mathbf{n} \rVert, \qquad
  \hat{\mathbf{n}} = \frac{\mathbf{n}}{\lVert \mathbf{n} \rVert}, \qquad
  \mathbf{c} = \mathbf{p}_0 + \tfrac13(\mathbf{e}_1 + \mathbf{e}_2).
$$

One square root serves area and normal. The corners are ordered
counter-clockwise seen from outside, so $\hat{\mathbf{n}}$ points outward
(@ref mftl_page_invariants). The centroid is built from seam-corrected offsets
and then wrapped: it is the true centroid of a straddling triangle *and* a
position `MFTLGetOwnerProc` accepts.

`MFTLComputeTriangleGeometry` streams `tris.verts` when the derived
connectivity is current and walks the half-edge records otherwise; the two paths
are bit-identical (`test_triangle_geometry_paths_are_bit_identical`).

### A normal is always a unit vector {#geom_unit_normal}

Every kernel evaluates

```c
const double mag = MFTL_SQRT(nx*nx + ny*ny + nz*nz);   /* unclamped */
const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
```

Clamping $\lVert\mathbf{n}\rVert^2$ before the square root floors the magnitude
(at $10^{-6}$ for a $10^{-12}$ clamp) and with it the area, so a degenerate
triangle could not be reported and the quality
$q = 4\sqrt3A/(a^2+b^2+c^2)$ could exceed its maximum of 1; clamping the divisor
only moves the floor. Either way small triangles get *short* normals that still
point the right way — every sign test passes while surface forces are silently
scaled. The test against zero costs one instruction (+1.1 %) and is exact at
every scale ($10^{-8}\ldots10^{8}$ in `test_triangle_normals_are_unit_at_every_scale`).

---

## Vertex normals {#geom_vertex_normals}

One rule, `MFTLComputeVertexNormals(obj, weighting)` for the whole pool and
`MFTLComputeVertexNormalAt(obj, v, weighting, n)` for one vertex, and one choice
on two levels (`MFTL_NormalWeighting`): a normal **from the triangles**, a
weighted sum of the face normals around the vertex, weighted by area or by
angle; or a **fitted** normal.

$$
  \mathbf{N}_v^{\text{area}} \propto \sum_{f \ni v} A_f\,\hat{\mathbf{n}}_f
  = \tfrac12 \sum_{f\ni v} (\mathbf{e}_1\times\mathbf{e}_2)_f,
  \qquad
  \mathbf{N}_v^{\text{angle}} \propto \sum_{f \ni v} \alpha_f(v)\,\hat{\mathbf{n}}_f ,
  \qquad
  \mathbf{N}_v^{\text{fit}} \propto \mathbf n - c_1\mathbf t_1 - c_2\mathbf t_2 .
$$

| | `MFTL_NORMAL_WEIGHT_AREA` | `MFTL_NORMAL_WEIGHT_ANGLE` | `MFTL_NORMAL_WEIGHT_FIT` |
|---|---|---|---|
| what it is | the direction of the volume gradient | the pseudo-normal whose inside/outside sign is exact | the normal of a one-ring paraboloid |
| accuracy on an irregular mesh | first order | first order | second order |
| cost per vertex, whole pool (Release, Apple M4) | 6–7 ns | 40–66 ns | about 240 ns |
| the library uses it for | the remesher's smoother (its tangent plane), the frame and samples of its placement fit and its projection's orientation test; the volume corrections | the signed distance and topology change; the remesher's Phong reference | curvature (`MFTL_CURVATURE_FIT`) and the surface tension with vertex normals |

**Area weighting** is free (the raw cross product carries it) and parallel to the
gradient of the enclosed volume on a closed fan,

$$
  \frac{\partial V}{\partial \mathbf{x}_v} = \tfrac13\sum_{f\ni v} A_f\,\hat{\mathbf{n}}_f ,
$$

so it is the direction in which a vertex changes the volume fastest, and the
normal of the one plane in which a vertex moves without changing the volume to
first order (`test_the_area_normal_is_the_direction_of_the_volume_gradient`).
The volume corrections move vertices along it, and the remesher's smoother moves
them in that plane. There the choice is load-bearing: with any other vertex
normal every smoothing move leaks volume at first order, and the leak roughens
the front. Over the LeVeque period without back-projection, angle normals in the
smoother left the uniform refinement-4 front 25 % of the radius off the exact
surface with a worst triangle quality of 0.004, and Max or fitted normals made
the curvature 2–15 times noisier (docs/analysis.md §16.9).

**Angle weighting** (Thürmer–Wüthrich) is invariant under re-triangulation of the
fan and is the only weighting of face normals whose pseudo-normal sign is provably
right (@ref geom_sdf). The corner angle is
$\alpha = \operatorname{atan2}(\lVert\mathbf e_1\times\mathbf e_2\rVert,\, \mathbf e_1\cdot\mathbf e_2)$
from the cross product the face normal needs anyway
and one dot product: both carry the factor $|\mathbf e_1||\mathbf e_2|$, so no
edge is normalised, and `atan2` is well conditioned at every angle, where `acos`
of the normalised dot product loses digits near 0 and $\pi$. On jagged meshes,
of 2277 points whose nearest mesh feature is a vertex, angle weights classify
all correctly and area weights get 10 wrong
(`test_vertex_signs_on_a_spiky_mesh_need_angle_weighting`). As the reference
normals of the remesher's Phong projection it measured best or tied everywhere.

**Max's (1999) weights are not offered.** They are exact when the one-ring lies
on a sphere and the most accurate of the three weightings on remeshed and
marching-cubes meshes (1.6–8 times angle weights, RMS, table below), but they
have no sign property (14 wrong in the test above), they are first order like
the others on a general surface, the fit makes their accuracy moot, and as Phong
reference normals they measured tens of percent worse than angle weights
(docs/analysis.md §16.9). In the remesher's placement fit, which kept a
private Max sum until 2026-11-20, area weights measured best or tied over the
LeVeque period (Max's were 10–20 % better only in the curvature error at uniform
refinement 5), so the fit now uses the library's area normals (§16.10).

<!-- figure: vertex-normals
Normals. (a) One square root serves both $A = \tfrac12\lVert\mathbf{e}_1\times\mathbf{e}_2\rVert$
and $\hat{\mathbf{n}}$; the corners are counter-clockwise from outside, so
$\hat{\mathbf{n}}$ points out. (b, c) One uneven fan weighted by area (shade
$\propto A_f$) and by corner angle (wedges $\alpha_f$): here the two vertex
normals differ by 27°. The area weighting is the volume gradient; the angle
weighting is invariant under re-triangulation of the fan and is the
pseudo-normal the signed distance needs.
-->

The whole pool is a **scatter** over triangles (flat loop, compact connectivity)
rather than a gather over fans (dependent loads, measured 3.4× slower); a
vertex→triangle CSR gather was a dead heat and is not stored
(docs/analysis.md §1.5). `verts.normal` stays xyz-interleaved: the three-double
read-modify-write is four memory operations per vertex instead of six
(1.20–1.25×, `examples/layout_benchmark.c`). One vertex is that gather, a walk of
its fan both ways from `out_he_idx`, for a caller that edits the mesh between two
normals and would find a whole-pool pass stale after its first edit (the
remesher's placement fit). Both take the corner angle and the fit from the same
code; they differ only in the order of the sum
(`test_one_vertex_is_the_whole_pool_pass`, every weighting, across a periodic
corner and at a mesh rim).

### The fitted normal {#geom_fit_normal}

`MFTL_NORMAL_WEIGHT_FIT` is not a weighting of face normals. In the frame
$(\mathbf t_1, \mathbf t_2, \mathbf n)$ of the area-weighted normal, a height
paraboloid $z = c_0 + c_1x + c_2y + c_3x^2 + c_4xy + c_5y^2$ is fitted to the
vertex and its one-ring positions (`MFTLFitHeightParaboloid`, lengths in units of
the ring's RMS edge), and the normal is that of the fitted surface at the vertex,
$\mathbf N \propto \mathbf n - c_1\mathbf t_1 - c_2\mathbf t_2$. A vertex with
fewer than three neighbours, a fit that cannot determine its slope, or a fitted
normal more than 75° from the frame keeps the area normal. Which weighting
gives the frame does not change the fitted normal (area, angle and Max frames
agree to three digits, docs/analysis.md §16.8), so the frame is the cheapest.

Why it exists: any weighting of face normals is **first-order** accurate on an
irregular mesh (second order only where the fan is symmetric); the fit is second
order on any mesh whose vertices lie on the surface. RMS error in degrees at the
finest of five levels on six meshes of the kind a run produces (39 000 to 562 000
vertices; docs/analysis.md §16.8), Max weights for comparison:

| mesh | area | angle | Max | fit |
|---|---|---|---|---|
| ellipsoid 1 : 0.7 : 0.5, isotropically remeshed | 8.1e-2 | 4.6e-2 | 2.4e-2 | 6.8e-3 |
| ellipsoid, curvature-graded (edges 0.3–2 h) | 3.2e-2 | 2.2e-2 | 1.4e-2 | 4.6e-3 |
| ellipsoid, marching cubes | 0.27 | 0.15 | 1.9e-2 | 2.1e-3 |
| torus $R = 1$, $r = 0.6$, jittered | 0.14 | 6.5e-2 | 4.0e-2 | 1.8e-3 |
| torus, isotropically remeshed | 1.4e-2 | 8.4e-3 | 8.7e-3 | 1.6e-3 |
| torus, marching cubes | 0.24 | 0.13 | 3.5e-2 | 1.3e-3 |

A curvature read off the normals (the fit of @ref geom_curvature, the forces of
@ref geom_surface_tension) inherits the normals' order divided by $h$: first-order
normals give an error that does not vanish. The fit costs one small QR per vertex,
about 240 ns on top of the area frame.

### Exactness needs a vertex-complete halo

The scatter runs over **all** triangles, ghosts included, so an owned vertex is
exact iff every incident triangle is present locally (the fitted normal needs the
same, the one-ring positions) — which the redistribution
halo guarantees (a triangle is sent to the owner of each of its corners). This
makes the pass communication-free.

@warning Ghost vertices hold a normalised **partial** sum, which looks exactly
         like a real normal. `MFTLEnsembleUpdateVertexNormals` adds the one ghost
         exchange that makes them exact.

---

## Closest point and signed distance {#geom_sdf}

`MFTLClosestPointOnTriangle` implements Ericson's Voronoi-region test: dot
products select one of seven regions (face, three edges, three corners) and only
that region's formula is evaluated. It also returns the **feature**; edge $k$
is half-edge $3t+k$. The side-of-an-edge decisions and the face point come from
an orthogonalised frame, not from Ericson's determinants of dot products
(@ref geom_closest_accuracy). So do the closest point's **barycentric weights**,
an optional output: one-hot at a corner, the edge parameter on an edge, the
frame's coordinates on the face. Recovering them from the point afterwards
through the 2x2 normal equations would bring the Gram determinant back; the
remesher's Phong projection and its field transfer read these instead. A triangle with $\sin^2\theta \le 10^{-15}$ at
corner 0 is treated as its three edges, because the face normal a signed
distance would read is not trustworthy there; that answer can be up to the
triangle's height too far.

`MFTLRasterizeSignedDistance` scatters, for each triangle, the value
$\pm\lVert\mathbf{x}-\mathbf{q}\rVert$ to every lattice node $\mathbf{x}$ within
the cutoff, where $\mathbf{q}$ is the closest point and the sign is that of
$(\mathbf{x}-\mathbf{q})\cdot\mathbf{n}_\text{pseudo}$:

| Closest feature | Pseudo-normal |
|---|---|
| face | $\hat{\mathbf{n}}_T$ |
| edge | $\hat{\mathbf{n}}_T + \hat{\mathbf{n}}_{T'}$ ($T'$ across the edge; $T$ alone at a mesh rim) |
| vertex | angle-weighted $\mathbf{N}_v$ (`verts.normal`, #MFTL_NORMAL_WEIGHT_ANGLE) |

For a closed, consistently oriented, non-self-intersecting surface this sign is
provably correct (Bærentzen & Aanæs, *IEEE TVCG* 11(3), 2005); area-weighted
vertex normals can fail on thin, sharply folded sheets — precisely where topology
changes are detected. Contributions are combined by `MFTLSdfMerge` (smaller
magnitude wins, ties to the negative value): a total order, so the merged field
is independent of triangle and rank order and a distributed rasterisation is
bit-identical to a serial one.

<!-- figure: closest-point
The closest point on a triangle. (a) Perpendiculars through the corners cut the
plane into seven regions: the face, three edges (edge $k$ is half-edge $3t+k$)
and three corners. Six dot products select the region before any formula is
evaluated. (b) A cross-section through a sharp edge. For $\mathbf{x}$ in the
edge's region the face normal alone gives the wrong sign; the edge
pseudo-normal $\hat{\mathbf{n}}_T+\hat{\mathbf{n}}_{T'}$ gives the right one.
-->

Nodes farther than the cutoff from every triangle stay `MFTL_SDF_UNKNOWN`. Every
corner of a lattice cell crossed by the surface lies within one cell diagonal of
it, so a cutoff above $\sqrt3\,\Delta x$ supplies every value marching cubes needs
without a flood fill. The lattice is **global** and periodic (nodes $g$ and
$g + N$ coincide), and a caller stores a *window* of it; triangles reach nodes by
unwrapped index in their own periodic image.

`MFTLRasterizeSignedDistanceBanded` gives marked nodes a WIDER cutoff. A node's
value is the merge over exactly the triangles within its own cutoff, provided the
caller flags every triangle that can come within the wide cutoff of a marked node.
Its nearest triangle is always one of those, so each node holds the value the plain
rasterisation at its own cutoff would give, to the bit. Topology change uses it to
pay the bias-wide band only near events. Both entry points skip a node before the
closest-point query when a lower bound on its distance — to the triangle's plane,
or to its bounding sphere minus that sphere's radius — exceeds the node's cutoff or
the value it already holds by more than a few ulps of the largest coordinate
involved. That skip is exact by construction: it drops only queries the exact test
would reject. It is checked bit for bit against a brute-force nearest-triangle
reference at the origin and a million units from it.

`MFTLRasterizeEdgeCrossings` records, per lattice edge, where the surface crosses
it: each triangle tests the lattice lines through its projected box, and a line
inside it names an edge (its lower node) and a fraction along it. A slot keeps that
fraction when exactly ONE triangle crossed the edge; two make it MANY
(`MFTLSdfCrossMerge`, commutative and associative like `MFTLSdfMerge`). A line
through an edge both its triangles count therefore costs that edge its crossing and
never puts it in a wrong place. Marching cubes puts the vertex of a once-crossed
edge on the crossing instead of where the linear interpolant of `phi` vanishes,
which is $O(\Delta x^2\kappa)$ off a curved surface (Wojtan et al.'s type-1
vertices).

`MFTLClosestPointsBetweenTriangles` returns the witnesses of two disjoint
triangles' minimum distance: six corner-face projections and nine
`MFTLClosestPointsOnSegments` pairs in a fixed order, so the pair it returns is a
function of its arguments. It has no sliver threshold (a triangle loses its face
term only when its corners are numerically collinear), and it is also the
distance of the contact predicate `MFTLTriTriProximity` (@ref bvh_distance).

### Closest points without the normal equations {#geom_closest_accuracy}

Every closest-point query is a small linear least-squares problem: the foot of a
perpendicular. Its $2\times2$ normal equations square the condition number. For
segments $P_0 + s\mathbf u$ and $Q_0 + t\mathbf v$ ($\mathbf w = P_0 - Q_0$) the
Gram determinant $D = ac - b^2$ of two directions at an angle $\theta$ is
computed with a relative error $\varepsilon/\sin^2\theta$, and the numerator
$be - cd$ cancels the same way. Ericson's triangle weights $d_1d_4 - d_3d_2$ are
the same thing for a sliver. Measured on the old code: a gap of $10^{-9}$ between
edges at an angle of $2\cdot10^{-8}$ came out as $10^{-8}$, a corner over a
1:$10^5$ sliver 26 times too far, and the "parallel" threshold
$D \le 10^{-15}ac$ that guarded the noise threw the interior answer away whenever
it fired.

What replaces them:

- **Segments.** $D = |\mathbf u\times\mathbf v|^2$ (Lagrange's identity, without
  the cancellation) and
  $be - cd = (\mathbf u\times\mathbf v)\cdot(\mathbf v\times\mathbf w)$,
  then Ericson's clamping cascade. There is no threshold:
  where $D$ is too noisy to trust, the valley the error runs along is just as
  flat, and the cascade loses a few ulps at most. $D = 0$ exactly (parallel)
  pins $s = 0$, and the cascade finds the overlap.
- **Triangles.** A Gram–Schmidt (QR) frame on the longest edge $\mathbf e = p_1 - p_0$:
  the apex offset $\mathbf g = (p_2 - p_0) - c\,\mathbf e$ with
  $c = (p_2-p_0)\cdot\mathbf e / \mathbf e\cdot\mathbf e \in [0,1]$. A point is
  $p - p_0 = s\,\mathbf e + t\,\mathbf g + \text{off}$, with $t$ taken from the
  residual $r = (p-p_0) - s\mathbf e$ alone, so no coordinate is a difference of
  products. The barycentric weights are $t$, $s - ct$ and $1 - s - (1-c)t$.
  What a sliver cannot pin down is the tilt of its plane about the long edge, and
  because the offset is measured from that edge, within the sliver's width of the
  projection, the tilt reaches the distance only to second order.
- **Local differences.** Every distance is the norm of a vector built from
  differences of nearby points, never from a returned point. A configuration far
  from the origin therefore keeps its accuracy; only the witnesses carry the
  offset's rounding.

The distance is then accurate to rounding: at most $1.5\,\varepsilon L$ ($L$ the
configuration's diameter) over $2\times10^5$ random near-parallel and sliver
pairs in general position, against up to $10^{10}\varepsilon L$ before.
`tests/test_geometry.c` checks it against configurations whose distance is known
EXACTLY (planes $z = 0$ and $z = \delta$ mapped by an integer matrix that is 3 or
7 times a rotation, with exactness of the mapping checked) and against a
double-double reference in random general position.

---

## Extent on a periodic axis, and the periodic extent rule {#geom_periodic_extent}

Per-body quantities on a periodic axis -- an enclosed volume, a centroid, a
reconstruction window -- need the whole body in ONE image. The library lifts a
coordinate by the nearest image about the body's centre,
$x \mapsto x - L\,\mathrm{nearbyint}((x-c)/L)$, which is exact when every vertex
is within $L/2$ of $c$. So the question is how to find $c$ when the stored
coordinates are wrapped and a body straddling the seam has a raw min/max
spanning the whole axis.

**The rule.** On every periodic axis, a body either WRAPS the axis (it closes on
itself only after going around the period: a film across a periodic box, the
travelling-wave ribbon) or is NARROWER THAN HALF THE PERIOD along it. A body
wider than $L/2$ is nearer its own periodic image than its own far side, which
no front-tracking setup wants. `MFTLEnsembleCheckPeriodicExtents` checks it and
returns `MFTL_ERROR` on every rank, naming the first offending body, when a body
breaks it.

**Four frames.** Each triangle contributes its bounds in four frames, cut at
$\min_k + \{0, L/2, L/4, 3L/4\}$: in frame $f$ the triangle is moved rigidly by
the period multiple that brings corner 0 (shifted by the frame offset) into the
box. A body whose projection on the axis misses a frame's cut is contiguous in
that frame and measures its true extent $E$. A cut lying INSIDE the projection
makes the frame measure at least $L$: every point near the cut is covered, from
one side or the other, by a triangle placed a period apart. So the smallest
frame extent below $L$ is always exact. One MIN reduction per body carries all
four frames (lower bound and negated upper bound, 24 slots).

| frames (cuts) | measures exactly every compact body narrower than |
|---|---|
| $\{0, L/2\}$ (the previous measurement) | $L/2$ -- and NO wider one that straddles both cuts |
| $\{0, L/4, L/2, 3L/4\}$ | $3L/4$ |

A body is COVERING an axis when no frame measures it narrower than $3L/4$. That
is every wrapping body, and every compact body at least $3L/4$ wide: with four
cuts such a body is measured exactly at some placements and not at others, so
the verdict is taken on the size alone. It also keeps the decision a quarter
period away from $L$, where an exact cover and a sliver of a gap differ in the
last bit (a marching-cubes vertex lying exactly on the seam). The rule's limit
is $L/2$ and the measurement's is $3L/4$ on purpose: a body growing through
$L/2$ is SEEN in the band $[L/2, 3L/4)$ before it becomes indistinguishable from
a wrapping one, provided the check runs at least once in the time any body
grows by a quarter period -- at output steps, and after a topology change,
which can merge two bodies into one wide one.

The two frames cut at 0 and $L/2$ are computed and compared exactly as the
two-frame code did, and the quarter frames are only read when neither measures
less than $3L/4$, so every extent that code got right is reproduced to the bit
(`test_a_chain_is_measured_exactly_below_three_quarters_at_every_offset`).

| consumer | depends on the extent how | a body in $[L/2, 3L/4)$ | wider |
|---|---|---|---|
| topology change (`MFTLTopoResolveExtents`) | window, nearest-image volume | exact | covering: whole-axis window, axis not used for the flux; no axis left = `n_clusters_unmeasured` |
| `MFTLEnsembleCheckPeriodicExtents` | the rule | reported, `MFTL_ERROR` | counted in `n_covering` |
| remesher (`MFTLRemeshMeasureBodies`) | circular-mean centre, strategy choice only | unaffected: operators are local | unaffected |
| spatial halo (`MFTLSpatialHaloResolveRadii`, the share extents) | envelope radius; a CONSERVATIVE send decision | exact | over-sends, never misses |

The two-frame measurement silently skipped the topology change's volume
correction for a closed body straddling both of its cuts on every axis: every
axis came out covered, no flux axis was left, and `volume_before` was 0. That is
what `test_a_body_straddling_both_half_frames_keeps_its_volume` pins.

**What lifting the rule would take.** Per-vertex periodic image counters
(LAMMPS's packed image flags, HOOMD-blue's `int3`) remove the limit entirely:
an `int16` column per axis in the vertex pool block (6 B/vertex) plus a per-body
`int64` base carried in the redistribution's body header, ghosts following
their owner by the minimum-image jump on receipt, every position writer (the
application's advection, split, collapse, the smoother, the topology change)
maintaining them, and a distributed canonicalisation (a weighted union-find
with lattice offsets, the label rounds of `topo_split_bodies.c`). A body's
winding needs none of it: it is the cycle sum of the per-edge minimum-image
wraps. Revive it only for a required case with a compact body of at least
$L/2$ on a periodic axis -- unit-cell slug flow with a bubble longer than half
the cell, a drop stretched past half a short periodic shear box.

---

## Area, volume and centroid of a body {#geom_body_integrals}

`MFTLEnsembleBodyIntegrals` (@ref mftl_mod_ensemble) returns, per body, its
area, the volume of the region it bounds and that region's centroid, each with a
status. The local half is here. Every quantity is the flux of a polynomial
field $\mathbf F$ with $\nabla\cdot\mathbf F$ known, so the divergence theorem
turns a sum over the owned triangles into a volume integral:

| field $\mathbf F$ | $\nabla\cdot\mathbf F$ | flux over the region's boundary |
|---|---|---|
| $y_k\,\mathbf e_k$ | 1 | $V$ |
| $\tfrac12 y_k^2\,\mathbf e_k$ | $y_k$ | $V\,(C_k - r_k)$ |
| $y_k y_j\,\mathbf e_k$, $j\ne k$ | $y_j$ | $V\,(C_j - r_j)$ |

with $\mathbf y = \mathbf x - \mathbf r$. The integrands are linear and quadratic
on a flat triangle, so with $S_{t,k}$ its area vector's component the sums are
exact:

$$
  \int_T y_k n_k\,dA = S_{t,k}\,\bar y_k,\qquad
  \int_T y_k y_j n_k\,dA = \frac{S_{t,k}}{12}\Big(\sum_i y_{k,i}y_{j,i} + \sum_i y_{k,i}\sum_i y_{j,i}\Big).
$$

**Which region.** The flux through the SURFACE equals the volume integral only if
every other part of the region's boundary contributes nothing:

| the body | the region | $k$ and $r_k$ | why nothing else contributes |
|---|---|---|---|
| closed, total area vector zero | the one it encloses | any axis it does not cover; its centre | there is nothing else |
| open, every rim on a wall | closed by the walls | rims on exactly one wall of $k$: that wall; else an axis whose walls carry no rim: its lower wall | $\mathbf F\cdot\mathbf n = 0$ on walls across the other axes, $\mathbf F = 0$ on the wall at $r_k$ |
| closed, total area vector non-zero | between the sheet and a wall | the axis the area vector points along, its lower wall | a sheet across periodic axes separates the box along that axis |

The open rule picks what every open-surface example hand-coded: Rayleigh–Taylor
and the travelling waves from the floor along $z$, the jet from the nozzle wall
along $x$; a drop hanging from the ceiling is measured from the ceiling. The
table is ONE function, `MFTLBodyFluxAxis`, which the topology change applies per
CLUSTER too, without the sheet row: it measures a closed cluster as enclosing,
which for a sheet is its volume up to a constant -- invisible to the before/after
comparison it needs, and possible without a wall. A body whose rim leaves the walls has no region
(UNDEFINED); one with rims on both walls of every non-periodic axis, or a sheet
across a fully periodic box, has one but no wall to measure it from
(UNSUPPORTED). A centroid component along an axis the body COVERS
(@ref geom_periodic_extent) is UNDEFINED: the body wraps it. On a periodic axis
every triangle is moved rigidly into the image nearest the body's centre, which
the four-frame extent makes exact for every body narrower than $3L/4$ -- wider
than the rule allows, and measured right anyway.

The volume is SIGNED, positive when the normals point away from the region:
outward for a closed body. A closed body's area vector is zero to round-off and a
separating sheet's is its cross-section, so the test against $10^{-8}$ of the area
has no length scale and no ambiguity.

**Compositions are the caller's arithmetic**, not machinery: the liquid around $N$
bubbles in a periodic box is $V_\mathrm{cell} - \sum_b V_b$ (its centre is
undefined, it wraps); the gas above a wall-closed interface is
$V_\mathrm{box} - V$; a slab between two films that each wrap $x$ and $y$ is the
difference of their volumes, each measured from the floor. Inside/outside queries
for a point are not provided: ray parity with an exact orientation test would be
the way, when IBM coupling needs one.

**Cost.** Two holder reductions (37 and 4 values per body) and two passes over
the owned triangles: 23 ns per triangle on a periodic domain against 5.7 for
`MFTLEnsembleUpdate` (Apple silicon, np = 1, 328 k triangles). The four-frame
extent (12 ns) and the lifted moments (10 ns) are the bulk; the extent's frames
keep the arithmetic the topology change is bit-identical with. A call for output
intervals and diagnostics, not for every timestep.

---

## Least squares and local height fits {#geom_surface_fit}

`MFTLLeastSquaresQRCP` solves $\min_x\lVert Ax-b\rVert_2$ for at most six unknowns
by Householder QR with column pivoting, and `MFTLFitHeightParaboloid` builds the
weighted system of a height paraboloid
$z = c_0 + c_1x + c_2y + c_3x^2 + c_4xy + c_5y^2$ on top of it. The remesher's
paraboloid placement is their only user (`src/remeshing/README.md`, "Vertex
placement").

**Never the normal equations.** $A^\top A$ squares the condition number. The
solver triangularises the weighted rows $\sqrt{w_j}\,(1, x, y, x^2, xy, y^2)$
directly: one reflector per column, applied to the remaining columns and to $b$,
then back substitution on $R$. Column norms are downdated and recomputed once
they have lost 90 % to cancellation, as LAPACK's `dlaqp2` does.

**Grouped pivoting, and what "rank-deficient" then means.** Columns form two
groups, factored in order: the lead group $\{1, x, y\}$ (height and slope) and
the trailing group $\{x^2, xy, y^2\}$ (curvature). Within a group the column of
largest remaining norm goes next. A group stops at the first pivot with
$|R_{kk}| \le \tau |R_{00}|$ ($\tau = 10^{-3}$ for the placement), and its
remaining columns get coefficient zero: the basic solution. In words: a
curvature component the samples cannot determine is set to zero, which is a
fit of lower degree in that direction, and the height is never traded for it.

LAPACK's `dgelsy` returns the MINIMUM-NORM solution instead, which spreads the
coefficients over the null space. When the null space couples the height with
the curvature, the height it returns depends on that choice rather than on the
data. Eight samples on the unit circle at height 1 are the clean case:
$x^2+y^2=1$ makes 1, $x^2$, $y^2$ dependent, and `dgelsy` returns
$c_0 = 0.667$ with $c_3 = c_5 = 0.333$; the grouped basic solution returns
$c_0 = 1$, the plane through the data. On a full-rank system both solve the same
problem and agree to $2\cdot10^{-15}$ (`studies/surface_fit/`).

**Scale.** The rank test is relative, so the columns must be of comparable size.
The caller passes coordinates in units of the local edge length; a uniform
scaling of $A$ and $b$ by a power of two then changes no pivot and no bit of
$x$ (`test_qrcp_is_independent_of_the_scale`). Unscaled coordinates give the
columns norms of order 1, $h$ and $h^2$, and a fixed relative tolerance then
drops the curvature columns as soon as $h^2$ falls below it. That was mftl's
fit, at `rcond` = 1e-10: on a sphere of radius $10^{-4}$ every edge lost its
curvature terms and the placed height was off by 4.5 sagittas.

**Cost**, Apple M4, per solve of the placement's systems (24 rows from
positions, 72 with normals): 0.39–0.53 µs and 1.2–1.3 µs, against 1.3–1.5 µs and
2.4–2.5 µs for `dgelsy` with a precomputed workspace: 1.9 to 3.4 times faster at
these sizes, where LAPACK's blocking and workspace logic is pure overhead.

---

## Signed curvature {#geom_curvature}

**The convention, stated once.** $\kappa = \kappa_1 + \kappa_2 = \nabla_s\cdot\hat{\mathbf n}$
with $\hat{\mathbf n}$ the OUTWARD normal (corners counter-clockwise seen from
outside): a sphere of radius $R$ has $\kappa = 2/R$, the Young–Laplace jump is
$p_\text{in} - p_\text{out} = \sigma\kappa$, and the surface-tension force density
$-\sigma\kappa\hat{\mathbf n}$ points inward. The remesher's curvature sizing reads
$\max(|\kappa_1|, |\kappa_2|)$ off the COTAN record.

`MFTLComputeVertexCurvature(obj, method, ncomp, out, &n_fallback)` writes a record
of `ncomp` doubles per vertex, the layout of a vertex field: $\kappa$; $\kappa_G$;
$\kappa_1 \ge \kappa_2$; a unit tangent along the principal direction of
$\kappa_1$. A shorter prefix (1, 2 or 4) skips the work it does not need. Owned
vertices are exact on a distributed mesh (one-ring quantities, vertex-complete
halo); `MFTLEnsembleComputeCurvature` makes the ghosts their owners' with one
field exchange.

### COTAN: Meyer, Desbrun, Schröder & Barr (2003) {#geom_curvature_cotan}

Every curvature operator of the paper, from one scatter over the triangles:

| quantity | formula | paper |
|---|---|---|
| mean-curvature normal | $\mathbf K_i = \dfrac{1}{2A_i}\sum_{j}(\cot\alpha_{ij} + \cot\beta_{ij})(\mathbf x_i - \mathbf x_j) = \dfrac{\nabla_i A}{A_i}$ | eq. 8 |
| signed curvature | $\kappa_i = \mathbf K_i\cdot\hat{\mathbf N}_i$, $\hat{\mathbf N}_i$ area-weighted | eq. 12 |
| Gaussian curvature | $\kappa_{G,i} = (2\pi - \sum_f\theta_f)/A_i$ | eq. 9 |
| principal curvatures | $\kappa_{1,2} = \tfrac{\kappa}{2} \pm \sqrt{\max(\kappa^2/4 - \kappa_G, 0)}$ | eqs. 10–11 |
| principal direction | least-squares curvature tensor from the edge normal curvatures $2(\mathbf x_i - \mathbf x_j)\cdot\hat{\mathbf n}/|\mathbf x_i - \mathbf x_j|^2$, trace fixed to $\kappa$ | §5.3 |

$A_i$ is the MIXED area: the Voronoi share of a non-obtuse triangle, half or a
quarter of an obtuse one (their Fig. 4), so the cells tile the surface. With the
trace fixed the tensor is $\tfrac{\kappa}{2}I + \begin{pmatrix}p & q\\ q & -p\end{pmatrix}$
in a tangent frame and an edge at angle $\theta$ predicts
$\tfrac{\kappa}{2} + p\cos2\theta + q\sin2\theta$: a two-unknown QR, no normal
equations. Each triangle of the fan contributes one row per edge it has at the
vertex with its half of Meyer's weight
$\tfrac18(\cot\alpha + \cot\beta)|\mathbf x_i - \mathbf x_j|^2$
(a negative half, behind an obtuse angle, is dropped), so an
edge's two rows add up to Meyer's one. Meyer's non-linear determinant constraint is
left out, as they found it unnecessary.

What it guarantees: zero on a plane; $\sum_i \kappa_{G,i}A_i = 2\pi\chi$ to
round-off on any closed mesh (`test_gauss_bonnet_holds_to_round_off`); and
$\sigma A_i\mathbf K_i$ IS the Frenet-vertex force of @ref geom_surface_tension
with triangle normals, to $2\cdot10^{-15}$
(`test_the_frenet_vertex_force_is_the_cotangent_operator`). What it does not: the
cotangent formula is **not pointwise convergent on an irregular mesh** (Xu 2004;
Wardetzky et al. 2007), only in an integral sense. Where $\kappa \approx 0$ and
$\kappa_G \approx 0$ (a plane), a rounding-sized $\kappa_G$ becomes its square root
in $\kappa_{1,2}$: $10^{-13}$ turns into $3\cdot10^{-7}$.

The rest of the paper -- anisotropic smoothing (§6.3) and operators on manifolds in
$n$ dimensions (§7) -- is mesh processing and image denoising, not a curvature of a
front, and is not in the library.

### FIT: a one-ring paraboloid through positions and normals {#geom_curvature_fit}

In the frame of the vertex normal every sample -- the vertex at the origin and
each one-ring neighbour -- gives three rows: its height and the two slopes
$\partial_x z = -n_x/n_z$, $\partial_y z = -n_y/n_z$ of its normal. Lengths are
divided by the ring's RMS edge, so heights and slopes are both $O(1)$ and one
weight serves both kinds of row. `MFTLFitHeightParaboloid` solves for
$z = c_0 + \dots + c_5y^2$ by QR with grouped pivoting, and the curvatures are
those of the fitted surface at the vertex:

$$
  I = \begin{pmatrix}1+c_1^2 & c_1c_2\\ c_1c_2 & 1+c_2^2\end{pmatrix},\qquad
  I\!I = -\frac{1}{W}\begin{pmatrix}2c_3 & c_4\\ c_4 & 2c_5\end{pmatrix},\qquad
  W = \sqrt{1+c_1^2+c_2^2},
$$

the shape operator is $S = I^{-1}I\!I$, $\kappa = \operatorname{tr}S$,
$\kappa_G = \det S$, and the principal direction is its eigenvector, mapped to
$u\,\mathbf x_u + w\,\mathbf x_v$.

Three normal-equation-free rows per sample make five unknowns well determined from
the one-ring alone, so the fit needs no topological ghost growth -- only the
neighbours' normals, which `MFTLEnsembleUpdateVertexNormals` makes exact on ghosts.
It is only as good as those normals: the slope rows turn a normal error $\delta$
into a curvature error $\delta/h$. Area-, angle- or Max-weighted normals are first
order on an irregular mesh and stall the fit at a few percent, so the fit always
reads `MFTL_NORMAL_WEIGHT_FIT` normals (second order, @ref geom_fit_normal), with
which it is as good as with exact normals; `MFTLEnsembleComputeCurvature` computes
and exchanges them. A vertex with fewer than three neighbours, fewer than three
whose normal is within 75° of its own (a fold), or a rank-deficient system gets the
COTAN values and is counted.

### Measured {#geom_curvature_measured}

Maximum error of $\kappa$ over the vertices, relative to the largest $|\kappa|$ on
the shape, at about 41 k vertices (246 k on the tori, 309 k on the marching-cubes
sphere), every vertex exactly on the surface (`examples/curvature_comparison.c`):

| mesh | COTAN | FIT, area normals | FIT, fitted normals | FIT, exact normals |
|---|---|---|---|---|
| icosphere | 1.2e-7 | 5.2e-2 | 2.1e-4 | 2.1e-4 |
| icosphere, vertices jittered by 0.3 h | 9.7 | 0.47 | 3.9e-4 | 3.7e-4 |
| isotropically remeshed sphere | 0.19 | 0.14 | 2.6e-4 | 2.5e-4 |
| marching-cubes sphere | 809 | 35 | 5.2e-3 | 6.9e-5 |
| ellipsoid 1 : 0.7 : 0.5 | 1.4e-2 | 4.2e-2 | 9.1e-4 | 6.2e-4 |
| torus, structured | 7.5e-5 | 2.2e-4 | 2.1e-4 | 1.9e-4 |
| torus, jittered by 0.2 cells | 0.92 | 0.46 | 9.8e-3 | 8.7e-3 |

COTAN is superb on the symmetric meshes (the icosphere, the structured torus) and
does not converge on the irregular ones; the error does not fall with $h$ (orders
around zero) and is largest where triangles are poorly shaped -- 800 times the
curvature at a marching-cubes sliver. The fit with fitted normals converges on
every mesh: at second order on the spheres and the ellipsoid, at first order on
the jittered torus, whose asymmetric stencils leave the cubic terms uncancelled
(the exact normals do no better there). The principal direction follows the same
pattern (RMS 0.06° FIT against 12° COTAN on the jittered torus).

Cost per vertex, Release, Apple M4, isotropically remeshed sphere of 41 k
vertices:

| | ns/vertex |
|---|---|
| COTAN, $\kappa$ | 14 |
| COTAN, $\kappa$ and $\kappa_G$ (three `atan2` per triangle) | 47 |
| COTAN, full record (Meyer's direction fit) | 268 |
| fitted normals | 233 |
| FIT, $\kappa$ (normals not included) | 421 |
| FIT, full record | 502 |

**Which to use.** `MFTL_CURVATURE_FIT` with `MFTL_NORMAL_WEIGHT_FIT` normals for
a pointwise curvature on a remeshed, topology-changed or otherwise irregular front
(the settings default). COTAN where only an integral matters, where the mesh is
symmetric, or where 14 ns per vertex matters more than accuracy.

---

## Surface tension {#geom_surface_tension}

The force on a piece of surface is the pull of the rest of the surface on its
boundary (Tryggvason et al. 2011, eqs. 7.52–7.53),

$$
  \delta\mathbf F = \oint_C \sigma\,\hat{\mathbf m}\,dl
  = \int_{\Delta A}\big(\nabla_s\sigma - \sigma\kappa\hat{\mathbf n}\big)\,dA,
  \qquad \hat{\mathbf m} = \hat{\mathbf t}\times\hat{\mathbf n},
$$

with $\hat{\mathbf m}$ the unit outward conormal: the force a front-tracking solver
spreads, INTEGRATED over the piece (a force, not a density). The two forms differ in
the piece, and each in the normal the conormal is built from
(`MFTL_SurfaceTensionParameters::contour_normal`).

**Frenet vertex** (eqs. 7.54–7.59): the barycentric cell of a vertex, whose boundary
inside a triangle runs from an edge midpoint to the centroid to the next midpoint.

- *Triangle normals* (the book's eq. 7.59): the two segments' pulls add up to half
  the pull on the opposite edge,
  $\mathbf F_v = \sum_{T\ni v}\tfrac12\sigma_T(\mathbf x_k - \mathbf x_j)\times\hat{\mathbf n}_T$
  for $(v, j, k)$ counter-clockwise -- exactly $-\partial E/\partial\mathbf x_v$ of
  $E = \sum_T\sigma_T A_T$. So the net force AND the net torque of a closed body
  vanish to round-off, the discrete surface energy is the one the force derives
  from, and for constant $\sigma$ the force is $-\sigma A_\text{mixed}\mathbf K$ with
  Meyer's operator. (Eq. 7.57 writes $\hat{\mathbf n}\times\hat{\mathbf t}$ where
  eq. 7.52 has $\hat{\mathbf t}\times\hat{\mathbf n}$; with outward normals the pull
  is the form given here, inward on a sphere.)
- *Vertex normals*: along each segment the vertex normals are interpolated
  linearly, and the segment's pull is evaluated ONCE and given to the two cells it
  separates with opposite signs. The force then converges pointwise to
  $-\sigma\kappa\hat{\mathbf n}$ times the vertex's BARYCENTRIC area -- the weight
  the coupler gives a vertex marker. The net force is still zero to round-off; the
  net torque only to truncation error.

**Frenet element** (eq. 7.53 on a triangle):
$\mathbf F_T = \sum_{e\in T}\sigma_e L_e\hat{\mathbf m}_e$,
the pull of the neighbours across each edge. With the
triangle's own normal on all three edges the pulls cancel exactly and nothing is
left, so the edge normal comes from both sides: the two triangles' normals summed
(*triangle normals*), or the two end vertices' normals summed (*vertex normals*).
The edge vector is a direct minimum-image difference and the normal a commutative
sum, so the two triangles of an edge compute the same pull with opposite signs, to
the bit: the net force of a closed body vanishes to round-off. The torque does not
(the force is applied at centroids).

The conormal is always $(L/|\mathbf c|)\,\mathbf c$ with
$\mathbf c = \mathbf e\times\mathbf s$ and $\mathbf s$ the unnormalised normal: normalising the cross product
keeps it a unit vector when $\mathbf s$ is not perpendicular to the piece (vertex
normals), and a vanishing $\mathbf c$ (a fold) falls back to the triangle's normal.

### What each variant conserves {#geom_surface_tension_conservation}

The four variants are named by form and contour normal: **V-T** (vertex form,
triangle normals: the book's eq. 7.59, the default), **V-F** (vertex form, fitted
vertex normals), **E-T** (element form, triangle normals) and **E-F** (element
form, fitted vertex normals).

**V-T is the gradient of the discrete surface energy** $E_h = \sum_T\sigma_TA_T$.
Inside a flat triangle the normal is constant, so the pull on the polyline from the
midpoint of edge $(v,j)$ through the centroid to the midpoint of edge $(v,k)$
depends on its ends only,
$\tfrac12(\mathbf x_k-\mathbf x_j)\times\hat{\mathbf n}_T = -\partial A_T/\partial\mathbf x_v$.
For uniform $\sigma$ this is the cotangent
formula, $\nabla_vA = A_\text{mixed}\mathbf K_v$ (Pinkall & Polthier 1993; Meyer et
al. 2003). Three exact properties follow from the invariances of $E_h$, and each
holds triangle by triangle:

- *momentum*: a translation does not change $E_h$, so $\sum_v\mathbf F_v = 0$;
- *angular momentum*: nor does a rotation. Differentiating $E_h(\mathsf R\mathbf x)$
  along any rotation $\boldsymbol\omega$ gives
  $\boldsymbol\omega\cdot\sum_v\mathbf x_v\times\mathbf F_v = 0$, so the net torque
  vanishes;
- *energy*: for any vertex velocities $\mathbf u_v$,
  $\sum_v\mathbf F_v\cdot\mathbf u_v = -dE_h/dt$ (the chain rule).

$E_h$ is also homogeneous of degree 2 in the positions, so
$\sum_v\mathbf x_v\cdot\nabla_vE_h = 2E_h$ (Euler): on a sphere about the origin the
Laplace pressure $-\sum\mathbf F\cdot\hat{\mathbf n}/A = 2\sigma/R$ is exact at every
resolution.

The energy property is the one coupling needs. With the markers moving at the
interpolated velocity $\mathbf u_v = (S^\mathsf T\mathbf u)_v$ and the force spread
with $S$ (the coupler's spread IS the exact transpose of its interpolation), the
power the force puts into the fluid is
$\langle S\mathbf F,\mathbf u\rangle = \sum_v\mathbf F_v\cdot\mathbf u_v = -dE_h/dt$:
kinetic plus surface energy can only decrease, by the viscous dissipation, in the
semi-discrete system. That is the mechanism of energy-stable immersed-boundary
schemes whose forces derive from an energy (Peskin 2002; Newren et al. 2007), and of
the unconditionally stable parametric finite elements of Barrett, Garcke & Nürnberg
(2015), whose piecewise-linear curvature is this same area gradient.

**The others conserve momentum only.** Each piece of contour is shared by two cells
and evaluated once, so its two pulls cancel and $\sum\mathbf F = 0$ to round-off in
all four variants (the conservative property of Tryggvason et al. 2001). The torque
is another matter. The continuous surface stress
$\sigma(\mathsf I - \hat{\mathbf n}\hat{\mathbf n})$ is symmetric and exerts none, but a discrete pull
$\mathbf P$ on a shared piece acts on two markers at different points and leaves the
couple $(\mathbf x_1-\mathbf x_2)\times\mathbf P$:

- *E-T and E-F*: the pull $\sigma L\hat{\mathbf m}$ across an edge acts on the two
  centroids, $\mathbf d = \mathbf c_1-\mathbf c_2$. In the edge frame
  ($\hat{\mathbf t}$ along the edge, $\hat{\mathbf m} = \hat{\mathbf t}\times\hat{\mathbf n}$)
  the couple is
  $\sigma L\,[-(\mathbf d\cdot\hat{\mathbf t})\,\hat{\mathbf n} + (\mathbf d\cdot\hat{\mathbf n})\,\hat{\mathbf t}]$:
  a twist about the normal wherever
  the two centroids are offset along the edge, and a bend about the edge wherever
  they sit at different heights, $O(\kappa h^2)$. Mirror-symmetric pairs cancel,
  which is why an icosphere shows nothing; an irregular mesh does not;
- *V-F*: a segment's pull acts on two vertices of the same edge, so its couple is
  small, but the three segments of a triangle now carry three different
  interpolated normals, and the triangle's corner forces no longer balance;
- *V-T*: the same per-segment couples exist, and those of each triangle sum to
  zero, because its three corner forces are the gradient of $A_T$.

None of V-F, E-T and E-F is the gradient of a function of the positions, so
$\sum\mathbf F\cdot\mathbf u \ne -dE_h/dt$: an $O(h^2)$ consistency error of
uncontrolled sign, work done on the fluid that the interface energy does not pay
for.

### Per marker and on the grid {#geom_surface_tension_grid}

**Per marker, the normal decides.** The force on a cell is
$\sigma\oint\hat{\mathbf t}\times\hat{\mathbf n}\,dl$ around a contour of length
$O(h)$. With triangle normals $\hat{\mathbf n}$ is a facet's, $O(h)$ off the
surface's on an irregular mesh: an $O(h^3)$ error in a force of $O(\kappa h^2)$,
which does not converge. Fitted normals are $O(h^2)$ accurate, which leaves the
force first-order accurate per marker. A vertex cell averages the errors of about
six triangles, where a triangle cell carries one triangle's alone (a sliver on a
marching-cubes mesh included), so V-F is better per marker than E-F, with half the
markers.

**On the grid a solver sees a spread sum, never one marker.** With
$\mathbf F_i^*$ the exact integral of $\mathbf f = -\sigma\kappa\hat{\mathbf n}$ over
marker $i$'s cell and $\delta_w$ a kernel of width $w$,

$$
  \sum_i\mathbf F_i\,\delta_w(\mathbf X-\mathbf x_i) - \int\mathbf f\,\delta_w(\mathbf X-\mathbf x)\,dA
  = \underbrace{\sum_i(\mathbf F_i-\mathbf F_i^*)\,\delta_w(\mathbf X-\mathbf x_i)}_{\text{force error}}
  + \underbrace{\Big[\sum_i\mathbf F_i^*\,\delta_w(\mathbf X-\mathbf x_i) - \int\mathbf f\,\delta_w\,dA\Big]}_{\text{lumping error}} .
$$

The lumping error is the quadrature error of putting a cell's force at its marker:
the same for every force formula, and $O((h/w)^2)$. V-T's per-marker error is in
divergence form (the error on a shared segment enters its two cells with opposite
signs), so it cancels under a smooth kernel. Three regimes follow.

1. *A smooth kernel* ($w \gg h$; what integrated quantities see, such as the net
   force on a region or a drop's motion): every variant converges at second order,
   and V-T is among the best.
2. *Plain spreading at grid scale* ($w \approx 2h$): at a fixed $h/w$ the lumping
   error is a fixed fraction (0.05–1 % on an irregular mesh), so no variant
   converges, whichever force it spreads. On top of it, V-T's force error cancels
   only partly over the twenty or so markers under a kernel: on irregular tori V-T
   is 2–11 times V-F, on ellipsoids within a factor 2. The element forms halve the
   lumping error with twice the markers.
3. *A balanced-force curvature* (the hybrid formulation of Shin et al. 2005, recipe
   below): $(\sigma\kappa)_h = -\mathbf F_h\cdot\mathbf G_h/(\mathbf G_h\cdot\mathbf G_h)$,
   with $\mathbf G$ each marker's normal times its cell area spread by the same
   kernel. Writing $\mathbf F_i = -\sigma\hat\kappa_i\mathbf G_i + (\text{tangential})$,
   the ratio is a kernel-weighted mean of the per-marker curvatures $\hat\kappa_i$.
   The lumping error is common to numerator and denominator and cancels, and what
   reaches the grid is the per-marker accuracy: V-T and E-T stall at 0.1–0.6 %, V-F
   and E-F converge (orders 1–2).

The last number is the parasitic-current number. In the balanced-force formulation
the grid force $(\sigma\kappa)_h\nabla_hI$ is an exact discrete gradient wherever
$(\sigma\kappa)_h$ is constant, and the pressure absorbs it without flow, so
spurious currents follow the variation of $(\sigma\kappa)_h$ along the interface
(Francois et al. 2006; Popinet 2018). A plainly spread force is not a discrete
gradient at all, and the lumping noise common to every variant drives the currents.

### Variable sigma: Marangoni stresses come for free

`sigma_field` names a registered one-component field. A TRIANGLE field is
$\sigma_T$, on an edge the mean of its two triangles. A VERTEX field is the linear
interpolant: on a triangle the mean of its corners (which makes
$E = \sum_T\sigma_T A_T = \int\sigma\,dA$ exact), along a piece of contour the mean
of its ends. No gradient operator is needed: the tangential force is what the
unequal pulls leave. On a flat sheet with $\sigma$ linear along it, every vertex of
both vertex forms carries exactly $\nabla\sigma$ times its barycentric area, and
both forms total $\Delta\sigma\,W$ (`test_a_linear_sigma_on_a_flat_sheet_is_the_marangoni_force`,
at np = 1–27, with the sigma ghosts poisoned so a missing exchange fails).

### Rims of open surfaces

The rim's line force is $\sigma\hat{\mathbf m}\,dl$ along its outward conormal.
`MFTL_RimForce` says what acts there:

| mode | meaning | a flat sheet between walls |
|---|---|---|
| `FREE` | nothing beyond the rim pulls; the rim is pulled into the surface | pulled inward at the rim; total zero |
| `WALL` | as FREE, plus the wall-normal component of the pull on every rim edge lying on a wall (both ends within `MFTL_BODY_WALL_TOL` of it), which the wall supplies | no force anywhere |
| `EXCLUDE` | rim vertices, and triangles with a rim edge, carry nothing | -- |

For the vertex forms the pull on a rim edge is shared by its ends as their cells
share the edge: by the hat functions with triangle normals,
$(2\sigma_a + \sigma_b)/6$ per unit pull, by halves with vertex normals,
$(3\sigma_a + \sigma_b)/8$. A spherical cap on a wall then presses on it, in WALL mode, with the
Laplace pressure times its base, $2\sigma\pi a^2/R$
(`test_a_cap_presses_on_its_wall_with_the_laplace_pressure`) -- to FIRST order
only: the rim's conormal comes from a one-sided normal (a rim triangle's, or a fit
on half a ring), 1.1–1.4 % off at 24 rings, half of what it is at 12. Contact
angles and accurate rim conormals are roadmap item C1.

### Distributed

Owned rows are exact without communication given current ghost values of what the
form reads: the vertex form is a scatter over every triangle the rank holds
(vertex-complete halo), and every owned triangle has its three edge neighbours.
`MFTLEnsembleComputeSurfaceTension` adds exactly those exchanges: the sigma
field's ghost rows when sigma is a field, and the fitted vertex normals (exact on
ghosts) with vertex normals on the contour. With a
uniform sigma and triangle normals it makes no collective call at all. Owned forces
equal the single-rank forces of the same body to $2\cdot10^{-15}$ at np = 1–27
(`test_the_distributed_force_and_curvature_are_the_serial_ones`).
`MFTLEnsembleSurfaceTensionStats` reduces each body's net force, surface energy
and area over its holders.

### Measured {#geom_surface_tension_measured}

Six meshes whose vertices lie exactly on a surface with known curvature, the kind a
run produces: an ellipsoid 1 : 0.7 : 0.5 isotropically remeshed, remeshed with
curvature sizing (edges 0.3–2 h), and from marching cubes (slivers); a torus
$R = 1$, $r = 0.6$ jittered, remeshed, and from marching cubes. 39 000 to 562 000
vertices; each cell gives the range over the six meshes at the finest level, with
the observed order between the two finest levels in brackets (2: the error quarters
when $h$ halves; 0: no convergence). Errors are relative to the largest $|\kappa|$
on the shape (docs/analysis.md §16.8):

| measure | V-T | V-F | E-T | E-F |
|---|---|---|---|---|
| per marker: $-\mathbf F\cdot\hat{\mathbf n}/(\sigma A) - \kappa$, RMS | 6.1e-3 – 7.2 (0) | 2.1e-4 – 2.4e-3 (1–1.8) | 5.8e-2 – 8.4 (0) | 4.9e-4 – 2.3e-2 (1–3) |
| net torque, $\lvert\sum\mathbf x\times\mathbf F\rvert/\sum\lvert\mathbf x\rvert\lvert\mathbf F\rvert$ | ≤ 4.3e-16 | 4.1e-9 – 6.1e-7 | 9.0e-8 – 8.3e-7 | 5.0e-9 – 1.1e-6 |
| energy, $\lvert\sum\mathbf F\cdot\mathbf u + \sigma\dot A\rvert/\lvert\sigma\dot A\rvert$ | ≤ 1.5e-11 (round-off) | 1.2e-6 – 6.5e-5 (2) | 3.8e-6 – 3.6e-5 (2) | 6.7e-6 – 6.0e-5 (2) |
| smooth kernel, $w = 0.4$ | 1.5e-5 – 7.2e-5 (2) | 4.2e-5 – 1.1e-4 (2) | 3.1e-5 – 2.5e-4 (1–2) | 2.7e-5 – 8.8e-5 (2) |
| plain spreading, $w = 2h$ | 2.3e-3 – 1.1e-2 (≈0) | 1.0e-3 – 4.4e-3 (≈0) | 2.6e-3 – 4.2e-2 (≈0) | 5.3e-4 – 1.5e-3 (≈0) |
| balanced-force $(\sigma\kappa)_h$, $w = 2h$ | 9.5e-4 – 3.5e-3 (0) | 5.4e-5 – 2.9e-4 (1–2) | 9.7e-4 – 6.2e-3 (0) | 4.3e-5 – 1.6e-4 (0.7–2) |
| markers per vertex; ns per vertex | 1; 9 | 1; 21 + 240 for the normals | 2; 24 | 2; 27 + 240 |

$A$ is the cell the force integrates over: Meyer's mixed area for V-T, the
barycentric area for V-F, the triangle for E-T and E-F. Read with the barycentric
area instead (what a solver dividing by the coupler's marker weight gets), V-T is
15 % off at the worst vertex even on an icosphere. $\mathbf u$ is a smooth non-rigid
velocity field. The kernel rows are the spread force against the exact surface
integral, $\lvert\sum_i\varphi(\mathbf x_i)\mathbf F_i - \int\varphi\mathbf f\,dA\rvert/\int\varphi\lvert\mathbf f\rvert\,dA$,
RMS over 64 (smooth) or 256 (grid-scale) kernel centres on the surface; the last
row is $\lvert(\sigma\kappa)_h/\sigma - \kappa_\text{ref}\rvert$ at 256 centres,
$\kappa_\text{ref}$ the same ratio of the exact integrals. The net force is below
$2\cdot10^{-14}$ of $\sum\lvert\mathbf F\rvert$ in every case.

**Which to use.**

- *A solver that spreads the force as it is*: **V-T, the default.** At grid scale
  every variant stalls at the lumping floor; V-F is 1–11 times lower there but costs
  thirty times as much, needs a normal exchange, and gives up the exact torque and
  energy balance.
- *A balanced-force curvature*: **V-F**, with $\mathbf G_v = \hat{\mathbf n}_vA_v$
  (fitted normal, barycentric area). It is the only vertex variant whose grid
  curvature converges.
- *Centroid markers*: **E-F**.

### Recipe: balanced force (Shin & Juric 2005; Tryggvason et al. 2011, eqs. 7.61–7.65)

Spreading $\sigma\kappa\hat{\mathbf n}\delta$ directly produces parasitic currents
when the pressure gradient cannot balance it on the grid. The hybrid formulation
balances them exactly, from library outputs and two spreads through the coupler:

1. spread the vertex force $\mathbf F_v$ and its partner $\mathbf G_v$, the normal
   and the cell the force is built from: $\hat{\mathbf n}_vA_v$ with the fitted
   normal and the barycentric area for V-F, the volume gradient
   $\sum_{T\ni v}\tfrac13A_T\hat{\mathbf n}_T$ for V-T
   (`MFTL_SPREAD_INTEGRATED`);
2. on the grid, $(\sigma\kappa)_h = -\mathbf F\cdot\mathbf G/(\mathbf G\cdot\mathbf G)$
   (positive on a drop: $\mathbf F$ points in, $\mathbf G$ out);
3. apply $(\sigma\kappa)_h\nabla_h I$ with the solver's OWN gradient operator
   $\nabla_h$ of its indicator.

Step 3 needs the solver's pressure-gradient operator, so it stays with the solver.

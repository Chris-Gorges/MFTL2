`src/geometry/` — Periodic-aware geometry {#mftl_mod_geometry}
=========================================

**Public header:** `include/mftl_geometry.h` · **Layer:** 2 ·
**Module group:** @ref mftl_geometry

Everything in this directory is **local** — no MPI:

| File | Produces |
|---|---|
| `triangle_geometry.c` | `tris.area`, `tris.normal`, `tris.center` for a mesh; the closest point on a triangle |
| `vertex_geometry.c` | area- or angle-weighted `verts.normal` for a mesh |
| `signed_distance.c` | signed distance to a mesh on a (periodic) lattice window |

The collective wrappers (`MFTLEnsembleUpdate`, `MFTLEnsembleUpdateVertexNormals`)
live in @ref mftl_mod_ensemble; the split is by *scope*, not by entity
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

$$
  \mathbf{N}_v^{\text{area}} \propto \sum_{f \ni v} A_f\,\hat{\mathbf{n}}_f
  = \tfrac12 \sum_{f\ni v} (\mathbf{e}_1\times\mathbf{e}_2)_f,
  \qquad
  \mathbf{N}_v^{\text{angle}} \propto \sum_{f \ni v} \alpha_f(v)\,\hat{\mathbf{n}}_f .
$$

**Area weighting** is free (the raw cross product carries it) and parallel to the
gradient of the enclosed volume on a closed fan,

$$
  \frac{\partial V}{\partial \mathbf{x}_v} = \tfrac13\sum_{f\ni v} A_f\,\hat{\mathbf{n}}_f ,
$$

the direction a volume-conserving front-tracking scheme moves vertices along. **Angle
weighting** (Thürmer–Wüthrich) is invariant under re-triangulation of the fan
and is the pseudo-normal of the signed-distance theorem below. The corner angles
come from one cross product and three dot products via `atan2`, which is well
conditioned where `acos` is not.

<!-- figure: vertex-normals
Normals. (a) One square root serves both $A = \tfrac12\lVert\mathbf{e}_1\times\mathbf{e}_2\rVert$
and $\hat{\mathbf{n}}$; the corners are counter-clockwise from outside, so
$\hat{\mathbf{n}}$ points out. (b, c) One uneven fan weighted by area (shade
$\propto A_f$) and by corner angle (wedges $\alpha_f$): here the two vertex
normals differ by 27°. The area weighting is the volume gradient; the angle
weighting is invariant under re-triangulation of the fan and is the
pseudo-normal the signed distance needs.
-->

The pass is a **scatter** over triangles (flat loop, compact connectivity) rather
than a gather over fans (dependent loads, measured 3.4× slower); a
vertex→triangle CSR gather was a dead heat and is not stored
(docs/analysis.md §1.5). `verts.normal` stays xyz-interleaved: the three-double
read-modify-write is four memory operations per vertex instead of six
(1.20–1.25×, `examples/layout_benchmark.c`).

### Exactness needs a vertex-complete halo

The scatter runs over **all** triangles, ghosts included, so an owned vertex is
exact iff every incident triangle is present locally — which the redistribution
halo guarantees (a triangle is sent to the owner of each of its corners). This
makes the pass communication-free.

@warning Ghost vertices hold a normalised **partial** sum, which looks exactly
         like a real normal. `MFTLEnsembleUpdateVertexNormals` adds the one ghost
         exchange that makes them exact.

---

## Closest point and signed distance {#geom_sdf}

`MFTLClosestPointOnTriangle` implements Ericson's Voronoi-region test: six dot
products select one of seven regions (face, three edges, three corners) and only
that region's formula is evaluated. It also returns the **feature**; edge $k$
is half-edge $3t+k$. A triangle with $\sin^2\theta \le 10^{-15}$ at corner 0 is
treated as its three edges.

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

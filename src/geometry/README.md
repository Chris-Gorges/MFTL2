`src/geometry/` — Periodic-aware geometry {#mftl_mod_geometry}
=========================================

**Public header:** `include/mftl_geometry.h` · **Layer:** 2 ·
**Module group:** @ref mftl_geometry

Two translation units, both **local** — no MPI anywhere in this directory:

| File | Produces |
|---|---|
| `triangle_geometry.c` | `tris.area`, `tris.normal`, `tris.center` for a whole mesh |
| `vertex_geometry.c` | `verts.normal` for a whole mesh |

The collective wrappers (`MFTLEnsembleUpdate`, `MFTLEnsembleUpdateVertexNormals`)
live in @ref mftl_mod_ensemble. The split is by **scope**, not by entity type —
see @ref arch_geometry_split.

Everything that reads a *position* is here or in the header; the small queries
(`MFTLGetEdgeVector`, `MFTLGetTriangleProperties`, `MFTLGetTriangleCorners`) are
`static inline` because their callers want them inlined into their own loops.

---

## The minimum image convention {#geom_minimum_image}

On a periodic axis the domain is a torus, so the difference of two stored
coordinates is **not** the vector between them. The shortest representative is

$$
  \Delta_k = (p_2 - p_1)_k - L_k \,\mathrm{round}\!\left(\frac{(p_2-p_1)_k}{L_k}\right)
$$

evaluated with `nearbyint`. Per axis that is one multiply, one round (a single
instruction on AArch64 and on SSE4.1+) and one fused multiply-subtract.
`MFTL_Domain::inv_domain_size` is precomputed so the division never appears.

### Why not the comparison form

The obvious alternative — `if (d > L/2) d -= L; else if (d < -L/2) d += L;` — is
two compare-multiply-add pairs, measurably more work in a hot loop, and it is
**correct only for $|d| < 1.5L$**, a precondition the round form does not need.
Ties at exactly $\pm L/2$ round to even, which is deterministic.

### Wrapping is the opposite trade

`MFTLApplyPeriodicity` *is* the comparison form, deliberately. It assumes the
displacement since the last wrap is smaller than the domain — true for any stable
simulation — and its branches are perfectly predicted. The branchless alternative
was measured **1.34–1.37× slower** in the per-timestep kernel, because it does
unconditional work on non-periodic axes too.

---

## Triangle area, normal and centroid {#geom_triangle}

With $\mathbf{e}_1 = \mathbf{p}_1 - \mathbf{p}_0$ and
$\mathbf{e}_2 = \mathbf{p}_2 - \mathbf{p}_0$ both taken under the minimum image
convention:

$$
  \mathbf{n} = \mathbf{e}_1 \times \mathbf{e}_2, \qquad
  A = \tfrac{1}{2}\lVert \mathbf{n} \rVert, \qquad
  \hat{\mathbf{n}} = \frac{\mathbf{n}}{\lVert \mathbf{n} \rVert}, \qquad
  \mathbf{c} = \mathbf{p}_0 + \tfrac13(\mathbf{e}_1 + \mathbf{e}_2)
$$

One square root serves both the area and the normalisation, and there is no
second square root anywhere in the loop.

The centroid is built from the *seam-corrected* offset and then wrapped, so it is
the true centroid even for a triangle straddling a periodic face — and it is a
wrapped position, which is what `MFTLGetOwnerProc` requires.

### A normal is always a unit vector {#geom_unit_normal}

Every kernel computes

```c
const double mag = MFTL_SQRT(nx*nx + ny*ny + nz*nz);   /* UNCLAMPED */
const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
```

Two spellings are wrong, and both were in the tree until 2026-08-15:

- **Clamping $\lVert\mathbf{n}\rVert^2$ before the square root** floors the
  magnitude at $\sqrt{10^{-12}} = 10^{-6}$ — and floors the derived area at
  $5\times10^{-7}$. So `quality.min_triangle_area` could never report a
  degenerate triangle, and $q = 4\sqrt3 A/(a^2+b^2+c^2)$ became unbounded above;
  an *average* $q$ of 1.023 was measured on a degenerate mesh.
- **Clamping the divisor** keeps the instruction count and only **moves** the
  floor: at `MFTL_VERYVERYSMALL` a triangle with $10^{-8}$ edges still came out
  with a normal of length 0.112.

A short normal is the most dangerous shape of wrong available here: **it points
the right way**, so every sign test, winding check and picture still passes,
while it silently scales surface tension and everything else built from it.

The test against zero has no cliff and costs one instruction (`fcmp`+`fcsel`
against `fmax`): **7.33 → 7.41 ns/tri, +1.1 %**, best of six over two rounds.

@warning The expression must stay **identical** in the walk and the stream
         kernels. `test_triangle_geometry_paths_are_bit_identical` compares the
         two paths with `memcmp`, so a difference of one operation here is a
         difference in the last bit there.

---

## Two connectivity paths, one entry point {#geom_dispatch}

`MFTLComputeTriangleGeometry` produces the same three outputs two ways, and which
one is available depends on the **state of the object**, not on what the caller
wants:

| Path | Reads | Valid when |
|---|---|---|
| `geometry_by_walk` | implicit half-edge indices and three gathered positions | **always**, including construction before derived arrays exist |
| `geometry_by_vertices` | compact `tris.verts` and three gathered positions | while #MFTL_DERIVED_TOPOLOGY is set |

The steady-state path unswitches periodicity outside the triangle loop. Its
non-periodic branch contains only compact-index loads, position gathers and
arithmetic; its periodic branch applies minimum-image corrections without a
per-triangle branch. The fallback preserves the construction-time API. The two
paths are bit-identical, tested on seam-straddling meshes.

---

## Area-weighted vertex normals {#geom_vertex_normals}

$$
  \mathbf{N}_v = \frac{\sum_{t \ni v} A_t\, \hat{\mathbf{n}}_t}
                      {\left\lVert \sum_{t \ni v} A_t\, \hat{\mathbf{n}}_t \right\rVert}
$$

Since $A_t\hat{\mathbf{n}}_t = \tfrac12 (\mathbf{e}_1 \times \mathbf{e}_2)$, the
area weight is **free** — the un-normalised cross product already carries it, and
the pass never divides per face.

### Scatter over triangles, not gather over vertices

| | Shape | Cost |
|---|---|---|
| **gather** | per vertex, walk the one-ring, recompute incident face normals | data-dependent trip count; every fan step is a *dependent* load (`he → twin → next`) |
| **scatter** | per triangle, gather positions and add its face normal into three vertices | compact connectivity and a flat constant body, but a read-modify-write into three indexed slots plus zeroing and normalisation passes |

### The CSR question, settled

The vertex→triangle CSR remains absent: it would add 16 bytes per triangle and a
topology rebuild solely to offer another adjacency traversal. Reconsider it only
with an end-to-end workload that amortises both costs.

### Why the scatter target stays interleaved

`verts.normal` is xyz-interleaved, and for *this* loop that is measurably right
rather than inherited. The read-modify-write of three contiguous doubles compiles
to a 128-bit load, a 2-wide vector add and a 128-bit store for $x,y$ plus one
scalar triple for $z$ — **four memory operations per vertex**. Against three
separate component arrays it would be six, because three separate load-add-stores
cannot pair. `examples/layout_benchmark.c` measures **1.20–1.25× in favour of
interleaved**, consistently and at every size.

### Exactness needs a vertex-complete halo

The pass accumulates over **all** triangles including ghosts, so an owned vertex
gets the exact answer *iff* every triangle incident to it is present locally.
That is what the redistribution's corner clause supplies, and it is what buys
this pass **zero communication**.

@warning Ghost vertices hold a **partial sum**, correctly normalised and quietly
         wrong — it looks exactly like a real normal.
         `MFTLEnsembleUpdateVertexNormals` closes that with one ghost exchange.

---

## Shared gathered loop {#geom_layering}

The bulk triangle and vertex-normal kernels use
#MFTL_FOREACH_TRIANGLE_GEOMETRY. Periodicity unswitching and compact-index
gathers therefore have one implementation shared with external hot loops.

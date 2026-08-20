`src/mesh_creation/` — Procedural generation {#mftl_mod_meshgen}
============================================

**Public header:** `include/mftl_meshgen.h` · **Layer:** 3 ·
**Module group:** @ref mftl_meshgen

| File | Job |
|---|---|
| `mesh_builder.c` | indexed triangle list → half-edge mesh. The single conversion point. |
| `sphere_creation.c` | the icosphere: an icosahedron plus $r$ rounds of 1-to-4 subdivision |

There is no mesh **reader** in this library, of any format. When one is written,
it goes through `MFTLBuildIndexedMesh`.

---

## One conversion point {#meshgen_one_builder}

Every generator that is not built by subdivision goes through
`MFTLBuildTopologyFromIndexList`, and so would a reader. Keeping one
implementation is what stops the half-edge conventions from being re-derived
slightly differently in each new generator:

- slot $k$ is the **origin** of half-edge $3t+k$;
- global IDs are dense per class, and a half-edge's is $3\,t_{\mathrm{gid}} + k$;
- twins linked, boundary recorded, derived arrays built.

What it does, in order:

1. `MFTLObjectSetCounts` — grows the triangle pool, and with it the half-edge
   array and every derived array, because they are **one allocation**.
2. Sets `next_triangle_gid = n` and `next_halfedge_gid = 3n`. Vertex IDs are
   deliberately **not** reset: the caller owns them, and subdivision carries old
   vertices over verbatim.
3. Clears every `out_he_idx` before re-deriving it — a stale index there would
   point into someone else's fan, which matters because this is called again for
   each subdivision level.
4. Writes each triangle's three half-edges, then `MFTLRebuildTwins`,
   `MFTLMarkMeshBoundary` and `MFTLObjectBuildDerived`.

`MFTLBuildIndexedMesh` is that plus the vertex-position copy, for callers holding
a complete `(positions, indices)` pair.

---

## The icosahedron {#meshgen_icosahedron}

Twelve vertices: the cyclic permutations of $(0, \pm 1, \pm\varphi)$ with
$\varphi = \tfrac{1+\sqrt5}{2}$, normalised onto the unit sphere. Twenty faces.

It is the starting point rather than an octahedron or a cube because it is the
**most regular** closed triangulation available: all twenty faces are congruent
equilateral triangles, so uniform subdivision inherits that regularity and
triangle quality does not degrade with $r$ — the mean ratio stays within a few
percent of the icosahedron's own.

### Face normals point outward, everywhere

The face table is wound counter-clockwise **as seen from outside**, and the
1-to-4 subdivision emits every sub-triangle in its parent's rotational order.

This is what makes the enclosed volume positive under the divergence theorem

$$
  V = \frac{1}{3}\oint_{\partial\Omega} \mathbf{x}\cdot\hat{\mathbf{n}}\,\mathrm{d}A
    = \frac{1}{3}\sum_{t} A_t\,\big(\mathbf{c}_t\cdot\hat{\mathbf{n}}_t\big)
$$

and it fixes the sign of surface tension and of the indicator function. Guarded
by `test_triangle_winding_is_outward` and
`test_enclosed_volume_is_positive_and_converges`.

---

## 1-to-4 subdivision {#meshgen_subdivision}

```
      v2
     /  \
   m2 -- m1
   / \  / \
 v0 -- m0 -- v1
```

Each triangle becomes four; each new vertex is the midpoint of an edge, projected
back onto the sphere.

### Counts

From Euler's formula $V - E + F = 2$ with $3F = 2E$ for a closed triangulation:

$$
  F_r = 20 \cdot 4^{\,r}, \qquad
  E_r = 30 \cdot 4^{\,r}, \qquad
  V_r = 10 \cdot 4^{\,r} + 2
$$

| $r$ | vertices | triangles |
|---|---|---|
| 0 | 12 | 20 |
| 1 | 42 | 80 |
| 2 | 162 | 320 |
| 3 | 642 | 1 280 |
| 4 | 2 562 | 5 120 |
| 5 | 10 242 | 20 480 |

Edge length scales as $2^{-r}$, so the target refinement for a given resolution
is $r \approx \log_2(L_{\mathrm{ico}}/L_{\mathrm{target}})$.

### Midpoints without a hash map {#meshgen_no_hash}

The usual implementation keys a midpoint on the unordered vertex pair $\{u,v\}$
and needs a hash map to find it again from the other incident triangle.

Here the edge a midpoint belongs to is identified by a **half-edge index**, and
`midpoint_map[h]` is filled for both $h$ **and its twin** at once. The second
half-edge of an edge then finds the vertex its partner created by a direct array
read.

$O(1)$ per half-edge with no hashing — and it is the clearest illustration of why
this library needs no hash maps at all: **on a half-edge mesh, "the other side of
this edge" is already an index.**

A closed mesh is required (every icosphere is one); a boundary half-edge would
have no twin to record the shared midpoint through.

### It produces an indexed list, so it needs no topology code

Subdivision computes midpoints and emits the sub-triangles as vertex indices;
`MFTLBuildTopologyFromIndexList` turns them into topology.

MFTL 5 hand-built 4 triangles and 12 half-edges per old triangle, duplicating
(slightly differently) the loop in `mesh_builder.c` — which is exactly how two
code paths drift apart.

It also no longer allocates and swaps three parallel arrays per level. Old
vertices keep their indices under subdivision (midpoints are **appended**), so the
vertex pool is grown in place and only the triangle pool is rewritten.

---

## Generation is collective, and rank 0 holds everything {#meshgen_collective}

`MFTLGenerateSphere` is collective, but the mesh it produces lives **entirely on
rank 0** — every other rank ends with an empty object. Spreading it is
`MFTLRedistribute`'s job, and separating the two is what keeps the generator free
of any partitioning concern.

Two consequences worth stating:

- The scatter from rank 0 reaches ranks that are **not** its neighbours, which is
  what forces redistribution's global communication path at $P \ge 8$. That is
  the case `test_redistribution_np8` exists for.
- A rank holding nothing must still enter every collective. `MFTLMeanEdgeLength`
  on such a rank returns **0**, which is why deriving a remeshing target from it
  after a collective generation is a real hazard — see
  `MFTLEnsembleMeanEdgeLength` and `MFTLRemeshParamsAgree` in
  @ref mftl_mod_remeshing.

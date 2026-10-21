`src/object/` — The mesh container {#mftl_mod_object}
==================================

**Public header:** `include/mftl_object.h` · **Layer:** 3 ·
**Module group:** @ref mftl_object

| File | Owns |
|---|---|
| `object.c` | the pool lifecycle — **the only file allowed to write a pool pointer, a `capacity` or a `block`** |
| `derived.c` | the derived connectivity arrays and the Morton renumbering of both pools |

---

## One pool is one allocation {#object_one_block}

An entity class needs an AoS record array *and* several SoA geometry arrays that
describe the same entities, so all of them must always have the same capacity.
Rather than keeping several capacity fields in step by convention, each pool is
**one** 64-byte-aligned block. `vertex_layout_compute` / `triangle_layout_compute`
walk one cursor through the fields,

$$
  o_0 = 0, \qquad
  o_{i+1} = L\left\lceil \frac{o_i + n_i s_i}{L} \right\rceil,
  \qquad L = \texttt{MFTL\_CACHE\_LINE} = 64,
$$

($n_i$ elements of size $s_i$), and the final cursor is the block size:

```
vertex block    rec[cap]  pos[3cap]  normal[3cap]                          64 B / vertex
triangle block  rec[cap]  he[3cap]  area[cap]  normal[3cap]  center[3cap]
                verts[3cap]  neighbors[3cap]                              140 B / triangle
```

One allocation, one `capacity`, one `free`: "the AoS grew but an SoA array did
not" cannot be expressed. The half-edges live in the triangle block because a
triangle has exactly three, at $3t\ldots3t+2$ (@ref arch_inv_implicit); a
separate half-edge capacity would describe an impossible mesh. Cache-line
alignment keeps arrays written by different kernels on different lines.

An icosphere of refinement $r$ ($F = 20\cdot4^r$, $V = 10\cdot4^r+2$) therefore
occupies $140F + 64V = 3440\cdot4^r + 128$ bytes, padding aside.

<!-- figure: pool-layout
The two blocks of an object, to scale in bytes per entity: 64 B per vertex and
140 B per triangle, every sub-array on its own 64 B cache line. (b) The access
pattern each layout serves: an operator edits a few records and most of each; a
geometry pass streams one component of every entity.
-->

---

## Optional fields {#object_fields}

Register `double` arrays collectively on the ensemble **before adding the first
body**. All ranks register the same names, entity classes, component counts and
modes in the same order. Registration returns a dense field id, or `MFTL_ERROR`
on all ranks for invalid or mismatched descriptors. The registry is fixed for
the ensemble's lifetime and applies to every body.

```c
int gamma = MFTLEnsembleRegisterField(ens, "concentration",
    MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_CONSERVATIVE_DENSITY);
int velocity = MFTLEnsembleRegisterField(ens, "velocity",
    MFTL_FIELD_VERTEX, 3, MFTL_FIELD_RECOMPUTED);
/* Add/generate bodies, then fill owned rows. */
double *c = MFTLObjectField(obj, gamma); /* c[t] */
double *u = MFTLObjectField(obj, velocity); /* u[3*v + component] */
```

| Mode | Storage and movement | Remeshing |
|---|---|---|
| `MFTL_FIELD_RECOMPUTED` | Allocated, initially zero; not redistributed | Contents unspecified afterwards; caller recomputes |
| `MFTL_FIELD_INTERPOLATED` | Persistent; follows entities through all permutations and migrations | Interpolated samples, no integral guarantee |
| `MFTL_FIELD_CONSERVATIVE_DENSITY` | Persistent, non-negative **triangle** density | Preserves owned `sum(area * value)` componentwise |

Each optional array is appended, cache-line aligned, to its entity pool's
existing allocation. There is no second capacity and no allocation per field.
Reserve/shrink preserve all live rows, including recomputed ones. The registry
owns descriptors; the object owns only a pointer table and its two pools.
An empty registry allocates no optional arrays or pointer table.

Reacquire field pointers after reserve, shrink, redistribution or reconstruction.
Reorder and remeshing also change entity indices; remeshing can renumber global
ids. Recomputed values must be filled whenever their physical inputs change,
as well as after mesh changes. MFTL does not infer their validity from a mesh
generation stamp.

Update owners first and use `MFTLUpdateGhostFields(ens, ids, nfields, false)` to
refresh selected ghost rows, including recomputed fields. Persistent **vertex**
ghosts must be current before redistribution: several triangle owners can send
copies of the same vertex. Pass `true` to bundle positions, then recompute
triangle geometry before redistribution. Field-only sync leaves the surface
generation unchanged.

Triangle density conservation here applies to library remeshing. The application
still owns density changes during physical motion, including area dilution,
sources and diffusion. Remapping uses local constant reconstruction, patch mass
corrections and shared-edge smoothing fluxes; it remains diffusive. See
@ref rem_fields and `examples/attached_fields.c` for its measured accuracy.

---

## The size API {#object_size_api}

```c
MFTLObjectReserve(obj, nv, nt);            /* exact; for bulk loads               */
MFTLObjectReserveAdditional(obj, dv, dt);  /* geometric; for incremental appends  */
MFTLObjectSetCounts(obj, nv, nvl, nt, ntl);/* reserves, then assigns              */
MFTLObjectShrinkToFit(obj, slack);         /* the only way down, explicit + gated */
MFTLObjectReleasePools(obj);               /* free both blocks of an empty object */
```

`MFTLObjectSetCounts` **reserves before it assigns**, so `n <= capacity` holds
unconditionally. It also clears the derived bit and stamps the body
*partition-dirty*: every entity creation or destruction passes through it, which
is what the redistribution's skip of unchanged bodies relies on
(MFTLObjectMarkPartitionDirty()).

**Growth is a ratchet.** The first three calls never lower a capacity, so a mesh
of stable size does not re-fault its pages every timestep, and redistribution
pre-grows a body and relies on the next reserve returning early.
`MFTLObjectShrinkToFit` is the only way down: a pool with
$\mathrm{capacity} > s\,n$ is moved to the smallest ladder capacity
$128\cdot2^k \ge n$. Because growth and shrink both leave
$\mathrm{capacity} \in [n, 2n)$, a gate $s = 2$ fires on any decrease and a
count dithering around a power of two would reallocate every step; $s = 4$
(#MFTL_SHRINK_SLACK_DEFAULT) re-fires only when the count halves, bounding the
number of reallocations over a run by $O(\log(n_{max}/n_{min}))$
(`test_shrink_to_fit_does_not_thrash_against_the_grow_ladder`). The shrink
changes no count, content, derived bit or partition stamp.

@warning Any call that can resize a pool may move both blocks. Raw pointers into
         `obj->verts.*` or `obj->tris.*` taken before it dangle afterwards —
         hold indices.

---

## Derived connectivity {#object_derived}

| Bit | Arrays | Rebuilt by | Invalidated by |
|---|---|---|---|
| #MFTL_DERIVED_TOPOLOGY | `tris.verts`, `tris.neighbors` | `MFTLObjectBuildDerived` (and redistribution's rebuild) | any topology change |
| #MFTL_DERIVED_GEOMETRY | `tris.area`, `tris.normal`, `tris.center` | `MFTLComputeTriangleGeometry(obj, tris.n)` | any topology change, a rebuild, `MFTLUpdateGhostPositions`, the remeshing operators |

The geometry bit is a lower bound, not a certificate. Moving owned vertices
through `verts.pos` does not clear it, because the library cannot see that; so a
clear bit means "not current" and a set bit means "current unless the caller
moved vertices since". `MFTLRedistribute` recomputes a body whose bit is clear
before routing it by its centres.

Both arrays are functions of the half-edge records:

```
verts[3t+k]     == he[3t+(k+2)%3].vertex_idx     (origin of half-edge 3t+k)
neighbors[3t+k] == he[3t+k].twin_idx / 3         (MFTL_ERROR without a twin)
```

They are redundant and stored for **traffic**: a connectivity-only loop (I/O,
vertex-normal scatter, redistribution's halo analysis, BVH build) streams
12 B per triangle instead of the 48 B of three half-edge records. The 4× figure
is arithmetic, not a measurement (docs/analysis.md).

**No position-derived array is stored.** Geometric consumers stream `verts`,
gather `verts.pos` and form minimum-image edges in registers
(#MFTL_FOREACH_TRIANGLE_GEOMETRY), so vertex motion invalidates nothing. Stored
per-triangle frames (72 B/triangle) were measured and removed: their build cost
did not amortise in moving or remeshed workflows (docs/analysis.md §1.2). A
vertex→triangle CSR was measured as a dead heat against the triangle scatter for
vertex normals while costing 16 B/triangle and a rebuild per topology change
(§1.5), and is not stored either.

---

## Morton renumbering {#object_morton}

`MFTLObjectReorderMorton` sorts both pools along a Z-order curve. Positions are
quantised to 21 bits per axis against the object's bounding box with one common
scale (cubic cells), and the bits are interleaved into a 63-bit key:

$$
  Z(x,y,z) = \sum_{i=0}^{20} \Big( x_i\,2^{3i} + y_i\,2^{3i+1} + z_i\,2^{3i+2} \Big).
$$

Points in a common octree cell share a key prefix, so sorting by $Z$ places
spatial neighbours close in memory. `MFTLMortonSpread3` performs the interleave
with five shift-and-mask steps.

<!-- figure: morton
Morton order, in 2D for legibility (the library interleaves 21 bits of each of
three axes). (a) Positions are quantised against the box with one common scale
and ordered along the Z-curve; each vertex is coloured by its resulting pool
index. (b) The key interleaves the coordinate bits, so points in one octree
cell share a key prefix. (c) Every edge of the same mesh as an arc between the
pool indices of its endpoints: for an arbitrary numbering, and after the
renumbering.
-->

1. Key per vertex; stable radix sort of `[0, n_local)` and, separately,
   `[n_local, n)` — the ownership prefix survives.
2. Permute every vertex array; remap `he.vertex_idx`.
3. Order triangles by their **smallest** (already Morton-ordered) vertex index —
   spatial by construction, no centroid (and no question of what a centroid is
   across a periodic seam), independent of which corner is corner 0. Records,
   half-edge triples and geometry arrays move together; `twin_idx` and
   `out_he_idx` keep their corner $h \bmod 3$ and remap $\lfloor h/3\rfloor$.
4. Rebuild the derived arrays.

The permutation is **applied to the pools**, never stored, so there is one index
space and one copy of each coordinate (@ref arch_inv_positions). Global ids are
untouched; pool indices are not stable across the call. A body straddling a
periodic seam gets a domain-wide box and coarser cells — a locality cost only.
The call is local (no MPI).

@note No wall-clock benefit of the renumbering has been measured on the
      development machine (Apple silicon), although it reduces the mean edge
      index distance 160×. It is kept for the single index space and as the LBVH
      prerequisite; re-measure on an x86 node with 1–2 MiB L2 per core.

---

## Hot-loop iterators {#object_iterators}

| Macro | Streams | In scope for the body |
|---|---|---|
| #MFTL_FOREACH_TRIANGLE_GEOMETRY | `tris.verts`, gathered `verts.pos` | `t`, `v0..v2`, corner `p0`, minimum-image edges `e1`, `e2` |
| #MFTL_FOREACH_VERTEX_POS | `verts.pos`, read-only | `v`, `px, py, pz` |
| #MFTL_FOREACH_OWNED_VERTEX | `verts.pos`, writable, `[0, n_local)` | `v`, `pos`, `rec` |

They are macros because C has no closures: the body is pasted into one flat loop
with `restrict` locals, a hoisted bound and no pointer chasing. The triangle
iterator decides periodicity once per call; its periodic branch applies
$d - L\,\mathrm{nearbyint}(d/L)$ without a per-triangle branch.

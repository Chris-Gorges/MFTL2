`src/object/` — The mesh container {#mftl_mod_object}
==================================

**Public header:** `include/mftl_object.h` · **Layer:** 3 ·
**Module group:** @ref mftl_object

Two translation units:

| File | Owns |
|---|---|
| `object.c` | the pool lifecycle. **The only file in MFTL allowed to write a pool pointer, a `capacity`, or a `block`.** |
| `derived.c` | the rebuildable per-triangle arrays, and the Morton renumbering of both pools |

---

## One pool is one allocation {#object_one_block}

A mesh entity class needs an AoS record array *and* a family of SoA geometry
arrays, and they describe the same entities, so they must always have the same
capacity.

MFTL 5 expressed that as a **rule**: three capacity fields (`capacity_*`,
`geom_cap_*`, `flat_cap_*`) and a comment saying every growth site must update
all three. A growth site that forgot one produced a heap overflow — not a wrong
number, a corruption somewhere else entirely.

Here the rule is replaced by a **structure**. `vertex_layout_compute` and
`triangle_layout_compute` walk one cursor through every field:

$$
  o_0 = 0, \qquad
  o_{i+1} = \left\lceil \frac{o_i + n_i s_i}{L} \right\rceil L,
  \qquad L = \texttt{MFTL\_CACHE\_LINE} = 64
$$

and the total is the block size. One `aligned_alloc`, one `capacity`, one pointer
to free. **"The AoS grew but the SoA did not" is not a bug that can be made — it
is not expressible.**

```
vertex block    rec[cap]  pos[3cap]  normal[3cap]  area[cap]  mean_curv[cap]  gauss_curv[cap]
triangle block  rec[cap]  he[3cap]   area[cap]  normal[3cap]  center[3cap]
                verts[3cap]  neighbors[3cap]  anchor[3cap]  edge[6cap]
```

### Field order is not arbitrary

`rec` and `pos` come first in the vertex block because they are the two arrays
every pass touches, so they share the front of the block and the TLB entries that
go with it.

### Why the half-edges live in the triangle block

Not packaging. A triangle has exactly three half-edges, at $3t \ldots 3t+2$
(@ref arch_inv_implicit), so a half-edge capacity that could differ from
$3\times$ the triangle capacity would describe **a mesh that cannot exist**.
Giving them their own pool would recreate exactly the problem this layout removes.

### What 64-byte alignment buys

- A streaming kernel over one component never shares a cache line with a
  neighbouring component, so a write to `tris.area` cannot false-share with a
  read of `tris.normal`.
- 64 bytes is a whole number of every element type in the pools (8 `double`,
  16 `int`, 4 #MFTL_Vertex records), so an aligned sub-array start is also an
  aligned *element* start and the vectoriser never needs a peel loop.

---

## The size API {#object_size_api}

```c
MFTLObjectReserve(obj, nv, nt);            /* exact, for bulk loads          */
MFTLObjectReserveAdditional(obj, dv, dt);  /* geometric, for incremental use */
MFTLObjectSetCounts(obj, nv, nvl, nt, ntl);
```

`MFTLObjectSetCounts` **reserves before it assigns**, so the postcondition
`n <= capacity` holds unconditionally. That is the function that makes "counts
and capacity disagree" unreachable, and it is why nothing else needs to check.

Setting a count invalidates the derived arrays: they describe a different number
of entities than they did a moment ago.

@warning Every pool-growing call **may reallocate both blocks**. Any raw pointer
         into `obj->verts.*` or `obj->tris.*` taken before such a call is
         dangling afterwards. Hold indices.

---

## Derived arrays {#object_derived}

`derived.c` maintains four per-triangle arrays that are *functions of the mesh*,
tracked by a two-bit word:

| Bit | Arrays | Rebuilt by | Invalidated by |
|---|---|---|---|
| #MFTL_DERIVED_TOPOLOGY | `verts`, `neighbors` | `build_topology_arrays` | any topology change |
| #MFTL_DERIVED_FRAMES | `anchor`, `edge` | `build_frames` | topology change **or** vertex motion |

### They are redundant, and that is the price

Both connectivity arrays are functions of the half-edge records at the same index:

```
verts[3t+k]     == he[3t+(k+2)%3].vertex_idx     (a fixed rotation)
neighbors[3t+k] == he[3t+k].twin_idx / 3         (a fixed arithmetic op)
```

They are stored because of **traffic**, not instructions. A kernel that wants only
connectivity — I/O, a triangle-to-vertex scatter, redistribution's halo analysis,
a BVH build — streams **12 bytes per triangle** out of one of these arrays.
Reading the same information from the half-edge records touches all three 16-byte
records, **48 bytes per triangle**, of which 36 are fields it does not want. On a
mesh that does not fit in cache, that 4× is the whole cost of the loop.

The redundancy is bounded and cheap to pay because both arrays are rebuilt by one
function, from one pass, and are marked stale by a single bit the moment topology
changes.

@note The 4× figure is arithmetic, not a measurement — see deferred issue 3 in
      `CLAUDE.md`. Re-examine when a consumer that isolates them exists.

### The frames

`anchor` holds corner 0; `edge` holds
$\{\mathbf{e}_1, \mathbf{e}_2\} = \{\mathbf{p}_1 - \mathbf{p}_0, \mathbf{p}_2 - \mathbf{p}_0\}$, **always
seam-consistent** — the minimum-image correction is applied once here, at frame
build time, and never again downstream. That is what lets
`MFTLComputeTriangleGeometry`'s streaming kernel do no periodic arithmetic at all.

### What was deleted, and why it stays deleted

MFTL 5 carried ten flattened arrays. The vertex→triangle **CSR** is the one worth
naming: `derived.c` originally asked for it to be reconsidered when a consumer
appeared. Vertex normals are that consumer, and §5 of
`docs/performance-analysis.md` measures a CSR gather as a **dead heat** with the
triangle scatter while costing 16 B/triangle of permanent state and a rebuild per
topology change. It stays deleted.

---

## Morton renumbering {#object_morton}

`MFTLObjectReorderMorton` sorts both pools along a Z-order curve. A position is
quantised to 21 bits per axis against the object's own bounding box, and the bits
are interleaved:

$$
  Z(x,y,z) = \sum_{i=0}^{20} \Big( x_i\,2^{3i} + y_i\,2^{3i+1} + z_i\,2^{3i+2} \Big)
$$

giving a 63-bit key in one `uint64_t`. Two points close in space have keys
agreeing in their high bits, so sorting by $Z$ puts spatial neighbours near each
other in memory.

The interleave itself (`MFTLMortonSpread3`) is the classic five-step magic-number
bit spread — $O(\log b)$ instead of the 21 iterations a bit-by-bit loop takes.

### What it does, in order

1. Morton key per vertex from `verts.pos`, quantised against the object's own
   bounding box so the resolution is spent where the mesh is.
2. A **stable sort within `[0, n_local)` and, separately, within
   `[n_local, n)`**, so the ownership prefix survives.
3. Permute every vertex array; remap `he.vertex_idx`.
4. Order triangles by their **smallest** (already Morton-ordered) vertex index —
   a spatial key by construction, needing no second geometric pass, no centroid,
   and no question about what a centroid means across a periodic seam. Using
   `min()` rather than the first corner makes the key independent of which corner
   the generator happened to call corner 0.
5. Remap `he.twin_idx` and `rec.out_he_idx`, then rebuild the derived arrays.

### This is where `sfc_perm` went

MFTL 5 stored a permutation *and* a second, SFC-ordered copy of every position.
That forced a second index space, a sync function, a staleness flag and an
"unwrapped or not" mode flag onto every consumer — and its own notes record it as
having caused a live silent-corruption bug.

Here the permutation is **applied to the pools**. Both the AoS records and every
SoA array move together and all indices are remapped, so afterwards there is
still exactly one index space and exactly one copy of every coordinate.

Global IDs are **not** touched: they are the stable name of an entity across
ranks and across reorderings. Pool **indices** are, by definition, not stable
across this call.

### A straddling object's bounding box legitimately spans the domain

The quantisation box is a plain min/max over the object's positions, so a body on
a periodic seam gets a box the size of the domain and the Morton cells coarsen
accordingly. That is an accepted cost, and it is the same accepted cost the BVH's
root box carries for the same reason.

@note **Morton renumbering has no proven benefit on the development machine.**
      It is kept because it is what makes one index space possible (its real
      justification) and because it is a prerequisite for an LBVH — not because
      it was measured to be faster. It cuts mean edge index distance 160× and
      buys no measurable time on Apple silicon; **needs re-measuring on an x86
      node with 1–2 MiB L2 per core.**

---

## The hot-loop iterators {#object_iterators}

Four macros in the header, each pinned to one access pattern:

| Macro | Streams | Asserts |
|---|---|---|
| #MFTL_FOREACH_TRIANGLE_EDGE | `tris.edge` | frames current |
| #MFTL_FOREACH_TRIANGLE_FRAME | `tris.anchor` + `tris.edge` | frames current |
| #MFTL_FOREACH_VERTEX_POS | `verts.pos`, read-only | — |
| #MFTL_FOREACH_OWNED_VERTEX | `verts.pos`, writable, `[0, n_local)` | — |

#MFTL_FOREACH_OWNED_VERTEX **clears** #MFTL_DERIVED_FRAMES on the way out, which
is what makes it impossible for a motion pass written with the iterator to leave
geometry a timestep stale. That is the whole reason to prefer it over a bare loop
with `MFTLVertexPosMut`.

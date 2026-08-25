Architecture {#mftl_page_architecture}
============

[TOC]

# The one idea {#arch_one_idea}

A mesh is described by two things that want opposite memory layouts, and each is
stored in the layout it wants.

**Topology and identity → Array of Structures.** Who is my twin, which vertex do
I end at, what is my global id, am I on the boundary. A remesher edits *one*
entity at a time and touches most of its record when it does; an edge collapse is
a handful of random accesses, not a sweep. AoS is right for that, and the records
are deliberately small so one cache line carries four of them:

| Record | Size | Fields |
|---|---|---|
| #MFTL_Vertex | 16 B | `out_he_idx`, `global_id`, `owner_proc`, `flags` |
| #MFTL_Triangle | 12 B | `global_id`, `owner_proc`, `flags` |
| #MFTL_HalfEdge | 16 B | `vertex_idx`, `twin_idx`, `global_id`, `flags` |

**Geometry → Structure of Arrays.** Positions, normals, areas, centroids,
curvatures, and the derived per-triangle frames. These are read and written by
kernels that touch *one* component of *every* entity, every timestep. That is the
access pattern AoS is worst at, and it is where all the time goes.

## Both live in one allocation per entity class {#arch_one_allocation}

#MFTL_VertexPool and #MFTL_TrianglePool each hold a single `block`, cut into
cache-line-aligned sub-arrays by MFTLLayoutAdd(), and each has exactly **one**
`capacity`:

```
MFTL_VertexPool   = one block  ->  rec[] pos[] normal[] area[] mean_curv[] gauss_curv[]
MFTL_TrianglePool = one block  ->  rec[] he[] area[] normal[] center[]
                                   verts[] neighbors[] anchor[] edge[]
```

The alternative — one capacity for the AoS records, a second for the geometry
arrays and a third for the derived ones, kept in step by a rule written in a
comment — fails at the first growth site that updates two of the three, and it
fails as a heap overflow rather than as an error. Here the AoS and every SoA
array are the same `malloc`, so **that class of bug is not expressible**.

@warning `src/object/object.c` is the **only** file allowed to write a pool
         pointer, a `capacity`, or a `block`. Everything else changes size
         through MFTLObjectReserve(), MFTLObjectReserveAdditional() or
         MFTLObjectSetCounts().

---

# Three invariants every module may assume {#arch_invariants}

## 1. Half-edges are triangle-indexed {#arch_inv_implicit}

Half-edge \f$3t+k\f$ belongs to triangle \f$t\f$, and

\f[
  \mathrm{next}(3t+k) = 3t + (k+1) \bmod 3
\f]

`he_idx`, `next_idx` and `triangle_idx` are therefore *not stored* — they are
arithmetic (MFTLTriFirstHe(), MFTLHeTriIdx(), MFTLHeNextIdx(), MFTLHePrevIdx()).
The half-edge array lives inside the triangle pool at 3× its capacity.

@warning Anything that creates a triangle must create its three half-edges at
         \f$3t\f$, \f$3t+1\f$, \f$3t+2\f$.

## 2. Ownership is a prefix {#arch_inv_prefix}

Entities \f$[0, \nlocal)\f$ are owned by this rank; \f$[\nlocal, n)\f$ are
ghosts. There is no `is_ghost` field — it would be a second copy of a fact the
index already carries. `owner_proc` survives only to answer *"whose is it"* for a
ghost, which the index cannot.

One notion of ownership, not three. An index test, an `is_ghost` flag and an
`owner_proc == rank` comparison can all be made to answer the question, and any
two of them can disagree; the prefix is the only one that cannot go stale,
because it *is* the storage order.

## 3. Positions live in the SoA and only there {#arch_inv_positions}

`verts.pos[3*v+k]` is authoritative. There is no second copy and no second index
space: the Morton ordering is applied to the *pool* (MFTLObjectReorderMorton()),
not stored as a permutation. Storing it as a permutation (`sfc_perm` /
`sfc_inv`) is the trap this avoids: a second index space forces a sync function,
a staleness flag and an "unwrapped or not" mode flag onto every consumer, and
every one of those is silent when it is wrong.

---

# Derived arrays and staleness {#arch_derived}

`tris.verts`, `tris.neighbors`, `tris.anchor` and `tris.edge` are rebuildable
caches, tracked by `obj->derived`:

| Bit | Arrays | Invalidated by |
|---|---|---|
| #MFTL_DERIVED_TOPOLOGY | `verts`, `neighbors` | any topology change |
| #MFTL_DERIVED_FRAMES | `anchor`, `edge` | topology change **or** vertex motion |

MFTLObjectBuildDerived() rebuilds both; MFTLObjectRefreshFrames() rebuilds only
the frames. Every `MFTL_FOREACH_*` iterator asserts on the relevant bit in Debug,
and #MFTL_FOREACH_OWNED_VERTEX clears the frame bit on the way out, so a motion
pass cannot silently leave geometry a timestep stale.

`tris.verts` and `tris.neighbors` are *redundant* with the half-edge records (a
rotation of `he.vertex_idx`, and `he.twin_idx / 3`). They are stored for
**traffic** — 12 B/triangle instead of 48 — not for instructions.

@note **There are no hash maps.** Global IDs are dense per entity class, and "the
      other side of this edge" is already an index — that is what a hash table
      would be doing work for. Do not reintroduce one without saying which lookup
      genuinely cannot be an array.

---

# Header layering {#arch_layering}

**One module = one `src/<module>/` directory = one `include/mftl_<module>.h`.**
Every `.c` includes its own header *first*, which makes headers self-contained by
construction. A header may only include headers from a strictly lower layer:

| Layer | Headers | Contains |
|---|---|---|
| 0 | @ref mftl_constants.h, @ref mftl_memory.h, @ref mftl_vectoroperations.h, @ref mftl_arrayfunctions.h | macros and plain-C helpers, no MFTL types |
| 1 | @ref mftl_types.h | all structs — the single place they are defined |
| 2 | @ref mftl_topology.h, @ref mftl_geometry.h | inline accessors over layer 1 |
| 3 | @ref mftl_object.h, @ref mftl_ensemble.h, @ref mftl_meshgen.h, @ref mftl_communication.h, @ref mftl_io.h, @ref mftl_remeshing.h, @ref mftl_bvh.h | module APIs |
| 4 | @ref mftl_core.h | umbrella for application code; library sources must not use it |

Two checks worth running after touching headers:

```bash
# every header must compile alone
for h in include/*.h; do printf '#include "%s"\n' "$(basename $h)" > /tmp/sc.c; \
  mpicc -std=c11 -fsyntax-only -Iinclude /tmp/sc.c || echo "NOT SELF-CONTAINED: $h"; done

# the library must export only MFTL* symbols, with no duplicates
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | grep -v _MFTL   # must be empty
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | sort | uniq -d  # must be empty
```

---

# Object model: Ensemble → Domain → Objects → pools {#arch_object_model}

- **#MFTL_Ensemble** — one #MFTL_Domain plus a list of `MFTL_Object*`.
  MFTLEnsembleCreate() is collective; MFTLEnsembleDestroy() cascades to every
  object it owns.
- **#MFTL_Domain** — the MPI/physical world shared by all objects: Cartesian grid
  (`cuts[3]`), `rank_map`, the 26-neighbour `comm_graph`, global bounds,
  periodicity. Objects hold a *borrowed* pointer, set by MFTLEnsembleAddObject().
- **#MFTL_Object** — one mesh, i.e. two pools plus ID counters and quality
  statistics.

@note Bodies are identified on the wire by their **index** in the ensemble's
      object list, not by `global_id`. Every rank must therefore hold the same
      object list in the same order; a Debug-only checksum reduction inside the
      redistribution reports a violation rather than silently merging entities
      into the wrong body.

---

# Where a geometry function lives {#arch_geometry_split}

Geometry appears in two modules and the split is by **scope**, not by entity
type:

| | one object, LOCAL, no MPI | every object, COLLECTIVE |
|---|---|---|
| triangles | MFTLComputeTriangleGeometry() (`src/geometry/`) | MFTLEnsembleUpdate() (`src/ensemble/`) |
| vertices | MFTLComputeVertexNormals() (`src/geometry/`) | MFTLEnsembleUpdateVertexNormals() (`src/ensemble/`) |

An `MFTLEnsemble*` function is a loop over the object list plus exactly the
collective its result needs; the arithmetic belongs to `src/geometry/`. The
naming carries the same information, so a call site says which it is without a
lookup.

The two collectives are of different **kinds**, and that is the interesting part:
the triangle statistics need a **reduction** (a minimum over one rank's partition
is not the mesh's minimum), while the vertex normals need an **exchange** (owned
normals are already exact; the ghosts are the half that is missing).

**One deliberate exception, and it is performance over tidiness.**
MFTLEnsembleUpdate()'s kernel is a file-static function in
`src/ensemble/ensemble_update.c` rather than in `src/geometry/`, because it
carries nine live reduction accumulators. Moving it across a translation-unit
boundary would force them into an address-taken struct, which
`docs/performance-analysis.md` §6 measures at 15 %. The kernel stays next to
the reduction it feeds.

---

# Conventions {#arch_conventions}

- Functions return `int` using `MFTL_SUCCESS` / `MFTL_ERROR`. `MFTL_ASSERT(...)`
  compiles to `assert()` in Debug and to nothing in Release — never put side
  effects inside it.
- `MFTL_ERROR` (−1) doubles as the "no such entity" index sentinel.
- Pool sizes change only through @ref mftl_object.h. Plain, genuinely independent
  arrays use #MFTL_ENSURE_CAPACITY (geometric) or #MFTL_RESERVE_ADDITIONAL
  (exact) from @ref mftl_memory.h. All allocation aborts on OOM by design.
- File-static helpers are `snake_case`; everything with external linkage is
  `MFTLPascalCase` (functions) or `MFTL_PascalCase` (types). Nothing but `MFTL*`
  may be exported.
- Formatting follows `.clang-format` (Google-based, 2-space indent, 160 columns,
  Allman braces).

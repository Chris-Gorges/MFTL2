`src/topology/` — Half-edge connectivity {#mftl_mod_topology}
========================================

**Public header:** `include/mftl_topology.h` · **Layer:** 2 ·
**Module group:** @ref mftl_topology

Navigation is `static inline` in the header (an arithmetic operation and a load
each). `topology.c` holds the four whole-mesh operations:

| Function | Job |
|---|---|
| `MFTLRebuildTwins` | pair every half-edge with its opposite, from vertex indices alone |
| `MFTLMarkMeshBoundary` | flag the half-edges and vertices on the surface's own boundary |
| `MFTLCheckMeshIntegrity` | validate the structure |
| `MFTLLabelConnectedComponents` | edge-connected components (used by topology changes) |

---

## The half-edge structure with implicit indexing {#topo_implicit}

An oriented triangle mesh is stored as *half-edges*: every edge is split into two
oppositely directed halves, one per incident triangle, and each triangle's three
half-edges run counter-clockwise seen from outside. The classical structure
stores $\mathrm{target}(h)$, $\mathrm{twin}(h)$, $\mathrm{next}(h)$ and
$\mathrm{face}(h)$ per half-edge. MFTL stores only the first two
(`vertex_idx`, `twin_idx`), because half-edge $3t+k$ **is** corner $k$ of
triangle $t$ — an invariant of every pool, not a convention:

$$
  \mathrm{face}(h) = \lfloor h/3 \rfloor, \qquad
  \mathrm{next}(h) = 3\lfloor h/3 \rfloor + (h+1) \bmod 3, \qquad
  \mathrm{prev}(h) = 3\lfloor h/3 \rfloor + (h+2) \bmod 3,
$$

$$
  \mathrm{origin}(h) = \mathrm{target}(\mathrm{prev}(h)).
$$

Consequences:

1. A broken next-cycle or a half-edge that does not point back at its triangle
   cannot be represented; the integrity check has nothing to test there.
2. The half-edge array needs no capacity of its own (3× the triangle pool's, in
   the same block).
3. The global id of half-edge $k$ of triangle $g$ is $3g+k$ mesh-wide, so
   half-edge ids are never transmitted.
4. The price: triangles only.

`MFTLGetTriangleVertices` and `tris.verts` both return slot $k$ as the **origin**
of half-edge $3t+k$; the remesher's triangle writer follows the same convention.
Mixing conventions silently rotates triangles.

---

## Rebuilding twins {#topo_twins}

`MFTLRebuildTwins` reconstructs every `twin_idx` from the vertex indices: the
twin of $u \to v$ is the half-edge $v \to u$. Half-edges are bucketed by their
**origin** vertex in an intrusive linked list (`v_heads[u]`, `he_next[h]`); the
partner of $u \to v$ is the unlinked half-edge in the bucket of $v$ that ends at
$u$. A bucket is as long as the valence, so the cost is
$O(V + H\,\bar d)$ with one allocation and no hashing. Only unlinked candidates
are paired, which keeps the relation symmetric on non-manifold input (three or
more half-edges on one edge): the surplus stays twin-less and is reported by the
integrity check.

@warning Pairing is by vertex **slot**. Two bodies sharing slot numbering —
         foreign triangles appended to a pool — would be welded together. This
         is one reason a spatial ghost is a separate shadow object
         (@ref mftl_mod_communication).

---

## Boundary is a property of the surface, not of the partition {#topo_boundary}

| Statement | Meaning | Partition-dependent? |
|---|---|---|
| `twin_idx == MFTL_ERROR` | no partner in **this** partition — an open surface's rim *or* a ghost halo's outer edge | yes |
| `flags & MFTL_FLAG_BOUNDARY` | no partner anywhere in the mesh | no |

`MFTLMarkMeshBoundary` writes the flag once, while the mesh is still complete
(every generator calls it), on each twin-less half-edge and both its endpoints.
From then on it travels with the entity through redistribution and remeshing.
Re-deriving it from `twin_idx` on a partitioned mesh would turn halo rims into
fake boundaries.

---

## Walking a fan {#topo_fan}

From an outgoing half-edge $h$ of vertex $v$, the next outgoing half-edge
counter-clockwise, and its inverse, are

$$
  h' = \mathrm{next}(\mathrm{twin}(h)), \qquad
  h'' = \mathrm{twin}(\mathrm{prev}(h)).
$$

On an interior vertex the rotation closes; on a boundary vertex it runs out of
twins. `MFTLVertexFanStart` rewinds clockwise to the fan's first half-edge (on a
closed fan this costs one full lap, which the walkers' `is_inner` flag skips).

**A vertex on an open fan has one more neighbour than incident triangles**: the
closing edge is reachable only as an incoming half-edge.

| Walker | Result at a boundary vertex |
|---|---|
| `MFTLGetVertexOneRing` (triangles) | one per incident triangle |
| `MFTLGetVertexOneRingVertices` | all neighbours; the extra one is emitted first, preserving rotational order |
| `MFTLGetVertexOneRingHalfedges` | outgoing half-edges only — one short of the incident edges; the missing one is `MFTLHePrevIdx(first)` |

Edge iterations (lengths, smoothing, remeshing candidates) must use the vertex
walker or add the closing edge (`test_one_ring_at_a_boundary_vertex_has_one_extra_neighbour`).
Every rotation is bounded by #MFTL_MAX_VALENCE steps or the caller's buffer, so a
non-manifold fan asserts in Debug instead of looping.

---

## Integrity checking {#topo_integrity}

`MFTLCheckMeshIntegrity` verifies that every vertex has an in-range outgoing
half-edge that really starts at it (isolated vertices are errors), that every
target and twin index is in range, that twins are symmetric
($\mathrm{twin}(\mathrm{twin}(h)) = h$) and have swapped endpoints. Twin-less
half-edges are counted, not rejected.

It does not check that half-edges were created in aligned triples — nothing at
the type level prevents a routine from breaking implicit indexing.
`inv_implicit_indexing()` in `tests/test_mesh_invariants.h` checks it in the test
suite.

---

## Connected components {#topo_components}

`MFTLLabelConnectedComponents` runs a breadth-first search over twin links,
$O(T)$, seeding in ascending triangle index so that labels are a function of the
pool alone. The library assumes an #MFTL_Object is **one connected surface**; a
topology change uses the labelling to give each component its own body. On a
partitioned mesh the result is local — parts connected only through another
rank's triangles count separately.

---

## The flag word {#topo_flags}

All bits of `flags` are allocated in `mftl_topology.h`, including those only the
remesher reads.

| Bit | Value | Owner | Lifetime |
|---|---|---|---|
| #MFTL_FLAG_BOUNDARY | 0x01 | topology | persistent — a property of the surface |
| #MFTL_FLAG_DEAD | 0x02 | remeshing | removed by compaction |
| #MFTL_FLAG_VISITED | 0x04 | remeshing | cleared between sweeps |
| #MFTL_FLAG_FEATURE | 0x08 | remeshing | re-derived each pass; splits hand it to both halves |
| #MFTL_FLAG_CORNER | 0x10 | remeshing | re-derived each pass; vertices only |
| #MFTL_FLAG_FROZEN | 0x20 | remeshing | one parallel phase; derived from the partition, never transmitted |

Application code must not reuse 0x02 or 0x04 for its own markers.

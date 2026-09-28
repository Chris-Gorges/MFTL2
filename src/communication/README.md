`src/communication/` — Distributed memory {#mftl_mod_communication}
=========================================

**Public header:** `include/mftl_communication.h` · **Layer:** 3 ·
**Module group:** @ref mftl_communication

| File | Job |
|---|---|
| `redistribute.c` | *the mesh moved — who owns what now?* Re-partition and rebuild the halo |
| `ghost_sync.c` | *the mesh deformed — refresh my ghosts.* One per-vertex 3-vector, topology untouched |
| `spatial_halo.c` | *what of body B is near my share of body A?* A geometric halo as shadow objects |
| `exchange.c` | MPI datatypes, the non-blocking multi-payload all-to-all-v, the neighbourhood allgather |
| `comm_internal.h`, `comm_stats.c` | the lazy `int` list, the global-id table, the diagnostics accumulator |

---

## Ownership and the halo rule {#comm_redistribute}

A vertex belongs to the rank whose Cartesian cell contains its position; a
triangle to the rank whose cell contains its **cached centre** (a body whose
#MFTL_DERIVED_GEOMETRY bit is clear is recomputed first)
($i_k = \lfloor (x_k-\min_k)\,\mathrm{cuts}_k/L_k\rfloor$). An owned triangle is
*additionally* sent to the owner of each of its three **corners** and each of its
three **edge neighbours**; that surplus is the ghost layer. Two properties
follow and every other module relies on them:

- **Vertex completeness.** Every triangle incident to an owned vertex is present
  locally (the corner clause). Vertex normals are therefore exact on owned
  vertices without communication, and every fan a remesher walks around an
  unfrozen vertex is complete.
- **Edge completeness.** Every owned triangle has all three edge neighbours
  locally (the neighbour clause), so a missing twin on an owned triangle really
  is a mesh boundary.

Read backwards, the rule characterises replication locally:
$$
  \text{owned triangle replicated} \iff
  \exists\,\text{corner} \ge \texttt{verts.n\_local}\ \lor\ \exists\,\text{neighbour} \ge \texttt{tris.n\_local},
$$
which is the remesher's freeze test (@ref mftl_mod_remeshing).

<!-- figure: halo-rule
The halo rule on an open patch split between two ranks. (a) A vertex belongs to
the rank whose cell holds it, a triangle to the rank holding its centre.
(b) Rank 0's pool: its owned triangles plus every triangle it receives because
it owns a corner (corner clause) or an edge neighbour (neighbour clause) of it.
Every owned vertex has its complete fan (coral), so vertex normals need no
communication. (c) Read backwards: an owned triangle is replicated, and so
frozen for the remesher, exactly when one of its corners or neighbours is a
ghost here.
-->

### What the wire carries {#comm_wire}

Only topology, identity and vertex **positions** travel:

| Record | Contents | Size |
|---|---|---|
| #MFTL_PackedVertex | position, global id, new owner, flags, body id | 40 B |
| #MFTL_PackedTriangleComplex | global id, new owner, flags, body id, 3 target-vertex ids, 3 half-edge flags | 40 B |

Areas, normals and centres are functions of positions and topology and are not
sent (shipping them would triple the payload). Half-edge ids ($3\,t_\mathrm{gid}+k$),
`next`/face (arithmetic), twins and `out_he_idx` are reconstructed at the
receiver. `vertex_gid[k]` is the **target** of half-edge $3t+k$, not
`tris.verts[3t+k]` (its origin). Boundary flags travel because a partition
cannot re-derive them. The MPI datatypes are one or two contiguous blocks with
extents pinned to the C `sizeof` (`MPI_Type_create_resized`), guarded by static
assertions.

**Postcondition.** A redistributed object has valid topology, valid positions and
#MFTL_DERIVED_TOPOLOGY set; `tris.area/normal/center` and `verts.normal` are
**unspecified** until recomputed (@ref mftl_page_timestep).

---

## The redistribution algorithm {#comm_algorithm}

1. **Classification.** `compute_owners` streams `verts.pos` and `tris.center`
   through `MFTLGetOwnerProc` (or the caller's #MFTL_OwnerMap) and writes the new
   owner into `owner_proc`, counting changes as it goes.
2. **Destination analysis.** Each owned triangle's destinations are its owner,
   its corners' owners and its neighbours' owners. Interior triangles, whose seven
   candidates coincide, take a register-only fast path. This rank is one
   destination among others: entities it keeps are flagged, not packed. Each
   destination's vertex list is de-duplicated on construction with one bit per
   (vertex, destination) — 32 slots suffice on the neighbour route; beyond them a
   list is sorted instead.
3. **Route.** One `MPI_Allreduce` (LOR) decides whether any destination lies
   outside the 26-neighbourhood. Graph route: `MPI_Neighbor_alltoallv` over
   `comm_graph`; global route (e.g. the initial scatter from rank 0):
   `MPI_Alltoallv` over `comm_cart`.
4. **Exchange.** `MFTLExchangeStart` sends three payloads — vertices, triangles,
   and a 7-int header per (body, destination) — after **one** merged count
   exchange. All bodies share the messages, so the message count is independent
   of the body count, and the headers let a receiver split the stream into per-body
   runs without sorting.
5. **Rebuild.** Kept entities are compacted in place into
   `[kept owned | new owned | kept ghost | new ghost]` (ownership stays a prefix);
   arrivals are de-duplicated by global id and appended. A twin between two kept
   triangles is a remap; only half-edges whose twin did not survive are re-paired.
   Kept-ness is decided per triangle, so this repair set is closed under the twin
   relation and the incremental repair finds every pair a full
   `MFTLRebuildTwins` would (checked by re-running it in
   `tests/test_redistribution.c`). `out_he_idx`, `tris.verts` and
   `tris.neighbors` are emitted inline. Bodies that arrive are created; bodies
   left empty are destroyed.

All scratch lives in a persistent per-ensemble workspace that only grows, so a
steady loop stops allocating after a few steps.

### The global-id table {#comm_gid_table}

"Do I already hold global id $g$, and in which slot?" is asked for every arriving
entity. Ids are dense per body, but a rank holds a scattered subset of a body's
range, so a direct-address array would be sized by the global count (measured
15–62 MB per rank per step at $P = 2$–8 on one 327 680-triangle body, 97 % empty
at $P = 8$). `mftl_gid_table` therefore has two modes, chosen per use:

- **direct** (`tag[gid]`) when $\text{range} \le 4\,n_\text{entries} + 64$;
- **sparse**: open addressing with linear probing, load factor $\le 1/2$, a
  multiplicative hash (the identity index clusters catastrophically on a
  partition's aliased id runs).

This is the library's only hash table. Entries carry an **epoch** in the high 32
bits of their tag, so retiring every entry is `epoch++`; nothing is ever deleted
(a probe stops at the first stale slot); capacity is a power of two in both
modes, so switching modes between bodies never reallocates.

### A body that did not move is not touched {#comm_skip}

A rank's state after a redistribution is a pure function of the global topology
and the owner map — read backwards, the halo rule *defines* the output. Positions
only decide the owner map. So a body for which neither input changed already
holds the answer, and `MFTLRedistributeSetSkipUnchangedBodies` (default on)
skips its analysis, packing and rebuild entirely (docs/analysis.md §2.4:
1.6–2.1× of the redistribution phase at 1000 bodies).

**The precondition is not "did topology change".** Under the remesher's freeze
discipline a topology change touches only unreplicated entities and is invisible
to the halo rule. What a skip breaks is **global ids and flags** of replicated
entities (renumbering relies on the rebuild to refresh ghost-triangle ids). The
precondition is therefore *no entity changed owner* **and** *not
`partition_dirty`* — the stamp set by `MFTLObjectSetCounts` (every creation and
destruction), by the renumbering sites and by `MFTLRemeshMarkFrozen`.

**Agreement.** A rank may receive a change it cannot see, so the verdict is
agreed per body — with **one** per-body MAX reduction over each body's holders
rather than one collective per body. The object loop is split: pass 1a
classifies every body and packs those this rank saw change; the vote; pass 1c
packs bodies dirty elsewhere by re-running the idempotent classification.

**A body must be clean twice.** A rebuild is not idempotent (an owned vertex whose
incident triangles are all remote is dropped and re-received at the end of the
owned run), so a body is rebuilt once more after it stops changing, and only then
skipped:

| changed now | changed before | action | layout |
|---|---|---|---|
| yes | — | rebuild | not necessarily a fixed point |
| no | yes | rebuild ("settle") | fixed point |
| no | no | **skip** | unchanged fixed point |

The skip is then exact slot for slot (`test_skipping_reproduces_the_full_rebuild`
compares set and slot-order digests), at the cost of one rebuild per
dirty-to-clean transition (`n_objects_settled`).

### Memory policy

Pools only grow; `MFTLRedistributeSetShrinkSlack` (default off) offers every
rebuilt body to `MFTLObjectShrinkToFit`. Bodies a rank no longer holds at all
are always released. After generation (every body built whole on rank 0) this
reduced rank 0's pool capacity from 80 MB to 33.5 MB at 256 bodies, refine 3,
$P = 4$; the optional shrink saves a further ~8 % for ~2 % of the phase time.

---

## Ghost synchronisation {#comm_ghost}

`MFTLUpdateGhostPositions` / `MFTLUpdateGhostVertexNormals` run one request/reply
round over `comm_graph`: each rank lists its ghost vertices (the pool suffix) by
owner and sends `(body id, vertex id)` requests (8 B); owners answer through the
id table with an #MFTL_GhostUpdatePayload (32 B); replies return in request order
and are applied by direct index. Three neighbourhood collectives. Only the
holder of a ghost knows it holds it, so a request phase is unavoidable; no
owner-to-holder state is cached between calls. One field per exchange: bundling
recomputable quantities with the position would ship bytes the receiver computes
more cheaply.

@warning A ghost whose owner is not a graph neighbour cannot be served: it keeps a
         stale value, the call returns `MFTL_ERROR` on that rank, and Debug
         builds assert. There is no global fallback, so a partition whose halo
         leaves the 26-neighbourhood (a coarsened remeshing partition) must not
         outlive the remeshing call; both remeshing drivers end with a plain
         `MFTLRedistribute`.

---

## The spatial halo {#comm_spatial}

| | Redistribution halo | Spatial halo |
|---|---|---|
| kind | topological | geometric |
| contents | neighbouring triangles of the **same** body | triangles of body **B** near this rank's share of body **A** |
| lives in | the body's own pools, past `n_local` | a standalone shadow #MFTL_Object per body |
| built by | `MFTLRedistribute` | `MFTLExtendSpatialHaloNearBodies` / `MFTLExtendSpatialHalo` |

Two bodies in contact are cut by the same rank boundaries, and the rank holding
one side of a contact may hold nothing of the other body; they share no edges, so
no topological halo growth can help. The halo is **transient**: redistribution
keeps only owned triangles, so it is rebuilt after every redistribution, and
each extension replaces earlier patches.

**Why a shadow object.** A patch is an ordinary object with `n_local == 0`, twins
linked within the patch and derived arrays built; it is not in
`ens->local_objects`, so redistribution and ghost syncs never see it. Appending
foreign triangles to the body's own ghost suffix would silently change every pass
bounded by `tris.n` (the Morton box, `MFTLMeanEdgeLength`, the vertex-normal
scatter, `MFTLRebuildTwins` — which would weld bodies sharing vertex slots), and
the arriving vertex ids would violate either vertex completeness or local id
uniqueness. Patch boundary flags are the transmitted ones: a patch's rim is a cut,
not a mesh boundary.

### Records, extents and the pair broad phase {#comm_halo_pairs}

1. Each rank publishes one record per body it **holds** — `(id, centre,
   half-width, global radius, envelope)`, nine doubles — to its graph neighbours
   (payload $O(\text{local bodies}\times\text{degree})$, independent of the
   ensemble size).
2. **Extents are a centre plus a half-width**, and "near" is
   $$
     \forall k:\ \bigl|\Delta^{\min}_k(c^A_k, c^B_k)\bigr| \le h^A_k + h^B_k + m,
   $$
   with $\Delta^{\min}$ the minimum-image difference. A `(lo, hi)` box fails
   both ways on a periodic domain: shares touching across the seam are a domain
   length apart (a **missed** contact, decided before the BVH ever sees the
   triangles), and a share containing one seam-straddling triangle spans the
   whole domain (every neighbour ships everything — measured at $P = 27$: 314
   ghosted triangles against 56). A share's extent is measured in the raw frame
   and in one shifted by half a domain, keeping the tighter per axis, which is
   exact while the share is narrower than $L/2$ and conservative otherwise.
3. The arriving records are binned into a **uniform grid** with cell
   $\ge 2\,h_{\max} + m_{\max}$ per axis, so overlapping extents are in adjacent
   cells and each query visits 27 cells: $O(\text{bodies}+\text{records})$ instead
   of the $O(B^2)$ all-pairs list (134 million records at 16 384 bodies). One
   domain-spanning body collapses the grid to a single cell — correct, and the
   right cost for a few large bodies.
4. Body B travels towards body A only when $A < B$ (and $A = B$ with
   `self_pairs`): the rank owning the A-side triangle reports the contact, so only
   it needs the copy. Per destination each triangle is packed once however many
   partners want it, and the margin used is the widest any matched partner needs —
   the pre-filter is always at least as permissive as the per-triangle test.
5. `halo->pair` returns every `(A, B)` with this rank holding A and seeing B,
   ascending and unique. Measured on six touching couples (6 of 66 pairs real):
   6/6/9/18/6 candidates at $P = 1, 2, 4, 8, 27$ against 66/132/264/528/1782 for
   all pairs (`test_the_broad_phase_finds_the_pairs_without_enumerating_them`).

<!-- figure: halo-extents
Why an extent is a centre and a half-width. One periodic axis, unrolled.
(a) Two shares touching across the seam have disjoint $(\mathrm{lo},\mathrm{hi})$
boxes, so the contact is missed before any triangle is compared. (b) A share
straddling the seam has a box spanning the whole domain, so every neighbour
ships it everything. Measured in the raw frame and in one shifted by half a
period, keeping the tighter, the centre and half-width are right in both cases
under the minimum image.
-->

`MFTLSpatialHaloResolveRadii` computes each body's radius — half the longest side
of its **global** box (so envelopes do not depend on the process count; $R$ for a
sphere) — over the holders in two frames, and publishes it one hop further,
because a rank deciding what to send needs the size of the body it sends
*towards*. A rank two hops from a body neither receives it nor can report a
contact with it — the same reach as the exchange.

The directory (#MFTL_SpatialHalo) is a sorted body-id array holding the bodies
this rank holds plus those it received a patch for, pruned after every exchange
(12 slots over all ranks for twelve bodies at $P = 27$, against 324 for a dense
table).

---

## Known limitations {#comm_deferred}

1. `MFTLUpdateGhostPositions` builds an `[nproc]` rank-to-neighbour table per
   call, and the halo exchange allocates two `[nproc]` send-buffer descriptor
   arrays per call — $O(P)$ per rank and timestep, harmless at the tested scales
   but a candidate for the persistent workspace at $P$ in the thousands.
2. The ghost exchange rebuilds its schedule every call; it could be cached for a
   solver that redistributes only every $N$ steps.
3. The geometry pass after a redistribution runs on freshly rewritten pools and is
   bound by first touch (6.4 ns/triangle against 2.9 on a resident mesh);
   fusing it into the rebuild was not attempted.
4. Nothing has been measured above 4 ranks on real hardware; the scaling of the
   graph against the global route at $P$ in the hundreds is unmeasured.

`src/communication/` — Distributed memory {#mftl_mod_communication}
=========================================

**Public header:** `include/mftl_communication.h` · **Layer:** 3 ·
**Module group:** @ref mftl_communication

Four translation units, four distinct jobs:

| File | Job |
|---|---|
| `exchange.c` | the MPI datatypes, and a non-blocking all-to-all-v of up to four independent payloads |
| `redistribute.c` | *"the mesh has moved — who owns what now?"* Re-partition, and rebuild the halo. |
| `ghost_sync.c` | *"the mesh has deformed — refresh my ghosts."* Topology untouched; one per-vertex 3-vector travels. |
| `spatial_halo.c` | *"fetch me the parts of body B near my share of body A."* A geometric halo, delivered as a shadow object. |

Plus `comm_internal.h`, holding the two data structures all four share: a lazy
`int` buffer, and the epoch-stamped global-id table that replaced the hash maps.

---

## What the wire carries, and what it deliberately does not {#comm_wire}

**Only topology and identity travel.** Everything geometric except the vertex
position is left behind and recomputed at the receiver, because recomputing it is
a streaming pass over contiguous arrays while shipping it is bytes on a wire, per
entity, per timestep:

| Payload | everything on the wire | here |
|---|---|---|
| #MFTL_PackedVertex | 96 B | **40 B** (position + 4 ints) |
| #MFTL_PackedTriangleComplex | 120 B | **40 B** (10 ints) |

A vertex **position** is the one geometric quantity that is authoritative
(@ref arch_inv_positions) — nothing can recompute it — so it travels. The vertex
normal, area and curvatures, and the triangle area/normal/centre, are all
functions of the positions and the topology, so they do not.

Three further fields that a general half-edge wire format would have to carry are
not sent, because implicit indexing already determines them:

- **A half-edge's global ID.** Half-edge $3t+k$ carries $3\,t_{\mathrm{gid}} + k$
  mesh-wide. The receiver reconstructs it.
- **A vertex's outgoing half-edge.** Re-derived at the receiver in the same pass
  that writes the half-edges.
- **`next`, `triangle` and `twin`.** The first two are arithmetic; the third is
  re-derived from the vertex ids that arrive anyway.

@warning `vertex_gid[k]` on the wire is the global id of the **target** of
         half-edge $3t+k$, i.e. `verts.rec[he[3t+k].vertex_idx].global_id`. It is
         **not** `tris.verts[3t+k]`, which is the *origin* of that half-edge. The
         two differ by a rotation, and mixing them silently rotates every triangle.

### Layout is pinned by static assertion

Both packed types are laid out as one run of doubles followed by one run of ints,
so the MPI datatype is two contiguous blocks rather than one block per field —
and both come to exactly 40 bytes with no padding. `_Static_assert` pins that,
because the datatype descriptions are only valid while it holds.

`commit_struct` additionally calls `MPI_Type_create_resized` to pin the extent to
the C `sizeof`. `MPI_Type_create_struct` derives the extent from the members it
was told about; that happens to match here, but only because of how the trailing
padding falls — and an array of a type whose extent disagrees with the C stride
reads **shifted garbage, silently**.

---

## Redistribution {#comm_redistribute}

### The ownership rule, and the halo rule

A vertex belongs to the rank whose Cartesian cell contains it. A triangle belongs
to the rank whose cell contains its **cached centre**.

A triangle is then *additionally* sent to the owner of each of its three
**corners** and each of its three **edge neighbours**. That surplus **is** the
ghost layer.

The corner clause is what makes the halo **vertex-complete**: for every owned
vertex, every triangle incident to it is present locally. That is a strictly
stronger statement than "one layer of edge-adjacent ghost triangles", and it is
exactly the precondition `MFTLComputeVertexNormals` needs to be
communication-free.

Read backwards, the same rule defines replication:

$$
  \text{replicated} \iff
  \text{corner} \ge \texttt{verts.n\_local} \ \lor\ \text{neighbour} \ge \texttt{tris.n\_local}
$$

Two streaming reads, six integer compares, no distance test and **no message**.
That is what `MFTLRemeshMarkFrozen` computes — see @ref mftl_mod_remeshing.

### Entities that stay never leave the pool

Treating "stays here" as just another destination routes **every** entity through
the wire format, including the ~90 % of a partition that does not move: pack into
an `MFTL_Packed*`, radix-sort the merged (kept + received) stream by global id,
de-duplicate, unpack, insert four hash-map entries per triangle. That round trip
is the single most expensive part of a redistribution and **none of it moves any
data anywhere**.

Here the kept entities are identified during the same destination analysis that
decides what to send — this rank is simply one more destination — and are then
**compacted in the pool** by a permutation. Concretely, per redistribution:

- **no pack/unpack of kept entities, and no global sort.** The merged sort is
  replaced by an $O(\text{kept})$ gather plus an $O(\text{received})$
  classification.
- **no hash maps.** Global ids are dense per entity class, so *"have I already got
  global id $g$"* is an epoch-stamped array probe.
- **no full twin rebuild.** A twin between two kept triangles survives as an index
  remap; only half-edges that actually lost their partner are re-paired, and they
  are collected for free while the remap runs.

### The epoch-stamped id table {#comm_gid_table}

`mftl_gid_table` in `comm_internal.h` is what replaced `khash`. The question every
rebuild answers is *"do I already hold global id $g$, and where?"* — and since ids
are dense, the answer is an **array index**.

Two things make a plain array practical:

- the id range is the **object's**, not the ensemble's, so a thousand droplets get
  a thousand small ranges rather than one huge one;
- the table is **never cleared**. `stamp[g] == epoch` is what makes an entry valid,
  and bumping the epoch retires every entry at once. Clearing a table sized by the
  id range once per object per timestep would cost more than the lookups it serves.

Epochs start at 1, so a zeroed slot is always *absent*. The one case where "never
cleared" cannot hold is after $2^{32}$ epochs, when the counter wraps to a value
some ancient stamp still carries — unsigned wraparound is defined, so it is
detectable, and the table is wiped exactly once every four billion objects.

### The vertex de-duplication bitmask

A vertex is a candidate for a destination once per incident triangle bound there,
so its list would arrive `valence` times over and have to be fixed afterwards with
a sort plus a unique pass — work to remove duplicates that never had to be created.

One bit per (vertex, destination) replaces it: `mask[v] & (1 << b)` is *"v is
already queued for the destination holding bit slot b"*, so the test-and-set is
$O(1)$ and nothing is sorted.

Bit slots are handed out on first contact. A remote destination set is bounded by
the 26-neighbourhood on the local path, so 32 bits suffice there by construction;
the global path (the initial scatter, or an object spanning more than 32 cells)
can exceed it, and those ranks get `NO_BIT` and fall back to the sort.
**Correctness never depends on the bound.**

### Two paths, one exchange engine

`MFTLExchangeStart` posts **two collectives regardless of the payload count**: one
`MPI_Alltoall` carrying `n_payloads` counts per target, then one non-blocking
`*alltoallv` per payload, all posted back to back.

Merging the count exchange matters at scale — a 1-int `Alltoall` is pure latency
and does not get cheaper with fewer bytes, so paying it once instead of once per
payload is what keeps the small-message cost of a redistribution flat.

| Path | Primitive | When |
|---|---|---|
| graph | `MPI_Neighbor_alltoallv` over `comm_graph` | every destination is inside the 26-neighbourhood — the production case |
| global | `MPI_Alltoallv` over `comm_cart` | some destination is not, e.g. the initial scatter from rank 0 |

The choice is reduced across ranks, never taken locally.

### A body that did not move is not touched {#comm_skip}

A rank's local state after a redistribution is a pure function of two inputs, the
global topology and the owner map. Read the halo rule backwards and it is a
definition of the OUTPUT rather than a procedure — *rank r holds triangle t iff r
owns t, or owns one of its corners, or owns one of its edge neighbours*, plus the
corners of everything it holds. **Positions are not an input.** They decide the
owner map, but once the map is fixed, moving a vertex inside its own cell changes
nothing about who holds what.

So a body for which neither input changed already holds the answer, and
everything in "Entities that stay never leave the pool" above is a very expensive
way of reproducing it. `MFTLRedistributeSetSkipUnchangedBodies` (default **on**)
leaves such a body alone: no destination analysis, no kept gather, no relink, no
arrival classification, no twin repair, no topology rebuild. The measured share
of a redistribution that removes is in `docs/analysis.md` §2.4.

**The precondition is NOT "did topology change".** Under the remesher's freeze
discipline a topology change is invisible to the halo rule — an unfrozen entity
has no ghost corner and no ghost edge neighbour, so every triangle an operator
builds out of it has all seven halo clauses resolving to the acting rank. What a
skip genuinely breaks is **global ids**: the spanning remesher renumbers owned
entities and deliberately leaves ghost *triangle* ids alone, because
redistribution drops and re-receives every ghost triangle and that is what
repairs them. The minimal correct precondition is therefore

> no entity of the body changed owner, **and** nothing rewrote the global ids or
> the flags of entities that other ranks hold copies of

and the second clause is `MFTLObjectMarkPartitionDirty` (@ref mftl_object.h),
stamped by `MFTLObjectSetCounts` — the choke point every entity creation and
destruction already passes through — plus the two renumbering sites and
`MFTLRemeshMarkFrozen`. Full derivation in
`docs/analysis.md` §2.4.

**Two passes and one collective.** The decision cannot be local: a rank with zero
local changes is still the receiver of a change made elsewhere. Agreeing per body
*inside* the object loop would be `num_objects` collectives per timestep, so
instead:

| | |
|---|---|
| pass 1a | classify ownership for **every** body; do the full analysis and packing for every body **this rank** saw change |
| vote | one `MFTLEnsembleReduceBodies` over each body's **holders** |
| pass 1c | the bodies that were clean here and dirty somewhere else |

**The vote used to be one `MPI_Allreduce(MPI_BOR)` over a bitset of one bit per
body**, which is as small as that shape can be made and still `O(bodies in the
ensemble)` on every rank — the last array here that grew with `P` in a weak sweep
at constant work per rank. Only the ranks HOLDING a body can have seen it change,
and only they will rebuild it, so the vote is exactly a per-holder reduction.
@ref MFTLEnsembleReduceBodies "MFTLEnsembleReduceBodies" is where that decision
lives; the workspace slices are now indexed by local slot.

@warning The pre-exchange half of the route test reads each body's ghost
         `owner_proc`, and `compute_owners` **overwrites** `owner_proc` with the
         DESTINATION rank while classifying. Asking one line later gives a
         plausible answer to the wrong question — the partition that is about to
         exist, not the one the vote has to reach — and it is wrong only when
         something moved far, i.e. exactly when the vote matters. So it is taken
         before pass 1a and handed to `MFTLEnsembleReduceBodiesPrescanned`, which
         agrees it together with the post-exchange certificate in one int.

An **arrival** has no vote slot and needs none: it is here because an entity of it
crossed into this rank's cell, so some rank saw it change, and `MFTLObjectCreate`
leaves `partition_settled` false — the skip clause is already decided.

Pass 1c does not need pass 1a's owner arrays kept alive — which is what would
otherwise cost `O(sum_verts + sum_tris)` of new storage. `compute_owners` is
idempotent, so it simply runs again on the few bodies it reaches.

**A body must be clean twice before it is skipped.** `rebuild_object` is not
idempotent: an owned vertex whose incident triangles all belong to other ranks is
not kept, so it is dropped, re-received, and lands at the *end* of the owned run
rather than where it was. Running the rebuild twice on an unchanged body
therefore moves entities the first run left in place. A skip taken before that
fixed point is reached still produces a correct partition — same entities, same
ownership, same halo, same ids, same positions — but a different **slot order**,
which is precisely what the order digest of
`parallel_advection_redistribute_list --digest` exists to catch, and it diverged
on the first step for bodies spanning three ranks. One extra bit per body closes
it:

| `D` now | `D` before | action | layout afterwards |
|---|---|---|---|
| 1 | — | rebuild | not necessarily a fixed point |
| 0 | 1 | rebuild ("settle") | a fixed point |
| 0 | 0 | **skip** | unchanged, so still a fixed point |

The cost is one rebuild per dirty-to-clean transition, reported as
`n_objects_settled`. What it buys is that the optimisation is invisible rather
than merely harmless.

### Postcondition

@warning A redistributed object comes back with valid topology, valid **positions**
         and #MFTL_DERIVED_TOPOLOGY set. Its `tris.area/normal/center` and
         `verts.normal` hold **unspecified** values.

`tris.verts` and `tris.neighbors` are not rebuilt by a closing pass; the relink
loop, the arrival loop and `repair_twins` emit them as they go, because every
input is already in a register there. Two thirds of what that saves comes back as
store traffic in a loop that is dependency-bound rather than streaming, which is
the measurement in `docs/analysis.md` §2.5 and the reason the
`derived: topology` bucket no longer exists.

That is what buys the 3× smaller wire format. The loop that satisfies it with
exactly one geometry pass is in @ref mftl_page_timestep, and the three things in
it that are silent when wrong are spelled out there.

---

## Ghost synchronisation {#comm_ghost}

`ghost_sync.c` implements one request/reply round:

1. every rank lists the global ids of its ghost vertices, grouped by owning rank;
2. one exchange sends the requests;
3. each owner scatters its **owned** vertices into the epoch-stamped table and
   answers with one #MFTL_GhostUpdatePayload per request;
4. one exchange returns the replies, which are applied by id.

### One vector per message, never a bundle

Shipping position, normal, both curvatures and the area together costs **80 B
against 32 B**, of which everything but the position is a function of the
positions and the topology — so four fifths of it is recomputable at the receiver
for less than it costs to send.

#MFTL_GhostUpdatePayload is therefore deliberately **generic** rather than named
after the position. The same 32 bytes, the same datatype and the same exchange
carry whichever per-vertex vector is being synchronised; the two callers differ
only in which array they read from and write into. Anything that genuinely needs
a second field sends a second exchange and pays only for that field.

The scatter is $O(\text{owned})$ and the requests are $O(\text{halo})$, so it is
only worth doing per object that actually received a request — which is why the
caller groups the requests by object first. In the many-body case that skips
almost everything.

Every call reconstructs the requests directly from the current ghost suffix.
No persistent owner-to-holder state or pool indices are retained between calls.

---

## The spatial halo {#comm_spatial}

A **different thing entirely** from the redistribution halo:

| | Redistribution halo | Spatial halo |
|---|---|---|
| kind | topological | geometric |
| contents | neighbouring triangles of the **same** body | triangles of body **B** near this rank's share of body **A** |
| lives in | the body's own pools, past `n_local` | a standalone shadow #MFTL_Object per body |
| rebuilt by | `MFTLRedistribute` | `MFTLExtendSpatialHalo` |

The patch is an ordinary object with `n_local == 0`, twins linked within the
patch, derived arrays built. It is deliberately **not** in `ens->local_objects`, so
redistribution and both ghost syncs never see it.

### The directory is sparse, and it is NOT the local object list {#comm_halo_dir}

#MFTL_SpatialHalo keys its patches and radii on a sorted `body_gid` array —
`MFTLSpatialHaloFindBody`, `MFTLSpatialHaloPatch` and `MFTLSpatialHaloRadius`
binary-search it — rather than on a `[n_bodies_global]` table. It holds two sets,
and the second is why it cannot simply *be* the ensemble's local list:

- the bodies this rank **holds**, so a resolved radius has somewhere to live;
- the bodies this rank **received a patch for**, which is precisely the set it
  holds nothing of. A rank with droplet A and none of droplet B is exactly the
  rank that must be told about B.

Entries that are neither held nor patched are pruned at the end of every
exchange, so the directory stays proportional to what is nearby rather than
growing with every droplet that ever drifted past. **Measured** on twelve bodies:
12 slots over all ranks at $P = 27$, against the 324 a dense table would have
cost.

### The candidate pairs are DERIVED, not enumerated {#comm_halo_pairs}

`MFTLExtendSpatialHaloNearBodies` is the form contact detection uses, and it
exists because the all-pairs request list its caller used to build was the last
$O(B^2)$ term in the library: $B(B-1)/2$ ordered records on every rank every call,
134 million of them at 16 384 bodies.

The list was justified by *"the halo derives its sequence of collectives from the
requests, so they must be identical on every rank"*. **They do not.** The halo's
collectives are one neighbourhood allgatherv of body extents and one of payload,
and neither's *count* depends on the requests — a request only says what THIS rank
is willing to send. So:

1. each rank publishes one record per body **it holds**: `(body id, centre,
   half-width, global radius, contact envelope)`, nine doubles. The dense form
   published `6 × n_bodies_global` doubles **to every neighbour**, i.e. 1.2 kB per
   body per rank on a full 26-neighbourhood;
2. the arrivals are binned into a **uniform grid** whose cell is at least
   `2 × widest half-width + largest envelope` on every axis. That is what makes
   "the extents overlap" imply "within one cell", so a query is the 27 cells
   around a centre and the whole phase is $O(\text{bodies} + \text{records})$;
3. body B travels towards body A only when `A < B` by body id — the existing
   "reported once" rule, since the holder of the smaller id is the rank that
   reports the pair and therefore the one that needs the copy;
4. `halo->pair` comes back holding every `(A, B)` with `A < B` such that this rank
   holds A and can see B, deduplicated.

The envelope going **on the wire** is what makes step 3 decidable locally: a rank
cannot judge whether its droplet is worth sending towards a neighbour's without
knowing that body's envelope, and the envelope scales with the body.

**Measured** on six well-separated touching couples, so 6 of the 66 unordered
pairs are contacts: 6 / 6 / 9 / 18 / 6 candidate pairs over all ranks at
$P = 1, 2, 4, 8, 27$, against 66 / 132 / 264 / 528 / 1782.

@note `MFTLSpatialHaloResolveRadii` reduces over each body's **holders** and then
      publishes `(gid, radius)` **one hop further**, to the 26 neighbours. That is
      not redundancy: the reduction tells a body's holders how big it is, and the
      rank that has to decide what to *send* needs the size of the body it is
      sending *towards* and holds none of it. One hop is exactly right, because
      the halo only ever ships along `comm_graph`. The corollary is that a rank
      two hops from a body has no envelope for it **and can never report a contact
      with it**, which is the same reach the exchange itself has.

### Why not append to the body's ghost suffix

That is what MFTL 2 did, and it looks free because the ownership prefix is
untouched. It fails for two reasons that are invisible until they bite:

1. **Four passes bounded by `tris.n` rather than `n_local` change silently:** the
   Morton bounding box in `reorder_vertices` (a partner body across the domain
   coarsens the cells for the *owned* mesh), `MFTLMeanEdgeLength` (each appended
   triangle contributes three twin-less half-edges, so the remeshing target length
   is set partly by another body — and it carries no ghost assertion), the
   vertex-normal scatter, and `MFTLRebuildTwins`, which re-pairs by vertex **slot**
   and can **weld two bodies together**.
2. **There is no good answer for the arriving vertex GIDs.** De-duplicate them onto
   existing slots and `inv_halo_is_vertex_complete` breaks; give them fresh slots
   and `inv_local_global_ids_are_unique` breaks. Both are in
   `inv_partition_is_sound`.

Doing it properly needs the two-way ownership split to become three-way, i.e. a
change to @ref arch_inv_prefix.

### An extent is a centre plus a half-width {#comm_extent}

Everything geometric in `spatial_halo.c` is a question of the form *"is this near
that"*, asked across a domain that **wraps**. A `(lo, hi)` box cannot express the
answer, and it gets it wrong in **both directions at once**:

- **It misses contacts.** Two shares meeting across the seam are a domain length
  apart in stored coordinates. Rank $p$ reports its part of A at $x \approx 9.9$
  and rank $q$ reports its part of B at $x \approx 0.1$, so a min/max comparison
  reports no overlap for two surfaces that are touching. The traversal tries every
  periodic image of B, but only for triangles it **has**, and this test decides
  whether they are sent at all. The miss lands one stage before anything the BVH
  can see.
- **It over-sends enormously.** A share containing even one seam-straddling
  triangle has a min/max box spanning the **whole domain**, because
  `MFTLGetTriangleBounds` is correctly seam-consistent and puts that triangle's
  corners just outside the box on one side. Such a share overlaps everything, and
  every neighbour ships its entire share of every partner body — a droplet on the
  seam turns the halo into an all-to-all broadcast of the mesh, every timestep,
  silently.

So an extent is a **centre on the circle plus a half-width**, and "near" is the
minimum-image distance between centres against the sum of half-widths:

$$
  \text{near} \iff \forall k:\;
  \big| \Delta_k(c^A_k, c^B_k) \big| \le h^A_k + h^B_k + \text{margin}
$$

A rank's own extent is measured **in two frames** — raw, and shifted by half a
domain — keeping the tighter per axis. That is what makes a straddling share
report its true size, and it is the same trick `MFTLSpatialHaloResolveRadii`
needs so a droplet on the seam does not report a bounding radius of half the box.

**Measured** at $P = 27$ on a seam contact: **314 triangles ghosted with boxes
against 56 with intervals**, and **320 against 0** for a placement with no
cross-rank contact at all. Payload is nine doubles per body a rank **holds**
either way — not per body in the ensemble.

### A pre-filter must be strictly MORE permissive than what it short-cuts

The per-triangle intervals are widened by the contact margin. An **un-widened**
macro extent is therefore *stricter* than what it stands in for, and rejects the
whole neighbour in exactly the regime the margin exists for: two shares that do
not overlap but are within contact range of each other.

Overlapping bodies hide it completely — their raw extents intersect anyway — so
only a **proximity** contact at a rank boundary reveals it, which is what
`test_the_tolerance_threshold_is_exact_from_both_sides` is.

### The halo is transient and cannot be otherwise

`collect_destinations` keeps only owned triangles, so a halo cannot survive a
repartition **even in principle**, and the documented loop redistributes every
step. Each `MFTLExtendSpatialHalo` **replaces** the patches of the bodies it was
asked about rather than accumulating; `MFTLExtendSpatialHaloNearBodies`, which
speaks for the whole directory, replaces all of them.

@note The overlap list is rebuilt from scratch every call and could be cached when
      the decomposition is static — but every rank's extent for every body changes
      every step, so the cache would be invalidated exactly as often as it was
      built. If a solver appears that redistributes every $N$ steps, it belongs on
      #MFTL_SpatialHalo, which the caller already owns — **not** on #MFTL_Domain,
      which is read-only during a redistribution.

---

## Deferred {#comm_deferred}

1. **Per-destination send buffers are allocated and freed on every call.** At the
   density-derived hint they are ~128 KB each, so a redistribution `malloc`s and
   frees a few megabytes per timestep and pays the first-touch faults again each
   time. Keeping them as scratch on the domain would remove it; not done because
   it puts mutable per-call state on a struct that is otherwise read-only during a
   redistribution.
2. **Step 5 of the per-timestep loop is bound by first touch, not by its kernel.**
   The geometry pass that restores what redistribution drops runs on pools the
   redistribution has just rewritten, and costs **6.4 ns/triangle against 2.9 ns**
   for the same function on a resident mesh. The remaining lever is fusing the
   recompute *into* the rebuild, where the arrays are already in registers — not
   attempted, because it would put geometry into a module that deliberately
   transmits none.
3. **Nothing has been measured above 4 ranks on real hardware.** The scaling of
   the exchange — graph path versus global path, and the behaviour at $P$ in the
   hundreds — is entirely unmeasured.

# Updated redistribution architecture recommendation

## Status and relationship to the first recommendation

This document rethinks [the original recommendation](redistribution-architecture-recommendation.md) under the following fixed requirements:

- The monolithic `MFTL_VertexPool` and `MFTL_TrianglePool` arrays remain.
- Their structures and the owned-prefix convention remain: owned entities occupy `[0, n_local)` and ghosts occupy `[n_local, n)`.
- New persistent auxiliary arrays and maps may be added to an object or ensemble.
- A rank must not retain an `MFTL_Object` that has no local owned or ghost vertices or triangles.
- The library will ship one redistribution implementation, not a full-rebuild engine beside an incremental engine.
- Both the neighbor path and the global path remain.
- Direct point-to-point communication and MPI neighborhood collectives must be measured rather than selected by assumption.
- Redistribution remains a mandatory collective timestep operation because CFD velocity interpolation acts only on locally owned vertices.

No implementation is made by this document.

## Decision

Rewrite the current `redistribute_core()` completely, while retaining the public role of `MFTLRedistribute()` and the existing ownership and halo semantics.

The replacement should be one unified, transactional delta redistribution:

```text
classify ownership
        ↓
derive topology and copy-set deltas
        ↓
select global or neighbor transport
        ↓
pack one record stream
        ↓
exchange
        ↓
apply records incrementally to the monolithic pools
        ↓
commit owners, topology, and cached ghost schedule
        ↓
remove locally empty objects
```

There must not be a separately maintained steady-state engine and full-rebuild engine. The same record types, packing code, transport interface, receive parser, and pool mutation functions must handle all cases:

- A normal timestep is a small delta from the committed state.
- Initial distribution is a large set of additions from an uncommitted state.
- Remeshing is a set of journaled additions, removals, and connectivity changes.
- A long-distance movement is the same delta stream sent through the global transport.
- A rank receiving a previously absent body creates it from the body header.

During development, the old implementation may exist behind a test-only build option as a differential correctness oracle. It should be removed from the production library after the replacement passes the validation and performance gates.

## Why a rewrite of the current core is preferable to incremental tuning

The current function already has one shared `redistribute_core()`, but its transaction is based on complete local reconstruction:

- it computes destinations from every owned triangle;
- it regenerates the complete required halo;
- it allocates rank-indexed ID lists and send buffers;
- it sends vertices, triangles, and body counts as separate payloads;
- it groups arrivals by globally replicated object-list index;
- it calls `rebuild_object()` for every object;
- it rewrites kept entities and repairs derived structures after the rewrite.

Retaining that control flow would preserve the dominant costs even after allocator and packing improvements. The replacement should keep useful low-level ideas—authoritative owners send, ghosts never forward, Cartesian owner lookup, one collective path decision, packed topology, and vertex-complete halo semantics—but replace the full-state plan and full-object rebuild.

The rewrite is larger than a local optimization, but smaller and safer than maintaining two redistribution architectures indefinitely.

## Required postconditions

The unified transaction must preserve these postconditions:

1. Every vertex and triangle is owned by the rank selected by the active owner rule.
2. Every authoritative triangle exists on:
   - its triangle owner;
   - the owners of its three vertices;
   - the owners of its three adjacent triangles.
3. Every local triangle has its three required vertices.
4. Owned vertices and triangles occupy their pool prefixes.
5. Ghost vertices and triangles occupy their pool suffixes.
6. Global IDs remain the cross-rank identities; local slots remain invalidatable implementation details.
7. Half-edge twins, `out_he_idx`, `tris.verts`, and `tris.neighbors` are valid after commit.
8. Frames are valid for retained and newly installed triangles after commit.
9. The owner-to-copy ghost synchronization schedule describes the committed state.
10. An object with no local entities is absent from the local ensemble.

The transaction should stage and validate all incoming operations before mutating pools. “Transactional” means a single consistency commit with validated inputs and pre-reserved memory; it does not imply recovery from an MPI process failure.

## Target state model

### 1. Sparse local object directory

`MFTL_Ensemble.objects` should contain only locally active objects. An active object has at least one owned or ghost vertex or triangle on that rank.

Add a persistent local mapping:

```text
body global ID → local object pointer or local object-list slot
```

The wire format must identify bodies by `MFTL_Object.global_id`, never by the position of an object in `ensemble->objects`. A body-run header should carry enough mesh metadata to construct an object on a rank that does not yet hold it:

- body global ID;
- global-ID counter high-water marks;
- global vertex and triangle counts if those fields remain eagerly maintained;
- a protocol/version field;
- any body-level flags that become necessary later.

Quality values and locally derived data should not be transferred.

At receive time:

- look up the body global ID;
- create and attach an empty local object if the stream contains an addition or handoff for an absent body;
- apply all records;
- remove and destroy the local object after commit if both pools are empty.

This changes `ensemble->num_objects` from a global replicated count to the local active-object count. It removes the present `O(Bglobal)` object scan, body-count payload, grouping arrays, and same-list/same-order checksum from the hot path.

If globally queryable body metadata is required, use a small distributed metadata directory with a deterministic home rank such as `hash(body_gid) % nproc`. That costs `O(Bglobal / P)` metadata per rank instead of retaining an empty mesh object for every global body. It is separate metadata, not an empty `MFTL_Object`.

### 2. Persistent per-object communication state

Add a private communication/topology-maintenance state to each active object. It should own auxiliary arrays and maps, but must not replace or split the existing pools.

A suitable state contains:

- a sparse vertex-GID-to-slot table sized by locally held vertices;
- a sparse triangle-GID-to-slot table sized by locally held triangles;
- an edge-key-to-half-edge table keyed by endpoint global IDs;
- a local triangle-reference count per vertex;
- a vertex-to-half-edge incidence structure;
- epoch-stamped dirty marks for vertices and triangles;
- optional `next_owner` arrays for the current transaction;
- a topology mutation journal;
- a distribution/topology generation number;
- owned-vertex copy-rank reference counts used by the ghost push schedule.

Do not size GID lookup tables unconditionally from `next_vertex_gid` or `next_triangle_gid`. That makes a rank holding a small patch of one large global surface allocate storage proportional to the global body. Use an open-addressed sparse table sized to the local owned-plus-ghost count. A dense table may be an adaptive optimization only when the global-ID range is demonstrably close to the local count.

The incidence structure is necessary because moving a vertex between pool slots requires updating every half-edge whose `vertex_idx` names that slot. A practical monolithic-compatible representation is:

```text
vertex_he_head[verts.capacity]
he_vertex_next[3 * tris.capacity]
he_vertex_prev[3 * tris.capacity]
```

These arrays make vertex-slot replacement proportional to local vertex valence rather than to the complete triangle pool.

### 3. Persistent per-ensemble communication workspace

Add one reusable workspace to the ensemble or its private communication state:

- local body-GID directory;
- neighbor rank/index mapping;
- outgoing action arena;
- incoming operation arena;
- byte counts and displacements;
- persistent send and receive byte buffers with controlled growth;
- destination-local vertex de-duplication table;
- current ghost owner-to-copy schedule;
- timing, byte, record, and pool-movement counters;
- selected transport backend and configuration.

The neighbor path should allocate and clear arrays proportional to the graph degree `d`, not `nproc`. Rank-sized arrays are allowed only when the global path is actually selected. Buffers should grow geometrically and be reused; a steady timestep with an unchanged maximum delta should perform no allocation.

## Incremental ownership and halo planning

### 1. Committed and next ownership

Do not overwrite `owner_proc` while planning. Compute a `next_owner` for every locally required entity, or record only changed entities plus a way to query unchanged owners.

The mandatory correctness path may still scan all locally held vertices and triangles:

- vertex ownership is computed from the current position;
- triangle ownership is computed from the current cached center;
- a cheap local-box test should precede the full Cartesian owner lookup.

This adds `O(Nlocal + Hlocal)` inexpensive classification work, but it is not the current dominant cost and does not communicate or rebuild the complete mesh. If the caller or an MFTL advection/geometry kernel can supply next-owner information while already streaming the arrays, the same planner may consume it; that is an optimization of the same algorithm, not a second redistribution implementation.

For explicit owner maps, maps remain associated with locally active objects and local pool slots. They no longer depend on identical object-list indices across ranks. The authoritative old owner is responsible for emitting the entity transaction.

### 2. Dirty migration frontier

Mark the minimum topology region whose replica set may have changed:

- a vertex whose owner changes marks every incident triangle;
- a triangle whose owner changes marks itself;
- a changed triangle marks its three adjacent triangles;
- a journaled topology edit marks the edited triangle, its vertices, and its edge neighbors.

Only an authoritative, currently owned triangle emits changes for that triangle.

For triangle `T`, define:

```text
Rold(T) =
    old_owner(T)
  ∪ old_owner(each of its three vertices)
  ∪ old_owner(each of its three adjacent triangles)

Rnew(T) =
    next_owner(T)
  ∪ next_owner(each of its three vertices)
  ∪ next_owner(each of its three adjacent triangles)
```

Both sets contain at most seven unique ranks. Compute them with a seven-element stack array and linear duplicate removal; a general set or allocation would cost more than the work saved.

The set difference determines the complete triangle transaction:

- `Rnew - Rold` receives `TRIANGLE_ADD`;
- `Rold - Rnew` receives `TRIANGLE_DROP`;
- the new owner receives or locally applies `TRIANGLE_HANDOFF` when ownership changes;
- ranks in the intersection retain the existing triangle and only commit the new owner value.

If an old owner remains in `Rnew`, it demotes its triangle from owned to ghost rather than deleting and recreating it. If the new owner already holds the triangle as a ghost, it promotes that slot rather than installing a duplicate.

### 3. Vertex presence and copy accounting

A `TRIANGLE_ADD` must guarantee that its three vertices exist at the destination. The packing pass should emit `VERTEX_UPSERT` records for required vertices, de-duplicated by `(destination, body_gid, vertex_gid)`. The receiver inserts a missing vertex or updates authoritative identity fields without replacing a retained vertex.

Maintain a local triangle-reference count for every vertex. A non-owned vertex is removed only when this count reaches zero. An owned vertex remains even if its current local triangle-reference count temporarily reaches zero during staged application.

The owner of each vertex also needs a sparse per-copy-rank reference count for the cached push schedule. Triangle replica changes generate aggregated reference deltas:

```text
(body_gid, vertex_gid, holder_rank, reference_delta)
```

Send those deltas to the vertex's next owner in the same record exchange. For a vertex ownership handoff:

- the old owner transfers the existing nonzero copy-rank count list to the new owner;
- triangle replica deltas for this transaction are addressed to the new owner;
- the new owner combines the transferred base with all deltas;
- retained holders commit the deterministic next owner;
- the resulting nonzero ranks become the next owner-to-copy schedule.

This avoids a second subscription exchange after every redistribution. It does add small, aggregatable records proportional to actual triangle replica changes.

### 4. Topology changes use a mutation journal

Delta redistribution cannot infer a deleted triangle after the triangle has already disappeared. Every topology-changing library operation must therefore append to an object mutation journal.

The journal needs enough pre-change information to remove old replicas:

- deleted triangle global ID;
- its old owner;
- its old replica set, or the old owners of its vertices and neighbors from which that set can be reconstructed;
- vertex global IDs and edge-neighbor global IDs needed for reference-count changes.

It also records:

- new triangles;
- connectivity changes;
- newly created vertices;
- deleted vertices;
- ownership-map changes introduced by remeshing.

Initial distribution is represented as additions from an uncommitted object state. It is not a separate full-distribution algorithm.

Add debug generation checks around all topology mutators. If the topology generation changes without matching journal entries, redistribution should report an invariant violation instead of silently leaving stale remote copies. All remeshing entry points must use the journal-aware mutators before the old production rebuild code is removed.

## One wire protocol

Use one variable-length byte stream per target. A stream is grouped by body and contains versioned, tagged records.

A minimal record family is:

- `BODY_BEGIN`
- `VERTEX_UPSERT`
- `VERTEX_HANDOFF`
- `VERTEX_REFERENCE_DELTA`
- `TRIANGLE_ADD`
- `TRIANGLE_DROP`
- `TRIANGLE_HANDOFF`
- `BODY_END`

`BODY_BEGIN` carries the body global ID and creation metadata. Record headers carry a tag and byte length so unknown or version-mismatched data can be rejected safely.

Triangle additions carry the authoritative fields presently sent by `MFTL_PackedTriangleComplex`:

- triangle global ID;
- next owner;
- flags;
- three vertex global IDs;
- three half-edge flags.

The receiving rank can compute center, area, normal, frames, and derived connectivity only for the newly arrived or locally touched triangles. Retained geometry must not be discarded.

Packing should be:

1. Append compact logical actions to a reusable action arena while planning.
2. Aggregate vertex reference deltas and eliminate duplicate vertex upserts.
3. Histogram final byte counts by target.
4. Prefix-sum target offsets.
5. Pack directly into the persistent rank-major or neighbor-major send buffer.

This removes:

- the per-destination `tri_ids` and `vert_ids` buffers;
- the per-object kept arenas;
- separate vertex, triangle, and body-count payloads;
- the flattening copy inside the current exchange layer;
- arrival regrouping by global object-list index.

Local-destination actions should pass through the same parser/apply representation without going through MPI. That keeps local and remote state transitions identical.

## Global and neighbor communication paths

### 1. Path selection remains

The planner marks `requires_global` when any record destination is outside the configured graph neighborhood. The ranks agree on the route with the current communicator-wide logical OR:

```text
MPI_Allreduce(local_requires_global, global_requires_global, MPI_LOR)
```

- False selects the neighbor path.
- True selects the global path.

This collective is a remaining fixed global latency cost in automatic mode. It is required for safe dynamic selection because a receiver cannot know that a non-neighbor sender selected the global communicator.

A later optional caller contract may permit an asserted-neighbor mode that skips the Allreduce when the CFD solver guarantees a displacement bound. Debug builds must verify that contract. The automatic mode, neighbor path, and global path still remain; this option changes only path-selection overhead.

### 2. Neighbor transport must be benchmarked

The neighbor route must expose the same stream to two low-level experimental backends:

1. MPI neighborhood collectives:
   - count exchange with `MPI_Neighbor_alltoall`;
   - data exchange with `MPI_Ineighbor_alltoallv`;
   - persistent variants only after stable buffer signatures exist.

2. Direct nonblocking point-to-point:
   - one compact count message per active graph edge;
   - posted receives and `MPI_Isend`/`MPI_Irecv` for nonempty payloads;
   - stable tags and request arrays cached by graph edge.

These are transports beneath one redistribution algorithm, not two redistribution functions.

Do not assume that direct point-to-point wins for sparse traffic or that neighborhood collectives win for regular traffic. MPI implementation, graph degree, eager limits, message size, progress behavior, and shared-memory ranks can reverse the result. Benchmark both on the MacBook and the target cluster.

After measurement:

- choose one production default;
- retain runtime selection only if different backends win in materially different supported regimes;
- otherwise remove the losing production backend after using it for validation.

### 3. Global transport remains

The global route accepts exactly the same packed streams and operation records. Its baseline transport should remain a count exchange plus `MPI_Ialltoallv`/`MPI_Alltoallv` over the Cartesian communicator.

Allocate `O(P)` count/displacement storage lazily only when this path is selected, then retain it for future global calls. A sparse global point-to-point variant is worth measuring only if receiver discovery and progress do not reintroduce a comparable collective cost.

Long jumps, initial rank-zero distribution, and remeshing owner maps require no special rebuild path: they merely select this transport when a destination is outside the neighborhood.

## Incremental mutation of the monolithic pools

### 1. Centralized slot primitives

All changes to pool order must go through a small, exhaustively tested set of internal primitives:

- `swap_vertex_slots(obj, a, b)`
- `swap_triangle_slots(obj, a, b)`
- `insert_owned_vertex` / `insert_ghost_vertex`
- `insert_owned_triangle` / `insert_ghost_triangle`
- `promote_vertex` / `demote_vertex`
- `promote_triangle` / `demote_triangle`
- `erase_vertex` / `erase_triangle`

`swap_vertex_slots` must update:

- every array in `MFTL_VertexPool`;
- the vertex GID-to-slot map;
- all incident half-edge `vertex_idx` values using the incidence lists;
- `out_he_idx` values;
- any dirty, next-owner, reference-count, and copy metadata indexed by vertex slot;
- affected derived connectivity/frame state.

`swap_triangle_slots` must update:

- every array in `MFTL_TrianglePool`;
- the triangle GID-to-slot map;
- all six groups of half-edge records involved in the swap;
- twins in adjacent triangles that point to moved half-edges;
- vertex `out_he_idx` values that point into moved half-edge ranges;
- incidence-list half-edge indices;
- edge-map values;
- affected `tris.neighbors` entries and auxiliary per-triangle arrays.

These functions are the critical correctness foundation. No other redistribution code may manually move individual pool arrays.

### 2. Prefix-preserving operations

The owned-prefix invariant can be maintained with constant-count slot operations:

```text
insert ghost:
    append at n

insert owned:
    append at n
    swap appended slot with n_local
    increment n_local and n

promote ghost:
    swap its slot with n_local
    increment n_local

demote owned:
    decrement n_local
    swap its slot with the new n_local boundary

erase ghost:
    swap with n - 1
    decrement n

erase owned:
    demote
    erase ghost
```

Equivalent triangle operations use the triangle slot primitive. A vertex swap costs `O(valence)` because incident half-edges must be redirected; a triangle swap is bounded by its three vertices and three neighbors. Neither requires copying the complete pool.

Use geometric capacity growth or a configurable slack policy. Before applying an object transaction, determine conservative final vertex and triangle counts and call the existing pool reserve functions once per pool. Occasional monolithic reallocation remains `O(Nobject)`, but it is amortized and should disappear from steady-state timing once capacity stabilizes.

### 3. Staged application order

Received and local actions should be applied in this order:

1. Parse and validate every record.
2. Resolve body global IDs and create required empty object shells.
3. Build per-object operation journals keyed by global entity ID.
4. Reject conflicting duplicate ownership or incompatible add/drop records.
5. Compute capacity requirements and reserve each monolithic pool once.
6. Upsert vertices required by additions and handoffs.
7. Add, update, and promote triangles.
8. Apply triangle demotions and drops.
9. Apply vertex ownership transitions.
10. Remove unreferenced non-owned vertices.
11. Repair edge twins and derived topology only for touched entities.
12. Recompute geometry/frames only for new or affected triangles.
13. Commit `owner_proc`, copy-rank counts, and the next ghost schedule.
14. Destroy locally empty objects and update the local body directory.

Operation journals should retain global IDs, not local slots, across mutation stages. A swap can invalidate a remembered slot, whereas a fresh GID lookup remains correct.

## Cached push-based ghost synchronization

The request/reply protocol in `ghost_sync.c` currently rediscovers holders on every call. The redistribution transaction will maintain that relationship persistently.

At commit:

- every ghost vertex stores its authoritative owner;
- every vertex owner stores the ranks with a nonzero copy-reference count;
- the ensemble workspace groups source slots and destination slots by communication edge and body/vertex GID;
- the schedule generation equals the committed redistribution generation.

The next position synchronization is:

```text
pack owned positions into cached send order
        ↓
one count-free exchange when schedule sizes are unchanged
        ↓
write received positions into cached ghost destinations
```

The stream does not need request IDs or reply IDs. When pool swaps change local slots, rebuild only the affected schedule entries from their global IDs after all pool mutations are complete. The topology transaction updates copy memberships, so the next schedule is ready before `MFTLRedistribute()` returns.

`MFTLUpdateGhostVertexNormals()` can use the same cached graph and slot schedules with a different field pointer. It remains a separate field transfer, not a separate schedule-discovery protocol.

Because redistribution is called every timestep, position synchronization still communicates `O(Hlocal)` vector data every timestep. The optimization removes request construction, two phases, repeated GID maps, repeated grouping, and most allocations; it cannot remove the required position bytes.

## Unified behavior for special cases

### Initial distribution

Only ranks that hold seed geometry have active objects. All ranks still enter the ensemble redistribution collective.

Seed objects are marked uncommitted. Their current authoritative triangles generate additions to `Rnew` and local drops from the seed rank when appropriate. Body headers create missing objects at receivers. The global path is selected if required. After commit, the same invariants and auxiliary structures exist as after any later timestep.

### Normal topology-stable CFD timestep

The ownership scan identifies a typically small migration frontier. No topology journal entries exist. If owners and replica sets do not change:

- no topology records are sent;
- no pool slots move;
- no topology or geometry rebuild occurs;
- the cached ghost position schedule remains valid;
- the collective path decision is the only redistribution communication in automatic mode.

### Remeshing or another topology edit

The mutation journal supplies additions and tombstones. Those records enter the same action arena as ownership-induced changes. The same replica-set rules determine new holders, the same stream carries the records, and the same pool primitives apply them.

There is no topology-epoch branch to the old full reconstruction algorithm.

### Long-distance movement

The planner detects a destination outside the graph. The agreed global route transports the same stream. The receiver applies the same records. No global/full rebuild implementation exists.

### Empty and reappearing bodies

After a body leaves a rank, the last local entity removal destroys the local object. If the body later returns, `BODY_BEGIN` recreates it. A rank with zero local objects still participates in collective communication and may receive a new object.

## API and header changes implied by sparse objects

The primary API should remain ensemble-wide:

```c
int MFTLRedistribute(MFTL_Ensemble *ensemble);
int MFTLRedistributeWithOwners(MFTL_Ensemble *ensemble,
                               const MFTL_OwnerMap *local_maps);
```

However, the current inline dispatch and single-object API conflict with removing empty object replicas:

- `MFTLRedistribute()` must never return early because the local object list is empty; that rank may receive geometry and must enter the collective.
- It must not dispatch based on the local `num_objects`, because local object counts differ.
- `MFTLRedistributeObject(MFTL_Object *object)` cannot be a safe collective when ranks without that body have no object pointer or domain through which to participate.

The clean architecture is:

- make the ensemble-wide entry point the one production redistribution API;
- replace the pointer-only single-object collective with an ensemble-and-global-ID form if single-body selection is still needed;
- make `MFTLRedistributeListOfObjects()` either the implementation of `MFTLRedistribute()` or remove the redundant public distinction;
- key explicit owner maps by locally active object/global ID rather than replicated list position.

This is an API migration, not a second redistribution implementation. A permanent compatibility path that recreates empty placeholders would defeat the scaling goal and should not be retained.

Sparse objects also affect code outside `redistribute.c`:

- `ghost_sync.c` must use body global IDs and the cached schedule.
- `spatial_halo.c` and `MFTL_SpatialHalo` must key patches and radii by body global ID rather than global list index.
- ensemble global-count and quality reductions must use sparse keyed aggregation rather than arrays of `num_objects`.
- remeshing owner maps and statistics must not assume identical object lists.
- examples must iterate local active objects and use global IDs when referring to a specific body.

These are necessary consequences of eliminating empty objects, not optional redistribution embellishments.

## Proposed replacement core

The intended high-level structure is:

```c
int redistribute_core(MFTL_Ensemble *ens,
                      const MFTL_LocalOwnerMaps *maps)
{
    begin_transaction(ens);

    classify_current_and_next_owners(ens, maps);
    import_topology_mutation_journals(ens);
    mark_migration_frontiers(ens);
    plan_triangle_replica_deltas(ens);
    plan_vertex_handoffs_and_reference_deltas(ens);

    aggregate_and_deduplicate_actions(ens);
    bool use_global = collectively_select_route(ens);
    direct_pack_record_streams(ens, use_global);
    exchange_record_streams(ens, use_global);

    parse_and_validate_records(ens);
    reserve_affected_monolithic_pools(ens);
    apply_all_object_transactions(ens);
    commit_owner_copy_schedules(ens);
    remove_empty_local_objects(ens);

    end_transaction(ens);
    return MFTL_SUCCESS;
}
```

There should be no `if (topology_changed) full_rebuild(); else incremental();` split. Differences between initial distribution, remeshing, neighbor migration, and global migration are expressed entirely in the planned records and chosen transport.

## Expected complexity

Let:

- `N` be locally held owned entities;
- `H` be locally held ghosts;
- `M` be entities changing owner;
- `ΔH` be triangle replica additions/removals;
- `J` be journaled topology mutations;
- `Blocal` be locally active bodies;
- `d` be graph degree;
- `P` be MPI ranks.

| Work or storage | Current behavior | Unified delta target |
|---|---:|---:|
| Ownership classification | `O(N + H)` | `O(N + H)`, optionally fused |
| Topology payload | approximately complete local halo | `O(M + ΔH + J)` |
| Redistribution pool work | `O(N + H)` per rebuilt object | `O((M + ΔH + J) · local valence)` |
| Retained geometry/topology | rewritten/rebuilt | unchanged |
| Ghost position data | `O(H)` plus request/reply overhead | `O(H)` one cached push |
| Object metadata/loops | `O(Bglobal)` per rank | `O(Blocal)` plus optional `O(Bglobal/P)` directory metadata |
| Neighbor-path target arrays | currently often `O(P)` | `O(d)` |
| Global-path target arrays | `O(P)` | `O(P)` only when used |
| Steady allocations | repeated | zero after buffer capacities stabilize |

This is scalable in the practical weak-scaling sense when local mesh size, local halo, migration rate, topology-change rate, active-body count, and graph degree remain bounded.

It is not perfectly scalable:

- automatic path selection retains one global Allreduce per call;
- the global route has `O(P)` count metadata and collective latency when selected;
- monolithic pool growth occasionally copies a complete local object pool;
- owned-prefix changes require slot swaps and incident-index updates;
- ghost position bytes remain proportional to the local halo;
- strong scaling still reaches the surface-to-volume and latency limits.

Those limitations are compatible with the fixed requirements and are much smaller than complete halo retransmission and full-object reconstruction every timestep.

## Detailed implementation plan

### Phase 0: Baseline and invariants

1. Add phase timers to the current implementation:
   - ownership;
   - destination planning;
   - path Allreduce;
   - packing/flattening;
   - count exchange;
   - payload wait;
   - arrival grouping;
   - pool reserve/copy;
   - topology repair;
   - derived rebuild;
   - ghost request, reply, and apply.
2. Add counters:
   - entities scanned;
   - owner changes;
   - dirty triangles;
   - triangle adds/drops/handoffs;
   - vertex upserts/handoffs;
   - bytes and records per target;
   - pool swaps and reallocations;
   - locally active and globally known body counts;
   - neighbor/global path frequency.
3. Record baselines for `parallel_advection_scaling` and `parallel_advection_redistribute_list`.
4. Add debug invariant checkers for prefix ownership, GID uniqueness, reference counts, edge twins, replica completeness, object directory uniqueness, and ghost schedule generation.

Exit gate: the current timing decomposition and byte counts explain the existing examples without changing their results.

### Phase 1: Monolithic pool mutation foundation

1. Add persistent sparse GID maps.
2. Add vertex reference counts and half-edge incidence arrays.
3. Add the persistent edge map.
4. Implement and unit-test vertex-slot swap.
5. Implement and unit-test triangle-slot swap.
6. Implement insert, erase, promote, and demote only through those swaps.
7. Add controlled geometric capacity growth.
8. Make derived-state invalidation precise to touched entities internally while keeping public derived-state postconditions.

Test every primitive against a complete `MFTLObjectBuildDerived()` reference after each operation.

Exit gate: randomized sequences of pool operations preserve all monolithic arrays, prefixes, GID maps, half-edge twins, incidence, and derived connectivity.

### Phase 2: Sparse object identity

1. Change wire identity from body list index to body global ID.
2. Add the ensemble body-GID directory.
3. Add internal create-on-arrival and remove-empty-object operations.
4. Make empty local ensembles participate in collectives.
5. Remove same-list/same-order assumptions from redistribution planning and receive grouping.
6. Update explicit owner-map association.
7. Update ghost sync, spatial halo, statistics, remeshing, and examples that use body indices.
8. If required, add the partitioned lightweight global body metadata directory.

Exit gate: a many-body test holds each object only on ranks intersecting its halo, ranks with no objects can receive one, and no communication code scans `Bglobal`.

### Phase 3: Unified protocol and transport seam

1. Define versioned body and record headers.
2. Implement the persistent action arena.
3. Implement aggregation and per-destination vertex de-duplication.
4. Replace the three current payloads with one direct-packed byte stream.
5. Refactor `exchange.c` so it does not make an internal flattening copy.
6. Implement the neighbor-collective backend.
7. Implement the direct point-to-point neighbor backend for measurement.
8. Keep the global collective backend using the same stream.
9. Add byte-for-byte parser validation and malformed-record tests.

At this phase the producer may initially emit a complete equivalent state transaction to exercise the protocol, but it must use the new single parser and pool primitives. That temporary development mode is removed when delta planning is enabled.

Exit gate: all transports produce identical committed states from identical action streams, including zero-byte ranks and empty receivers.

### Phase 4: Ownership and replica deltas

1. Add committed/next owner storage.
2. Add epoch-stamped frontier marking.
3. Compute `Rold` and `Rnew` for authoritative triangles.
4. Emit triangle add/drop/handoff actions.
5. Emit and de-duplicate required vertex upserts.
6. Add vertex holder reference counts and aggregated reference-delta actions.
7. Add vertex ownership handoff with copy-rank state transfer.
8. Preserve retained ghosts and geometry.
9. Apply records using the staged monolithic transaction.
10. Remove the complete kept-list/rebuild-object flow.

Exit gate: a topology-stable timestep with no owner or replica-set change sends zero topology bytes and moves zero pool slots.

### Phase 5: Topology mutation journal

1. Define journal records for triangle/vertex create, delete, and connectivity change.
2. Route every MFTL topology mutator through journal-aware helpers.
3. Integrate remeshing operations.
4. Represent initial distribution as additions from the uncommitted generation.
5. Add debug detection of unjournaled topology changes.
6. Remove any remaining production branch to the old reconstruction algorithm.

Exit gate: initial distribution, topology-stable movement, remeshing, and arbitrary explicit owner maps all execute the same transaction core.

### Phase 6: Cached ghost push

1. Build owner-to-copy membership from vertex holder reference counts.
2. Freeze GID-based schedules into current local slots after transaction commit.
3. Replace per-call request construction and reply matching in `ghost_sync.c`.
4. Reuse counts, displacements, requests, and data buffers.
5. Support positions and vertex normals through the same cached schedule.
6. Add generation checks that reject use of a schedule older than the committed redistribution.

Exit gate: ghost position synchronization uses one data exchange, no request/reply phase, no GID lookup on the steady apply path, and no steady allocation.

### Phase 7: Transport measurements and selection

Benchmark at least:

- MPI ranks: 1, 2, 4, 8 on the MacBook and a larger range on the target cluster;
- graph degrees caused by face, edge, and corner neighborhoods;
- zero, tiny, medium, and large deltas;
- 0%, 0.1%, 1%, 10%, and high migration fractions;
- one large connected surface and many small spheres;
- topology-stable and remeshing timesteps;
- same-node and multi-node placement;
- neighbor-collective and direct point-to-point backends;
- blocking, nonblocking, and persistent variants where supported;
- neighbor and global route frequency.

Measure median, tail latency, CPU time, bytes, message count, allocations, wait time, and overlap—not only total wall time.

Select the backend by evidence. Avoid payload-size heuristics unless benchmarks show a stable crossover on more than one supported MPI implementation.

Exit gate: the production default is documented with benchmark evidence, and any retained alternate backend has a demonstrated regime where it is beneficial.

### Phase 8: Remove the old production implementation

1. Run the old and new implementations as differential test oracles on randomized small cases.
2. Archive benchmark results and invariant comparisons.
3. Remove the old implementation and temporary full-state producer from production sources.
4. Keep only the unified core and chosen transport configuration.
5. Update `include/mftl_communication.h` and `src/communication/README.md` to describe the new actual behavior.

Exit gate: there is one production redistribution algorithm and no duplicated full/incremental maintenance burden.

## Validation plan

### Correctness fixtures

Cover:

- no ownership changes;
- face, edge, and corner crossings;
- simultaneous opposite-direction crossings;
- triangle handoff when the new owner already holds a ghost;
- vertex handoff with several copy ranks;
- an old owner that remains a ghost holder;
- a body leaving a rank completely;
- a body arriving on a previously empty rank;
- a rank with an empty local ensemble entering the collective;
- several bodies creating and disappearing on the same step;
- periodic seams;
- physical boundaries;
- a destination outside the neighbor graph;
- rank-zero initial distribution;
- explicit owner maps;
- remeshing additions and tombstone removals;
- ID holes and renumbering;
- zero-length messages and zero-neighbor cases.

### Differential checks

Compare old and new states by body and entity global ID:

- owner ranks;
- owned and ghost sets;
- triangle replica sets;
- vertex-complete halo;
- positions and flags;
- triangle vertex order;
- half-edge flags and twins;
- `out_he_idx`;
- `tris.verts` and `tris.neighbors`;
- frames and geometry within tolerance;
- global counts;
- cached owner-to-copy membership.

### Performance acceptance criteria

For a topology-stable timestep with unchanged ownership:

- zero topology payload bytes;
- zero pool swaps;
- zero full derived rebuilds;
- zero communication-buffer allocations after warm-up;
- one cached field exchange for ghost positions.

For a timestep with migration:

- topology bytes proportional to `M + ΔH`;
- pool work proportional to changed entities and local valence;
- no complete kept-entity copy;
- no scan or temporary storage proportional to `Bglobal`;
- neighbor-path count/displacement work proportional to `d`;
- identical result through global and neighbor transports where both are legal.

For weak scaling with fixed local problem size and migration fraction:

- local planning/apply time remains approximately flat;
- bytes per rank remain approximately flat;
- active-object metadata remains approximately flat;
- any remaining growth is attributable to the path-selection collective or MPI transport, not local reconstruction.

## Final recommendation

Do not keep the current full-rebuild function as the foundation, and do not add a second incremental redistribution API beside it.

Replace the internals of the existing ensemble redistribution with one sparse, delta-based transaction architecture. Preserve the monolithic arrays and owned prefixes through carefully centralized swap/insert/remove primitives. Remove locally empty mesh objects and identify bodies by global ID. Preserve both global and neighbor routes beneath the same wire protocol and benchmark point-to-point against neighborhood collectives. Use a topology mutation journal so initial distribution, remeshing, and normal CFD timesteps remain cases of the same algorithm.

This will not make every cost perfectly independent of `P`: automatic route selection and the exceptional global route retain collective scaling limits, and monolithic growth occasionally copies a pool. It does remove the dominant avoidable work—complete halo retransmission, full local object reconstruction, replicated empty objects, request/reply ghost discovery, and per-step allocation—and is the strongest scalable architecture compatible with the fixed storage requirements.

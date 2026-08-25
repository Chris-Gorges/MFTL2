# Final redistribution architecture recommendation

## Status

This document is the final recommendation after critically reviewing:

- [the original recommendation](redistribution-architecture-recommendation.md);
- [the constraint-aware update](updated_redistribution-architecture-recommendation.md);
- [the subsequent review](redistribution-architecture-recommendation-reviewed.md);
- the current implementation in `src/communication/`, `src/object/`, `src/remeshing/`, and the public headers.

It supersedes the earlier architectural recommendations. It does not implement any change.

## Executive decision

Rewrite the internals of the existing ensemble redistribution as one stateful, validate-then-apply redistribution engine.

The final engine must:

- keep the monolithic vertex and triangle pools;
- keep owned entities in a prefix and ghosts in a suffix;
- keep both neighbor and global communication routes;
- use one ownership/replica planner, one record protocol, one parser, and one commit path;
- retain existing entities and ghosts when their required membership does not change;
- communicate topology proportional to the actual change in the normal CFD regime;
- handle high-churn remeshing and initial distribution with a dense batch representation inside the same engine;
- maintain a cached owner-to-copy schedule for one-phase ghost field pushes;
- ultimately store full `MFTL_Object` instances only for locally active bodies;
- use a mandatory distributed body-metadata catalog when sparse objects are enabled;
- explicitly audit the persistent distributed state;
- remove the old full-rebuild production implementation after differential validation.

The engine is adaptive in data representation and local application density, but it is not two redistribution solutions. Sparse and dense batches share:

- ownership classification;
- replica-set calculation;
- action semantics;
- transport;
- receive validation;
- the final per-body operation plan;
- topology repair;
- postconditions;
- consistency auditing.

Only the encoding of a large action set and the low-level method used to move many pool slots differ.

## Critical assessment of the reviewed recommendation

The review materially improves the earlier proposal, but not every conclusion should be adopted unchanged.

| Review finding | Final assessment | Final decision |
|---|---|---|
| F1: small deltas do not describe remeshing | Accept the workload distinction; modify the proposed snapshot | Add authority-scoped dense batches and adaptive local apply, not an under-specified body-wide snapshot |
| F2: removing empty objects does not remove all `O(Bglobal)` state | Partly correct | Keep sparse objects as the final target, make the distributed metadata catalog mandatory, and use an active list only as a transition |
| F3: “transactional” implies rollback | Accept | Use “validate-then-apply”; validation failure is fatal |
| F4: persistent state loses the current stateless safety property | Accept | Add generations, exact local invariants, scalable membership fingerprints, and a forced global reconcile mode |
| F5: persistent edge hash duplicates touched-set twin repair | Accept | Do not add an edge map |
| F6: doubly linked vertex incidence is excessive | Mostly accept | Use the existing half-edge fan after twin repair; add at most a singly linked incidence fallback if measurement proves necessary |
| F7: slot erasure can degrade locality | Accept as a risk, reject it as an established cost | Instrument long-run locality and kernel time; do not automatically re-Mortonise until a target-machine regression is measured |
| F8: redistribution should not eagerly rebuild frames | Accept, with an explicit API change | Return with topology valid and frames invalid |
| F9: NBX eliminates the global/neighbor distinction | Reject as stated | Benchmark NBX as a global-sparse or universal transport, but retain the required global and neighbor routes |
| F10: remeshing renumbering is a generation break | Accept | Give renumbering explicit generation and cache-rebuild/remap semantics |
| F11: sparse tables make renumbering removable “for free” | Reject | Treat disabling renumbering as a later project requiring a density audit and an ID-overflow policy |
| F12: a local-box pre-test is not automatically cheap | Accept | Make it measurement-only; keep unconditional owner lookup as the baseline |
| F13: persistent-state memory was not budgeted | Accept | Add a memory budget and keep per-call state in ensemble workspace |
| F14: the earlier phase order had excessive blast radius | Accept | Land low-risk measured improvements first and sparse object identity last |
| F15: validation should name repository guards | Accept with corrected names | Use the actual invariant helpers and test functions from this tree |
| F16: smaller implementation corrections | Mostly accept | Incorporate generation headers, geometric growth, direct packing, and removal of the 32-destination de-duplication cliff |

### Where the review itself needs correction

#### 1. A body-wide snapshot is not locally selectable

The review proposes selecting `BODY_SNAPSHOT` per body from a local threshold. A distributed body has several authoritative owners. Their local churn ratios can differ, so they cannot independently decide that the receiver should discard the complete body. A receiver that gets a snapshot from one owner and deltas from another would not know which existing records the snapshot supersedes.

A full body snapshot also has no single source in the spanning case. Today, the complete body is distributed across authoritative triangle owners.

The safe dense unit is therefore an authority-scoped fragment:

```text
(body_gid, old_authority_rank, source_generation)
```

An `AUTHORITY_SNAPSHOT` replaces only the receiver's replicas whose committed authority was that source rank. The old authority can choose delta or snapshot independently because those scopes are disjoint. It sends an empty snapshot/clear marker to every correct old holder that must lose its last replica from that authority.

The receiver converts both delta and snapshot records into the same final operation plan. No body-wide mode agreement is required.

#### 2. Empty-object removal can save much more than 288 bytes

The review correctly observes that a newly created empty `MFTL_Object` is approximately 288 bytes including its list pointer. That is not the upper bound.

Pools never shrink. An object that previously held a generated or migrated mesh can have `n == 0` while retaining large vertex and triangle blocks. `MFTLObjectDestroy()` frees those blocks; merely skipping the object in an active list does not. This is especially relevant on the rank-zero generation path documented in `docs/scaling-analysis.md`.

Therefore:

- an active-index list is the correct low-risk first step for loop cost;
- destroying locally empty objects remains valuable for resident memory;
- active nonempty objects whose capacity is far above their live count also need a measured shrink policy;
- removing mesh objects still does not eliminate globally required metadata, so sparse identity and distributed metadata must be designed together.

#### 3. Snapshot resynchronization is not automatically complete

An authority can send a full description to every holder it believes exists. It cannot remove a stale copy on an unknown rank if its copy-rank state is already corrupt.

Consequently, an audit failure must not silently trigger an ordinary authority snapshot and continue. The default response is a fatal consistency error with diagnostics.

An optional repair mode must be a communicator-wide reconcile generation:

1. all ranks enter the same reconcile generation;
2. all non-authoritative replicas are discarded;
3. current authoritative owned entities are reclassified geometrically;
4. ownership handoffs are completed;
5. authoritative fragments repopulate the exact halo;
6. copy schedules and fingerprints are rebuilt from empty.

This is a forced dense invocation of the same planner, protocol, parser, and commit machinery. It is not an ordinary per-body snapshot.

#### 4. Re-Mortonisation is not yet justified

Swap-with-tail can degrade locality, but the current measurements show that a roughly 160-fold improvement in the edge-index-distance metric produced no measurable kernel improvement on the development MacBook. The reviewed document converts an unmeasured long-run risk into a scheduled remedy too early.

It also overstates reproducibility. Morton keys derived from positions are deterministic, but stable-sort tie order can depend on prior slot order, and changing triangle order can change floating-point accumulation order in downstream scatter kernels.

The final recommendation is:

- track a disorder metric and downstream ns/triangle over long runs;
- benchmark 10³–10⁴ timesteps on the target cluster;
- enable reordering only if kernel degradation exceeds its amortized reorder cost;
- if enabled, tie-break by global ID, run inside commit, rebuild all slot caches, and define whether bitwise or tolerance reproducibility is required.

#### 5. NBX does not simultaneously remove and retain route selection

NBX can exchange sparse messages with arbitrary ranks without an `O(P)` count array or a prior destination-discovery Allreduce. It is a valuable transport candidate.

However:

- if the engine still dynamically chooses between a neighborhood collective and a global collective, ranks still need a consistent path decision;
- the Allreduce disappears only when NBX is selected as the transport for the entire redistribution call, or when the caller guarantees neighbor-only traffic;
- an NBX barrier is still a communicator-wide termination operation;
- progress and probe performance vary materially between MPI implementations.

Both required routes therefore remain. NBX is benchmarked as an additional global-sparse backend and, optionally, as a universal configured backend. It does not erase the architectural distinction by declaration.

#### 6. Stopping ID renumbering is not free

Sparse local GID tables remove the largest memory reason for dense IDs, but other issues remain:

- the current invariants and tests require dense IDs;
- half-edge IDs are derived as `3 * triangle_gid + k`;
- IDs are 32-bit integers in records and wire formats;
- indefinite split/collapse churn can overflow monotonic 32-bit counters;
- remeshing, I/O, and other modules must be audited for direct indexing and density assumptions.

Keep renumbering initially and handle it as an explicit generation break. Disabling it is a follow-on only after an exhaustive audit and either a safe ID-reuse protocol or migration to a sufficiently wide ID type.

## Fixed constraints and final postconditions

### Storage constraints

The existing structures remain:

- `MFTL_VertexPool` is one monolithic allocation containing all vertex arrays.
- `MFTL_TrianglePool` is one monolithic allocation containing all triangle, half-edge, geometry, and derived arrays.
- Owned vertices and triangles occupy `[0, n_local)`.
- Ghost vertices and triangles occupy `[n_local, n)`.
- Positions remain authoritative only in `verts.pos`.
- Half-edges remain implicitly triangle-indexed.

Auxiliary arrays may be attached to an object or ensemble. They do not create a second vertex or triangle storage layout.

### Redistribution postconditions

After a successful commit:

1. Every vertex and triangle is owned by the rank selected by the active owner rule.
2. Every authoritative triangle is present on:
   - its triangle owner;
   - the owners of its three vertices;
   - the owners of its three edge-neighbor triangles.
3. Every local triangle has its three vertices.
4. Owned and ghost prefixes/suffixes are valid.
5. Half-edge twin symmetry and `out_he_idx` are valid.
6. `tris.verts` and `tris.neighbors` match the half-edge records.
7. `MFTL_DERIVED_TOPOLOGY` is set.
8. `MFTL_DERIVED_FRAMES` is clear.
9. The owner-to-copy schedule matches every locally held ghost vertex and carries the committed generation.
10. The local body directory contains exactly the bodies with local mesh state.
11. The distributed body catalog contains the metadata required for every global body.

Triangle and vertex geometry arrays remain unspecified after redistribution, as they are today. Retained values may remain in memory but are not a public validity guarantee.

### Validate-then-apply guarantee

All received data is parsed, bounds-checked, generation-checked, de-duplicated, conflict-checked, and converted into per-body operation plans before any pool mutation begins. Required capacities are reserved before mutation.

After apply starts, the code path contains no expected allocation or validation failure. There is no rollback. A validation error is a fatal distributed invariant violation.

## Final state model

### 1. Per-object persistent state

Only data that must survive between redistributions belongs here:

- sparse vertex-GID-to-slot table;
- sparse triangle-GID-to-slot table;
- local triangle-reference count per vertex;
- owner-side sparse copy-rank reference counts for vertices;
- committed distribution generation;
- topology generation;
- coalesced topology mutation journal;
- optional long-run locality statistics.

The GID tables are flat open-addressed arrays sized from the local owned-plus-ghost count. This is the justified exception to the repository's no-hash-map preference: IDs are dense across the global body but sparse on a rank, so direct-address tables are proportional to the global body rather than the local partition.

Do not add:

- a persistent edge-key map;
- a doubly linked half-edge incidence structure;
- per-call dirty arrays;
- per-call next-owner arrays.

If fan-walk vertex updates are later measured to be a bottleneck, the only permitted incidence fallback is a singly linked target list:

```text
vertex_he_head[verts.capacity]
he_vertex_next[3 * tris.capacity]
```

It must be justified by measurements before becoming persistent state.

### 2. Per-ensemble reusable workspace

Per-call and transport state belongs here:

- next-owner scratch;
- epoch-stamped vertex and triangle dirty marks;
- touched half-edge list and twin-repair buckets;
- outgoing logical action arena;
- incoming record arena;
- per-body final operation plans;
- destination-local vertex-upsert de-duplication;
- aggregated copy-reference deltas;
- neighbor counts/displacements and request arrays;
- lazily allocated global counts/displacements;
- persistent send and receive byte buffers;
- temporary fan-walk lists;
- cached ghost field schedules;
- timers, counters, generation diagnostics, and fingerprint buffers.

All arrays grow geometrically and are reused. A steady call whose previous high-water marks are sufficient performs no allocation.

### 3. Sparse local objects and the body catalog

The final state has two levels:

```text
MFTL_Ensemble
├── local active MFTL_Object instances
│   └── full monolithic mesh pools and per-object persistent state
└── distributed body metadata catalog
    └── small metadata entries partitioned by deterministic home rank
```

Wire records and APIs use `body_gid`, never object-list position.

The catalog is not optional once empty objects are removed. It owns or routes:

- body global ID;
- global vertex and triangle counts;
- next-ID/generation metadata;
- centroid, radius, weight, or other global remeshing measurements when requested;
- the set or directory information needed to publish metadata to current holders;
- any body-level parameters currently assumed replicated.

Per-body reductions become:

```text
holders produce keyed partials
        ↓
partials route to home(body_gid)
        ↓
home aggregates deterministically
        ↓
results publish to requesting holders/algorithms
```

The current dispersed load balancer depends on a replicated global body list and deterministic list-index ordering. Sparse objects therefore require an explicit remeshing redesign:

- either retain a compact replicated metadata array as an interim compatibility layer;
- or replace the global greedy ordering with a deterministic distributed ordering/assignment keyed by `(weight, body_gid)`.

The latter is the asymptotically scalable target. The former is transitional and still costs `O(Bglobal)` metadata per rank.

When an object's local pools become empty, destroy the local object and remove its directory entry. For a nonempty object whose capacity is much larger than its live count, benchmark a `shrink-to-fit` threshold so rank-zero generation capacity and past migration peaks do not remain resident indefinitely.

## Unified planning

### 1. Ownership classification

Do not overwrite committed `owner_proc` values while planning.

Compute next owners into workspace for all locally required vertices and triangles:

- vertices from current wrapped positions;
- triangles from current cached centers;
- explicit owner maps when supplied.

The baseline calls `MFTLGetOwnerProc` unconditionally. A local-box shortcut or branchless same-cell comparison is measurement-only because the current owner lookup is already very cheap and a prior conditional extension regressed it.

The caller must still synchronize ghost positions and compute triangle geometry for owned and ghost triangles before redistribution. That is what gives every old authority consistent next owners for corners and edge neighbors.

### 2. Migration frontier

Mark:

- every vertex whose owner changes;
- every incident triangle of such a vertex;
- every triangle whose owner changes;
- the three edge neighbors of a changed triangle;
- every topology-journal entity and its affected neighborhood.

Only the current authoritative owner emits state changes for an existing triangle.

For triangle `T`:

```text
Rold(T) =
    old_owner(T)
  ∪ old_owner(three vertices)
  ∪ old_owner(three adjacent triangles)

Rnew(T) =
    next_owner(T)
  ∪ next_owner(three vertices)
  ∪ next_owner(three adjacent triangles)
```

Use a seven-element stack array and linear duplicate removal.

- `Rnew - Rold` produces triangle additions.
- `Rold - Rnew` produces triangle removals.
- A changed authority produces a handoff.
- The intersection retains the existing triangle and commits only changed identity fields.

If the old owner remains in `Rnew` it demotes the existing slot. If the new owner already holds a ghost, it promotes that slot.

### 3. Vertex presence and copy accounting

Every triangle addition implies three vertex upserts, de-duplicated by:

```text
(destination_rank, body_gid, vertex_gid)
```

The per-vertex local triangle-reference count determines lifetime:

- a non-owned vertex is removed when its count reaches zero;
- an owned vertex is retained through temporary zero counts during apply;
- counts are recomputed from topology during forced reconcile.

For the push schedule, aggregate:

```text
(body_gid, vertex_gid, holder_rank, signed_reference_delta)
```

and send it to the vertex's next owner.

On vertex handoff:

1. the old owner transfers the current nonzero holder counts;
2. topology deltas are addressed to the next owner;
3. the next owner merges base counts and deltas;
4. zero entries are removed;
5. the result becomes the committed owner-to-copy set.

### 4. Topology mutation journal

The journal records net distributed changes, not every internal instruction performed by a remeshing operator.

It must retain enough pre-change state to remove old replicas:

- deleted triangle GID and authority;
- old vertex GIDs;
- old neighbor GIDs or old replica set;
- old copy-reference contributions.

It must retain final state for:

- new triangles and vertices;
- connectivity changes;
- ownership-map changes;
- GID renames.

Journal entries are coalesced by global ID before planning. A created-then-deleted entity disappears from the final journal. A repeatedly modified triangle produces one old tombstone and one final upsert.

All topology mutators increment a topology generation. Debug builds assert that generation changes are represented by the journal. No MPI is introduced into serial remeshing operators; the journal is local state consumed by the existing remeshing communication layer.

## Sparse and dense action representation

### Why both densities must be efficient

Normal CFL-limited advection typically changes a thin partition rim. Remeshing migrations, dispersed-body gathering, generation scatter, and global-ID renumbering can change a large fraction of a local body.

A sparse-only engine would optimize the user's CFD path but risk making remeshing slower. A full-state-only engine is the current bottleneck. The unified engine therefore uses one logical operation plan with two density encodings.

### Sparse delta batches

Sparse records describe:

- vertex upserts and handoffs;
- vertex copy-reference deltas;
- triangle additions, removals, and handoffs;
- GID renames when sparse;
- body metadata generation changes.

They are used when encoded delta bytes and predicted apply work are below measured thresholds.

### Authority-scoped dense snapshots

The dense record is:

```text
AUTHORITY_SNAPSHOT_BEGIN
    body_gid
    old_authority_rank
    source_generation
    target_generation
    final vertex/triangle record counts for this receiver

fixed-size vertex and triangle arrays

AUTHORITY_SNAPSHOT_END
```

Its semantics are precise:

> Replace the receiver's currently committed replicas authored by this old authority and body with the supplied final replicas for the target generation.

This supports:

- high-churn remeshing;
- initial distribution from rank zero;
- a dense GID-renumber generation;
- authority-local consistency refresh.

It does not claim to repair an unknown stale copy on an arbitrary rank; only forced global reconcile makes that guarantee.

### Selection

Selection is authority-local and requires no per-body collective because each snapshot replaces a disjoint old-authority scope.

The cost model includes:

- encoded bytes;
- record count;
- predicted GID probes;
- predicted slot moves;
- fraction of locally held triangles affected;
- whether every GID changes;
- whether the receiver must already rebuild most topology.

Thresholds are benchmark results, not constants chosen in the design document.

## One wire protocol and packing path

Use one versioned byte stream per target, grouped by body and authority scope.

Record families:

- `BODY_BEGIN` / `BODY_END`;
- `AUTHORITY_SNAPSHOT_BEGIN` / `AUTHORITY_SNAPSHOT_END`;
- `VERTEX_UPSERT`;
- `VERTEX_HANDOFF`;
- `VERTEX_REFERENCE_DELTA`;
- `VERTEX_RENAME`;
- `TRIANGLE_ADD`;
- `TRIANGLE_DROP`;
- `TRIANGLE_HANDOFF`;
- `TRIANGLE_RENAME`;
- `RECONCILE_GENERATION`.

Every header contains:

- protocol version;
- record tag;
- byte length;
- body GID;
- source and target generation where relevant.

Do not cast unaligned byte-buffer addresses directly to record structs. Pack and unpack with explicit byte offsets or `memcpy` into aligned local structs.

Packing:

1. append logical actions to a reusable arena;
2. coalesce topology actions and copy-reference deltas;
3. de-duplicate vertex upserts without the current 32-destination bit budget;
4. choose sparse or authority-snapshot encoding;
5. histogram bytes by target;
6. prefix-sum target offsets;
7. pack directly into the final persistent send buffer.

This removes:

- per-destination triangle and vertex ID buffers;
- kept-entity arenas from the steady path;
- the separate body-count payload;
- separate vertex/triangle payload collectives;
- the flattening copy in the exchange layer;
- grouping by globally replicated object-list index;
- the `MFTL_DEDUP_BITS` fallback cliff.

The current exchange already merges payload counts into one count exchange. The benefit of one byte stream is therefore one data collective instead of two or three, fewer buffers, no body-count payload, and no extra flattening—not the removal of three independent count round trips.

Local-destination actions enter the same parsed operation plan without MPI.

## Communication routes and transport measurements

### Required route model

The production architecture retains:

1. a neighbor route for destinations in `domain->comm_graph`;
2. a global route for arbitrary destinations.

In automatic collective mode, ranks agree on the route with one communicator-wide logical OR. This is necessary when the chosen transport requires every rank to invoke the same collective.

An asserted-neighbor mode may skip that decision only when the CFD caller supplies a displacement/communication-radius contract. Debug builds verify the contract.

### Neighbor route candidates

Benchmark:

- `MPI_Neighbor_alltoall` plus `MPI_Ineighbor_alltoallv`;
- direct nonblocking point-to-point over graph edges;
- persistent neighbor collective or persistent point-to-point forms after schedule signatures stabilize.

Neighbor arrays are `O(d)`, where `d <= 26` for the current graph.

### Global route candidates

Benchmark:

- count exchange plus `MPI_Ialltoallv` as the dense/global baseline;
- NBX using synchronous nonblocking sends, probing receives, and nonblocking-barrier termination for sparse arbitrary destinations;
- sparse direct point-to-point only if it has a correct termination protocol.

Rank-sized count/displacement arrays are allocated only for the global collective backend and retained after first use.

### Optional universal NBX configuration

NBX may be configured for an entire redistribution call. In that configuration it handles neighbor and non-neighbor destinations and needs no prior path Allreduce. The neighbor and global collective routes still remain available and tested, satisfying the fixed architecture requirement.

Do not switch per rank or per body between NBX and collectives in the same call. Transport choice is communicator-consistent configuration.

Measure on the MacBook and target MPI systems:

- empty and near-empty calls;
- small halo-sized messages;
- dense remeshing messages;
- graph degree;
- same-node and cross-node placement;
- MPI progress behavior;
- median and tail latency;
- CPU polling cost;
- collective count;
- bytes and message count.

Retain more than one low-level backend only if each has a demonstrated supported regime. This does not create more than one redistribution algorithm.

## Validate-then-apply on monolithic pools

### 1. Central slot operations

All sparse pool movement uses tested internal primitives:

- swap vertex slots;
- swap triangle slots;
- insert owned/ghost vertex;
- insert owned/ghost triangle;
- promote/demote vertex;
- promote/demote triangle;
- erase vertex/triangle.

No redistribution code moves an individual pool array independently.

Triangle-slot movement updates:

- every triangle-pool array;
- triangle GID map;
- three half-edge records;
- neighboring twin indices;
- vertex `out_he_idx` values into moved half-edge ranges;
- touched derived topology.

Vertex-slot movement occurs only after triangle mutations and twin repair. The existing half-edge fan is then walkable. Gather the incident outgoing half-edges into workspace, update the corresponding incoming target references, and update:

- every vertex-pool array;
- vertex GID map;
- `out_he_idx`;
- local reference counts;
- slot-indexed schedule scratch;
- touched `tris.verts` entries.

Boundary/open fans and isolated vertices require explicit unit fixtures.

### 2. Batched sparse apply

Do not perform an unordered erase immediately for each received record.

For each object:

1. resolve all operations by GID;
2. collect holes, tail candidates, promotions, and demotions;
3. reserve once;
4. execute a batched relocation plan so each surviving entity moves at most once where practical;
5. append additions into final owned or ghost regions;
6. update maps and topology through the central movement functions.

Batching reduces random swap churn and makes sparse and medium-density cases smoother than a sequence of independent swap-with-last operations.

### 3. Dense local apply

When the final operation plan affects most of an object, build one old-slot-to-final-slot permutation in workspace and perform a bulk gather/copy of authoritative arrays, followed by one topology relink pass.

This is not the old redistribution engine:

- it does not re-plan ownership;
- it does not regenerate messages;
- it does not have a second receive parser;
- it consumes the same validated final operation plan;
- it uses the same GID maps, twin repair, generation commit, and audits.

It is a dense execution kernel for the same local plan, analogous to choosing bulk copy instead of repeated element moves.

The sparse/dense local apply crossover is measured independently of the sparse/dense wire crossover.

### 4. Apply order

1. Parse and validate all streams.
2. Verify protocol and generations.
3. Resolve or create local body shells.
4. Build final per-body GID-keyed operation plans.
5. Detect incompatible duplicate authority, add/drop, or rename records.
6. Compute final counts and reserve each monolithic pool once.
7. Select sparse batched or dense bulk local apply.
8. Upsert required vertices without ownership-prefix movement.
9. Apply triangle additions, updates, promotions, demotions, and removals.
10. Repair twins over the closed touched set using the existing bucket method.
11. Complete vertex promotions, demotions, and removals using valid fan walks.
12. Write `tris.verts` and `tris.neighbors` for touched/moved topology.
13. Commit entity owners, local reference counts, generations, and copy-rank counts.
14. Freeze the new ghost schedule from GIDs into current local slots.
15. Set topology derived and clear frames derived.
16. Destroy locally empty objects and update the body directory.
17. Run configured audits.

### 5. Capacity policy

Use geometric growth when final counts exceed capacity. The current `MFTLObjectSetCounts()` exact reserve can reallocate a complete pool for a one-entity upward jitter; fix that independently.

Shrinking is separate:

- never shrink for small steady oscillations;
- consider shrink when `capacity > γ * live_count` for a measured `γ`;
- always free both pools when the local object becomes empty;
- perform shrink only at commit boundaries;
- include shrink cost and RSS benefit in instrumentation.

## Derived topology, frames, and geometry

Redistribution should not call full `MFTLObjectBuildDerived()`.

The relink/apply pass already knows:

- each triangle's three vertex slots;
- each half-edge's repaired twin.

Write `tris.verts` and `tris.neighbors` inline or in a touched-only pass.

Do not rebuild `tris.anchor` and `tris.edge` in redistribution. Every CFD timestep moved vertices, so frames were already invalid before redistribution. Building 72 bytes per triangle merely to enable a later geometry fast path has not paid in the measured production loop.

New contract:

```text
after redistribution:
    topology valid
    positions valid
    frames invalid
    geometry fields unspecified
```

Callers that immediately require `MFTLEnsembleUpdate()` or `MFTLComputeVertexNormals()` must call `MFTLObjectRefreshFrames()` first, or those APIs must later gain an explicitly designed fallback. This is a public behavioral change and must be reflected in `include/mftl_communication.h`, examples, and tests.

## Cached ghost field synchronization

At commit, sort schedule entries deterministically by:

```text
(peer_rank, body_gid, vertex_gid)
```

Both owner and holder derive:

- the same entry count;
- the same order;
- current local source or destination slots;
- the same schedule generation.

Normal position synchronization becomes:

```text
pack owned positions in cached order
        ↓
one data exchange using cached counts
        ↓
verify generation
        ↓
write directly to cached ghost slots
```

No per-call request list, GID scatter, reply construction, or reply matching remains.

The generation word is mandatory. A mismatch is fatal; it is not treated as a recoverable missing message.

The schedule transport follows its actual peer set:

- neighbor exchange when every owner/copy edge is in the graph;
- global or NBX exchange when a remeshing/global placement creates non-neighbor copy edges.

`MFTLUpdateGhostVertexNormals()` reuses the same schedule with a different field pointer.

## Persistent-state safety

### Generation model

Track:

- topology generation per body;
- distribution generation per body;
- GID generation per body;
- ghost schedule generation.

Every wire scope names its expected source generation and target generation. Apply rejects:

- stale records;
- skipped generations;
- duplicate authority for one generation;
- a schedule generated from a different committed distribution.

### Local exact checks

Debug builds check after every commit:

- pool counts and prefixes;
- GID map bijection with live slots;
- triangle-reference counts;
- twin symmetry;
- `out_he_idx` validity;
- derived topology agreement;
- local GID uniqueness;
- vertex-complete local halo;
- schedule slots matching their GIDs and owners.

### Scalable distributed fingerprints

Do not add an `O(Bglobal)` Allreduce per audit.

Build two independently mixed 64-bit fingerprints plus counts for:

1. expected triangle memberships emitted by authoritative triangles:
   `(body_gid, triangle_gid, holder_rank, owner_rank)`;
2. actual locally held triangle memberships;
3. actual ghost vertex subscriptions emitted by holders:
   `(body_gid, vertex_gid, holder_rank, owner_rank)`;
4. owner-side schedule memberships.

Reduce a constant number of scalar fingerprints/counts across the communicator. Equality is probabilistic, so:

- Debug failures trigger detailed keyed diagnostics;
- tests perform exact gathers/comparisons for bounded fixtures;
- Release audits use the fingerprints as corruption detection, not mathematical proof.

Run:

- every redistribution in Debug;
- after every remesh in Release;
- every configurable `K` CFD steps in Release;
- whenever a generation or transport mismatch is suspected.

### Failure and reconcile policy

Default: report diagnostics and fail collectively.

Optional explicit repair: enter a global reconcile generation, discard ghosts everywhere, reclassify authoritative owned state, and rebuild all replicas and schedules through authority snapshots. Never attempt an ordinary local snapshot as silent recovery.

## Global-ID generation breaks

Keep current renumbering behavior during the initial rewrite.

When remeshing renumbers:

1. finish/coalesce topology mutations in the old generation;
2. obtain the owned and ghost old-to-new mappings;
3. emit rename records or choose dense authority snapshots;
4. rebuild/remap GID tables;
5. rebuild copy-reference keys and schedules;
6. bump the GID and distribution generations;
7. reject any old-generation journal or field schedule.

Because a dense renumber changes nearly every identity, authority snapshots are likely to win, but this is measured.

Only after sparse GID maps are stable should the project audit every density dependency and evaluate disabling renumbering. That decision must include:

- 32-bit exhaustion bounds for the expected remeshing duration;
- half-edge implicit-ID overflow;
- I/O format compatibility;
- tests and invariant changes;
- deterministic ID reuse or a 64-bit migration.

## Locality policy

Sparse batched movement intentionally sacrifices stable order for low migration cost. Treat locality decay as an observable, not an assumption.

Add:

- mean and percentile edge-adjacent triangle slot distance;
- vertex-gather distance;
- geometry ns/triangle;
- vertex-normal ns/triangle;
- BVH build/refit time;
- measurements over 10³–10⁴ steps.

If and only if downstream loss exceeds the amortized reorder cost on a supported target:

- reorder inside commit;
- sort owned and ghosts separately;
- use global ID as a deterministic tie-break;
- rebuild GID maps and cached schedules;
- keep frames invalid rather than calling the current full `MFTLObjectBuildDerived()`;
- test the required reproducibility level explicitly.

## Memory budget

Current live pool payload is approximately 88 bytes per vertex and 212 bytes per triangle before alignment.

Target persistent additions:

| State | Approximate scaling | Decision |
|---|---:|---|
| sparse vertex GID table | about 8 B/entry divided by load factor | keep |
| sparse triangle GID table | about 8 B/entry divided by load factor | keep |
| local vertex triangle-reference count | 4 B/local vertex | keep |
| owner copy-rank counts | proportional to boundary holders | keep |
| generations | constant per object | keep |
| coalesced mutation journal | `O(J)` between commits | keep |
| edge map | potentially per half-edge | reject |
| doubly linked incidence | 24 B/triangle + 4 B/vertex | reject |
| dirty and next-owner arrays | `O(N + H)` workspace high-water mark | not per object |
| action, receive, fan, and repair scratch | proportional to peak call | ensemble workspace |

Exact structure size and load factor must be reported by instrumentation before merge. The target is roughly 16 bytes per locally held entity plus boundary copy state, not the +20–40% persistent growth of the earlier proposal.

Memory acceptance also includes:

- zero steady allocations after warm-up;
- no direct-address array sized by global body entity count;
- no full mesh pools for locally empty bodies in the final state;
- no per-rank full body metadata in the final distributed-catalog state;
- measured shrink or destruction of stranded pool capacity.

## Complexity

Let:

- `N` = locally owned entities;
- `H` = local ghosts;
- `M` = ownership changes;
- `ΔH` = replica membership changes;
- `J` = journaled net topology changes;
- `Blocal` = local active bodies;
- `d` = graph degree;
- `P` = ranks.

| Cost | Normal advection, sparse change | Dense remeshing/initial distribution |
|---|---:|---:|
| owner classification | `O(N + H)` | `O(N + H)` |
| logical planning | `O(M + ΔH + J)` after frontier marking | `O(N + H)` |
| topology bytes | `O(M + ΔH + J)` | `O(N + H)` authority snapshots |
| local pool movement | proportional to affected entities × valence | one bulk permutation/relink |
| topology derived work | touched set | one linear relink |
| frame work | none | none |
| ghost positions | `O(H)` one cached push | `O(H)` one cached push |
| body mesh iteration | `O(Blocal)` | `O(Blocal)` |
| neighbor metadata | `O(d)` | `O(d)` when legal |
| global collective metadata | none with NBX; otherwise `O(P)` | `O(P)` for global collective backend |
| allocation after warm-up | zero | zero unless a new high-water mark is reached |

Remaining limits:

- the `O(N + H)` correctness classification scan unless fused with caller work;
- automatic collective route-selection latency;
- global collective metadata when that backend is selected;
- NBX barrier/progress overhead when NBX is selected;
- occasional monolithic pool growth or shrink;
- ghost bytes proportional to halo size;
- optional locality restoration;
- physical strong-scaling surface-to-volume limits.

## API consequences

The final primary collective is ensemble-wide:

```c
int MFTLRedistribute(MFTL_Ensemble *ensemble);
int MFTLRedistributeWithOwners(MFTL_Ensemble *ensemble,
                               const MFTL_LocalOwnerMaps *maps);
```

Required changes:

- `MFTLRedistribute()` is no longer a local-count-based inline dispatcher.
- A rank with zero local objects must still enter the collective.
- `MFTLRedistributeObject(MFTL_Object *)` is removed or replaced by an ensemble-plus-body-GID collective.
- `MFTLRedistributeListOfObjects()` becomes redundant with the primary ensemble call.
- owner maps are keyed by local object/body GID, not global list position.
- spatial halo results are keyed by body GID.
- I/O and examples distinguish global body metadata from locally active mesh objects.
- the frame-valid postcondition is removed from `include/mftl_communication.h`.

During the active-list transition, current APIs may remain replicated. The final sparse-object cutover must not retain compatibility placeholders that recreate empty `MFTL_Object` instances.

## Implementation plan

### Phase 0 — Measure the current implementation

Add timers for:

- ownership;
- destination collection;
- path decision;
- packing;
- exchange count and data phases;
- wait/progress;
- arrival grouping;
- pool growth;
- kept-entity gather/copy;
- relink/twin repair;
- topology and frame rebuild;
- ghost request, answer, and apply.

Add counters for:

- scanned entities;
- owner changes;
- replica additions/removals;
- actual and encoded bytes;
- messages and targets;
- per-object affected fraction;
- allocations and copied capacity bytes;
- active, empty, and globally known bodies;
- stranded capacity on empty/small objects;
- route/backend frequency.

Run both advection examples, one large connected mesh, both remeshing strategies, and long-run locality probes.

Exit gate: phase totals reconcile with existing benchmark buckets and establish sparse/dense crossovers.

### Phase 1 — No-regret current-core improvements

- Geometric growth in `MFTLObjectSetCounts()`.
- Persistent ensemble communication workspace.
- Cached neighbor-rank indexing.
- Reused send/receive/count buffers.
- Direct packing without the exchange-layer flattening copy.
- Destination-local de-duplication without `MFTL_DEDUP_BITS`.
- Active-object index for hot local loops while lists remain replicated.

Exit gate: identical results, zero steady allocator calls, reduced copied bytes, and no hot communication loop over empty bodies.

### Phase 2 — Local mutation substrate

- Sparse local GID tables.
- Vertex reference counts.
- Slot swap/insert/erase/promote/demote primitives.
- Batched sparse relocation planner.
- Dense bulk local apply consuming the same synthetic operation plan.
- Touched-set twin repair using existing bucketing.
- Fan-based vertex reference updates.
- Randomized property tests.

Exit gate: sparse and dense apply produce identical states and match a full topology rebuild.

### Phase 3 — Unified record protocol and transports

- Versioned stream and generation headers.
- Sparse action batches.
- Authority-scoped snapshots.
- One receive parser and per-body plan builder.
- Neighbor collective backend.
- Direct neighbor point-to-point backend.
- Global Alltoallv backend.
- NBX experimental backend.

Exit gate: identical action plans commit identically through all legal backends, including zero-byte ranks.

### Phase 4 — Delta ownership and halo planner

- Committed/next ownership.
- Frontier marking.
- `Rold/Rnew` differences.
- Vertex upsert de-duplication.
- Authority handoffs.
- Copy-reference deltas.
- Sparse/dense wire and apply cost selection.
- Remove the kept-list/full-object rebuild flow from the new core.

Exit gate: unchanged ownership sends zero topology bytes and moves zero slots; sparse and authority-snapshot forms agree exactly.

### Phase 5 — Derived-state contract and ghost push

- Inline/touched topology arrays.
- Leave frames invalid.
- Build deterministic owner/copy schedules.
- Replace request/reply ghost synchronization.
- Add schedule generations and non-neighbor schedule routing.

Exit gate: one field data exchange, no request phase, no steady GID lookup, and correct explicit frame behavior.

### Phase 6 — Persistent-state audits and reconcile

- Local exact invariant suite.
- Constant-size distributed fingerprints.
- Detailed Debug diagnostics.
- Injected lost/duplicate/stale-record tests.
- Explicit forced global reconcile generation.

Exit gate: every injected state divergence is detected; reconcile restores exact membership in tests.

### Phase 7 — Remeshing journal and GID generation

- Coalesced topology journal.
- Tombstones and final upserts.
- Explicit owner-map integration.
- Renumber generation records or dense snapshots.
- Update all remeshing paths without moving MPI out of `remesh_comm.c`.
- Initial distribution as an uncommitted authority snapshot.

Exit gate: initial scatter, advection, spanning remesh, dispersed remesh, and explicit maps use one core.

### Phase 8 — Sparse object identity and distributed metadata

This is deliberately late because it has the largest API and module blast radius.

- Body-GID wire identity.
- Local active-object directory.
- Create on arrival.
- Destroy on empty.
- Distributed metadata homes and keyed aggregation.
- Sparse global counts, extents, spatial halos, I/O lookup, and quality data.
- Deterministic distributed replacement for list-index remeshing decisions.
- Remove the transitional replicated list and active index.

Exit gate: local mesh memory and hot loops scale with `Blocal`; metadata scales with `Bglobal/P`; empty ranks safely receive bodies.

### Phase 9 — Long-run locality and transport selection

- 10³–10⁴-step locality tests.
- Target-cluster kernel drift measurements.
- Reorder only if justified.
- Full transport matrix on MacBook and cluster.
- Select production defaults and remove unhelpful experimental backends.

Exit gate: defaults are supported by archived data and any retained alternative has a demonstrated regime.

### Phase 10 — Cutover and cleanup

- Differential old/new runs on randomized bounded cases.
- Remove old production redistribution.
- Remove temporary full-state producer and replicated-object transition code.
- Update headers, module README, timestep documentation, and examples.
- Run strict documentation builds.

Exit gate: one production planner/protocol/parser/commit engine remains.

## Validation gates

### Existing repository gates

Use the actual current tests and helpers:

- `tests/test_bvh_ghosting.c` at np = 1, 2, 4, 8, and 27, including cross-rank non-vacuity checks;
- `ghosts_agree_with_owners()` in `tests/test_remeshing_parallel.c`;
- `frozen_matches_replication()`;
- `test_the_two_strategies_reach_the_same_mesh()`;
- `inv_partition_is_sound()`;
- `inv_halo_is_vertex_complete()`;
- `test_twin_repair_matches_full_rebuild()`;
- `test_remeshing_mpi_is_confined`;
- `examples/leveque_remeshing_parallel.c` reproducibility;
- `tests/test_object.c` under ASan/UBSan;
- strict Doxygen and MkDocs builds after header changes.

All MPI assertions must reduce their verdict before any rank returns.

### New correctness fixtures

Add:

- no ownership or replica change;
- face, edge, and corner crossings;
- opposite simultaneous crossings;
- handoff to a rank already holding a ghost;
- old authority remaining a ghost;
- vertex handoff with multiple holder ranks;
- authority-local sparse and snapshot encodings mixed for one body;
- empty authority snapshot clearing a final old replica;
- periodic seam and walled boundary;
- global non-neighbor jump;
- empty local ensemble receiving its first body;
- body leaving a rank and freeing retained capacity;
- initial rank-zero scatter;
- topology create/delete/connectivity coalescing;
- renumber generation break;
- stale, duplicated, skipped-generation, and malformed records;
- zero-byte targets and zero-neighbor communicator cases;
- forced global reconcile after an injected unknown stale copy.

### Differential comparisons

Compare by body and entity GID:

- authority;
- owned and ghost sets;
- triangle replica memberships;
- vertex completeness;
- positions and flags;
- triangle vertex order;
- half-edge IDs, flags, and twins;
- `out_he_idx`;
- `tris.verts` and `tris.neighbors`;
- global counts;
- copy-rank membership;
- schedule generations.

Frames are expected to be invalid after the new redistribution and are not compared as current data.

### Performance acceptance

Normal unchanged timestep:

- zero topology payload;
- zero pool moves;
- zero frame/topology full rebuild;
- zero steady allocations;
- one cached ghost-position data exchange;
- no loop over globally empty mesh objects.

Sparse migration:

- bytes proportional to `M + ΔH + J`;
- apply work proportional to changed entities and valence;
- retained geometry/topology arrays not bulk-copied;
- no direct-address table proportional to global entity count.

Dense migration/remeshing:

- authority snapshots beat sparse records at the measured crossover;
- dense local apply is no slower than the current bulk rebuild baseline;
- no duplicate planner, parser, or ownership logic exists.

Weak scaling:

- local planning and apply approximately flat at fixed local load/churn;
- bytes per rank approximately flat;
- local mesh object memory proportional to `Blocal`;
- catalog metadata proportional to `Bglobal/P`;
- any remaining growth attributable to chosen MPI transport or physical halo growth.

Long run:

- downstream kernel time is stable, or measured reordering recovers more time than it costs;
- generation and membership audits remain clean over 10⁴ steps.

## Final recommendation

Adopt a complete internal rewrite of the current redistribution core, not a second public or production redistribution implementation.

The final design is one validate-then-apply engine with:

- delta replica planning for normal CFD steps;
- authority-scoped dense batches for initial distribution, remeshing, renumbering, and other high-churn cases;
- adaptive sparse or bulk mutation of the unchanged monolithic pools;
- persistent sparse local GID maps and copy schedules;
- touched-set twin repair and fan-based vertex updates instead of permanent edge/incidence structures;
- topology-valid, frames-invalid postconditions;
- cached one-phase ghost pushes;
- explicit generations, audits, and a communicator-wide reconcile mode;
- both neighbor and global routes, with neighborhood collectives, point-to-point, and NBX measured beneath the same protocol;
- local active mesh objects plus a mandatory distributed global metadata catalog.

This retains the constraints that make the rest of MFTL fast and safe while removing the main redistribution costs: complete halo retransmission, full-object reconstruction in sparse timesteps, request/reply schedule discovery, per-step allocation, global-size GID tables, empty-object pool residency, and hot loops over globally replicated empty meshes.

It is not perfectly scalable. Ownership classification, ghost bytes, occasional monolithic growth, global-route termination, and strong-scaling halo physics remain. It is, however, the most scalable coherent architecture compatible with the monolithic arrays, owned prefixes, mandatory per-timestep redistribution, required global and neighbor paths, and the requirement to ship only one redistribution engine.

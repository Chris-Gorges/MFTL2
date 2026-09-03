# Send-only ghost synchronization implementation and test plan

## Topologically filtered broadcast to all MPI graph neighbours

Date: 2026-09-01

Related documents:

- [`ghost-sync-architecture-analysis.md`](ghost-sync-architecture-analysis.md)
- [`scalability-update-implementation-plan.md`](scalability-update-implementation-plan.md)
- [`scaling-analysis.md`](scaling-analysis.md)

## Decision and scope

Implement an experimental owner-side, send-only ghost-field backend with these
rules:

1. derive exactly which owned vertices are replicated from local topology (accordingly to `MFTLRemeshMarkFrozen()`);
2. send every such vertex to every rank in `domain->comm_graph`;
3. do not perform exact owner-to-holder discovery;
4. let each receiver match the incoming `(body, vertex)` key against the ghosts
   it actually holds and discard the rest;
5. use neighbourhood collectives only;
6. retain the current request/reply implementation as the reference backend;
7. do not change `MFTLRemeshMarkFrozen()` until the experiment has passed the
   performance and scaling gate.

This is a deliberately staged decision. The topological predicate is exact, but
the recipient set is a conservative superset. The experiment asks whether one
fewer neighbourhood collective and removal of owner-side request processing
outweigh sending the filtered list to neighbours that do not hold every entry.

The backend is valid only under the same graph-local ownership condition the
current `MFTLUpdateGhostPositions()` relies on:

```text
for every ghost vertex held by rank H:
    owner(ghost) is an in-neighbour of H in domain->comm_graph
```

The prototype must detect violations after the exchange as missing updates. It
must not silently leave a stale ghost value.

## Non-goals for the first experiment

The first implementation will not:

- construct or maintain an exact owner-to-holder schedule;
- derive recipients geometrically;
- add motion bounds or process-box skin tests;
- add a global arbitrary-rank transport fallback;
- change redistribution;
- change sparse-object or global-body-list semantics;
- replace the existing request/reply production default;
- modify the public remeshing API;
- use MPI persistent collectives.

Those changes would make it impossible to tell whether the simple send-only
algorithm itself is responsible for a measured difference.

## 1. The algorithm in simple steps

Assume rank A owns vertex `v` and one or more neighbouring ranks hold ghost
copies.

### Current request/reply

```text
holder -> owner:  please send (body, vertex)
owner:            build GID lookup and find the owned slot
owner -> holder:  (vertex value, vertex GID)
holder:           write the reply into the recorded ghost slot
```

This needs three neighbourhood collectives: counts, requests and replies.

### Experimental send-only broadcast

```text
owner:                 identify replicated owned vertices from topology
owner -> all neighbours: broadcast (body, vertex, value)
each neighbour:        update the key only if it owns a matching ghost copy
```

This needs two neighbourhood collectives:

1. exchange one candidate count with each graph neighbour;
2. exchange the variable-length candidate payloads.

There is no holder request and no reply phase. False-positive arrivals are
expected and are discarded, but every actual ghost must match exactly one value
from its recorded owner.

## 2. Exact topological send filter

### Required predicate

An owned triangle is remotely replicated exactly when at least one of these is
true:

- one of its three corner slots lies in the ghost vertex suffix;
- one of its three edge-neighbour slots lies in the ghost triangle suffix.

Every ghost triangle is replicated by definition. An owned vertex must be sent
when it is incident to at least one replicated triangle.

In terms of the current owned prefixes:

```text
owned triangle t is replicated iff
    any tris.verts[3*t+k]     >= verts.n_local
    or any valid
        tris.neighbors[3*t+k] >= tris.n_local

ghost triangle t is replicated unconditionally

owned vertex v is a candidate iff
    v is a corner of any replicated owned or ghost triangle
```

The edge-neighbour clause is mandatory. “An owned vertex with a ghost triangle
in its one-ring” is not sufficient: an owned triangle may travel because its
edge neighbour is remote, taking all three of its vertices with it.

### Prototype implementation

Do not call `MFTLRemeshMarkFrozen()` in the first experiment. It currently:

- clears and writes `MFTL_FLAG_FROZEN` on vertices and triangles;
- calls `MFTLObjectMarkPartitionDirty()`;
- provides the base for later remeshing dilation.

Calling it from every ghost field update would dirty otherwise clean bodies and
could overwrite a dilated remeshing band.

Instead, add a private prototype collector in the ghost-sync implementation:

```c
collect_replicated_owned_vertices(const MFTL_Object *obj,
                                  unsigned char *vertex_mark,
                                  mftl_int_buffer *owned_slots);
```

Its intended work is:

1. clear `vertex_mark[0..verts.n_local)`;
2. stream the owned triangle topology arrays;
3. for every replicated owned triangle, mark its owned corner slots;
4. stream the ghost triangle suffix and mark its owned corner slots;
5. compact the marks into `owned_slots` in deterministic object/slot order.

The mark array and slot buffer belong to `comm_workspace` and grow
geometrically. The collector must not modify the object, flags, derived bits,
counts, ownership, pool order or `partition_dirty`.

Duplicating the small predicate temporarily is intentional: it isolates the
communication experiment. The duplication is removed only after the go/no-go
decision in §10.

## 3. Wire record and MPI exchange

### Payload

Use a field-independent key and one three-vector:

```c
typedef struct {
  double value[3];
  int body_gid;
  int vertex_gid;
} MFTL_GhostBroadcastPayload;
```

Expected size: 32 bytes. The two integers use the padding present in the current
three-double-plus-one-int reply layout. Confirm this with a `_Static_assert` and
an MPI datatype extent test rather than assuming the ABI.

Use `MFTL_Object.global_id` as `body_gid`. The new wire format must not introduce
another dependency on equal local object-list positions. While the replicated
list still exists, maintain a persistent `body_gid -> MFTL_Object *` lookup in
ensemble communication state and rebuild it only when the ensemble list changes.

### Neighbour broadcast

The same candidate payload is sent to every outgoing graph neighbour, so prefer
the collective that expresses exactly that:

1. `MPI_Ineighbor_allgather` exchanges one payload count per neighbour;
2. pack the local candidate list while the count request is in flight where the
   implementation permits useful overlap;
3. compute receive displacements;
4. `MPI_Neighbor_allgatherv` sends the same packed list to every graph neighbour.

This avoids constructing `send_bufs[rank]` copies of an identical list. If the
target MPI implementation performs poorly for neighbourhood allgatherv, retain a
compact `[d]` `MPI_Neighbor_alltoallv` implementation as a benchmark alternative,
not as a different algorithm.

Query and record the graph communicator's in-degree and out-degree rather than
assuming they are identical. The current Cartesian neighbourhood is intended to
be symmetric; assert that property in Debug and add a construction test. The
allgather sends to out-neighbours and receives from in-neighbours, so receiver
owner validation must use the incoming peer list.

Every rank enters both collectives, including ranks with:

- zero objects;
- zero owned vertices;
- zero candidates;
- zero received candidates.

Use non-null one-element fallback buffers for zero counts, matching the existing
MPI portability policy.

### Traffic model

Let:

- `F_r` be the number of topologically selected owned vertices on rank `r`;
- `d_r` be its number of distinct graph neighbours;
- `G` be the global number of actual ghost copies;
- `A = sum_r(F_r * d_r)` be the number of broadcast arrivals.

Ignoring count metadata:

```text
current pull bytes       = 40 * G
broadcast bytes          = 32 * A
broadcast / pull         = 0.8 * A/G
```

With `F = sum_r F_r` and replication multiplicity `m = G/F`, a regular grid is
approximately:

```text
broadcast / pull ~= 0.8 * d/m
```

The broadcast wins wire volume only when `d/m < 1.25`. This is unlikely on a
full 3-D neighbourhood, but wire volume is not the only cost: the prototype also
removes one collective, request construction/grouping, and the owner-side
`O(owned vertices)` GID scatter. Measurements must decide the result.

## 4. Receiver matching and correctness

### Ghost-only lookup

Build a workspace-owned lookup keyed by:

```text
(body_gid, vertex_gid) -> (object pointer, ghost slot, expected owner rank)
```

Populate it only from each object's ghost vertex suffix. Owned vertices must
never be insertion targets.

The first implementation may rebuild this lookup once per broadcast call. Time
it separately. If the backend passes the main gate and lookup construction is
material, cache it by a pool-layout generation in a later, isolated change.

For each incoming peer slice:

1. look up `(body_gid, vertex_gid)`;
2. if absent, count a false-positive arrival and discard it;
3. if present but `expected_owner_rank != sender_rank`, report an invariant
   failure and do not write;
4. if the same ghost is matched twice, report an invariant failure;
5. otherwise copy the three doubles into the ghost field slot and mark it
   updated.

After application, every local ghost vertex must have been updated exactly once.
Any unmatched ghost includes the current non-neighbour-owner/orphan case and
causes `MFTL_ERROR` after all ranks have completed the neighbourhood
collectives. Debug builds assert at the cause; Release builds report counts.

### Field behavior

The position and normal entry points share the algorithm and differ only in the
selected source/destination array, as they do today.

After a position update, derived-state invalidation must be identical to the
current request/reply path. A normal update must not change topology, positions,
ownership or unrelated derived fields.

Do not fuse positions and normals. The benchmark must include both “positions
only” and “positions plus normals” workloads because eliminating one request
phase per field may change the result substantially.

## 5. Backend selection and instrumentation

Keep both implementations in the same build during evaluation:

```text
MFTL_GHOST_SYNC_PULL_REFERENCE
MFTL_GHOST_SYNC_TOPO_BROADCAST
```

Use an internal/test setter whose value must agree on every rank. Do not change
the public default until the production gate passes. Avoid compile-only A/B
executables because small build differences can obscure a sub-millisecond phase.

Add per-backend counters and timers for:

- total calls and field type;
- topological classification;
- selected vertices `F`;
- neighbour degree `d`;
- candidate arrivals `A`;
- actual ghost copies `G`;
- matched, discarded, duplicate and missing arrivals;
- count collective;
- payload collective;
- packing;
- ghost-lookup construction;
- application;
- teardown;
- logical and MPI payload bytes sent/received;
- allocations, frees and workspace high-water bytes;
- maximum-rank total phase time.

Report at least:

```text
A/G                         over-send factor
32*A / (40*G)               predicted wire ratio
matched/A                   useful-arrival fraction
classification/total        topological-filter share
lookup/total                receiver-matching share
broadcast_time/pull_time    observed phase ratio
```

## 6. Correctness test plan

### Unit tests for the topological collector

Add tests covering:

- serial object with no ghosts: empty candidate set;
- owned triangle with one ghost corner: all owned corners selected;
- owned triangle with only a ghost edge neighbour: corners selected;
- ghost triangle incident to an owned vertex: that owned vertex selected;
- several replicated triangles sharing one vertex: vertex emitted once;
- a free owned interior vertex: not emitted;
- mesh boundaries containing `MFTL_ERROR` neighbours: not mistaken for ghosts;
- empty objects and objects with vertices but no triangles;
- deterministic candidate ordering;
- input object and all flags bit-identical before and after collection;
- `partition_dirty`, counts, capacities and derived bits unchanged.

On MPI fixtures, compare the candidate truth with the existing
`frozen_matches_replication` construction: an owned vertex is selected if and
only if another rank actually holds it as a ghost.

### End-to-end field tests

Run both backends from identical cloned ensemble states and require bit identity
for:

- all owned and ghost positions;
- all owned and ghost vertex normals;
- ownership prefixes;
- vertex and triangle GIDs;
- topology and slot-order digests;
- derived-state bits expected to change or remain valid;
- return code and invariant counters.

Fixtures must include:

- one face-neighbour cut;
- process-grid edge and corner neighbours;
- graph in/out-neighbour symmetry and ordering;
- `2x1x1`, `2x2x1`, `2x2x2` and a grid with the full distinct-neighbour regime
  when hardware permits;
- periodic seams and deduplicated periodic neighbours;
- one large spanning body;
- many dispersed bodies;
- multiple bodies reusing the same per-body vertex GIDs;
- zero-candidate ranks participating beside active ranks;
- a rank with zero local objects;
- redistribution followed by broadcast;
- ownership handoff;
- remeshing and GID redensification followed by broadcast;
- `MFTLObjectReorderMorton` before broadcast;
- `MFTLObjectShrinkToFit` before broadcast;
- positions only, normals only, and both fields;
- deliberately corrupted owner rank and missing owner-neighbour relation.

For the orphan fixture, the new backend must detect a missing update rather than
silently succeeding. This is stricter than the current Release behavior and is a
required production property.

### Long-running differential test

Run at least 10,000 advection/redistribution iterations on bounded meshes with:

- backend A and backend B starting from cloned partitions;
- bit-exact field comparison after every ghost sync;
- set and slot-order partition digests after every redistribution;
- mesh invariants after every configured verification interval;
- periodic remeshing and Morton reordering;
- ASan and UBSan Debug variants.

Any difference stops the performance evaluation. A faster wrong ghost update is
not a candidate backend.

## 7. Performance experiment design

### Fair A/B protocol

For every case:

1. build one initial mesh and committed partition;
2. clone or regenerate an identical state for each backend;
3. warm both paths until workspace capacities stabilize;
4. alternate backend execution order between repetitions;
5. use identical compiler, optimization, MPI binding and verification settings;
6. exclude setup and validation from timed production regions, but report them
   separately;
7. measure at least seven repetitions and report median, minimum and dispersion;
8. reduce phase time with `MPI_MAX`, because the slowest rank sets timestep time;
9. compare both isolated ghost time and the complete timestep;
10. retain raw JSON/CSV results and process-grid shape.

Instrumentation builds establish cause. Uninstrumented builds decide speed.

### Workloads

Measure at least these two regimes:

1. **Spanning body:** one large body cut by multiple ranks. This is where the
   current ghost sync anti-scales and where the optimization has useful upside.
2. **Dispersed bodies:** hundreds or thousands of small bodies at fixed total
   triangle count. Ghost sync is a small timestep share here, so regressions must
   remain below noise at end-to-end level.

For each, measure:

- positions only;
- positions plus vertex normals;
- no movement after a settled redistribution;
- moving geometry with redistribution every step;
- remeshing at the existing representative interval.

### Process-grid shapes

Do not accept a result measured only on `4x1x1`. Test increasing neighbour
degree independently from rank count:

```text
slab:       Px1x1
pencil:     PxQx1
small 3-D:  PxQxR
interior:   a grid large enough to expose ranks with 26 distinct neighbours
```

At equal `P`, compare alternative shapes where possible. Record minimum, mean
and maximum distinct neighbour degree because periodic deduplication can make
small grids misleading.

### Strong scaling

Hold the global mesh fixed and increase `P`. Record:

- owned and ghost vertices per rank;
- `F`, `G`, `A`, `A/G` and `d`;
- phase and end-to-end speedup;
- parallel efficiency relative to the first non-serial point;
- time split between classification, count latency, payload, lookup and apply.

The broadcast is not considered scalable merely because `d <= 26`: in strong
scaling the boundary-to-volume ratio grows, and `d` also grows as the grid gains
dimensions. Compare its degradation directly with the pull backend.

### Weak scaling

Hold owned mesh work per rank approximately fixed while increasing `P`. Use a
problem construction that retains representative cut surfaces instead of merely
adding disconnected serial bodies.

Record whether:

- `F` and `G` remain bounded per rank;
- `A` grows only with bounded neighbour degree;
- workspace bytes remain `O(local mesh + received candidates + d)`;
- maximum-rank ghost time remains flat;
- the broadcast/pull time ratio worsens as interior 3-D ranks appear.

Laptop measurements are suitable for correctness, profiling and slab/pencil
trends. A production scalability decision requires cluster runs with enough
ranks to create representative 3-D interiors.

## 8. Performance go/no-go gate

The topological broadcast proceeds to production refactoring only if all of the
following are true.

### Correctness gate

- All differential tests are bit-exact.
- Every actual ghost receives exactly one update.
- No owned vertex is overwritten.
- Missing/non-neighbour owners fail diagnostically.
- No object state is modified by candidate classification.

### Efficiency gate

- The maximum-rank median ghost phase is at least 10% faster than pull in the
  representative spanning workload, outside measured run-to-run noise.
- Positions-plus-normals also wins; the result is not limited to one specially
  selected field sequence.
- The dispersed end-to-end timestep regresses by no more than 2%.
- Classification and receiver lookup do not merely replace the removed request
  work with equal or larger CPU time.
- Steady workspace allocations stop after warm-up.

### Scalability gate

- The backend remains faster than pull at the largest representative 3-D rank
  count, not only on slab grids.
- Its broadcast/pull time ratio does not deteriorate by more than 10% between
  the first and largest representative 3-D weak-scaling points.
- At fixed neighbour degree, workspace and local metadata do not grow with
  communicator size.
- No result depends on oversubscribing the four performance cores of the current
  laptop.

If correctness passes but performance or scaling fails, stop here:

- keep the request/reply implementation as production;
- archive the broadcast measurements;
- do not refactor `MFTLRemeshMarkFrozen()` merely to support a losing backend;
- use the data to decide whether to build the separate exact schedule instead.

The thresholds above should be fixed before examining final timings. Changing
them after seeing the result turns a go/no-go gate into a justification exercise.

## 9. Conditional production hardening

This phase is entered only after §8 passes.

Work:

- make the broadcast backend the opt-in production alternative;
- persist `[d]` counts, displacements and buffers;
- remove per-call allocation after warm-up;
- retain body-GID lookup generations;
- add complete statistics to scaling JSON;
- document the graph-local owner precondition;
- retain the request/reply backend as a test oracle for at least one release;
- rerun the entire redistribution, remeshing, BVH, I/O and sanitizer suite;
- repeat cluster scaling without heavy instrumentation.

Changing the default requires a second explicit decision based on the hardened,
uninstrumented results. Passing the prototype gate is permission to refactor,
not automatic permission to replace the default.

## 10. Conditional shared frozen-classification refactor

This phase is also entered only after §8 passes.

### Required semantic split

The exact topological rule is general replication information. “Frozen” is a
remeshing policy applied to that information and may later be dilated. Separate
them conceptually:

```text
pure replication classification
        |-> ghost sync: private owned-vertex send mask
        `-> remeshing: base FROZEN flags -> optional dilation
```

Do not make ghost sync call the current side-effecting
`MFTLRemeshMarkFrozen()`. Do not add a `mark_dirty` Boolean that asks callers to
know which side effects are safe. Instead, extract a shared, side-effect-free
classification kernel and retain `MFTLRemeshMarkFrozen()` as the remeshing
wrapper that deliberately writes flags and marks the partition dirty.

One possible private/public split is:

```c
/* Pure: writes caller-provided scratch, changes no MFTL_Object state. */
int MFTLClassifyReplicatedEntities(const MFTL_Object *obj,
                                   unsigned char *vertex_replica,
                                   unsigned char *triangle_replica);

/* Existing public behavior retained. */
void MFTLRemeshMarkFrozen(MFTL_Object *obj);
```

The exact API should be selected after profiling the prototype. In particular,
avoid forcing the remesher to allocate `O(vertices + triangles)` scratch on every
phase if a shared inline triangle predicate or existing remesh workspace can
preserve the current streaming cost.

### Refactor implementation order

1. Freeze the current flag and `partition_dirty` behavior in tests.
2. Extract the owned-triangle replication predicate, including the ghost
   edge-neighbour clause, into one shared pure definition.
3. Extract or share the ghost-triangle-to-owned-vertex marking rule.
4. Route the ghost broadcast collector through the pure classifier and its
   private workspace mask.
5. Route `MFTLRemeshMarkFrozen()` through the same base rule.
6. Keep the remeshing wrapper responsible for clearing/writing flags and marking
   the body partition-dirty.
7. Keep `MFTLDilateFrozenBand()` exclusively remeshing-side and never visible to
   ghost send selection.
8. Delete the prototype's duplicate predicate.
9. Re-profile both ghost classification and remeshing freeze time.

### Refactor correctness gates

- The pure classifier is bit-identical to the existing exact
  `frozen_matches_replication` truth before dilation.
- Calling the pure classifier changes no byte in the object or its pools.
- Calling it does not change `partition_dirty` or derived bits.
- `MFTLRemeshMarkFrozen()` produces exactly the same base flags and dirty stamp
  as before the refactor.
- Dilation produces exactly the same expanded band as before.
- Ghost sync selects only the base replicated set, never the dilated remeshing
  set.
- Repeated and interleaved calls to classifier, remeshing mark, dilation and
  ghost sync do not affect each other's results.
- Edge-neighbour-only replication remains covered by a named regression test.

### Refactor performance gates

- Ghost candidate classification does not regress by more than 3% from the
  accepted prototype.
- The remeshing freeze phase does not regress by more than 3% on spanning and
  dispersed fixtures.
- No new per-phase allocation is introduced after workspace warm-up.
- Persistent memory added solely for classification is reported and justified;
  an unconditional byte per global vertex is not accepted without measurement.

If the shared abstraction fails these performance gates, keep one shared inline
predicate and separate optimized output loops rather than forcing both consumers
through a slower generic callback or allocation-heavy interface.

## 11. Recommended commits

Keep the changes independently reviewable:

1. instrumentation and backend selector;
2. private topological collector plus unit tests;
3. broadcast payload and MPI datatype tests;
4. neighbour count/payload exchange;
5. receiver ghost lookup, missing/duplicate detection and field application;
6. differential correctness suite;
7. benchmark driver and raw result schema;
8. performance report and explicit go/no-go decision;
9. only after GO: production workspace hardening;
10. only after GO: shared pure replication classifier and
    `MFTLRemeshMarkFrozen()` refactor.

Do not combine commit 8 with commits 9 or 10. The evidence must be reviewable
before the conditional implementation work begins.

## 12. Final acceptance statement

The send-only broadcast is accepted only when it is both correct and faster on
representative 3-D process grids. A slab-only improvement is useful evidence but
not a scalability result.

The topological filter is already justified for correctness. The uncertain part
is economic: whether sending `F` filtered vertices to every neighbour costs less
than requesting and replying to the exact `G` ghost copies. This plan measures
that uncertainty before changing shared remeshing behavior or committing the
library to the broadcast architecture.

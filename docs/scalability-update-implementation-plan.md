# Scalability update and implementation plan

## Cached ghost schedules, bounded-rank metadata, and sparse local objects

Date: 2026-09-01

## Status and relationship to the redistribution recommendation

This document is a focused update for three findings from the laptop scaling
analysis:

1. ghost synchronization reconstructs its request/reply schedule every timestep;
2. redistribution retains communicator-wide collectives and rank-sized metadata
   even on the bounded-degree neighbor path;
3. every rank retains the complete object list, including empty mesh objects.

It turns those findings into an implementation order and answers which parts can
be changed independently without weakening correctness. It is a companion to
[`redistribution-architecture-recommendation_final.md`](redistribution-architecture-recommendation_final.md),
not a competing redistribution design. In particular, it keeps that document's
requirements:

- one redistribution engine;
- monolithic vertex and triangle pools;
- owned prefixes and ghost suffixes;
- both neighbor and global communication routes;
- global body IDs as distributed identity;
- validate-then-apply semantics;
- a dense representation for high-churn remeshing and initial distribution;
- a sparse/delta representation for steady advection.

No implementation is made by this document.

## Executive decisions

### 1. Cache ghost field schedules, and make redistribution own them

`MFTLUpdateGhostPositions()` and `MFTLUpdateGhostVertexNormals()` should become
one-way owner-to-copy pushes over a schedule committed by redistribution. A
steady field synchronization should contain:

```text
pack values from cached owner slots
        -> one payload exchange
        -> write directly to cached ghost slots
```

It must not rebuild requests, group them by object, construct an owner GID table,
send replies, or match replies by GID on every call.

The first implementation can rebuild a complete schedule snapshot while the
current redistribution still rematerializes complete dirty bodies. The final
delta redistribution maintains the same schedule with signed copy-reference
deltas. Designing the snapshot form with body GIDs and generations from the
start avoids replacing it during sparse-object migration.

### 2. Do not remove global collectives indiscriminately

The goal is not "no collectives." The goal is to distinguish:

- constant-size global decisions whose semantics are genuinely global;
- per-body global arrays caused by the replicated object list;
- `O(P)` metadata required only by the exceptional global collective backend;
- bounded-degree neighbor metadata that should be `O(d)`, with `d <= 26`.

Keep the communicator-wide route decision while automatic collective routing is
used. Keep scalar remeshing convergence reductions. Do not remove the current
dirty-body vote before its full-rebuild skip semantics are replaced: doing so
would allow one rank to skip a body that another rank is sending to it.

Independently refactor the neighbor path so its hot metadata is `O(d)`. Allocate
and retain `O(P)` count/displacement arrays only after the global Alltoallv
backend is selected. Benchmark NBX or another correctly terminated sparse global
transport before using it to eliminate the route Allreduce.

### 3. Use an active pointer view first, but do not confuse it with sparse objects

An array of pointers to locally active objects is a useful transitional change:

```text
replicated objects[]          remains the identity/compatibility list
active_objects[]              contains pointers plus stable global-list indices
```

It can remove empty-object work from advection, geometry, BVH construction,
ghost collection, and parts of redistribution while current wire records still
use global list indices. It does not remove empty `MFTL_Object` instances, global
per-body arrays, global-list memory, or the requirement that every rank know
every body.

The final state is different:

```text
MFTL_Ensemble
|- local_objects[]                         O(Blocal)
|- body_gid -> local object directory      O(Blocal)
|- communication workspace                O(local mesh + d)
`- distributed body metadata catalog      O(Bglobal / P) per home rank
```

Wire records, owner maps, spatial-halo requests, remeshing measurements, and I/O
metadata use `body_gid`, never a local or globally replicated list position.

### 4. Treat sparse local objects as a library-wide migration

Changing the meaning of `ensemble->objects` is not a redistribution-only
optimization. It changes public structure semantics and assumptions in:

- ensemble lifecycle and global counts;
- redistribution and both ghost field exchanges;
- both parallel remeshing strategies and GID assignment;
- dispersed remeshing load balancing;
- spatial-halo construction and radius aggregation;
- ensemble BVH pair discovery;
- quality statistics and global body measurements;
- mesh generation and first distribution;
- I/O, examples, diagnostics, and tests.

Land it after body-GID protocols, local object creation/destruction, and keyed
metadata aggregation exist. An early cutover would replace simple replicated
state with several incomplete special cases.

## Current contracts that must remain true

The implementation may change representation, but it must preserve these
semantic requirements.

### Collective participation

Every rank in the ensemble communicator enters redistribution, ghost field
updates, collective remeshing, spatial-halo operations, and shared-file I/O in
the same order. A rank with zero local objects is not allowed to return early: it
may receive a body or contribute an empty payload while another rank has work.

The current early returns based on `ensemble->num_objects == 0` must therefore be
audited before `num_objects` becomes a local count.

### Body identity

`MFTL_Object.global_id` is the cross-rank identity. Local object pointer, local
slot, active-list slot, and catalog-home slot are never wire identities.

### Halo membership

After redistribution, a triangle is present on its triangle owner and on the
owners of its corners and adjacent triangles. Every local triangle has all three
vertices. The cached field schedule must describe exactly the ghost vertex
suffix produced by this committed membership.

### Remeshing determinism

Sparse objects must not let ranks select different remeshing strategies, owner
maps, GID ranges, convergence outcomes, or dispersed assignments. Replicated
arrays currently provide agreement implicitly; their replacement must provide
agreement explicitly through keyed aggregation and deterministic tie-breaking.

### Neighbor and global movement

Normal geometric advection uses the graph-neighbor route. Initial distribution,
large displacement, dispersed-body gathering, remeshing coarsening, or explicit
owner maps may require arbitrary destinations. Both routes remain supported by
one protocol and commit path.

## Finding 4: cached owner-to-copy ghost schedules

### Current cost and why a cache is valid

`ghost_sync_vec3()` currently performs the following for every synchronized
field:

1. scan all local ghost vertices and build per-neighbor requests;
2. exchange request counts;
3. exchange `(object index, vertex GID)` requests;
4. group received requests by object;
5. scatter every requested object's owned vertices into a GID-to-slot table;
6. construct replies;
7. exchange replies;
8. match and apply replies;
9. free the per-call schedule storage.

Who owns a vertex and which ranks hold copies are properties of the committed
partition, not of the field being transferred. Positions and vertex normals use
the same relation. The schedule is invalidated by redistribution, object
creation/destruction, GID renumbering, and topology changes that alter replica
membership; it is not invalidated merely because a field value changes.

### Target schedule representation

Add a private ensemble-owned schedule. Per peer it stores deterministic entries
sorted by:

```text
(peer_rank, body_gid, vertex_gid)
```

Owner-side send entry:

```text
body_gid
vertex_gid
current owned source slot
holder rank
```

Holder-side receive entry:

```text
body_gid
vertex_gid
current ghost destination slot
owner rank
```

Schedule-level state:

```text
distribution generation
GID generation
route/backend
peer ranks
cached counts and displacements
send and receive byte-buffer high-water marks
valid/invalid state and invalidation reason
```

Slots are cached only after pool mutation and final ordering are complete. GIDs
remain in the entries so Debug checks can verify that cached slots still name the
expected vertex.

### Schedule construction on the current full-rebuild core

This is the lowest-risk path to a useful cache before delta redistribution.

1. While `collect_destinations()` determines every triangle holder, emit
   candidate subscriptions `(body_gid, vertex_gid, vertex_owner, holder_rank)`
   for its three corners.
2. De-duplicate by `(vertex_owner, holder_rank, body_gid, vertex_gid)`.
3. Piggyback subscription snapshots in the redistribution exchange. The current
   multi-body exchange uses three of the four supported payload slots, so a
   schedule payload can initially be evaluated without adding a second
   communication round.
4. After `rebuild_object()` fixes final pool slots, each holder constructs its
   receive entries directly from the ghost suffix.
5. Each vertex owner resolves received subscription GIDs against its final owned
   slots and constructs send entries.
6. Replace schedule entries for every globally dirty body. Retain entries for a
   body that the current two-pass vote skipped unchanged.
7. Include an explicit body-scope clear/snapshot header so owners remove stale
   subscribers when a dirty body now has no copy on a previous holder.
8. Freeze, sort, fingerprint, and increment the schedule generation only after
   redistribution commits successfully.

The schedule payload must be included in the neighbor/global route decision. A
record addressed to a non-neighbor owner is not allowed to be silently omitted.

### Schedule maintenance in the delta engine

When the redistribution planner computes `Rnew - Rold` and `Rold - Rnew`, it also
emits signed copy-reference changes:

```text
(body_gid, vertex_gid, holder_rank, +reference)
(body_gid, vertex_gid, holder_rank, -reference)
```

The next vertex owner aggregates them. Reference counts are needed because
several held triangles may require the same ghost vertex. The owner-to-copy
entry is created at the first positive reference and removed when the count
reaches zero. A vertex handoff transfers existing nonzero copy counts to the new
owner before applying the current transaction's deltas.

This turns schedule maintenance from a full snapshot into work proportional to
replica membership change.

### One-phase field push

`MFTLUpdateGhostPositions()` becomes:

1. reject an invalid or wrong-generation schedule;
2. pack three doubles per send entry from the cached source slots;
3. add a per-peer header containing protocol, field, generation, and entry count;
4. perform one neighbor/global/NBX payload exchange with cached counts;
5. validate headers;
6. write three doubles per receive entry to cached destination slots;
7. clear the affected derived-state bits for positions, as today.

`MFTLUpdateGhostVertexNormals()` uses the same schedule and a different source
and destination array.

No request count exchange is needed because both sides committed the counts.
No GID table is built on the steady path. No request/reply buffers are allocated
after warm-up.

### Generation and failure rules

- Every topology, distribution, and GID generation used by the schedule is
  explicit.
- Any pool operation that may reorder vertices invalidates cached slots until
  redistribution refreezes the schedule.
- Generation mismatch, entry-count mismatch, wrong GID at a cached slot, or an
  owner/holder disagreement is a collective invariant failure, not a skipped
  update.
- Debug builds compare constant-size fingerprints of holder subscriptions and
  owner send entries after every redistribution.
- Tests use exact gathered schedule comparisons on bounded fixtures.
- Keep the old request/reply builder only as a test oracle and initialization
  fallback during migration. Record every fallback; production steady tests must
  report zero.

### Ghost-schedule acceptance gates

- Position and normal results are bit-identical to the request/reply path.
- Exactly one data exchange occurs per field after redistribution.
- No steady-path GID-table build occurs.
- No steady allocation occurs after high-water marks stabilize.
- A rank with no local objects still enters the exchange and can receive zero
  bytes safely.
- Schedule rebuild works after body arrival, body departure, ownership handoff,
  remeshing, GID redensification, neighbor routing, and global routing.
- Cached slot checks remain clean for at least 10,000 moving/remeshing steps.

## Finding 3: collectives and rank-sized redistribution metadata

### Classify before optimizing

| Current operation/state | Purpose | Decision |
|---|---|---|
| dirty-body bitset `MPI_Allreduce(BOR)` | all ranks agree whether a full body may be skipped | retain until full-body skip is replaced by delta membership or catalog-home agreement |
| one-int route `MPI_Allreduce(LOR)` | all ranks choose the same collective communicator/backend | retain in automatic collective mode |
| neighbor count/payload exchange | variable-size bounded-neighbor transfer | retain; metadata must be `O(d)` |
| global `Alltoall(v)` counts/displacements | arbitrary destinations | retain as dense fallback; allocate `O(P)` state lazily |
| scalar remeshing convergence/profile reductions | global termination and max-rank timing | retain unless a different global convergence algorithm is designed |
| per-body count/quality/radius/GID arrays in Allreduce/Exscan | agreement indexed by the replicated global object list | replace with keyed aggregation or redesign |
| `is_neighbor[P]`, send buffers `[P]`, rank-to-neighbor `[P]` on neighbor path | convenient rank indexing | replace with compact peer-indexed structures |
| `domain.rank_map[P]` | fast per-entity Cartesian owner lookup | audit separately; it is persistent domain state, not per-redistribution metadata |

The constant-size Allreduces are not the first performance target. The per-body
payloads and `O(P)` hot metadata violate weak-scaling assumptions more directly.

### Refactor the neighbor path to `O(d)`

Replace interfaces that require `send_bufs[rank]` with a target-list interface:

```c
typedef struct {
  int rank;
  MFTL_SendBuffer payload[MFTL_MAX_EXCHANGE_PAYLOADS];
} MFTL_TargetBuffers;
```

The exact type may differ, but the semantics are:

- neighbor entries are stored in `domain->comm_graph` neighbor order;
- rank-to-neighbor lookup is a linear scan over at most 26 entries or a small
  fixed open-addressed table, never an array of length `P`;
- only touched arbitrary ranks are stored for sparse global traffic;
- statistics iterate touched targets, not every communicator rank;
- neighbor counts/displacements are `[payload][d]` and persist in the ensemble
  workspace;
- send/receive byte buffers grow geometrically and persist;
- `is_neighbor[P]` becomes `MFTLDomainNeighborIndex(rank)` or an equivalent
  compact lookup;
- payload flattening writes directly into the persistent transport image instead
  of copying per-rank buffers into a second image.

For the global Alltoallv backend, allocate `[P]` counts and displacements only on
its first use and retain them. A job that stays neighbor-only should never touch
that memory.

### Route-decision options

#### Option A: automatic collective routing -- initial default

Keep the route Allreduce. It is a constant-size safety operation and guarantees
that every rank invokes the same collective. This is the safest initial state and
allows arbitrary movements to fall back correctly.

#### Option B: asserted neighbor-only contract -- optional fast path

Allow a caller/ensemble configuration to promise that every destination lies in
the graph neighborhood. This can be justified by a displacement and halo-radius
contract in a normal CFD timestep.

- Release builds skip the route Allreduce.
- Debug/verification runs check every local destination and may use a collective
  only to report a global violation.
- A Release violation is fatal; it cannot safely switch one rank to a different
  collective after the choice point.

This mode is opt-in and must not replace the automatic default without cluster
measurements.

#### Option C: communicator-consistent universal sparse transport

Configure the entire redistribution call to use NBX or another sparse arbitrary-
destination protocol. This removes the neighbor/global collective choice and its
route Allreduce, but it introduces communicator-wide termination detection and
MPI-progress sensitivity. It must be selected consistently for the ensemble, not
per rank or per body.

Benchmark it against neighbor collectives and global Alltoallv. Retain it only if
supported target MPI implementations have a demonstrated regime where it wins.

### Replacing the dirty-body vote

Do not try to replace the current global bitset with one round of neighbor flags.
A spanning body may have holders more than one graph hop apart, and a rank that
is currently clean may receive a change produced elsewhere. One-hop agreement
would be incorrect.

There are two correct eventual replacements:

1. delta redistribution eliminates the need for a body-wide skip decision;
   changed ownership and replica sets directly name the ranks that must act;
2. while any body-wide decision remains, keyed votes route to `home(body_gid)`,
   the home combines them, and the result publishes to the known holder set.

The second requires the distributed catalog and costs two sparse communication
rounds. It is not automatically faster than a compact Allreduce at small scale,
so it should exist because sparse object identity requires it, not merely to
remove the word "Allreduce."

### Rank-metadata acceptance gates

- Neighbor-only steady calls allocate and initialize `O(d)`, not `O(P)`, rank
  metadata.
- A test can raise `P` while holding `d` fixed and observe constant workspace
  bytes for the neighbor path.
- Global arrays are absent until a global backend is first selected.
- Empty, one-target, full-neighbor, and arbitrary-target exchanges are identical
  across retained backends.
- All ranks choose the same backend, including zero-byte ranks.
- The automatic route remains behaviorally identical before experimenting with
  asserted-neighbor or NBX defaults.

## Finding 2: local object lists and distributed metadata

### Options and recommendation

#### Option 1: active pointer/index view

Representation:

```text
objects[Bglobal]              replicated MFTL_Object shells, unchanged
active[{global_index, ptr}]   only objects with local owned or ghost entities
```

Benefits:

- low API risk;
- wire format and global arrays remain unchanged;
- local geometry/advection/normal/BVH-build loops skip empty bodies;
- redistribution can scan active mesh state while still setting votes by stable
  global index;
- useful performance counters become easy to add.

Limitations:

- memory remains `O(Bglobal)` per rank;
- global reductions remain `O(Bglobal)` per rank;
- empty `MFTL_Object` shells and any stranded pools remain resident;
- list-index identity remains embedded in communication and remeshing;
- BVH all-pairs and spatial-halo arrays remain global.

Decision: implement as a measured transition, then remove it at sparse-object
cutover. Do not describe it as the final weak-scaling solution.

#### Option 2: sparse mesh objects plus replicated compact metadata

Representation:

```text
local MFTL_Object directory                    O(Blocal)
replicated MFTL_BodyMetadata[Bglobal]          O(Bglobal)
```

Benefits:

- destroys locally empty mesh pools and object shells;
- local mesh kernels become `O(Blocal)`;
- preserves simple deterministic global remeshing and body enumeration;
- provides a practical compatibility bridge for applications with moderate body
  counts.

Limitations:

- not asymptotically flat weak scaling when `Bglobal` grows with `P`;
- per-body reductions and BVH candidate enumeration can remain global;
- list-index assumptions can survive invisibly in the metadata layer.

Decision: permitted only as an explicit transition phase. Keep body-GID APIs and
wire records even if a compact replicated metadata vector temporarily provides
ordering.

#### Option 3: sparse mesh objects plus a sharded body catalog -- target

Assign every body a stable metadata home:

```text
home(body_gid) = stable_hash(body_gid) % nproc
```

Each home stores compact metadata and the current holder/subscriber set. Local
mesh ranks store only active `MFTL_Object` instances and a GID-to-object map.

Benefits:

- mesh state and hot loops scale with `Blocal`;
- catalog state averages `O(Bglobal/P)` per rank;
- keyed partials can replace global per-body arrays;
- object arrival/departure becomes explicit and testable;
- body list order may differ safely between ranks.

Costs:

- global body enumeration becomes an explicit catalog query;
- metadata aggregation and publication need a sparse global transport;
- algorithms that require all bodies everywhere must be redesigned;
- holder membership, generations, deletion, and home failure diagnostics become
  persistent distributed state;
- the public exposed `MFTL_Ensemble` layout and direct-field users require a
  migration plan.

Decision: this is the final scalable representation. Use a hash home rather than
a spatial home initially: it is stable and load-balanced. A spatial home moves as
the body moves and couples catalog migration to geometry without evidence that
the locality benefit pays for that complexity.

#### Option 4: replicated catalog below a threshold, sharded above it

This hybrid can be attractive for small jobs but creates two metadata algorithms
and a collective mode switch. Do not implement it initially. A sharded catalog
may cache a read-only replicated snapshot for a bounded query without changing
the authoritative representation.

### Target local object data model

Introduce private structures before changing public semantics:

```text
MFTL_LocalObjectEntry
  body_gid
  MFTL_Object *object
  local state/generation
  catalog home

MFTL_LocalObjectDirectory
  dense local entry vector
  body_gid -> vector slot map

MFTL_BodyMetadata
  body_gid
  lifecycle generation
  topology/distribution/GID generations
  global vertex and triangle counts
  next-ID high-water marks while retained
  requested centroid/radius/weight/quality values and their epoch
  current holder ranks or holder-set routing state
  body-level parameters that callers currently assume are replicated
```

Use swap-with-tail removal in the local entry vector and update the directory.
No wire record stores that vector slot.

When records arrive for an absent body:

1. validate the body header and catalog generation;
2. create an empty local object attached to the ensemble domain;
3. reserve required pools;
4. install records through the common commit plan;
5. add the object directory entry only at commit.

When both pools become empty after commit:

1. invalidate object schedules and GID maps;
2. notify the metadata home that this rank is no longer a holder;
3. remove the directory entry;
4. destroy the object and release all pools.

The ensemble collective itself never returns early merely because the local
directory is empty.

### Keyed catalog aggregation

Replace replicated per-body reductions with a common primitive:

```text
local holders emit (body_gid, operator, payload, epoch)
        -> route to home(body_gid)
        -> home combines in deterministic rank/GID order
        -> publish to current holders or requesting algorithm
```

Operators initially needed:

- sum owned vertex/triangle counts;
- area-weighted centroid components;
- maximum radius;
- min/max/sum quality statistics;
- next-ID/generation metadata;
- active holder set;
- optional dirty/body-wide decisions during transition.

The generic layer owns transport and generation validation. Modules own payload
meaning. Do not create one ad hoc home exchange in remeshing, another in BVH, and
a third in ensemble statistics.

### Module impact and required redesign

#### Ensemble lifecycle and public API -- critical

Current `MFTL_Ensemble.objects` is public and documented as the same list in the
same order on every rank. Sparse objects change that contract.

Required work:

- add `MFTLEnsembleFindLocalObject(body_gid)`;
- add local-object iteration APIs/macros;
- add explicit local and global body-count queries;
- add private create-on-arrival and destroy-on-empty operations;
- distinguish local mesh object count from global catalog body count;
- decide the compatibility/ABI policy for direct access to `objects` and
  `num_objects`;
- make collective entry points legal on an empty local directory;
- replace `MFTLEnsembleUpdateGlobalCounts()` arrays with keyed catalog
  aggregation;
- give globally deleted bodies a lifecycle generation/tombstone.

Recommendation: add new private/local fields first while retaining the old
public list. Change public semantics only in a named cutover phase, with compiler-
visible API changes rather than silently changing what `num_objects` means.

#### Redistribution and exchange -- critical

Required work:

- change body-run and action headers from object index to body GID;
- group arrivals through the local GID directory;
- create an absent object from a validated body header;
- destroy locally empty objects after commit;
- key explicit owner maps by body GID/local object entry;
- remove same-list/same-order checks and replace them with generation and catalog
  checks;
- make zero-local-object ranks participate;
- make per-body operation plans sparse rather than `[Bglobal]`;
- integrate cached ghost schedule construction/maintenance;
- use compact neighbor target arrays and lazy global arrays.

This is why final sparse-object cutover should follow the unified body-GID record
protocol rather than be retrofitted deeply into the old list-index stream.

#### Remeshing -- critical and high difficulty

The parallel remesher currently uses arrays of `2 * num_objects` or more for:

- body measurements and strategy selection;
- per-phase created-entity GID ranges;
- GID watermarks and redensification;
- convergence counters combined with per-body counts;
- explicit owner maps;
- dispersed destination selection;
- ghost GID synchronization.

Required changes:

1. `MFTLRemeshMeasureBodies` emits keyed local partials to catalog homes. A
   constant-size global maximum can still decide SPANNING versus DISPERSED after
   homes compute body radii.
2. Spanning remeshing iterates only local active bodies. Per-body GID allocation
   becomes keyed by body GID; keep existing renumbering as an explicit generation
   break during the first migration.
3. `MFTLRemeshSyncGhostIds` uses body GIDs and later the cached copy schedule.
4. Explicit owner maps are a set of local `(body_gid, maps)` records, not an array
   indexed by the global list.
5. Dispersed remeshing's global greedy load balancer cannot merely iterate a
   local list: every rank currently obtains the same assignment from replicated
   centroids and weights. Use a replicated compact metadata view during the
   transition, then implement deterministic distributed ordering/assignment
   keyed by `(weight, body_gid)`.
6. Whole-body gathering may create a body on a previously inactive rank and must
   delete it from ranks that become empty.
7. Phase convergence remains a scalar global operation; do not expand it back
   into per-body replicated reductions.

Remeshing is a separate major work package inside the sparse-object project. The
spanning and dispersed strategies both need differential tests against their
replicated-list versions before cutover.

#### BVH and spatial halos -- critical and separate algorithmic issue

The current ensemble intersection path:

- builds one BVH for every globally listed body;
- enumerates every global body pair;
- requires identical pair requests on every rank;
- carries extent arrays proportional to `Bglobal` through neighbor collectives;
- traverses every pair locally.

An active pointer list reduces empty BVH construction but does not fix the
`O(Bglobal^2)` pair list or global extent payloads. Simply iterating local object
pairs after sparse cutover is incorrect: two bodies can need a cross-rank contact
test without both complete bodies already coexisting on one rank.

Required target design:

1. publish compact body bounds/radii keyed by body GID through a distributed
   spatial broad phase;
2. discover candidate body pairs spatially rather than enumerating all pairs;
3. assign each candidate pair a deterministic pair owner, for example from the
   ordered `(min_gid, max_gid)` key;
4. request only the patches required by candidate pairs;
5. exchange all keyed requests in a fixed number of batched operations whose
   sequence does not depend on a rank's local pair count;
6. store halo patches by body GID, not object-list slot;
7. replace `12 * Bglobal` radius/extents Allreduces and `6 * Bglobal * d`
   neighbor buffers with keyed home/broad-phase aggregation;
8. test candidate completeness across periodic seams and ranks with zero local
   bodies.

This is not required to land the first active-list or ghost-schedule change, but
it is required before the whole library can claim sparse many-body scaling.

#### Ensemble quality/statistics -- medium

`MFTLEnsembleUpdate()` currently reduces arrays for every globally listed body.
Local geometry computation naturally becomes local-object iteration. Global
quality values become keyed catalog reductions and are cached in body metadata
with an epoch. A local `MFTL_Object` may mirror the current catalog value for API
convenience, but it is not the authority.

#### I/O -- low to medium

The VTU writer already concatenates locally held pieces, writes `obj->global_id`
as `BodyID`, and supports collective zero-byte ranks. It should adapt relatively
cleanly when `num_objects` is explicitly local.

Required work:

- iterate local objects explicitly;
- preserve collective calls on zero-object ranks;
- distinguish local piece counts from global catalog body counts in metadata and
  diagnostics;
- ensure filenames/index files do not assume every rank lists every body;
- add tests where bodies exist on only a subset of ranks.

A future restart reader would need catalog reconstruction, but no reader exists
today.

#### Generation and initial distribution -- medium to high

Rank-zero generation currently creates every body before the opening scatter.
Sparse object identity allows non-root ranks to construct a body on arrival, but
does not by itself remove the serial generation peak.

Implement in two steps:

1. express the existing initial scatter as body-GID authority snapshots that
   create local objects and catalog entries;
2. later add a distributed generator/reader so rank zero never holds the entire
   global mesh.

Do not make distributed generation a prerequisite for the object-directory API,
but do not count sparse object shells as a fix for the generation peak.

#### Examples, diagnostics, and tests -- broad but mechanical after core APIs

Every loop over `ens->objects` must be classified as one of:

- local mesh iteration;
- global metadata iteration;
- collective participation independent of local count;
- an algorithm incorrectly relying on global replication.

Examples should print both local active bodies and global known bodies. Scaling
JSON should report:

- `Blocal` min/mean/max;
- catalog entries owned by each rank;
- holder memberships;
- empty local ranks;
- cached schedule entries and rebuilds;
- neighbor/global metadata bytes;
- route/backend counts;
- catalog bytes and messages.

## Ordered implementation plan

### Phase 0 -- freeze contracts and extend measurements

Work:

- record all current collectives by purpose, payload scaling, and call frequency;
- record workspace bytes split into local-entity, per-body, per-neighbor, and
  per-rank classes;
- add counters for active/global/empty bodies and stranded object pool bytes;
- add ghost request/answer/table/apply timers and schedule-size counters;
- add MPI-call counters to the scaling JSON;
- archive current advection, spanning/dispersed remeshing, BVH, and I/O baselines.

Exit gate:

- phase totals reconcile with end-to-end timers;
- every `O(P)` and `O(Bglobal)` allocation in the relevant modules is listed;
- the old request/reply field sync is a differential oracle.

Difficulty: low. Risk: low.

### Phase 1 -- transitional active-object view

Work:

- add `active_objects[]` entries containing pointer and stable global-list index;
- maintain it after add, redistribution, pool-count change, and object release;
- replace unquestionably local hot loops in examples, geometry orchestration,
  normals, ghost collection, and BVH construction;
- retain global-index loops where dirty votes, remeshing arrays, or identical pair
  lists still require them;
- instrument active-view rebuild time and memory.

Exit gate:

- bit-identical behavior;
- no local hot kernel visits empty objects;
- current global wire/list contracts remain intact;
- measurements show whether empty-object iteration was material.

Difficulty: low to moderate. Risk: low if global and local loops are named
explicitly.

### Phase 2 -- bounded-degree neighbor metadata

Work:

- change exchange input from `[P]` send-buffer arrays to compact target lists;
- replace `is_neighbor[P]` and `rank_to_neighbor[P]` with bounded lookups;
- persist `[d]` neighbor counts/displacements and byte buffers;
- allocate `[P]` state lazily only in the global Alltoallv backend;
- remove the exchange flattening copy if direct packing benchmarks confirm the
  expected byte reduction;
- preserve the automatic route Allreduce.

Exit gate:

- neighbor-path workspace is constant at fixed `d` as `P` grows;
- no steady allocation after warm-up;
- all exchange backends pass the same randomized payload tests.

Difficulty: moderate. Risk: moderate because it changes shared transport code,
but not mesh semantics.

### Phase 3 -- snapshot cached ghost schedule on the current core

Work:

- add schedule structures, invalidation, generations, fingerprints, and counters;
- generate full subscription snapshots during dirty-body redistribution;
- piggyback them in the redistribution exchange;
- freeze owner and holder slots after pool commit;
- implement one-phase position and normal pushes;
- retain request/reply behind a test/build switch and explicit first-use fallback;
- add schedule comparison and corruption-injection tests.

Exit gate:

- one field exchange per steady call;
- no steady request construction or owner GID scatter;
- exact agreement with the old path through ownership changes and remeshing.

Difficulty: moderate to high. Risk: high enough to require generations and exact
fixtures, because stale cached slots produce plausible wrong geometry.

### Phase 4 -- body-GID protocol and local directory substrate

Work:

- introduce the private local GID-to-object directory while the replicated list
  still exists;
- change new redistribution/action/schedule records to body GID;
- add versioned body headers and create-on-arrival plans;
- add destroy-on-empty primitives and holder notifications;
- key owner maps and ghost schedule entries by body GID;
- make collective entry points legal with zero local active objects;
- retain a replicated compact metadata/order view temporarily for remeshing and
  BVH compatibility.

Exit gate:

- ranks may hold different active-list orders without changing results;
- absent bodies can arrive, leave, and reappear;
- no new protocol uses global list position.

Difficulty: high. Risk: high. This is the identity migration, but not yet the
final removal of global metadata.

### Phase 5 -- common distributed catalog and keyed aggregation

Work:

- implement stable body homes and metadata generations;
- maintain holder membership on local object create/destroy;
- implement generic keyed reduce/publish operations;
- migrate global counts, centroid/radius/weight, and quality statistics;
- add catalog enumeration/query APIs and constant-size fingerprints;
- select and validate the sparse global transport used to reach arbitrary homes;
- retain a test-only exact gather for bounded fixtures.

Exit gate:

- catalog memory averages with `Bglobal/P`;
- local holders receive exact metadata values;
- list order and local absence do not affect aggregate results;
- stale, duplicate, and skipped generations fail diagnostically.

Difficulty: very high. Risk: very high because this introduces persistent
distributed authority.

### Phase 6 -- delta schedule maintenance and sparse redistribution cutover

Work:

- integrate copy-reference deltas with `Rold/Rnew` replica planning;
- transfer copy counts during vertex handoff;
- replace schedule snapshots with incremental updates in normal advection;
- make per-body plans sparse and eliminate the global dirty bitset;
- use catalog-home agreement only for any transitional body-wide operation;
- destroy empty local objects and remove the replicated `MFTL_Object` list;
- retain dense authority snapshots for initial scatter and high-churn remeshing.

Exit gate:

- unchanged membership changes no schedule entries;
- sparse migration work follows actual membership change;
- local mesh memory and hot loops scale with `Blocal`;
- global object shells are absent.

Difficulty: very high. Risk: very high. This depends on the broader delta engine
described in the final redistribution recommendation.

### Phase 7 -- remeshing migration

Work:

- migrate measurements, GID assignment, owner maps, and ghost-ID sync to GID keys;
- implement explicit generation breaks for redensification;
- migrate the spanning strategy first;
- retain compact replicated metadata temporarily for the dispersed assignment;
- replace dispersed global greedy assignment with a deterministic distributed
  algorithm;
- validate whole-body gathering and return-to-geometric-partition with absent
  object creation/destruction;
- keep scalar convergence/profile collectives.

Exit gate:

- both strategies match replicated-list results and invariants;
- no remeshing array is sized by a globally replicated object list;
- ranks cannot diverge in strategy, destination, GID, or convergence decisions.

Difficulty: very high. Risk: very high.

### Phase 8 -- spatial halo and BVH candidate-discovery redesign

Work:

- replace index-based spatial-halo requests with body-GID keys;
- build a distributed spatial broad phase for body bounds;
- generate complete candidate pairs without all-pairs enumeration;
- assign deterministic pair owners;
- batch patch requests and replies independently of local pair count;
- migrate global radii/extents to catalog/broad-phase aggregation;
- test periodic and cross-rank non-vacuity exhaustively.

Exit gate:

- candidate completeness matches an exact all-pairs oracle on bounded cases;
- production memory/work is not `O(Bglobal^2)` per rank;
- sparse local object lists do not lose cross-rank intersections.

Difficulty: very high. Risk: very high. This is algorithmically separate from
redistribution even though it uses the same identity/catalog substrate.

### Phase 9 -- I/O, examples, public cutover, and cleanup

Work:

- migrate VTU and diagnostics to explicit local iteration/global queries;
- update public headers and direct-structure contracts;
- update examples and documentation terminology;
- add an upgrade guide for `objects`/`num_objects` users;
- remove active-view/global-list compatibility state;
- remove the old ghost request/reply production path and obsolete list-index
  formats;
- run strict documentation and ABI/API checks.

Exit gate:

- one local-object representation, one body identity, one field schedule, and one
  redistribution protocol remain in production.

Difficulty: moderate mechanically, high compatibility risk.

## Validation matrix

### Correctness fixtures

Add fixtures for:

- zero local objects on one or many ranks;
- a body created on a rank by incoming redistribution;
- a body destroyed locally and later returning;
- different local object orders on every rank;
- one spanning body and thousands of dispersed bodies;
- vertex handoff with existing copy ranks;
- a dirty body whose copy set becomes empty on one owner;
- GID renumber followed by position and normal push;
- neighbor-only and forced-global routes;
- stale schedule generation, duplicate subscription, missing removal, and wrong
  cached slot;
- remeshing strategy and destination agreement;
- BVH contact whose two bodies initially live on different ranks;
- periodic-seam contact and spatial-halo completeness;
- shared I/O with ranks that own zero pieces.

### Existing tests to retain as gates

- `tests/test_redistribution.c`, including twin repair and skip equivalence;
- `tests/test_remeshing_parallel.c` ghost-owner agreement;
- `tests/test_remeshing_dispersed.c` strategy and balance fixtures;
- `tests/test_bvh_ghosting.c` at all supported MPI grid sizes;
- `tests/test_bvh_parallel.c`;
- `tests/test_io.c`;
- `tests/test_ensemble_parallel.c`;
- all mesh invariants and ASan/UBSan object tests;
- partition set and slot-order digests during differential migration.

### Performance gates

Ghost field update:

- one data exchange per field;
- zero schedule discovery on the steady path;
- no steady allocation;
- bytes proportional to ghost entries plus bounded headers.

Neighbor redistribution:

- metadata/workspace proportional to `d`, not `P`;
- no global-backend arrays allocated in neighbor-only runs;
- route Allreduce separately visible rather than hidden in another phase.

Sparse objects:

- mesh objects and local hot-loop iterations proportional to `Blocal`;
- catalog entries proportional to `Bglobal/P` within hash imbalance;
- no `O(Bglobal)` per-rank collective payload in the production sparse path;
- no empty mesh pools retained;
- zero-object ranks remain safe participants.

BVH/spatial halo:

- candidate work follows spatial candidates, not all global pairs;
- exact bounded oracle proves no missed pairs;
- extent traffic follows local/candidate bodies.

Run the gates both on the MacBook for deterministic differential testing and on
the target cluster for transport/scaling decisions. Instrumented and
uninstrumented timing builds are separate comparisons.

## Difficulty and delivery assessment

| Work package | Relative difficulty | Main risk |
|---|---|---|
| active object pointer view | 3/10 | accidentally using a local view where identical global order is still required |
| `O(d)` neighbor metadata | 4/10 | shared exchange regressions and zero-byte edge cases |
| cached ghost schedule snapshot | 6/10 | stale generations/slots causing silent wrong fields |
| body-GID wire protocol and local directory | 7/10 | identity and create/destroy lifecycle errors |
| sharded catalog/keyed reductions | 9/10 | persistent distributed-state disagreement |
| sparse redistribution cutover | 9/10 | pool mutation, replica membership, and schedule consistency |
| remeshing migration | 9/10 | deterministic strategy/GID/destination agreement |
| BVH/spatial broad phase | 9/10 | missed cross-rank collision candidates |
| public/API cleanup | 6/10 | downstream source and ABI compatibility |

The first three packages are incremental and independently valuable. The complete
sparse-object conversion is a major library architecture change, not a local
optimization. For one experienced contributor it is realistically a multi-month
project; roughly four to eight person-months is a reasonable planning range for
the complete communication, catalog, remeshing, BVH, validation, and cutover
work. That estimate is intentionally broad and excludes waiting for access to a
representative cluster or redesigns prompted by transport measurements.

The recommended release boundary is:

1. release active-object, bounded-neighbor metadata, and cached ghost push after
   their independent correctness/performance gates;
2. develop body-GID protocol, catalog, sparse redistribution, remeshing, and BVH
   behind a feature/build boundary with differential testing;
3. change the public object-list contract only when the whole library no longer
   relies on replicated ordering.

## Final answer to the three findings

1. A cached ghost schedule is feasible without weakening semantics. Build a
   generation-checked snapshot in the current redistribution first, then maintain
   it with copy-reference deltas. The steady field update becomes one push.
2. Some global synchronization is intentional and should remain. Make the
   neighbor data structures `O(d)`, global arrays lazy, and per-body agreement
   keyed. Remove the route Allreduce only under an asserted local contract or a
   measured universal sparse transport.
3. A local active pointer list is useful but transitional. The scalable target is
   a local body-GID directory plus a sharded metadata catalog. Reaching it affects
   redistribution, remeshing, BVH/spatial halos, global statistics, generation,
   I/O, examples, and public API semantics; remeshing and BVH are the most
   difficult dependent subsystems.

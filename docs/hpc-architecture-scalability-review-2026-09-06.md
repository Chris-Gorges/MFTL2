# MFTL: architecture, correctness, and scalability review

Date: 2026-09-06. Review of the working tree based on commit
`35b4ea98f53662050dfbc2e32ccfaff45cba9d80`, including the in-progress dispersed
remeshing changes. This supplements
[the earlier review](full-library-code-review-2026-09-06.md).
Only this report and standalone diagnostic sources were added; library sources
and the earlier review were not modified.

## Assessment

**Retain the core data structures.** The implicit half-edge indexing, unified
pool allocations, geometry separated from topology records, owned prefixes, and
sparse local body directory are a sound foundation for a CPU/MPI surface library.
Production readiness does not require replacing them.

However, I would not yet treat this version as a reliable general-purpose
distributed geometry component. Some communication optimizations assume more
than their checks establish. This review reproduces incorrect distributed sums
and prefixes on a valid connected closed surface, missed distributed contacts,
and a heap access outside its allocation with only two small triangles.
These take priority over improving benchmark timings.

The most useful architectural addition is explicit ownership of reusable
communication and geometry state: who holds each body/entity, which layouts a
schedule refers to, and when those facts change. That can be built around the
existing pools. It would improve correctness and enable several of the largest
scalability opportunities at the same time.

## 1. New correctness findings

### F1 — Critical: the holder-routing proof is false for a connected surface

Evidence: `src/ensemble/ensemble_reduce.c:156–222`, `:700–725`;
the proof in `include/mftl_ensemble.h:372–381`;
the prefix consumer in `src/remeshing/remesh_parallel.c:569–573`.

`MFTLEnsembleBodyHoldersFitTheGraph()` examines the owners visible through each
rank's ghost pools. It accepts when they are neighbors and their local coordinate
offsets span at most one cell per axis. The documentation concludes that
connectedness makes this sufficient to establish that **all holders** are mutual
graph neighbors. That implication is incorrect.

A connected tube can follow this path through a nonperiodic `3 × 2 × 1` grid:

```text
cell (0,0) -> cell (1,0) -> cell (1,1) -> cell (2,1)
rank    0          2             3             5
```

The two x crossings are separated by a y crossing. No intermediate holder needs
to see ghost owners on opposite sides along the same axis. Nevertheless, ranks
0 and 5 are not neighbors. A neighborhood allgather cannot deliver their
contributions to each other.

The [reproducer](review-reproducers-2026-09-06/bent_holder.c) constructs a capped,
connected tube, checks its topology, and calls the actual redistribution before
comparing AUTO reductions with GLOBAL reductions. Results from six ranks:

| Rank | Owned triangles | AUTO sum | Correct sum | AUTO prefix | Correct prefix |
|---|---:|---:|---:|---:|---:|
| 0 | 98 | 386 | 484 | 0 | 0 |
| 2 | 160 | 484 | 484 | 98 | 98 |
| 3 | 128 | 484 | 484 | 258 | 258 |
| 5 | 98 | 386 | 484 | 288 | 386 |

Every rank votes that the graph route is valid. Mesh construction and
redistribution return success. The mesh checker reports a consistent closed
surface with 244 vertices and 484 triangles. An open connected strip also fails.

The prefix discrepancy is more serious than inaccurate statistics: assigning
triangle IDs using these prefixes gives rank 3 `[258,386)` and rank 5
`[288,386)`, overlapping by 98 IDs. This reproduces the prefix error and its
range consequence; it does not depend on a particular remeshing sweep producing
the same mesh. The remesher uses this very prefix mechanism for identity.

Affected consumers also include global body counts, quality, body radii, and
redistribution's dirty-body vote. A missed dirty contribution can make different
holders disagree about whether to rebuild a body.

**Immediate containment:** use the global route for unproven holder sets, covering
both AUTO entry points and call sites that explicitly select HOLDERS after this
test. Fix F4's rank-zero handling at the same time. This is a small correctness
patch with a performance cost.

**Durable correction:** establish a complete holder set, or a sound certificate
of its communication reachability, when residency changes; cache it with explicit
invalidation. A conservative globally established holder-coordinate extent is
another starting point. Temporary remeshing owner maps and periodicity must be
covered. Do not derive the certificate using the reduction whose validity it is
supposed to prove. Connectedness alone does not repair the present test.

### F2 — High: spatial-grid sizing overflows before its cap is applied

Evidence: `src/communication/spatial_halo.c:907–966`.

`grid_build()` converts domain-width/search-width ratios to `int`, multiplies
the three dimensions into `long cells`, and only afterward shrinks the grid to
`4*n_items + 64` cells. The initial product can overflow even though the desired
final grid is tiny. The floating-to-integer conversion is also unchecked.

With two triangles approximately `1e-7` across in a unit domain, a single-rank
contact query reports:

```text
spatial_halo.c:924: signed integer overflow:
99900064980004 * 9995002 cannot be represented in type 'long'
AddressSanitizer: heap-buffer-overflow in grid_build, spatial_halo.c:966
```

This is a reachable memory-safety defect with six input vertices, not merely a
theoretical limit for billion-element meshes. It is reproduced by the `tiny`
mode of [contact_limits.c](review-reproducers-2026-09-06/contact_limits.c).

**Correction:** impose the item-based budget before converting or multiplying;
use checked arithmetic and ensure that the allocated product agrees with the
indexing dimensions. Reducing the cell count enlarges cells and preserves the
conservative broad phase. Effort: small, within the current implementation.

### F3 — High: contact detection silently misses pairs beyond one graph hop

Evidence: `src/communication/spatial_halo.c:1363–1379`, `:1467` onward;
`src/bvh/bvh_traverse.c:380–507`;
the absolute physical envelope supported by `MFTL_BVHBuildParams`.

The spatial halo publishes extents and sends triangles only over `comm_graph`.
There is no reach-based expansion or rejection when the requested contact
distance extends beyond that neighborhood.

Four-rank reproduction: domain `[0,4] × [0,1] × [0,1]`, two small triangles in
the planes `x=0.9` and `x=2.1`, absolute proximity floor `1.3`. Their separation
is `1.2`, so they should be reported as a proximity pair. Before redistribution,
with the meshes held together on rank 0, the collective query reports one pair.
After redistribution their owners are ranks 0 and 2, and the same query reports
zero pairs. Every API call returns success.

```text
ranks=4 before=1 after=0 rc_before=0 redistribute=0 rc_after=0
```

The normal mode of [contact_limits.c](review-reproducers-2026-09-06/contact_limits.c)
reproduces this. The issue becomes relevant to strong scaling because the
physical envelope can stay fixed while process cells shrink. Large triangles
also require accounting for primitive extent relative to their owning centroid.

**Correction:** define and enforce a conservative graph-reach condition using
primitive extent and the contact envelope. Reject unsupported reach collectively
as an initial patch. Subsequently support an expanded graph, multiple forwarding
rounds, or a suitable sparse fallback. The transport needs extension; the mesh
pools do not need replacement.

### F4 — High portability defect: rank-zero Exscan handling is wrong in I/O too

Evidence: `src/ensemble/ensemble_reduce.c:511–547` and
`src/io/vtu_writer.c:1016–1026`.

The earlier report correctly identifies the body-prefix problem, but incorrectly
describes the VTU writer as handling it correctly. `scan_placement()` also zeros
the result **before** `MPI_Iexscan` and does not restore rank zero's base after
completion. Its comment says that MPI leaves the buffer untouched.

MPI specifies rank zero's exclusive-scan output as undefined. Initialization
before the call is insufficient. Set rank zero's output explicitly after the
wait in both locations. The tested MPI happens to produce the expected values;
this finding follows from the implementation and the
[MPI exclusive-scan contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node143.htm).

### F5 — Large-output defect: VTU offsets have an earlier limit than point IDs

Evidence: `src/io/vtu_writer.c:1038–1045`, `:1188–1190`, `:1228–1229`.

The shared writer checks that the number of points fits Int32. Its cell offsets
are also Int32, however, and store `3 * cumulative_triangle_count`. Consequently
the offset limit is 715,827,882 triangles, even when all point IDs fit. At the
next triangle, the offset is no longer representable. Distributed bodies can
collectively reach this limit while every individual rank's buffers remain
below the separate MPI byte-count limit.

This is established from the layout arithmetic; no giant allocation was made.
Add a collective preflight check or emit Int64 connectivity/offset arrays.
Do this before filling arrays and before subsequent collective I/O.

## 2. Findings from the earlier review that remain relevant

Five standalone reproducers from the earlier review were rebuilt against the
fresh Release library and rerun:

| Finding | Observed result | Implication |
|---|---|---|
| Absolute geometry thresholds | Equilateral side `1e-7`: quality `0`, vertex normal `(0,0,0)` | Physical unit choices change dimensionless quality and normal validity |
| Non-manifold input | Three faces sharing an edge accepted; integrity returns success | The public checker does not establish manifoldness |
| Stale remesh normals | Maximum component difference after recomputation `0.19900926206771047` | Returned normal data violates the documented postcondition |
| BVH translation sensitivity | Translating a unit triangle to `1e9` changes fallback radius from about `0.5` to `256` | Contact tolerance and traversal efficiency depend on coordinate origin |
| Degenerate triangle distance | Squared distance `3`, correct value `2` | Degenerate primitive handling is incomplete |

For the translation case, ensemble global-radius hints repair the tolerance
calculation but do not repair the loose absolute-coordinate float AABBs.

Additional earlier findings were checked in source:

- Ghost synchronization counts orphaned non-neighbor ghosts but returns status
  from `unserved` alone (`ghost_sync.c:458`). Its neighbor-index allocation also
  still scales with process count (`:209–211`). A successful return cannot be
  relied on to mean all requested ghost fields are current.
- `MFTLRemeshParamsAgree()` compares only two lengths (`remesh_comm.c:335–351`),
  and entry-point checks are assertions. Strategy, phase budget, and other
  collective control parameters require an agreed Release contract. Define
  identical collective arguments as a public requirement and validate the
  dangerous controls before branching, preferably once per parameter revision.
- `redensify_global_ids()` rewrites owned identity after a failed ghost-ID sync,
  and its caller continues into redistribution (`remesh_parallel.c:691–738`,
  `:1698–1712`). Failure needs an agreed stop or a defined restoration path before
  another state-changing collective.
- Shared I/O returns locally for oversized blocks before `MPI_File_open`
  (`vtu_writer.c:1064–1067`). A rank-local validation failure must be agreed before
  peers enter the next collective.
- Serial remeshing never performs the integrity check advertised by its return
  contract (`remesh_driver.c:315–343`). Its final smoothing pass also changes
  positions after computing normals.
- Domain bounds, size arithmetic, and several public memory-safety preconditions
  need Release validation. Unchecked `3*T`, ID increments, growth, and MPI
  displacement accumulation should be treated as one library-wide audit.

The distinction between an exact orientation predicate and an entirely robust
contact implementation should also be explicit: the coplanar path uses a plain
2-D determinant (`bvh_predicates.c:1017–1020`). Its own comment acknowledges this.
Adaptive `orient3d` alone does not establish exactness of all branches, and a
proximity fallback cannot undo a false-positive intersection accepted early.
This is a robustness gap identified in source, not a new failing pair reproduced
in this review.

## 3. Scalability improvements that fit the current design

Effort below is relative: **small** means a localized change with focused tests;
**medium** means a module/API extension; **large** means coordinated work across
modules. None of this table requires replacing the half-edge representation.
Expected benefits are structural hypotheses to measure, not claimed speedups.

| Priority | Improvement | Effort | Why it matters / condition |
|---|---|---|---|
| 1 | Reuse exact owner-to-holder ghost schedules | Medium | Stable partitions currently repeat count exchange, requests, ID lookup, and allocation for every field update |
| 2 | Cache valid holder membership and reduction routes | Medium–large | Eliminates repeated discovery and enables correct sparse reductions without the F1 assumption |
| 3 | Remove `O(P)` bookkeeping from normal neighbor paths | Small–medium | Local work should follow active peers, not all ranks in the job |
| 4 | Reuse exchange, reduction, and remeshing scratch | Small–medium | Existing persistent redistribution storage stops short of several heavily repeated allocation paths |
| 5 | Preserve BVHs and refit between structural changes | Medium | The ensemble contact pipeline builds, sometimes rebuilds, refits, and destroys acceleration state every call |
| 6 | Avoid duplicate radius publication/refits in the ensemble contact pipeline | Small–medium | Radius information is published separately and then included in the nine-double body records; repeated traversals also occur during radius/halo setup |
| 7 | Dispatch/remesh subsets of bodies by need and regime | Medium–large | One large or troubled body currently affects the strategy/cost paid for unrelated small bodies |
| 8 | Make the body broad phase tolerate a wide size distribution | Medium | A uniform grid sized by the largest body can put many separated small bodies into the same bucket |
| 9 | Generate and register initial bodies without a rank-zero bottleneck | Medium | Batched publication reduces latency but generation and transient replicated registration still limit startup |
| 10 | Introduce measured computation/communication overlap | Medium | Begin/end APIs allow useful interior work while a stable halo exchange runs |

### Exact cached ghost exchange

Keep the existing request/reply mechanism as schedule construction and as a
reference implementation. Store the resulting send/receive slot lists in an
ensemble-owned context. While valid, pack only current values, communicate the
known lengths, and scatter directly to known ghost slots. Persistent MPI
requests are optional; the largest change is reusing knowledge of the recipients.

Invalidate on redistribution that changes residency, topology edits, pool
reordering/compaction that changes indices, body creation/destruction, and any
address changes relevant to persistent requests. A dirty boolean that is later
cleared is insufficient to identify which layout a cached schedule describes.
Use monotonically advancing versions and central mutation hooks. Commit a
schedule only after construction succeeds everywhere it is required.

Positions and normals can share this schedule while remaining separate exchanges:
normal computation depends on current positions. Do not force their payloads
into the same phase. For steady communication, three doubles per vector suffice;
IDs are necessary during discovery, not on every update.

This develops the existing
[ghost-sync architecture proposal](ghost-sync-architecture-analysis.md), whose
exact cached schedule is a better long-term target than broadcasting all
replicated vertices to every neighbor. The latter pays for unused recipients.

### Truly sparse per-body reductions

Even after F1 is repaired, AUTO currently makes a **single ensemble-wide route
decision**. One body that does not fit forces dense global staging for every
body (`ensemble_reduce.c:700–725`). Under weak scaling, a fixed local droplet
population can therefore acquire an `O(B_global)` memory and bandwidth term
because of one large interface elsewhere.

Use certified shared-body membership to send each contribution only where it is
needed. Whole bodies held on one rank need no reduction payload. For arbitrary
holder sets, an amortized sparse coordinator/holder protocol can reduce and
redistribute records while batching many bodies together. Avoid creating one
MPI communicator per droplet as the default answer for millions of droplets.
This is substantial communication work, but the object pools and operators
remain usable.

### Workspaces and neighbor indexing

The redistribution workspace retains rank-indexed send tables and rank maps
(`redistribute.c:1100–1127`). Ghost sync allocates/initializes an `nproc` inverse
map on every call. Spatial halo allocates two `nproc` send tables. Generic
exchange allocates count arrays and flattened buffers each time
(`exchange.c:118–199`), and the holder reduction repeatedly allocates staging.

Use neighbor-slot indexing on the common route and separately allocated or
sparse destination structures for exceptional global migration. Retain the
ability to migrate farther than one cell; it is part of redistribution's
correctness. Expand high-water-mark workspaces rather than just moving mallocs
from one function to another. Include peak retained storage in measurements.

### Contact acceleration

`MFTLEnsembleIntersectObjects()` first builds local trees, refits after setting
radii, rebuilds trees receiving patches, refits again, and then frees everything.
Some bodies thus pay multiple geometry/tree passes for one query. Start with
building once with final parameters and patch membership. Then persist local
trees and refit as positions change, rebuilding when deformation degrades tree
quality or primitive membership changes.

A separate local tree and patch tree can avoid rebuilding the local hierarchy
when remote membership changes; it requires extra traversal combinations and
duplicate-ownership tests, so benchmark it. Cache broad-phase candidates only
with a conservative motion bound and invalidation on remeshing; stale candidates
without a bound would miss contacts.

The uniform-grid broad phase is output-sensitive, not unconditionally linear.
If one body forces very large bins, small-small candidate enumeration can become
quadratic even when those small bodies do not overlap. Size classes, a separate
large-body list, or a hierarchy over body bounds can address this without changing
triangle storage.

### Scheduling and load balance

The chooser uses a global maximum radius
(`remesh_dispersed.c:228–260`). The dispersed driver also falls back for the
whole ensemble when any body is too large. A batch of small droplets plus one
large interface is therefore a distinct workload that needs its own benchmark.

Introduce body-selection masks and run the existing strategies on agreed subsets
in a fixed collective sequence. Skip bodies whose measured quality and motion
do not require remeshing. Price migration and restoration as part of the decision;
offloading a small body can cost more than processing it where it already lives.

The current uncommitted dispersed balancer uses neighborhood advertisements,
grants, and verdicts instead of the older replicated global body/load tables.
That is a useful scalability improvement already present in this working tree.
Its limitation is locality: one round can use nearby spare capacity, but cannot
balance a concentrated cloud across arbitrarily distant idle ranks. Coarse-level
balancing or several carefully bounded rounds could extend it. Load weights
should eventually include observed remeshing work and memory growth, since
equal triangle counts need not have equal split/collapse/smoothing cost.

### Overlap and measurement

A nonblocking MPI call is an opportunity for overlap, not evidence that the
network progressed while the CPU computed. Measure initiation, useful work,
and exposed wait time on the target MPI implementation. The
[MPI nonblocking-collective specification](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node145.htm)
allows implementation-dependent progress behavior. Begin/end halo APIs can
expose useful overlap across all bodies without introducing a collective per
body. Keep any MPI polling or progress-thread policy configurable and measured.

The byte counters also need precise semantics. For example,
`holder_gather()` adds `scount + total` to a counter documented as sent plus
received bytes (`ensemble_reduce.c:275`), but its send block goes to every
out-neighbor. Datatype extent is likewise a storage stride, not necessarily the
MPI payload size: the ghost reply occupies 32 bytes in C but its registered type
contains three doubles and one int. Distinguish packed storage, logical MPI
payload, estimated network volume, and allocation churn.

## 4. Production engineering within the existing architecture

These are at least as valuable as faster kernels:

1. **Define a supported input envelope.** Finite coordinates and domain bounds,
   orientable manifold topology, boundary rules, supported valence, periodic
   representation constraints, count limits, and communication reach. Validate
   once at ingestion and where an operation can invalidate a condition. Internal
   assertions should complement those checks.
2. **Make state validity explicit.** Topology, layout, ghost fields, geometry,
   and quality have different lifetimes. A caller writing positions needs a
   clear update protocol; that does not require maintaining another copy of
   positions. A stable body ID remains useful while an `MFTL_Object *` can be
   invalidated when redistribution removes its local body.
3. **Define the failure model.** Distinguish invalid arguments, unsupported
   geometry, nonconvergence, and MPI/runtime failure. `MPI_ERRORS_RETURN` alone
   does not make continued execution safe. Local detection must lead to agreed
   decisions before further collective mutation. An ID-coherence failure cannot
   be treated like an ordinary quality shortfall.
4. **Add a solver-field lifecycle.** A physics solver may attach velocity,
   surfactant, temperature, or history to vertices/triangles. Reorder,
   redistribution, split, collapse, and renumbering need usable maps/events or a
   field-registration mechanism. Returning new geometry without a transfer
   contract makes conservative coupling difficult. Start with permutation and
   migration maps, then operator-specific interpolation/conservative transfer.
5. **Separate mesh quality from physical fidelity.** Tangential smoothing and
   midpoint collapse do not impose exact volume conservation or a bound on
   distance to the pre-remesh surface. Add per-body signed-volume and geometry
   error budgets; investigate constrained operations and conservative remapping.
   A better-looking mesh can still be a worse interface for the solver. Treat
   intended physical deformation separately from remeshing-only error.
6. **Support return communication for contact forces.** A contact callback may
   refer to a remote B-side primitive. If consumers accumulate forces there,
   they need an owner-directed accumulation/reduction path and an explicit
   exactly-once contract. This can be a complementary field-exchange API.
7. **Make long jobs restartable.** Visualization output is not a checkpoint
   contract. Provide, or explicitly delegate to the host solver, versioned
   serialization of authoritative geometry, identity, fields, and relevant
   parameters with restart at a different process count.
8. **Make the library easy to embed.** Add install/export CMake targets, package
   versioning, a small external-consumer build test, configurable diagnostics,
   and a documented thread/MPI lifecycle. Process-global mutable statistics,
   datatype registration, and lazy predicate initialization need attention
   before concurrent calls are supported.

Optional geometry fields are a reasonable later memory improvement. The three
currently unproduced scalar vertex fields cost 24 bytes per vertex, but that is
not 27% of the entire mesh: a closed triangular surface has approximately
`V = T/2`, and current nominal pools cost about `88*V + 140*T`, or 184 bytes per
triangle before capacity slack and ghosts. Those fields account for roughly
6.5% of that total. Ghost amplification, scratch, minimum pool capacity for tiny
bodies, and temporary migration peaks deserve measurement alongside them.

For numerical reproducibility, require decomposition-independent ownership and
correct results first. Separate that from bitwise identity across different MPI
reduction trees, architectures, or remeshing orders. Document which operations
promise which property.

## 5. Larger changes, and the actual architectural limits

| Change | Classification | Recommendation |
|---|---|---|
| 64-bit global counts, IDs, and large MPI messages | Large cross-cutting API/wire-format change; no full mesh rewrite | Start with checked arithmetic and 64-bit aggregate counts; plan IDs separately from 32-bit local indices |
| Sparse directory/reductions for arbitrary body holder sets | Communication-subsystem extension | Worth doing when mixed or domain-spanning bodies are a target; preserve pools |
| Hybrid MPI plus threads over independent bodies/geometry | Medium–large | Plausible after removing shared mutable initialization/statistics hazards; do not assume arbitrary half-edge edits can be threaded |
| Persistent mesh ownership decoupled from Cartesian CFD ownership | Major architectural extension | Can improve clustered-interface load balance, but introduces solver/mesh routing and changes ownership assumptions across modules |
| Concurrent remeshing across partition boundaries | Major algorithm/protocol replacement | Needed only if shifted/frozen bands remain the dominant limit after simpler work; requires conflict ownership and cross-rank edit agreement |
| GPU-resident topology editing or distributed asynchronous mesh tasks | Major redesign | A different long-term execution model; not the next production-readiness step |
| Breakup/coalescence with changing body identity and conservative fields | Major functional extension | Existing local pools remain useful, but distributed connectivity, identity, and transfer semantics need substantial new machinery |

The spanning algorithm has a real strong-scaling limit. As process cells shrink
relative to mesh edges and the frozen band, fewer entities are eligible for an
independent local operation. Coarsening restores operable regions by reducing
the effective parallelism. This is a reasonable fallback, but cannot deliver
unbounded strong scaling. Report active remeshing ranks, frozen share, hidden
edges, and residual quality alongside time.

The dispersed algorithm has the complementary limit: whole-body work is
indivisible. Useful sweep parallelism is bounded by the number and cost
distribution of bodies, and one large body sets a lower bound on completion
time. A neighborhood load balancer cannot remove that bound.

These are reasons to select and schedule the two algorithms carefully, not
reasons to discard the current data structures.

## 6. Verification and the next development sequence

Verification in this review:

- Fresh AppleClang Release build, with examples and tests, completed.
- The sequential CTest run passed **48/48** tests, including existing MPI cases
  up to 27 ranks. These are single-machine correctness runs, not cluster scaling
  measurements.
- Independent six-rank connected strip and closed-tube reproductions failed the
  AUTO-versus-GLOBAL reduction check. The closed tube also demonstrates incorrect
  exclusive prefixes.
- The four-rank proximity reproduction changed from one contact to zero after
  redistribution, with success statuses throughout.
- A separate Debug library with ASan/UBSan reproduced grid overflow and a heap
  access outside its allocation with two small triangles on one rank.
- Five earlier focused reproducers were rebuilt and confirmed as listed above.
- F4, F5, the coplanar robustness gap, and performance opportunities are
  source-derived findings; no large-output or multi-node performance experiment
  is claimed.

The earlier report's parallel I/O test collision and test-parser alignment issues
were inspected but not rerun as failure experiments here. Isolate output paths
and repair byte decoding before treating parallel sanitizer CTest runs as a
reliable gate.

Recommended order:

1. Contain F1, repair both rank-zero Exscan paths, and fix F2. Establish regressions
   from the supplied reproducers before changing the implementations.
2. Enforce ghost/contact reach and collective error/control contracts. Harden
   mesh ingestion, normalize scale tests, and make remesh postconditions true.
3. Introduce versions and persistent workspaces, then exact cached ghost
   schedules. Measure their complete cost including invalidation/reconstruction.
4. Correct sparse holder certification and reductions, remove hot `O(P)` metadata,
   reuse contact acceleration, and test mixed-size ensembles explicitly.
5. Add field transfer, conservation/error budgets, checkpoint integration, and
   package/CI support. Introduce per-body remeshing selection and mixed strategies
   when workload measurements justify them.

For scaling acceptance, use weak scaling with fixed local mesh/body work and
strong scaling separately. Include a connected bent filament, one large body
among many small ones, a clustered cloud, periodic seams, empty ranks, and
contact distances approaching/exceeding cell width. Measure maximum-rank wall
time, useful sweep time, exposed MPI wait, peak memory, ghost/owned ratio,
candidate/narrow-phase counts, route fallbacks, bytes, and physical error.
Repeat on at least two MPI implementations and on a real multi-node system.
Oversubscribed laptop runs establish communication coverage, not efficiency.

### Running the new reproducers

They are diagnostic programs, not library changes. The first two MPI runs below
are expected to return nonzero on the reviewed version; the sanitizer run aborts
on the reproduced memory error. Use a clean CMake build directory of your choice.

```sh
mpicc -O2 -DNDEBUG -Iinclude docs/review-reproducers-2026-09-06/bent_holder.c \
  /path/to/release/lib/libMFTL.a -lm -o /tmp/mftl_bent_holder
mpirun -n 6 /tmp/mftl_bent_holder closed

mpicc -O2 -DNDEBUG -Iinclude docs/review-reproducers-2026-09-06/contact_limits.c \
  /path/to/release/lib/libMFTL.a -lm -o /tmp/mftl_contact_limits
mpirun -n 4 /tmp/mftl_contact_limits

mpicc -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude \
  docs/review-reproducers-2026-09-06/contact_limits.c \
  /path/to/sanitized-debug/lib/libMFTL.a -lm -o /tmp/mftl_contact_limits_san
mpirun -n 1 /tmp/mftl_contact_limits_san tiny
```

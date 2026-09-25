# Full Library Code Review

**Date:** 2026-09-06  
**Review perspective:** Distinguished HPC Performance/Software Engineering  
**Scope:** Current working tree, including the core data model, topology, geometry,
MPI communication, redistribution, serial and distributed remeshing, spatial halo,
BVH construction/traversal/predicates, I/O, build configuration, and tests.

## Executive assessment

The library has a strong core design, especially its compact half-edge
representation, hybrid AoS/SoA pools, sparse ensemble directory, vertex-complete
halo, neighbor/global redistribution routing, and measured parallel-remeshing
strategies.

The current checkout is not yet production-safe for arbitrary physical scales,
malformed input, large MPI jobs, or threaded callers. The review found several
confirmed correctness defects, including ones capable of corrupting distributed
global IDs, returning stale geometry as current, accepting non-manifold meshes,
and making contact tolerances depend on the mesh's absolute coordinate origin.

No source changes were made as part of the review. Focused standalone reproducers
were built outside the repository under `/tmp`.

## 1. Architecture and data flow

| Layer | Current model | Assessment |
|---|---|---|
| Entities | 16-byte vertex and half-edge records; 12-byte triangle records | Compact and cache-conscious |
| Geometry | SoA arrays cut from two cache-line-aligned allocations | Good streaming layout, but carries unused fields |
| Topology | Half-edges `3*t+k`; origins derived from `prev(h)` | Elegant, but places a high burden on validation |
| Ownership | Owned prefix followed by ghost suffix | Efficient and consistently exploited |
| Ensemble | Sparse, sorted local body directory with dense global body IDs | Strong improvement over replicated empty objects |
| MPI topology | Duplicated world communicator, Cartesian grid, up to 26-neighbor graph | Sensible locality model |
| Redistribution | Owner classification, one-layer vertex-complete halo, neighbor/global exchange, pool rebuild | Well structured; persistent workspace is valuable |
| Parallel remeshing | Shifted-plane spanning mode plus whole-body dispersed mode | Sophisticated, but collective control and failure handling need strengthening |
| BVH | Morton LBVH, float conservative AABBs, exact orientation predicates | Good overall structure; origin sensitivity is serious |
| I/O | Per-rank VTU or collective MPI-IO | Reasonable design, with a large-message collective deadlock path |

The principal distributed data flow is:

```text
authoritative vertex positions + half-edge topology
                    |
                    v
       triangle geometry / owner classification
                    |
                    v
   local retention + destination-specific packing
                    |
                    v
 neighbor exchange when reachable, global exchange otherwise
                    |
                    v
 deduplication + owned/ghost pool rebuild + topology repair
```

The spanning remesher repeatedly changes the ownership planes, freezes shared
entities, runs the serial operators on safe interiors, collectively allocates
new global IDs, and finally restores the canonical decomposition. The dispersed
remesher instead migrates each sufficiently small body to one rank, performs a
serial remesh there, and redistributes the result.

## 2. High-severity correctness findings

### 2.1 Rank 0 reads undefined `MPI_Exscan` output

**Location:** `src/ensemble/ensemble_reduce.c:511-547`

The global body-prefix path initializes `pre` with `calloc`, calls
`MPI_Iexscan`, and subsequently reads `pre` on every rank. MPI leaves the
exclusive-scan receive buffer undefined on rank 0. The fact that the allocation
was initially zero is not a valid guarantee after the MPI call.

These prefixes are used during parallel remeshing to allocate global-ID ranges.
An MPI implementation that modifies rank 0's receive buffer can therefore
produce overlapping vertex or triangle IDs.

**Required correction:** Explicitly zero rank 0's `pre[0:block]` after
`MPI_Waitall`, or exclude rank 0 from copying the undefined result. The VTU
writer already recognizes this requirement in its own Exscan path.

### 2.2 Geometry and remeshing are not scale invariant

Several kernels use absolute thresholds for quantities whose physical dimension
changes with mesh scale:

- `src/geometry/vertex_geometry.c:148-150` rejects a vertex normal when its
  squared accumulated area vector is at most `1e-12`. The area vector scales as
  `L^2`, so this tested quantity scales as `L^4`.
- `src/ensemble/ensemble_update.c:151-153` sets mean-ratio quality to zero when
  the sum of squared edge lengths is at most `1e-12`, even though the resulting
  quality is dimensionless and should be scale invariant.
- `src/remeshing/remesh_internal.h:289-294` rejects squared area vectors below
  `1e-15` in collapse/flip fold tests. This is another `L^4` threshold.
- Related absolute thresholds occur in feature classification and smoothing at
  `src/remeshing/remesh_smooth.c:296`, `:391`, `:491`, and `:577`.

A focused optimized-build test with an equilateral triangle of side `1e-7`
produced:

```text
q=0 area=4.3301270189221933e-15 vnormal=(0 0 0)
```

The exact quality is 1 and the vertex normal is nonzero. This behavior also
contradicts the warning against clamped normalization in
`include/mftl_constants.h:121-126` and the geometry design discussion in
`src/geometry/README.md:74-99`.

**Required correction:** Use zero or relative/dimensionless degeneracy tests as
appropriate. Normalize with a robust scaled norm or `hypot`-style computation
where the supported coordinate range requires protection from overflow and
underflow. Add metamorphic tests that uniformly rescale identical meshes.

### 2.3 Invalid and non-manifold topology is accepted in Release

**Locations:**

- `src/mesh_creation/mesh_builder.c:71-73`
- `src/topology/topology.c:76-95`
- `src/topology/topology.c:192-232`

Repeated triangle vertices are rejected only by a debug assertion. The Release
runtime check verifies index bounds but not `origin == target`. A resulting
self-edge can then twin with itself.

For a non-manifold edge with three incident faces, two half-edges are paired and
the surplus is silently classified as a normal boundary. The integrity checker
treats every missing twin as a genuine boundary and consequently cannot
distinguish this case from an open manifold surface.

A focused non-manifold input produced:

```text
[MeshCheck] Mesh is consistent and open (5 vertices, 3 triangles, 7 boundary half-edges).
nonmanifold rc=0 integrity=0
```

This also contradicts the comment in `topology.c` that says the checker can
report surplus half-edges.

**Required correction:** Reject repeated vertices, self-twins, duplicate
directed edges, duplicate faces, and undirected edges with more than two
incident faces. Validation must execute in Release when the public API promises
an error return.

### 2.4 Serial remeshing violates its documented postconditions

**Locations:**

- Public contract: `include/mftl_remeshing.h:1416-1433`
- Smoother: `src/remeshing/remesh_smooth.c:514-592`
- Driver: `src/remeshing/remesh_driver.c:315-343`

The public API promises current vertex normals and `MFTL_ERROR` if the mesh fails
its integrity check. In the implementation:

- Smoothing computes normals at the beginning of each sweep and then changes
  positions, with no final normal recomputation.
- With zero smoothing sweeps, normals might not be computed at all.
- The driver rebuilds derived connectivity and triangle geometry but never calls
  `MFTLCheckMeshIntegrity`.

On an irregular tetrahedron, returned vertex normals differed from an immediate
fresh recomputation by a maximum component delta of:

```text
0.19900926206771047
```

**Required correction:** Recompute vertex normals after the final position
update, or weaken the contract explicitly. If integrity failure remains an
advertised return path, perform the check and propagate its result.

### 2.5 Ghost synchronization can report success with stale ghosts

**Location:** `src/communication/ghost_sync.c:213-260,458`

Ghosts whose owner lies outside the graph neighborhood increment `orphaned` and
are deliberately skipped. The final return checks only `unserved`, not
`orphaned`. Release builds can therefore return `MFTL_SUCCESS` while ghost
positions or normals remain stale.

**Required correction:** Include `orphaned` in the error verdict or implement
the global fallback already used by remesh global-ID synchronization. If a
fallback is too expensive for every timestep, cache the schedule and only use
the global route when the collectively agreed topology requires it.

### 2.6 Collective control parameters are not fully agreed across ranks

**Locations:**

- `src/remeshing/remesh_comm.c:335-351`
- `src/remeshing/remesh_driver.c:355-388`
- `src/remeshing/remesh_parallel.c:1284-1307`
- `src/remeshing/remesh_dispersed.c:789-802`

`MFTLRemeshParamsAgree` compares only `min_edge_length` and
`max_edge_length`, and it is invoked only through debug assertions. Collective
structure and ownership decisions also depend on `parallel_strategy`,
`parallel_phase_budget`, coarsening/gather controls, iteration count, smoothing,
and feature parameters.

Different `parallel_strategy` values route ranks immediately into different
collective algorithms. Different phase budgets can cause different numbers of
collectives. Different coarsening inputs can make ranks compute different owner
maps.

`MFTLWriteVtu` has the same class of problem: different layouts or enabled
arrays lead through different collective sequences, while the public API does
not explicitly require or validate identical options and filename prefixes.

**Required correction:** Validate every control value that affects collective
sequence, message layout, ownership, or loop count. Perform one collective
agreement before entering the algorithm and return the same verdict everywhere.

### 2.7 Collective MPI-IO has a rank-local early-return deadlock

**Location:** `src/io/vtu_writer.c:1038-1088`

The shared writer rejects a local data block larger than `INT_MAX` and returns
before the collective `MPI_File_open`. Local block sizes can differ by rank, so
an imbalanced job can have one rank return while the remaining ranks enter the
collective file operation and hang.

The path-length check before `MPI_File_open` has the same collective-safety
requirement if ranks are allowed to receive different prefixes.

**Required correction:** Compute all local preflight errors, reduce them over
the communicator, and return collectively before the first subsequent
collective. Large blocks can alternatively be emitted through chunked
collective writes or MPI large-count APIs.

### 2.8 BVH behavior depends on absolute translation

**Locations:**

- `src/bvh/bvh_internal.h:41-67`
- `src/bvh/bvh_build.c:247-295`

Float AABB expansion uses relative slack proportional to the absolute coordinate.
The standalone BVH fallback radius is subsequently measured from the expanded
float root box. As the coordinate origin grows, float spacing and the explicit
relative slack can dwarf the physical body.

The same unit triangle produced:

```text
x=0          radius=0.500000119 tolerance=0.000500000119
x=1000000000 radius=256         tolerance=0.256
```

Translation alone inflated the physical contact threshold by 512 times. The
ensemble pipeline's global radius hint repairs the threshold, but its broad-phase
boxes remain hundreds of units wide and can cause a large increase in false
positive traversal and narrow-phase work.

**Required correction:** Store float boxes relative to a body/local origin, or
use double AABBs. Compute the physical bounding radius directly from double
geometry before conservative float conversion.

### 2.9 Degenerate triangles produce incorrect BVH distances

**Locations:**

- `src/bvh/bvh_predicates.c:1208-1285`
- Public contract: `include/mftl_bvh.h:680-710`

The parallel/degenerate branch in the segment-segment distance routine pins the
first segment parameter rather than explicitly handling point-segment and
point-point cases. A triangle against a repeated-point triangle returned squared
distance `3`; the correct answer was `2`.

The public API describes closed triangles and does not state that triangles must
be non-degenerate.

**Required correction:** Handle zero-length segments explicitly before the
parallel-segment branch, and add point/segment/collinear degenerate triangle
tests to both intersection and distance APIs.

### 2.10 Parallel-remesh failure handling continues with incoherent IDs

**Locations:**

- `src/remeshing/remesh_parallel.c:691-730`
- `src/remeshing/remesh_parallel.c:1689-1712`
- `src/remeshing/remesh_comm.c:972-978`

`redensify_global_ids` can fail to update some ghost IDs, but still overwrites
owned IDs and global counters. Its caller records the error and then proceeds
into the closing redistribution. The implementation's own comments state that
stale IDs can be duplicated during that redistribution.

**Required correction:** Agree failure status collectively before the next
state-changing collective. Define a recoverable rollback/restoration path, or
make an ID-coherence failure a collective fail-stop condition rather than
continuing with a known-invalid identity space.

## 3. Scaling and architectural findings

### 3.1 Per-rank `O(P)` metadata remains on neighborhood paths

Redistribution permanently allocates several arrays of length `nproc`, including
two ID-buffer tables, three send-buffer tables, and multiple rank maps:
`src/communication/redistribute.c:1100-1127`.

The spatial halo allocates two more `nproc` send tables on every call:
`src/communication/spatial_halo.c:1467-1468`.

This overhead is independent of how few graph neighbors or destinations are
active. It should be indexed by graph-neighbor slot on the normal path, with a
separate sparse or temporary global-destination mapping only when the global
route is collectively selected.

### 3.2 Contact detection discards reusable acceleration state

**Location:** `src/bvh/bvh_traverse.c:380-507`

Every ensemble contact call builds all local BVHs, constructs a spatial halo,
builds BVHs for remote patches, and destroys all of them before returning. This
bypasses the public refit capability for the local hierarchies.

Introduce a topology epoch on each object and persist its local BVH. Refit while
the primitive set is unchanged and rebuild only after remeshing, redistribution,
or excessive deformation. Remote patch BVHs may still require rebuilds when the
halo payload changes.

### 3.3 Body reductions allocate repeatedly on hot paths

The holder and global reduction routes repeatedly allocate send, receive,
count, displacement, grouping, and staging arrays. These reductions occur
multiple times during geometry updates, contact resolution, and remeshing.

Move this storage into an ensemble-owned reduction workspace, analogous to the
persistent redistribution workspace, and grow it only when observed high-water
marks increase.

### 3.4 Global body-reduction fallback is `O(B_global)` per rank

The arbitrary-holder fallback stages values by dense global body ID on every
rank. This is correct but becomes a memory and collective-bandwidth bottleneck
when body count scales with process count.

Longer-term alternatives include cached communicators for holder components, a
distributed body directory, or a sparse world exchange keyed by body ID. The
current neighbor-holder route is appropriate when the documented connected
holder-set assumptions hold.

### 3.5 Integer and MPI-count limits are insufficiently guarded

Entity counts, global IDs, half-edge counts, BVH capacities, MPI counts, and MPI
displacements are all `int`. Examples of unchecked arithmetic include:

- `3 * n_triangles` and `3 * next_triangle_gid`;
- `2 * n_prims` for BVH capacity;
- `b * (n + 1)` for body-reduction messages;
- `count + needed` in growth macros;
- byte and layout arithmetic in `include/mftl_memory.h:94,147-151`;
- casts from `size_t` products to MPI `int` counts.

Several boundary checks exist only as debug assertions and disappear in Release.

**Recommendation:** Add checked add/multiply/alignment helpers, enforce explicit
maximum entity counts at public boundaries, and use MPI large-count APIs or
chunked exchanges where large data sets are intended to be supported.

### 3.6 MPI error policy is inconsistent

There are roughly 80 MPI call sites, while only about 11 explicitly inspect an
MPI return status. Most public functions advertise `MFTL_ERROR`, but communicator
operations normally use MPI's fatal error handler and generic exchange functions
return `void`.

Choose one explicit policy:

1. Fail-stop MPI errors, with `MFTL_ERROR` reserved for library validation and
   consistency failures; or
2. Install `MPI_ERRORS_RETURN` on library-owned communicators and propagate MPI
   failures collectively through status-returning APIs.

The current mixture suggests recoverability that most code paths do not provide.

### 3.7 Morton ordering has an absolute-scale cliff

**Locations:**

- `src/object/derived.c:279-294`
- `src/bvh/bvh_build.c:429-454`

Ranges at most `1e-12` force `inv_range` to zero, so all Morton keys collapse to
the same value. Correctness remains intact through stable tie handling, but
spatial ordering and tree quality disappear for small valid objects.

Use `range > 0` with robust normalization or normalize in a scale-free local
frame.

### 3.8 Topology construction is not unconditionally linear

Twin rebuilding buckets half-edges by origin but scans the destination vertex's
entire bucket for each half-edge. Complexity is therefore
`O(V + sum(degree(v)^2))`, not unconditionally `O(V + H)`.

The intended bounded-valence assumption can make this effectively linear, but
the input builder and integrity checker do not validate that assumption. Fan
traversal is separately capped by `MFTL_MAX_VALENCE`; in Release it can return a
partial or arbitrary traversal after the cap at
`include/mftl_topology.h:506-524`.

Either validate and document the valence ceiling as a hard input invariant, or
use dynamic traversal/storage for valid high-valence meshes.

### 3.9 Twenty-four bytes per vertex have no library producer

**Location:** `include/mftl_types.h:168-201`

`verts.area`, `verts.mean_curv`, and `verts.gauss_curv` are permanently allocated,
copied, compacted, and permuted, but no library routine computes them. These
three arrays cost 24 bytes per active vertex, approximately 27% of the nominal
88-byte vertex payload, plus memory traffic during pool growth and permutation.

Implement their documented geometry lifecycle with validity tracking, or make
them optional/remove them until a producer exists.

### 3.10 Thread safety is undefined

**Locations:**

- `src/bvh/bvh_predicates.c:127-168,791-815`
- `src/communication/exchange.c:28-94`
- global communication and measurement statistics

Exact-predicate initialization uses an unsynchronized lazy global flag and
mutable global constants. Concurrent first calls are a C data race. MPI datatype
registration and process-global statistics have similar shared-state behavior.

Document a single-threaded or `MPI_THREAD_FUNNELED` calling contract, or use
`call_once` and context-owned state for routines intended to be thread-safe.

## 4. API and documentation inconsistencies

### 4.1 Sparse ensemble behavior conflicts with older API documentation

`include/mftl_communication.h:423-425` says every rank must hold the same object
list in the same order for list redistribution. The current redistribution code
instead supports a sparse local directory and creates arriving bodies by global
body ID.

`include/mftl_communication.h:680-683` similarly says ghost requests carry local
object indices and require replicated list order, while the implementation sends
global body IDs and resolves them to local slots.

`include/mftl_remeshing.h:1511-1513` repeats the replicated-list precondition for
the spanning driver. These contracts should distinguish the initial collective
body-registration requirement from the sparse post-redistribution local lists.

### 4.2 Domain validation is incomplete

`MFTLEnsembleCreate` validates positive cuts and their product, but does not
validate finite bounds or `global_max >= global_min`. Negative or NaN domain
sizes can silently produce unusable inverse sizes, ownership factors, and
periodic behavior. The cut-product expression can also overflow `int` before it
is compared with communicator size.

### 4.3 Release behavior relies excessively on debug assertions

Several public preconditions that protect memory safety, MPI collective order,
topology correctness, or owner-map validity are assertion-only. Assertions are
valuable for internal invariants, but public functions advertising an error
return should perform the corresponding Release validation whenever continuing
can corrupt state or deadlock peers.

## 5. Test infrastructure findings

### 5.1 Parallel CTest runs race on fixed I/O filenames

**Location:** `tests/CMakeLists.txt:210-220`

The same `test_io` executable is registered at one, two, and four ranks. The
tests use fixed output prefixes and have neither a CTest resource lock nor
distinct working directories.

Running `ctest -j4` produced 46 of 48 passing tests, with one truncation assertion
and one allocator crash. All three I/O configurations passed individually and
in the sequential suite, confirming a test-isolation race rather than a normal
I/O implementation failure.

Use unique per-test temporary directories or declare a shared resource lock.

### 5.2 The binary VTU test parser performs unaligned typed loads

**Locations:** `tests/test_io.c:273-322,784-806,840-849`

The parser casts byte offsets in an XML appended-data buffer directly to
`double *` and `int *`. UBSan reports misaligned loads. This can fail on
strict-alignment architectures even when it appears to work on x86.

Decode scalar values with `memcpy`, then apply any required byte-order handling.

### 5.3 Important metamorphic and failure-path tests are missing

Add tests for:

- identical geometry scaled over several powers of ten;
- identical BVHs translated to large positive and negative origins;
- repeated triangle vertices, duplicate faces, and non-manifold edges in a
  Release configuration;
- degenerate point/segment triangles in intersection and distance predicates;
- rank-0 body-prefix results on the global Exscan route;
- ghost owners outside the graph path;
- deliberately different collective options on ranks, with clean collective
  rejection rather than a hang;
- large-count rejection using injected boundary values, without allocating
  multi-gigabyte buffers;
- concurrent first use of exact predicates under ThreadSanitizer if thread-safe
  operation is intended.

## 6. Validation performed

- A fresh optimized Release build completed successfully without compiler
  warnings.
- The sequential sanitizer-enabled test suite passed **48/48 tests**.
- A parallel `ctest -j4` run passed **46/48 tests**; the two failures were traced
  to colliding I/O test files.
- Individual I/O runs at one, two, and four ranks passed functionally.
- MPI correctness tests exercised configurations up to 27 ranks. These are
  oversubscribed correctness runs, not evidence of multi-node scalability.
- Focused optimized reproducers confirmed:
  - small-scale normal and quality failure;
  - non-manifold topology accepted as valid;
  - stale normals after serial remeshing;
  - translation-dependent BVH radius and tolerance;
  - incorrect degenerate-triangle distance.
- `git diff --check` reported a trailing blank line in
  `src/communication/spatial_halo.c`; this was pre-existing working-tree content
  and was not modified during the review.

No `clang-tidy`, `cppcheck`, or `scan-build` executables were available in the
review environment.

## 7. Positive design observations

The findings above should not obscure several unusually good design decisions:

- Implicit half-edge indexing removes redundant next/triangle fields and gives
  compact, deterministic topology.
- Keeping authoritative positions in a streaming SoA while retaining compact
  identity/topology records is appropriate for both geometry kernels and
  remeshing.
- The owned-prefix/ghost-suffix invariant removes per-entity ownership branches
  from many hot loops.
- The sparse local object directory avoids an `O(B_global)` object footprint in
  normal operation.
- Redistribution combines all bodies into a bounded number of messages rather
  than paying collective latency per body.
- The neighbor/global routing decision correctly favors locality while retaining
  a correctness fallback for arbitrary migration.
- The spanning remesher's shifted-plane schedule, frozen-band accounting, and
  convergence instrumentation show careful reasoning about partition-boundary
  correctness.
- The dispersed remesher explicitly recognizes its load-balancing and
  serialization regimes.
- Exact adaptive `orient3d` predicates are the correct foundation for narrow
  phase intersection decisions.
- The test suite covers a broad range of rank topologies and contains strong
  decomposition invariants, even though several boundary classes remain absent.

## 8. Recommended remediation order

1. Fix rank-0 Exscan handling, ghost-sync status, and collective control
   agreement.
2. Remove absolute geometric thresholds and add scale/translation metamorphic
   tests.
3. Harden mesh ingestion and integrity validation.
4. Make serial remesh postconditions true, including final vertex normals and
   integrity checking.
5. Correct BVH coordinate representation and degenerate distance handling.
6. Add checked count/size arithmetic and collective-safe large-message
   rejection.
7. Define and consistently implement the MPI error model.
8. Persist reduction, spatial-halo, and local-BVH workspaces.
9. Reconcile sparse-ensemble documentation and define the thread-support policy.
10. Implement or remove the unused per-vertex geometry arrays.

## 9. Suggested release gates

Before treating the library as production-ready, require all of the following:

- Scale and translation metamorphic tests pass in Debug and optimized Release.
- Invalid topology is rejected identically with and without assertions.
- All collective entry points either validate replicated control state or state
  a precise, tested fail-stop contract.
- No rank-local early return exists between collectively dependent operations.
- A ThreadSanitizer policy is established: either a clean supported threaded
  configuration or an explicit single-thread/FUNNELED API contract.
- Large entity/count limits are documented and checked without signed overflow.
- Multi-node weak- and strong-scaling runs separately report computation,
  migration, collectives, allocation/first-touch, geometry, and BVH costs.
- The full test suite passes both sequentially and with parallel CTest scheduling.

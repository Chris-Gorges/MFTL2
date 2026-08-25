## Recommendation

I would change the internal redistribution architecture, while keeping the public API, geometric ownership rule, packed identity formats, and vertex-complete halo semantics.

The present implementation should remain as:

- the initial distribution engine;
- the post-remeshing/topology-change engine;
- the arbitrary long-distance migration fallback;
- a correctness reference for testing.

Normal CFD timesteps should use a new incremental, topology-stable migration engine.

Mandatory invocation every timestep does not imply mandatory full reconstruction every timestep. The CFD requirement is only:

> At the end of each timestep, every vertex must be owned by the rank containing its new position, and the required triangle/vertex replicas must be correct for the next timestep.

That can be achieved by communicating ownership and halo deltas rather than retransmitting and rebuilding the entire existing halo.

## Why optimizing only the current implementation is insufficient

The current architecture has three fundamental properties that prevent fully scalable per-step redistribution:

1. Owned and ghost entities must occupy prefixes and suffixes of monolithic pools. Any ownership change can force a large permutation.

2. Existing ghosts are effectively discarded and regenerated. Owned triangles are resent to their full destination sets every redistribution, even when those destination sets did not change.

3. Rebuilding one object rewrites all kept entities, local indices, topology arrays, and frames, not only the migrating boundary.

Allocator improvements, better packing, and fewer collectives would make this substantially faster, but the work would remain proportional to the complete locally held mesh for every dirty object.

For many small spheres, object-level fast paths could work very well. For one large connected interface crossing at least one partition boundary every timestep, that entire object would remain dirty and the fundamental scaling problem would remain.

## State-of-the-art architectural model

There is no single “SOTA redistribution function.” Scalable distributed mesh systems converge on several design patterns:

- explicit owner-to-copy communication graphs, as represented by PETSc’s root/ghost-leaf [`PetscSF`](https://petsc.org/release/manualpages/PetscSF/PetscSF/);
- separate migration, overlap generation, and field distribution, as in [`DMPlexDistribute`](https://petsc.org/release/manualpages/DMPlex/DMPlexDistribute/);
- batched ownership and ghost modifications committed as one consistency transaction, as documented by [STK Mesh](https://trilinos.github.io/pages/packages/stk/STKManual_2024-07-12-final.pdf);
- compact array-oriented parallel mesh storage, such as [Omega_h](https://github.com/sandialabs/omega_h);
- bounded-degree neighborhood communication rather than global all-to-all communication.

For MFTL, I would specialize those ideas to its Cartesian geometric decomposition and surface-triangle topology.

## Proposed per-timestep architecture

The normal timestep would become:

```text
CFD velocity interpolation on owned vertices
                │
                ▼
Advect owned vertices
                │
                ├── begin cached owner → ghost position push
                │       └── overlap with interior triangle geometry
                ▼
Complete boundary geometry after ghost positions arrive
                │
                ▼
Determine next owners and mark the migration frontier
                │
                ▼
Compute old/new halo destination-set differences
                │
                ▼
One sparse neighbor delta exchange
                │
                ▼
Apply additions, handoffs and removals incrementally
                │
                ▼
Install owner/copy schedule for the next timestep
```

### 1. Fuse ownership detection into required kernels

All owned vertices already pass through advection, and all triangles already pass through geometry. Ownership detection should be fused into those loops:

- after moving a vertex, compare it with the local CFD subdomain bounds;
- after computing a triangle center, compare it with those bounds;
- only call the complete Cartesian owner calculation when an entity has left its current box.

The overall timestep still has unavoidable `O(Nlocal)` physics work, but redistribution no longer adds another independent full-volume scan.

Store `next_owner` only for entities whose owner changed, or store a compact change list. Do not overwrite the committed owner until the migration transaction completes.

### 2. Maintain a migration frontier

When an owner changes:

- a changed vertex marks all its incident triangles dirty;
- a changed triangle marks itself and its three edge neighbors dirty;
- those triangles are the only ones whose halo destination sets can have changed.

For an authoritative triangle `T`, the current replication rule is:

```text
replicas(T) =
    owner(T)
  ∪ owners(three vertices)
  ∪ owners(three adjacent triangles)
```

Compute this set using old ownership and next ownership. Both contain at most seven unique ranks.

The difference gives:

- `new replicas − old replicas`: triangle additions;
- `old replicas − new replicas`: triangle removals;
- intersection: retained copies needing, at most, an owner update;
- changed triangle owner: authority handoff.

This changes halo work from “send the complete current halo” to “send only halo membership changes.”

### 3. Use one packed delta stream per neighbor

Each destination should receive a compact stream containing body-run headers followed by tagged records such as:

- `VERTEX_HANDOFF`
- `VERTEX_OWNER_CHANGED`
- `TRIANGLE_ADD`
- `TRIANGLE_DROP`
- `TRIANGLE_HANDOFF`
- `COPY_SUBSCRIBE`
- `COPY_UNSUBSCRIBE`

A triangle addition carries its topology and any referenced vertices not guaranteed to exist at the receiver. Vertices are deduplicated per destination.

The stream should be constructed using:

1. per-neighbor histograms;
2. prefix sums;
3. direct packing into persistent rank-major buffers.

That eliminates per-destination buffers, the subsequent flattening copy, the separate vertex/triangle/body-count collectives, and most per-step allocation.

A normal step then needs approximately:

- one small neighbor count exchange;
- one neighbor variable-length byte-stream exchange.

Whether neighborhood collectives or direct nonblocking point-to-point communication performs better should be measured. MPI-4 also standardizes persistent neighborhood communication, but it is most useful when buffer signatures remain stable; MFTL should first cache graphs and storage, then benchmark persistent requests rather than assuming they help. [MPI-4.1 documents the persistent communication model](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm).

### 4. Replace request/reply ghost synchronization with an owner/copy graph

At the end of redistribution, the library knows:

- every locally owned vertex;
- every rank holding a copy;
- every ghost vertex and its owner.

That is precisely an owner-root/ghost-leaf communication graph. Store it.

The next position synchronization becomes:

```text
owners pack positions using cached owned slots
        ↓
one owner-to-copy exchange
        ↓
receivers write directly into cached ghost slots
```

No request IDs, request counts, GID lookup, reply construction, or reply matching are needed.

When redistribution changes ownership or halo membership, incrementally update this schedule for the next timestep. Rebuilding it from the local halo in `O(Hlocal)` would still be much cheaper than the current request/reply protocol, but an incremental update is the eventual target.

### 5. Change the local storage model

This is the most consequential architectural change.

The current monolithic pool and owned-prefix invariant are excellent for streaming kernels but hostile to incremental migration. I would replace them internally with chunked, dense owned and ghost arenas:

```text
object/rank state
├── owned vertex chunks
├── ghost vertex chunks
├── owned triangle chunks
└── ghost triangle chunks
```

Important properties:

- Owned vertices remain contiguous within chunks for CFD interpolation.
- Ghosts do not have to be shifted merely because the owned count changed.
- Fixed-capacity chunks avoid reallocating a complete body pool.
- Deleted slots are filled using local swap-delete or a free list.
- Kernels still iterate dense contiguous chunks.
- Stable handles identify entities across chunk movement.

A stable handle can be `(chunk, slot, generation)`, while persistent local maps resolve:

- `(body_gid, vertex_gid) → vertex handle`;
- `(body_gid, triangle_gid) → triangle handle`;
- undirected edge key → local half-edge handle.

The present direct integer indices are fast but force every reference to be rewritten after large permutations. Stable handles or sparse-set indirection confine updates to entities incident to the few slots that moved.

For vertex removal, only incident half-edges need updating, which is proportional to vertex valence. For triangle removal, only three vertices, three twins, and three edge-map entries are affected.

### 6. Make topology incrementally maintainable

During advection, topology itself has not changed. Only which pieces are present on a rank has changed.

Maintain these structures persistently:

- vertex-to-incident-triangle adjacency;
- global-ID-to-local-handle maps;
- edge-key-to-half-edge map;
- local vertex reference counts;
- triangle neighbor global IDs or handles.

When adding a triangle:

1. resolve or insert its three vertices;
2. insert its three edges into the edge map;
3. connect any available twins;
4. increment vertex reference counts;
5. populate its hot `verts` and `neighbors` data immediately.

When removing one:

1. unlink its twins;
2. remove its edge entries;
3. decrement vertex reference counts;
4. remove vertices whose count reaches zero and which are not locally owned.

There is then no full twin repair or full topology-derived rebuild.

### 7. Preserve existing geometry for retained entities

Before migration, the required geometry pass has already calculated current centers, areas, normals, and possibly frames.

Therefore:

- retained triangles keep their current geometry;
- a triangle that was already present as a ghost and becomes owned keeps it;
- only genuinely new triangle copies require geometry/frame calculation after arrival.

This removes both the full `MFTLObjectBuildDerived()` from redistribution and potentially the post-redistribution full geometry pass. Newly arrived boundary triangles are only `O(ΔH + migration)`.

### 8. Remove normal-step global collectives

The CFD coupling should provide a bounded-movement contract such as:

```text
maximum vertex displacement per step
    < configured Cartesian communication radius
```

The neighborhood communicator should cover that radius plus the topological-halo requirement. Under the usual CFL-limited case, destinations remain within a fixed small graph.

Then the normal path needs no global path-selection Allreduce.

For exceptional long jumps, options are:

- route records through multiple Cartesian hops;
- invoke the existing global/full fallback;
- reject the step in debug mode if the caller promised bounded movement.

The exceptional fallback may use a global agreement. The normal timestep should not.

### 9. Stop replicating the complete body directory

For fully scalable many-body weak scaling, each rank should store only bodies for which it holds owned or ghost entities, keyed by global body ID.

Replicating the entire body list makes per-rank metadata grow with global problem size even if local mesh size is fixed. This is incompatible with asymptotically flat weak scaling.

Global body metadata can be:

- replicated only if genuinely small;
- partitioned by a deterministic owner rank;
- cached locally for active bodies.

This change reaches beyond redistribution into the ensemble API, so it should be a later phase.

## Expected complexity

Let:

- `N` be owned entities on the rank;
- `H` be current ghosts;
- `M` be entities changing owner;
- `ΔH` be halo additions/removals;
- `Blocal` be locally active bodies;
- `d` be fixed neighbor degree.

| Work | Current | Incremental target |
|---|---:|---:|
| Required advection/geometry | `O(N + H)` | `O(N + H)` |
| Extra ownership detection | separate `O(N + H)` scans | fused into required loops |
| Position synchronization | `O(H)`, three collectives | `O(H)`, one cached push |
| Topology sent | approximately complete halo | `O(M + ΔH)` |
| Local mesh reconstruction | `O(N + H)` | `O(M + ΔH)` |
| Body metadata | `O(Bglobal)` | `O(Blocal)` |
| Rank metadata | partly `O(P)` | `O(d)` |
| Global synchronization | every call | exceptional fallback only |

This is the appropriate weak-scaling architecture: if `N`, `H`, `M`, `ΔH`, `Blocal`, and `d` remain fixed, per-rank redistribution work remains fixed.

Strong scaling will still eventually stop because halo surface-to-volume ratio and fixed communication latency grow relative to local work. No architecture can remove that physical limit.

## If the current architecture must be retained

I would make these changes, in order:

1. **Global/object clean fast path.** Every call still happens, but an object with no owner changes keeps its topology and halo and receives only `MFTLObjectRefreshFrames()`.

2. **Rebuild only dirty objects.** Use outgoing and incoming body metadata to skip `rebuild_object()` for unaffected objects.

3. **Cache the ghost owner/copy schedule.** Convert position synchronization from request/reply into one push exchange.

4. **Use controlled pool growth slack.** Prevent small boundary ingress from reallocating entire monolithic pools repeatedly.

5. **Retain all communication workspace.** Counts, displacements, ID lists, GID tables, send/receive storage, and grouping arrays should persist on the ensemble.

6. **Direct-pack final buffers.** Perform a histogram and prefix sum, then pack directly into the final MPI buffers.

7. **Use one packed multi-body stream.** Remove the third count payload and separate vertex/triangle data collectives.

8. **Add an explicit neighbor-only steady-state mode.** Remove the per-step global path Allreduce under a caller-provided movement bound.

9. **Fuse reconstruction passes.** Fill derived topology while relinking, use epoch maps instead of clearing old maps, and apply an in-place permutation or a single old-to-new copy rather than gather-plus-copy-back.

10. **Preserve ghosts for clean objects.** Do not regenerate their existing halo.

This could give a large practical improvement for many droplets because most individual bodies do not cross a cut on most timesteps. It still cannot provide true migration-proportional scaling for a single large dirty object.

## Recommended implementation roadmap

### Phase 1: Instrument and optimize reusable pieces

- Add internal timings and counters for ownership changes, halo adds/removes, bytes, pool growth, packing, waits, reconstruction, and derived rebuild.
- Add persistent communication workspaces.
- Add controlled pool slack.
- Implement cached push-based ghost position synchronization.
- Add neighbor-path configuration.

These changes benefit both engines.

### Phase 2: Add a topology-stable delta engine

Introduce a topology epoch:

```text
topology epoch unchanged → incremental redistribution
topology epoch changed   → existing full redistribution
```

Initially, use existing pools but:

- detect dirty objects;
- preserve clean objects;
- communicate only dirty bodies;
- rebuild only locally affected objects.

This provides a safe intermediate result.

### Phase 3: Introduce incremental local storage

- Separate owned and ghost storage.
- Add chunked arenas and stable handles.
- Add persistent GID and edge maps.
- Implement triangle/vertex add, drop, and handoff.
- Incrementally maintain topology and reference counts.

At this point, normal redistribution becomes migration-proportional.

### Phase 4: Delta halo and sparse body directory

- Compare old/new replica sets.
- Send only additions, removals, and ownership handoffs.
- Transfer owner/copy schedules with ownership.
- Replace global object arrays in the hot path with locally active body maps.

### Phase 5: Overlap and on-node parallelism

- Split interior and boundary triangle work.
- Overlap cached ghost pushes with interior geometry.
- Parallelize frontier marking, histograms, packing, and arrival application.
- Benchmark neighborhood collectives, point-to-point schedules, and persistent MPI requests independently.

## Validation strategy

The existing full redistribution should serve as an oracle. In test mode, run both engines from the same pre-step state and compare by global ID:

- vertex and triangle owners;
- local owned/ghost entity sets;
- vertex-complete halo;
- positions and flags;
- triangle connectivity;
- half-edge twins;
- derived topology;
- periodic-boundary behavior.

Critical fixtures should include face, edge, and corner crossings; several entities crossing in opposite directions; an authority handoff where the new owner already has a ghost; empty ranks; long-distance fallback; many bodies; and the first step after remeshing.

The decisive success metrics would be:

- zero topology bytes when no ownership or halo membership changes;
- no full pool copies in a normal timestep;
- no per-step global collective on the fast path;
- one cached position push plus one topology-delta exchange;
- topology bytes proportional to `M + ΔH`;
- flat weak scaling when local entities, halo, migration rate, and active-body count are fixed.

So my answer is: preserve the current redistribution as a robust full-rebuild engine, but do not try to turn that engine alone into the final scalable CFD path. Build an incremental migration/ghost-transaction architecture beside it and make that the normal per-timestep implementation. No code was changed.

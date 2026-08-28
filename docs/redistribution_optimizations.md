The four phases are mostly local memory-management and topology reconstruction—not MPI work. Together they account for about 24.4 ms of the 39.9 ms redistribution:

| Phase | ms/step | Share |
|---|---:|---:|
| Kept gathering | 7.61 | 19.1% |
| Destination classification | 6.80 | 17.1% |
| Frame reconstruction | 5.28 | 13.2% |
| Relinking | 4.71 | 11.8% |

The relevant pipeline is:

```text
classify owners
    ↓
find destinations + build kept-index lists
    ↓
pack/exchange arrivals
    ↓
gather kept records into scratch
    ↓
classify arrivals and determine final owned/ghost slots
    ↓
place kept records back
    ↓
relink kept topology + insert arrivals + repair twins
    ↓
rebuild topology arrays
    ↓
rebuild geometric frames
```

## 1. Kept gathering

A “kept” entity is already on the correct rank and remains necessary either as an owned entity or a ghost. It should not travel through MPI.

The difficulty is that the final pools must have this layout:

```text
[ kept owned | newly arrived owned | kept ghosts | newly arrived ghosts ]
```

The number of newly arrived owned entities is unknown until arrivals have been received and deduplicated. Therefore, the old pool cannot simply be compacted from left to right.

The current implementation gathers the authoritative parts of every kept entity into reusable scratch in [redistribute.c](/Users/christian/git/MFTL_6/src/communication/redistribute.c:1165):

- Vertex: 16-byte record + 24-byte position = 40 bytes.
- Triangle: 12-byte record + three 16-byte half-edges = 60 bytes.
- Derived geometry is deliberately not copied.

After arrivals are classified and final slots become known, those records are copied back into their final owned/ghost positions.

For this benchmark, roughly 2.52 million vertices and 5.01 million triangles are kept across all ranks. That represents about 401 MB of authoritative records. Gather plus placement causes approximately 1.6 GB of aggregate read/write traffic per step.

Possible optimizations:

1. Skip rebuilding unchanged bodies

This is the highest-potential optimization. If no entity belonging to a sphere changed owner and its topology did not change, its existing ownership prefix and halo remain valid. That body does not need destination analysis, gathering, relinking or topology reconstruction; it only needs its frames refreshed if positions moved.

This requires:

- A per-object “partition topology is still current” state, so a remeshed object is never incorrectly skipped.
- Detecting local owner changes during the existing owner scan.
- A global per-object OR reduction—preferably a compact bitset—to ensure every rank makes the same skip decision.

With 1,000 independently moving spheres, many bodies probably do not cross a partition boundary during a particular step even though some sphere always does globally. Per-object skipping could therefore be far more effective than an ensemble-wide skip.

2. Copy consecutive runs with `memcpy`

The kept-index lists are ascending. Since about 97% of entities are retained, they should consist mainly of long consecutive runs separated by small holes.

Instead of copying records element by element:

```c
old = kept[i];
scratch[i] = pool[old];
```

the gather can detect runs such as `[100,101,...,146]` and copy each run using `memcpy`. That does not reduce the byte count, but it gives the compiler/libc a much better bulk-copy shape.

This is a relatively low-risk prototype.

3. In-place permutation

A cycle-following permutation could eliminate scratch and the copy-back pass. I would not start here: it replaces one random-read/sequential-write stream with random reads and random writes across several arrays. Memory systems generally prefer the current gather direction.

It might become attractive only if implemented as a specialized “fill holes created by departing entities” algorithm rather than a general cycle permutation.

## 2. Destination classification

This phase decides where every owned triangle must exist after redistribution. The rule in [collect_destinations](/Users/christian/git/MFTL_6/src/communication/redistribute.c:381) is:

```text
triangle centre owner
+ owner of each of its 3 vertices
+ owner of each of its 3 edge-neighbour triangles
= at most 7 candidate ranks
```

The centre owner receives the owned triangle. The other ranks receive ghosts required to preserve the vertex-complete halo.

For each owned triangle, the code:

1. Reads its three vertex indices and three neighbour indices.
2. Looks up the seven candidate owners.
3. Removes duplicate ranks.
4. Sets local keep flags or appends the triangle to a remote destination list.
5. Adds its vertices to each destination, using a per-vertex 32-bit destination mask to avoid duplicates.

The bucket also includes `build_kept_lists()`, which scans vertices twice and triangles twice:

```text
pass 1: append kept owned entities
pass 2: append kept ghost entities
```

That ordering is necessary because owned entities must form the pool prefix.

Possible optimizations:

1. Count while setting keep flags, then fill in one pass

While classifying destinations, the code can detect the first transition from `keep == 0` to `keep == 1` and count kept-owned versus kept-ghost entities.

Once those counts are known, one subsequent scan can write both runs:

```c
owned_cursor = 0;
ghost_cursor = n_kept_owned;

if (keep[i]) {
    kept[owner[i] == my_rank ? owned_cursor++ : ghost_cursor++] = i;
}
```

That replaces four scans—vertices owned/ghost and triangles owned/ghost—with two. It preserves ascending order in both runs, which the gather relies on for locality.

This is probably the best small, contained optimization in this phase.

2. Specialize the all-same-owner fast path

The existing code already recognizes that most triangles have all seven candidates on one rank. However, it first constructs the candidate array and then checks it.

A specialized path could load and compare the owners directly, without materializing `cand[7]`:

```text
r = triangle owner
if all vertex and neighbour owners equal r:
    process exactly one destination
else:
    construct and deduplicate candidate list
```

The benefit is likely modest, but this path runs for essentially every interior triangle.

3. Maintain an incremental boundary frontier

A triangle’s destination set can only change when one of these changes:

- Its own owner.
- One of its vertex owners.
- One of its neighbour-triangle owners.
- Its topology.

Therefore, destination classification could retain the previous destination state and revisit only:

- Previously boundary/replicated triangles.
- Triangles incident to entities whose owners changed.
- Triangles affected by topology changes.

This is potentially large, but it becomes an incremental redistribution design rather than a local loop optimization. It also requires receiver-side removals of ghosts whose destination relationship disappeared.

4. Avoid fusing packing immediately

Packing currently costs only about 0.30 ms/step. Packing directly inside destination classification would eliminate temporary ID lists, but would scatter writes across multiple send buffers and mix record/position reads into an otherwise compact connectivity pass. It could easily hurt cache locality more than it saves.

## 3. Frame reconstruction

A triangle frame consists of:

```text
anchor = position of vertex 0           3 doubles
edge 1 = minimum-image(v1 - v0)         3 doubles
edge 2 = minimum-image(v2 - v0)         3 doubles
```

That is 72 bytes written per triangle. The implementation is in [derived.c](/Users/christian/git/MFTL_6/src/object/derived.c:108).

Frames exist so downstream geometry, normals, BVH and IO kernels can process triangles using contiguous arrays without gathering three vertex positions or handling periodic seams themselves.

In the requested periodic domain, frame reconstruction gathers three positions per triangle and performs six minimum-image corrections. Across 5.12 million triangles, it writes approximately 369 MB of frame data per step.

Possible optimizations:

1. Use a seam-free object fast path

The current dispatch uses the periodic kernel whenever the domain has any periodic axis. But most small spheres are nowhere near a global periodic seam and need no minimum-image corrections.

While vertices are already being gathered or placed, compute the local wrapped AABB of each body. If its extent along every periodic axis is less than half the domain length, every raw vertex difference is already the minimum-image difference. That body can use the much cheaper non-periodic frame loop.

The test is conservative: a large object may be classified as potentially crossing a seam even when its individual triangles do not, but it cannot produce incorrect frames.

With 1,000 small spheres in a `4×4×4` domain, this could remove periodic rounding from most bodies.

2. Generate frames during relinking and arrival insertion

Once a triangle’s final vertex indices and all final vertex positions are known, its frame can be generated immediately. This would fuse frame writes into loops already touching the triangle and its half-edges, eliminating the later traversal through `tris.verts`.

The operation order would need a small adjustment: arriving vertices must be placed before kept triangles are relinked so every final position is available.

This is a medium-risk but promising fusion.

3. Fuse frame and post-redistribution geometry

The benchmark subsequently computes area, normal and centre. Those values are direct functions of the anchor and two edge vectors.

A fused kernel could:

- Gather the three positions once.
- Apply periodic correction once.
- Store anchor and edges.
- Compute and store area, normal and centre from the same register values.

For this benchmark, that could attack the combined 5.28 ms frame reconstruction plus 9.46 ms post-redistribution geometry.

It should be an explicit fused operation rather than a changed redistribution default: the documented production timestep does not always require geometry immediately after redistribution, whereas redistribution promises current frames to all callers.

## 4. Relinking

Compaction changes local pool indices. Every retained half-edge still names vertices and twins using the old indices, so those references must be translated.

The relinking phase in [redistribute.c](/Users/christian/git/MFTL_6/src/communication/redistribute.c:1371) builds:

```text
old vertex index   → final vertex slot
old triangle index → final triangle slot
```

Then, for every kept triangle and its three half-edges, it:

1. Translates the target vertex index.
2. Translates the twin triangle while preserving the twin’s corner number.
3. Marks a twin unresolved if its old partner did not survive.
4. Rebuilds a vertex’s `out_he_idx`.
5. Adds unresolved half-edges to the small twin-repair set.

The implementation is already clever in one important way: it does not rebuild every twin through a global lookup. Twins between two retained triangles survive through a simple index remap; only broken/new relationships enter twin repair.

Possible optimizations:

1. Emit derived topology during relinking

After relinking one triangle, all three final vertex indices are known. The code can immediately write `tris.verts`.

For a retained twin, the final neighbour triangle is also known, so it can immediately write `tris.neighbors`. For unresolved edges, initialize the neighbour to `MFTL_ERROR`; when `repair_twins()` pairs two edges, update both neighbour entries there.

That removes the subsequent full `MFTLObjectBuildTopologyArrays()` pass, currently another 2.44 ms/step, without changing the public contract.

This is my strongest concrete fusion candidate.

2. Build mappings earlier

The old-to-logical mapping can be populated while constructing the kept lists or gathering records. After arrival counts are known, logical indices are converted to final slots.

This could eliminate a later pass over every kept index, although ghost slots require an offset based on the number of newly arrived owned entities. A compact encoded logical mapping could defer that offset calculation to relinking.

Whether this wins depends on whether one extra arithmetic operation per half-edge is cheaper than the eliminated mapping pass.

3. Avoid epoch-stamped maps here

Epoch maps help when only a small fraction of the index range is populated. Here almost every old entity is retained, so the mappings are dense. Clearing a dense integer array is likely cheaper than doubling its traffic with a stamp array.

## Recommended order

I would investigate these in this order:

1. Split the destination timer into actual destination analysis and kept-list construction.
2. Count kept-owned/ghost entities during classification and build each kept list in one scan.
3. Emit `tris.verts` and `tris.neighbors` during relinking/twin repair, eliminating the topology-array rebuild.
4. Add a seam-free per-object frame kernel.
5. Prototype run-based kept gathering.
6. Design per-object unchanged-partition skipping—the largest opportunity, but the one requiring the strongest new state invariant.
7. Only then consider a fully incremental halo/destination scheme.

The best near-term targets are the one-pass kept-list construction and topology emission during relinking. The likely largest long-term gain is skipping redistribution work for bodies whose ownership and halo did not change.

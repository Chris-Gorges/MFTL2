> Resolution (2026-08-30): persistent triangle frames were removed from the
> object and every consumer now gathers current positions through compact
> connectivity. Frame-related API names below describe the reviewed historical
> implementation and are retained as decision evidence.


## Implemented result

The frame removal was measured in the same Release + LTO build, before and after
the change, with four MPI ranks, 256 spheres, refinement level 3 (327,680
triangles), 500 steps, and an output digest:

```text
mpirun -n 4 ./build/parallel_advection_redistribute_list \
  2 2 1 1 1 1 --spheres 256 --refine 3 --steps 500 --digest
```

| timing, median of 7 runs | with frames | without frames | result |
|---|---:|---:|---:|
| whole timestep loop | 3.945 s | 3.217 s | **1.226x throughput; 18.5% less time** |
| redistribution + deferred post-geometry | 2.717 s | 2.128 s | **1.277x throughput; 21.7% less time** |

The whole-loop ranges were 3.536--4.121 s before and 3.023--3.340 s after; all
seven frame-free samples were faster than every frame-backed sample. The
partition set/order digest was identical in every run:
`set 1dbb389d394d61a5 order 6f79c8f911b7139d`.

The harder consumer case adds `--quality --normals`, forcing the gathered
geometry/statistics and vertex-normal paths every step. Its median whole-loop
time fell from 5.283 s to 4.344 s over five runs: **1.216x throughput, or 17.8%
less elapsed time**. Thus the frame-free result still wins after charging the
new gathered consumers, rather than merely moving work out of redistribution.

Persistent storage also fell by **72 bytes per triangle of pool capacity** (one
3-double anchor plus two 3-double edges): 22.5 MiB for this benchmark's 327,680
triangles before ghost/capacity overhead.

I read `redistribute.c`, `comm_internal.h`, `derived.c`, and §7b–7d of `docs/performance-analysis.md`. Every number below is from the phase breakdown already measured in that doc (1000 bodies, 1 280 000 triangles, np = 4, phase ≈ 33.3 ms/step), so these are ranked against a real profile rather than against structure.

## Verdict

| idea | share of phase it can touch | correctness risk | difficulty | verdict |
|---|---|---|---|---|
| **1. Skip unchanged bodies** | up to **68 %** (many-body only; **0 %** for one big body) | **high — silent** | medium | **do it, but measure the ceiling first with a counter** |
| **2a. Kept-list scans 4→2** | ~2–3 % | none | trivial | do it — but do the cheaper fix I describe below first |
| **2b. All-same-owner fast path** | part of 16 % | none | low | try it, measure; may be gather-bound not stack-bound |
| **3a. Seam-free frame path** | ~1.6 % | none | low | **no** |
| **3b. Frames during relink** | ~2–3 % | low | medium | **no** |
| **3c. Fuse frames + geometry** | ~1 ms + contract change | none | high | **no — delete the frames instead** |
| **4a. Emit derived topology during relink** | **5 %** | low, test-guarded | low | **do it — best effort/reward in the list** |
| **4b. Build mappings earlier** | ~0 | none | medium | **no** (but one free sub-win, below) |
| **4c. Avoid epoch maps** | negative | none | — | **no — the premise is wrong** |

---

## OPTIMIZATION 1 — skip unchanged bodies

**Your reasoning is sound and this is the biggest lever in the list.** The local state of rank *r* after a redistribution is a pure function of (global topology, owner map): rank *r* holds triangle *t* iff *r* owns *t*, one of its three corners, or one of its three edge neighbours, and holds the corners of everything it holds. Neither input changed ⇒ neither does the output. Positions moved, but positions are not an input to that function.

**What it saves.** Everything except the owner scan, the frames rebuild, and the collectives:

| removed by a skip | 22.6 ms | 68 % |
|---|---:|---:|
| kept gather 5.72, destination analysis 5.29, relink 3.85, topology arrays 1.79, arrivals 1.69, pool growth 1.67, place-back 1.65, twin repair 0.93 | | |
| **survives** — owners 2.72, frames 3.60, route 2.25, wait 0.98, misc 1.20 | 10.8 ms | 32 % |

That is a higher ceiling than the delta-planner architecture in `docs/` (measured at 1.6× on the phase) for a fraction of the work. But it is **entirely a many-body optimization**. For one large body spanning the grid, no step will ever have zero owner changes, and you pay the compare plus a collective for nothing. It needs to be free when it doesn't fire.

### On your "is it not simply enough to check for owner changes"

Almost. Three things it does not cover:

**Topology change — why it matters.** Two reasons, and the second is the fatal one:
1. The halo rule reads `tris.verts`/`tris.neighbors`. A flip changes which rank needs *t* as a ghost without moving anything.
2. **Remeshing renumbers global IDs and creates entities whose halo was never communicated.** A split midpoint between two vertices on the same rank produces a *new* triangle with zero owner changes. So `changed == 0` is true and the partition is wrong. Worse: `MFTLRemeshEnsembleParallel` and `MFTLRemeshEnsembleDispersed` both *close* with a plain `MFTLRedistribute` precisely to restore the canonical fine partition (and deferred issue 19 says `MFTLUpdateGhostPositions` drops silently on a coarse partition, so that closing call is load-bearing). A skip would elide exactly the call that is not optional.

Don't add a hand-maintained "partition topology is current" flag — the invalidation set is open-ended and one missed site is a silent corruption that surfaces steps later. **Piggyback on a discipline that already exists and is already asserted:** add a monotone `topology_generation` bumped by `MFTLObjectBuildTopologyArrays`, and store `partition_generation` on the object at the end of a rebuild. Skippable requires equality. Any topology change must clear `MFTL_DERIVED_TOPOLOGY`, and `collect_destinations` already asserts `MFTLObjectHasTopology` — so the counter is exactly as strong as a contract the whole library already enforces, rather than a new one. Keep the `MFTLObjectHasTopology` assert on the skip path too, or you lose it.

**Ghost flags go stale, not just ghost positions.** This one isn't in your list. A full rebuild replaces every ghost triangle over the wire each step, so ghost `flags` are refreshed for free. `MFTLUpdateGhostPositions` syncs positions; nothing syncs flags. A skip freezes them. Today only the remesher writes flags (`MFTL_FLAG_FROZEN`) and that path changes topology, so it is safe — but it must be written down as a precondition, or the next thing that mutates a flag without touching topology breaks silently.

**The reduction must be global per body**, as you say — a rank with zero local changes can still be the *receiver* of a change made elsewhere. A bitset in one `MPI_Allreduce(MPI_BOR)` over `ceil(num_objects/32)` ints is 125 B for 1000 bodies; message count doesn't grow with the ensemble.

### The objection you'd expect, and why it's weaker than it looks

The bitset reduction must happen *before* the destination analysis, so it cannot be fused into the existing route `Allreduce` (which needs `requires_global`, only known afterwards). That looks like paying the route bucket twice — 7 %, 2.25 ms. It isn't: the header comment on `t_route` says it is the *first* collective after the per-object loop and therefore absorbs load imbalance the way `t_wait` does. Insert the skip reduction there and *it* becomes the imbalance sink; the route reduction that follows finds the ranks already synchronised and costs pure latency. You add one latency (log P), not one imbalance absorption.

### Do this first, and it costs nothing

The instrumentation from commit `2272622` already gives you the harness. Add `n_objects_skippable` to `MFTL_RedistributeStats`, compute the decision, **and don't act on it.** One run of `parallel_advection_redistribute_list` then tells you the skippable *body* fraction — which is the number the whole idea turns on and which nothing measured so far reports. The 95 %-unchanged figure in the doc is per **entity**; per **body** it will be far lower, because a body straddling a cut plane almost certainly has some triangle cross every step. My estimate for the 1000-body case is 60–75 % skippable, but that is arithmetic on a body-radius/cell-width ratio, not a measurement, and it is exactly the kind of geometric prediction CLAUDE.md records over-predicting by 2× for `frozen_fraction`.

### The acceptance test is already built

For an unchanged body the full rebuild produces a permutation that should be the identity for the owned run and a stable re-arrival order for the ghost run — so a skip should be **bit-identical** to a rebuild. `parallel_advection_redistribute_list --digest` compares both the set form and the slot-order form, and that is precisely how the five no-regret fixes were proven behaviour-neutral. Use the same protocol. If the digests differ for a skipped body, the skip is unsound and you'll know immediately rather than three hundred timesteps later.

---

## OPTIMIZATION 2 — destination classification

### 2a. The four scans

Correct that it's four scans and correct that a two-cursor pass preserves ascending order in both runs. But **there is a cheaper fix you've missed, and I'd do it before the restructure**: `build_kept_lists` reads `vr[v].owner_proc` out of the 16-byte `MFTL_Vertex` record and `tr[t].owner_proc` out of the 12-byte `MFTL_Triangle`, when `vert_owner[]` and `tri_owner[]` are dense `int` arrays sitting in the workspace holding exactly that value, still hot from `compute_owners`. Same four passes, 4 B/entity instead of 16 and 12. Two-line change, no counting, no hot-loop change, ~4× less traffic in that function.

Then, if you still want 4→2: don't count inside `collect_destinations`. Setting `keep_v[v]` there is currently an unconditional store; counting first transitions turns it into a read-modify-write with a loop-carried dependency in the hottest loop of the phase, to save a pass elsewhere. **One pass with two cursors and no counting** is available instead: fill owned forward from index 0 and ghost *backward* from the top of the body's arena slice (which is already sized `verts.n`), then reverse the ghost run in place and `memmove` it up against the owned run. The ghost run is O(halo), so the fix-up is negligible. One pass, no extra state, ascending in both runs.

Worth maybe 2–3 % of the phase. Safe, but not where the money is.

### 2b. Avoiding `cand[7]`

The mechanism you name is real: `cand[n_cand++]` uses a *variable* index, which almost certainly forces the array to the stack rather than to registers — so it's 7 stores and 7 loads per triangle that a branch-free XOR-OR reduction would not do:

```c
const int r = tri_owner[t];
int diff = (vert_owner[v0]^r) | (vert_owner[v1]^r) | (vert_owner[v2]^r);
for (int j = 0; j < 3; ++j) { const int a = tn[base+j]; if (a != MFTL_ERROR) diff |= (tri_owner[a]^r); }
if (!diff) { /* single destination */ } else { /* build cand[7], dedup */ }
```

**But I'd temper the expectation.** That loop also does four scattered gathers per triangle (`vert_owner[v0..v2]`, `tri_owner[adj]`) into arrays of 4 B × N. If it is gather-bound rather than stack-bound, removing the stack traffic buys little. Destination analysis is the second-largest bucket (16 %, 5.29 ms), so even 15 % off it is worth having — but this is a "check the disassembly, then measure" change, not one to predict. Zero correctness risk either way.

---

## OPTIMIZATION 3 — frames

**All three sub-ideas optimise work that is already known to be dead, and the right move is the one you didn't list: delete it.**

The doc measures the frame rebuild at 3.60 ms, 11 % of the phase, and states it plainly: redistribution builds `tris.anchor`/`tris.edge` as its last act; step 1 of the next iteration moves owned vertices and `MFTL_FOREACH_OWNED_VERTEX` *clears* `MFTL_DERIVED_FRAMES` on the way out; nothing reads them in between; step 3's `MFTLComputeTriangleGeometry` then takes its walk path *because* they are gone. 72 B/triangle written and discarded unread.

**3a (seam-free fast path).** Your test is *correct* — a straddling body has a raw AABB of nearly `L` and fails conservatively, and raw extent < L/2 does imply every raw difference is already the minimum image. It's also cheap. But the periodic correction is only ~16 % of the frame build (§9: streaming frames 1.64 ns/tri non-periodic vs 1.90 periodic), so the ceiling is 16 % of 11 % ≈ **1.6 % of the phase**, and only for bodies that pass. Your own instinct is right.

**3b (frames during relink).** Legal — the arrival-vertex loop only needs `v_final_of_logical` and `recv_v_logical`, both fixed in step 3, so hoisting it above the relink is safe. But it saves the streaming read of `tris.verts` and one loop's overhead; the 3 scattered position gathers and the 72 B write remain, and they are the cost. ~2–3 %, and the frames now get written in two places instead of one.

**3c — and your question about `MFTLEnsembleUpdate`.** No, it doesn't already do this. `MFTLEnsembleUpdate` *consumes* frames (it dispatches on `MFTL_DERIVED_FRAMES`) and writes area/normal/centre plus nine quality reductions; it never writes `anchor`/`edge`. So the fusion you describe is genuinely absent. And it would genuinely help — this is deferred issue 7 in CLAUDE.md, which measures the post-redistribution geometry pass at 6.4 ns/tri against 2.9 ns for the same function on a resident mesh, i.e. bound by first touch, and names fusion into the rebuild as the only remaining lever. But it pushes geometry into the one module that deliberately transmits none, and it moves the contract in the *opposite* direction from where the measurement points.

**The coherent change is to weaken the postcondition:** return with `MFTL_DERIVED_FRAMES` **cleared**, and let the caller that actually wants frames (`MFTLEnsembleUpdate`, `MFTLComputeVertexNormals`, `MFTLSpatialHaloResolveRadii`, `MFTLExtendSpatialHalo`) call `MFTLObjectRefreshFrames` itself. That is 11 % recovered outright, for a one-line change plus a documentation and test update — larger than 3a+3b+3c combined and simpler than any of them. The header already flags it: *"this is the first thing to renegotiate if the phase needs to get faster."* The cost is that the four call sites above must be audited, since the documented loop's "exactly one geometry pass" argument is written around frames being current on return.

---

## OPTIMIZATION 4 — relinking

### 4a. Emit derived topology during relinking — **yes, do this**

This is the best-value item after Optimization 1. `MFTLObjectBuildTopologyArrays` reads 48 B and writes 24 B per triangle for values that are already in registers in the relink loop:

- `tris.verts[3s+k]`: `build_topology_arrays` computes `tv[b+0] = he[b+2].vertex_idx` etc. The relink loop already forms **exactly that rotation** two lines later for `out_he_idx`. Free.
- `tris.neighbors[3s+k]`: `tw_tri` is already in a register on the retained-twin branch; `MFTL_ERROR` otherwise. Free.
- Arriving triangles: same, with all neighbours `MFTL_ERROR`.
- `repair_twins` writes both `he[h].twin_idx` and `he[h2].twin_idx`, so it adds two `tn[] = MFTLHeTriIdx(...)` stores per repaired pair. The repair set is O(halo).

The writes are sequential — `slot` is the identity for the kept-owned run and `i + n_new_t_owned` for the kept-ghost run, so both are contiguous streams. You trade one full pass (72 B/tri of traffic) for 24 B/tri of stores into lines you're already sequential over. **5 % of the phase, 1.79 ms.** Set `obj->derived |= MFTL_DERIVED_TOPOLOGY` by hand, and `inv_derived_matches_topology()` (`tests/test_mesh_invariants.h:291`, reached from `inv_mesh_is_sound`) catches any mistake directly — it re-derives both arrays from the records and compares.

### 4b. Build mappings earlier — no, but take the free part

The trade is not "one extra arithmetic operation per half-edge". Deferring the logical→final conversion into the relink loop makes `h3[k].vertex_idx = v_final_of_logical[v_map_of_old[...]]` — a **dependent random load** per half-edge, three per triangle, against one sequential store per kept entity that you'd eliminate. That's the wrong direction.

**The free part:** the clear `for (v) s->v_final_of_old[v] = MFTL_ERROR` exists only to feed the Debug assert *"a kept triangle references a vertex that was not kept"* — the value is never tested in Release, because a kept triangle's corners are always kept by construction. Move that clear under `#ifndef NDEBUG` and you drop `n_old_v × 4 B` of writes per body per step. The **triangle** clear must stay: `t_final_of_old[tw/3] == MFTL_ERROR` is how the relink retires a twin whose triangle was not kept, and that is load-bearing.

### 4c. Avoid epoch-stamped maps — the premise doesn't hold

Two different maps are being conflated:

- `v_final_of_old` / `t_final_of_old` **are already plain dense cleared `int` arrays.** No epoch, no stamp. Nothing to change.
- The epoch-stamped maps are `s->v_tbl` / `s->t_tbl`, and they are keyed by **global ID, not old pool index.** Global IDs are dense across the whole *body*; a rank holds a partition, so the IDs it looks up are a sparse subset scattered over the body's range. That is measured, not argued: one 327 680-triangle body at np = 2/4/8 asked for 15.2 / 31.0 / **62.0 MB** of direct-addressed table per rank per step, **97 % empty at np = 8** (`comm_internal.h:265`). "Almost every old entity is retained, so the mappings are dense" is true of the first pair and false of the second.

And the stamp does not double traffic: the epoch and the key are packed into **one** `uint64_t tag[]` word specifically so a probe is a single load and a single compare (`comm_internal.h:296–305`). Removing it would mean clearing the range on every call — which is the per-timestep megabytes fix 2 of the last commit just removed, at 77–183× on the id-table bytes.

---

## What I'd do, in order

1. **Add `n_objects_skippable` as a counter only** and run it. It's an afternoon, it costs nothing at runtime, and the skippable *body* fraction is the single number that decides whether Optimization 1 is a 1.9× or a rounding error. Nothing measured so far reports it.
2. **4a** — emit `tris.verts`/`tris.neighbors` during relink. 5 %, low risk, guarded by an existing invariant.
3. **Delete the frames rebuild** (postcondition change) rather than any of Optimization 3. 11 %, one line plus an audit of four call sites.
4. **2a's cheap half** — read `vert_owner[]`/`tri_owner[]` instead of the records in `build_kept_lists`; and the `#ifndef NDEBUG` on the vertex clear from 4b.
5. **Optimization 1**, if step 1 says the body fraction justifies it — with the `topology_generation` counter, the ghost-flag precondition written down, and `--digest` bit-identity as the acceptance criterion.
6. **2b** last, disassembly first.

Steps 2–4 together are ~18 % of the phase for changes that are individually small and individually testable — comparable to the entire delta-planner architecture's measured 1.6× ceiling, without the rewrite. One caution carried from §7c and §7d of the doc: the last five "obviously correct" fixes removed 2.6× the allocator traffic and made the phase **no faster**, because macOS hands out large blocks as lazily-mapped zero pages. Anything here that isn't a reduction in *touched bytes* should be assumed to buy nothing until an interleaved A/B at ≥200 steps says otherwise.

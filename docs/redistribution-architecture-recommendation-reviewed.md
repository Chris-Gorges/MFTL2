# Redistribution architecture: reviewed recommendation

> Historical design note: persistent triangle frames were removed on
> 2026-08-30. Frame-state postconditions below describe the implementation this
> recommendation reviewed.

## Status

This document reviews [`updated_redistribution-architecture-recommendation.md`](updated_redistribution-architecture-recommendation.md)
against the code, and then restates the architecture with the corrections folded in. It supersedes
that document. No implementation is made here.

Every code claim below carries a `file:line`. Every number is quoted from the tree or from
`docs/performance-analysis.md` / `docs/scaling-analysis.md` / `docs/parallel-remeshing-analysis.md`;
none is estimated.

**Verdict.** The direction is right and most of the mechanism is sound. Three load-bearing claims do
not survive contact with this codebase, one proposed structure violates a stated invariant while
duplicating machinery that already exists, and one cost category is missing entirely. The
sparse-object requirement does not buy what the original says it buys.

| # | Finding | Severity |
|---|---|---|
| F1 | The "small delta" premise is false for the dominant caller (remeshing) | critical |
| F2 | Removing empty objects does not remove the `O(B_global)` replication | critical |
| F3 | "Transactional" is misleading — there is no rollback | wording |
| F4 | Statelessness is the current design's main safety property, and losing it is unnamed | critical |
| F5 | The edge-key hash map should be cut — it duplicates `repair_twins` | design |
| F6 | The vertex incidence lists are mostly redundant with the half-edge fan | design |
| F7 | Slot swapping destroys spatial locality; nothing addresses it | missing |
| F8 | Redistribution should stop building frames, not "recompute only affected" | design |
| F9 | The path-selection Allreduce is not irreducible — NBX removes it | missed |
| F10 | Global-ID renumbering is under-specified, and it happens every remesh | gap |
| F11 | Sparse GID tables let you stop renumbering altogether — under-claimed | missed |
| F12 | The "cheap local-box test" has a measured precedent against it | correction |
| F13 | Memory budget for the new persistent state is absent | gap |
| F14 | Phase ordering front-loads the widest blast radius with no payoff | plan |
| F15 | Validation does not name this repo's actual guards | plan |
| F16 | Six smaller corrections | correction |

---

## What the original gets right

Kept unchanged, and restated in Part II:

- **One engine, not two.** A steady-state engine beside a full-rebuild engine is a maintenance and
  correctness trap.
- **Delta planning via `Rold` / `Rnew` set difference** over a seven-element stack array with linear
  duplicate removal. This mirrors `collect_destinations`'s existing fast path
  (`src/communication/redistribute.c:384-394`) and is sound. Both sets are computable locally: the
  vertex-complete halo guarantees a triangle's owner holds all three corners and all three edge
  neighbours, and `owner_proc` is stored on ghosts.
- **Sparse GID→slot tables sized by the local count, explicitly not by `next_*_gid`.** This is the
  worst defect in the current code. `redistribute.c:799` seeds `v_range` from
  `obj->next_vertex_gid` — the *global* vertex count — and `:836` sizes the table from it, at 8 B per
  global id (`comm_internal.h:172-178`). Because `rebuild_scratch` is a stack local (`:1470`) freed at
  `:1495`, the epoch mechanism that `comm_internal.h:138-158` argues for at length **never amortises
  across calls**, and the whole stamp array is memset every timestep. The same defect is in
  `ghost_sync.c:132`, followed by a scatter of every owned vertex into it at `:133-137`.
- **Push-based ghost synchronisation with owner-side copy-rank reference counts.** This is the right
  answer to `CLAUDE.md` deferred issue 5, whose stated blocker — *"only the holder of a ghost knows
  that it holds it"* — is dissolved exactly by maintaining holder counts inside the transaction.
- **Neighbour-degree-sized arrays instead of `nproc`-sized ones**, with rank-sized storage only when
  the global path is selected. `redistribute_core` currently allocates eight `nproc`-sized structures
  and runs five `nproc` loops per call (`:1237`, `:1256-1262`, `:1264`, `:1269`, `:1271`,
  `:1287-1289`, `:1354`, `:1394`), and `ghost_sync.c:203-205` does the same **per timestep** to build
  a ≤26-entry inverse map.
- **Direct-pack into the send buffer**, removing `tri_ids`/`vert_ids`, the separate payloads, and the
  flattening `memcpy` at `exchange.c:197`.
- **Point-to-point versus neighbourhood collectives must be measured, not assumed**, and no
  payload-size heuristic without a crossover reproduced on more than one MPI implementation.
- **Instrument before rewriting.** `git grep MPI_Wtime -- src/` returns one hit, in
  `remesh_comm.c:295`. There are no timers in `src/communication/` at all, so every cost attribution
  in either document is currently inference from code, not measurement.

---

# Part I — Findings

## F1 (critical). The "small delta" premise is false for the dominant caller

The original's payoff — topology payload `O(M + ΔH + J)` instead of the full halo — assumes `M`,
the number of entities changing owner, is small.

That holds for the CFD advection step. `docs/scaling-analysis.md:128-136` measures **130 triangles
per step of ownership churn against 490 of wire traffic** at np = 4.

It is **false for remeshing**, which is where the time actually is:

- `docs/parallel-remeshing-analysis.md:174-176` — *"Between **96 % and 99.5 %** of the timestep is
  inside `MFTLRemeshEnsemble`."*
- `:140-141` — *"**4.8–4.9 migrations per remeshed timestep**"*, each of them a
  `MFTLRedistributeWithOwners`.
- The **spanning** strategy shifts the cut planes by `h/2` over the `{0, h/2}³` schedule, so a half
  box of entities changes owner at every phase. `M ≈ O(N_local)`, not a rim.
- The **dispersed** strategy moves whole bodies onto the rank owning their centroid. `M = N_body`.

For those calls the delta engine degenerates to a full transaction **with strictly higher per-entity
constants than today**: tagged variable-length records instead of a fixed-size struct, GID-map probes
instead of array indexing, and one-at-a-time slot swaps instead of the two bulk `memcpy`s at
`redistribute.c:938-961`.

So the proposal as written can make the dominant path slower while speeding up the minor one.

**Correction — add a bulk record form.** `BODY_SNAPSHOT`: *"discard your local state for this body;
here is the whole thing."* Its payload is exactly today's `MFTL_PackedVertex` /
`MFTL_PackedTriangleComplex` arrays and its apply path is today's gather-and-two-memcpys. It keeps
"one engine, one parser, one transport" — a snapshot is one more record tag, not a second
implementation — while giving remeshing and the initial scatter a bulk path.

The planner chooses per body per call on a measured threshold (`M > α·N_local`, α to be measured; the
crossover is where per-record overhead exceeds the bytes saved). Both forms must produce **identical
committed state**, which is a cheap differential test and the strongest correctness gate available for
the delta path.

## F2 (critical). Removing empty objects does not remove the `O(B_global)` replication

The original treats *"a rank must not retain an `MFTL_Object` with no local entities"* as a fixed
requirement. Its cost and its benefit are both different from what is claimed.

### What it saves, measured

`sizeof(MFTL_Object)` is **280 B**. `MFTLObjectCreate` (`src/object/object.c:256-272`) `calloc`s the
struct and leaves both pools at capacity 0 with no block, so a replicated *empty* object costs 280 B
plus 8 B of list pointer — **288 B per body per rank**.

| bodies | replicated empty objects, per rank |
|---|---|
| 10^3 | 0.3 MB |
| 10^4 | 2.9 MB |
| 10^5 | 29 MB |
| 10^6 | 288 MB |

The original never states this number, so there is no way to tell which regime it is optimising for.
Note that this is *not* the memory problem `CLAUDE.md` deferred issue 20 describes — that one is
about **pool capacity** on the generating rank (`3680 · 4^refine + 176` bytes per body), which
survives redistribution because pools never shrink, and which removing empty objects does not touch.

### What it cannot remove

The distributed remesher's determinism argument requires every rank to hold metadata for **every**
body. `src/remeshing/remesh_dispersed.c:255-273`:

> *"Every rank runs this loop and every rank must reach the same answer, because `MFTL_OwnerMap` is
> undefined — not slow, undefined — if two ranks name different destinations for the same entity.
> There is no reduction anywhere below: the centroids and the weights came out of
> `MFTLRemeshMeasureBodies`' Allreduce and are already identical to the bit… Determinism extends to
> the tie-breaks. Two ranks with equal load are ordered by rank id and two bodies of equal weight by
> **body index**."*

`MFTLRemeshMeasureBodies` (`src/remeshing/remesh_comm.c:404`) is
`MPI_Allreduce(acc, sum, 8 * num_objects, MPI_DOUBLE, MPI_SUM, ...)`, indexed by list position, and
documented at `:353` as *"the module's only `O(num_objects)` collective"* — deliberately widened from
stride 7 to stride 8 to avoid needing a second one. With sparse local lists that reduction cannot be
formed, the balancer cannot run, and the index tie-break is undefined.

At least **eight** collectives are sized and indexed by object-list position:

| site | payload |
|---|---|
| `src/ensemble/ensemble.c:303` | `2 · num_objects` ints (global counts) |
| `src/remeshing/remesh_comm.c:280` | `n` ints, MAX |
| `src/remeshing/remesh_comm.c:286` | `n` doubles, SUM |
| `src/remeshing/remesh_comm.c:292` | `n` doubles, MAX |
| `src/remeshing/remesh_comm.c:404` | `8 · n` doubles (measure bodies) |
| `src/remeshing/remesh_comm.c:477` | `n` doubles (radii) |
| `src/communication/spatial_halo.c:364` | `12 · n` doubles (extents) |
| `src/remeshing/remesh_comm.c:759-799` | ghost-ID grouping by object index |

plus `src/io/vtu_writer.c:1170`.

So replicated per-body **metadata** — gid, centroid, radius, weight, counts ≈ 48 B — is required
regardless. Sparse identity saves roughly **240 B of 288, i.e. 6×, not elimination**, and costs an API
break plus edits to `ghost_sync.c`, `spatial_halo.c`, `ensemble.c`, four remeshing files, the VTU
writer and every example.

Worth noting that `spatial_halo.c:367-371` already handles *"nobody holds this body"* explicitly
(`halo->global_radius[b] = 0.0`). The current design expects possibly-empty replicated lists and
degrades gracefully; nothing is broken today.

### Correction — split the requirement and stage it

**B1 — active-set iteration. Cheap, no API break, do it first.**
Keep `ensemble->objects` replicated and index-addressed. Add `ensemble->active[]`, a compact list of
locally non-empty object indices, maintained by redistribution's commit. Every hot loop iterates
`active` (`O(B_local)`); every collective keeps iterating the dense list unchanged. This removes every
`O(B_global)` hot-path loop — the actual per-timestep cost — with **zero blast radius**, and captures
most of the original's Phase 2 benefit.

**B2 — true sparse identity. Only if measured to matter.**
Body-GID wire format, GID directory, create-on-arrival, destroy-on-empty — *plus* a distributed
per-body metadata directory with a deterministic home rank (`hash(body_gid) % nproc`) and a two-stage
sparse reduction (holders → home rank → holders) to serve the eight collectives above. The original
lists that directory as *"optional"*; it is **mandatory** for B2, because the dispersed balancer
cannot function without it, and building it is a larger job than the redistribution change itself.

Gate B2 on a measured RSS contribution from empty objects at the target body count. At 10^4 bodies it
is 2.9 MB and the answer is no.

## F3. "Transactional" is misleading — there is no rollback

The original says *"stage and validate all incoming operations before mutating pools"* and then
*"reserve each monolithic pool once"* before applying. Once application begins there is no rollback,
and there cannot be a collective abort. `:86` half-concedes this; the word "transactional" appears
throughout and will be read as atomicity.

**Correction.** Call it **validate-then-apply**, and state the guarantee exactly: *inputs are parsed
and validated, and capacity is reserved, before any pool is mutated — so application cannot fail.
There is no rollback and no abort. A validation failure is a fatal invariant violation, not a
recoverable condition.*

## F4 (critical, and unnamed in the original). Statelessness is what the current design buys

Today's redistribution is **stateless**. Every rank re-derives the whole partition from replicated data
every step, so any transient inconsistency is erased on the next call. That is why `MFTL_OwnerMap`'s
contract can be *"a geometric rule evaluated on identical coordinates satisfies this by
construction"* (`include/mftl_communication.h:427-433`), and it is why the parallel remesher needs no
negotiation at all.

The delta design replaces that with **persistent distributed state**: committed owners, copy-rank
reference counts, GID→slot maps, the ghost push schedule, generation numbers. One missed or misordered
record and the state is silently wrong **forever** — and the failure mode is the worst one this
library has. `src/remeshing/remesh_comm.c:612-624` describes it for the analogous case: divergent
positions on two holders → divergent triangle centres → divergent destinations next step → a triangle
quietly loses an edge neighbour, while `MFTLCheckMeshIntegrity` reports a perfectly valid mesh.

This is the single biggest risk in the proposal and the original does not name it.

**Correction — two mechanisms, both cheap.**

1. **A consistency audit.** Debug: every call. Release: every `K` calls and after every remesh.
   Reduce a per-body checksum over the owned `(global_id, owner_proc)` multiset plus the copy-rank
   membership, and compare it against a locally re-derived one. This is the distributed analogue of
   what `tests/test_redistribution.c` already does serially — re-run `MFTLRebuildTwins` after a
   redistribution and assert it changes nothing.
2. **`BODY_SNAPSHOT` doubles as resync.** With F1's bulk form, recovery is *"emit a full transaction
   from committed state"* — the same engine, the same parser, one record tag. The original already
   frames initial distribution that way; make it a first-class, permanently supported mode rather
   than a startup special case.

## F5. The edge-key → half-edge hash map should be cut

The original proposes *"an edge-key-to-half-edge table keyed by endpoint global IDs"* as persistent
per-object state, and requires `swap_triangle_slots` to update *"edge-map values"*.

Three problems.

It is a **hash map**, and `CLAUDE.md` states: *"There are no hash maps… Do not reintroduce one
without saying which lookup genuinely cannot be an array."* The original makes no such argument.

It must be maintained through **every** slot swap, which is pure overhead on the path the whole
redesign exists to make cheap.

And it is **redundant**. `repair_twins` (`redistribute.c:721-752`) already answers "find the partner
of this half-edge" with transient origin-vertex bucketing over the repair set only, in
`O(|repair set|)`, with no persistent state and no hashing:

```c
for (int j = 0; j < n_u; ++j) v_head[he[MFTLHePrevIdx(u_list[j])].vertex_idx] = MFTL_ERROR;
for (int j = 0; j < n_u; ++j) { ... u_next[j] = v_head[origin]; v_head[origin] = j; }
```

Note it clears only the buckets it will use, so it is not an `O(V)` pass despite `v_head` being sized
by `verts.n`.

**Correction.** Delete the edge map. Keep `repair_twins`, applied to the touched set. The file header
at `redistribute.c:39-53` proves the repair set is closed under the twin relation, and the argument
carries over to the delta case unchanged: both sides of a surviving twin are either both retained or
both re-linked, so a half-edge that lost its partner can only pair with another such half-edge.

## F6. The vertex incidence lists are mostly redundant — `he_vertex_prev` in all cases

The original proposes

```text
vertex_he_head[verts.capacity]
he_vertex_next[3 * tris.capacity]
he_vertex_prev[3 * tris.capacity]
```

which is **24 B per triangle plus 4 B per vertex of new permanent state** (+11 % on a 212 B/triangle
pool), maintained through every swap.

Its purpose is "find every half-edge naming this vertex slot". The half-edge structure already answers
that: the fan walk from `out_he_idx` via twin/next, implemented in `include/mftl_topology.h` as
`MFTLGetVertexOneRingHalfedges` (`:636`) with `MFTLGetVertexOneRingVertices` (`:581`) covering the
open-fan case. It is `O(valence)` with **zero** extra state.

The catch is real: mid-transaction the twins may be temporarily inconsistent, so the fan is not
walkable.

**Correction.** Order the application phases so that fans are walkable when vertices move — apply all
triangle adds, drops and promotions and repair twins **first**, then do all vertex slot movement in a
single pass at the end. That deletes all three arrays.

If profiling later shows the fan walk is too slow, add `vertex_he_head` + `he_vertex_next` only. A
singly-linked list suffices because a vertex's list is only ever rebuilt wholesale (`O(valence)`),
never spliced from the middle. `he_vertex_prev` is unnecessary in every scenario and should not be
built.

## F7 (missing entirely). Slot swapping destroys spatial locality

The original specifies `erase ghost: swap with n - 1; decrement n` — the classic unordered erase, and
the only `O(1)` one.

Today's rebuild produces a pool in a deterministic order derived from the old one (kept entities in
ascending old-index order, arrivals in arrival order — `redistribute.c:588-600`), so the Morton
locality that `MFTLObjectReorderMorton` establishes decays slowly. Under swap-with-last it decays
**fast**: at the measured churn of ~130 triangles per step into an ~80 k pool, the ordering is
thoroughly scrambled within ~1000 steps.

Everything downstream is a streaming or gathering kernel over the pool: `build_frames`' three random
gathers into `verts.pos` (`src/object/derived.c:165-167`), `MFTLComputeVertexNormals`' scatter, the
LBVH build, the 2.9 ns/triangle geometry pass. The original's complexity table says *"Retained
geometry/topology: unchanged"* — the retained *data* is unchanged, but the **rate at which it can be
read falls over time**, and no benchmark in this repo would catch it. They all run 10–100 steps.

**Correction.** State it as a known cost with a measured remedy:

- Keep swap-with-last. It is the only `O(1)` erase and the alternative — stable compaction — is the
  `O(n_local)` term the redesign exists to remove.
- Re-Mortonise a body every `K` redistributions, triggered by a cheap disorder metric: the mean
  |Δslot| between edge-adjacent triangles, which is the statistic `docs/performance-analysis.md` §2
  already uses (Morton cut it 160×). Re-Mortonisation is a deterministic function of positions, so
  bitwise reproducibility survives.
- It must run **inside** the commit, not beside it: it invalidates every GID→slot map, the ghost push
  schedule and every cached slot in the workspace.
- Add a long-run drift test — 10^3–10^4 steps, reporting the geometry pass's ns/triangle over time.
  Nothing in the suite currently does this, so the decay is invisible today.

## F8. Redistribution should stop building frames, not "recompute only affected"

The original's *"Recompute geometry/frames only for new or affected triangles"* looks like a saving
and is not: on a CFD timestep **every vertex moved**, so every triangle's frame is affected. The
saving is zero and the bookkeeping is real.

The right move is the opposite. `build_frames` (`derived.c:150-211`) writes 72 B/triangle plus three
random gathers, and the only thing it feeds is the frame dispatch inside
`MFTLComputeTriangleGeometry`, measured at **~1 %** in the one production loop here
(`docs/performance-analysis.md` §9). The caller already owes a full geometry pass — it is step 5 of
the documented timestep and costs 13–15 % of it (`performance-analysis.md:310-312`).

`tris.verts` and `tris.neighbors` are a different matter: they are pure functions of
`he[].vertex_idx` and `he[].twin_idx`, both of which the relink loop (`redistribute.c:983-1014`) and
the arrival loop (`:1049-1060`) already hold in registers.

**Correction.** New postcondition: topology valid, positions valid, `MFTL_DERIVED_TOPOLOGY` **set**
(written inline by the relink and arrival loops, deleting `build_topology_arrays` from this path),
`MFTL_DERIVED_FRAMES` **clear**. This is an API change to `include/mftl_communication.h` and needs an
explicit decision, but it deletes a full pass over the pool from every timestep and simplifies the
delta bookkeeping rather than complicating it.

## F9. The path-selection Allreduce is not irreducible — NBX removes it

The original calls the `MPI_Allreduce(MPI_LOR)` at `redistribute.c:1373` *"a remaining fixed global
latency cost… required for safe dynamic selection because a receiver cannot know that a non-neighbor
sender selected the global communicator."*

That is precisely the problem **NBX** solves (Hoefler, Siebert, Lumsdaine, *Scalable Communication
Protocols for Dynamic Sparse Data Exchange*): `MPI_Issend` to each destination, `MPI_Ibarrier` once
all local sends are locally complete, `MPI_Iprobe` + receive until the barrier fires. No rank ever
forms an `O(P)` count array, and **no rank needs to know in advance whether a non-neighbour sender
exists** — which collapses the neighbour/global dichotomy, the `requires_global` flag and the
Allreduce into a single mechanism.

The original lists point-to-point only as a *neighbour* transport and never considers it as a
path-selection mechanism.

**Correction.** Add NBX as a third measured backend, evaluated on the axis the other two are not:
collectives per step.

| | per redistribution | ghost sync | total |
|---|---|---|---|
| today | 4–5 (`Allreduce` + count `alltoall` + 2–3 `alltoallv`) | 3 | **7–8** |
| original's target | 3 (`Allreduce` + count + data) | 1 | **4** |
| with NBX | 1 | 1 | **2** |

At the measured halo fraction (2–3 %, payloads in the tens of KB) the phase is latency-bound, not
bandwidth-bound — `performance-analysis.md:369-371` says so explicitly for the ghost sync — so
collective count is the right figure of merit.

## F10. Global-ID renumbering is under-specified, and it is not rare

`include/mftl_remeshing.h:89-95`: the serial path re-densifies with `MFTLRemeshRenumberGlobalIds`, and
*"The distributed driver re-densifies with an `MPI_Exscan` instead, and pays one ghost exchange for
it."* `renumber_global_ids` defaults **on** (`:1048`). So after **every** remesh pass, every global
ID in the body changes.

That invalidates, simultaneously on every rank: both GID→slot maps, the copy-rank reference-count
tables (keyed by vertex GID), the ghost push schedule, and every journal entry keyed by GID. The
original mentions renumbering once, as a test-fixture line item.

**Correction.** Treat a renumbering as a named **generation break** with its own specified handling.
Because the Exscan plus ghost exchange gives every rank the old→new map for everything it holds, the
cached structures can be *remapped* rather than rebuilt; specify that remap as an explicit step with
its own invariant check, and bump the generation word so any stale schedule is rejected loudly (F16).

## F11 (under-claimed). Sparse GID tables let you stop renumbering altogether

`CLAUDE.md` deferred issue 15 states the reason remeshing must re-densify: *"`src/communication/
redistribute.c` and `ghost_sync.c` size their arrival lookup tables from `next_*_gid`, so holes
accumulated over a few hundred timesteps become an unboundedly growing table for a mesh of constant
size."*

Once those tables are sparse and sized by the **local** count, that reason evaporates.

Turning `renumber_global_ids` off would remove a full `O(V + T)` pass from every remesh, the
`MPI_Exscan` and its ghost exchange from every distributed pass, and the generation break in F10. It
would also demote `inv_global_ids_are_dense` from an invariant to an optimisation.

**Correction.** Record it as an explicit follow-on, gated on the sparse tables landing, and audit what
else relies on density before promising it — `redistribute.c` and `ghost_sync.c` are the two callers
`CLAUDE.md` names, but the check should be exhaustive.

## F12. The "cheap local-box test" has a measured precedent against it

The original proposes *"a cheap local-box test should precede the full Cartesian owner lookup."*

`MFTLGetOwnerProc` (`include/mftl_geometry.h:166-177`) is branch-free and measured at 1.79 ns/entity.
The one documented attempt to put a conditional in that exact loop is recorded on the function itself
(`:163-164`):

> *"Adding the shift and its periodicity branch here cost **1.70x** — 1.79 to 3.05 ns/entity, 42 to
> 89 instructions for the same loop."*

That regression is the entire reason `MFTL_OwnerMap` exists as a seam.

**Correction.** Keep the idea, mark it **measure-only**, and cite the precedent so nobody lands it on
argument. A branchless variant — compute the cell index and compare it against this rank's own cell
index, with no early exit — is more likely to survive, and costs one compare rather than a branch.

## F13. Memory budget for the new persistent state is absent

The original adds roughly ten persistent structures and never totals them. Current pool cost is
**88 B per vertex and 212 B per triangle** (`src/object/object.c:75-86`, `:181-195`), and pools never
shrink (`:120`, `:225`).

As written, the proposal adds:

| structure | cost | verdict |
|---|---|---|
| `he_vertex_next` + `he_vertex_prev` | 24 B/tri | **cut** (F6) |
| `vertex_he_head` | 4 B/vert | cut, or 4 B/vert if the fallback is needed |
| edge-key map | ~16 B/half-edge | **cut** (F5) |
| sparse vertex GID table | ~16 B/vert | keep |
| sparse triangle GID table | ~16 B/tri | keep |
| per-vertex triangle refcount | 4 B/vert | keep |
| epoch dirty stamps | ~8 B/entity | **move to workspace** — per-call, not persistent |
| `next_owner` | 4 B/entity | **move to workspace** — per-call, not persistent |
| copy-rank refcount lists | ~8 B per (owned vertex, holder) | keep; bounded by halo |
| journal | `O(J)` | keep; transient per pass |

As proposed that is roughly **+20 % on the triangle pool and +40 % on the vertex pool** before the
copy-rank structure — in a library whose stated scaling problem includes memory. With F5 and F6
applied and the per-call arrays moved to the ensemble workspace, it is ~16 B/entity plus the
copy-rank lists.

**Correction.** Put this table in the architecture, with per-item justification, and state the rule:
**per-call state lives in the ensemble workspace; only state that must survive between calls lives on
the object.**

## F14. Phase ordering front-loads the widest blast radius with no payoff

The original's Phase 2 rewrites body identity across `ghost_sync.c`, `spatial_halo.c`, `ensemble.c`,
four remeshing files, the VTU writer and every example — **before** the new redistribution exists in
Phase 4. Maximum regression risk, zero measurable benefit, and it lands on top of the eight
index-addressed collectives in F2.

**Correction.** See Part III. Every phase ships a measurable win with contained risk, and the sparse
identity change moves to the end where it is optional.

## F15. Validation does not name this repo's actual guards

The fixture list is generic. This repo has specific, hard-won tests that are exactly the ones a delta
engine breaks. They should be named as gates. See Part IV.

## F16. Smaller corrections

- **"one count-free exchange when schedule sizes are unchanged."** A receiver cannot know sizes are
  unchanged without being told. Both sides derive the schedule from the same committed transaction, so
  it does work — but it is fragile. Put a **generation word in the payload header** and assert on it.
  It is free, and it converts a silent desync into a loud failure.
- **`MFTLRedistributeObject`.** The original's reasoning is correct. Note additionally that
  `MFTLRedistribute` is itself a `static inline` dispatcher in `include/mftl_communication.h:387`, so
  this is a header/ABI change visible to every example and test.
- **"global vertex and triangle counts if those fields remain eagerly maintained."** They are.
  `MFTLEnsembleUpdateGlobalCounts` (`src/ensemble/ensemble.c:303`) is one Allreduce of
  `2 · num_objects` ints, and it is one of the eight index-addressed collectives in F2.
- **"sends vertices, triangles, and body counts as separate payloads."** Accurate, but they already
  share **one** count exchange and are posted back-to-back non-blocking (`exchange.c:139-219`), so the
  saving is one or two collectives, not three round trips.
- **"monolithic pool growth occasionally copies a complete local object pool."** Worse than stated
  today. `MFTLObjectSetCounts` (`src/object/object.c:317`) calls `MFTLObjectReserve` with the **exact**
  counts — no `MFTLGrowCapacity` — so a halo that jitters upward by one entity reallocates the whole
  block, `memset`s all of it (`:130`, `:230`) and `memcpy`s all nine triangle sub-arrays
  (`:235-243`), including `area`, `normal`, `center`, `verts`, `neighbors`, `anchor` and `edge`, every
  one of which the redistribution postcondition declares undefined. The original's *"controlled
  geometric capacity growth"* fixes a present bug, not just a future one, and the one-line version is
  worth landing on its own.
- **`MFTL_DEDUP_BITS`.** The original does not mention it. The current 32-slot budget is per object
  (`redistribute.c:1312` resets `dd.n_bits`), and above 32 distinct remote destinations the fallback
  is a radix sort with an internal `malloc` per (object, destination) — a path `np ≤ 27` cannot reach
  and nothing tests (`CLAUDE.md` deferred issue 18). Direct-packing with a destination-local
  de-duplication table removes the cliff entirely; say so.

---

# Part II — The revised architecture

## Decision

Rewrite `redistribute_core()` as **one unified, validate-then-apply delta engine**, retaining the
public role of `MFTLRedistribute()` and the existing ownership and halo semantics.

```text
classify current and next ownership
        ↓
import topology mutation journals
        ↓
mark migration frontier; derive Rold / Rnew deltas
        ↓
choose per body: DELTA or BODY_SNAPSHOT
        ↓
aggregate, de-duplicate, direct-pack one record stream per target
        ↓
transport (neighbour collective | point-to-point | NBX)
        ↓
parse and validate; reserve pools once
        ↓
apply: triangles, twins, then vertices
        ↓
commit owners, copy-rank counts, ghost schedule, active set
        ↓
audit (debug: always; release: every K)
```

There is no `if (topology_changed) full_rebuild(); else incremental();` split. The difference between
initial distribution, remeshing, neighbour migration and global migration is expressed entirely in
the **records planned** and the **transport chosen** — including `BODY_SNAPSHOT`, which is a record
tag, not a second code path.

## Postconditions

1. Every vertex and triangle is owned by the rank the active owner rule selects.
2. Every authoritative triangle exists on its triangle owner, the owners of its three vertices, and
   the owners of its three edge neighbours.
3. Every local triangle has its three vertices present (vertex-complete halo).
4. Owned entities occupy the pool prefix; ghosts occupy the suffix.
5. Global IDs are the cross-rank identity; local slots are invalidatable.
6. Half-edge twins, `out_he_idx`, `tris.verts` and `tris.neighbors` are valid.
7. **`MFTL_DERIVED_TOPOLOGY` is set; `MFTL_DERIVED_FRAMES` is CLEAR.** *(changed — F8)*
8. The owner-to-copy ghost schedule describes the committed state and carries its generation.
9. `ensemble->active[]` lists exactly the locally non-empty objects. *(B1)*
10. *(B2 only)* An object with no local entities is absent from the local ensemble.

**Guarantee:** inputs are parsed and validated, and capacity is reserved, before any pool is mutated,
so application cannot fail. There is **no rollback and no abort** (F3).

## State model

### Ensemble workspace — one, reusable, grown geometrically, never freed between calls

Local body directory and `active[]`; rank→neighbour-slot map; outgoing action arena; incoming
operation arena; counts and displacements sized by graph degree `d` (rank-sized **only** when the
global path runs); persistent send and receive byte buffers; destination-local vertex de-duplication
table; the current ghost schedule; per-call `next_owner` and dirty stamps (F13); counters and timers;
transport selection.

A steady timestep with an unchanged maximum delta performs **zero allocations**.

### Per-object persistent state — only what must survive between calls

Sparse vertex GID→slot table; sparse triangle GID→slot table; per-vertex local triangle reference
count; owner-side copy-rank reference counts; distribution generation number; the mutation journal.

**Explicitly not present:** the edge-key map (F5), `he_vertex_next` / `he_vertex_prev` /
`vertex_he_head` (F6), `next_owner` and dirty stamps (F13).

GID tables are **open-addressed and sized by the local owned-plus-ghost count**. The justification
required by `CLAUDE.md`'s no-hash-maps rule is: global IDs are dense **mesh-wide**, but a rank holds
`1/P` of them, so the direct-address array that is correct at `P = 1` is `P`-times oversized at scale.
This is the one lookup in the library that genuinely cannot be an array. A dense table remains a valid
adaptive optimisation when the global range is demonstrably close to the local count.

## Planning

### Committed and next ownership

Do not overwrite `owner_proc` while planning; compute `next_owner` into workspace scratch. The
mandatory correctness scan is `O(N_local + H_local)` over positions and cached centres — cheap, and
not the current dominant cost. A local-box pre-test is **measure-only** (F12).

Ghost centres must be current, exactly as today (`redistribute.c:263-264`): the caller runs
`MFTLComputeTriangleGeometry(obj, obj->tris.n)` covering ghosts, or two ranks disagree about a ghost's
destination.

### Frontier and replica sets

A vertex whose owner changes marks every incident triangle; a triangle whose owner changes marks
itself and its three edge neighbours; a journalled edit marks the edited triangle, its vertices and
its edge neighbours. Only a currently owned, authoritative triangle emits changes for itself.

```text
Rold(T) = old_owner(T) ∪ old_owner(its 3 vertices) ∪ old_owner(its 3 edge neighbours)
Rnew(T) = next_owner(T) ∪ next_owner(its 3 vertices) ∪ next_owner(its 3 edge neighbours)
```

Both are at most seven ranks; compute with a seven-element stack array and linear duplicate removal,
as `collect_destinations` already does at `redistribute.c:369-394`.

- `Rnew − Rold` → `TRIANGLE_ADD`
- `Rold − Rnew` → `TRIANGLE_DROP`
- ownership change → `TRIANGLE_HANDOFF`
- intersection → retain the triangle, commit the new owner only

An old owner still in `Rnew` **demotes** rather than deleting and recreating. A new owner already
holding it as a ghost **promotes** rather than duplicating.

### Vertex presence and copy accounting

A `TRIANGLE_ADD` implies `VERTEX_UPSERT` for its three vertices, de-duplicated by
`(destination, body, vertex_gid)`. A non-owned vertex is erased only when its local triangle reference
count reaches zero; an owned vertex survives a transient zero.

Copy-rank accounting rides the same exchange as aggregated
`(body, vertex_gid, holder_rank, delta)` records addressed to the vertex's **next** owner. On a vertex
handoff the old owner transfers its non-zero copy list, the new owner combines base plus deltas, and
the resulting non-zero set becomes the next push schedule — no second subscription round.

### Choosing DELTA or SNAPSHOT (F1)

Per body, per call: if the planned record volume exceeds `α · N_local`, emit `BODY_SNAPSHOT` instead.
Expected outcomes — advection: DELTA. Remeshing migrations and the initial scatter: SNAPSHOT.
`α` is measured, not assumed, and both forms must commit identical state.

## Wire protocol

One variable-length byte stream per target, grouped by body, versioned and tagged.

```text
BODY_BEGIN          body id, generation word, protocol version, creation metadata
BODY_SNAPSHOT       n_verts, n_tris, then today's MFTL_PackedVertex / PackedTriangleComplex arrays
VERTEX_UPSERT
VERTEX_HANDOFF
VERTEX_REFERENCE_DELTA
TRIANGLE_ADD        gid, next owner, flags, 3 vertex gids, 3 half-edge flags
TRIANGLE_DROP
TRIANGLE_HANDOFF
BODY_END
```

Every record header carries a tag and a byte length, so unknown or version-mismatched data is rejected
rather than misparsed. The generation word in `BODY_BEGIN` is what makes a stale ghost schedule or a
missed renumbering fail loudly (F10, F16).

Packing: append logical actions to the arena while planning → aggregate reference deltas and
de-duplicate upserts → histogram bytes by target → prefix-sum → **pack directly into the persistent
send buffer**. Local-destination actions go through the same parser and apply path without touching
MPI, so local and remote transitions are identical by construction.

This removes `tri_ids` / `vert_ids`, the per-object kept arenas, the three separate payloads, the
flattening `memcpy` at `exchange.c:197`, arrival regrouping by object-list index, and the
`MFTL_DEDUP_BITS` cliff.

## Transport

Three backends beneath one algorithm, selected by measurement:

1. **Neighbourhood collectives** — `MPI_Neighbor_alltoall` for counts, `MPI_Ineighbor_alltoallv` for
   data; persistent variants once buffer signatures are stable.
2. **Direct non-blocking point-to-point** — one count message per active graph edge, `MPI_Isend` /
   `MPI_Irecv` for non-empty payloads, request arrays and tags cached per edge.
3. **NBX** (F9) — `MPI_Issend` + `MPI_Ibarrier` + `MPI_Iprobe`. No `O(P)` count array, and **no path
   decision at all**: it handles arbitrary sparse destination sets natively, which retires
   `requires_global`, the `MPI_Allreduce` at `redistribute.c:1373`, and the neighbour/global
   dichotomy.

If a path decision is still needed (backends 1 and 2), it stays as today's communicator-wide
`MPI_LOR`, and rank-sized count storage is allocated lazily on first global use and retained.

Do not assume point-to-point wins for sparse traffic or that collectives win for regular traffic. MPI
implementation, graph degree, eager limits, message size, progress behaviour and shared-memory
placement can all reverse it. Benchmark on the laptop and on the target cluster; keep a second backend
only if it wins in a materially different supported regime.

## Pool mutation

All order changes go through a small, exhaustively tested primitive set:
`swap_vertex_slots`, `swap_triangle_slots`, insert/erase/promote/demote for both classes. No other
code may move a pool array by hand.

`swap_vertex_slots` updates every `MFTL_VertexPool` array, the vertex GID→slot map, every incident
half-edge's `vertex_idx` **found by fan walk** (F6), `out_he_idx`, and slot-indexed workspace metadata.

`swap_triangle_slots` updates every `MFTL_TrianglePool` array, the triangle GID→slot map, the six
half-edge records, twins in adjacent triangles pointing at moved half-edges, `out_he_idx` values
pointing into moved ranges, and `tris.neighbors`.

Prefix-preserving forms are as the original states them: insert-then-swap-to-boundary, promote by swap
with `n_local`, demote by decrement-then-swap, erase by swap with `n − 1`. A vertex swap is
`O(valence)`; a triangle swap is bounded by three vertices and three neighbours.

Capacity uses **geometric growth** (`MFTLGrowCapacity`), fixing the present exact-reserve bug (F16),
with one reserve per pool per object transaction computed from conservative final counts.

### Application order (fixed by F6)

1. Parse and validate every record; reject conflicting ownership or incompatible add/drop.
2. Resolve body IDs; create shells for arriving bodies.
3. Build per-object journals **keyed by global ID**, never by slot — a swap invalidates a remembered
   slot, a GID lookup stays correct.
4. Compute capacity; reserve each pool once.
5. Upsert vertices required by additions and handoffs (append only; no slot movement yet).
6. Add, update and promote triangles; write `tris.verts` / `tris.neighbors` inline (F8).
7. Apply triangle demotions and drops.
8. **Repair twins over the touched set** (`repair_twins`, F5). *Fans are now walkable.*
9. Apply vertex ownership transitions and erase unreferenced non-owned vertices — all vertex slot
   movement happens here, in one pass, using the fan walk.
10. Commit `owner_proc`, copy-rank counts, the ghost schedule and its generation.
11. Update `active[]`; *(B2)* destroy locally empty objects.
12. Re-Mortonise if the disorder metric has crossed its threshold (F7).
13. Audit (F4).

## Pool order and locality (new — F7)

Swap-with-last is kept; the decay it causes is answered by in-commit re-Mortonisation on a measured
trigger, and the trigger is a disorder metric, not a step counter. Re-Mortonisation is deterministic
in the positions, so bitwise reproducibility survives; it invalidates every cached slot, so it runs
inside the commit before the schedule is frozen.

## Cached push-based ghost synchronisation

At commit, every ghost vertex knows its owner, every owner knows its non-zero copy ranks, and the
workspace holds source and destination slot lists grouped by communication edge, stamped with the
commit generation. The next sync is: pack owned positions in cached order → one exchange (with the
generation word in the header) → write into cached destinations. No request phase, no reply matching,
no GID lookup on the steady apply path, no allocation.

`MFTLUpdateGhostVertexNormals` reuses the same schedule with a different field pointer.

This does not reduce the bytes — position data stays `O(H_local)` per step — it removes two phases,
the per-call GID scatter (`ghost_sync.c:132-137`), the `nproc`-sized inverse map
(`ghost_sync.c:203-205`) and the allocations. It also removes the silent-drop failure at
`ghost_sync.c:213-242`, since the owner-side schedule cannot contain a non-neighbour it did not
choose.

## Consistency audit (new — F4)

Debug: every call. Release: every `K` calls and after every remesh. Per body, reduce a checksum over
the owned `(global_id, owner_proc)` multiset and the copy-rank membership; compare against a locally
re-derived one. On mismatch, emit `BODY_SNAPSHOT` for that body from committed state — the same
engine, one record tag — and report the violation.

## Complexity (corrected — F1, F7, F8)

`N` local owned, `H` local ghosts, `M` owner changes, `ΔH` replica changes, `J` journalled mutations,
`B_local` / `B_global` bodies, `d` graph degree, `P` ranks.

| | today | **advection regime** (`M ≪ N`) | **remeshing regime** (`M ~ N`) |
|---|---|---|---|
| ownership classification | `O(N + H)` | `O(N + H)` | `O(N + H)` |
| topology payload | ≈ full halo | `O(M + ΔH + J)` | `O(N + H)` via `BODY_SNAPSHOT` |
| pool work | `O(N + H)` per object | `O((M + ΔH + J) · valence)` | `O(N + H)`, two bulk memcpys |
| derived rebuild | full `BuildDerived` | inline topology; **no frames** | inline topology; **no frames** |
| ghost positions | `O(H)` + request/reply | `O(H)`, one cached push | `O(H)`, one cached push |
| body loops | `O(B_global)` | `O(B_local)` | `O(B_local)` |
| neighbour-path arrays | often `O(P)` | `O(d)` | `O(d)` |
| collectives/step | 7–8 | 2 (NBX) – 4 | 2 (NBX) – 4 |
| steady allocations | every call | zero | zero |

Remaining non-scalable terms, stated honestly: the path-selection Allreduce if a non-NBX backend is
chosen; `O(P)` count metadata when the global path actually runs; occasional geometric pool
reallocation; periodic re-Mortonisation at `O(N log N)` amortised over `K` steps; ghost position bytes
proportional to the halo; and the surface-to-volume limit on strong scaling.

---

# Part III — Revised phase plan (F14)

Each phase ships a measurable win. Risk rises monotonically, and the widest-blast-radius change is
last and optional.

| # | Content | Risk | Exit gate |
|---|---|---|---|
| **0** | Phase timers and counters inside the **current** implementation; per-phase breakdown in `parallel_advection_scaling.c`; add a one-large-mesh configuration; resolve the mixed `results/scaling/` store | none | the breakdown reconciles with the existing `t_redist` bucket; §7 numbers reproduce |
| **1** | Persistent ensemble workspace; **sparse GID tables**; neighbour-slot indexing; direct-pack; geometric growth in `MFTLObjectSetCounts` | low — `src/communication/` + one line in `object.c` | zero steady allocations; GID-table bytes `O(local)`; peak RSS drops; all tests unchanged |
| **2** | Slot primitives + randomized property tests, behind the old algorithm | low | random operation sequences preserve every pool array, prefixes, GID maps, twins and derived connectivity against a full `MFTLObjectBuildDerived` reference |
| **3** | `ensemble->active[]` and active-set iteration (**F2/B1**) | low | no communication path scans `B_global`; every collective unchanged |
| **4** | One wire protocol incl. `BODY_SNAPSHOT`; transport seam; `exchange.c` stops flattening | medium | all transports commit identical state from identical action streams, incl. zero-byte ranks |
| **5** | Delta planning; `Rold`/`Rnew`; drop the kept-list/`rebuild_object` flow; postcondition change (F8) | medium | a step with no owner or replica change sends zero topology bytes and moves zero slots; DELTA and SNAPSHOT commit identical state |
| **6** | Mutation journal; remeshing integration; renumbering remap (F10) | high | initial distribution, advection, remeshing and explicit owner maps all execute the same core; unjournalled mutation is detected in Debug |
| **7** | Cached ghost push + consistency audit (F4) | medium | one data exchange, no request phase, no steady allocation; audit catches an injected desync |
| **8** | Locality: disorder metric, in-commit re-Mortonisation, long-run drift test (F7) | low | geometry-pass ns/tri is flat over 10^4 steps |
| **9** | Transport measurement incl. NBX (F9) | low | production default documented with evidence; any retained backend has a demonstrated regime |
| **10** | Sparse identity (**F2/B2**) — *only if the RSS number justifies it* | very high | a body is held only on ranks intersecting it; an empty rank can receive one |
| **11** | Remove the old implementation | — | one production algorithm |

During development the old implementation lives behind a **test-only** build option as a differential
oracle, and is removed from the production library at Phase 11.

---

# Part IV — Validation (F15)

## Named gates from the existing suite

- **`tests/test_bvh_ghosting.c`** at np = 1, 2, 4, 8 and **27**. Multiset equality of gathered
  `(gid_A, gid_B)` contact pairs against a pipeline run before partitioning, with per-row non-vacuity
  assertions. `CLAUDE.md` calls it *"the one test that finds ghosting bugs without knowing what they
  are"*; it separates missing / duplicate / spurious / divergent at once. **Primary gate for F1, F4
  and any halo delta.**
- **`test_every_ghost_agrees_with_its_owner`** (bitwise) and **`frozen_matches_replication`**. The
  "replicated but not frozen" direction is the fatal one, and `MFTL_FLAG_FROZEN` is derived from the
  ownership prefix and the halo rule — so any change to who holds what surfaces here.
- **`test_the_two_strategies_reach_the_same_mesh`**. The two remeshing strategies share no code past
  `MFTLRemeshIterate`, so it is evidence about the answer rather than one implementation.
- **`inv_partition_is_sound` / `inv_halo_is_vertex_complete`**. Vertex-completeness is what makes
  `MFTLComputeVertexNormals` communication-free; a delta that under-sends breaks it silently.
- **`test_twin_repair_matches_full_rebuild`**. Re-runs `MFTLRebuildTwins` after a redistribution and
  asserts nothing changes — the direct gate for F5.
- **`test_remeshing_mpi_is_confined`** (a CMake source check). The journal must not put MPI outside
  `src/remeshing/remesh_comm.c`.
- **`examples/leveque_remeshing_parallel.c`**. Bitwise determinism across runs, which both F7's
  re-Mortonisation and F1's dual record forms threaten.
- **`tests/test_object.c` under `--debug`**. It writes the last addressable element of every pool
  array, so any array not grown with the rest is an ASan failure — the gate for the geometric-growth
  change and the new per-object arrays.

## New fixtures

No ownership change; face / edge / corner crossings; simultaneous opposite crossings; handoff to a
rank already holding a ghost; vertex handoff with several copy ranks; an old owner that stays a ghost
holder; a body leaving a rank entirely; a body arriving on a previously empty rank; a rank with an
empty local ensemble entering the collective; periodic seams; walled boundaries; a destination outside
the neighbour graph; rank-zero initial distribution; explicit owner maps; remeshing additions and
tombstones; renumbering across a generation break; zero-length messages; zero-neighbour cases; and
**DELTA versus `BODY_SNAPSHOT` committing identical state on every one of the above**.

## Differential comparison, old versus new

By body and entity global ID: owner ranks; owned and ghost sets; triangle replica sets; vertex
completeness; positions and flags; triangle vertex order; half-edge flags and twins; `out_he_idx`;
`tris.verts` and `tris.neighbors`; global counts; copy-rank membership. Geometry is compared only
where the postcondition still promises it (F8).

## Performance acceptance

Topology-stable step with unchanged ownership: zero topology bytes, zero pool swaps, zero derived
rebuilds, zero communication allocations after warm-up, one cached field exchange.

Step with migration: topology bytes `O(M + ΔH)` in the advection regime; pool work proportional to
changed entities times valence; no complete kept-entity copy; no storage or scan proportional to
`B_global`; neighbour-path count work `O(d)`; identical results through every legal transport.

Weak scaling at fixed local size and migration fraction: local planning and apply time flat; bytes per
rank flat; active-object metadata flat; any residual growth attributable to the transport or the
path-selection collective, not to local reconstruction.

Long-run: geometry-pass ns/triangle flat over 10^4 steps (F7).

---

# Part V — Invariants this changes

Nothing here may move silently. Each row needs an explicit decision and a `CLAUDE.md` update.

| Invariant today | Change | Why | Finding |
|---|---|---|---|
| Bodies are identified on the wire by **list index**; every rank holds the same list in the same order | → body **global ID**; lists may differ | scaling at `B_global` | F2 (B2 only) |
| Redistribution's postcondition sets **both** derived bits | → `FRAMES` left clear | 72 B/tri of writes for a ~1 % downstream gain | F8 |
| **There are no hash maps** | → one open-addressed GID→slot table per entity class | ids are dense mesh-wide but a rank holds `1/P` of them; a direct-address array is `P`× oversized | Part II |
| **Global IDs are a dense permutation of `[0, n)`** (`inv_global_ids_are_dense`) | → density becomes an optimisation, not a requirement | sparse tables remove the only reason to re-densify | F11 |
| Redistribution is **stateless** — every partition re-derived from replicated data | → persistent committed state, guarded by audit + snapshot resync | delta payload | F4 |
| Pool order derives deterministically from the previous order | → history-dependent under swap-with-last, re-Mortonised on a metric | `O(1)` erase | F7 |
| `MFTLRedistributeObject(MFTL_Object *)` is a collective | → removed, or re-expressed as ensemble + body GID | a rank without the body has no handle | F16 |
| `MFTLObjectSetCounts` reserves exactly | → geometric growth | present bug: full-block realloc on a one-entity jitter | F16 |

---

# Final recommendation

Adopt the original's core judgement — **one delta engine, one wire protocol, one parser, transports
measured rather than assumed** — with five substantive corrections:

1. **Add `BODY_SNAPSHOT`.** Without a bulk record form the design pessimises remeshing, which is
   96–99.5 % of a remeshed timestep and calls redistribution 4.8× per step. It also becomes the resync
   primitive that makes the persistent state safe.
2. **Do not remove empty objects yet.** Measure first: they cost 288 B/body/rank, the dispersed
   balancer requires replicated per-body metadata regardless, and eight collectives are indexed by
   list position. Ship `active[]` now; ship sparse identity only if the number justifies it.
3. **Name the statelessness trade and pay for it** with a consistency audit and snapshot resync. This
   is the largest risk in the proposal and it is currently unmentioned.
4. **Cut the edge map and the incidence lists.** `repair_twins` and the half-edge fan walk already do
   both jobs with no persistent state, and one of the two proposed structures violates a stated
   invariant.
5. **Address pool-order decay**, and **stop building frames**. The first is a cost nothing in the
   repo would currently detect; the second is a full pass over the pool, every timestep, for a
   measured ~1 %.

Plus one opportunity the original passes over: **NBX** removes the path-selection Allreduce it calls
irreducible, taking collectives per timestep from 7–8 to 2 — and at the measured payload sizes this
phase is latency-bound, so that is the figure of merit.

And one that follows for free: once GID tables are sparse, the reason remeshing re-densifies global
IDs after every pass disappears, along with an `O(V + T)` pass, an `MPI_Exscan` and a ghost exchange
per distributed remesh.

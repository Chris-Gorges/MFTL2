# Ghost synchronisation: what the pull costs, and what a push would buy

## An architecture study, and a companion review of the scalability update plan

Date: 2026-09-01

Final review update: 2026-09-01

This document answers two questions that arrived together.

1. Could `MFTLUpdateGhostPositions` be faster and more scalable as an owner-side
   **push** — every rank sending its own vertices outward — instead of the
   **pull** it is today, where the holder of a ghost asks its owner for a value?
   Specifically: is there a topological condition that narrows *which* local
   vertices need to be sent, and a positional one that narrows *which* of the 26
   neighbours they go to, far enough that a push wins?
2. Is [`scalability-update-implementation-plan.md`](scalability-update-implementation-plan.md)
   right about how to get there?

The short answers are that the topological condition **exists, is exact, and is
already implemented and under test** (§3); that the destination question is
**not** locally answerable and the gap is precise and reachable in ordinary
configurations (§4); that a broadcast to all neighbours has an unfavourable
14–17× wire-volume model on a full 3-D process grid but still deserves an
instrumented prototype because it removes a collective and the owner-side
request table (§5 and §7); and that an exact cached push remains the likely final
architecture (§8).

The final review changes one important ownership decision: the ghost schedule
should be owned and committed by a separate ghost-schedule module. Redistribution
may later provide replica events to that module, but schedule construction should
first be implemented as an independent holder-registration phase. Adding a
fourth redistribution payload is not free under the current exchange layer: it
posts another non-blocking data collective even though it shares the existing
count exchange.

The requested topological neighbour-broadcast experiment and its go/no-go gates
are specified separately in
[`send-only-ghost-sync-implementation-test-plan.md`](send-only-ghost-sync-implementation-test-plan.md).

The review of the plan is §10. It is a companion, not a rebuttal: the plan's
classification is sound and its phase ordering is right.

No implementation is made by this document.

## 1. What the pull costs, split by regime {#ghost_regimes}

`ghost_sync_vec3` (`src/communication/ghost_sync.c`) does nine things per
synchronised field: collect ghosts grouped by owner, exchange request counts,
exchange requests, group arrivals by object, scatter each requested object's
owned vertices into a GID→slot table, build replies, exchange replies, apply
them positionally, free the schedule. Three neighbourhood collectives, 40 bytes
per ghost vertex on the wire (8 out, 32 back).

The cost of that is not one number, because the two regimes this library serves
disagree about it:

| | dispersed: 1000 bodies, 1 280 000 tris, np = 4 | spanning: one body, 5 242 880 tris |
|---|---|---|
| ghost sync | 0.52 ms | 11.30 ms |
| whole timestep | 14.14 ms | ~79 ms |
| **share** | **3.7 %** | **~14 %** |

Sources: [`performance-analysis.md`](performance-analysis.md) §7e and
[`scaling-analysis.md`](scaling-analysis.md) §3.

**The spanning number is the one that matters, because it is the one that goes
the wrong way.** Over the same P = 2 → 4 step on 5.24 M triangles, advection
scales 1.78×, the two geometry passes scale, redistribution is flat — and ghost
sync goes **5.456 → 11.297 ms**. It doubles while the per-rank share halves.
[`scaling-analysis.md`](scaling-analysis.md) §3 gives the cause: more cut planes
through one body grow the halo (0.13 → 0.31 % of the mesh), and this phase is
sized by the cut planes rather than by the per-rank share.

The ceiling on any change here is therefore about **2 % of a timestep** in the
dispersed regime and something like **7–10 %** in the spanning one. That
asymmetry decides everything below: an architecture that wins in the dispersed
case and is neutral for spanning is not worth the risk, and one that wins for
spanning may spend a little in the dispersed case.

## 2. The halo rule, restated as an information problem {#ghost_information}

Redistribution sends a triangle to the owner of each of its three corners and to
the owner of each of its three edge neighbours (`collect_destinations`,
`src/communication/redistribute.c:451-473`), and every triangle carries its three
vertices. So a rank holds vertex `v` iff it holds some triangle incident to `v`,
and for `v` owned by rank `A`:

```
holders(v) = union over t incident to v of:

    { owner(t) }  ∪  { owner(c) : c a corner of t }  ∪  { owner(n) : n an edge neighbour of t }
```

A push needs two things the pull never asks for:

- **WHICH** of `A`'s owned vertices are held by somebody else at all;
- **WHERE** each of them goes.

These have different answers, and conflating them is the trap. The comment at
the head of `ghost_sync.c` states the pull's justification as *"Only the HOLDER
of a ghost knows that it holds it; the owner has no record of who copied its
vertex."* The first half of that sentence is false and the second half is true.

## 3. WHICH: exactly derivable, no communication, already in the tree {#ghost_which}

An owned vertex is replicated somewhere **if and only if** it carries
`MFTL_FLAG_FROZEN`:

```
triangle t (owned) is replicated  ⟺  a corner of t is a ghost
                                     OR an edge neighbour of t is a ghost
vertex v is replicated            ⟺  some incident triangle is replicated
                                     (a triangle that is itself a ghost always counts)
```

With ownership as a prefix, "is a ghost" is `index >= n_local`, so the whole
classification is two streaming reads of 12 B/triangle and six integer compares —
`MFTLRemeshMarkFrozen` (`src/remeshing/remesh_parallel.c:176-249`).

**Proof, both directions.** `A` holds every `t ∈ T(v)`, because the halo is
vertex-complete (`inv_halo_is_vertex_complete`). If some `R ≠ A` holds `v`, then
`R` holds some `t ∈ T(v)`; that `t` is on `A` too, so it is either a ghost on `A`
(and then `v` is frozen by the ghost-triangle clause) or owned by `A` and held
elsewhere (and then `t` is replicated, so `v` is frozen). Conversely, if `v` is
not frozen then every incident triangle is owned by `A` with no ghost corner and
no ghost edge neighbour, so redistribution sent none of them anywhere, so nobody
else has `v`.

**This is not an argument that needs to be trusted: it is asserted in both
directions by an existing test.** `frozen_matches_replication`
(`tests/test_remeshing_parallel.c:411-473`) builds what every rank actually holds
as a ghost, `MPI_Allreduce`s it, and compares the result against the flag for
every owned entity — including the "replicated but not frozen" direction, which
is the fatal one.

So the proposed condition — *send every local vertex that has a ghost triangle in
its one ring* — is right, and it is stronger than "probably valid": it is an
exact characterisation with no false positives and no false negatives.

One correction to the phrasing. The rule is **not** "has a ghost triangle in its
one ring". An owned vertex all of whose incident triangles are owned can still be
replicated, when one of those triangles has a ghost **edge neighbour** — that
triangle travelled to the neighbour's owner and took `v` with it. Dropping that
clause loses sends, silently. The correct rule is the two-level one above.

Cost note: the classification is `O(owned triangles)`, which for the spanning
body at np = 4 is 1.3 M triangles × 24 B of streaming reads. That is not free —
it is the same order as the whole phase it would serve. §8 puts it somewhere it
already happens.

## 4. WHERE: not locally derivable, and the gap is precise {#ghost_where}

`A` knows `owner(t)` for every `t ∈ T(v)` (it is `owner_proc` on the record) and
`owner(c)` for every corner of every local triangle (vertex-completeness puts all
three corners in the pool, each carrying `owner_proc`). The edge-neighbour clause
splits in two:

- **`t` owned by `A` — exact.** All three edge neighbours of `t` are present on
  `A`, because `n` travels to the owner of each of *its* edge neighbours and `t`
  is one of them. So `A` can evaluate the full seven-way clause.
- **`t` a ghost on `A` — incomplete.** `t`'s edge neighbours need not be present
  at all; `tris.neighbors[3t+k]` reads `MFTL_ERROR` and a holder is missed.

**A counter-example, and it is an ordinary boundary configuration.** Vertex
ownership is decided by position and triangle ownership by centroid, so the two
differ across a cut plane routinely. Take `v` owned by `A`; a triangle `t`
incident to `v` owned by `B`; and `n`, adjacent to `t` across the edge *opposite*
`v`, owned by `C`. Then:

- `C` receives `t` under the edge-neighbour clause, so **`C` holds `v`**;
- `A` receives `t` under the corner clause, so `A` sees `t`;
- `A` has no reason to hold `n` — `n` is not incident to `v`, `A` does not own
  it, and `n` travels to `owner(t) = B` rather than to `A` — so **`A` cannot name
  `C`**.

The failure mode is a **missing send**, not an over-send. `ghost_sync.c:229-241`
already documents where a missing update goes: the ghost keeps a stale position,
this rank and the owner compute different centres for the same triangle,
disagree about its destination at the next redistribution, and someone's owned
triangle quietly loses an edge neighbour. Silent and downstream.

**Conclusion: a stateless owner push needs a provable superset of
`holders(v)`.** The next two sections price the two candidates.

## 5. Superset 1 — broadcast to all neighbours {#ghost_broadcast}

Let `F` be the number of frozen owned vertices on a rank, `G` the number of ghost
copies of that rank's vertices held anywhere, and `m = G/F` the mean replication
multiplicity. `m` is small and dominated by process-box **face** vertices, which
are held by exactly one other rank; box-edge vertices are held by 3 and box-corner
vertices by 7, and there are far fewer of them. `m ≈ 1.2–1.5`.

| | wire per rank |
|---|---|
| pull (today) | `40 B × G` — 8 B request out, 32 B reply back |
| broadcast | `32 B × nn × F` — the GID must ride along, because the receiver has to identify what arrived |

The ratio is `32·nn·F / (40·m·F)` = **`0.8 · nn / m`**:

| process grid | `nn` | bytes vs today |
|---|---|---|
| `4×1×1` slab | 2 | **1.1–1.3×** |
| `2×2×2` (periodic dedup collapses ±1 per axis) | 7 | 4–5× |
| large 3-D grid | 26 | **14–17×** |

Receiver-side work moves the same way: every neighbour tests `F` arrivals against
its own ghost set instead of receiving `G/nn` matched records, so the apply phase
grows by the same factor.

**Verdict: viable only in slab and pencil decompositions, which is the opposite
of where scaling is wanted.**

There is a trap worth naming, because it would produce a convincing and wrong
benchmark. The anti-scaling spanning rows in
[`scaling-analysis.md`](scaling-analysis.md) §3 were collected on a `4×1×1` grid,
where `nn = 2`. A broadcast would measure at parity there and collapse at the
grid shape a production run has. §5 of that same document already shows the shape
is not a detail — pinning `2×2×1` against `4×1×1` costs the spanning case 1.09×.

## 6. Superset 2 — the geometric skin stencil {#ghost_skin}

The second half of the question — *narrow the destinations by where the vertex
sits in the owning rank's box* — can be made **rigorous** rather than heuristic,
and this is the strongest form of the stateless push.

**Claim.** `R ∈ holders(v)` ⟹ `R` owns an entity within distance `D = 2·h_max` of
`v`, where `h_max` is the longest edge in the body.

**Proof.** `R` holds `v` because it owns some entity in the seven-way clause of
some `t ∈ T(v)`. If `R = owner(t)`, its centroid is within `diam(t)` of the corner
`v`. If `R` owns a corner `c` of `t`, `|c − v| ≤ diam(t)`. If `R` owns an edge
neighbour `n` of `t`, then every point of the shared edge is within `diam(t)` of
`v` (both its endpoints are corners of `t`), and `n`'s centroid and corners are
within `diam(n)` of that edge. A triangle's diameter is its longest edge, so
every case is bounded by `diam(t) + diam(n) ≤ 2·h_max`. ∎

`h_max` costs nothing: `MFTL_ObjectQualityStatistics.max_edge_length_sq`
(`include/mftl_types.h:275`) is already reduced globally per body by
`MFTLEnsembleUpdate`.

The test costs three axes of arithmetic on fields already on `MFTL_Domain`. With
`f_k = (p_k − global_min_k) · grid_factors_k`, `c_k = (int)f_k` and
`frac = f_k − c_k`, cell `c_k − 1` is a candidate iff `frac < D · grid_factors_k`
and `c_k + 1` iff `1 − frac < D · grid_factors_k`. The at most 27 resulting cells
resolve through `rank_map`. Branch-free, no allocation, no new state.

The over-send factor collapses to ≈ `m` when `D` is close to the true halo depth,
because "which neighbour cells are within `D` of `v`" is very nearly the same
question as "which ranks hold a triangle incident to `v`" — a face vertex names
one neighbour, a box-edge vertex three, a box-corner vertex seven. So the wire
cost lands near **0.8× today's, even at `nn = 26`**, and one of the three
collectives goes away.

**Three defects, and together they rule it out as the production path.**

1. **`D = 2·h_max` is a global maximum over the body, so one large triangle
   widens the band everywhere.** On a remeshed mesh the stretch bound keeps
   `h_max/h_mean` bounded; on a mesh that is only advected, nothing does.
2. **The order of operations makes the test stale.** The documented per-timestep
   loop runs `MFTLUpdateGhostPositions` **before** `MFTLRedistribute` — on the
   partition committed last step, with positions already moved this step. So the
   test must be evaluated with `D + dt·v_max`, a caller-supplied motion bound. It
   is the same precondition as deferred issue 21 in `CLAUDE.md`, with the same
   property: a solver that understates its bound gets a silently wrong answer,
   which here means a stale ghost rather than a wrong partition.
3. **A conservative superset forces the receiver to identify each record.** That
   puts the GID back on the wire (32 B rather than 24 B) and needs a
   GID→ghost-slot lookup on arrival — or a per-peer sorted two-cursor merge,
   which is itself a cached per-peer ordering, i.e. a schedule by another name.

Two implementation qualifications are also required. The current
`MFTLRemeshMarkFrozen()` is not directly reusable as the ghost-send predicate:
it writes shared entity flags, stamps `partition_dirty`, and its result may later
be dilated for remeshing. Ghost sync needs the same exact base classification in
a private bitset without those side effects. In addition, the at-most-27-cell
geometric stencil is valid only while the conservative distance is smaller than
one process-cell width per axis. A general implementation must enumerate every
cell covered by the distance, handle periodic wrapping, or fail an explicit
bound and select a global path.

Point 3 generalises, and it is the central result of this study:

> **Every push that is not exact re-introduces a lookup at the far end. Only the
> exact schedule costs nothing on either side.**

## 7. The collective count is the other axis, and it is not 3 → 1 {#ghost_collectives}

A push does not automatically become one message. The receiver of an
`MPI_Neighbor_alltoallv` must supply exact receive counts, and in a stateless
push it does not know them.

| architecture | neighbourhood collectives |
|---|---|
| pull (today) | **3** — `Ineighbor_alltoall` counts, `Neighbor_alltoallv` requests, `Neighbor_alltoallv` replies |
| stateless push (§5 or §6) | **2** — the count exchange survives |
| cached push | **1** — counts are committed on both sides |

So the stateless push saves one collective of three; caching saves two of three.
At `P` in the hundreds with `nn = 26` the fixed cost of a neighbourhood
collective is what dominates a small payload — which is precisely the spanning
body's regime, where the payload per rank is a thin band and the phase is
already anti-scaling.

## 8. Final recommendation: separate ownership, measured broadcast, then an exact schedule {#ghost_recommendation}

The likely final data path is still a one-way owner→holder push over an exact
schedule. The final review changes who owns and initially builds that schedule:

- a dedicated ghost-schedule module owns entries, generations, buffers and
  validation;
- redistribution invalidates or advances a committed partition generation;
- the first implementation builds the exact schedule in a separate collective
  holder-registration phase;
- only after measurement may redistribution feed replica-membership events to
  the schedule module as an optimisation.

Before that cached design, the topological all-neighbour broadcast described in
§5 will be implemented as an experimental backend. Its cost model is poor for a
full 3-D neighbourhood, but a measurement is still justified: it replaces the
three-stage pull with a two-stage push and removes the owner-side request
grouping and owned-vertex GID scatter. The implementation and test gates are in
[`send-only-ghost-sync-implementation-test-plan.md`](send-only-ghost-sync-implementation-test-plan.md).

### Exact schedule variant A: separate holder registration -- initial design

After a successful redistribution, or lazily at the first field sync after it:

1. each holder scans its ghost suffix and groups `(body_gid, vertex_gid)` by
   owner;
2. a count exchange and registration exchange deliver those exact subscriptions;
3. the holder retains its ghost slots in registration order;
4. the owner resolves each registration once and retains its owned slots in the
   matching receive order;
5. the reversed counts and order become the owner-to-holder push schedule.

This keeps schedule construction completely outside redistribution. Rebuilding
the schedule and pushing one field costs the same three collective calls as the
current pull, but moves 32 rather than 40 bytes per ghost relation and makes every
additional field update one 24-byte exchange. It is also lazy: a caller that
does not request a ghost field pays no schedule-build cost.

### Exact schedule variant B: redistribution event source -- optional optimisation

Redistribution temporarily knows the exact replica relation, so it can later
emit replica events to the separate schedule module. The useful facts are below,
but they do not justify putting schedule state or policy into `redistribute.c`.

**Holder-side discovery needs no communication.** After `rebuild_object` the ghost vertex suffix **is** the
receive schedule: group `[verts.n_local, verts.n)` by `owner_proc` and record the
slots. `O(ghosts)`, no communication. This is the same work `ghost_sync.c` step 1
does today (`ghost_sync.c:215-250`), moved from once-per-field-per-step to
once-per-redistribution.

**Owner-side membership is available during destination analysis.** The missing
information from §4 — *which rank holds my vertex* — is known to the owner of
each **triangle**, not to the owner of the vertex. It must therefore be routed.
The routing can reuse destinations already discovered, for a reason worth
stating carefully:

> When rank `S` decides to send triangle `t` to destination `dst`, it emits a
> subscription `(vertex_gid, dst)` addressed to `vert_owner[v]` for each corner
> `v` of `t`. **`vert_owner[v]` is provably already in `dests[]`**, because
> `collect_destinations` puts the owner of every corner into the candidate set
> (`redistribute.c:452,467-469`). So the subscription travels on a message `S` is
> already sending — to a *different* peer from the one that occasioned it.

The emission point is exactly the existing de-duplication block at
`redistribute.c:510-524`, where `mask[v] & m` already answers "have I already
listed `v` for `dst`". Volume is 8 B per `(ghost vertex, holder)` pair globally —
identical to the current request phase's volume. The earlier claim that the
fourth payload slot is free is incorrect under the current exchange
implementation. The count exchange is shared across payloads, but
`MFTLExchangeStart()` posts one `MPI_Ineighbor_alltoallv` or `MPI_Ialltoallv` per
payload, so a fourth payload adds a fourth data collective. It may overlap with
the other payloads; it is not the same message. A truly fused form would have to
extend an existing wire record or tagged stream and must be measured separately.

Two details the emission needs. Duplicates survive **across** senders — two ranks
owning different triangles incident to `v` both send to `dst` — so the owner
de-duplicates on arrival. And the `MFTL_DEDUP_NO_BIT` overflow path (above 32
distinct destinations) does not set the mask, so it produces same-sender
duplicates too; the same arrival-side dedup covers both.

**Resolution can reuse existing redistribution work, and this may be the largest
saving.** The owner must turn
subscription GIDs into its own final slots. That table already exists inside
redistribution: `rebuild_object` builds `s->v_tbl` over kept plus received
vertices and `v_final_of_logical` over the final ordering
(`redistribute.c:1380-1402`, used at `1686`). **The `O(owned vertices)` GID
scatter that `answer_object_requests` rebuilds on every ghost-sync call
(`ghost_sync.c:138-139`) is a duplicate of work redistribution has already
done.** Reusing it removes an `O(owned mesh)` pass from a phase whose payload is
`O(halo)` — ~650 k `gt_put` per call for the spanning body at np = 4.

**The steady call becomes:** pack 3 doubles per entry from cached source slots →
one `MPI_Neighbor_alltoallv` with cached counts into persistent buffers → write 3
doubles per entry to cached destination slots. **24 B per ghost, one collective,
no table on either side.** Plain `MPI_Neighbor_alltoallv`, so no MPI-4
dependency; `MPI_Neighbor_alltoallv_init` + `MPI_Start` drops in behind the same
schedule later if the target cluster ever makes that assumable.

### Both regimes, which is the constraint

**Dispersed.** In the incremental or redistribution-event variant, the schedule
survives every skipped body, because
`MFTLRedistributeSetSkipUnchangedBodies` guarantees the pool is bit-identical
**slot for slot**, not merely equal as a set — that is what
`test_skipping_reproduces_the_full_rebuild` compares, and the order digest is the
half that caught the non-idempotence. At the measured 77–86 % clean-body
fraction, most bodies rebuild no schedule at all. Trust a schedule only after the
settling rebuild that the existing "clean twice" rule already provides. The
ceiling on the win here is ~2 % of a timestep, so the production design must add
nothing material to the clean path. A standalone full registration must be
measured separately because it may still rebuild global schedule state.

**Spanning.** One body, cut by planes, dirty every step: **zero amortisation
across steps.** An event-fed exact schedule can still win in principle, for
reasons that have nothing to do with amortisation:

- the `O(owned)` GID scatter is *reused* from redistribution rather than rebuilt;
- the steady field update removes two of three neighbourhood collectives;
- the payload goes 40 → 24 B per ghost.

**This is the direct answer to deferred issue 5 in `CLAUDE.md`.** Its premise —
*"the per-timestep loop this library is built around redistributes every step and
there would be nothing to amortise over"* — is correct, and it does not settle
the question, because the saving is a **relocation** of work into a pass that
already holds the inputs in registers, not an amortisation over steps.

**What dispersed pays for spanning in the optional event-fed variant:** one
predicted branch and one 8-byte append inside `collect_destinations`' dedup block,
on band triangles only. Target: below
1 % of the redistribution phase. That loop deserves the caution — `CLAUDE.md`
records a **1.70×** regression from a previous modification to `compute_owners`,
and the same discipline applies (hoist the decision per body, keep the inner loop
shape).

### What must be kept, and what must invalidate

**Keep the pull**, for two reasons rather than one. It is the differential oracle
the acceptance gate needs — positions are doubles copied verbatim, so
bit-identity is achievable and is the right gate. It is also the **only side that
can detect an orphan**: a ghost whose owner is not a graph neighbour is visible
to the holder (`ghost_sync.c:224-244`) and invisible to the owner. A push must
name its equivalent signal — an undeliverable subscription — or deferred issue 19
gets quieter rather than fixed.

**Invalidate the schedule** when membership, GIDs or integer slots can change.
`MFTLObjectMarkPartitionDirty` is an existing conservative choke point for count,
GID and flag changes, but a schedule needs an explicit generation rather than
silently borrowing one Boolean. `MFTLObjectReorderMorton` invalidates integer
slots and must advance that generation. `MFTLObjectShrinkToFit` does **not**:
it reallocates storage while preserving counts, contents and slot ordering, so it
is safe for schedules that store integer slots rather than raw pointers.

A full snapshot also needs an explicit clear/removal rule. Positive
subscriptions alone cannot delete a holder that disappeared in the new
partition. A separate full registration build naturally clears the old schedule.
An incremental or redistribution-fed design must carry body-scope snapshots or
signed reference removals and must commit them transactionally.

## 9. Summary of the four architectures {#ghost_summary}

| | collectives | bytes/ghost | owner-side table | holder-side table | persistent state |
|---|---|---|---|---|---|
| pull (today) | 3 | 40 | `O(owned)` per call | none | none |
| broadcast to `nn` | 2 | `32·nn/m` ⇒ 640 at `nn=26` | none | `O(nn·F)` tests | none |
| geometric skin push | 2 | ~32 | none | `O(received)` lookup | none, but needs `dt·v_max` from the caller |
| **schedule push** | **1** | **24** | **none** | **none** | schedule + generation |

The schedule row describes a field update after a valid schedule exists. A
separate exact schedule rebuild costs a count exchange plus a registration
exchange. Those costs must be included whenever a benchmark rebuilds the
schedule every timestep.

## 10. Review of `scalability-update-implementation-plan.md` {#ghost_plan_review}

The classification in Finding 3 is good work and should survive unchanged. The
refusal to replace the dirty-body `MPI_BOR` with one round of neighbour flags is
correct and correctly argued (a spanning body has holders more than one graph hop
apart). Choosing a hash home over a spatial home is right, and for the right
reason. Bit-identity as the acceptance gate for a field push is the right gate.

Thirteen amendments.

**R1 — Finding 4 does not say which regime it pays in, and they disagree.** §1
above: 3.7 % of a timestep in the dispersed case against ~14 % and anti-scaling
in the spanning one. The plan reads as though the win were uniform, and the
"cached ghost schedule snapshot" work package is priced at 6/10 difficulty
without that split. State the ceiling per regime.

**R2 — the subscription routing in "Schedule construction on the current
full-rebuild core", step 3, is incomplete, and making it explicit is what makes
the design credible.** The step says *piggyback subscription snapshots in the
redistribution exchange … the current multi-body exchange uses three of the four
supported payload slots*. The proposed record already contains `vertex_owner`,
so it may intend the correct route, but it does not state it. A subscription
`(vertex_gid, holder)` is produced by the owner of the **triangle**
and must reach the owner of the **vertex**, which is not the destination `dst`
that occasioned it — so it does not fit in the message to `dst`. It fits in a
message that is already being sent, because `owner(v)` is provably in `dests[]`
(§8). That proof is the load-bearing step and the plan omits it. The plan also
does not mention that duplicates survive across senders and need an arrival-side
dedup.

**R3 — the largest per-step cost the change removes is not on the wire, and the
plan never ranks it.** `answer_object_requests` scatters **all** `n_local` owned
vertices into a GID table on every call, for every body that received any request
(`ghost_sync.c:138-139`) — `O(owned mesh)` per field per step against an
`O(halo)` payload. The acceptance gate *"No steady-path GID-table build occurs"*
names the symptom; the classification never says this is the item, and §8 shows
the table already exists inside redistribution.

**R4 — two per-call allocations in `ghost_sync.c` that the Finding-3 table lists
for redistribution but not here.** `ghost_sync.c:210-212` mallocs
`rank_to_neigh[nproc]` and writes all `nproc` entries **on every call, per field,
per step**; `ghost_sync.c:324` and `:340` do `calloc(num_objects+1)` and
`malloc(num_objects)` likewise. The communication graph is fixed for the domain's
life, so `rank_to_neigh` belongs on `MFTL_Domain`, and the grouping scratch
belongs in the workspace beside the GID table that was moved there for exactly
this reason (`ghost_sync.c:359-364`). **Independent of everything else in the
plan and fixable today** — see §11.

**R5 — the schedule entry is over-fat.** `body_gid`, `vertex_gid`, source slot and
holder rank is 16 B/entry. Grouped by peer and then by body, `body_gid` and the
peer are *headers*; the production entry is one `int`. Keep the GIDs in a parallel
Debug-only array so the verification the plan wants survives without costing the
pack loop its cache footprint.

**R6 — invalidation needs an explicit generation and a precise operation list.**
*"Any pool operation that may reorder vertices invalidates cached slots"* is the
right rule. `MFTLObjectMarkPartitionDirty` is the existing conservative stamp for
much of this class, while `MFTLObjectReorderMorton` must also invalidate cached
integer slots. The earlier review incorrectly included
`MFTLObjectShrinkToFit`: it moves storage but preserves slots, and is safe when
the schedule stores indices rather than raw pointers.

**R7 — the plan inherits `orphaned` (deferred issue 19) and does not say which
side reports it.** In a pull the holder detects it; in a push the owner is the
side that would have to, and a locally-derived destination set omits such a
holder silently. Designable, but it has to be designed.

**R8 — Phase 1 (the active pointer view) is unconditional work whose value the
plan's own exit gate defers to measurement.** *"Measurements show whether
empty-object iteration was material"* is an exit gate for work already done. Make
Phase 1 conditional on the Phase 0 measurement of `B_local/B_global`; in the
measured dispersed case every rank holds a share of most bodies, so the empty set
is small.

**R9 — Phases 5–8 are unschedulable on the stated hardware, and the plan does not
say so.** Deferred issues 10, 11 and 18 in `CLAUDE.md` all record that nothing is
measured above np = 27 or four performance cores, and the plan's own Finding-3
gate is *"raise `P` while holding `d` fixed and observe constant workspace
bytes"*. Cluster access belongs in the Phase 0 exit gate, or the four-to-eight
person-month estimate describes a schedule with no blocking dependency, which is
not the schedule that exists.

**R10 — a fourth exchange payload is not a free piggyback.** The current
`MFTLExchangeStart()` shares one count exchange across its payloads but posts a
separate non-blocking all-to-all-v for every payload. The plan must count a
subscription payload as another data collective unless it is genuinely encoded
inside an existing wire stream.

**R11 — schedule ownership and schedule information sourcing must be separated.**
The ghost-schedule module should own schedule state and commit it. A standalone
holder-registration builder is the clearest first implementation. Redistribution
may later emit membership events into that module without becoming responsible
for ghost-sync policy or storage.

**R12 — snapshots need removal semantics.** Re-emitting only current positive
subscriptions leaves stale owner-to-holder entries after a copy disappears. A
full rebuild must clear the old body scope before commit; a delta design must
carry signed references and ownership-handoff state.

**R13 — reuse the replication predicate, not the current remeshing side effects.**
The base frozen predicate is the exact topological answer to WHICH, but
`MFTLRemeshMarkFrozen()` currently writes flags, marks the partition dirty and is
followed by optional dilation. If the broadcast experiment wins, extract a pure
replication classifier that can write either a private ghost-send bitset or the
remesher's base frozen flags. Do not add a Boolean such as `mark_dirty=false` to
the current public routine; keep the pure classification and remeshing policy as
separate layers.

## 11. Two changes that are independent of all of the above {#ghost_do_now}

Neither is implemented. Both are small, both are behaviour-neutral, and neither
requires committing to a push.

1. **Move `rank_to_neigh` to `MFTL_Domain`.** The graph is built once at ensemble
   creation and never changes; `ghost_sync.c` rebuilds an `nproc`-length inverse
   of it on every call. At `P = 10⁵` that is a 400 KB allocation and fill per
   field per timestep, for information that is constant.
2. **Move the request-grouping scratch (`obj_off`, `cursor`) into the existing
   `comm_workspace`.** They are `O(num_objects)` per call, next to a GID table
   that was already moved there because *"building and releasing one per TIMESTEP
   was the last per-step allocation on either path"* (`ghost_sync.c:359-364`) —
   which is no longer true while these two remain.

## 12. How the claims here can be checked {#ghost_verification}

- **§3 (the frozen ⟺ replicated equivalence)** — `python3 build_all.py --debug`
  then `ctest --output-on-failure -R remeshing_parallel`. That is the claim, and
  it is already a gate.
- **§4 (the destination gap)** — constructible as a test, and worth building
  before anything is written: on a partitioned sphere, compute the
  locally-derivable holder set for each owned vertex and compare it against the
  `MPI_Allreduce`d truth `frozen_matches_replication` already assembles; assert
  it is a strict subset for at least one vertex at np ≥ 4. If it never is, §4 is
  wrong, the geometric superset of §6 is unnecessary, and the stateless push
  becomes much more attractive.
- **§5 and §7 (the cost models)** — arithmetic over
  `sizeof(MFTL_GhostUpdatePayload) == 32` (`include/mftl_communication.h:204`)
  and the two-int request at `ghost_sync.c:246`. The multiplicity `m` is
  measurable today: `MFTLCommStats.g_n_ghosts_requested` against a frozen-vertex
  count.
- **§8's main non-wire argument (R3's magnitude)** — build with
  `-DMFTL_COMM_PROFILE=ON` and read `g_t_table` against `g_t_total` from
  `examples/parallel_advection_scaling.c` on the spanning case. **If the GID
  scatter is not a material share of the phase, the recommendation loses its
  largest claimed saving** and this document must be amended rather than the code
  written.
- **The topological neighbour broadcast** — compare it bit-for-bit with the pull
  on the same committed partition, then compare maximum-rank phase time,
  end-to-end timestep time, bytes, candidate-to-match ratio and scaling slope
  across slab, pencil and 3-D grids. A win on the current `4x1x1` laptop grid is
  not sufficient evidence for production because that grid has only two
  neighbours. The exact matrix and stop gates are in
  [`send-only-ghost-sync-implementation-test-plan.md`](send-only-ghost-sync-implementation-test-plan.md).

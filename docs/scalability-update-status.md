# The scalability update: what changed, and what is still missing

**Status document, 2026-09-04.** Written to be read start to finish by somebody
who did not follow the work. It says in words what was wrong, what was done about
it, what that measurably bought, and — at the end, at length — what is still
broken. The numbers behind every claim are in
[`scaling-analysis.md`](scaling-analysis.md); the invariants the change
established are in `CLAUDE.md` and in the `src/*/README.md` chapters. Nothing here
is a plan: everything described as done is in the tree and covered by tests.

**Second pass, same day.** Stages 0–3 below are unchanged. What was written up as
the two largest remaining items — contact detection still enumerating every body
pair, and the remesher's per-body collectives ported rather than fixed — is now
**Stage 4**, and §6 says what genuinely remains.

Three things this document itself had wrong are corrected in place:

1. it said contact detection and the spatial halo *genuinely need* every body,
   "because their request lists must be identical on every rank". **They do not**,
   and the reason is in Stage 4. The claim had also been copied into
   `mftl_ensemble.h`, `mftl_types.h`, `mftl_communication.h`, `mftl_bvh.h` and
   `src/ensemble/ensemble.c`, and is corrected in all of them;
2. §5's API break was being violated inside this repository, by a benchmark, for
   the whole of Stages 0–3 — see "the bug the benchmarks caught and the tests did
   not";
3. `MFTLRemeshSyncGhostIds` documented its argument as `[n_local_objects]` while
   the code indexed it by body id. The code now matches the contract.

`MFTLEnsembleAcquireDenseView` consequently has **no caller left in the library**.
It is kept for a caller that genuinely needs every body; §6 says when to delete
it.

**Third pass, 2026-09-04, after a review against the tree.** Two of Stage 4's
claims about the remesher were not true of the code, and both are now. They are
written up in full as **Stage 5** below rather than folded into Stage 4, because
what makes them worth reading is not the fix — each is a one-line change — but
that the whole 48-entry suite passed at every rank count with both defects in
place, and every recorded mesh digest reproduced bit for bit:

1. the spanning driver's per-phase id scan was **pinned to
   `MFTL_BODY_REDUCE_SCOPE_GLOBAL`**, so the one call site the Stage 4 table
   names as its headline change was still staging and reducing
   `O(bodies in the ensemble)` on `comm_world`, once per phase;
2. the explicit owner maps every remeshing migration builds were still an array
   **indexed by body gid and sized by the id space** — the last
   `O(bodies in the ensemble)` allocation in the library that is not the
   dispersed balancer, and it is on the per-phase path. It is item 4 of the
   implementation plan's own remeshing checklist, and it had not been done.

§6 is corrected accordingly: the owner maps leave the "what is still missing"
list, and a new smaller item takes their place.

---

## 1. The problem in one paragraph

MFTL used to require that **every rank hold an `MFTL_Object` for every body in the
ensemble**, because a body was identified on the wire by its *position* in
`MFTL_Ensemble.objects` and an `MPI_Allreduce` aligns element *k* of every rank's
buffer. So a rank kept a 264-byte object plus a pointer for every droplet in the
simulation, entered every per-body loop for every droplet, and appeared in every
per-body collective for every droplet — whether or not it held a single triangle
of it.

That is affordable when there are ten bodies. It is the first thing that breaks
when there are a million. In a **weak-scaling** sweep the body count grows with
the rank count, so all of that per-body cost grows **linearly with the number of
ranks** while the useful work per rank stays flat. At the design target — `P` in
the thousands to tens of thousands, `B` in the millions — a rank would have held
272 MB of empty object shells and reduced 88 MB of quality statistics per
timestep in order to learn about the hundred bodies it actually had.

## 2. What was done

Five stages, in an order chosen so that each one is separately testable.

### Stage 0 — the instrument, and the free wins

Nothing can be fixed that is not measured, and none of this was. The
redistribution's persistent workspace is now reported **split by what it is
proportional to** — `ws_bytes_per_body`, `ws_bytes_per_rank`,
`ws_bytes_per_entity` — as high-water marks rather than an allocation-churn sum,
so the figures are a footprint. The per-entity column is the control: weak
scaling holds work per rank constant, so that is what "flat" looks like, and
without it the other two columns say nothing. `n_objects_empty_local` counts the
bodies a rank paid for and held nothing of.

Two costs were removed outright while the instrument was going in. Mesh
generation used to make one `MPI_Bcast` and one `MPI_Allreduce` **per body, in
strict sequence**, on the one rank building everything; `MFTLGenerateSphereLocal`
plus `MFTLPublishGeneratedBodies` is two collectives for the whole ensemble
instead of `2N`. And `MFTLEnsembleAddObject` used to `realloc` the object list on
every single add, which is `O(N²)` bytes moved to build an ensemble; it now grows
geometrically.

### Stage 1 — identity moved from a list slot to a dense body id

`MFTL_Object.global_id` became **the** identity of a body. `MFTLEnsembleAddObject`
hands the ids out densely from a counter on the ensemble, exactly as vertex and
triangle ids are handed out, so a body id is an array index and needs no hash. The
three wire protocols that named bodies by list position — the redistribution count
payload, the ghost-sync request pairs, the remesher's id exchange — were re-keyed
to the id, and so was `MFTL_OwnerMap`.

**This stage deliberately changed no behaviour.** The list was still replicated,
so `global_id ==` the list position and every number was unchanged to the byte.
Splitting identity from storage is what made each half testable on its own: the
whole suite had to pass unmodified, and the only edits it needed were two
assertions that had hard-coded the *old* identity scheme.

### Stage 2 — the local object list became sparse

**A rank now holds an `MFTL_Object` exactly for the bodies it holds entities of.**
`MFTL_Ensemble.local_objects` is a subset **sorted by ascending body id**,
`n_local_objects` is what this rank holds, and `n_bodies_global` is the size of
the id space — a scalar, never an array.

The sort order is load-bearing three times over, which is why insert and erase
are ordered `memmove`s rather than the cheaper swap-with-tail: it makes
`id → slot` a **binary search** rather than a hash map (which the library forbids
without an argument); it makes local iteration order equal to the global id order
restricted to this rank, so every deterministic tie-break that says "bodies in
order" keeps meaning the same thing; and it lets arrival runs, which arrive
grouped, join against the directory with a merge instead of a search per record.

Redistribution grew a lifecycle to match: a body whose entities have all left is
**destroyed**, and one arriving where the rank held nothing is **created**. Both
sites already existed within a few lines — the "nothing kept and nothing
received" branch already released the pools — so the incremental cost is one
`malloc`/`free` of a 264-byte shell per body per rank-boundary crossing, against a
rebuild that was happening anyway.

Two modules were carried rather than redesigned *at this stage*. Contact detection
and the spatial halo were believed to need every body in the ensemble, because
their request lists had to be identical on every rank; they read through
`MFTLEnsembleAcquireDenseView`, a read-only `[n_bodies_global]` array in which an
absent body points at one shared, permanently empty object. The remesher got a
mechanical port: loops over local objects, staging arrays still
`[n_bodies_global]` but indexed by id. **Stage 4 removed both**, and the premise
turned out to be false: the request list never drove the *sequence* of
collectives, only their contents.

### Stage 3 — per-body reductions go to that body's holders

Three collectives in a timestep reduce one small record per body: the
redistribution skip vote (1 value), `MFTLEnsembleUpdateGlobalCounts` (2), and
`MFTLEnsembleUpdate`'s quality statistics (11). All three staged into a buffer
indexed by body id and called `MPI_Allreduce` on `comm_world`.

`MFTLEnsembleReduceBodies` (`src/ensemble/ensemble_reduce.c`) is now the one place
in the library where a per-body collective chooses its communicator. Two things
about it matter more than the collective itself:

**The caller's buffer is indexed by local slot** and reduced in place. Moving the
id-indexing *inside* the function is most of the saving — otherwise the message
shrinks and the per-rank memory does not. `MFTLEnsembleUpdate` used to allocate
`22 × bodies` doubles on every rank on every call — local and global min, max and
sum, i.e. 176 bytes per body — so **148 kB at 864 bodies and 176 MB at a
million**, to learn eleven numbers about the bodies it holds.

**The route is decided, not assumed.** A reduction over a body's holders is exact
only if every holder of every body is inside every other holder's
26-neighbourhood, and that is established by a purely local test: a rank checks
that each body it holds owns at least one entity here, that every other holder
named by its ghosts' `owner_proc` is a graph neighbour, and that those
neighbours' cell offsets span at most one cell per axis. The property is a
conjunction over ranks, so one `MPI_LAND` of a single int *is* the answer. The
test reads only the ghost suffix, so a body held whole costs nothing, and it is
re-derived on every call rather than cached — a cached verdict is invalidated by
any holder-set change anywhere in the job, and a stale one is a silently wrong
reduction rather than a crash.

### Stage 4 — the two modules that were carried, and the remesher's own collectives

Stage 3 built the mechanism. Stage 4 is what happens when the two places that
were exempted from it stop being exempt, and it removed the last
`O(bodies in the ensemble)` term from every per-timestep and per-phase path.

#### Contact detection: the request list was never information

`MFTLEnsembleIntersectObjects` enumerated `B*(B-1)/2` ordered pairs into an array,
on every rank, every call. At 16 384 bodies that is **134 million records and two
gigabytes** before a triangle is looked at, and it grows quadratically in exactly
the quantity the sparse list exists to stop paying linearly for. It also built one
BVH per body in the *ensemble* — a rank near ten droplets out of a million
allocated a million empty hierarchies per timestep.

The list was justified by "the request list drives a sequence of collectives and
so must be identical on every rank". That was **wrong**, and finding out why is
most of the change: the halo exchange's collectives are one neighbourhood
allgather of extents and one neighbourhood alltoallv of payload, and neither's
*count* depends on the requests. The requests only decide what a rank is willing
to **send**, which is a statement about that rank's own bodies.

So the pairs are derived rather than enumerated:

- a rank publishes one record per body **it holds** — `(body id, centre,
  half-width, global radius, contact envelope)`, nine doubles — through a
  neighbourhood allgatherv. The dense form published `6 × n_bodies_global` doubles
  to every neighbour, i.e. 1.2 kB per body per rank on a full 26-neighbourhood;
- the arriving records are binned into a **uniform grid** whose cell is at least
  as wide as the widest body plus the largest envelope, so "extents overlap"
  implies "within one cell" and a query is the 27 cells around a centre;
- body B travels towards body A only when `A < B` by body id, which is the
  existing "reported once" rule (the holder of the smaller id is the rank that
  reports the pair, so it is the rank that needs the copy);
- `MFTLExtendSpatialHaloNearBodies` returns the candidate pair list, and
  `MFTLEnsembleIntersectObjects` traverses that instead of all pairs.

`MFTL_SpatialHalo` became a **sparse, gid-keyed directory** rather than a
`[n_bodies_global]` table of patch pointers and radii, and it deliberately holds
more than the local object list: the bodies this rank *holds*, plus the bodies it
*received a patch for*. The second set is the point of a spatial halo — a rank
with droplet A and none of droplet B is exactly the rank that must be told about
B, and it has no local object to hang that on. `MFTLSpatialHaloPatch()` and the
new `MFTLSpatialHaloRadius()` binary-search it.

`MFTLSpatialHaloResolveRadii` moved onto `MFTLEnsembleReduceBodies`: twelve MIN
slots per **local** body (a lower corner and a negated upper corner, in the raw
frame and in one displaced by half a domain) instead of an `MPI_Allreduce` on two
`[12 × n_bodies_global]` staging buffers. It gained one small extra step — each
rank publishes `(body id, radius)` to its 26 neighbours — because a rank deciding
what to *send* needs the size of the body it is sending *towards*, and one hop is
exactly the right distance: the halo only ever ships along `comm_graph`.

Measured on twelve bodies arranged as six well-separated touching couples, so
exactly 6 of the 66 unordered pairs are contacts:

| np | candidate pairs, all ranks | all-pairs would be | directory slots, all ranks | dense would be |
|---:|---:|---:|---:|---:|
| 1  | 6  | 66   | 12 | 12  |
| 2  | 6  | 132  | 16 | 24  |
| 4  | 9  | 264  | 23 | 48  |
| 8  | 18 | 528  | 44 | 96  |
| 27 | 6  | 1782 | 12 | 324 |

`test_the_broad_phase_finds_the_pairs_without_enumerating_them` asserts both
halves and neither alone is enough: the contact count must equal an **exhaustive
`O(nA·nB)` sweep over all 66 pairs**, run on one rank while it still holds the
whole ensemble and sharing no code with the BVH, the halo or the grid; and every
pair the broad phase kept must be one of the six, because a grid that admitted a
far-apart pair would still give the right answer and counting hits alone cannot
see it.

#### The remesher

Four collectives, all of them staged by body id on `comm_world`:

| what | was | is |
|---|---|---|
| `MFTLRemeshMeasureBodies` | `8B` doubles SUM, then `B` doubles MAX | 9 SUM/MIN values per **local** body, 1 MAX value per local body, and **one scalar `MPI_MAX`** for the largest body in the ensemble |
| the per-phase id block | `MPI_Exscan` + `MPI_Allreduce` of `2B` ints, and `11B + 2E` ints of scratch per pass | `MFTLEnsembleExscanBodies`, a per-body **scan over that body's holders**, and 6 doubles + 3 ints per local slot |
| the dispersed id counters | `2B + 1` ints MAX | 2 values per local body plus one scalar |
| `watermarks_agree` (Debug) | `4B` ints MAX | 4 values per local body |

`MFTLEnsembleExscanBodies` is the new primitive and the interesting one. A
reduction is not enough to hand out global ids: two ranks that both split an edge
of one body each create a vertex, and the two need *different* ids, so each holder
needs the count of what the holders **before** it created. That is an exclusive
scan, not a combination. It reuses the reduction's record exchange and its
ascending-source-rank order, falls back to `MPI_Exscan` + `MPI_Allreduce` on the
same test, and is checked by `test_the_scan_tiles_every_body_exactly_once`: every
holder's range must abut the previous one and the last must close exactly on the
total, gathered and walked per body on rank 0.

Two things did **not** move at this stage, and both are stated rather than hidden.
The convergence certificate used to ride the id `Allreduce` for free; now that the
ids go over the 26-neighbour graph it pays its own 3-int `Allreduce` per phase —
one small latency, to remove an ensemble-sized payload from every phase. And the
**dispersed load balancer still needed replicated per-body data**, because its
greedy is over rank *loads* and a rank's load sums bodies held anywhere in the
job. That replication was made explicit here (`replicate_for_balancer`, one
`MPI_Allreduce` of `4B` doubles instead of the measurement's `9B`), attributable
to one switch, and **skipped entirely** when `dispersed_balance_slack <= 0`. It is
gone in Stage 6.

#### The evidence that Stage 4 changed no answer

`examples/leveque_remeshing_parallel` reproduces the mesh digests recorded in
[`parallel-remeshing-analysis.md`](parallel-remeshing-analysis.md) §3 **exactly**
— `f45a22917d6f42d0`, `d61ab5fdc0d81af4`, `0275cc47ff9d5fa4` at np = 1, 2 and 4
over 40 remeshed timesteps at budget 4. Those digests predate the whole
scalability update, so the distributed remeshing answer is bit-for-bit what it was
before any of it, *through* a per-body scan over holders.

`examples/remesh_regime_sweep -only l` reproduces the balancer table of
[`parallel-remeshing-analysis.md`](parallel-remeshing-analysis.md) §7.4 digit for
digit at np = 4, 8 and 27: clustered imbalance 4.00 / 8.00 / 27.00 → 1.25, idle
ranks 3/4, 7/8, 26/27 → 0, 0, 1, busiest-rank ops 12 608 / 25 216 / 85 104 →
3 940, and 103 of 108 bodies reassigned at np = 27. One row moved: the np = 4
*lattice* case reports a pre-balance imbalance of 1.25 rather than the recorded
1.50. That is a last-ulp tie-break, not a defect — a lattice puts body centroids
exactly on cut planes, the holder route sums in ascending rank order and
`MPI_Allreduce` in an MPI-chosen one, and a centroid on a knife-edge tips to the
other cell. Every rank still agrees, which is the property that matters.

### Stage 5 — the two things Stage 4 said it had done, and had not

Found by reading the tree against this document rather than by a failing test,
which is the whole point of the section. **Neither defect was observable in any
mesh MFTL produces.** The full 48-entry suite passed at np = 1, 2, 4, 8 and 27 in
Release and in Debug with ASan and UBSan; `leveque_remeshing_parallel`
reproduced `f45a22917d6f42d0`, `d61ab5fdc0d81af4` and `0275cc47ff9d5fa4` exactly;
`remesh_regime_sweep -only l` reproduced the balancer table digit for digit. Both
defects were **scaling** defects on a machine that cannot measure scaling.

#### The route was pinned, so the scan went over `comm_world` anyway

`gid_reduce` in `remesh_parallel.c` — the per-phase id block, and the single call
site the Stage 4 table names as its headline change — passed
`MFTL_BODY_REDUCE_SCOPE_GLOBAL` rather than `MFTL_BODY_REDUCE_SCOPE_AUTO`. So
`MFTLEnsembleExscanBodies` was reached, its contract was honoured, its
local-slot-indexed buffer was used, and it then took `exscan_over_comm_world`:
`3 × n_bodies_global × 2` doubles of staging per rank and two `comm_world`
collectives over a third of that, **once per phase**, four to eight times per
remesh. At a million bodies that is 48 MB staged and 32 MB on the wire, per rank,
per phase, to hand out ids for the hundred bodies the rank holds.

**Why no test could see it.** The two routes compute the same answer to the bit —
that is the property the rest of `test_body_reduce.c` exists to establish, and
`test_both_scan_routes_agree` asserts it directly. A test of the ANSWER therefore
cannot distinguish them, ever. The only observable is which communicator ran.

The fix is one token. What is registered with it is
`test_a_droplet_remesh_takes_no_global_reduction`, the sibling of the timestep
gate: it resets `MFTLEnsembleGetBodyReduceStats`, runs
`MFTLRemeshEnsembleParallel` on the twelve-droplet fixture, and asserts the split.

| np | holder reductions | global | with the pin restored |
|---:|---:|---:|---|
| 1  | 0  | 0 | 0 / 0 — the single-rank identity, correctly invisible |
| 2  | 11 | 0 | 6 / **5** |
| 4  | 11 | 0 | 6 / **5** |
| 8  | 11 | 0 | 6 / **5** |
| 27 | 11 | 0 | 6 / **5** |

Five is four phase scans plus the closing re-densification, which is exactly the
per-phase term. The test fails on every rank with the pin restored and passes
without it, and it asserts two non-vacuity conditions so it cannot pass by
accident: the pass must run more than one phase, and the coarsening fallback must
have stayed at `{1,1,1}` — on a coarsened partition a holder is `group` cells
away, `MFTLEnsembleBodyHoldersFitTheGraph` correctly says so, and the global
route is then the right answer rather than the regression.

**The strategy and the route are independent questions and the test separates
them.** The fixture is dispersed and the driver is the spanning one, deliberately:
the strategy asks how big a body is against a rank box, the route asks whether a
body's holders fit in one 26-neighbourhood. That is not a contrived pairing —
`MFTLRemeshEnsemble` dispatches per ENSEMBLE, so one body wider than a rank box
sends a million droplets down the spanning driver, and `MFTLRemeshEnsembleParallel`
is a public entry point in its own right.

#### The owner maps were still sized by the id space

Every remeshing migration goes through `MFTL_OwnerMap`, and `migrator_reserve`
sized `MFTL_RemeshMigrator.maps` at `MAX(n_bodies_global, n_local)` and keyed it
by `global_id`. So each migration cleared and then partly filled two pointers for
every body in the **job**, on a rank that may hold none of them: 16 B per body per
rank, four to eight times per remesh, plus the `O(B)` clearing pass. It was the
last `O(bodies in the ensemble)` allocation in the library apart from the
dispersed balancer, and unlike the balancer it is on the **per-phase** path.

It is item 4 of the implementation plan's own remeshing checklist — *"explicit
owner maps are a set of local records, not an array indexed by the global list"* —
and it had not been done.

**The reason it survived is that the rule it violates is a correct rule applied
one step too far.** Everything that crosses a rank boundary indexed by body must
be keyed by `global_id`, because a list position is only an identity while every
rank holds every body; that is Stage 1, and `MFTLRedistributeWithOwners`' own
header said so. But **the map never crosses a rank boundary.** The caller fills it
walking `ens->local_objects`, and the function reads it walking the same list in
the same order on the same rank. Slot `o` names the same body at both ends by
construction, and no identity has to survive anything.

That is the same correction `MFTLEnsembleReduceBodies` makes to its caller's
buffer, and for the same stated reason: move the id indexing inside the one place
that genuinely needs it, or the message shrinks and the per-rank memory does not.
`MFTL_OwnerMap` is now documented and indexed as `[ens->n_local_objects]` in
`ens->local_objects` order, in the public header, in `redistribute_core` and in
the migrator. The test for it is the existing one — every remeshing entry at every
rank count, in Release and under ASan, plus the three unchanged digests — because
this half genuinely is a pure refactor of an index.

The rule that replaces the over-applied one, now in `CLAUDE.md`: **ask whether the
index has to mean the same thing on two ranks, not whether it is about a body.**

### The bug the benchmarks caught and the tests did not

`examples/leveque_remeshing_parallel` held the `MFTL_Object*` that
`MFTLObjectCreate` returned and advected through it every timestep. That is
precisely the API break §5 below describes, and the example was never updated for
it. The consequences are worth spelling out because they are the shape this
failure always has:

- **at np = 1 it is correct**, because nothing is ever destroyed;
- at np ≥ 2 one rank's share of the surface simply never moves;
- **every structural invariant survives**. The mesh stays a valid closed sphere,
  Euler characteristic 2, zero inverted triangles for the first few steps, mean
  triangle quality above 0.98. `MFTLCheckMeshIntegrity` reports nothing;
- the enclosed volume is the only thing that moves, and it moves a *lot*:
  **V/V₀ = 0.074 after twelve steps against 0.995**;
- no CTest entry saw it, because no CTest entry advects a distributed mesh for
  more than a handful of steps and then measures a physical quantity.

It is now guarded by `test_leveque_parallel_conserves`, registered at **np = 2**
because np = 1 cannot see it, and by the example's own `-verify` flag. Two
sibling reporting defects of the same family were fixed with it: the benchmark's
mesh digest keyed on the local list *slot* rather than the body id (harmless at
one body, wrong at several), and `remesh_regime_sweep` summed `global_n_triangles`
over its local list and printed a column that halved when the rank count doubled.

**The lesson is registered rather than remembered**: the sparse local list moved a
correctness obligation onto callers, and the only thing that catches a caller
getting it wrong is a *physical* measurement over enough timesteps for the drift
to be unmistakable.

### Stage 6 — the dispersed balancer, and the end of the `O(B)` terms

The one term Stages 4 and 5 both named and neither removed, because it needed a
different *algorithm* rather than a different route. `assign_destinations` is a
greedy over rank LOADS; a rank's load sums bodies held anywhere in the job; so as
long as every rank ran the greedy, every rank needed every body. `MFTL_OwnerMap`
is undefined — not slow, undefined — if two ranks name different destinations for
one entity, and that is what made "every rank runs the same greedy over the same
replicated table" look like the only safe shape.

**The way out is that agreement does not require replication; it requires
delivery.** Let `home(o)` be the owner of body `o`'s reduced centroid — bit-identical
on every holder, because the centroid came out of `MFTLEnsembleReduceBodies`. Then
one rank decides `o` and the rest read the verdict.

The precondition of that is reachability, and it is a **local test**, which is
what makes the whole thing work: each holder checks that `home(o)` is itself or
one of its 26 graph neighbours, and one `MPI_LAND` turns those local checks into

> for every body `o` and every holder `H` of `o`: `home(o) ∈ {H} ∪ N(H)`.

Two consequences fall straight out, and they are exactly the two messages needed.
`home(o)` **can be told about the body**: the graph is symmetric, so
`lowest_holder(o)` — already reduced, already agreed, and provably a holder — is
one of its neighbours. And `home(o)` **can tell every holder**, in one
neighbourhood gather. Note what is *not* required: the holder set need not be a
clique, no holder need own an entity, and `home` need not hold the body.
`MFTLEnsembleBodyHoldersFitTheGraph` is strictly stronger and deliberately not
what is tested here. A failure reports `MFTL_RemeshStats::n_bodies_unreachable`
and reverts the pass to plain centroid ownership — never to replicating, because
a fallback that is `O(B)` *is* an `O(B)` term.

**What a local rule costs instead is a negotiation.** A replicated greedy can hand
a body to the least-loaded rank in the job knowing nobody else is doing the same;
a local one cannot, because every overloaded rank sees the same idle neighbour.
Three directed doubles per neighbour per round buy it: a rank asks its neighbours
for room in proportion to what they advertise, and each neighbour divides the room
it actually has among the askers, neediest first. `Σ grants ≤ room` by
construction, so no rank can be pushed past its cap however many neighbours ship
at once.

Two failures of that rule were found by measurement, and neither is visible to any
test of the answer — a balancer that silently does nothing produces a perfectly
valid mesh:

- **a body is indivisible, so a proportional ask is wrong.** Splitting the excess
  over 26 neighbours returns grants smaller than one body and nothing is
  placeable: 108 bodies over 27 ranks stayed at an imbalance of 2.00 with
  `n_bodies_reassigned == 0`, while the clustered fixture — one uncontended asker
  — balanced perfectly;
- **a tie-break identical on every rank starves the same rank.** Four overloaded
  ranks looking at four equally empty neighbours all pick the same one: 1 of 8
  left idle. Ties now break on the neighbour's cell *offset*, which is relative to
  the rank doing the looking.

The result reproduces the replicated balancer digit for digit — 2.00 / 4.00 / 8.00
/ 27.00 → 1.25 at np = 2 / 4 / 8 / 27 on the clustered fixture, zero ranks idle
(one better than the replicated greedy at np = 27), 1.45× to 5.0× less wall clock.

**The observable is the payload, and it needed a new instrument.** Both balancers
give a correct, reproducible, equally balanced mesh, so every digest and every
imbalance figure in the suite is blind to the difference — the same blindness
`MFTLEnsembleGetBodyReduceStats` exists to cure for pinned reduction scopes. So
`MFTLRemeshBalancerPayload()` counts the doubles a rank stages, and
`test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold` asserts the sharp
form: **double the ensemble and a rank holding none of it must stage the same
number**. Measured 25 / 49 / 85 doubles at np = 2 / 4 / 8, unchanged from 16 to 32
bodies, against `4B = 64 → 128` on *every* rank under replication.

Two smaller things went with it. The imbalance report no longer reduces a
`[nproc]` load table — it is four scalars — so the dispersed driver has no
`O(nproc)` term of its own either. And the balanced and unbalanced paths are now
one function rather than two, so the diagnostic costs the same either way.

## 3. What it measurably bought

**Stages 0–3 only.** Stage 4's evidence is of a different kind and is reported
with it: counted quantities (candidate pairs, directory slots) and bit-for-bit
digest comparisons against numbers recorded *before* any of this work, because
none of the paths it touches is on the critical path of a four-core laptop and a
timing would measure the machine.

All figures below are a **differential measured in one session**: the pre-change
commit built and run against the current tree on the same laptop with
byte-identical flags. Full tables in
[`scaling-analysis.md` §scaling_before_after](scaling-analysis.md).

**Per-rank body work went from tracking the rank count to nearly flat.** Weak
sweep, 32 bodies per rank:

| np | bodies | body visits per rank per step | empty bodies |
|---|---|---|---|
| 2 | 64  | 64 → **36.4** (1.76×)  | 2.4 % → **2.1 %** |
| 4 | 128 | 128 → **41.0** (3.12×) | 18.5 % → **4.3 %** |
| 8 | 256 | 256 → **46.8** (5.48×) | 34.6 % → **6.3 %** |

The reduction *factor* is the result, not the absolute: it rises with the rank
count, which is precisely the weak-scaling signature the change was for. The
residual growth — 36 to 47 — is halo sharing, not the ensemble. On the other axis,
at fixed `np = 4`, 16 384 bodies went from 16 384 visits per rank per step to
**4 353**, and the empty-body fraction from 6.5 % to 2.4 %.

**Nothing regressed.** Strong scaling on a fixed 327,680-triangle problem measures
1.00×, 0.99×, 1.01×, 0.99× at `P = 1, 2, 4, 8` while per-rank body visits fall by
1.00×, 1.76×, 3.00×, 5.46×.

### And the wall clock did not move, which is the honest headline

**Every timing above is a wash: 0.98× to 1.16×, against a noise floor wider than
most of that.** Five independent runs of the same configuration give step medians
spanning **±28 %**. Every counted column, by contrast, is bit-stable across runs.

That is the expected outcome. Four performance cores reduce through shared memory
at roughly 200 GB/s, and a loop iteration over a body a rank holds nothing of is a
handful of correctly predicted branches. What this work removes is **memory per
rank** and **message volume per rank**, and neither is on the critical path of a
four-core laptop. Resident memory barely moves either (280.9 → 281.4 MB at 16 384
bodies), for a related reason: the per-body arrays made local are tens of bytes
per body against the ~5 kB of mesh each body carries, so on any ensemble that fits
on a laptop the pools dominate. The arrays that were removed scale with
`bodies × ranks`, which is the one axis a laptop cannot grow.

**Stage 3 in isolation is not a speed-up at all, and at these rank counts it moves
MORE bytes.** Forcing the three call sites back onto `comm_world` — same sparse
list, only the route different — changes no timing outside the noise, and the
holder route's payload is 1.5×–2.0× *larger*. That is not a defect; it is the
model working. The neighbourhood route sends every held body to every neighbour,
so its payload is `b_local × (1+d)` records against the old route's `B_global`.
With the graph degree `d` capped at 26 the ratio is `1.4 × (1+s) × (1+d) / P`,
which crosses 1 at about **45 ranks** and then improves linearly: 22× fewer bytes
at `P = 1000`, 220× at `P = 10 000`. Per consumer the crossover differs by two
orders of magnitude — quality at `P ≈ 35`, counts at `P ≈ 96`, and the skip vote
at `P ≈ 4100`, because a one-bit-per-body bitset is simply a very good encoding
and nothing built from `(id, value)` records beats it on a small ensemble.

So the correct summary of Stage 3 is: **a measured change of shape whose payoff is
off the end of this machine.** What the experiment does confirm is that both
routes are live, that a droplet ensemble never falls back, and that the byte ratio
tracks the model rather than drifting.

## 4. Six bugs the work exposed, all of them silent

Worth reading even if the rest is skimmed, because each was a plausible-looking
piece of code that produced a wrong answer or a hang with no diagnostic. The first
four are Stages 0–3; the last two are Stage 4.

**A collective guard on a local count is a deadlock.** About seventeen functions
early-returned on `num_objects == 0`. That used to be a *global* condition, so
every rank returned together. Once the count became local, every one of them was
a rank sitting out a collective the others were inside. All of them now read
`n_bodies_global`.

**A switch derived from a local count changes a message count.** Redistribution
decided whether to ship the per-body count payload with `num_objects > 1`, which
was the same number on every rank only because every rank held every body. It
became rank-dependent, and ranks posted different numbers of `Alltoallv` calls.
It now reads "does this call have an ensemble".

**An empty holder is invisible, so it has to veto.** The holder-set test starts
from "a rank holds an entity of a body only if it *owns* one". A rank holding an
*empty* object owns nothing, appears in no other holder's halo, and can therefore
neither be discovered nor served — it would sit out the exchange and keep whatever
was in its buffer. That is not a corner case: it is the state a *generated*
ensemble is in before its first redistribution, and on a 4×1×1 grid rank 2 is not
even a neighbour of rank 0. An existing test caught it; such a rank now vetoes for
the whole ensemble.

**`comm_graph` carries a self-loop on a thin grid.** The neighbour construction
skips the `(0,0,0)` *offset*, not the own *rank*: on an axis with a single cut and
periodicity every offset along it wraps back onto this cell, so the 26-stencil
legitimately resolves to this rank. An exchange does not care — the block arrives
and the data is right. A **reduction** does: the self block is folded in a second
time, which MIN and MAX hide while every SUM silently doubles. Grid shapes
`{2,1,1}` and `{2,2,1}` have the self-loop and `{2,2,2}` and `{3,3,3}` do not,
which is exactly the set of rank counts at which the remeshing suite failed.

**A per-body verdict summed over ranks counts a shared body once per HOLDER.**
Every holder of a body comes out of a reduction with the same value — that is the
guarantee the whole mechanism is built to give — so `sum over ranks of (this
body's radius exceeds a rank box)` counts a body held by four ranks four times.
The dispersed remesher printed it as **"4 of 1 bodies are wider than 0.40 × h_min"**
and only the arithmetic gave it away; the *decision* it feeds (`> 0` ⇒ fall back
to the spanning driver) was right either way, so no test could see it. The fix is
one slot: a `MIN` of the contributing rank is the body's **lowest-numbered
holder**, and `lowest_holder[o] == rank` is the licence to count body `o`. It
rides a reduction already being taken and needs nobody to know the holder set.
This is the general answer whenever a per-body quantity has to be counted exactly
once across the job, and it did not exist before the object list became sparse —
with every rank holding every body, "count it on the rank that owns its centroid"
was available and now is not.

**A caller holding an `MFTL_Object*` across a redistribution loses volume, not
structure.** The sixth is §5's API break, found in this repository's own
benchmark. It gets its own account under Stage 4 because the interesting part is
not the mistake but what it looked like: valid mesh, valid topology, valid
quality, 93 % of the enclosed volume gone, and correct at np = 1.

One further hazard was designed out rather than fixed. Two holders of one body
must end up with **bitwise identical** reduced values, because a solver branches
on `quality` to decide whether that body needs remeshing and that decision is
collective — two ranks differing in the last ulp is two ranks entering different
collectives. Floating-point addition is not associative and the neighbour list is
in an MPI-chosen order that differs from rank to rank, so contributions are folded
in **ascending source-rank order** with this rank spliced in at its own position.
`MPI_Allreduce` gives that guarantee for free; here it had to be arranged.

## 5. The one API break, and it will affect solver code

**An `MFTL_Object*`, a list slot and the local body count are all invalid after a
redistribution.** A rank that ends up holding no entity of a body destroys its
object; one that receives a body it had nothing of creates a new one.

The handle that survives is `global_id`, and `MFTLEnsembleFindLocalObject(ens,
gid)` is the way back — it returns `MFTL_NULL` when this rank no longer holds the
body. It is the same rule global ids and pool indices already follow across a
remesh. The failure mode is a use-after-free on a pointer that was valid one line
earlier, and it **hides for a long time** on any fixture where every rank keeps a
share of everything, which is what made it most of the test churn in Stage 2.

Two consequences follow: a guard on the *local* body count in front of a
collective is a deadlock, so every such guard reads `n_bodies_global`; and
anything indexed by body must be indexed by `global_id` rather than by a list
slot.

**And it did in fact affect solver code, in this repository, for the whole of
Stages 0–3.** `examples/leveque_remeshing_parallel` advected through the pointer
`MFTLObjectCreate` returned and lost 93 % of its enclosed volume in twelve steps
at np = 2, with every structural invariant intact and the np = 1 answer correct.
Stage 4 found it, fixed it and registered a physical check for it; the account is
under "the bug the benchmarks caught and the tests did not" above. Read that
before writing a solver against this API — it is what the failure looks like from
outside.

## 6. What is still missing

**No `O(bodies in the ensemble)` term is left on a per-timestep or per-remesh
path.** The dispersed load balancer was the last one and Stage 6 removed it; the
remesher's explicit owner maps were the one before that, in Stage 5. What is left
below is either a measurement this machine cannot take, a module that was never
part of this work, an `O(nproc)` term, or a small item with a stated reason for
not having been done.

| what | where | shape |
|---|---|---|
| generation builds every body on rank 0 | `MFTLGenerateSphere`, once per run | serial, and the Amdahl ceiling |
| `[nproc]` metadata inside a redistribution | `redistribute_core`, `MFTLRemeshSyncGhostIds` | `O(P)` per call, allocated and freed |
| the halo reaches one hop | `MFTLExtendSpatialHalo`, by construction | a contact two cells away is missed, silently |
| the balancer's weight moves one cell per pass | `assign_destinations`, by construction | a pile-up wider than a cell is only partly fixed |
| a route choice has no test of its own unless one counts routes | every `MFTL_BODY_REDUCE_SCOPE_*` call site | invisible to every digest; two gate tests, and no more |
| nothing measured above four ranks | the machine | every scaling claim is a model, not a measurement |

### Nothing above four ranks has been measured on real hardware

This is the largest gap and it invalidates no result above, because every result
above is either a counted quantity, a bit-for-bit digest comparison or an explicit
"no change". But the *point* of the work is `P` in the hundreds and thousands, and
the development machine has four performance cores. Above four ranks half the
ranks land on efficiency cores and absolute per-triangle costs roughly triple.
Specifically unmeasured: whether the holder route's predicted crossover at
`P ≈ 45` is real, how a neighbourhood collective compares against a `comm_world`
one at `P = 10 000`, and whether the `O(B)` terms that remain below actually
become the limit.

### The balancer's weight moves one cell per pass, and that is now the limit

Stage 6 replaced the replicated greedy with a rule that decides each body on the
owner of its centroid and delivers the verdict one hop along `comm_graph`. The
`O(B)` term is gone; what took its place is a **reach** limit, and it is worth
being precise about because it is not the same weakness.

A body can only be reassigned to a graph neighbour of its home rank. A pile-up
confined to one rank box is therefore fixed completely — measured 27.00 → 1.25 at
np = 27 with no rank idle — and so is any imbalance whose surplus has somewhere to
go within one cell, which is what droplet clustering looks like. A pile-up
*spanning many cells* is only partly fixed, and successive remeshing passes do not
compound the fix, because the closing `MFTLRedistribute` restores the geometric
partition every time.

The honest comparison is that **the replicated greedy was not much better here**:
its own policy tried the 26 cells around a body before anywhere else, and its
global fallback is precisely the case where a body leaves the stencil and
redistribution drops to its all-to-all path. What it could do and this cannot is
place a body on an idle rank on the far side of the machine. Whether that is worth
an `O(B)` broadcast is a question a cluster measurement should answer, not this
document.

Two smaller properties of the new rule, both recorded rather than fixed:

- **The rounds are bounded, not convergent.** Three ask/grant/ship rounds run
  whatever the state; running out leaves an imbalance the pass reports in
  `load_imbalance`, never a wrong mesh. Measured, two rounds suffice on every
  fixture up to np = 27 and one does not (np = 27 spread stalls at 1.50).
- **The precondition is a gate, not a per-body exception.** One unreachable
  (holder, body) pair anywhere disables balancing for the whole pass, because
  skipping just that body would leave the rank loads describing a different
  assignment from the one that ran. `n_bodies_unreachable` reports it. Reaching it
  needs a body wider than the admission test allows; the realistic route is
  `parallel_coarsen` relaxing the yardstick to a coarsened box.

### Generation is still rank-0 serial, and it is the Amdahl ceiling

`MFTLGenerateSphere` builds every body whole on rank 0 — `3680 × 4^refine + 176`
bytes each, 57.5 MiB for 256 bodies at refine 3, 460 MiB for 512 at refine 4 —
before the opening redistribution scatters it. The peak is real, it is serial, and
nothing in this work touches it. In a weak sweep setup time therefore grows
linearly with the rank count while the timed loop does not, which is why setup
must never be folded into a scaling total. Fixing it needs a generator that builds
distributed, which does not exist. Batching is *not* the fix: it shaves only the
packed-buffer transient and costs `O(N²/K)` because the halo analysis is
re-derived per call.

### The halo reaches one hop, and a contact two hops away is missed

Not new and not introduced here — `MFTLExtendSpatialHalo` has only ever shipped
along `comm_graph` — but Stage 4 made it visible, because the radius publish now
has the same reach and `test_the_distributed_answer_matches_the_unpartitioned_one`
had to be told so. A rank two hops from body B receives no patch of it, holds none
of it, and returns from the traversal before it looks at a single box; its
envelope for that body is unobservable, so the test's "the tolerance did not move"
assertion is taken on the ranks that can actually report. That is the whole of the
property, not a weakening of it — but it does mean **a contact between two bodies
whose holders are more than one cell apart is not found**, which is the same
condition the per-body reduction's holder route already requires and the
distributed remesher already dispatches on. It has no diagnostic.

### Smaller items, in rough order of how likely they are to bite

**Only two call sites are gated, and there are eleven.** `MFTLEnsembleReduceBodies`
and `MFTLEnsembleExscanBodies` are reached from the quality statistics, the global
counts, the skip vote, the halo radii, the remesher's body measurement, its
watermark check, its per-phase id scan and its re-densification — and exactly two
of those routes are asserted, by `test_a_droplet_timestep_takes_no_global_reduction`
(3 holder, 0 global) and `test_a_droplet_remesh_takes_no_global_reduction` (11, 0).
Between them they happen to cover every site a droplet ensemble reaches, which is
why the counts are asserted as exact numbers rather than as inequalities: a new
call site that pins its scope changes one of the two totals and fails. A site
reached only in a regime neither fixture produces would still be unguarded, and
Stage 5 is what that costs. Adding a third gate is cheap; knowing which regime it
should run in is the part that needs thinking about.

**The route verdict is recomputed on every per-body collective, and there are now
more of them.** One `MPI_LAND` of a single int each: three in a timestep, plus one
for the halo radii and two per remeshing pass, plus one per phase for the id scan.
It is latency and not volume, and caching it was rejected on purpose — a stale
verdict is a silently wrong reduction and the invalidation rule would have to be
maintained by every future holder-set change. If a cluster measurement ever shows
those latencies mattering, the right fix is a partition epoch owned by
redistribution, not a cache in the reduction.

**The uniform grid degrades to one cell when a body spans the domain.** Cell width
is `2 × widest half-width + largest envelope`, so one domain-spanning body makes
the grid a single bucket and the broad phase falls back to all-pairs *within the
neighbourhood*. That is the correct answer and the right cost for the case, but it
is a cliff rather than a slope, and a two-level grid or a body-size histogram
would remove it. Not done: the regime it fires in is the one the distributed
remesher calls SPANNING, where the ensemble has a handful of bodies.

**A one-slot reduction deserves a narrower record.** The skip vote is the one
consumer for which the old bitset was better at any realistic ensemble size on a
laptop, and its crossover is `P ≈ 4100`. A 32-bit record type would move it down
by a factor of two. It was not done because two payload encodings is two
mechanisms; revisit only with a measurement asking for it.

**`MFTLEnsembleReduceBodies` and `MFTLEnsembleExscanBodies` allocate per call.**
Five small buffers on the neighbourhood route and two or three on the global one,
several times a timestep and once per remeshing phase. It is the same first-touch
churn `performance-analysis.md` §7 already identifies as why a first benchmark run
is slower than every repeat, and the remedy is the same — persistent scratch on
the ensemble. Not done, for the same reason the redistribution workspace took a
while to get one: it puts mutable state on a struct that is otherwise read-only
during a call.

**The holder-set exactness argument assumes a body is a connected surface.** Two
components of one body could sit on ranks that neither see each other nor have
anything between them, and no local test could tell. Nothing in MFTL produces such
a body — generation emits a closed sphere, remeshing preserves connectivity, and
the fan walks, the mean edge length and the halo rule all read one surface — but a
caller that assembles one from an index list owes
`MFTL_BODY_REDUCE_SCOPE_GLOBAL`. This is characterised, not asserted away.

**The workspace per-body figure is a ratchet and reads high.** `cap_objects` never
shrinks, and every rank held the whole dense list once — during generation, before
the opening scatter — so the reported footprint is the high-water mark rather than
the steady state. It is the workspace analogue of the stranded pool capacity in
§scaling_trap_pools, with the same remedy (an explicit shrink) and the same reason
for not having one: it is a trade, and this library ships trades disabled.

**The ghost-exchange schedule is still rebuilt every timestep**, and
`MFTLUpdateGhostPositions` still has no global fallback — a ghost whose owner is
not a graph neighbour is counted as orphaned and skipped. Both predate this work
and both are unchanged by it; the second is safe today only because every
coarsened partition is undone by a plain redistribution before the next timestep.

**`MFTLEnsembleAcquireDenseView` now has no callers in the library.** It is kept
because it is the honest answer for a *caller* that genuinely needs every body,
and because deleting a documented entry point is a separate decision from no
longer needing it. If nothing outside the tests uses it by the next review, it
should go.
---

## Verification, so the claims above are checkable

48 CTest entries pass in Release **and** in Debug with ASan and UBSan, including
the distributed remeshing and contact suites at up to 27 ranks. New in this work:

- `test_body_reduce` at `np = 1, 2, 4, 8, 27` — both routes plus a **third
  reference implementation written in the test**, compared **bitwise**; per-holder
  bit-identity; the empty-holder veto; the spanning fallback; a rank holding zero
  bodies; and the gate itself, that a droplet timestep takes no `comm_world`
  reduction with a body-proportional payload.
- `inv_local_directory_is_sound` — the directory is sorted and strictly
  increasing, holds no empty object, and every id is in range — wired into every
  `ensemble_is_sound`.
- `test_a_body_arrives_leaves_and_returns`, which asserts the create/destroy churn
  is non-zero so that the lifecycle cannot pass vacuously.
- `parallel_advection_redistribute_list --digest --quality` matches on **both** the
  set digest and the slot-order digest with the body skip on and off, at `np = 2`
  and `np = 4`. The order digest is the one that catches a skip taken before the
  layout reached its fixed point.

New in Stage 4:

- `test_the_scan_tiles_every_body_exactly_once` and `test_both_scan_routes_agree`
  at `np = 1, 2, 4, 8, 27`. The tiling is the property a scan has and a reduction
  does not: every holder's range must abut the previous one in ascending rank
  order and the last must close exactly on the total. It reports how many bodies
  have more than one holder — 6, 6, 6 and 9 of 12 at np = 2, 4, 8 and 27 — and
  asserts it, because on a fixture where every body sits on one rank the scan is
  the identity and the tiling is satisfied by a function returning zero.
- `test_the_broad_phase_finds_the_pairs_without_enumerating_them` at the same five
  rank counts: the contact count against an exhaustive `O(nA·nB)` sweep over all
  66 pairs, *and* the assertion that every candidate pair the grid kept is one of
  the six real ones, *and* that all six were proposed, *and* that the halo
  directory summed over ranks is below `nproc × bodies`.
- `test_leveque_parallel_conserves` at `np = 2` — the enclosed volume, the winding
  and the Euler characteristic of a distributed, remeshed, advected mesh after
  twelve steps. It is the entry that catches a caller holding an `MFTL_Object*`
  across a redistribution, which no structural check can see.
- The two example digests that were already recorded before the scalability update
  are reproduced exactly: `leveque_remeshing_parallel` at np = 1, 2, 4 and the
  `remesh_regime_sweep` balancer table at np = 4, 8, 27.

New in Stage 5:

- `test_a_droplet_remesh_takes_no_global_reduction` at `np = 1, 2, 4, 8, 27`. It
  counts ROUTES, not values, because the two routes agree to the bit and no test
  of a mesh can tell them apart. 11 holder reductions and 0 global at every rank
  count above one; 6 and 5 with the pin restored, on which it fails on every
  rank. Two non-vacuity conditions guard it: more than one phase must have run,
  and the coarsening fallback must have stayed at `{1,1,1}`.
- The owner-map re-indexing has no test of its own and does not need one: it is a
  pure change of index on an argument that never leaves the rank, and it is
  exercised by every remeshing entry at every registered rank count, in Release
  and under ASan and UBSan, with all three recorded mesh digests unchanged and
  the `remesh_regime_sweep` balancer table reproduced digit for digit.

New in Stage 6:

- `test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold` at
  `np = 2, 4, 8`. It counts BYTES STAGED, not values, for the same reason the
  route tests count routes: the replicated and the local balancer reach the same
  imbalance on the same fixture, so no digest, volume or triangle count can tell
  them apart. Two runs, 16 bodies and 32, all inside one rank's box; a rank
  holding none of them must stage the SAME number of doubles both times, and the
  rank holding all of them must stage at least `3 x 16` more. Measured 25 / 49 /
  85 doubles on a non-holder at np = 2 / 4 / 8, identical at both body counts,
  against `4B = 64 -> 128` on every rank under replication. Two non-vacuity
  conditions guard it: some rank must genuinely hold nothing, and the balancer
  must have moved bodies in both runs.
- `MFTLRemeshBalancerPayload()` / `MFTLRemeshResetBalancerPayload()`, the
  instrument that makes the above expressible at all — the sibling of
  `MFTLRemeshMeasurementCount()` and of `MFTLEnsembleGetBodyReduceStats()`.
- `test_the_balancer_evens_out_a_pile_up` and `test_the_balancer_is_deterministic`
  were already there and both pass unchanged, which is the point: they are
  sensitive to the answer and blind to how it was reached.
- One test tolerance moved and it is recorded rather than quietly widened.
  `test_four_split_droplets_are_centred_and_remeshed` asserted that fewer than
  2 % of edges end outside the length band; which rank remeshes a body decides
  the order its pool arrives in, split and collapse are order-dependent, and the
  quantity spans **1.31 % to 2.17 %** over np = 1, 2, 4, 8 with the balancer on
  and off — all equally correct. The threshold is 3 %, with the measured spread
  in the comment. What it is for is catching a pass that did not converge, which
  reports tens of per cent.

Re-verified after Stage 6: 48 of 48 CTest entries in Release and in Debug with
ASan and UBSan; `remesh_regime_sweep -only l` gives clustered imbalance 2.00 /
4.00 / 8.00 / 27.00 -> 1.25 at np = 2 / 4 / 8 / 27 with **zero** ranks idle and
busiest-rank ops 6 304 / 12 608 / 25 216 / 85 104 -> 3 940.

Re-verified after Stage 5: 48 of 48 CTest entries in Release and in Debug with
ASan and UBSan; `leveque_remeshing_parallel` at np = 1, 2, 4 gives
`f45a22917d6f42d0`, `d61ab5fdc0d81af4`, `0275cc47ff9d5fa4`; `remesh_regime_sweep
-only l` at np = 4, 8, 27 gives clustered imbalance 4.00 / 8.00 / 27.00 → 1.25,
idle ranks 3/4, 7/8, 26/27 → 0, 0, 1, busiest-rank ops 12 608 / 25 216 / 85 104 →
3 940, and 103 of 108 bodies reassigned at np = 27.

Both documentation manuals build with zero warnings, every header compiles alone,
and the library exports only `MFTL*` symbols with no duplicates. The library
itself now compiles the whole tree, tests included, with **no compiler warnings**:
`test_body_reduce.c` was passing `const double[3]` to `MFTLGenerateSphere`, which
is four `-Wincompatible-pointer-types-discards-qualifiers` in a file this work
added.

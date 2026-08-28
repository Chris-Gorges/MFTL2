# Skipping unchanged bodies in `MFTLRedistribute`: what the precondition actually is

This document supersedes one section of `docs/redistribution_optimizations_review.md` — the part of
"OPTIMIZATION 1" headed *"Topology change — why it matters"* (`:36-40`). That review is left as it
stands; it is a record of what was argued, and this is the correction beside it rather than on top of
it. Everything else in the review is untouched by what follows.

**The status of every claim here is "proved from the code", not "measured".** Nothing in this document
was reproduced at runtime, because the optimisation it is about does not exist yet: there is no
`n_objects_skippable`, no `partition_generation` and no skip path anywhere in `src/` or `include/`.
What is offered is a derivation from the rules the library already enforces, with a `file:line` behind
each step so that each one can be checked rather than believed. Where a claim is an estimate rather
than a derivation, it says so.

## 1. The question

The proposed optimisation is: at the top of a redistribution, decide per body whether any of its
entities changed owner; agree on that globally; and if a body's answer is "none", leave it entirely
alone — no destination analysis, no kept gather, no relink, no topology rebuild.

The review asserts that an owner-change test is not sufficient, and that a **topology** test is needed
as well, for two reasons: that a flip changes which rank needs a triangle as a ghost without moving
anything, and that a split midpoint between two vertices on the same rank produces a new triangle with
zero owner changes, so `changed == 0` is true while the partition is wrong.

**Both of those reasons are wrong**, and the objection they were raised against — that MFTL's
non-negotiable promise is that no topology-changing algorithm ever touches a replicated entity — is
sound. The review's *conclusion* is nevertheless right: the closing `MFTLRedistribute` of a remeshing
pass must not be skipped. The mechanism is different, it is narrower, and it fires on a more common
case than anything the review describes.

## 2. The rule redistribution implements

`collect_destinations` sends an owned triangle to seven candidate ranks — the owner of its centre, of
each of its three corners, and of each of its three edge neighbours (`src/communication/redistribute.c:401-431`).
The prose form is at `redistribute.c:350-364`:

> A triangle goes to the owner of its centre (that rank will own it), and to the owners of its three
> vertices and its three edge neighbours (those ranks need it as a ghost). Sending that surplus here is
> what produces a complete, VERTEX-COMPLETE one-layer halo without a second communication round.

Read backwards, that is a definition of the output rather than a description of a procedure:

> rank *r* holds triangle *t* **iff** *r* owns *t*, or *r* owns one of *t*'s three corners, or *r* owns
> one of *t*'s three edge neighbours

together with the corners of everything it holds. The local state of rank *r* after a redistribution
is therefore a pure function of two inputs: the global topology, and the owner map. **Positions are not
an input.** They determine the owner map — `compute_owners` reads `verts.pos` and `tris.center`
(`redistribute.c:297-314`) — but once the map is fixed, moving a vertex within its own cell changes
nothing about who holds what.

This is why an owner-change test is the natural trigger in the first place: it tests one of the two
inputs directly. The whole question is whether the *other* input can change in a way that matters.

## 3. Why a topology change under the freeze discipline cannot change the answer

The freeze classification is an equivalence, not a distance (`src/remeshing/remesh_parallel.c:58-75`):

> an owned entity is frozen ⟺ some other rank holds a copy of it

and a rank evaluates the right-hand side alone, because it is the same rule redistribution sent by. In
storage terms, with ownership as a prefix, `MFTLRemeshMarkFrozen` computes
(`remesh_parallel.c:210-216`):

```c
const int replicated =
    (tv[b] >= nvl) | (tv[b + 1] >= nvl) | (tv[b + 2] >= nvl) | (tn[b] >= ntl) | (tn[b + 1] >= ntl) | (tn[b + 2] >= ntl);
```

Every ghost triangle is then frozen unconditionally (`:219`), and a vertex is frozen iff any incident
triangle is (`:232-240`).

The four-point exactness argument at `remesh_parallel.c:77-103` establishes that an operator admitted
by that gate has its whole modified set owned exclusively by the acting rank, and that every fan an
approval predicate walks is complete. Two further steps are needed for the question at hand, and
neither is spelled out there.

**New entities need no ghost anywhere.** An unfrozen triangle has no ghost corner and no ghost edge
neighbour — that is exactly what the six compares above test — so all three of its corners and all
three of its edge neighbours are owned by the acting rank *r*. A split, collapse or flip therefore
builds its replacement triangles out of vertices owned by *r*, with edge neighbours owned by *r*, plus
whatever midpoint it created, which is also *r*'s. Apply the halo rule of §2 to any such new triangle
and every clause resolves to *r*: no other rank owns its centre, none of its corners, none of its edge
neighbours. **No other rank's halo can name it.** The split midpoint the review offers as a
counterexample is in fact the canonical case of this: it is precisely the entity nobody else needs.

**Edge-neighbour completeness survives, not only vertex-completeness.** `remesh_comm.c:205-211` argues
the vertex-complete half; the halo rule is strictly stronger than that, and `inv_partition_is_sound`
checks the extra statement (every owned triangle has all three edge neighbours present). Suppose rank
*r* owns triangle *T* and rank *s* owns its edge neighbour *N*, so *r* holds *N* as a ghost. Then *N*
is replicated, hence frozen on *s* — so *s* may not split, collapse or flip it. Nor can *s* reach *N*
indirectly: a collapse's gate tests the two endpoint vertices (`remesh_internal.h:373-376`), and a
vertex is frozen if any incident triangle is, so every corner of *N* is frozen too. The only way *T*'s
edge neighbour could change identity is if the triangle across that edge were rewritten, and that
triangle is *N*. It cannot be.

So under the freeze discipline, a topology change is invisible to the halo rule. Both of the review's
stated reasons fail on the same step.

## 4. The library already relies on exactly this

`MFTLRemeshMigrateShifted` implements the same optimisation at ensemble scope — one
`MPI_Allreduce(MPI_LOR)` on "did anything move", and an early return that never enters
`MFTLRedistributeWithOwners` at all (`src/remeshing/remesh_comm.c:212-217`). Its soundness comment
(`:205-211`) is the argument of §3, written before the question was asked:

> Skipping is sound because a redistribution that moves nothing changes nothing: the ownership prefix,
> the halo and the derived arrays are all functions of the assignment, and the assignment is what did
> not change. The halo in particular is still vertex-complete, because a neighbour's sweeps can only
> have created entities whose corners it owns — an unfrozen edge has all of its corners on the rank
> operating on it — so nothing incident to one of THIS rank's owned vertices can have appeared
> elsewhere.

Note what that skip is placed between. It sits *inside* the phase loop, after sweeps that changed
topology on the previous phase and before sweeps that will change it on this one. It is not a
special case where topology happened to be static; it is the case the review says needs a topology
check, taken deliberately, on the hot path, every phase 0 of every remeshing call.

## 5. Where a per-body skip does break: ghost triangle global ids

The hazard is real, it is not the halo, and it is not caught by any argument about replication.

`redensify_global_ids` closes a spanning remeshing pass by renumbering every **owned** entity to
`base + slot` from an `MPI_Exscan` (`remesh_parallel.c:641-648`), and syncing ghost **vertex** ids
through `MFTLRemeshSyncGhostIds` (`:633`). Ghost **triangles** are deliberately left alone, and the
function says why (`remesh_parallel.c:611-614`):

> Ghost TRIANGLES are deliberately not exchanged. Redistribution keeps only owned triangles and
> re-receives every ghost, so a ghost triangle's id is replaced wholesale by the caller's closing
> migration — which is also why that migration has to come AFTER this function and not before it.

The mechanism it names is one line of `build_kept_lists` (`redistribute.c:675`):

```c
const int nt = obj->tris.n_local;  // only owned triangles can ever be kept
```

Every ghost triangle is dropped and re-received on every redistribution, and that is what repairs the
ids. **A skip removes precisely that repair.** The renumbering is upstream of the skip decision and
has already happened; the migration that was supposed to clean up after it never runs.

Three things make this worse than it first sounds.

**The renumbering is not the identity, even when the sweeps did nothing.** It might seem that a body
nothing happened to would be renumbered back onto the ids it already had. It is not: the previous
`MFTLRedistribute` rewrote the pool into `[ kept owned | new owned | kept ghost | new ghost ]`
(`redistribute.c:689-704`), with the kept-owned run in ascending *old* index order. Slot order and gid
order part company there, so the next `base + slot` pass assigns different ids to entities that never
changed.

**The skip's trigger and the hazard coincide on the most common case, not a corner one.** Consider a
converged body in a per-timestep remeshing loop. Phase 0's shift is the canonical `{0,0,0}`
(`remesh_parallel.c:157-159`), so the phase-0 migration is already skipped by `remesh_comm.c:212-214`;
the sweeps perform no operations and hide no edges, so the driver takes the fast exit at `p == 0`
(`remesh_parallel.c:1555-1559`) with the canonical placement still in force; `redensify_global_ids`
runs anyway; and if the smoother moved no vertex across a cut plane, the closing `MFTLRedistribute`
sees zero owner changes for that body. A naive per-body skip fires — on a good mesh, every step.

**The failure is silent and delayed.** The pool is left holding ghost triangles whose global ids belong
to the previous numbering, colliding with the freshly-assigned ids of unrelated owned triangles. That
violates `inv_local_global_ids_are_unique`, and can violate `inv_global_ids_are_dense` whenever a pass
netted out collapses and `next_triangle_gid` shrank below a stale ghost's id. Nothing in the production
loop reads a ghost triangle's gid immediately, so the wrong values sit there until the next full
redistribution replaces them — which means the symptom, if it ever surfaces, surfaces somewhere other
than the cause.

**The dispersed driver is not exposed to this.** `remesh_dispersed.c:760-767` asserts
`n_local == n` for every body before the serial kernel runs, so a body that could be skipped at that
driver's closing redistribute has no ghosts for stale ids to live in. `MFTLRemeshRenumberGlobalIds`
carries the same precondition as an assertion of its own (`remesh_compact.c:419-421`).

## 6. So the guard is not "did topology change"

The minimal correct precondition for skipping body *b* is:

> no entity of *b* changed owner, **and** nothing rewrote the global ids of entities that other ranks
> hold copies of.

The second clause is what the review was reaching for. A topology test would catch it, but only
incidentally and only because renumbering happens to accompany remeshing; it would also decline to
skip in cases that are provably safe, and — more importantly — it would be hooked in the wrong place.
The review proposes bumping a counter inside `MFTLObjectBuildTopologyArrays`. That is a rebuild
function, and rebuilds are a consequence of topology changes rather than the thing itself.

Two hooks are better, and between them they are complete:

**`MFTLObjectSetCounts` (`src/object/object.c:308-327`)** is the single function any entity creation or
destruction must pass through — it is the enforced choke point that CLAUDE.md's pool discipline is
built around, and it already ends with

```c
  /* The derived arrays describe a different entity count than they did a moment
   * ago, so whatever they hold is no longer a description of this mesh. */
  obj->derived = 0u;
```

so a partition stamp is one adjacent store in a function that is already doing exactly this kind of
bookkeeping. This is what makes the skip's soundness independent of an unenforceable convention.
That matters, because the freeze discipline of §3 is enforced by the *sweeps*, not by the type system:
`MFTLSplitEdge`, `MFTLCollapseEdge` and `MFTLFlipEdge` are public API
(`include/mftl_remeshing.h:1153, 1226, 1266`) and can be called on a distributed object with no gate
whatsoever. `tests/test_remeshing_parallel.c:594-603` does precisely that, re-implementing the gate by
hand at the call site. Redistribution cannot verify that a caller obeyed the rule, and the failure if
one does not is silent — so the cheap conservative stamp is worth its one store.

**The two renumbering sites** — `redensify_global_ids` (`remesh_parallel.c:620-656`) and
`MFTLRemeshRenumberGlobalIds` (`remesh_compact.c:416-439`) — cover §5 directly, including the case
where a pass performed zero operations and therefore never called `MFTLObjectSetCounts` at all. This
is the hook the review's topology test would have missed on its own reasoning, and gained only by
accident.

One per-body bitset then carries both votes — owner-changed **or** dirty — through a single
`MPI_BOR`, as the review already proposes (`redistribution_optimizations_review.md:44`). The stamp is
cleared per body at the end of a successful rebuild.

## 7. Two corrections to the review's supporting claims

**Ghost flags are not all refreshed for free.** The review states (`:42`) that a full rebuild replaces
every ghost over the wire each step, so ghost `flags` are refreshed for free and only a skip would
freeze them. That is true for ghost **triangles** and their half-edges, which are always dropped and
re-received (§5). It is false for ghost **vertices**: kept entities are entered into the arrival table
first (`redistribute.c:1229`), and an arriving copy whose gid is already present is dropped
(`:1243-1247`, *"the first copy wins and the rest are dropped"*). A retained ghost vertex therefore
keeps its **local** record — flags included — across every redistribution today. That weakens the
premise of §8 below rather than the skip proposal, but it should not be carried forward unexamined.

**`--digest` is sufficient for this hazard and not in general.** The review offers
`parallel_advection_redistribute_list --digest` bit-identity as the acceptance criterion (`:56`). For
the §5 break it works, and it works well: the digest hashes each ghost's `global_id` and `owner_proc`
(`examples/parallel_advection_redistribute_list.c:320-338`), so a stale ghost triangle id changes both
the set digest and the order digest immediately. But everything else about a ghost is inside the
`if (owned)` branch — flags, positions, and the whole twin structure are hashed for owned entities
only. So the digest would **not** catch a stale ghost flag, a stale ghost position, or a wrong ghost
twin. It is the right acceptance test for the id hazard and an incomplete one for the ghost-staleness
class of hazards as a whole.

## 8. A defect found while checking the above

Out of scope for the skip, recorded here because it was found on the way and nothing covers it.

`MFTLDetectFeatures` reclassifies over the **full** pool, ghosts included — `n_v = obj->verts.n`
(`src/remeshing/remesh_smooth.c:145-146`), cleared for every vertex at `:153`, and
`MFTL_FLAG_FEATURE` / `MFTL_FLAG_CORNER` set at `:307` and `:316`. The redistribution halo is
vertex-complete only for **owned** vertices (`remesh_parallel.c:86-88`), so a ghost vertex's count of
incident feature edges is taken over a partial fan and can put it in a different class than its owner
computes. The half-edge half of the function has a guard for exactly this shape of problem
(`remesh_smooth.c:198-204`, the `MFTLHeIsGhost` branch, whose comment explains that "no twin here" and
"mesh boundary" are different statements); the vertex classification has no equivalent.

The divergence survives the closing redistribution, because a retained ghost vertex keeps its local
record (§7). It is harmless inside a remeshing pass — every ghost is frozen, and both the collapse gate
and the smoother's skip mask (`remesh_smooth.c:554`) reject frozen vertices before any feature flag is
consulted — and **no failure has been observed**. It is recorded as a latent inconsistency, not as a
bug with a symptom.

Nothing tests it. `ghosts_agree_with_owners` (`tests/test_remeshing_parallel.c:681-723`) is the test
that would, and it compares **positions only** (`:708-714`); its own header comment is explicit that
the claim it certifies is about positions. Worth noting alongside: three documents cite that test under
a name it does not have — `test_every_ghost_agrees_with_its_owner`, at `src/remeshing/README.md:939`,
`CLAUDE.md:352` and `docs/redistribution-architecture-recommendation-reviewed.md:809`.

## 9. What to measure before any of this is built

The review's step 1 stands unchanged and is the gate on everything above:
add `n_objects_skippable` to `MFTL_RedistributeStats` (`include/mftl_communication.h:855-986`) as a
**counter only**, compute the decision, do not act on it, and run
`examples/parallel_advection_redistribute_list.c` once.

The reason is that the number the whole idea turns on has never been measured. The 95 %-unchanged
figure in `docs/redistribution_optimizations.md` is per **entity**; what decides whether this is a 1.9×
win or a rounding error is the skippable **body** fraction, and a body straddling a cut plane almost
certainly has *some* entity cross every step. The review estimates 60–75 % for the 1000-body case and
flags its own estimate as arithmetic on a radius/cell-width ratio rather than a measurement — the same
kind of geometric prediction that CLAUDE.md records over-predicting `frozen_fraction` by a factor of
two.

And the standing warning from §7c–7d of `docs/performance-analysis.md` applies here as much as
anywhere: the last five obviously-correct fixes removed 2.6× the allocator traffic and made the phase
no faster, because macOS hands out large blocks as lazily-mapped zero pages. Anything that is not a
reduction in *touched bytes* should be assumed to buy nothing until an interleaved A/B at ≥ 200 steps
says otherwise. A skip is a reduction in touched bytes, which is the reason to expect it to work — but
only for as many bodies as actually qualify.

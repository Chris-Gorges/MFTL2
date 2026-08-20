`src/remeshing/` — Isotropic surface remeshing {#mftl_mod_remeshing}
==============================================

**Public header:** `include/mftl_remeshing.h` · **Layer:** 3 ·
**Module group:** @ref mftl_remeshing

The largest module, and the only one whose MPI is confined to a single file by a
**build-time** check.

| File | Job | MPI? |
|---|---|---|
| `remesh_ops.c` | split, collapse, flip — an approval predicate and an execution function each | no |
| `remesh_smooth.c` | feature detection, and the three-subspace tangential smoother | no |
| `remesh_compact.c` | remove dead entities, re-establish the ownership prefix, re-densify global ids | no |
| `remesh_driver.c` | the sweep loop, and the dispatch on rank count | no |
| `remesh_parallel.c` | the shift/freeze phase schedule — move the **planes** | no |
| `remesh_dispersed.c` | body centering — move the **bodies** | no |
| `remesh_comm.c` | **every MPI call the module makes** | **yes** |

`remesh_internal.h` holds the shared helpers; `remesh_comm.h` declares the
communication layer and the shifted ownership rules.

---

## The serial algorithm {#rem_algorithm}

Botsch & Kobbelt's incremental isotropic remesher. One iteration is:

```
DETECT → SPLIT → COLLAPSE → COMPACT → FLIP → SMOOTH
```

against a target edge length $L$, with

$$
  \ell_{\min} = \tfrac45 L, \qquad \ell_{\max} = \tfrac43 L
$$

### The bounds are not tuning

Splitting an edge of length $\ell$ produces two of length $\ell/2$. For a split to
immediately create a collapsible edge we would need $\ell > \tfrac43 L$ **and**
$\ell/2 < \tfrac45 L$, i.e. $\ell < \tfrac85 L$. Conversely a collapse merges two
short edges into one of length at most $\ell_1 + \ell_2 < \tfrac85 L$, which is
below $\ell_{\max}$ only if the bound holds.

$[\tfrac45 L, \tfrac43 L]$ is the widest pair for which the two sweeps cannot
oscillate. Any wider and split/collapse ping-pong forever; any narrower and the
mesh is over-constrained.

### The sweep order is not arbitrary {#rem_order}

- **SPLIT before COLLAPSE**, because splitting is the operation that *cannot
  fail*: no link condition, no fold test. Doing it first gives the collapse sweep
  a mesh where the short edges are genuinely short rather than merely short
  relative to their over-long neighbours.
- **COMPACT between COLLAPSE and FLIP.** MFTL 5 compacted once at the very end,
  so flip and smoothing ran over a pool containing dead entities and had to defend
  against them individually. The payoff is not tidiness: the smoother calls
  `MFTLObjectRefreshFrames` and `MFTLComputeVertexNormals`, which **stream the
  whole pool** and would otherwise accumulate dead triangles into live vertices'
  normals.
- **FLIP before SMOOTH**, because the valence criterion is combinatorial — it
  reads the connectivity the previous two sweeps produced, and smoothing does not
  change connectivity. Running it after would delay its effect by an iteration.
- **DETECT at the top of every iteration**, not once per call — see
  @ref rem_features.

### Every sweep takes a fresh snapshot of the half-edge count

`MFTLSplitEdge` appends triangles, so $3 n_{\text{tris}}$ grows **during** the
split sweep. The bound is captured once, before the loop, **on purpose**: the
half-edges the sweep creates are the two halves of an edge it has just decided was
too long, and re-examining them in the same sweep is how a remesher turns one long
edge into a burst of tiny ones. They are examined on the next iteration, measured
against a mesh that has been smoothed in between.

---

## What implicit indexing does to a topological operator {#rem_operators}

Half-edge $3t+k$ **is** corner $k$ of triangle $t$. An operator therefore cannot
allocate a half-edge, move one between triangles, or patch a `next` pointer —
none of those are representable. What it does instead:

> **Rewrite whole triangles, then re-link the patch's outer twins.**

A triangle is written by storing three *target* vertices, and the patch is closed
by re-pointing the twins of the half-edges on its rim. Three consequences, and
they are why these operators are **shorter** than MFTL 5's rather than longer:

| Operator | Allocation | Work |
|---|---|---|
| **FLIP** | none at all | both triangles rewritten in place, four rim twins re-pointed. No `next` cycle to repair — which was most of MFTL 5's flip. |
| **SPLIT** | one triangle per side (two interior, one at a boundary) | its three half-edges come with it, at the right indices, by construction |
| **COLLAPSE** | **cannot free anything** | a triangle's half-edges *are* its storage, so it marks #MFTL_FLAG_DEAD and compaction removes them later |

A half-edge's global id is $3\,t_{\mathrm{gid}} + k$, a function of the **slot**,
never of the logical edge — so rewriting a triangle leaves the ids alone and the
invariant holds for free.

@warning What does **not** come for free is the **flag word**. When triangle $t$ is
         rewritten, the logical edge that used to sit in slot 1 may end up in slot
         2, and #MFTL_FLAG_FEATURE / #MFTL_FLAG_BOUNDARY have to travel with the
         **edge**, not with the slot. Every operator captures the rim flags before
         it writes anything and replaces them afterwards.

### Approval and execution are separate, on purpose

The predicate is the entire correctness argument; the execution assumes it has
already been made. A distributed driver needs to evaluate predicates on one rank
and replay executions on another, and that is only possible if they share no state.

### Every clause of `MFTLCollapseApproved` is silent when removed {#rem_approval}

| Clause | What its absence produces |
|---|---|
| **link condition** — endpoints share exactly two neighbours (one at a boundary edge) | welds two sheets into a non-manifold vertex, after which every one-ring walk from it silently visits half the fan |
| **wing valence** — both wing vertices keep valence ≥ 3 | a **tetrahedron passes the neighbour count and still dissolves** |
| **stretch bound** — no edge of the merged vertex exceeds $\ell_{\max}$ | split and collapse oscillate forever |
| **fold-over** — no surviving normal turns by more than `max_normal_deviation_cos` | an inverted triangle |
| **feature/boundary link** | two vertices on a feature curve joined by an edge **not** on that curve are a chord across the surface; collapsing it **pinches the curve inwards** |

The last two are easy to omit because the obvious form of the link condition does
not imply them. On an open patch the feature clause's absence serrates the rim; on
a cube it merges two sharp edges into the middle of the face between them.

Guarded by `test_collapse_refuses_to_dissolve_a_tetrahedron` and
`test_remesh_of_an_open_patch_keeps_its_rim`.

The fold-over guard has a measured second role: **worth nothing on a correct
configuration, but it bounds a runaway once features are spurious** — 16 702
triangles against 101 048.

### Valence optimisation

A flip is accepted on a **strict** decrease of

$$
  E = \sum_{v \in \{a,b,c,d\}} \big(\deg(v) - \deg^{*}(v)\big)^{2},
  \qquad \deg^{*} = \begin{cases} 6 & \text{interior} \\ 4 & \text{boundary} \end{cases}
$$

$\deg^{*} = 6$ is the valence of a regular triangulation of the plane; 4 is its
half. Strictness matters — an equality would let two flips alternate forever.

Additionally rejected when the two opposite vertices are already adjacent (the
flip would create a duplicate edge), and when it would turn a normal past the
fold-over threshold, which is what a flip across a **non-convex** quad does.

---

## Features and smoothing {#rem_features}

`MFTLDetectFeatures` produces two outputs from two linear passes and **no fan
walks at all**:

- **half-edges** — #MFTL_FLAG_FEATURE on both directions of every edge whose
  dihedral angle exceeds the threshold, and on every mesh-boundary half-edge
  unconditionally. Cleared elsewhere, so a fold that has since been smoothed out
  stops being a feature.
- **vertices** — FEATURE for exactly two incident feature edges continuing roughly
  straight through the vertex (a *ridge point*); CORNER for one, for three or
  more, or for two that turn by more than `feature_corner_angle`; neither for zero.

Doing the vertex classification **here** rather than in the smoother is what keeps
the smoothing sweeps free of fan walks — the smoother runs several times per
iteration and would otherwise re-derive the same counts each time.

The count is over **undirected** edges, visited once through the half-edge of
smaller index, which is also what makes the boundary case come out right: a fan
walk over *outgoing* half-edges misses the closing edge of an open fan entirely,
so a boundary vertex counted that way is short by one and lands in the wrong class.

### Detection must describe the mesh being SMOOTHED

`MFTLRemeshObject` calls `MFTLDetectFeatures` **twice per iteration**.

A split of a rim edge creates a midpoint that *is* on the rim. A vertex the
smoother believes to be free gets the 2D tangent-plane treatment and slides
straight off it — after which the surface has a serrated notch **no structural
check can see**, because the result is a perfectly valid manifold with a perfectly
valid boundary.

MFTL 5 detected once, so from the first split onwards the smoother was classifying
vertices against a feature set describing a mesh that no longer existed.

### Two feature edges make a ridge point only if the curve runs STRAIGHT

Counting alone makes the four corners of an open patch ridge points, free to slide
along the diagonal between their neighbours — off the patch.

`feature_corner_angle` is a **separate** threshold from `feature_angle_threshold`
because it applies where that one does not: **a mesh boundary is a feature curve
at any dihedral threshold, including a disabled one.**

### Smoothing is a projection {#rem_smoothing}

The feature class chooses the subspace the Laplacian displacement is projected
onto. Stated that way, the three cases are one idea:

| Class | Subspace | Update |
|---|---|---|
| free | 2D tangent plane | $\mathbf{d} \leftarrow (\mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^{\!\top})\,\mathbf{d}$ |
| ridge | 1D curve tangent | $\mathbf{d} \leftarrow (\hat{\mathbf{t}}\hat{\mathbf{t}}^{\!\top})\,\mathbf{d}$ |
| corner | 0D | $\mathbf{d} \leftarrow \mathbf{0}$ |

MFTL 5 moved a ridge vertex **to** the midpoint of its two feature neighbours —
the full displacement, no projection, ignoring its own relaxation factor. That is
a **chord cut**: it pulls the vertex off the curve towards the straight line
between its neighbours, so a repeatedly smoothed ridge **shrinks**. A regular
24-gon collapses to 0.84 of its radius in 20 sweeps — a large systematic error,
applied precisely where the geometry was supposed to be *preserved*.
`test_a_regular_ridge_is_a_fixed_point` asserts drift below $10^{-12}$.

The free case's normal comes from `MFTLComputeVertexNormals`, refreshed each
sweep, which is both more accurate and **faster** than accumulating a cross
product per vertex over its one-ring: it streams `tris.edge` with no fan walk and
no dependent loads (measured at 3.4×).

Every difference is a minimum-image vector and every result is wrapped, so the
smoother is correct across a periodic seam. A per-vertex step clamp
(`smooth_max_step_fraction`) costs one comparison against a length the smoother
has already computed — the tangential projection makes a fold-over *unlikely*
rather than impossible, because the projection is against a normal that is itself
meaningless where the one-ring is already degenerate.

### Feature detection on a smooth surface is a bug, not a safety net {#rem_features_off}

The LeVeque test's poor final mesh was caused by **feature detection on a smooth
surface**, not by any single operator.

A sphere has no features, so **every flagged edge is a false positive**, and each
removes smoothing degrees of freedom from vertices that needed them. Four spurious
edges are enough: at the step they appear, the worst triangle in the mesh drops
from **0.797 to 0.045** and then **freezes there**, because a constrained vertex is
simultaneously exempt from smoothing, from collapse and from flip.

| `feature_angle_threshold` | final min quality |
|---|---|
| 180 (detection off) | **0.76** |
| 45 | 0.0002 |

Monotone across five thresholds.

@warning **Set `feature_angle_threshold` to 180 for any smooth closed interface.**
         A mesh boundary is still treated as a feature at that setting, because it
         is a feature of the *surface* rather than of its curvature.

---

## Compaction {#rem_compaction}

A collapse cannot free anything, so **compaction is not an optimisation here — it
is the deallocator.**

Two properties make the pass simpler than MFTL 5's:

- **Half-edges move as triples.** `he[3*t_new + k] = he[3*t_old + k]`, so the
  implicit indexing invariant survives by construction, and so do the implicit
  half-edge global ids. Neither needs a fix-up pass. MFTL 5 compacted half-edges as
  an independent array with its own map, which is why it needed `next_idx` and
  `triangle_idx` remapped too.
- **One allocation per pool.** Compacting a pool is a loop over its sub-arrays, and
  a sub-array someone forgot **cannot exist** — it would have to have been
  allocated separately, and nothing outside `src/object/object.c` can do that.

### `out_he_idx` is re-derived, not remapped

Remapping would faithfully preserve whatever each operator left behind —
**including an index the operator failed to repair**. Re-deriving it from the
surviving half-edges is the same single-pass rule `MFTLBuildTopologyFromIndexList`
uses, costs $O(H)$ rather than $O(V)$, and cannot be stale.

It gives up nothing: the fan walkers rewind to the start of an open fan themselves,
so *which* outgoing half-edge a vertex names has never mattered.

### It is where the ownership prefix is re-established

An operator appends, and an append lands at the end of the pool — **past the
ghosts**. So during a parallel sweep the pool is

```
[ owned ...... | ghosts ...... | created this sweep ...... ]
               ^ n_local
```

which violates @ref arch_inv_prefix for exactly as long as the sweep runs. This
function ends that window, and it is the natural place: it is already a full
permutation of both pools with a full index remap behind it, so restoring the
prefix costs one more block move.

That is also why the parallel gate is #MFTL_FLAG_FROZEN rather than
`index >= n_local` — during the sweep the index test is **wrong**.

### Renumbering, and why it has to happen

Splits hand out fresh ids and collapses retire them, so ids stay unique but stop
being **dense**. Density is not cosmetic: `src/communication/` sizes its arrival
lookup tables from `next_*_gid`, so holes accumulated over a few hundred timesteps
become an unboundedly growing table for a mesh of constant size.

The cost is that **pool indices and global ids are both unstable across a remesh**,
so nothing external may hold either.

---

## Distributed: two strategies, one ratio {#rem_parallel}

An edge whose two endpoints are owned by different ranks cannot be collapsed by
either of them: each holds half of the one-ring the link condition is a statement
about, and each would have to rewrite records the other owns.

The obvious answer is a protocol — lock the edge, ask the neighbour, agree, act —
and it is the **wrong** one. It puts a round trip inside the innermost loop of a
remesher, and it makes the *result* depend on message arrival order, so the same
mesh remeshes differently on the same hardware twice in a row.

Two answers that remove the problem instead of managing it:

| | `remesh_parallel.c` — SPANNING | `remesh_dispersed.c` — DISPERSED |
|---|---|---|
| regime | $R_{\text{body}} \gg h$ | $R_{\text{body}} \ll h$ |
| what moves | the **cut planes** | the **bodies** |
| phases | 4–17 measured ($P = 1..8$) | 1 |
| migrations | one per phase, plus one | exactly **2** |
| frozen | $O(\delta/h)$ of the mesh | **nothing** |
| id cost | one Exscan + Allreduce per phase | one `Allreduce(MAX)`, total |
| weakness | phase count at large $P$ | load balance at $N_{\text{bodies}} < P$ |

`MFTLRemeshChooseStrategy` measures the bodies and decides.
`dispersed_radius_fraction` defaults to **0.4, not 0.5**, and the margin is not
arbitrary: object-centering is a 26-neighbour exchange exactly while every
triangle lies within one rank box of the centroid, and at $R = 0.5h$ a body
centred on a corner reaches two boxes away on the diagonal and the redistribution
falls back to the **global** communicator — correct, but an all-to-all where the
design promised a stencil. The gap absorbs one pass's worth of growth.

@warning **Both drivers must be reached by every rank.** The choice is made from
         reduced data and never from anything local. Two ranks entering different
         drivers is an immediate deadlock with no diagnostic.

**They share no code past `MFTLRemeshIterate`** — different migrations, different
id schemes, different convergence criteria. That is what makes them each other's
cross-check: `test_the_two_strategies_reach_the_same_mesh` is evidence about the
**answer** rather than about one implementation. Its tolerance is 4 % on the
triangle count and 1 % on the volume and **must not be tightened to equality**:
split and collapse are order-dependent, so the two legitimately land on different
triangulations of the same surface at the same resolution. Measured 0.4–2.5 % and
0.3 % at $P = 1, 2, 4, 8$.

---

## The spanning driver {#rem_spanning}

> **Don't move the mesh to suit the cut planes. Move the cut planes.**

Run the untouched serial kernel on the **interior** of each rank's share, where
the whole modified neighbourhood is exclusively that rank's; refuse every
operation whose footprint reaches the partition boundary; then move the planes by
half a box and run again. What was on the boundary is now in the middle of
somebody's share.

Nothing is negotiated because nothing overlaps, and the answer does not depend on
timing because no two ranks ever touch the same entity.

### The geometric guarantee

Let $\delta$ be the half-width of the frozen band and $h$ the rank box width. A
point frozen in the canonical placement is at distance $\le \delta$ from a plane,
hence at distance $\ge h/2 - \delta$ from the nearest plane of the half-shifted
placement. It is interior there iff

$$
  \frac{h}{2} - \delta > \delta
  \quad\Longleftrightarrow\quad \delta < \frac{h}{4}
  \quad\Longleftrightarrow\quad \boxed{h > 4\delta}
$$

That is the **per-axis** statement. In 3D an entity can sit near the planes of
one, two or three axes at once, and the diagonal shift $(1,1,1)$ only rescues the
first case:

| Case | Near | Measure | Phases needed |
|---|---|---|---|
| **slab** | 1 axis | $O(\delta/h)$ | 2: $(0,0,0)$, $(1,1,1)$ |
| **tube** | 2 axes | $O((\delta/h)^2)$ | 4: the even-parity set |
| **corner** | 3 axes | $O((\delta/h)^3)$ | 8: all of $\{0, h/2\}^3$ |

because an entity near the $x$ and $y$ planes but not $z$ is still near the
*shifted* $x$ and $y$ planes after $(1,1,1)$ — both moved, and it was between them.
Only a shift that moves **one** of the two rescues it, which is what the
odd-parity phases are for.

The phase table is ordered so the phases that matter most run first: two phases
handle all but $O((\delta/h)^2)$ of the mesh, and phases 3–8 typically find a
handful of entities each and exit almost immediately.

### "Frozen" is derived, not communicated, and it is exact {#rem_frozen}

$$
  \text{an owned entity is frozen} \iff \text{some other rank holds a copy of it}
$$

and this rank can evaluate the right-hand side **alone**, because it is the same
rule the redistribution used when it decided what to send:

$$
  \text{triangle } t \text{ is replicated} \iff
  \text{a corner of } t \text{ is a ghost} \ \lor\ \text{an edge neighbour is a ghost}
$$

With ownership as a prefix, "is a ghost" is `index >= n_local`. **Two streaming
reads, six integer compares, no distance test, no tolerance, no message.**

Being exact rather than conservative is what makes the dilation the implementation
plan called for unnecessary, and the argument is short enough to state in full:

1. A vertex is frozen if **any** incident triangle is. So an unfrozen vertex has
   every incident triangle unfrozen — owned, replicated nowhere.
2. An unfrozen triangle has no ghost corner, so all three of its corners are
   **owned**. With 1: every vertex an operator reads while judging an operation on
   unfrozen entities is owned by this rank.
3. The redistribution halo is **vertex-complete**, so every triangle incident to
   an owned vertex is present locally. With 2, every fan walk an approval predicate
   performs traverses a **complete** fan and gets the same answer it would on the
   whole mesh.

And a fourth consequence, which is what removes the per-phase ghost sync:

> **Every ghost in a rank's pool is frozen on its owner.** It is here because this
> rank owns a corner or an edge neighbour of it, which from the owner's side *is*
> the definition of replicated. So ghost copies are still exact when the sweeps
> finish, and a phase needs **no ghost position sync** before the next migration.

`test_every_ghost_agrees_with_its_owner` checks that to the bit, and
`frozen_matches_replication` checks the equivalence in **both** directions — the
"replicated but not frozen" direction is the fatal one, and its symptoms appear
one or two migrations after the cause.

### A shifted cut plane needs a FLOOR, not a truncation {#rem_shift_floor}

`MFTLRemeshOwnerProc` biases by `cuts` before the `(int)` cast on a periodic axis.
Without the shift the argument is non-negative for any wrapped position, so
`(int)` **is** `floor`; with a shift of up to $h/2$ it reaches $-0.5$, where
`(int)` truncates *towards zero* and attributes the sliver
$[\text{min}, \text{min}+\text{shift})$ to cell 0 instead of to cell
$\text{cuts}-1$ across the seam.

@warning What makes that hard to catch is that **the wrong rule is still a rule.**
         Every rank computes the same answer, the partition stays consistent, and
         every structural invariant still holds — cell 0 just grows to $1.5h$ and
         cell $\text{cuts}-1$ shrinks to $0.5h$, which is exactly what a
         non-periodic axis does on purpose. The one thing that changes is that
         **the seam plane never moves**, so an entity on it is frozen in every
         phase and ends up in a permanently un-remeshed band.

A body away from the seam never enters the sliver, so both original spanning tests
passed under the broken rule. Found by **mutation testing**, guarded by
`test_a_sphere_on_the_periodic_seam_remeshes_everywhere`, which fails on
convergence *and* reports the sphere as an open surface with 78 boundary
half-edges.

The same reasoning applies one level up: `MFTLRemeshMeasureBodies` averages a
periodic coordinate as an **angle** (circular mean, `atan2` of the area-weighted
mean direction), because the arithmetic mean of a seam-straddling body's
coordinates lands in the middle of the domain and its radius comes out as half the
box. Nothing fails — the body is simply classified SPANNING forever and takes the
phase schedule it does not need.

### A rank must not judge work it only holds as a GHOST {#rem_certificate}

The first convergence certificate counted *"approved, then refused by the freeze
gate"*. It reported 3 permanently blocked operations on a $P = 4$ sphere that was
in fact fully converged: a rank examines every half-edge in its pool, ghosts
included, and around a ghost vertex its one-ring is **incomplete**, so it approves
a collapse the owner correctly rejects.

The gate therefore runs **before** the approval (it is cheaper anyway), and the
certificate is `count_hidden_edges` — out-of-band edges with a frozen endpoint,
which needs no fan walk and so **cannot be given a wrong answer by a partial
view**.

Convergence is `ops == 0 && hidden == 0`, with a full-table streak of silent
placements as the fallback for the one false alarm `hidden` still has: an
out-of-band edge no operator would fix anyway, sitting on a band.

### Coarsening: a correctness escape hatch, not a performance feature

Raise the rank count on a fixed domain and $h$ shrinks until $h > 4\delta$ fails.
Past that point the schedule **degrades silently**: a lattice of bands along the
cut planes that no placement ever offers to anyone.

`parallel_coarsen` merges `group[k]` boxes per axis into one, multiplying the box
width by the same factor. It needs **no second communicator**, because
redistribution already accepts an arbitrary destination per entity
(#MFTL_OwnerMap) — a coarser decomposition is just a different *owner function*.

The price is exactly what it sounds like: $1/(g_0 g_1 g_2)$ of the ranks do **all**
the sweep work. Two cheaper answers exist and should be preferred: remesh to a
**longer target length**, or use the dispersed strategy if the bodies fit in a rank
box (it has no band and so no precondition).

The factors are **per axis**, because the precondition is per axis: a domain thin
in $x$ only costs 2× parallelism rather than 8×.

---

## The dispersed driver {#rem_dispersed}

For a body that fits inside a rank box the question never arises: migrate the
whole body onto one rank and there are **no split edges at all**.

The dispersed regime is what a real emulsion, spray or bubbly flow is in, and
there the difference is not marginal: a pass over ten thousand droplets costs
**two migrations instead of thirty-four**.

### Why the ids need almost nothing

The spanning driver's most delicate machinery is the per-phase `MPI_Exscan` that
makes concurrently-created global ids disjoint. **None of it is needed here**, and
the reason is worth stating because it is easy to add back out of habit:

> After centering, exactly **one** rank holds any entity of a given body, so that
> rank's own counter is already globally unique **for that body**.

Two ranks remeshing two different bodies cannot collide, because ids live in a
per-body space. So the operators run completely untouched,
`MFTLRemeshRenumberGlobalIds` makes the ids dense on the owner, and the only thing
the other ranks need is the resulting **counts** — one `Allreduce(MAX)` over
$2 N_{\text{bodies}}$ ints, because a rank holding nothing contributes zero and the
owner's value wins.

### The one thing that must not happen

The locality check comes **before** any remeshing. If some body did not become
local, this pass must not have modified anything yet — the fallback hands the whole
ensemble to the spanning driver, and a half-remeshed ensemble entering it would
carry per-rank id blocks that no Exscan agreed on.

The ordering costs **nothing**: the radii it tests were already reduced for the
strategy decision, so the check is a loop over replicated data.

---

## All the MPI is in one file {#rem_mpi_confined}

Every MPI call the remesher makes is in `remesh_comm.c`.
`test_remeshing_mpi_is_confined` — a **CMake source check**, not a runtime one —
fails the build if that stops being true, because no runtime test can see it.

The rule exists because the first version broke it in two places that were
individually small, correct and fully tested, and **collectively the wrong shape**:
a feature used a few times per hundred timesteps paid for once per timestep, in the
two loops the library is most careful about.

- `MFTLGetOwnerProc` gained a per-axis cut-plane offset and a periodicity branch.
  That function **is** the body of `compute_owners`, i.e. it runs once per vertex
  and once per triangle on every redistributed timestep. **Measured at 1.70× —
  1.79 → 3.05 ns/entity, 42 → 89 instructions for the same loop** (`-O3`, Apple
  silicon, best of seven, interleaved). The end-to-end redistribute timing on the
  development laptop varies by 2× between sessions and cannot resolve it; the
  isolated number is the honest one.
- The per-timestep ghost exchange gained a third field, *"tell me my new global
  id"*, and with it two `field ==` tests inside the loop that applies replies.

Both are reverted. What replaces them is #MFTL_OwnerMap — **one** seam, a NULL
pointer on the production path, and one predicted branch per body hoisted out of
both loops. It is also what the dispersed strategy needs, and that one is not a
plane test at all — so the seam had to be an ownership **array** rather than a
cleverer ownership function.

`MFTLRemeshWallTime` looks like a pointless wrapper around `MPI_Wtime` and is not:
the confinement check is a source scan for `MPI_*(`, so a bare `MPI_Wtime()` in
`remesh_parallel.c` **fails the build**. The rule is worth more than the
indirection, and the call is not on any hot path.

---

## `parallel_phase_budget` is the one parameter a solver must set {#rem_budget}

The default (32) means **run to a fixed point**: keep placing planes until no rank
anywhere wants to perform an operation. That is right for a **one-shot** remesh —
bringing a mesh to a target length it is far from — and it is measurably **wrong**
inside a per-timestep loop, in both directions at once.

Measured on the LeVeque deformation, 150 remeshed timesteps, sphere in a
time-reversed vortex. The budget is the only thing varied:

| budget | $P{=}1$ phases | $V/V_0$ | min $q$ | ms/step | $P{=}4$ $V/V_0$ | min $q$ | ms/step |
|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.751 | 0.876 | 7.1 | — | — | — |
| 2 | 2.0 | 0.769 | 0.856 | 20.4 | 0.745 | 0.897 | 12.9 |
| **4** | 3.7 | 0.733 | 0.836 | 37.8 | **0.771** | **0.879** | **16.7** |
| 8 | 5.7 | 0.683 | 0.459 | 52.7 | 0.670 | 0.796 | 35.4 |
| 32 | 13.3 | 0.682 | 0.278 | 99.0 | 0.639 | 0.621 | 84.7 |

Serial reference (`MFTLRemeshObject`): $V/V_0$ 0.751, min $q$ 0.876, 7.1 ms/step.

Two things to read off it. **Volume conservation and triangle quality are
non-monotone and get worse as the budget grows** — re-converging an already-good
mesh to a fixed point every timestep is not free polish, it is repeated
split/collapse churn that erodes an under-resolved surface. And the cost is linear
in the phase count, because every phase is a migration.

At budget 4 the distributed answer at $P = 4$ **matches or beats the serial one on
every correctness measure**. At the default it is worse on all of them and five
times slower.

| Use | Setting |
|---|---|
| **per-timestep loop** | 2–8. Four covers slab and tube conflicts and was the best measured compromise; eight is a full covering set of the table. `converged` will often be false and that is expected — the mesh is being *maintained*, not converged. |
| **one-shot remesh** | leave it at 32 or raise it, and check `converged`. A mesh far from its target needs several sweeps of the table: 4 phases at $P=1$, 6 at $P=2$, 19 at $P=4$, 17 at $P=8$ going 1280 → 3830 triangles. |

The default stays at 32 because the one-shot case **fails loudly** when the budget
is too small (`converged` comes back false) and the loop case **fails quietly**
when it is too large. A default that is wrong in the loud direction is the safer of
the two.

@note At a fixed budget of 4 the phase count **saturates** with rank count
      (3.83 → 3.96 from $P = 2$ to $P = 8$), overhead on a fixed problem grows
      33 % for 4× the ranks, and the distributed mesh is *better* than the serial
      one on every quality measure at an adequate resolution. Still open above
      $P = 8$ and on real hardware: the development machine has four performance
      cores, so $P = 8$ is oversubscribed and every absolute time above $P = 4$ is
      a lower bound.

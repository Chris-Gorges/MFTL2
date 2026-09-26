`src/remeshing/` — Isotropic surface remeshing {#mftl_mod_remeshing}
==============================================

**Public header:** `include/mftl_remeshing.h` · **Layer:** 3 ·
**Module group:** @ref mftl_remeshing

| File | Job | MPI |
|---|---|---|
| `remesh_ops.c` | split, collapse, flip: an approval predicate and an execution function each | no |
| `remesh_smooth.c` | feature detection, tangential smoothing | no |
| `remesh_compact.c` | remove dead entities, restore the ownership prefix, re-densify ids | no |
| `remesh_driver.c` | the sweep loop, the serial driver, the dispatch | no |
| `remesh_parallel.c` | SPANNING: shift the cut planes | no |
| `remesh_dispersed.c` | DISPERSED: move the bodies | no |
| `remesh_comm.c` | **every MPI call of the module** | yes |

`remesh_internal.h` holds shared helpers and the freeze gates; `remesh_comm.h`
declares the communication layer and the shifted ownership rules.

[TOC]

## Notation {#rem_notation}

| Symbol | Meaning | Where |
|---|---|---|
| $L$ | target edge length | `params.target_length` |
| $\ell_{\min}, \ell_{\max}$ | length band, $\tfrac45 L$ and $\tfrac43 L$ | `params.min/max_edge_length` |
| $P$ | rank count | `domain.nproc` |
| $h_k$, $h$ | rank box width on axis $k$; its minimum over cut axes | `MFTLDomainGetHMin` |
| $\delta$ | half-width of the frozen band | `stats.band_measured`, `band_bound` |
| $g_k$, $G$ | per-axis coarsening factors, $G = g_0g_1g_2$ | `MFTLRemeshCoarseningFactors` |
| $R_b$, $w_b$ | body radius, body weight (triangles) | `MFTLRemeshMeasureBodies` |
| $N_b$, $B$ | bodies this rank holds; bodies in the ensemble | `n_local_objects`, `n_bodies_global` |

---

## The serial algorithm {#rem_algorithm}

Botsch & Kobbelt's incremental isotropic remesher (SGP 2004). One iteration of
`MFTLRemeshIterate`:

```
DETECT → SPLIT (ℓ > ℓmax) → COLLAPSE (ℓ < ℓmin) → COMPACT → FLIP → DETECT → SMOOTH
```

repeated `params.iterations` times or until an iteration performs at most
`min_convergence_ops` operations. `MFTLRemeshObject` wraps it with an assertion
that the object holds no ghosts, the id renumbering and a closing geometry pass.

### The length band {#rem_band}

A split maps $\ell \mapsto \ell/2$, so an edge longer than $\ell_{\max}$ gives
halves longer than $\tfrac23 L$: $|\tfrac43 L - L| = |\tfrac23 L - L|$, i.e. a
split never moves an edge further from $L$. The band does not by itself exclude
split/collapse ping-pong — an edge in $(\tfrac43 L, \tfrac85 L)$ splits into
halves below $\ell_{\min}$ — and three mechanisms do:

1. every sweep bounds its loop by the half-edge count taken **before** it, so new
   halves are not re-examined until the next iteration, after smoothing;
2. `MFTLCollapseApproved` refuses a collapse that would create an edge longer
   than $\ell_{\max}$ (the stretch clause);
3. smoothing between sweeps pulls the length distribution into the band.

`MFTL_REMESH_CUSTOM` lets the caller set any band (then `MFTLRemeshParamsFinalise`).

### The sweep order {#rem_order}

- **SPLIT before COLLAPSE:** splitting cannot fail (no link condition, no fold
  test), so the collapse sweep sees edges that are genuinely short.
- **COMPACT right after COLLAPSE:** flip and smoothing never see a dead entity,
  so `MFTLComputeVertexNormals` can run over the whole pool.
- **FLIP before SMOOTH:** the valence criterion reads the connectivity just
  produced; smoothing does not change it.
- **DETECT twice:** splits and collapses change which edges exist. A split of a
  rim edge creates a midpoint on the rim; classified as free, it would slide off
  the rim, leaving a notch no structural check can see.

---

## Operators under implicit indexing {#rem_operators}

Half-edge $3t+k$ *is* corner $k$ of triangle $t$, so an operator cannot allocate
a half-edge or patch a `next` pointer. It **rewrites whole triangles and
re-links the twins on the rim of its patch**:

| Operator | Allocation | Work |
|---|---|---|
| FLIP | none | two triangles rewritten in place, four rim twins re-pointed |
| SPLIT | one triangle per side (with its three half-edges) | midpoint vertex appended, 2–4 triangles written |
| COLLAPSE | none, and nothing freed | one or two triangles and the origin vertex flagged `MFTL_FLAG_DEAD` for compaction |

Half-edge ids, $3\,t_\mathrm{gid}+k$, belong to the slot and stay valid. The
flag word does not: when a triangle is rewritten a logical edge may move to
another slot, so every operator captures the rim flags (FEATURE, BOUNDARY)
first and places them explicitly. A split's midpoint inherits the edge's
classification, so the output is consistent before the next detection.

Approval and execution are separate; the predicates carry the whole correctness
argument. SPLIT and FLIP are offered once per undirected edge (the smaller
half-edge index); COLLAPSE is offered **both directions**, because its feature
rules are asymmetric. `MFTL_FLAG_VISITED`, cleared once per iteration, keeps an
entity from being operated on twice before the next smoothing.

### The collapse predicate {#rem_approval}

Collapsing $h = (a\to b)$ retires $a$, moves $b$ to $p_m$ and rewrites every
triangle of $a$'s fan. With $N(v)$ the one-ring and $W = N(a)\cap N(b)$ the wing
vertices, in evaluation order:

| # | Clause | Statement | Without it |
|---|---|---|---|
| 1 | wanted | $\lVert e_h\rVert < \ell_{\min}$ | — |
| 2 | features | $a$ not a corner; $a$ on a feature curve ⟹ $b$ too; $a$ on the boundary ⟹ $b$ too | the curve or rim is dragged off itself |
| 3 | curve link | not ($h$ not a feature edge ∧ $a, b$ both on feature curves); not ($h$ interior ∧ $a, b$ both on the boundary) | a **chord** collapses: an open patch's rim is serrated, two sharp edges of a cube merge |
| 4 | link condition | $\lvert W\rvert = 2$ ($1$ at a boundary edge) | two sheets welded into a non-manifold vertex |
| 5 | merged valence | $3 \le \lvert N(a)\rvert + \lvert N(b)\rvert - \lvert W\rvert - 2 \le$ `MFTL_MAX_VALENCE` | untraversable vertex |
| 6 | wing valence | $\lvert N(w)\rvert \ge 4$ for $w \in W$ | a **tetrahedron** passes clause 4 and dissolves |
| 7 | stretch | $\lVert p_m - p_w\rVert \le \ell_{\max}$ for all $w \in N(a)\cup N(b)$ | split/collapse oscillation |
| 8 | fold-over | $\cos\angle(\mathbf n_\text{old}, \mathbf n_\text{new}) \ge$ `max_normal_deviation_cos` for every surviving triangle; degenerate results refused | inverted triangles |

$p_m = p_b$ if $b$ is a corner or $b$ alone lies on a feature curve or boundary;
otherwise the minimum-image midpoint (exact on a straight feature, $O(h^2)$ off a
curved one, acceptable because the edge is short). None of these clauses fails
loudly; clauses 3 and 6 are the ones the textbook link condition does not imply
(`test_collapse_refuses_to_dissolve_a_tetrahedron`,
`test_remesh_of_an_open_patch_keeps_its_rim`). The fold test is relative to the
triangle's own edges (squared sine of a corner angle against $10^{-15}$), so the
result is independent of the length unit:
`test_remeshing_is_independent_of_the_length_unit` requires a sphere remeshed at
scale $2^{-20}$ to equal the unit result times $2^{-20}$ bit for bit. The
fold-over guard also bounds a runaway once features are spurious (16 702
triangles against 101 048).

### Valence optimisation

A flip re-cuts the quad $(a,b,c,d)$, so $a, b$ lose a neighbour and $c, d$ gain
one. It is accepted only on a **strict** decrease of

$$
  E = \sum_{v \in \{a,b,c,d\}} \big(\deg v - \deg^* v\big)^2,
  \qquad \deg^* = 6 \text{ interior},\ 4 \text{ boundary}
$$

($\deg$ counts neighbours; 6 is the regular valence of the plane, 4 of the
half-plane). Equality would let two flips alternate forever. Refused for
feature and boundary edges, when $\deg a$ or $\deg b \le 3$, when $c$ and $d$ are
already adjacent (a duplicate edge), and when a new triangle would turn past the
fold threshold against the quad's normal (a non-convex quad).

---

## Features and smoothing {#rem_features}

`MFTLDetectFeatures` recomputes `tris.normal` and classifies, in linear passes
over undirected edges (a fan walk over outgoing half-edges misses the closing
edge of an open fan):

- an edge is a feature if $\hat{\mathbf n}_1\cdot\hat{\mathbf n}_2 < \cos\theta_\text{feature}$,
  or if it is a mesh-boundary edge (a missing twin on a *ghost* triangle is the
  halo rim, not a boundary; `MFTL_FLAG_BOUNDARY` decides there);
- a vertex with exactly two feature edges $\mathbf d_1, \mathbf d_2$ is a ridge
  point iff $-\mathbf d_1\cdot\mathbf d_2 / (\lVert\mathbf d_1\rVert\lVert\mathbf d_2\rVert) > \cos\theta_\text{corner}$;
  with one, three or more, or a sharper turn it is a corner.

`feature_corner_angle` is separate from `feature_angle_threshold` because a
boundary is a feature curve at any threshold: without it the four corners of an
open patch are ridge points that slide off the patch.

### Smoothing is a projection {#rem_smoothing}

| Class | Subspace | Displacement |
|---|---|---|
| free | tangent plane | $(\mathbf I - \hat{\mathbf n}\hat{\mathbf n}^\top)\,\mathbf d$ |
| ridge | curve tangent $\boldsymbol\tau$ | $(\boldsymbol\Delta\cdot\boldsymbol\tau / \lVert\boldsymbol\tau\rVert^2)\,\boldsymbol\tau$ |
| corner | none | $\mathbf 0$ |

For a free vertex $\mathbf d = \sum_i A_i\mathbf c_i/\sum_i A_i - \mathbf p$
over the incident triangles (`MFTL_SMOOTH_TRIANGLE_BASED`; area and centroid
offset both come from the two edge vectors leaving $\mathbf p$), or the umbrella
mean of the neighbours (`MFTL_SMOOTH_VERTEX_BASED`); $\hat{\mathbf n}$ is the
area-weighted vertex normal, refreshed every sweep, and a vertex whose normal is
zero stays put. For a ridge vertex with offsets $\mathbf d_1, \mathbf d_2$ to its
feature neighbours, $\boldsymbol\tau = \mathbf d_2 - \mathbf d_1$ and
$\boldsymbol\Delta = \tfrac12(\mathbf d_1+\mathbf d_2)$. Moving it *to*
$\boldsymbol\Delta$ instead cuts the chord: a regular 24-gon ridge shrinks to
$\cos^{20}(\pi/24) = 0.842$ of its radius in 20 sweeps
(`test_a_regular_ridge_is_a_fixed_point` asserts drift below $10^{-12}$).

The area weight is the mechanism that equalises triangle size; without it the
operator is nearly the umbrella one (`docs/analysis.md` §4: 6.0 % volume loss
against 3.5 %). Updates are **Jacobi** (independent of pool order), and the step
is

$$
  \mathbf p \leftarrow \mathbf p + \sigma\mathbf d,\qquad
  \sigma = \min\!\big(\lambda,\ \phi\,\ell^\text{fan}_{\min}/\lVert\mathbf d\rVert\big),
$$

with $\lambda$ = `smooth_relaxation`, $\phi$ = `smooth_max_step_fraction`: the
projection makes a fold unlikely, not impossible, where the normal is ill-defined.
All differences are minimum-image vectors and positions are wrapped. The sweep
reads ghost and frozen vertices but moves neither, which makes a distributed
sweep exact without communication.

### Feature detection on a smooth surface {#rem_features_off}

On a sphere every detected feature is a false positive, and it removes degrees of
freedom a vertex needed: in the LeVeque test four spurious edges dropped the
worst triangle from 0.797 to 0.045, where it froze, because a constrained vertex
is exempt from smoothing, collapse and flip. Final minimum quality 0.76 with
detection off against 0.0002 at 45°, monotone over five thresholds
(`docs/analysis.md` §4). **Set `feature_angle_threshold` to 180 for smooth closed
interfaces**; boundaries remain features.

---

## Compaction {#rem_compaction}

A collapse cannot free anything, so compaction is the deallocator. Half-edges
move as triples with their triangle, which preserves implicit indexing and the
implicit half-edge ids, and one allocation per pool means no sub-array can be
forgotten. `out_he_idx` is **re-derived** from the surviving half-edges rather
than remapped, so an index an operator failed to repair cannot survive.

During a parallel sweep appended entities sit past the ghosts,

```
[ owned ...... | ghosts ...... | created this sweep ...... ]
               ^ n_local
```

violating the ownership prefix (invariant 2 of `mftl_types.h`) until compaction
swaps the two trailing blocks. That is why the parallel gate is
`MFTL_FLAG_FROZEN`, not `index >= n_local`.

Splits create and collapses retire ids, so ids stay unique but stop being dense;
since redistribution sizes its arrival tables from `next_*_gid`, every pass ends
with dense ids. Pool indices and global ids are therefore both unstable across a
remesh: nothing external may hold either.

---

## Distributed remeshing: the obstacle {#rem_parallel}

An operator has a **modified set** (split, flip: the two incident triangles;
collapse: $a$, $b$ and $a$'s whole fan) and an **approval set** it must read
(for a collapse $N(a)\cup N(b)$ and both fans). A rank may act iff it owns the
modified set exclusively and sees the approval set completely. An edge whose
endpoints two ranks own satisfies neither; a lock/ask protocol would put a round
trip in the innermost loop and make the result depend on message timing. Two
exact strategies avoid negotiation altogether:

| | SPANNING (`remesh_parallel.c`) | DISPERSED (`remesh_dispersed.c`) |
|---|---|---|
| regime | $R_b \gtrsim h$ | $R_b \le 0.4\,h$ |
| what moves | the cut planes | the bodies |
| placements | ≈ 4 at budget 4; 4–19 for a one-shot remesh | 1 |
| migrations | one per placement (identity skipped), plus one | exactly 2 |
| frozen | $O(\delta/h)$ of the mesh | nothing |
| id cost | per-body prefix sums over the holders, per placement | one per-body MAX |
| weakness | placements, band precondition | idle ranks when $B < P$ |

They share no code past `MFTLRemeshIterate` (different migrations, id schemes
and convergence criteria), so `test_the_two_strategies_reach_the_same_mesh` checks
the *answer*: within 4 % in triangle count and 1 % in volume (measured 0.4–2.5 %
and 0.3 % at $P$ = 1, 2, 4, 8). The tolerance must not become equality: split and
collapse are order-dependent.

### Dispatch {#rem_dispatch}

```
MFTLRemeshEnsemble(ens, params, stats)                              COLLECTIVE
├─ nproc == 1 ─▶ MFTLRemeshObject per object       no MPI; converged = true
└─ parallel_strategy
   ├─ SPANNING  ─▶ MFTLRemeshEnsembleParallel
   ├─ DISPERSED ─▶ MFTLRemeshEnsembleDispersed      (measures the bodies itself)
   └─ AUTO      ─▶ MFTLRemeshChooseStrategyMeasured
                   ├─ MFTLEnsembleComputeTriangleGeometry               local
                   ├─ MFTLRemeshMeasureBodies    2 per-body holder reductions
                   │                             + 1 scalar MPI_MAX (r_max)
                   └─ r_max > dispersed_radius_fraction * box ?
                      ├─ yes ─▶ MFTLRemeshEnsembleParallel
                      └─ no  ─▶ MFTLRemeshEnsembleDispersedMeasured
                                (the measurement is handed on, not retaken)
```

The box is the narrowest rank box, or the coarsened box (@ref rem_yardstick).
`dispersed_radius_fraction` is 0.4 rather than 0.5 because at $R = 0.5h$ a body
centred on a box corner reaches two boxes diagonally and the centering migration
leaves the 26-neighbourhood; the margin absorbs one pass of growth. The decision
uses only reduced data (`r_max` is bit-identical everywhere): **two ranks in
different drivers deadlock without a diagnostic.**

---

## The spanning driver {#rem_spanning}

Run the unmodified serial kernel only where the whole modified neighbourhood is
this rank's alone, then move the cut planes by half a box and run again: what was
on a boundary is now in somebody's interior. Nothing overlaps, so nothing is
negotiated and the result does not depend on timing (digests are bit-identical on
repeated runs, `docs/analysis.md` §5.3).

### One pass {#rem_spanning_flow}

```
MFTLRemeshEnsembleParallel(ens, params, stats)                      COLLECTIVE
├─ 0  guards: band > 0 (else MFTL_ERROR); Debug: params and id counters agree
├─ 1  COARSENING PRE-PASS   if (parallel_coarsen || serial_gather) and P > 1
│     freeze + probe band and frozen share        2 small Allreduces
│     choose group g (GATHER, BAND, FROZEN triggers, cap)     @ref rem_coarsen
├─ 2  distinct placements: de-duplicate the phase table over active axes
│     (cuts[k]/g[k] > 1)                         ▸ n_distinct_placements
├─ 3  PHASE LOOP, at most parallel_phase_budget placements
│     ├─ MFTLRemeshPhaseShift: shift_k = s_k * g_k h_k / 2          local
│     ├─ geometry; MFTLRemeshMigrateShifted
│     │    per entity MFTLRemeshCoarseOwnerProc; 1 LOR: moved anything?
│     │    no ─▶ skip the migration (always for the first placement)
│     ├─ MFTLRemeshMarkFrozen, MFTLDilateFrozenBand, count_hidden_edges,
│     │  band probe; record id watermarks                            local
│     ├─ MFTLRemeshIterate, FULL iteration count, gated by MFTL_FLAG_FROZEN
│     ├─ assign_phase_gids: MFTLEnsembleExscanBodies (per body, holders)
│     │                     + MFTLRemeshSumInts (ops, hidden_long, hidden_short)
│     └─ converged?  ops == 0 and hidden == 0                   ─▶ done (exact)
│                    ops == 0, hidden_long == 0, streak == n_distinct ─▶ done
│                    hidden_long > 0 or ops > 0                 ─▶ streak = 0
└─ 4  CLOSING (order forced)
      redensify_global_ids: prefix sum of OWNED counts per body;
         MFTLRemeshSyncGhostIds (keyed on the OLD id); overwrite gid = base + slot
      MFTLRedistribute to the canonical planes (replaces stale ghost triangle ids)
      clear FROZEN, global counts, triangle geometry, profile (1 Allreduce MAX)
      warn once if h <= 4 delta and no coarsening happened
```

Per placement: one migration (unless the identity), one LOR, the per-body id
scan and one 3-int Allreduce. Everything else — the freeze classification and
every approval predicate — is local.

### The geometric guarantee {#rem_covering}

Placement $s\in\{0,1\}^3$ puts the planes of axis $k$ at
$x_k = \min_k + s_k h_k/2 + j h_k$. A point within $\delta$ of a plane of one
placement is at least $h_k/2 - \delta$ from every plane of the other, so it is
interior there iff

$$
  \frac{h_k}{2} - \delta > \delta \iff \boxed{h_k > 4\delta}.
$$

Under this condition a point is near the planes of at most one placement per
axis: for each axis there is at most one *bad* value $b_k$, and the point is
interior in placement $s$ iff $s_k \ne b_k$ on every axis where $b_k$ is defined.

### The phase table {#rem_phase_table}

| Case | axes in conflict | measure | placements that cover every case |
|---|---|---|---|
| slab | 1 | $O(\delta/h)$ | $\{000, 111\}$ |
| tube | 2 | $O((\delta/h)^2)$ | an orthogonal array of strength 2: $\{000, 011, 101, 110\}$ |
| corner | 3 | $O((\delta/h)^3)$ | all of $\{0,1\}^3$ (only $\bar b$ works) |

`MFTL_REMESH_PHASES` orders $\{0,1\}^3$ so that prefixes are useful:
000, 111 (all slabs), 011, 101, 110 (with 000: all tubes), 100, 010, 001 (all
corners). Two placements handle all but $O((\delta/h)^2)$ of the mesh; the tail
rows find a handful of entities and are cheap, because a silent
`MFTLRemeshIterate` stops after one iteration. On a grid with an uncut axis rows
coincide (2 distinct placements on $2\times1\times1$, 4 on $2\times2\times1$).

### The frozen share {#rem_frozen_share}

A frozen entity is migrated and then not touched, so the frozen share is the
share of every migration that buys nothing. The uniform model
$f = 1 - \prod_k (1 - 2\delta/h_k)$ gives 87.5 % at the limit $h = 4\delta$, and
**over-predicts by about two** in practice (71 % against 36.6 % measured at
$P = 27$) because a body does not sample the planes uniformly. Use the measured
`MFTL_RemeshStats::frozen_fraction`.

### "Frozen" is derived locally, and exactly {#rem_frozen}

$$
  \text{owned entity frozen} \iff \text{another rank holds a copy of it},
$$

and a rank evaluates this alone from the rule redistribution sends by:

$$
  t \text{ replicated} \iff \exists k:\ \texttt{tris.verts}[3t{+}k] \ge n^V_\text{local}
  \ \lor\ \texttt{tris.neighbors}[3t{+}k] \ge n^T_\text{local},
  \qquad
  v \text{ frozen} \iff \exists\, t \ni v \text{ frozen}.
$$

Two 12 B/triangle streams and six integer compares; `MFTL_ERROR` ($-1$) is below
every count, so a genuine boundary does not freeze a triangle. Exactness makes a
dilation unnecessary:

1. an unfrozen vertex has only unfrozen (owned, unreplicated) incident triangles;
2. an unfrozen triangle has only owned corners, so every vertex an approval reads
   is owned;
3. the halo is vertex-complete, so every fan an approval walks is complete and
   gives the answer it would give on the whole mesh.

Hence the gates are two flag tests: `freeze_blocks_pair` (split, flip: either
incident triangle frozen) and `freeze_blocks_collapse` (either endpoint frozen).
And **every ghost is frozen on its owner** (it is here because this rank owns a
corner or edge neighbour of it), so ghost copies are still exact after the sweeps
and no ghost position sync is needed inside the loop
(`ghosts_agree_with_owners` in `tests/test_remeshing_parallel.c`; `frozen_matches_replication` checks
the equivalence both ways — "replicated but not frozen" is the fatal direction).

### The band is measured {#rem_band_width}

The freeze rule's reach bounds $\delta \le (3 + \texttt{rings})\,\ell_{\max}$, but
the band actually observed on a sphere in $[-1,1]^3$ is $\approx 0.84\,\ell_{\max}$
for targets 0.06–0.50; at target 0.12 the bound predicts $4\delta = 1.92 > h = 1$
while the pass converges with nothing hidden. So `local_band_probe` measures it:
the minimum distance over axes to the **current** placement's planes, maximised
over frozen vertices and over all phases (a coarsening pass lengthens edges, and
the band grows: 0.216 at phase 0, 0.375 later at target 0.40).

### A shifted plane needs a floor {#rem_shift_floor}

`MFTLRemeshOwnerProc` computes $\sigma = (x_k - \min_k - \text{shift}_k)/h_k$.
Without a shift $\sigma \ge 0$ and `(int)` is $\lfloor\cdot\rfloor$; with a shift
$\sigma$ reaches $-0.5$, where `(int)` truncates towards zero and gives the
sliver $[\min_k, \min_k + \text{shift}_k)$ to cell 0 instead of cell
$\texttt{cuts}_k - 1$ across the seam. On a periodic axis the index is therefore
$\lfloor\sigma + \texttt{cuts}_k\rfloor \bmod \texttt{cuts}_k$. The wrong rule is
still a consistent rule — only the seam plane never moves, so entities on it are
frozen in every phase (found by mutation testing;
`test_a_sphere_on_the_periodic_seam_remeshes_everywhere`).

For the same reason `MFTLRemeshMeasureBodies` averages a periodic coordinate as
an angle,

$$
  \bar\theta_k = \operatorname{atan2}\Big(\sum_t A_t\sin\theta_{k,t},\ \sum_t A_t\cos\theta_{k,t}\Big),
  \qquad \theta = \frac{2\pi(c-\min_k)}{\text{size}_k};
$$

the arithmetic mean puts a seam-straddling body in the middle of the domain with
half the domain as radius
(`test_a_seam_straddling_droplet_is_not_measured_as_domain_sized`).

### Global ids {#rem_ids}

Ids must be unique across ranks and dense at the end (a rank-encoded 64-bit id
would be sparse by construction).

**Per placement.** The operators draw ids from the object's counter, so two
ranks that both split an edge of a body produce the same id — harmless, because
nothing communicates before the fix-up. Entities created in the phase are those
with $\texttt{gid} \ge w$, the watermark recorded before the sweeps. With $c_r$
the count this rank created and $\mathrm{Exscan}$ the exclusive prefix sum over
the body's holders in ascending rank order (`MFTLEnsembleExscanBodies`),

$$
  \texttt{gid} \leftarrow w + \mathrm{Exscan}(c_r) + \text{local ordinal},
  \qquad w' = w + \textstyle\sum_r c_r .
$$

A scan, not a reduction: two holders need *different* ranges, which no
combination provides. Its ranges tile $[w, w')$ exactly
(`test_the_scan_tiles_every_body_exactly_once`).

**At the end.** `redensify_global_ids` renumbers owned entities by slot,
$\texttt{gid}(v) = \mathrm{Exscan}(n^V_\text{local}) + v$ (likewise triangles,
half-edges $3\,\texttt{gid}(t)+k$), in a forced order: scan first; then
`MFTLRemeshSyncGhostIds`, keyed on the OLD id (renumbering first would make every
request unanswerable, and ghosts would silently keep stale ids); overwrite last.
Ghost triangles are not exchanged: the closing redistribution re-receives them,
which is why it comes after.

### The convergence certificate {#rem_certificate}

Counting "approved, then refused by the gate" reported 3 permanently blocked
operations on a converged $P = 4$ sphere: around a ghost vertex the one-ring is
incomplete, so a rank approves collapses the owner rejects. The gate therefore
runs first, and the certificate counts out-of-band edges hidden by it, each edge
attributed to the owner of the incident triangle with the smaller global id (so
it is counted once mesh-wide):

- $\text{hidden}_\text{long}$: $\ell > \ell_{\max}$ and an incident triangle frozen
  (the split gate);
- $\text{hidden}_\text{short}$: $\ell < \ell_{\min}$ and an endpoint frozen (the
  collapse gate).

| Exit | Test | Why sound |
|---|---|---|
| fast | ops $= 0$ ∧ hidden $= 0$ | every out-of-band edge was offered to a rank allowed to act, and it declined on merit |
| slow | ops $= 0$ ∧ hidden$_\text{long} = 0$ for `n_distinct` consecutive placements | a covering set: under $h > 4\delta$ every entity was interior in one of them |

$\text{hidden}_\text{long}$ is exact (an offered long edge is always split), so
non-zero resets the streak; $\text{hidden}_\text{short}$ has false alarms (a
short edge no collapse may remove, sitting on a band), which the slow exit
absorbs. The certificate is measured before the sweeps of a phase, so the
smoothing after the last sweep can still move a few edges out of the band.

### Coarsening {#rem_coarsen}

When more ranks shrink $h$ below $4\delta$, bands along the cut planes are
silently never remeshed. A coarser decomposition needs no second communicator,
because redistribution accepts any destination per entity (`MFTL_OwnerMap`):
`MFTLRemeshCoarseOwnerProc` maps a fine cell $i_k$ to the representative of its
group, $\lfloor i_k/g_k\rfloor g_k$, with the remainder absorbed into the last
group (a short group would have a narrower $H$ and could re-create a frozen seam).
The covering argument holds with $H_k = g_k h_k$, and

$$
  g_k = \min\!\big(\texttt{cuts}_k,\ \lceil 4\delta/h_k\rceil\big)
$$

per axis: a domain thin in $x$ loses a factor 2, not 8. `parallel_coarsen = 1`
(default) arms the decision; on a healthy partition the factors are
$\{1,1,1\}$ and the mesh is bit-identical
(`test_the_fallback_stays_off_on_a_healthy_partition` compares `mesh_digest`;
arming costs about 0.5 % of a timestep). When active, $1/G$ of the ranks do all
the sweeping; a longer target length or the dispersed strategy is cheaper when
available.

| Trigger | Fires when | Default | Character |
|---|---|---|---|
| BAND | $h_k \le 4\delta$ on a cut axis | on | required; the alternative is an un-remeshed lattice |
| FROZEN | `frozen_fraction > coarsen_frozen_fraction` | off | trade: 36.6 % frozen → converged instead of 32 hidden edges, for 2× wall clock |
| GATHER | owned triangles $\le$ `serial_gather_max_triangles` | off | trade: crossover ~50–100 triangles per rank |

GATHER short-circuits the others with $g_k = \texttt{cuts}_k$; BAND and FROZEN are
evaluated through the same `MFTLRemeshCoarseningFactors`, FROZEN by inflating the
band: requiring each of $n$ cut axes to keep an unfrozen share
$u = (1-f^*)^{1/n}$ gives $g_k = \lceil 2\delta/((1-u)h_k)\rceil$, i.e. the band
$\delta' = \delta/(2(1-u))$. A cap `parallel_coarsen > 1` shrinks the largest
factor first and reports once if it binds below what BAND required. The decision
uses $\max(\text{measured band}, \ell_{\max})$, since a coarsening pass lengthens
edges.

The gather is the safest coarse map: nothing is replicated or frozen, there is
one distinct placement, and a pass costs exactly two migrations; every rank sends
to one destination. A coarse destination lies outside the 26-neighbourhood, so
redistribution and `MFTLRemeshSyncGhostIds` take their global paths (both detect
it). `MFTLUpdateGhostPositions` has no global fallback, which is why a coarse
partition never outlives the call: both drivers close with `MFTLRedistribute`.
Tables: `docs/analysis.md` §5.6.

---

## The dispersed driver {#rem_dispersed}

For bodies smaller than a rank box — emulsions, sprays, bubbly flows — migrate
each body whole onto one rank and remesh it serially: no shared edges, no
freezing, two migrations per pass.

### One pass {#rem_dispersed_flow}

```
MFTLRemeshEnsembleDispersed(ens, params, stats)                      COLLECTIVE
│  (public entry: geometry + MFTLRemeshMeasureBodies; AUTO hands its own on)
▼
MFTLRemeshEnsembleDispersedMeasured
├─ 0  guards (band > 0; Debug: params agree)
├─ 1  ADMISSION, before anything moves
│     too_large = bodies with R > fraction * box, counted by lowest holder;
│     reachability veto (home(o) a neighbour of every holder)   1 Allreduce
│     too_large > 0 ─▶ whole ensemble to MFTLRemeshEnsembleParallel
├─ 2  DESTINATIONS (assign_destinations), neighbourhood only
│     dest = MFTLGetOwnerProc(centroid)            the production rule
│     lowest holders publish (gid, home, weight)  1 neighbourhood gather
│     total, heaviest                              2 scalar Allreduces
│     balance if slack > 0, P > 1 and no veto:
│       advertise room; 3 rounds of ask / grant / ship  (directed, 1-2 doubles)
│       home publishes (gid, dest, weight) for moved bodies   1 gather
│     peak load; idle and reassigned counts        2 scalar Allreduces
├─ 3  MFTLRemeshMigrateToRanks: one destination per body; Debug: no ghosts left
├─ 4  MFTLRemeshObject per held body; MFTLRemeshRenumberGlobalIds (always)
├─ 5  id counters: per-body MAX over the holders; busiest rank: 1 Allreduce
└─ 6  MFTLRedistribute back to the geometric partition; counts; geometry; profile
```

**Admission is on the radius, not on the outcome.** A one-destination migration
makes *any* body local — a domain-spanning jet would arrive whole on one rank,
report itself perfectly local and serialise the largest object onto one rank's
memory, with a migration that leaves the stencil. Deciding before anything moves
also means the spanning fallback never sees a half-remeshed ensemble. The count
uses `lowest_holder` so that a body held by four ranks is counted once (without it
the driver printed "4 of 1 bodies are wider than a rank box").

**Ids need almost nothing.** After centering exactly one rank holds any entity of
a body, and ids live in a per-body space, so the owner's counter is unique. The
renumbering is unconditional (density is required by the next redistribution),
and the other holders learn the counters by a per-body MAX, to which a rank
holding nothing contributes zero (not a stale value, which would be too large for
a body that shrank).

### Load balance {#rem_balance}

Bodies go where their centroids are, and centroids cluster. Two situations look
alike in a profile and call for opposite responses:

| | $B \ge P$ | $B < P$ |
|---|---|---|
| cause | bodies sit in few boxes | fewer bodies than ranks |
| fixable | yes: reassign | no: a body is the smallest unit of work |
| read | `load_imbalance` | `n_ranks_idle` |

With weights $w_b$ (global triangle counts),

$$
  \text{cap} = \max\!\Big(\texttt{slack}\cdot\frac{\sum_b w_b}{P},\ \max_b w_b\Big),
  \qquad
  \texttt{load\_imbalance} = \frac{P\max_r \text{load}_r}{\sum_b w_b};
$$

a cap below the heaviest body would ask for an impossible assignment.

#### One rank decides each body {#rem_balance_local}

`MFTL_OwnerMap` is undefined if two ranks name different destinations for one
entity. Instead of replicating every body's centroid and weight on every rank
($O(B)$ per rank), the owner of body $o$'s reduced centroid, $\mathrm{home}(o)$,
decides and publishes the verdict one hop along `comm_graph`. The precondition

$$
  \forall o\ \forall \text{holders } H:\quad \mathrm{home}(o) \in \{H\}\cup N(H)
$$

is tested by each holder locally and agreed by one reduction. It implies both
messages: home can be told (the graph is symmetric and `lowest_holder` is a
holder) and home can tell every holder. A failure (`n_bodies_unreachable`) falls
back to plain centroid ownership, never to replication.

#### A negotiation, not a greedy {#rem_balance_negotiate}

Every overloaded rank sees the same idle neighbour. So a rank advertises its room
(cap − load); an overloaded rank asks a few neighbours, emptiest first, for all
their room until its excess plus one body is covered; each neighbour grants its
real room to askers neediest first, so
$\sum_s \text{grant}_{s\to r} \le \text{room}_r$ and no rank is pushed past its cap. Bodies are placed heaviest
first. Two silent failures were found by measurement: a proportional ask over 26
neighbours returns grants smaller than one body, so nothing moves (108 bodies on
27 ranks stuck at imbalance 2.00); and a tie-break identical on every rank
starves one target (1 of 8 ranks idle), so ties break on the neighbour's cell
offset relative to the asker. Rounds are bounded at three (two sufficed
everywhere measured); running out leaves a reported imbalance, never a wrong mesh.

$4P$ droplets packed into one rank box (`remesh_regime_sweep -only l`):

| $P$ | imbalance | idle ranks | moved | wall clock |
|---|---|---|---|---|
| 2 | 2.00 → 1.25 | 1 → 0 | 3 / 8 | 1.45× less |
| 4 | 4.00 → 1.25 | 3 → 0 | 11 / 16 | 2.5× |
| 8 | 8.00 → 1.25 | 7 → 0 | 27 / 32 | 5.0× |
| 27 | 27.00 → 1.25 | 26 → 0 | 103 / 108 | 3.3× |

No digest can distinguish a replicated from a local balancer — both give a
correct, balanced mesh — so the scaling property is tested on the payload:
`test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold` requires a rank
holding no body to stage the same number of doubles when the ensemble doubles
(25 / 49 / 85 at $P$ = 2 / 4 / 8). Weight moves at most one cell per pass. The
balancer decides who remeshes, not who holds the mesh afterwards.

### Coarsening as a yardstick {#rem_yardstick}

$R_{\max} > 0.4\,h_{\min}$ would send a spray to the spanning driver merely
because more ranks shrank $h$. With `parallel_coarsen` armed the admission test
uses the coarsened box $g_k h_k$; the destinations do not change (coarse routing
would restrict them to $P/G$ ranks and undo the balancer). Unbounded, this is
harmful — on a $2\times1\times1$ grid the coarse box is the whole domain and a jet
would be serialised onto one rank — so `dispersed_box_min` relaxes only when
$B \ge P$ (otherwise ranks are idle anyway) and $G < P$ (merging everything is
the gather, a spanning decision bounded by a triangle count). Chooser and driver
use the same function. `test_small_bodies_at_a_fine_grid_stay_dispersed` checks
both directions on a $\{P,1,1\}$ grid.

---

## All the MPI is in one file {#rem_mpi_confined}

| Function | Collective | Volume |
|---|---|---|
| `MFTLRemeshMigrateShifted` | LOR + `MFTLRedistributeWithOwners` | up to 87.5 % of the mesh per placement change |
| `MFTLRemeshMigrateToRanks` | `MFTLRedistributeWithOwners` | whole bodies |
| `MFTLRemeshSumInts` | Allreduce(SUM) | 3 ints per placement; the dispersed gates |
| `MFTLRemeshAgree*` | Allreduce | a few scalars |
| `MFTLRemeshNeighborGather` / `Allgather` / `Alltoall` | `comm_graph` | 3 doubles per body a rank speaks for or moves; 1–2 doubles per neighbour per round |
| `MFTLRemeshMeasureBodies` | 2 × `MFTLEnsembleReduceBodies` + scalar `MPI_MAX` | 9 + 1 doubles per local body |
| `MFTLRemeshSyncGhostIds` | request/reply over `comm_graph` (global fallback) | 4 B per ghost vertex |
| `MFTLRemeshParamsAgree` | Allreduce | Debug only |

Nothing in `src/remeshing/` allocates or sends a quantity proportional to $B$;
proportional to $P$ are only the `[nproc]` inverse neighbour table of
`MFTLRemeshSyncGhostIds` and redistribution's own arrays.

`test_remeshing_mpi_is_confined` is a CMake source check (a scan for `MPI_*(`
outside `remesh_comm.c`; hence the `MFTLRemeshWallTime` wrapper). It exists
because the first version broke the rule twice, each change small and correct: a
cut-plane offset in `MFTLGetOwnerProc`, the body of `compute_owners`, cost
**1.70×** on the per-timestep ownership loop (1.79 → 3.05 ns/entity, 42 → 89
instructions), and a "new global id" field added two tests per ghost to the
per-timestep ghost exchange. Both were replaced by one seam, `MFTL_OwnerMap`
(NULL on the production path) — an ownership *array*, which the dispersed
strategy needs anyway.

---

## `parallel_phase_budget` {#rem_budget}

The default, 32, runs to a fixed point: right for a one-shot remesh far from its
target, wrong in a per-timestep loop. LeVeque deformation, 150 remeshed steps
(`docs/analysis.md` §5.1):

| budget | $P{=}1$ phases | $V/V_0$ | min $q$ | ms/step | $P{=}4$ $V/V_0$ | min $q$ | ms/step |
|---|---|---|---|---|---|---|---|
| 2 | 2.0 | 0.769 | 0.856 | 20.4 | 0.745 | 0.897 | 12.9 |
| **4** | 3.7 | 0.733 | 0.836 | 37.8 | **0.771** | **0.879** | **16.7** |
| 8 | 5.7 | 0.683 | 0.459 | 52.7 | 0.670 | 0.796 | 35.4 |
| 32 | 13.3 | 0.682 | 0.278 | 99.0 | 0.639 | 0.621 | 84.7 |

Serial reference: $V/V_0$ 0.751, min $q$ 0.876, 7.1 ms/step. Volume and quality
get **worse** as the budget grows — re-converging an already good mesh every step
is split/collapse churn that erodes an under-resolved surface — and the cost is
linear in the placements. At budget 4 the distributed result at $P = 4$ matches
or beats the serial one on every measure.

| Use | Setting |
|---|---|
| per-timestep loop | 2–8; 4 was the best compromise. `converged == false` is normal: the mesh is maintained, not converged. |
| one-shot remesh | ≥ 32, and check `converged`. Going 1280 → 3830 triangles took 4 / 6 / 19 / 17 placements at $P$ = 1 / 2 / 4 / 8. |

The default stays at 32 because a too-small budget fails loudly in the one-shot
case, while a too-large one fails quietly in the loop. At budget 4 the placement
count saturates (3.83 → 3.96 from $P$ = 2 to 8) and the overhead on a fixed
problem grows 33 % for 4× the ranks; nothing has been measured above $P = 8$ or
on a cluster.

---

## Reading the diagnostics {#rem_diagnostics}

| Symptom | Read | Likely cause, remedy |
|---|---|---|
| `converged == false`, one-shot pass | `n_phases`, `n_edges_hidden_long` | budget too small (raise it), or work hidden (next row) |
| `n_edges_hidden_long > 0`, persistent | `band_measured` vs $h/4$, `coarsen_trigger` | $h \le 4\delta$: longer $L$, fewer ranks, or let BAND coarsen |
| pass bound by migration | `frozen_fraction`, `t_migrate` vs `t_sweep` | 30–60 % frozen: `coarsen_frozen_fraction` may help; below, the budget |
| idle ranks, DISPERSED | `n_ranks_idle`, `load_imbalance` | balanced but idle ⟹ $B < P$ (unfixable); imbalanced ⟹ check `n_bodies_unreachable`, slack |
| DISPERSED became SPANNING | `n_bodies_not_centred`, `strategy` | a body outgrew the admission box (e.g. after coalescence) |
| quality collapses at one step | `n_feature_edges` | features on a smooth surface: `feature_angle_threshold = 180` |
| `MFTL_ERROR`: non-positive band | — | the band came from `MFTLMeanEdgeLength` on a distributed object on a rank holding nothing; use `MFTLEnsembleMeanEdgeLength` |

The timers are maxima over ranks (the slowest rank sets the wall clock of a
synchronised loop) and are reduced unconditionally — a collective gated on the
caller passing a stats struct would deadlock.

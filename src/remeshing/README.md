`src/remeshing/` — Isotropic surface remeshing {#mftl_mod_remeshing}
==============================================

**Public header:** `include/mftl_remeshing.h` · **Layer:** 3 ·
**Module group:** @ref mftl_remeshing

| File | Job | MPI |
|---|---|---|
| `remesh_ops.c` | split, collapse, flip: an approval predicate and an execution function each; the volume-preserving collapse target | no |
| `remesh_smooth.c` | feature detection, tangential smoothing | no |
| `remesh_sizing.c` | the decision makers of the two sizing modes: edge predicates, the curvature field | no |
| `remesh_fit.c` | paraboloid placement of split and collapse vertices | no |
| `remesh_project.c` | the smoother's back-projection onto the Phong surface of the call's input | no |
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
| $L$ | target edge length (the coarsest under curvature sizing) | `params.target_length` |
| $L(v)$, $\varepsilon$ | per-vertex curvature target; approximation tolerance | `verts.sizing`, `params.curvature_tolerance` |
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

<!-- figure: length-band
The length band. A split halves an edge, and $\tfrac43L$ and $\tfrac23L$ are
equally far from $L$, so a split never moves an edge away from the target.
Edges in $(\tfrac43L, \tfrac85L)$ (hatched) split into halves below
$\tfrac45L$, which is why the band alone does not rule out split/collapse
ping-pong.
-->

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

Jiao et al.'s §5.4 order (smooth, collapse, flip, split, flip, smooth) and
length-sorted sweeps (longest split first, shortest collapse first) were
implemented, measured over the LeVeque period and removed (`docs/analysis.md`
§14.8). Jiao's order lost more volume in every mode, even at equal smoothing work:
uniform sizing at refinement 4 lost 9.5 ± 1.1 % against 3.9 ± 0.1 %, at 1.8–2.0
times the cost. Sorting changed nothing beyond the run-to-run spread.

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

<!-- figure: remesh-operators
The three operators, before (top) and after (bottom). A split appends the
midpoint and one triangle per side. A collapse moves $b$ to $p_m$, rewrites
$a$'s fan, and marks $a$ and the two triangles on the edge dead for compaction.
A flip re-cuts a quad only if the valence energy strictly drops, here from 4
to 0.
-->

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
the point of the rim curve (@ref rem_rim) along an edge of an open surface's rim;
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

<!-- figure: collapse-predicate
Four clauses of the collapse predicate, none of which fails loudly. (a) An
allowed collapse: $a$ and $b$ share exactly the two wing vertices. (b) A third
common neighbour $x$ violates the link condition; the collapse would weld two
sheets. (c) A tetrahedron passes the link condition, but its wings have valence
3 and it would dissolve. (d) $a$ and $b$ both lie on the rim but the edge
between them does not: collapsing that chord pinches the rim inward.
-->

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

<!-- figure: smoothing
Smoothing is a projection, and the feature class chooses the subspace. (a) A
free vertex moves by the tangential part of $\mathbf{d}$ (drawn at 1.8×; the
step is $\sigma\mathbf{d}$). (b) A ridge vertex moves along
$\boldsymbol\tau$ only; the full $\boldsymbol\Delta$ would cut the chord.
(c) A corner does not move. (d) Moving a regular 24-gon ridge to its midpoints
shrinks it to $\cos^{20}(\pi/24) = 0.842$ of its radius in 20 sweeps; the
projection leaves it fixed.
-->

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

### The rim curve {#rem_rim}

The chord midpoint of two rim vertices lies on the rim only if the rim is
straight; on a curved one it cuts inside by $h^2\kappa/8$, and a remesher splits,
collapses and smooths the rim every pass. Measured on a unit disk refined and
coarsened eight times (`target` 0.07 and 0.2, three iterations each): the rim
ended at $r \in [0.977, 0.988]$. Every position the remesher MAKES on a rim -- the
split of a rim edge, the target of a rim-edge collapse, the smoother's move of a
rim vertex -- is therefore taken on the RIM CURVE through the rim's own vertices.

Over the edge $a$-$b$ the curve is the average of two circular arcs: through
$\mathrm{pred}(a), a, b$ and through $a, b, \mathrm{succ}(b)$. With
$\mathbf v = b - a$ ($L = \lVert\mathbf v\rVert$) and $\mathbf u$ the third point
less $a$, the arc bulges away from the third point's side, half its central
angle $\varphi$ given by

$$
  \sin\varphi = \frac{\lVert\mathbf u\times\mathbf v\rVert}{\lVert\mathbf u\rVert\,\lVert\mathbf u - \mathbf v\rVert},
$$

and its point at fraction $f$ is

$$
  a + \tfrac{\mathbf v}{2} + \tfrac{L}{2}\Big(\frac{\sin\psi}{\sin\varphi}\,\hat{\mathbf c}
  + \frac{\cos\psi - \cos\varphi}{\sin\varphi}\,\hat{\mathbf n}\Big),\qquad \psi = \varphi(2f-1).
$$

| property | why it matters |
|---|---|
| exact on a circle at ANY spacing | a remeshed rim is unevenly spaced by construction; the same disk keeps $r = 1$ to $6\cdot10^{-16}$ |
| the chord on a straight rim | straight rims and their corners stay exactly where they are |
| never past the edge's ends | arcs project into the chord; a 10:1:1 spacing splits at the midpoint, where a uniform spline lands outside the edge |
| a planar wall kept to the bit | every term is a combination of offsets from $a$, so a coordinate the three points share stays zero; a wall both $a$ and $b$ lie on is also held explicitly |

The first choice was the centripetal Catmull-Rom spline (no cusps within a
segment, Yuksel et al. 2011). It is $7\cdot10^{-6}$ inside a regular 48-gon but
$4\cdot10^{-4}$ inside once the spacing is 1:2 -- $O(h^2)$, five times better
than the chord and still a shrinking rim ($r \in [0.994, 0.998]$ on the same
disk). Arcs are parameter-free, which is what uneven spacing needs.

The curve BREAKS at a corner -- a vertex flagged `MFTL_FLAG_CORNER`, or one the rim
turns at by more than `feature_corner_angle` (`MFTLDetectFeatures`'s own test,
so an unclassified mesh breaks at the same places); that side's arc is dropped,
and with neither the point is the chord's. The corner rule also bounds every arc
used to a quarter circle, so its sagitta is at most $0.21 L$.

**A rim split checks for a fold, which a chord split never needed.** The chord
midpoint lies on $a$-$b$, so $(a, m, c)$ and $(m, b, c)$ keep the orientation of
$(a, b, c)$; a point off the chord can turn one over when $c$ sits closer to the
chord than the arc's bulge -- a sliver on a curved rim. The split then falls back
to the chord midpoint, and is refused only if that folds too (which takes a
projector).

**Smoothing a rim vertex** keeps the ridge rule's step -- half the difference of
its two rim edges, times $\lambda$, capped at $\phi$ times the shorter -- but
measures it ALONG the curve and evaluates the curve there, rather than stepping
along the tangent, which leaves a curved rim by $O(\text{step}^2\kappa)$.

**The rim constraint.** `MFTL_RemeshParameters::rim_project` is an optional
`MFTL_RimProjector`: a pure function, the same on every rank, that moves a
position onto what the rim's own vertices cannot know (a nozzle circle is not
among those -- the arcs reproduce it -- but a curved support may be). It is
applied after each of the three placements above, BEFORE the approval clauses
see the position; a refusal cancels the operation. A topology change whose
`MFTL_TopologyOptions::remesh` carries it applies it to every rim vertex it
reconstructs and inside its volume correction. Its presence is agreed across
ranks with the rest of the block.

Rim neighbours are found by walking the fans of $a$ and $b$, which are complete
for every entity an operation may touch (an unfrozen entity's one-ring is
complete). `MFTLRemeshRimFallbacks` counts placements whose walk failed and
took the chord; it reads 0 in every test. A closed surface never reaches any of
this, so its results are bit-identical to the chord remesher's.

### Feature detection on a smooth surface {#rem_features_off}

On a sphere every detected feature is a false positive, and it removes degrees of
freedom a vertex needed: in the LeVeque test four spurious edges dropped the
worst triangle from 0.797 to 0.045, where it froze, because a constrained vertex
is exempt from smoothing, collapse and flip. Final minimum quality 0.76 with
detection off against 0.0002 at 45°, monotone over five thresholds
(`docs/analysis.md` §4). **Set `feature_angle_threshold` to 180 for smooth closed
interfaces**; boundaries remain features.

---

## Sizing: uniform and curvature-adaptive {#rem_sizing}

`params.sizing` chooses how the remesher judges an edge, once per run. The
driver, the sweep order, the freeze discipline and both distributed strategies
are the same for both; what differs is four decision makers:

| Decision | `UNIFORM` | `CURVATURE` (Dunyach et al. 2013) |
|---|---|---|
| split wanted | $\ell > \ell_{\max}$ | $\ell > \tfrac43\min(L(a),L(b))$ |
| collapse wanted | $\ell < \ell_{\min}$ | $\ell < \tfrac45\min(L(a),L(b))$ |
| collapse stretch | new edge $> \ell_{\max}$ | new edge $> \tfrac43\min(L_m, L(w))$ |
| free-vertex target | area-weighted centroid | centroid weighted by area $/\bar L^2$ |

Flips (valence) and new-vertex positions (chord midpoint) are the same in both.

Every judgement goes through `MFTLRemeshEdgeWantsSplit` and
`MFTLRemeshEdgeWantsCollapse` — the operators' predicates, the collapse's
stretch clause and the spanning driver's convergence certificate alike — so they
cannot disagree about what "too long" means. The `UNIFORM` branches are the
expressions the remesher had before sizing existed, and every uniform digest is
unchanged. The topological operators themselves are untouched: a split
interpolates the curvature target to its new vertex and a collapse to its
survivor, one line each.

`max_edge_length` is always the longest edge the mode can want —
$\tfrac43 L_{\max}$ under curvature sizing — because the frozen band
($3\,\ell_{\max}$) and the parameter check read it.
`MFTLRemeshUseCurvatureSizing` sets the mode and derives that envelope; the
settings layer does the same from `remesh.sizing`.

### Curvature-adaptive {#rem_sizing_curvature}

The target of a vertex comes from its maximum absolute curvature $\kappa$ and the
approximation tolerance $\varepsilon$ — the largest gap between an edge and the
surface it cuts — through the circumcircle of an equilateral triangle:

$$
  L(v) = \sqrt{6\varepsilon/\kappa - 3\varepsilon^2}\ \in [L_{\min}, L_{\max}],
  \qquad \kappa = H + \sqrt{\max(H^2 - K, 0)} .
$$

$H = \tfrac12\lVert\Delta\mathbf x\rVert$ and $K = (2\pi - \sum\theta)/A_\text{mixed}$
are the paper's operators (Meyer et al. 2003: cotangent Laplacian, angle defect,
mixed Voronoi area with the obtuse cases), computed by
`MFTLComputeVertexCurvatures` in `src/geometry/`. `MFTLRemeshCurvatureSizing`
evaluates the formula without dividing before the clamps: $\kappa \le 0$ is the
coarsest target, $\kappa \ge 2/\varepsilon$ the finest, and no input gives NaN.
An edge's target is the smaller of its ends'. A feature, corner or rim vertex has
no curvature of its own — a crease is unbounded, a half fan is not a curvature —
and takes the mean of its non-feature neighbours (paper, Sect. 3, item 1).

**One smoothing pass over the one-ring, before the length is taken.** Not in the
paper; in its reference implementation (pmp-library smooths the curvature tensor
once), and it is load-bearing. The edge rule takes a MINIMUM of two noisy
estimates, which is biased low, so without the pass a remeshed mesh reads about
6 % more curvature than its input and every pass that recomputes the field
refines further: five repeated serial remeshes of an ellipsoid went
4578 → 5092 → 5040 → 4952 → 4958 triangles and were still operating
(`docs/analysis.md` §14.1). With it they reach a fixed point — no operation
from the sixth call on — within 4 % of the first.

**The field is computed once per kernel call and then carried.** A split
interpolates it to the new vertex, a collapse to the survivor, compaction and
reordering permute it (`verts.sizing`, an optional array of the vertex pool's one
block: zero bytes unless enabled). Recomputing after every operation (what the
old mftl did) would feed the chord midpoints the operators create straight back
into the curvature.

**Smoothing** moves a free vertex to
$\sum_t |t|\rho_t\mathbf b_t / \sum_t |t|\rho_t$ with the density
$\rho_t = \bar L_t^{-2}$ ($\bar L_t$ the mean of the three corner targets). This is a deliberate deviation from the printed Eq. 6,
which weights by $|t|\,L(\mathbf b_t)$: the Optimal Delaunay Triangulation the
paper cites weights by a density that DECREASES with the target, and so does
pmp-library. The printed weight pulls vertices towards the coarse side, against
the sizing field; `test_the_smoother_moves_towards_the_fine_side` asserts the
sign. A ridge or rim vertex follows the 1D form of the same rule along its curve
(edge midpoints weighted by length over squared target), whose fixed point is
$\ell_1/L_1 = \ell_2/L_2$ and which is the uniform rule when the targets agree.

**Distributed.** The field is a one-ring quantity, exact on owned vertices (the
halo is vertex-complete) and partial on ghosts. No admitted operation reads a
ghost's value; the convergence certificate does, and `hidden_long` is exact only
if every copy of a vertex holds the same target. So
`MFTLRemeshEnsemblePrepareSizing` runs the curvature pass, gives the ghosts their
owners' values (`MFTLRemeshSyncGhostSizing`, `remesh_comm.c`, the route of the
ghost-id exchange including its global fallback), runs the neighbour pass, and
exchanges again. A redistribution does not carry the field, so the spanning
driver recomputes it after every migration; each recomputation on a part-remeshed
mesh is another chance to split at a dip the band's hysteresis then keeps, and
the distributed fixed point carries 1.3 / 3.9 / 6.0 / 4.0 % more triangles than
the one-rank fixed point at np = 2 / 4 / 8 / 27 (`docs/analysis.md` §14.2).
Carrying the field through the remesher's migrations would remove that and
needs a payload in the redistribution's wire format; not done.

**Run ONE iteration per placement under curvature sizing.** The spanning driver
converges every placement fully, so a budget of 4 is four times the serial
kernel work per remesh, each time with a recomputed field. A uniform mesh is
near converged every step and does not care; a curvature-sized mesh tracks a
curvature that moves, and every extra pass splits and collapses for real: the
LeVeque period at np = 4 kept 0.717 of its volume with three iterations per
placement and 0.881 with one (serial: 0.903), at 45 against 19 ms/step
(`docs/analysis.md` §14.5).

### Removed: anisotropic adaptation and quadric-error placement {#rem_sizing_removed}

Jiao et al.'s anisotropic mode (2010: a quadric metric for the flip and the
smoother, the paper's split and collapse criteria, quadric-error placement of new
vertices) was implemented as a third mode, measured over the LeVeque period, and
removed with its tests (`docs/analysis.md` §14.4, §14.7–§14.9;
`studies/leveque_remeshing/REPORT.md` §7; the code is archived as
`studies/leveque_remeshing/archive/sizing_with_anisotropic_and_qem.patch`
against commit `bb3c785`). It held the enclosed volume best of the three modes
(+0.06 % at refinement 5, uniform −1.17 %), but the follow-up located that in
how rarely it COLLAPSES under a stretching flow, not in its metric: the
volume a remesher loses on a convex surface is lost by collapses and valence
flips, and a mode that tolerates stretched triangles makes a seventh of the
collapses. It carried three times the triangles, cost 3–4 times as much, and
returned the noisiest curvature of all modes. Quadric-error placement, which
it used and which was also an option for the other two, lowered the volume and
position errors of uniform and curvature sizing and raised their curvature
noise 1.4–3 times. A back-projection onto a smooth reconstruction of the
call's input, measured in the same study, beat both on every measure.

---

## Vertex placement and back-projection {#rem_placement}

A remesher that splits at chord midpoints, collapses to them and smooths in the
tangent plane loses volume on a convex front, and only two of its sweeps do:
a collapse replaces two surface points by one on the chord, and a valence flip
away from the convex diagonal cuts off a tetrahedron. Each call's loss becomes
the next call's input. Over the LeVeque period at refinement 4 that is −2.7 %
(collapses) and −1.2 % (flips) under uniform sizing; a midpoint split is exactly
neutral and tangential smoothing gains +0.2 % (`docs/analysis.md` §14.9). Three
independent options attack it, all off by default and bit-identical to the old
remesher when off: where splits and collapses put their vertex (chosen
separately), a volume-preserving collapse, and a back-projection of the
smoother.

### Placement: a paraboloid through the edge's two-ring {#rem_placement_fit}

`params.split_placement` = `MFTL_REMESH_PLACE_PARABOLOID` puts a split's new
vertex, and `params.collapse_placement` = `MFTL_REMESH_PLACE_PARABOLOID` a
collapse's merged vertex between two free vertices, on a weighted
least-squares height paraboloid above the edge midpoint
(`MFTLRemeshFitEdgePoint`). The two are separate switches; both default to the
midpoint:

- **Frame.** Origin at the minimum-image midpoint $m$, $z$ along the normalised
  sum of the endpoints' vertex normals, $x$ along the edge projected into the
  tangent plane. Normals use Max's weights
  $\sum (\mathbf e_1\times\mathbf e_2)/(|\mathbf e_1|^2|\mathbf e_2|^2)$,
  exact for vertices on a sphere; they measured better than angle weights here.
- **Samples.** The vertices of the edge's two-ring, 20–30 of them, each weighted
  by Wendland's C2 function $(1-q)^4(1+4q)$ of
  $q = |\mathbf x - m| / (\rho h)$, with $h$ the mean length of the edges at
  both endpoints and $\rho$ = `fit_radius`. A sample whose normal is more than 60° from $z$ is
  dropped: it is on the far side of a fold or a sheet.
- **Scale.** Coordinates in units of $h$, so every column is O(1) and the rank
  tolerance is dimensionless. A mesh scaled by $2^{-20}$ remeshes to the scaled
  result bit for bit.
- **Solve.** `MFTLFitHeightParaboloid`, QR with column pivoting grouped as
  {height, slopes} then {curvatures} (`src/geometry/README.md`, "Least squares").
  An undetermined curvature component is set to zero; an undetermined height, a
  correction beyond a quarter of the edge, fewer than six samples or, for a
  split, a fold of a new triangle fall back to the midpoint.
- **With normals** (`MFTL_REMESH_PLACE_PARABOLOID_NORMALS`), every owned sample
  with a complete fan off any feature adds its normal's slopes
  $\partial_x z = -n_x/n_z$, $\partial_y z = -n_y/n_z$ as two more equations.

**Partitioned meshes.** An operation is only allowed on an unfrozen edge, whose
one-ring is owned and unreplicated, so both endpoint fans are complete. The
two-ring may reach ghosts (about 30 % of the fits at np = 4): their positions
are exact (a ghost is frozen on its owner), their normals partial, so a ghost
contributes its position only. Giving the ghosts their owners' normals (one
exchange per placement) measured the same within the spread, so there is none.

**What changed against mftl's fit.** mftl fitted the same quadric to eight
vertices (the edge's two triangles and the four across their outer edges),
unweighted, in physical coordinates, through LAPACK `dgelsy` with
`rcond = 1e-10`, the frame normal summed over the rank's owned vertices only.
Its coordinate transformation was right: it formed the cofactor matrix over the
determinant, i.e. the inverse transposed, which for an orthonormal frame is the
frame itself, the rotation it needed. Its failure was the scale: below an edge
length of about $10^{-5}$ in the units of the coordinates the curvature
columns fall under `rcond` and the height comes out wrong by several sagittas
(4.5 on a sphere of radius $10^{-4}$). Scaled, its eight samples are as
accurate on an exact sphere as a two-ring with radius 2; the wider, weighted
stencil trades a little of that accuracy for noise rejection on a real front.

**Cost.** A collapse's position is fitted only for candidates whose midpoint
already passes the stretch and fold clauses, then judged again at the fitted
point, and its execution reuses the approval's fit (cached on the mesh's
generation stamp). Without that a run spent five times as many fits on
candidates the fold or stretch clause then refused.

### The volume-preserving collapse {#rem_placement_volume}

`params.collapse_preserves_volume` moves the point `collapse_placement` chose,
midpoint or fitted, so that the collapse leaves the enclosed volume unchanged
(the volume constraint of Lindstrom & Turk 1998). Measured from the merged
position $\mathbf p$, a surviving triangle of the patch (the triangles at the two
endpoints) holds $\mathbf p$ as a corner and contributes nothing after the
collapse, so the change is minus the patch's old contribution,

$$
  \Delta V(\mathbf p) = -\tfrac16 \sum_{\text{patch}}
  \det(\mathbf x_0-\mathbf p,\ \mathbf x_1-\mathbf p,\ \mathbf x_2-\mathbf p),
$$

independent of the origin because the patch's rim is the same closed loop
before and after. Every surviving triangle holds $\mathbf p$ exactly once, so
$\Delta V$ is LINEAR in $\mathbf p$, with gradient
$\mathbf g = \tfrac16\sum_{\text{surviving}} (\mathbf y-\mathbf p)\times(\mathbf z-\mathbf p)$
over each triangle's other two corners in cyclic order (a third of the new
fan's vector area). One step

$$
  \mathbf p \leftarrow \mathbf p - \Delta V\,\mathbf g/|\mathbf g|^2
$$

reaches the plane $\Delta V = 0$ at its point nearest the chosen one; it is
exact up to rounding, so there is no iteration
(`test_a_volume_preserving_collapse_keeps_the_volume`: below $10^{-15}$ of V
per collapse, against $2\cdot10^{-4}$ without). The move is along the fan's
normal by $3|\Delta V|/A_\text{fan}$, a fraction of the sagitta.

- **Scope.** An interior edge between two free vertices, whose link is a closed
  loop. Rim collapses keep the rim curve, a collapse into a pinned vertex keeps
  its position, and a truncated fan or a zero gradient leaves the chosen point.
- **Judged.** The stretch and fold clauses judge the moved point, the midpoint
  candidate and the fitted one alike; the move costs one pass over the two fans.
- **Partitioned meshes.** Only unfrozen edges collapse, so both fans are owned
  and complete and no ghost is read; `test_leveque_parallel_conserves_vpcollapse`
  runs it at np = 2.

**Why collapses.** Over the LeVeque period at refinement 4, 94 % of the
collapse loss comes from patches bent by more than 5°, which are 47 % of the
collapses: the stretched filament's under-resolved rims. The flow stretches one
direction while it compresses the other, so splits and collapses run on the
same material at the same time (8 777 collapses against 15 417 splits even in
the growing half of the period), and every midpoint collapse of a bent patch
cuts off a cap. A midpoint split leaves the triangulated surface unchanged.

**Measured** (uniform sizing, four orientations, `docs/analysis.md` §15.4):

| refinement 5, at T | V/V(0) − 1 | d_rms [% R0] | E_H | N_H |
|---|---|---|---|---|
| projection | −0.182 % | 0.247 | 0.17 | 0.082 |
| projection + volume-preserving collapse | −0.175 % | 0.236 | 0.15 | 0.066 |
| midpoint, no projection | −1.18 % | 0.63 | 1.29 | 1.52 |
| volume-preserving collapse, no projection | −0.41 % | 0.26 | 1.78 | 2.18 |

With the projection it improves every measure, more at refinement 4 (E_H
0.55 → 0.38, N_H 0.33 → 0.22), for +0.5–1.0 % of a step. Without the projection the
position improves and the curvature noise grows: a patch's deficit is put back
at one vertex, a bump the next smoothing does not remove.

**Measured and not taken, for flips.** A flip's volume change is
$\det(\mathbf d,\mathbf b,\mathbf c)/6$ from $\mathbf a$. Refusing a flip whose
change exceeds a threshold halves the flips' loss, but the valences it leaves
make the collapses lose more (net −3.74 against −3.90 % at refinement 4).
Restoring it exactly by a common shift of the quad's four vertices along its
normal helps only without the projection, which takes the shift back.

### Back-projection onto the Phong surface of the call's input {#rem_placement_project}

`params.projection` = `MFTL_REMESH_PROJECT_PHONG` moves every free vertex the
call's last smoothing sweep moved onto the Phong surface (Boubekeur & Alexa 2008,
$\alpha = \tfrac12$) of the mesh as THIS kernel call received it:

$$
  \mathbf p \leftarrow \mathbf p_\flat - \tfrac12 \sum_i u_i\,
  \big((\mathbf p_\flat - \mathbf x_i)\cdot\mathbf n_i\big)\,\mathbf n_i ,
$$

$\mathbf p_\flat$ the closest point on the reference's flat triangles, $u_i$ its
barycentrics, $\mathbf n_i$ the reference's angle-weighted normals.

- **Why the call's input, and why curved.** Botsch & Kobbelt project onto the
  original surface kept as a mesh, fixed and usually finer. In front tracking
  that mesh is the previous step's at the same resolution, and projecting onto
  its flat triangles puts every smoothed vertex on a chord: −12.5 % of the
  volume over the LeVeque period at refinement 4. Its Phong surface passes
  through the same vertices and is curved like the front, so the smoother
  re-lifts what this call's collapses and flips cut off. A reference rebuilt
  from the current mesh does not: it reconstructs the surface from the samples
  being smoothed and loses more volume than no projection (`docs/analysis.md`
  §14.9).
- **What it means physically.** The call receives the surface the flow advected;
  remeshing is meant to change its discretisation, not its position. Every
  normal displacement the operators and the tangential smoother make is a
  discretisation error, and the projection takes it back to the advected
  surface, known through its vertices and their normals. It does not touch
  what the advection put there, noise included, and it undoes any normal
  motion made on purpose INSIDE the call: a denoising or volume-correcting step
  (TSUR3D, say) belongs outside the remesher, or must produce the reference.
  Mesh quality is unaffected: the correction is a sagitta, the smoother's
  moves are a fraction of an edge (mean quality at T 0.9767 against 0.9775
  without, uniform sizing at refinement 4; `docs/analysis.md` §14.10).
- **Once per call.** The projection runs after the last smoothing sweep of the
  call's last iteration. After every sweep it measured the same within the
  run-to-run spread (under-resolved uniform sizing at refinement 4: −1.49
  against −1.60 % volume) at about three times the cost; the intermediate
  sweeps and operators work on a slightly perturbed surface and the final
  projection puts every moved vertex back.
- **The reference** (`MFTL_RemeshReference`) is copied once per kernel call,
  after the first feature detection, and kept for all its iterations:
  positions, normals, corners, edge neighbours, and the vertex global ids,
  sorted. A corner that is a feature, corner, boundary or ghost vertex gets no
  normal, so the Phong surface degrades to the flat triangle towards it.
- **Finding the closest triangle without a search structure.** A kernel call does
  not renumber global ids, so a vertex the call received is found in the copy
  by binary search on its id, a vertex a split created through a neighbour. A
  walk over the copy's edge neighbours then reaches the closest triangle, and
  each later sweep starts from the triangle the previous one found. No grid, no
  communication: a moved vertex is unfrozen and its fan owned. Since the
  projection runs once, each lookup starts cold (about 120 ns a vertex). A projection
  that would bridge more than half the vertex's shortest edge is not made.
- **Normals.** Angle weights. Max's weights are exact on a sphere and make the
  lift fourth-order on an irregular sphere mesh, where angle weights leave it
  second-order with a three times smaller constant than no projection
  (`test_projected_smoothing_stays_on_the_sphere`), but over the LeVeque period
  they were worse at refinement 4 and equal at 5.

### Measured {#rem_placement_measured}

LeVeque period, four orientations, one rank (`docs/analysis.md` §14.10):

| curvature sizing, V/V(0) − 1 at T, N_H | refinement 4 | refinement 5 |
|---|---|---|
| midpoint, no projection | −3.15 %, 2.57 | −0.97 %, 1.58 |
| paraboloid placement (positions) | −0.56 %, 0.58 | −0.32 %, 0.38 |
| paraboloid placement (+ normals) | +0.01 %, 0.90 | −0.22 %, 0.36 |
| Phong projection | −0.089 %, 0.075 | −0.030 %, 0.023 |
| projection + paraboloid placement | −0.089 %, 0.079 | −0.030 %, 0.022 |

**The projection is the one to switch on**; with it the placement adds nothing
measurable. Projecting once per call costs 3.3–3.7 % of the remesh time in
projections and 2.4–2.8 % in reference copies, +5–10 % per triangle-step, and a
transient 53 B per triangle during a kernel call against the pool's ~172. The
placement on its own is the cheaper partial remedy: 2.3 µs (positions) or
3.1 µs (with normals) per fit, 2.4–5.4 % of the remesh time.

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

<!-- figure: remesh-regimes
The two distributed strategies. (a) A body narrower than a rank box moves whole
to the rank owning its centroid and is remeshed serially: nothing shared,
nothing frozen. (b) A body wider than a rank box stays; the cut planes move by
half a box instead. (c) Clustered bodies are spread one hop along `comm_graph`
by a negotiation that never pushes a rank past its cap.
-->

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

<!-- figure: spanning-placements
The frozen band follows the planes. An icosphere on a periodic 2×2×2 grid, with
ownership, halo and frozen set computed by the rules of this chapter: (a) with
the canonical planes, (b) with every plane shifted by $h/2$. (c) On one axis, a
point within $\delta$ of a plane of one placement is at least $h/2-\delta$
from every plane of the other, so it is interior there when $h > 4\delta$.
-->

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
(`test_a_seam_straddling_droplet_is_not_measured_as_domain_sized`). The
remesher is the one per-body consumer that does NOT depend on the periodic
extent rule (@ref geom_periodic_extent): its operators see only minimum-image
edges, and this centre and radius choose a strategy, which a body wider than half
the period still gets right -- the spanning tests remesh a sphere 0.7 of the
period wide.

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
| `MFTLRemeshAgreeParams` | Allreduce(MIN) | 49 doubles, once per call, at the first collective |

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

## Attached-field transfer {#rem_fields}

The optional arrays described in @ref object_fields follow every permutation
and migration, including the temporary placements of both parallel strategies.
The local operators share the same transfer rules in serial and distributed
runs. Replicated entities remain frozen; their values travel with the existing
migrations. Recomputed fields require no numerical transfer.

For a conservative triangle component, let $m_T=A_T\rho_T$ with fresh
minimum-image triangle areas. Transfer uses constant reconstruction and only the
triangles changed by an operation; no neighbouring field rows supply material.

- **Split:** distribute each parent's amount among its own children in proportion
  to their new areas, conserving the two parents separately.
- **Collapse/flip:** project each new triangle centroid onto the nearest old
  triangle in the changed patch and take that donor's concentration. Collapse
  includes the full union of endpoint fans, including surviving moved triangles.
  Resolve distance ties by global id with a scale-relative tolerance. For the
  predicted amounts $\widehat m_j=A_j^{new}\widehat\rho_j$, apply the local correction
  $m_j^{new}=M_{patch}^{old}\widehat m_j/\sum_k\widehat m_k$. A zero predictor with
  positive old material uses the original positive area-based transfer; zero
  old material remains zero. This is approximate interpolation, not an overlap map.
- **Smooth:** retain the Jacobi geometry proposals and old field values. Each
  moving internal edge defines a signed swept quadrilateral in the frame given
  by its adjacent old unit normals. Its signed area times the upwind concentration
  gives a mass flux, applied with opposite signs to the two triangles. Scale a
  donor's outgoing fluxes together if they exceed its available mass. Fixed edges
  have zero flux, and open rims supply no external material. Divide final amounts
  by actual new triangle areas. A degenerate edge frame selects the old per-triangle
  area-ratio rule for the entire sweep, before any field writes.

These rules conserve patch/sweep amount to round-off and preserve non-negativity.
They do not guarantee an upper bound on concentration when areas shrink or a patch
correction is needed. Invalid old densities or nonpositive new areas reject an
edit before topology, positions or fields are changed. All triangle components
are prepared before mutation and committed after reacquiring any reallocated
pool pointers. Smoothing rejects the entire candidate sweep if its transfer is
invalid or an affected triangle reverses orientation. Unchanged triangle rows
are never written; replicated entities remain frozen.

Ordinary triangle samples inherit on
split, use an old-area-weighted patch average on collapse/flip, and stay unchanged
on smoothing. Vertex samples use the clamped endpoint fraction on split/collapse;
on smoothing they use barycentric interpolation at the closest point in the old
incident fan, reading the old values for the entire Jacobi sweep.

**Conservation does not establish spatial accuracy.** The constant reconstruction
remains diffusive. Planar tests show first-order behaviour; curved-surface swept
areas are local approximations, and no second-order guarantee is made. On a fixed
planar domain, small smoothing moves preserve constant density through the signed
area balance. On curved surfaces constant density cannot stay constant when total
area changes while amount is fixed. The demonstrated accuracy gain does not turn
remeshing into a physical surfactant transport solver.
`examples/attached_fields.c` measures mass, profile error and concentration
centroid movement separately on a stationary sphere; `docs/analysis.md` §12
records its refinement study and the before/after transfer comparison. Linear
gradients are deferred: the measured constant-reconstruction upgrade already
substantially reduces the profile error without a wider read stencil.

Physical advection remains the solver's responsibility. Breakup/coalescence
reconstruction currently rejects ensembles with persistent fields before any
mutation; recomputed-only ensembles are supported and receive fresh storage.

## Reading the diagnostics {#rem_diagnostics}

| Symptom | Read | Likely cause, remedy |
|---|---|---|
| `converged == false`, one-shot pass | `n_phases`, `n_edges_hidden_long` | budget too small (raise it), or work hidden (next row) |
| `n_edges_hidden_long > 0`, persistent | `band_measured` vs $h/4$, `coarsen_trigger` | $h \le 4\delta$: longer $L$, fewer ranks, or let BAND coarsen |
| pass bound by migration | `frozen_fraction`, `t_migrate` vs `t_sweep` | 30–60 % frozen: `coarsen_frozen_fraction` may help; below, the budget |
| idle ranks, DISPERSED | `n_ranks_idle`, `load_imbalance` | balanced but idle ⟹ $B < P$ (unfixable); imbalanced ⟹ check `n_bodies_unreachable`, slack |
| DISPERSED became SPANNING | `n_bodies_not_centred`, `strategy` | a body outgrew the admission box (e.g. after coalescence) |
| quality collapses at one step | `n_feature_edges` | features on a smooth surface: `feature_angle_threshold = 180` |
| `MFTL_ERROR`, "band is invalid on at least one rank" | — | the band came from `MFTLMeanEdgeLength` on a distributed object on a rank holding nothing; use `MFTLEnsembleMeanEdgeLength` |
| `MFTL_ERROR`, "different parameter blocks" | the field named on stderr | a parameter derived from rank-local data; every field must be bitwise identical on every rank |

The timers are maxima over ranks (the slowest rank sets the wall clock of a
synchronised loop) and are reduced unconditionally — a collective gated on the
caller passing a stats struct would deadlock.

`src/coupling/` — Coupling to a flow solver {#mftl_mod_coupling}
===========================================

**Public header:** `include/mftl_coupling.h` · **Layer:** 3 ·
**Module group:** @ref mftl_coupling

**Open work:** `plans/00-roadmap.md`, item B5.

A flow solver owns velocity components and force targets on ITS samples — cell centres,
staggered faces, the centroids of an unstructured mesh — distributed its own way. A
front tracker needs those velocities at the surface's markers, and returns surface
forces to the samples. This module is that exchange, for any solver: the solver
registers what it owns, MFTL builds one interpolation operator per binding and
applies its exact transpose to spread, and only the band of samples the markers'
kernels can reach ever travels.

| File | Job | MPI? |
|---|---|---|
| `solver_grid.c` | registration: locations, lattice blocks, point sets, coverage, ownership regions | no |
| `coupling_kernels.c` | the kernel table: support, lattice moments, sign, smoothness | no |
| `coupling_rows.c` | one general row: base weights, moment correction, quality checks | no |
| `coupling_markers.c` | markers from the surface, dual-area weights, quadrature points and their vertex schedule, the generation snapshot | no |
| `coupling_directory.c` | lattice index arithmetic and images, box algebra, indexed ownership/resolution certificates | no |
| `coupling_index.c` | compact bounds over provider tiles and irregular cached samples | no |
| `coupling_stencil.c` | direct Cartesian brick lookup, factored Peskin rows and their transpose | no |
| `coupling_memory.c` | module allocation accounting, peak and optional capacity limit | no |
| `coupler.c` | the state machine: directory, request, window, binding, transfers | no |
| `coupling_comm.c` | **every MPI call the module makes** | **yes** |

`coupling_internal.h` declares what the files share; `test_coupling_mpi_is_confined`
fails the build if an MPI call appears anywhere else. The coupler communicates on its
own duplicate of the ensemble's communicator and never probes a message it did not
cause. Fatal allocation/budget failure aborts the MPI job, including if it occurs
during registration before a private communicator exists.

---

## The operator {#cpl_operator}

For marker $i$ at $\mathbf x_i$ and the samples $j$ of one location, at $\mathbf X_j$
with measures $q_j$,

$$
\psi_{ij} = \frac{q_j}{\det\tilde H_i}\prod_k \phi\!\left(\frac{(X_j-x_i)_k}{\tilde H_{i,k}}\right),
\qquad
M_i = \sum_j \psi_{ij}\,p(d_{ij})\,p(d_{ij})^T,\quad M_i c_i = e_0,\quad
W_{ij} = \psi_{ij}\,p(d_{ij})^T c_i,
$$

with the linear basis $p(d) = (1, d_x, d_y, d_z)^T$, the offset
$d_{ij} = \tilde H_i^{-1}(X_j - x_i)$ in units of the marker's kernel metric $\tilde H_i$. Interpolation is $U = W u$, spreading is $b = W^T F$ with
the SAME stored rows, so

$$
u^T b \;=\; U^T F
$$

holds to round-off (`test_spread_is_the_brute_force_transpose` checks it with random
fields and loads, and checks every owned sample's $b_j$ against a transpose recomputed
by brute force from every marker of every rank). With LINEAR reproduction
$\sum_j W_{ij} = 1$ and $\sum_j W_{ij}(X_j - x_i) = 0$: the spread conserves force and
torque (`test_mac_vectors_force_and_torque`, per component on a MAC grid, where each
component's first moment is its own location's).

**A row whose kernel already has the moments is left alone, bit for bit.** Peskin's
function over a complete uniform lattice reproduces constants and linears
($M_i e_0 = e_0$) although $M_i$ is not the identity — its second moments are not
those of the unit — so no solve runs. Everywhere else the moment matrix is solved
through the eigen-decomposition of its equilibrated form (cyclic Jacobi, 4×4):
rank-revealing, with the exact condition number, and cheaper than the accumulation
that built it.

**Four checks, each on its own.** A row is built only over a support certified
COMPLETE (next section); it must have full numerical rank and a condition number below
`max_condition`; it must reach the requested moments within
`max(moment_tolerance, 10^3 ε κ)`; and it must not amplify,
$\sum_j|W_{ij}| \le$ `max_amplification`. A successful pivot implies none of them: a
cloud squashed into a plane is rank-deficient, a marker outside its samples' hull
extrapolates with weights of both signs (about 22 in absolute sum in a corner,
`test_extrapolation_is_refused_by_amplification`). A failure is BAD_STENCIL on every
rank; growing the support (`MFTL_STENCIL_GROW`, within `max_growth`, sized into the
window) and dropping to constant reproduction (`MFTL_STENCIL_LOWER_ORDER`) are explicit
choices, counted per row. A row is never silently renormalised.

### Kernels {#cpl_kernels}

| kernel | half-width | lattice moments | sign | smoothness |
|---|---|---|---|---|
| `PESKIN4` (default) | 2 | constants, linears | ≥ 0 | C1 |
| `ROMA3` | 1.5 | constants, linears | ≥ 0 | C0 |
| `COSINE4` | 2 | constants | ≥ 0 | C1 |
| `HAT2` | 1 | constants, linears | ≥ 0 | C0 |
| `BSPLINE3` | 2 | constants, linears | ≥ 0 | C2 |
| `WENDLAND_C2` (radial) | 2 | none | ≥ 0 | C2 |

`test_coupling_kernels` checks every declaration by independent means — published
values, the unit integral by quadrature, the lattice moments over 1024 shifts in BOTH
directions (a declared order holds, the next does not), Peskin's even/odd and
sum-of-squares identities, continuity at the breakpoints. An application kernel is a
callback evaluated once per marker and axis over all candidates, never once per
sample.

### The metric {#cpl_metric}

A location whose samples all share one resolution (a uniform lattice, a point set with
`h_uniform`) uses it: the kernel metric is `support_scale` times $h$. Otherwise,
each sample influences the metric within `pilot_factor` times **its own** directional
resolution. With $a =$ `pilot_factor`, define

$$
r_j^2 = \sum_k \left(\frac{X_{j,k}-x_k}{a h_{j,k}}\right)^2,\qquad
\omega_j = \frac{q_j}{h_{j,x}h_{j,y}h_{j,z}}(1-r_j)^4(4r_j+1)
$$

for $r_j<1$, and zero otherwise. The marker resolution is
$h(x)=\sum_j\omega_j h_j/\sum_j\omega_j$, with `support_scale` applied afterwards.
This compact Wendland C2 average varies smoothly wherever the denominator is
positive; no contributor means BAD_STENCIL. There is no global-maximum-radius
pilot. Refining or coarsening a distant region changes neither the metric nor its
required samples unless that region's own influence reaches the marker.

The directory bounds every contributor over the whole marker bin plus motion skin
before a window is built. The metric is evaluated from the certified samples in
canonical order, independently of their rank ownership. This is a numerical change
from the earlier global-bound average; variable-resolution results need not match
the old ones. Independent whole-mesh references test the formula and a distant
coarsening test checks locality and unchanged band size.

**An anisotropic point location** gives every sample a full CELL matrix $H$ (its
columns the cell's edge vectors) and rows use $d = \tilde H^{-1}(X - x)$, normalised by
$|\det \tilde H|$: a tensor kernel then follows a rotated or sheared cell's own axes.
A cloud that is a complete lattice in those axes — a rotated anisotropic lattice of
points — then has the kernel's lattice moments: Peskin's rows need no correction and
stay nonnegative, where the axis-aligned treatment of the same cloud corrects every
row (`test_a_rotated_lattice_in_its_own_frame`). The marker's matrix is the Shepard
average of its neighbours' cells with the same sample-centred weights; their per-axis
extents $\sum_j |H_{kj}|$ are the resolution bounds the directory and sizing use.
The averaged matrix must still pass the existing invertibility and stencil-quality
checks; arbitrary cell orientations do not guarantee a usable average.

## Coverage: a contract, never a guess {#cpl_coverage}

The numerical requirement is the same for every input: every active sample a row's
kernel can touch must be present, exactly once. Two representations, because they
admit two different proofs.

**A LATTICE location** (uniform or rectilinear) is declared GLOBALLY — identical
tables on every rank, checked by hash — and owned in index blocks. The requester
therefore knows how many samples a request box must hold, and every owner it asked
reports how many of its own it has there: active, masked (no such sample: a solid) or
owned-but-unexported. The sum must equal the count exactly. Less is an ownership gap,
reported as UNPROVIDED; more is two owners of one sample, INVALID; unexported ones are
NEED_COVERAGE with the exact missing index boxes handed to their owner
(`test_a_one_cell_hole_is_found_and_a_masked_cell_is_not` carves ONE cell out of a
selected export and gets exactly that index back — the remaining samples would have
given a perfectly conditioned linear fit). On an unmasked FULL block the count is
arithmetic.

**A POINT location** cannot be enumerated by anyone but its owner, so its owners
certify regions: `MFTL_ProviderRegions` (where a rank owns samples, exported or not)
and `MFTL_CoverageDesc` (where its export is complete). A FULL point set defaults to
tiles of at most 64 Morton-consecutive samples' cell extents, split further when
their resolution ratio would exceed two on any axis. Two limits are stated rather than
hidden. A false certificate cannot be detected from a point cloud: a withheld sample
inside a certified region goes through, and only an independent full input shows the
difference (`test_the_selected_point_protocol_and_its_limit` asserts both halves). And
a hole in ONE registered cloud thinner than a tile is claimed by the tile's box —
certified empty — while a gap between ranks or between separately registered sets is
reported (`test_an_ownership_gap_is_never_a_wall`); explicit regions make it exact.
Closed certificates also cover degenerate provider regions (planes, lines and
points), including the zero-width slabs that can occur on empty ranks.

**Unknown is never a wall.** A region no rank claims is UNPROVIDED — never a truncated
row. Walls are DECLARED: the domain's non-periodic faces, the lattice's own extent, and
masked (solid) cells.

**A truncated row is counted, not guessed.** A lattice is declared globally, so a row
counts by arithmetic the lattice positions its support holds — over the lattice
extended past its ends by the end spacing — and compares with the samples it got. A
row missing a position with weight is TRUNCATED: past a wall, past the edge of a
lattice smaller than the domain, or on a masked cell; corrected under
`MFTL_BOUNDARY_TRUNCATE`, refused under `MFTL_BOUNDARY_ERROR`. A support that merely
crosses a wall with no position beyond it — a marker 1.6 cells from it, Peskin's reach
being two — is complete and not refused (`test_truncation_is_counted_exactly`). The
same count is a self-check: a row missing an unmasked interior sample cannot happen
once coverage is proved, and is reported as an internal error rather than corrected
away (a mutation that drops one sample from the band is caught on the first row).

**Mirror walls.** `MFTL_BOUNDARY_MIRROR` completes a support beyond a wall with mirror
images of the location's own samples, reflected about the wall plane (in x, then y,
then z at an edge or corner). Each face has a parity: EVEN images carry the sample's
value (a zero normal derivative), ODD ones $2u_w - u$, with the wall value $u_w$ given
per field. The stored row is composed onto the real samples; an ODD face's share is a
separate per-row coefficient of $u_w$, which interpolation adds and spreading reports
as the face's REACTION, so

$$
\sum_j b_j + \sum_f R_f = \sum_m F_m, \qquad U \cdot F = u \cdot b + \sum_f u_{w,f} R_f .
$$

On a wall-aligned lattice the virtual lattice is complete, so Peskin's rows need no
correction at a wall at all and EVEN rows stay nonnegative; a profile linear through
the wall value is reproduced exactly at any distance from an ODD wall. The reference
in `test_coupling_boundaries` never reflects anything: it maps virtual lattice indices
to the indices they mirror by integer arithmetic.

**Constrained samples.** A sample whose value the solver BUILDS — a hanging node,
$u_c = \sum_j P_{cj} u_j + u_{bc}$ — is registered on its owner with its row of P
(`MFTL_ConstraintDesc`); every free-owner claim is checked collectively when the grid is
taken up. Owner ranks use the solver communicator's ordering and are translated to the
coupler's ordering, so a reordered solver communicator works for constraints too.
Interpolation reads the stored value; spreading applies the exact transpose:
the load a constrained sample accumulates goes to its free samples as $P_{cj} b_c$
(remote ones through one more exchange), its own element receives nothing, and $b_c$ is
reported so that the affine part's work closes the power identity,
$U \cdot F = \sum_{free} u_j b_j + \sum_c u_{bc,c} b_c$.

**FULL and SELECTED give the same answer.** A SELECTED band built by the protocol
alone — export nothing, export exactly what each NEED_COVERAGE names, retry —
converges in a few rounds and then interpolates bit-identically to the FULL export,
while exporting 24.5 % of the grid
(`test_a_selected_band_built_from_requests_equals_full`). A selection-only update keeps
the directory.

---

## Discovery and the window {#cpl_window}

```
create/update  publish ownership and local metric influence certificates
               to the rank boxes they touch                              (NBX)
request        occupied marker bins plus skin: look up metric certificates
               and bound each bin's possible resolution                  (NBX)
               look up ownership for bin + skin + local support bound    (NBX)
               request samples from the certified owners                 (NBX)
               replies: counts per box, each sample once per requester   (known peers)
               coverage verified, status agreed                          (one MAX)
window         band (routes, images, bins), pack lists, served table     (no messages)
```

**Discovery is geometric and sparse.** The ranks a box touches are enumerated
arithmetically from the process grid — never a loop over P. Requests and directory
entries go by Hoefler–Siebert–Lumsdaine's NBX: synchronous sends, probe and receive,
a non-blocking barrier once this rank's sends have been matched; a phase's messages
can never leak into the next. Nothing assumes 26 neighbours: a kernel wider than a
rank box reaches owners several boxes away (`test_a_support_wider_than_a_rank_box`,
nine or more peers on 3×3×3). Directory shards use compact bounds trees. A marker
far from its rank looks up certificates in the relevant shards; it does not grow
every rank's directory (`test_user_markers_far_from_their_rank`). The active lookup
cache is reset at each new support request, so motion does not gradually replicate
the global directory. `directory_margin` is optional extra advertisement padding,
not a globally enlarged numerical reach.

Uniform locations already have exact resolution bounds. Requests involving only
uniform locations skip the metric-lookup exchanges entirely.

**Resolution certificates are separate from ownership.** FULL point providers use
actual sample tiles for metric bounds even if their declared ownership box spans
the domain. Rectilinear locations use small index tiles derived from the declared
lattice, including positions nobody owns; otherwise an unowned coarse cell could
shrink the pilot until the request missed that ownership gap. Tile construction is
distributed across ranks, separately from sample ownership. SELECTED point providers must
declare conservative bounds for unseen samples: a broad coarse certificate can
still over-request. No tighter bound is inferred from an incomplete export.
Every certificate bounds the sample-centred pilot influences that can touch the
movement envelope. Their maximum resolution bounds both the pilot contributors
and the final averaged kernel, including allowed support growth. The subsequent
ownership/sample request therefore covers all stages admitted by that window.

**Requests overlap, replies do not.** A request box is a marker bin widened by the
local reach and the skin, so a sample can lie in many of them.
Lattice ownership boxes are checked for overlap before their counts are accepted:
a duplicate masked sample must not cancel an unowned hole. Every box is COUNTED,
but a sample is returned, packed and
fetched once per requester. The first version enumerated, sent and sorted every sample
once per overlapping box. A historical comparison measured 148 ms versus 7 ms per
window on a 64³ MAC grid; these are not timings of the current architecture.

**Routes.** A band sample is SELF (owned here, read in place), HALO (owned elsewhere
but held in a ghost layer this rank's block declares, read in place when the field
says its ghosts are valid, its loads returned to the owner) or REMOTE (fetched; loads
returned). Reading valid ghosts in place is the same arithmetic as fetching, to the
bit, with fewer bytes (`test_the_halo_route_changes_no_bit` — whose third run corrupts
the ghosts while still claiming them valid and gets a different answer, which is what
proves the halo was read at all).

**Periodicity.** A lattice image is the canonical coordinate of `i mod N` plus
`floor(i/N)` periods — one formula, used by requester and provider alike — and a
support plus the skin must stay below half a period, so a marker meets one image of a
sample. The rows on a periodic box equal brute force across the seam.

**The skin.** Marker and stage positions within `skin` of a bin that held a
requesting marker stay inside the window: binds need no new request and no new fetch.
Anything farther is STALE on every rank, never a row over uncertified samples
(`test_a_marker_one_bin_outside_the_window_is_stale` — which a mutation that dropped
the skin test passed until it existed). A position belongs to the bin `floor()` puts
it in, judged by the very expression the request binned with: the reconstructed
interval $[\min + i b, \min + (i+1) b]$ can exclude a position one ulp below a face
whose quotient rounded up, and such a marker used to be STALE in the window requested
for it — forever, since a new request reproduced the same bins
(`test_a_marker_an_ulp_below_a_bin_face_binds`).

**Migration.** With `migration` set, every rank also sends its bins to the ranks whose
box lies within that distance of them (geometric discovery, one NBX), and their windows
hold those bins too: after a redistribution has moved markers that far onto another
rank they bind in the window it already has, with no request and no fetch. On a
connected surface the new owner's own neighbouring markers usually hold those bins
anyway; the allowance makes it a guarantee
(`test_a_migrated_user_marker_binds_in_the_old_window`).

**What an update keeps.** A change of what is OWNED or of a constraint map stales
every window (constraint indices and transpose routes are cached); a change of
EXPORTS alone keeps every window whose served samples are all still exported, checked
by the window's next collective use. A selection that slides past a band costs that
band nothing; one that withdraws a served sample stales it on every rank, and
restoring the export revives it. Re-registering identical ownership keeps the
directory (`test_a_selection_change_keeps_windows_that_stay_exported`).

---

## Fast paths and reuse {#cpl_fast_paths}

The intermediate Cartesian structure contains **original fluid samples**. It is a
search and communication cache, never a second interpolated velocity mesh.

On complete uniform lattices with the default four-point Peskin policy, sparse
4×4×4 bricks map global `ijk` indices to cached sample columns. A bind computes the
stencil by floor/index arithmetic and evaluates twelve 1-D kernel values. The row
stores those twelve factors and 32-bit local columns (360 bytes with the present
layout, versus 1024 bytes for 64 double weights and 64-bit columns). Interpolation
and spreading evaluate the same products in the same order. Sample ownership has
no role in the local stencil loop. `BindingQuality.n_cartesian` reports this path.

Walls, masks, mirror boundaries, stretched coordinates, changed support scale,
other kernels, and irregular point sets use the checked general row path as
needed. Missing coverage is never replaced by a guessed stencil. Mirror locations
always use their existing wall-aware builder. Bands beyond the compact column
index range also use general rows.

Point providers keep an index over their Morton tiles. Irregular cached samples
are ordered spatially inside Cartesian bins and indexed in groups of 32. Queries
skip disjoint groups while retaining a global, partition-independent accumulation
order. All coordinates and bounds are physical; bin width does not replace the
numerical kernel metric.

General rows lazily construct one candidate-image list per occupied marker bin and
location, covering that bin's certified movement and support envelope. Stage binds
and later bindings reuse the list, filter it to their actual support and recompute
all weights. Remeshing can reuse the same lists while the window remains valid.
`n_candidate_builds` and `n_candidate_reuses` expose this reuse. A successful direct
Cartesian stencil needs no such list. Only occupied bins exist; global minimum
resolution sets their search granularity, never a dense finest-level grid.

Windows cache peer-to-band segments. Transfer preparation caches packed offsets
for the field batch and agrees validation plus signatures in one reduction.
Repeated transfers use these maps and reusable communication buffers; released
fetch buffers are reused without overwriting any still-live snapshot. General-row
candidate scratch is also reused across binds. A checked convenience interpolation
uses three validation reductions instead of six; prepared Begin/End uses two.
These checks remain collective so a stale handle or bad load on one rank cannot
strand the others or partly modify targets.

Both LeVeque examples first bind in the existing window and refresh collectively
only when it is stale. They request a migration allowance as well as a motion skin.
Weights are still rebuilt at every RK stage and fields are fetched at each distinct
stage time. Remeshing invalidates bindings, not every fluid window.

---

## Bindings and transfers {#cpl_binding}

A binding is the rows of an ordered set of markers at one geometry: VERTICES (owned
vertices, weight the dual area $\sum_{T\ni v}A_T/3$ over the vertex-complete halo),
CENTROIDS (owned triangles, weight the area) or USER. Both kinds of weight sum to the
surface area to round-off (`test_marker_weights_sum_to_the_area`). `BindStage` builds
the rows of the same markers at RK stage positions inside the same window — one status
agreement, no request, no fetch — and equals a fresh bind at those positions, bit for
bit.

**Rows do not depend on the partition.** Candidates are gathered in a GLOBAL bin order
and fetched values are copies, so the interpolated values at a marker are the same bits
at np = 1, 2, 4, 8 and 27, for any solver decomposition and at a length unit of
$2^{-20}$ (`test_results_are_bitwise_partition_and_unit_independent`, against a rerun on
rank 0 alone). Spreading sums contributions from several ranks, in ascending source
rank order on every owner: reproducible at a fixed decomposition, equal to round-off
across decompositions.

### Surface quadrature {#cpl_quadrature}

`MFTL_MARKERS_QUADRATURE` cuts every owned triangle into $m^2$ congruent sub-triangles
with one point at each sub-centroid, weight $A/m^2$ — exact for the vertices' linear
hat functions, so a vertex's points weigh its dual area. $m$ is chosen per triangle so
that a sub-triangle's longest edge stays within `quadrature_ratio` of the finest fluid
resolution the directory knows near it: the surface mesh need not be refined to the
fluid. A quadratic integrated over the facets converges at $1/m^2$ (ratios 3.98–4.00
measured). Two collective conversions form a composite adjoint pair,
with the lumped mass $M_S$ the points define:

$$
U_v = M_S^{-1} B^T Q\, U_q, \qquad F_q = Q B\, M_S^{-1} F_v, \qquad U_v \cdot F_v = U_q \cdot F_q,
\quad \sum_q F_q = \sum_v F_v .
$$

A point contributes to the three corners of its triangle, which may be ghosts; the
binding holds a ghost schedule (built once, by NBX) that completes per-vertex sums and
delivers owned values to the ghost copies. The projection equals its definition
recomputed from geometry vertex by vertex, and the same surface on one rank alone, to
round-off (`test_coupling_quadrature` — a swapped barycentric pair keeps constants,
masses and the adjoint identity, and is caught only there).
Conversions agree the map identity, direction, component count and load form before
exchanging data. Quadrature areas, including stage weights, must be finite and positive;
degenerate triangles and invalid stage areas return INVALID.

**Local stage binds.** `MFTLCouplerBindStageLocal` builds the same rows as
`BindStage`, to the bit, with no communication — it may run while a fetch is in flight
— and returns THIS rank's status; the caller agrees it in its own step decision or with
`MFTLCouplerAgreeStatus` before anything depends on it.

**The solver can complete the ghost route.** With `halo_transpose` a spread leaves the
halo route's loads in the ghost elements they were read from and zeroes every other
element of the declared ghost layers, so the solver's single reverse halo accumulation
completes $W^T F$ and adds nothing its arrays held before. A ghost element that is, in
memory, an owned sample of another block (two blocks over one array) is left alone.
Refused on a constrained location, whose transpose the solver would bypass.

**Transfers are agreed before they start.** `PrepareTransfer` agrees the direction,
the locations and flags in order, and the handles' currency. The token retains the
window and binding. Begin agrees the token identity and rechecks staleness and inputs
before posting any messages; End agrees its arguments before completing the transfer.
This catches changes between preparation and use, and a missing load pointer on just
one rank cannot strand its peers. A non-finite load on one rank is refused on every
rank before any target is written. Tokens cannot be released during an active transfer.
Fetch and spread are split into Begin/End
over fixed schedules; messages larger than the chunk limit travel as several,
reassembled in order (`test_chunking_and_repetition_change_no_bit` at 40-byte chunks).

### Staleness {#cpl_staleness}

Every library call that changes a mesh renews the object's GENERATION
(`MFTL_Object::generation`, `MFTLObjectMarkChanged`): entity counts, pool order, ghost
positions, the remeshing operators and smoother, the topology change's corrections. A
binding records its bodies' generations and refuses itself afterwards; a redistribution
that rebuilds a body renews it even when the caller forgot to announce moving the
vertices (`test_a_redistribution_stales_a_binding`); a skipped body keeps its stamp,
because its pool is bit-identical. A caller writing `verts.pos` owes the call; a Debug
checksum of the bound positions catches the forgotten one. Registration edits invalidate
local consumers immediately, before `UpdateGrid`; collective calls agree that rejection
across ranks. `UpdateGrid` then invalidates what the change reaches: an ownership or constraint-map change
every window, fetch and transfer token; an export-only change only the windows whose
served samples left the export. Failed interpolation validates all fields and ranks before writing any
results. A skin may span several bins; both fresh and stage binds search that full allowance.

---

## The three surface loads {#cpl_surface_loads}

`Spread` takes integrated marker loads; it does not multiply by marker area.
For three components, pass three target fields and `loads[3*i + component]`.
The same code handles all three requested uses:

| Source | Integrated load at marker i |
|---|---|
| Surface tension | The integrated force supplied by the surface-force calculation; if supplied as traction, multiply by the marker area once |
| IBM | The integrated reaction force from the caller's IBM enforcement step |
| Indicator gradient | For indicator 1 inside and 0 outside, `-normal[i] * area[i]`, with outward normals |

Use `MFTL_SPREAD_INTEGRATED` for per-sample integrated loads, or
`MFTL_SPREAD_DENSITY` to divide each load by the registered sample measure.
Quadrature markers provide the area weights needed for triangle sources. Normals
must correspond to the geometry being bound; a stage's moved marker positions
alone do not determine its triangle normals. An indicator-gradient source is
spread as a vector source; this does not reconstruct the indicator or solve its
Poisson equation. No solenoidal projection is applied.

For power consistency, pair spreading with interpolation on the same binding and
at the same geometry. Vertex↔quadrature conversion has its own exact transpose;
an independently assembled quadrature interpolation is not automatically the
vertex interpolation composed with that conversion.

---

## The solver loop {#cpl_loop}

```
window = none
for each step:
    b0 = bind(window)                           # rows at X^n, if a window exists
    if no window or any rank is STALE:
        window = request + prepare at X^n, with motion skin and migration allowance
        b0 = bind(window)
    k1      = interpolate(b0, fetch(u at t))
    b2      = bind_stage(b0, X^n + dt/2 k1);  k2 = interpolate(b2, fetch(u at t + dt/2))
    b3      = bind_stage(b0, X^n + dt/2 k2);  k3 = interpolate(b3, same fetch)
    b4      = bind_stage(b0, X^n + dt   k3);  k4 = interpolate(b4, fetch(u at t + dt))
    X^{n+1} = X^n + dt/6 (k1 + 2 k2 + 2 k3 + k4);  MFTLEnsembleMarkChanged
    ghosts, geometry, redistribute (or remesh)
```

If a later stage leaves the window, release its dependent handles, refresh and
retry the step before committing positions. A time-dependent field is fetched at each distinct stage time; reusing the stage-1
rows at every stage would be Euler. Frozen-field RK4 (one fetch) is fourth order for
the frozen field only, not for a coupled scheme.

---

## Measured on LeVeque's vortex {#cpl_leveque}

`examples/leveque_coupled_cartesian.c` and `examples/leveque_coupled_unstructured.c`
advect LeVeque's sphere with the velocity known only on a solver's samples, and
compare with the same RK4 on the analytic field. The following are historical
measurements, before the current cache/stencil refactor and local metric: np = 4, laptop (Apple M-series),
Release, refine 3 (642 vertices), T = 3, dt = 0.02.

**Pass A** — no remeshing; every vertex against its own exact trajectory at maximum
deformation (t = T/2):

| solver samples | h | rms \|Δx\| | max \|Δx\| | max \|Δu\| | volume vs direct | order |
|---|---|---|---|---|---|---|
| MAC 16³, z-slabs | 0.0625 | 1.01e-1 | 1.73e-1 | 1.95e-1 | +10.9 % | |
| MAC 32³ | 0.0313 | 2.67e-2 | 4.77e-2 | 5.04e-2 | +2.0 % | 1.93 |
| MAC 64³ | 0.0156 | 6.73e-3 | 1.20e-2 | 1.27e-2 | +0.45 % | 1.99 |
| tetrahedra 3 072, Morton chunks | 0.0688 | 1.65e-1 | 3.12e-1 | 3.42e-1 | +18.5 % | |
| tetrahedra 24 576 | 0.0344 | 4.63e-2 | 9.00e-2 | 1.07e-1 | +3.7 % | 1.83 |
| tetrahedra 196 608 | 0.0172 | 1.18e-2 | 2.34e-2 | 2.71e-2 | +0.66 % | 1.97 |

Second order in the fluid spacing on both — on the unstructured mesh, which is graded
(cells 1.9× larger inside than at the walls) and jittered, through moment-corrected
rows and a Shepard metric. The state at T is NOT a coupling measure: the interpolated
field $\cos(\pi t/T)\,W u(\mathbf x)$ is exactly time-reversible too, so its
trajectories return to their start as the exact ones do (both end within 1e-6).

**Pass B** — remeshed every step, the way a front tracker runs; at t = T:

| | direct | MAC 64³ | tetrahedra 196 608 |
|---|---|---|---|
| V/V0 | 0.747 | 0.738 | 0.723 |
| A/A0 | 1.173 | 1.166 | 1.148 |
| max \|\|x−c\|−R\| | 0.138 | 0.141 | 0.140 |
| min triangle quality | 0.71 | 0.76 | 0.41 |

The 25 % volume loss is the remesher's at this resolution (the original example's too);
the coupling adds 1 % on the lattice, 3 % on the tetrahedral mesh, where the
interpolated field is rougher — which is also what costs the remeshed run its
worst triangle.

**The solver's decomposition changes no digit and most of the traffic.** The same
64³ MAC run with the solver's blocks aligned to MFTL's boxes and two valid ghost layers
gives the same errors to every printed digit as z-slabs, and fetches 36 MB instead of
249 MB over the run: per rank the band is 34 000 samples read in place, 5 000 from
ghost layers, 4 200 remote, against 9 500 / 0 / 33 700.

**Cost**, coupling only, per RK4 step on 4 ranks: 15.6 ms on the 64³ MAC grid
(window 7.8, four binds 7.1, three fetches 0.3), 30 ms on 196 608 tetrahedra
(window 14, binds 15). Getting there took three measured fixes, each recorded in
`docs/analysis.md` §11. Factored Cartesian rows and reusable candidate lists have
since been implemented; their performance has not been measured here.

---

## Found by attack {#cpl_attack}

`tests/test_coupling_adversarial.c` breaks the habits the first tests were written
around — several blocks per rank in a random k-d partition, a ghost width per block
and axis, padding that holds NaN, anisotropic lattices on a box that does not start at
the origin, periodic on some axes only, interleaved components, more ghosted blocks
than the halo route tracks, two couplers interleaving their transfers — against brute
force over the global lattice, and at np = 1 … 27 to the bit against one rank alone.
Two bugs fell out, both silent in the sense that matters (a plausible status, not a
crash):

- **A USER marker more than one period away** was wrapped by one period, binned in a
  frame no provider answers, and came back as NEED_COVERAGE for samples that existed.
  Every marker is now wrapped onto $[\min, \max)$ however far away it is, and the
  position one ulp below `min` — which plus a period rounds onto `max` — onto `min`.
- **A marker one ulp below a bin face** was STALE in the window requested for it,
  forever (see the skin, above).

The truncation count, the non-finite-load refusal and the rank-local assertions in the
tests themselves (a `TEST_ASSERT` inside `if (rank holds the marker)` strands the other
ranks in the next collective) came out of the same exercise.

## Scope and limits {#cpl_deferred}

The module transfers velocity components to front-tracking/IBM markers and returns
integrated surface loads to fluid samples. It does not interpret a solver's
populations, basis functions, modal coefficients or flux DOFs. A solver supplies
physical component samples and their integration measures. Active AMR leaves can
be supplied as point sets; covered parent cells must not also be exported.

The former C3 directional reconstruction, arbitrary gradient operators, imported
rows and nearest-surface queries have been removed, including their public APIs and
feature-specific tests. Nearest-surface/indicator reconstruction is a separate
geometry/solver task. Scalar field batches again have one value per marker per
field; `BindingWallCoefficients` takes no component argument. Callers using C3
must migrate; there are no compatibility stubs.

There is no divergence-free Cartesian B-spline/Bao mode. Ordinary Peskin transfer
is not pointwise divergence-free. The existing small scalar kernel table (including
`BSPLINE3`) is unrelated to a staggered divergence-free scheme and adds no separate
architecture. The default remains `PESKIN4`.

Packed nonblocking point-to-point MPI remains the transport. There are no PETSc,
LBM or FEM dependencies, framework adapters, GPU transport or Fortran wrappers.
Both decompositions use the same MPI process group, although rank order and spatial
partitions may differ. Point-set changes still require re-registration; constraint
maps are single-level and within one location.

The variable-resolution metric is local, but genuinely coarse samples whose
influence reaches a marker can still enlarge its neighbourhood. Coarse SELECTED
provider certificates can also over-request. Arbitrarily sparse or badly shaped
clouds can fail the numerical stencil checks; search cannot fix missing numerical
support. The directory and bands are sparse; there is no replicated process-count
or process-count-squared routing table.

Released fetch storage is cached in at most three snapshots per window and is freed
with that window. Live snapshots are immutable. Candidate lists also live with the
window. The optional memory limit described below covers both kinds of storage;
it does not evict caches or recover from exhaustion. Persistent MPI requests and
automatic overlap of communication with interior interpolation are not implemented.
Performance claims require measurements; the historical timings above are not
timings of this revision.

## Allocation accounting and budget {#cpl_memory}

`MFTLCouplerGetMemoryStats` reports the current and peak accounted allocation
capacity across **all grids and couplers on this rank**, including registry data,
directories, live handles, rows, snapshots, retained caches and temporary buffers.
Each allocation carries a small aligned size header, also counted. This measures
module-owned capacity, not process RSS: solver arrays, MFTL surface storage, MPI
buffers and the C allocator's internal storage are excluded. The older
`MFTL_CouplerStats.bytes_memory` remains an estimate of directory/window payload;
use the new API for allocation totals.

For example, before grid registration, set a 256 MiB limit on every rank:

```c
if (MFTLCouplerSetMemoryLimit((size_t)256 * 1024 * 1024) != MFTL_SUCCESS)
  return MFTL_ERROR; /* A proposed limit below current usage is rejected. */
MFTL_CouplingMemoryStats memory;
MFTLCouplerGetMemoryStats(&memory);
```

Zero disables the limit (the default). Configure it only while no other thread is
inside coupling calls on this rank. `MFTLCouplerResetMemoryPeak` sets the peak to
current usage without freeing storage. Read at a quiescent point for a consistent
snapshot of the counters.

An allocation that would exceed the limit fails **before allocation** and aborts
the MPI job with the allocation name, requested bytes, live bytes and limit.
This follows MFTL's fatal allocation-failure policy, rather than returning from one
rank while peers wait or a transfer is partly applied. It is not a recoverable
NEED_COVERAGE condition. The budget test exhausts one rank while another waits in
MPI, checks the diagnostic and requires prompt process termination; lifecycle
tests check that releasing all module objects returns live bytes and blocks to zero.

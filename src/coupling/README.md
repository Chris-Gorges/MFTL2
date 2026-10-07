`src/coupling/` — Coupling to a flow solver {#mftl_mod_coupling}
===========================================

**Public header:** `include/mftl_coupling.h` · **Layer:** 3 ·
**Module group:** @ref mftl_coupling

A flow solver owns velocities, pressures and forces on ITS samples — cell centres,
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
| `coupling_rows.c` | one row from candidates: base weights, moment correction, the four quality checks | no |
| `coupling_markers.c` | markers from the surface, dual-area weights, the generation snapshot | no |
| `coupling_directory.c` | lattice index arithmetic and images, box algebra, the directory's grid and cover | no |
| `coupler.c` | the state machine: directory, request, window, binding, transfers | no |
| `coupling_comm.c` | **every MPI call the module makes** | **yes** |

`coupling_internal.h` declares what the files share; `test_coupling_mpi_is_confined`
fails the build if an MPI call appears anywhere else. The coupler communicates on its
own duplicate of the ensemble's communicator and never probes a message it did not
cause.

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
`h_uniform`) uses it: the kernel metric is `support_scale` times $h$. Otherwise the metric is
a Shepard average of the samples' resolutions over a pilot neighbourhood of radius
`pilot_factor` times the location's largest resolution, with a Wendland weight — a
smooth function of the marker's position, computed from samples the window
certified, so it is the same on every rank.

---

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
tiles of 64 Morton-consecutive samples' cell extents. Two limits are stated rather than
hidden. A false certificate cannot be detected from a point cloud: a withheld sample
inside a certified region goes through, and only an independent full input shows the
difference (`test_the_selected_point_protocol_and_its_limit` asserts both halves). And
a hole in ONE registered cloud thinner than a tile is claimed by the tile's box —
certified empty — while a gap between ranks or between separately registered sets is
reported (`test_an_ownership_gap_is_never_a_wall`); explicit regions make it exact.
Closed certificates also cover degenerate provider regions (planes, lines and
points), including the zero-width slabs that can occur on empty ranks.

**Unknown is never a wall.** A region no rank claims is UNPROVIDED — never a truncated
row. Walls are DECLARED: the domain's non-periodic faces, and the lattice's own extent.
A row cut by a wall is corrected over the samples inside under `MFTL_BOUNDARY_TRUNCATE`
(counted), refused under `MFTL_BOUNDARY_ERROR`.

**FULL and SELECTED give the same answer.** A SELECTED band built by the protocol
alone — export nothing, export exactly what each NEED_COVERAGE names, retry —
converges in a few rounds and then interpolates bit-identically to the FULL export,
while exporting 24.5 % of the grid
(`test_a_selected_band_built_from_requests_equals_full`). A selection-only update keeps
the directory.

---

## Discovery and the window {#cpl_window}

```
create/update  directory: every rank publishes its ownership extents, widened by
               the directory reach, to every rank box they touch          (NBX)
request        markers binned at half the kernel reach; per bin and location
               one box: the bin widened by the reach and the skin; per box
               the owners the directory names                            (NBX)
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
nine or more peers on 3×3×3). The directory's reach grows, collectively, when a
request needs more (USER markers far from their rank,
`test_user_markers_far_from_their_rank`).

**Requests overlap, replies do not.** A request box is a marker bin widened by the
reach and the skin — about ten bins across — so a sample lies in hundreds of them.
Lattice ownership boxes are checked for overlap before their counts are accepted:
a duplicate masked sample must not cancel an unowned hole. Every box is COUNTED,
but a sample is returned, packed and
fetched once per requester. The first version enumerated, sent and sorted every sample
once per overlapping box: 148 ms per window on a 64³ MAC grid, against 7 ms now.

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
the skin test passed until it existed).

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

**Transfers are agreed before they start.** `PrepareTransfer` agrees the direction,
the locations and flags in order, and the handles' currency. The token retains the
window and binding. Begin agrees the token identity and rechecks staleness and inputs
before posting any messages; End agrees its arguments before completing the transfer.
This catches changes between preparation and use, and a missing load pointer on just
one rank cannot strand its peers. Tokens cannot be released during an active transfer.
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
across ranks. `UpdateGrid` then invalidates every window, including existing fetches and
transfer tokens. Failed interpolation validates all fields and ranks before writing any
results. A skin may span several bins; both fresh and stage binds search that full allowance.

---

## The solver loop {#cpl_loop}

```
for each step:
    window  = request + prepare at X^n, skin covering every stage of the step
    b0      = bind(window)                       # rows at X^n
    k1      = interpolate(b0, fetch(u at t))
    b2      = bind_stage(b0, X^n + dt/2 k1);  k2 = interpolate(b2, fetch(u at t + dt/2))
    b3      = bind_stage(b0, X^n + dt/2 k2);  k3 = interpolate(b3, same fetch)
    b4      = bind_stage(b0, X^n + dt   k3);  k4 = interpolate(b4, fetch(u at t + dt))
    X^{n+1} = X^n + dt/6 (k1 + 2 k2 + 2 k3 + k4);  MFTLEnsembleMarkChanged
    ghosts, geometry, redistribute (or remesh)
```

A time-dependent field is fetched at each distinct stage time; reusing the stage-1
rows at every stage would be Euler. Frozen-field RK4 (one fetch) is fourth order for
the frozen field only, not for a coupled scheme.

---

## Measured on LeVeque's vortex {#cpl_leveque}

`examples/leveque_coupled_cartesian.c` and `examples/leveque_coupled_unstructured.c`
advect LeVeque's sphere with the velocity known only on a solver's samples, and
compare with the same RK4 on the analytic field. np = 4, laptop (Apple M-series),
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
`docs/analysis.md` §11. Re-measure before optimising further; factored tensor rows
on lattices and the per-bin binary search of the candidate gather are the levers
not taken.

---

## What is not here {#cpl_deferred}

From plan 06's milestones, these are not implemented and nothing refers to them:
composite AMR constraints and constrained-DOF (boundary, hanging-node) transpose maps;
directional normal-velocity samples and imported native FE/DG rows; gradient operators;
adaptive surface quadrature markers; incremental grid deltas (a grid update is a full
replacement that keeps the directory when ownership did not change); persistent
point-to-point and neighbour-collective transport backends (the baseline is packed
non-blocking point-to-point); local stage binds without a status agreement; the
solver-completed transpose halo (HALO loads always go back to the owner); windows
seeded with the envelopes of markers arriving from other ranks (a window is per rank's
own markers; a marker that migrated in is STALE and the step re-requests). Factored
tensor rows are not used: every row is CSR.

The plan's performance gates also remain open: allocation-free repeated operations,
a configurable memory budget and complete peak-allocation accounting, and removal of
the temporary rank-count-sized scratch lists used in discovery. `bytes_memory` estimates
the main directory/window payload, not total allocation. The measurements above predate
the additional transfer validation; they are not timings of the reviewed implementation.

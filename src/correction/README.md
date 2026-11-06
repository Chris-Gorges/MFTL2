`src/correction/` — Front corrections {#mftl_mod_correction}
==========================================

**Public header:** `include/mftl_correction.h` · **Layer:** 3 ·
**Module group:** @ref mftl_correction

Two corrections a front-tracking solver makes between flow steps, both along the
surface normal and both conserving what they should exactly:

| call | does | conserves |
|---|---|---|
| MFTLEnsembleRemoveUndulations() | TSUR3D: removes grid-scale undulations edge by edge | every body's volume, per operation, to round-off |
| MFTLEnsembleCorrectVolumes() | restores each body's #MFTL_Object::reference_volume by a uniform normal offset | -- it is the restoration |

[TOC]

# Why they are not inside the remesher {#corr_outside}

The remesher's Phong back-projection (#MFTL_REMESH_PROJECT_PHONG,
`src/remeshing/README.md`) moves every smoothed vertex, once at the end of a
kernel call, back onto the surface that call RECEIVED. That is what makes the
remesher's own tangential smoothing leave the shape alone, and it means that any
normal motion made inside the call is undone by the call. A denoising step or a
volume correction made between the call's input copy and its projection would be
projected away.

Called before the remesh, a correction is part of the surface the remesh
receives, and the projection keeps it; called after, the remesh never sees it.
Both are therefore separate, collective calls in the solver's own loop. The
remesher calls neither and knows nothing of them.

# TSUR3D {#corr_tsur}

## The edge operation

TSUR3D (de Sousa, Mangiavacchi, Nonato, Castelo, Tomé, Ferreira, Cuminato and McKee,
J. Comput. Phys. 198 (2004) 469–499, doi:10.1016/j.jcp.2004.01.032 -- the paper's
"mass-conserving filter" for small undulations) acts on one edge \f$a\f$–\f$b\f$ at a time:

1. \f$\vect n_a, \vect n_b\f$: the area-weighted normals of the two vertex fans,
   computed with BOTH ends moved to the centroids of their one-rings -- an end's
   own displacement must not tilt the normal it is judged by;
   \f$\vect n = (\hat{\vect n}_a + \hat{\vect n}_b)/\lVert\cdot\rVert\f$.
2. With the midpoint \f$\vect m\f$ as origin the ends sit at heights
   \f$h_a = (\vect a - \vect m)\cdot\vect n = -h_b\f$.
3. Both ends move along \f$\vect n\f$ to ONE height \f$h\f$: the edge ends up
   perpendicular to the smoothed normal, which is what a smooth surface's chord
   is to third order in the edge length.

The height is chosen so that the enclosed volume does not change. Moving \f$\vect a\f$
by \f$d\,\vect n\f$ changes the volume by exactly \f$d\,\vect n\cdot\vect S_a/3\f$, where
\f$\vect S_a = \tfrac12\sum (\vect x - \vect a)\times(\vect y - \vect a)\f$ is the
vector area of its fan, summed over its link edges \f$(\vect x, \vect y)\f$. That
change is linear in \f$d\f$, because each tetrahedron's volume is linear in each
corner. \f$\vect S_a\f$ depends only on the link, so moving \f$\vect b\f$ along
\f$\vect n\f$ does not change it. And the two triangles holding both ends gain no
cross term, because \f$\det(\vect n, \vect n, \cdot) = 0\f$. With the projected
fan areas \f$P = \vect n\cdot\vect S\f$,

\f[
  h = \frac{h_a P_a + h_b P_b}{P_a + P_b}
\f]

conserves the volume EXACTLY. This is the published
\f$h = (V_1 + V_2)/(V_1/h_1 + V_2/h_2)\f$ with the local fan volumes
\f$V_i = h_i P_i/3\f$ written out and cancelled.

## What changed against mftl's implementation

| | mftl `tsur3d.c` | here |
|---|---|---|
| the height | \f$(V_1+V_2)/(V_1/h_1+V_2/h_2)\f$: **0/0 on an edge that is already flat**, which came back NaN and was skipped | the cancelled form: an exact zero, no move; a convex combination of \f$h_a, h_b\f$, so neither end moves further than the edge's own kink |
| a folded fan | divided through | refused when \f$P_a \le 0\f$ or \f$P_b \le 0\f$, and when the move would turn any fan triangle over |
| the domain | every vertex clamped `MFTL_SMALL` inside the box, which **breaks the volume conservation it computes** | minimum-image geometry and the periodic wrap; nothing clamped |
| features, rims | moved | `BOUNDARY`, `FEATURE` and `CORNER` vertices stay fixed |
| parallel | every vertex gathered onto the rank holding the most | exact on the partition as it stands (below); no gather, no migration |
| scalars | `concentration *= old/new area` per fan triangle, per operation | the same rule, applied once per call (path independent), on ghost copies too |
| the trigger | a dihedral-angle variance whose sum reads `roughness_array[counter]` (one past the end) | the operation's own residual: the kink angle, `min_angle` |

## On a smooth surface it is third order; on an under-resolved one it is a smoother

For an edge of length \f$\ell\f$ on a smooth curve the kink is \f$\ell^2 f'''/12\f$:
a resolved sphere is nearly a fixed point. A clean icosphere at refinement 4 moves
by at most 0.3 % of an edge (`test_a_jittered_sphere_goes_back_to_a_sphere_at_constant_volume`),
and a flat sheet does not move to the bit. Where the mesh does NOT resolve the
geometry -- the rim of a filament one or two edges thick -- the kink is of order one
and TSUR3D flattens it like noise. At the grid scale nothing local can tell an
under-resolved feature from noise: they are the same frequency. Only time can,
because noise is renewed every step and geometry persists.

Measured on the LeVeque deformation at refinement 4 (rms distance from the exact
sphere at T, in % of R0; one sweep every step):

| noise | none | every edge | kinks > 5° | kinks > 15° |
|---|---|---|---|---|
| 0 | 2.12 | 52.9 | 13.8 | 2.64 |
| 0.05 | 5.80 | 63.5 | 26.4 | 5.53 |

and on a rigid rotation of the same sphere, which the mesh resolves at all times:

| noise | none | every edge | kinks > 5° | kinks > 15° |
|---|---|---|---|---|
| 0 | 0.006 | 0.093 | 0.006 | 0.006 |
| 0.05 | 2.81 | 0.53 | 0.77 | 1.08 |

So on a resolved interface TSUR3D removes about five sixths of the noise. On the
LeVeque filament it destroys the solution, because it is a volume-preserving
curvature smoothing applied 200 times: the area at T/2 fell from 4.01 to 3.40
while every call conserved the volume to 1e-15.

**`min_angle` therefore defaults to 15°**, not to the published 0:

- **LeVeque:** neutral in position, better in curvature, and with velocity noise
  better in both (r5: 3.67 → 2.66 % R0, E_H 14.2 → 4.4).
- **Resolved shape:** a clean icosphere is a fixed point to the bit.
- **What it leaves:** noise down to the threshold's own level, about
  \f$\tan 15°\,\ell/2\f$ of height (`test_the_default_spares_a_resolved_surface_and_still_removes_noise`).
- **Resolved but noisy interfaces:** where the mesh resolves the interface and the
  noise is what matters, 0 or a few degrees removes more (rotating sphere at r5:
  0.51 % R0 at 0°, 0.60 % at 5°, 0.80 % at 15°).

The four-orientation study is in `docs/analysis.md` §15.

## Distributed, exactly {#corr_tsur_parallel}

Every operation reads two fans and writes two vertices, so two operations on
different ranks must neither write the same vertex nor read one the other
writes. The vertices are classified once per call, from the rule redistribution
SENT by (`MFTLRemeshMarkFrozen()`'s):

| class | meaning |
|---|---|
| fixed | not owned, or flagged boundary, feature or corner |
| private | owned, and every incident triangle owned with every corner and edge neighbour owned: nobody else holds any of it |
| shared | owned, and some incident triangle is held elsewhere too |

**Pass 0** treats the edges with both ends private, on every rank at once. A
private vertex's whole fan is owned here, read and written by nobody else.
**Pass 1** treats the remaining edges with both ends owned, one COLOUR at a time:

- **The colouring:** per axis, parity, plus a third colour for the last cell of an
  odd periodic axis.
- **Why it is safe:** two ranks of one colour are never 26-neighbours, and a
  triangle holding a vertex of each would make them neighbours.
- **Synchronisation:** each round ends with MFTLUpdateGhostPositions(), so the next
  colour reads exact copies.
- **Skipping:** one Allreduce of at most 27 flags per call skips the colours with
  no work.

The volume is then exact at any process count, and the ghosts equal their owners
to the bit (`test_the_distributed_sweep_is_exact_and_agrees`, np = 1 … 27).

**What it does not do: an edge whose two ends have different owners is never
moved.** Each end still moves through its other edges. Writing the far end would
need a reverse exchange to its owner, and two active ranks of one colour could
both want to write it. Measured on a jittered sphere, three sweeps:

| np | 1 | 2 | 4 | 8 | 27 |
|---|---|---|---|---|---|
| cross-rank edges per sweep | 0 | 190 | 378 | 564 | 894 |
| residual rms against one rank | 1 | 1.017 | 1.029 | 1.048 | 1.101 |

That is a cost that grows with the cut planes' share of the surface. It is stated
here rather than hidden. Cost: `n_colours` ghost exchanges per sweep, 8 on a
2×2×2 grid and 27 on a periodic 3×3×3.

## Scalars

A normal motion carries no material across an edge, so a CONSERVATIVE_DENSITY
keeps its amount per triangle, \f$\rho' A' = \rho A\f$. Every operation conserves
the amount of every triangle it touches, so the composite of a whole call is
\f$\rho_\mathrm{end} = \rho_\mathrm{start} A_\mathrm{start}/A_\mathrm{end}\f$: one
rescale at the end, whatever touched the triangle. Every holder gathers a
triangle's corners in the same order from the same synchronised positions, so the
factor is the same bits on the owner and on every ghost. The copies stay
consistent with no field exchange, and an untouched triangle has
\f$A_\mathrm{end} = A_\mathrm{start}\f$ exactly and is left alone. Vertex fields
keep their values: the vertex is the same material point. The remesher's
smoother re-interpolates them because it moves vertices TANGENTIALLY, over the
material. Tested to 3e-16 per triangle and bit-identical ghosts.

# The volume correction {#corr_volume}

## What it does

Every owned vertex of a body moves by \f$d\,\vect u\f$. The field \f$\vect u\f$ is the
area-weighted unit vertex normal (MFTLBodyVolumeDisplacement()), with a rim vertex
on a wall keeping only its in-wall components. Its exact first variation is

\f[
  \frac{dV}{dd} = \sum_t \vect S_t\cdot\frac{\vect u_0 + \vect u_1 + \vect u_2}{3},
\f]

and the step is Newton's, \f$d = (V_\mathrm{ref} - V)/(dV/dd)\f$, repeated a fixed
number of times (3). Each step is one per-body holder reduction of
\f$(V, dV/dd)\f$ and one ghost exchange. The region and its measure are
MFTLEnsembleBodyIntegrals()'s: closed bodies, and open ones closed by the walls
their rims lie on. The optional `rim_project` is applied after every step, and the
next step measures what it did. The count is fixed so that every rank enters the
same collectives: a stopping test on a reduced double is two ranks disagreeing in
the last ulp.

This is the topology change's cluster correction as a standalone call; both use
MFTLBodyVolumeDisplacement(), so the arithmetic is shared and the topology digests
are unchanged.

## Against mftl's `VolumeRecovery`

mftl moves every vertex along the same unit normals by one step of
\f$d = (V_\mathrm{old} - V)/A\f$. The differences are two, and they are of
different size:

- **Area instead of the first variation.** \f$\sum_t \vect S_t\cdot\bar{\vect u}_t\f$
  is the area weighted by the cosine between face and vertex normals,
  \f$A(1 - O(\kappa^2\ell^2))\f$. So the area step is wrong by a relative
  \f$O(\kappa^2\ell^2)\f$ of itself and converges only linearly. Measured: 0.03 per step
  on a coarse satellite (topology change, §13).
- **One step instead of several.** \f$V(d)\f$ is a cubic, so ONE exact step from a
  deficit \f$e\f$ leaves about \f$e^2/3\f$ on a sphere: 8.7 % → 2.8e-3 → 2.6e-6 → 2.3e-12
  over three steps (`test_a_shrunken_sphere_is_restored_in_three_newton_steps`).

In a time loop with an ABSOLUTE target the per-step error is small (at most 6.7e-4
on LeVeque at refinement 4) and neither matters much. One Newton step gives the
same mesh as three to every digit of every metric. mftl's step leaves about 2e-6 of
the volume and the same shape within the spread over orientations (rms off the
sphere 0.545 against 0.547 % R0 at refinement 5). Neither residual accumulates,
because the next call corrects to the same target. For a solver the exact form is
still the right default:

- it is exact at any curvature and resolution, so a satellite one edge across is
  restored as well as a resolved drop;
- a large deficit -- a topology change, a restart, a phase-change source term --
  converges in three steps rather than slowly;
- it handles walls.

Measured in §15.

## What a uniform offset does to a shape

It changes a shape least for a given volume -- a sphere stays a sphere, and an
offset \f$d\f$ moves each point by \f$|d|\f$ -- but it is GLOBAL. It puts volume a
remesh lost at a few high-curvature collapses back over the whole surface, and on a
deforming flow that misplacement is carried along. Measured over the LeVeque period
(`docs/analysis.md` §15.1, refinement 5, four orientations):

| | V/V(0) − 1 | rms off the sphere at T | E_H at T | N_H at T |
|---|---|---|---|---|
| no projection, no correction | −1.17 % | 0.64 % R0 | 1.29 | 1.52 |
| no projection, correction | 0 | 0.55 % R0 | 1.36 | 1.50 |
| Phong projection, no correction | −0.18 % | 0.25 % R0 | 0.17 | 0.08 |
| Phong projection, correction | 0 | 0.28 % R0 | 0.35 | 0.21 |

Without projection the correction is a net gain: the position improves and the mean
bias goes from −0.39 to +0.005 % R0. With the projection, which already holds the
volume to 0.18 %, it doubles the curvature roughness at T. Applying it only before
or only after 3T/4 isolates the cause: the offsets made while the filament is
stretched. They are uniform in the stretched configuration, but the volume was
lost at collapses on the filament, and when the flow compresses the surface back a
normal displacement is scaled by the local stretch.

So: the right call where the loss is genuinely global (an advection scheme's
divergence error, a source term) or large (a topology change), and a trade where
the remesher already conserves well. The remedy for the trade is a LOCAL
correction -- the remesher recording a per-vertex volume defect, put back where it
arose. It is not implemented.

## The reference volume travels with the body

The target is library state, not a caller argument, because every holder of a
body needs it and a redistribution creates holders that never held the body.
#MFTL_Object::reference_volume (NaN = none):

- **Setting it:** MFTLEnsembleSetReferenceVolumes() measures it. An application may
  set it itself, to the same bits on every holder (a phase change, a source term).
- **Redistribution** carries it in the per-body header -- two more ints, beside
  the id counters -- to the ranks that newly hold the body, and asserts in Debug
  that the holders agree (`test_the_reference_volume_reaches_new_holders`, which
  counts the new holders).
- **A topology change** hands it on, because a merged body that kept id A must not
  be shrunk back to A's volume. Each body a cluster becomes gets the cluster's
  reference total \f$R_c\f$ in proportion to its volume, \f$V_b R_c / V_c\f$. That
  conserves the total through mergers, breakups and discarded satellites. A cluster
  in which any body has no reference hands on NaN. Tested:
  `test_a_coalescence_hands_the_sum_of_the_references_on`, to 1e-14, with
  references deliberately different from the volumes.

# In a time loop {#corr_loop}

```
advect owned vertices
MFTLUpdateGhostPositions
[MFTLEnsembleRemoveUndulations]      the undulations come from the advection; the
                                     remesh's projection then keeps the smoothing
geometry, MFTLRemeshEnsemble (or MFTLRedistribute), geometry
[MFTLEnsembleApplyTopologyChanges]   hands the references on
MFTLEnsembleCorrectVolumes           LAST: restores what the advection, the remesh
                                     and the reconstruction moved
```

**Once per step, last, is enough. A second call right after the advection changes
the shape at second order only**, because the correction is a uniform offset and
the errors it removes add at first order. Measured in §15: the same volume (exact
either way) and the same shape (rms off the sphere 0.547 ± 0.008 against
0.547 ± 0.005 % R0 at refinement 5, E_H 1.36 against 1.37), at 7.4 against 3.9 % of
the step. A loop that does not remesh every step still calls it every step: the
advection alone drifts. A solver that evaluates surface forces BETWEEN the
advection and the remesh, on the uncorrected surface, has its own reason for an
earlier call; the shape argument gives none.

**TSUR3D after the advection, before the remesh** is where the undulations it
removes come from, and the remesh's projection keeps the smoothing. Its cost at
refinement 5 is 14.7 % of the step on one rank and 4.4 % at np = 4 (every edge,
one sweep); with a 15° threshold, 9.0 and 2.6 %. The volume correction is 4.1 and
0.9 %.

# What the tests check {#corr_tests}

`tests/test_correction.c`, at np = 1, 2, 4, 8 and 27:

- **One isolated edge:** volume to 2e-16; both moves along one direction to 3e-16;
  the edge left perpendicular to it; neither end beyond its kink.
- **A jittered sphere:** the noise above the clean sphere's own discretisation
  error falls to 0.44× in one sweep and 0.23× in five, monotonically, at constant
  volume. A jittered periodic sheet falls to 0.19× in ten. A flat sheet is a fixed
  point to the bit.
- **The seam:** the same sphere centred on the corner of a periodic box ends within
  1.3e-15 of the unshifted one, with the same moves.
- **Fields:** a density's amount per triangle to 3e-16; a vertex field untouched.
- **Distributed:** the volume and the density amount to round-off. Ghost
  positions and ghost densities equal their owners' to the bit, and the partition
  stays sound. Non-vacuity: the coloured rounds must have moved edges and
  cross-rank edges must exist at np ≥ 2. The residual noise must be within 15 % of
  one rank's.
- **Volume:** a shrunken sphere is restored in 3 Newton steps at the quadratic rate.
  A sunk cap on the floor is restored to 1e-12 with every rim vertex still exactly
  on the wall.
- **References:** they reach new holders, and a coalescence hands on their sum.
- **Parameters:** out-of-range parameters are refused on every rank, and an
  ensemble with no body enters no unmatched collective.

`test_correction_mpi_is_confined` keeps the module's own MPI in
`correction_comm.c`.

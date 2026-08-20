Invariants worth preserving {#mftl_page_invariants}
===========================

[TOC]

Every entry below is a property that must survive any change, the reason it
exists, and the test that guards it. They share one characteristic: **breaking
any of them is silent.** The mesh stays structurally valid, the integrity checks
still pass, and the picture still renders — the numbers are just wrong.

---

# Geometry {#inv_geometry}

## A normal is always a unit vector, at every scale {#inv_unit_normal}

MFTLGetTriangleProperties(), `src/geometry/triangle_geometry.c`,
`src/ensemble/ensemble_update.c` and MFTLNormalizeVector() all compute

\f[
  m = \sqrt{\norm{\vect{v}}^2} \quad \text{(unclamped)}, \qquad
  \mathrm{inv} = \begin{cases} 1/m & m > 0 \\ 0 & \text{otherwise} \end{cases}
\f]

Two spellings are wrong and both were in the tree until 2026-08-15:

- **Clamping \f$\norm{\vect{v}}^2\f$ before the square root** floors the
  magnitude at \f$\sqrt{10^{-12}} = 10^{-6}\f$ — and, where an area is derived
  from the same \f$m\f$, floors that at \f$5\times10^{-7}\f$. So
  `quality.min_triangle_area` could never report a degenerate triangle, and
  \f$q = 4\sqrt{3}A/(a^2+b^2+c^2)\f$ became unbounded above; an *average* \f$q\f$
  of 1.023 was measured on a degenerate mesh.
- **Clamping the divisor** with `MFTL_MAX(mag, tiny)` keeps the instruction count
  but only *moves* the floor. At `tiny = MFTL_VERYVERYSMALL` a triangle with
  \f$10^{-8}\f$ edges still came out with a normal of length 0.112.

@warning A short normal is the most dangerous shape of wrong available here: it
         points the right way, so every sign test, winding check and picture
         still passes, while it silently scales surface tension and every other
         quantity built from it.

The test against zero has no cliff and costs one instruction (`fcmp`+`fcsel`
against `fmax`): **7.33 → 7.41 ns/tri, +1.1 %**. Pay it.

**Guarded by** `test_triangle_normals_are_unit_at_every_scale` (\f$10^{-8}\f$ …
\f$10^{8}\f$, all three kernels), `test_normalize_is_unit_across_magnitudes`
(\f$10^{-150}\f$ … \f$10^{150}\f$), `test_normals_of_a_whole_mesh_are_unit`,
`test_a_very_small_triangle_is_measured_exactly`, and
`inv_triangle_normals_are_unit` / `inv_vertex_normals_are_unit` in
`tests/test_mesh_invariants.h` — which are deliberately *not* part of
`inv_mesh_is_sound()`, because redistribution leaves `tris.normal` unspecified by
design.

The expression must stay identical in the walk and stream kernels, or
`test_triangle_geometry_paths_are_bit_identical` fails.

## Face normals point outward, everywhere {#inv_winding}

The icosahedron face table in `src/mesh_creation/sphere_creation.c` is wound
counter-clockwise as seen from outside, and the 1-to-4 subdivision emits every
sub-triangle in its parent's rotational order. This is what makes the enclosed
volume positive under the divergence theorem and fixes the sign of surface
tension and of the indicator function.

**Guarded by** `test_triangle_winding_is_outward`,
`test_enclosed_volume_is_positive_and_converges`.

## Never compare two identical vectors with `acos(a·b)` {#inv_no_acos}

The arc cosine has infinite derivative at 1, so a dot product one ulp below 1
reports \f$1.5\times10^{-8}\f$ rad — a noise floor of \f$\sim10^{-6}\f$ degrees
regardless of how exact the inputs are. Use the chord \f$\norm{\vect{a} -
\vect{b}}\f$.

`tests/test_vertex_normals.c` records the false failure this produced.

---

# Topology {#inv_topology}

## "No twin" and "mesh boundary" are different statements {#inv_boundary}

Only one of them is partition-independent.

| Statement | Meaning | Partition-dependent? |
|---|---|---|
| `twin_idx == MFTL_ERROR` | no partner in **this** partition | yes — true at an open surface's rim *and* at a ghost halo's outer edge |
| `flags & MFTL_FLAG_BOUNDARY` | no partner anywhere in the mesh | no — a property of the surface |

#MFTL_FLAG_BOUNDARY is written once by MFTLMarkMeshBoundary() while the mesh is
still complete, and from then on travels with the entity. Re-deriving it from
`twin_idx` after redistribution would promote every halo rim into a fake
boundary.

## A boundary vertex has one more neighbour than incident triangles {#inv_open_fan}

The closing edge of an open fan is reachable only as an *incoming* half-edge.
MFTLGetVertexOneRingVertices() handles it; MFTLGetVertexOneRingHalfedges()
deliberately does not, because its contract is *outgoing* half-edges.

**Guarded by** `test_one_ring_at_a_boundary_vertex_has_one_extra_neighbour`.

## A half-edge's global ID is `3 * triangle_gid + k`, mesh-wide {#inv_he_gid}

Established by MFTLBuildTopologyFromIndexList(), preserved by
MFTLObjectReorderMorton() (which moves half-edge triples as a unit) and by
redistribution. It is the wire-format half of implicit indexing: half-edge global
IDs are **never transmitted**, the receiver reconstructs them.

**Guarded by** `inv_halfedge_gids_are_implicit`.

## Global IDs are a dense permutation of [0, n) per entity class {#inv_dense_gid}

And `next_*_gid == n_*` after generation. Subdivision retires the whole previous
generation of triangles and half-edges, so those counters restart at zero;
`next_vertex_gid` does not, because old vertices are carried over and keep their
IDs.

Density is what lets a receiver size a lookup table by ID range instead of
hashing — which is why remeshing must renumber at the end of every pass.

**Guarded by** `inv_global_ids_are_dense`.

---

# I/O {#inv_io}

## Only OWNED entities go into a `.vtu` {#inv_owned_only}

A ghost is a copy of a triangle another rank owns, so writing the whole pool —
which MFTL 5 did — puts every halo triangle in the merged dataset once per rank
holding it. The picture is unchanged (the copies are coincident) and **every
integral the visualisation tool computes over the surface is wrong by the halo
fraction, silently.**

`include_ghosts` restores the old behaviour for looking *at* the halo.

**Guarded by** `test_ghost_inclusion`, `test_cell_count_is_the_owned_count`.

## The periodic-wrap test has no tolerance, on purpose {#inv_wrap}

It asks whether a reconstructed corner \f$\vect{p}_0 + \vect{e}\f$ left the
domain box. A false positive costs three extra points carrying correct
coordinates and renders identically; a false negative draws a triangle across the
whole domain. **Any slack added to suppress the first buys the second**, so there
is none.

**Guarded by** `test_wrap_detection_paths_agree`.

## MFTLWriteVtu() must never write `tris.area` or `tris.normal` {#inv_no_stale_fields}

Nothing on the object records whether they are current — `derived` tracks the
topology and the frames, not the geometry — and a redistribution leaves them
undefined by design, so a production file containing them would look right and be
stale.

MFTLWriteVtk() writes them deliberately, because the raw dump's contract is to
show what is actually in memory.

---

# Communication {#inv_comm}

## An extent is a centre plus a half-width, never a (lo, hi) box {#inv_extent}

In `src/communication/spatial_halo.c`, for what a rank publishes about its share
and for each triangle's own contribution. "Near" is the minimum-image distance
between centres against the sum of half-widths.

A box gets it wrong in **both directions at once**:

- It **misses.** Two shares meeting across the seam are a domain length apart in
  stored coordinates, so min/max says they do not touch — and because the
  classification decides what is *sent*, the miss lands one stage before anything
  the periodic image loop can see, so nobody reports the contact.
- It **over-sends.** A share holding one seam-straddling triangle has a box
  spanning the whole domain, because MFTLGetTriangleBounds() is correctly
  seam-consistent and puts that triangle's corners just outside the box on one
  side.

**Measured** at np = 27 on a seam contact: 314 triangles ghosted with boxes
against 56 with intervals, and 320 against 0 for a placement with no cross-rank
contact at all. Payload is six doubles per body per rank either way.

A rank's own extent is measured in two frames, raw and shifted by half a domain,
keeping the tighter per axis.

**Guarded by** `test_contact_through_the_seam_survives_the_decomposition`,
`test_a_seam_straddling_body_is_not_measured_as_domain_sized`.

## A pre-filter must be strictly MORE permissive than the test it short-cuts {#inv_prefilter}

The halo's per-triangle intervals are widened by the contact margin, so an
un-widened macro extent is *stricter* than what it stands in for and rejects the
whole neighbour in exactly the regime the margin exists for.

Overlapping bodies hide it completely — their raw extents intersect anyway — so
only a **proximity** contact at a rank boundary reveals it.

**Guarded by** `test_the_tolerance_threshold_is_exact_from_both_sides`.

## A spatial ghost is a triangle in a SHADOW OBJECT {#inv_shadow}

MFTLExtendSpatialHalo() puts what the neighbours sent into
`MFTL_SpatialHalo.patch[b]` — an ordinary, self-consistent #MFTL_Object with
`n_local == 0`, twins linked within the patch, derived arrays built. It is
deliberately **not** in `ens->objects`, so redistribution and both ghost syncs
never see it.

Appending them to the body's own ghost suffix instead — which is what MFTL 2 did,
and which looks free because the ownership prefix is untouched — fails for two
reasons that are invisible until they bite:

1. **Four passes bounded by `tris.n` rather than `n_local` change silently:** the
   Morton bounding box in `reorder_vertices` (a partner body across the domain
   coarsens the cells for the *owned* mesh), MFTLMeanEdgeLength() (each appended
   triangle contributes three twin-less half-edges, so the remeshing target
   length is set partly by another body), the vertex-normal scatter, and
   MFTLRebuildTwins() (which re-pairs by vertex **slot** and can weld two bodies
   together).
2. **There is no good answer for the arriving vertex GIDs.** De-duplicate them
   onto existing slots and `inv_halo_is_vertex_complete` breaks; give them fresh
   slots and `inv_local_global_ids_are_unique` breaks.

Doing it properly needs the two-way ownership split to become three-way, i.e. a
change to @ref arch_inv_prefix.

**Guarded by** `test_halo_does_not_touch_the_bodies`,
`test_halo_patch_is_a_valid_object`.

---

# Remeshing {#inv_remesh}

## MFTL_FLAG_FROZEN means REPLICATED, not "near a cut plane" {#inv_frozen}

An owned entity is frozen exactly when some other rank holds a copy of it, and
MFTLRemeshMarkFrozen() derives that from the rule redistribution **sent** by:

```
replicated  ⟺  corner >= verts.n_local  ||  neighbour >= tris.n_local
```

Two streaming reads, six integer compares, no distance test and **no message**.
Three things follow, and all three are load-bearing:

- A vertex is frozen if **any** incident triangle is, so an unfrozen vertex has
  every incident triangle owned and unreplicated — which is why gating a
  **collapse**, whose modified set is an entire fan, is two flag tests.
- Combined with the vertex-complete halo, every fan an approval predicate walks
  is complete, so the link condition, the wing valences and the fold-over test
  give the same answers they would on the whole mesh.
- **Every ghost in a rank's pool is frozen on its owner.** It is here because
  this rank owns a corner or an edge neighbour of it, which from the owner's side
  *is* the definition of replicated. So ghost copies are still exact when the
  sweeps finish, and a phase needs **no ghost position sync** before the next
  migration.

**Guarded by** `test_every_ghost_agrees_with_its_owner` and
`frozen_matches_replication` — the "replicated but not frozen" direction is the
fatal one, and its symptoms appear one or two migrations after the cause.

## A shifted cut plane needs a FLOOR, not a truncation {#inv_shifted_floor}

MFTLRemeshOwnerProc() biases by `cuts` before the `(int)` cast on a periodic
axis. Without the shift the argument is non-negative for any wrapped position, so
`(int)` **is** `floor`; with a shift of up to \f$h/2\f$ it reaches \f$-0.5\f$,
where `(int)` truncates *towards zero* and attributes the sliver
\f$[\text{min}, \text{min} + \text{shift})\f$ to cell 0 instead of to cell
\f$\text{cuts}-1\f$ across the seam.

@warning What makes that hard to catch is that **the wrong rule is still a
         rule.** Every rank computes the same answer, the partition stays
         consistent, and every structural invariant still holds. The one thing
         that changes is that **the seam plane never moves**, so an entity on it
         is frozen in every phase of the schedule and ends up in a permanently
         un-remeshed band.

A body away from the seam never enters the sliver, so both original spanning
tests passed under the broken rule. Found by mutation testing.

**Guarded by** `test_a_sphere_on_the_periodic_seam_remeshes_everywhere`, which
fails on convergence *and* reports the sphere as an open surface with 78 boundary
half-edges.

The same reasoning applies to MFTLRemeshMeasureBodies() one level up: it averages
a periodic coordinate as an **angle** (circular mean, `atan2` of the
area-weighted mean direction), because the arithmetic mean of a seam-straddling
body's coordinates lands in the middle of the domain and its radius comes out as
half the box. Nothing fails — the body is simply classified SPANNING forever.
**Guarded by** `test_a_seam_straddling_droplet_is_not_measured_as_domain_sized`.

## Remeshing reaches the network through its own file {#inv_mpi_confined}

Every MPI call the remesher makes is in `src/remeshing/remesh_comm.c`.
`test_remeshing_mpi_is_confined` — a CMake source check, not a runtime one —
fails the build if that stops being true, because no runtime test can see it.

The rule exists because the first version broke it in two places that were
individually small, correct and fully tested, and collectively the wrong shape: a
feature used a few times per hundred timesteps paid for once per timestep, in the
two loops the library is most careful about.

MFTLGetOwnerProc() gained a per-axis cut-plane offset and a periodicity branch.
That function **is** the body of `compute_owners`, i.e. it runs once per vertex
and once per triangle on every redistributed timestep. **Measured at 1.70× —
1.79 → 3.05 ns/entity, 42 → 89 instructions for the same loop.**

What replaces both is #MFTL_OwnerMap — one seam, a NULL pointer on the production
path, and one predicted branch per body hoisted out of both loops.

## A rank must not judge work it only holds as a GHOST {#inv_no_ghost_judgement}

The first parallel convergence certificate counted "approved, then refused by the
freeze gate". It reported 3 permanently blocked operations on an np = 4 sphere
that was in fact fully converged: a rank examines every half-edge in its pool,
ghosts included, and around a ghost vertex its one-ring is incomplete, so it
approves a collapse the owner correctly rejects.

The gate therefore runs **before** the approval (it is cheaper anyway), and the
certificate is `count_hidden_edges` — out-of-band edges with a frozen endpoint,
which needs no fan walk and so cannot be given a wrong answer by a partial view.

## Every clause of an approval predicate is silent when removed {#inv_approval}

MFTLCollapseApproved() refuses on five grounds and none of them announce
themselves:

| Clause | What its absence produces |
|---|---|
| link condition | welds two sheets into a non-manifold vertex whose one-ring walk then silently visits half the fan |
| fold-over | an inverted triangle |
| stretch bound | split and collapse oscillate forever |
| **wing-valence floor** | a tetrahedron passes the neighbour count and still dissolves |
| **feature/boundary link condition** | two vertices on a feature curve joined by an edge *not* on that curve are a chord across the surface; collapsing it pinches the curve inwards |

The last two are easy to omit because the obvious form of the link condition does
not imply them. On an open patch the feature clause's absence serrates the rim;
on a cube it merges two sharp edges into the middle of the face between them.

**Guarded by** `test_collapse_refuses_to_dissolve_a_tetrahedron`,
`test_remesh_of_an_open_patch_keeps_its_rim`.

## Feature classification must describe the mesh being SMOOTHED {#inv_features_twice}

Not the one the sweeps started from. MFTLRemeshObject() calls MFTLDetectFeatures()
**twice per iteration** for this reason.

A split of a rim edge creates a midpoint that *is* on the rim, and a vertex the
smoother believes to be free gets the 2D tangent-plane treatment and slides
straight off it — after which the surface has a serrated notch that no structural
check can see, because the result is a perfectly valid manifold with a perfectly
valid boundary.

## Smoothing is a PROJECTION {#inv_smoothing_projection}

The feature class chooses the subspace: 2D tangent plane, 1D ridge tangent, 0D
locked. Stated that way the three cases are one idea.

MFTL 5 moved a ridge vertex **to** the midpoint of its two feature neighbours —
the full displacement, no projection — which is the 2D shrinkage problem
reintroduced on a 1D curve: a regular 24-gon ridge collapses to 0.84 of its
radius in 20 sweeps.

**Guarded by** `test_a_regular_ridge_is_a_fixed_point`, which asserts drift below
\f$10^{-12}\f$.

## Two feature edges make a ridge point only if the curve runs STRAIGHT {#inv_corner_angle}

Counting alone makes the four corners of an open patch ridge points, free to
slide along the diagonal between their neighbours — off the patch.
`feature_corner_angle` is a **separate** threshold from
`feature_angle_threshold` because it applies where that one does not: a mesh
boundary is a feature curve at any dihedral threshold, including a disabled one.

---

# Contact detection {#inv_bvh}

## Periodicity is two problems, and they need two different fixes {#inv_bvh_periodic}

- **Within an object**, a triangle straddling a seam has wrapped vertices, so a
  centroid built from raw coordinates lands in the middle of the domain and its
  AABB spans the box. Fixed by building every primitive from the seam-consistent
  `tris.anchor` / `tris.edge` frame.
- **Between objects**, two bodies on opposite faces are neighbours and no
  per-triangle arithmetic can see it. Fixed by traversing the pair once per
  periodic **image** (up to 27, each rejected by one root-AABB test).

A straddling object's **root** box legitimately spans the domain and is not a
bug — same accepted cost, for the same reason, as MFTLObjectReorderMorton().

**Guarded by** `test_periodic_morton_of_a_straddling_object`,
`test_contact_across_a_periodic_face`.

## The contact envelope scales with the body, not the process count {#inv_envelope}

\f[
  \tau = \max\big(\texttt{proximity\_floor},\ \texttt{proximity\_factor} \cdot R\big)
\f]

replaces MFTL 2's hardcoded \f$10^{-4}\f$ (AABB inflation) and \f$10^{-6}\f$
(proximity), both absolute and therefore correct at exactly one length scale.

\f$R\f$ is half the **longest side** of the root box, not half its diagonal, so
it equals the radius for a sphere and `factor` means what it says. It is measured
over one rank's partition, so the distributed pipeline overwrites it with the
body's **global** radius from one `MPI_Allreduce` — otherwise splitting a droplet
over eight ranks would change the separation at which contact is detected.

**Guarded by** `test_detection_at_two_disparate_length_scales`,
`test_tolerance_is_independent_of_the_partition`.

## A distributed contact is reported by the A-side owner, and only that rank {#inv_a_side}

So a BVH is built over **owned** triangles only (including the redistribution
halo makes two ranks report the same contact, exactly as writing ghosts into a
`.vtu` double-counts the surface), only object B's triangles travel, and the
traversal skips any leaf pair whose A side is remote.

Get it wrong in one direction and cross-rank contacts are missed; wrong in the
other and every cross-rank contact force is applied twice.
`test_distributed_contacts_match_the_serial_answer` compares the gathered **set**
against the answer computed on one rank while it still held the whole mesh, which
is the only check that rules out both at once.

## MFTL_Orient3D is adaptive; nothing else in src/bvh/ is {#inv_adaptive}

It is the only predicate whose **sign** decides the answer — Guigue–Devillers
branches on it up to twelve times per pair, and a wrong sign gives a clean "no
intersection" for two triangles that cross. Everything else there degrades
continuously.

`examples/orient3d_predicate_profile.c` measures the trade both ways and the
answer is not one-sided: the plain determinant did not misclassify once in 20 000
samples at gaps down to \f$10^{-12}\f$ of the body size (three orders below where
a lubrication model works), and only fails at \f$10^{-14}\ldots10^{-16}\f$. It is
kept because the cost is ≤ 6.2 % of a full MFTL_TriTriProximity() call (2.00× on
the predicate, but the narrow phase is dominated by its fifteen distance
sub-tests) and the failure it removes is silent.

---

# How the tests express these {#inv_test_helpers}

`tests/test_mesh_invariants.h` holds the structural checks a mesh must satisfy —
implicit indexing, twin symmetry, `out_he` really outgoing, derived arrays
agreeing with the records, ownership prefix, dense global IDs.

Use `inv_mesh_is_sound()` after any operation rather than re-deriving them, and
`inv_partition_is_sound()` after anything distributed. The latter adds the three
statements that only mean something once a mesh is spread over ranks:

1. local global-ID uniqueness,
2. a **vertex-complete halo** (`inv_halo_is_vertex_complete`), which is exactly
   what MFTLComputeVertexNormals() needs to be communication-free,
3. every owned triangle having all three edge neighbours.

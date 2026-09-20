# 13 — Periodicity completion end-to-end

Status: NOT STARTED
Depends on: 02, 06, 07, 10, 11
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Ensure Cartesian periodicity (X/Y/Z, any combination) is correct across **every** subsystem —
integrals, curvature, point queries, remeshing, and intersection — not just the primitives.
Resolve the outstanding periodic-correctness question in the BVH and add end-to-end periodic
validation.

## Relevant existing code
- MFTL2 primitives (correct, tested): `MFTLGetMinimumImageVector`, `MFTLApplyPeriodicity`,
  `MFTL_Domain.is_periodic[3]`/`domain_half_size[3]`; `MPI_Cart_create` with `periods[]`
  (`src/general/ensemble.c`); `test_internals.c` (`test_apply_periodicity`, `test_min_image_vector`,
  `test_edge_length`). Open item: `src/general/mftl_bvh.c:535`
  "ATTENTION: IS THIS CORRECT FOR PERIODIC DOMAINS?".
- Consumers wired in earlier stages: 02 (integrals unwrap per triangle), 06 (curvature one-ring),
  07 (point-in-object nearest image), 08–10 (remesh edge length / midpoint via min-image), 11
  (overlap min-image shift of B into A).

## Proposed changes
- **Audit + wire**: verify each stage's periodic path is exercised, and fill any gaps:
  - Integrals (02): per-triangle unwrap relative to `v₁`.
  - Curvature (06): one-ring computed on min-image-unwrapped neighbours across seams.
  - Point-in-object (07): ray-cast against the nearest periodic image of the object; periodic
    ray re-entry or image selection.
  - Remeshing (08–10): edge length/midpoint/fit use `MFTLGetMinimumImageVector`; new/moved
    vertices optionally `MFTLApplyPeriodicity`-wrapped; disable any wall-clamp on periodic axes.
  - Intersection (11): min-image shift of B into A's frame; broad phase tests periodic images.
- **BVH periodic correctness** (`mftl_bvh.c:535`): make LBVH node bounds / traversal periodic-aware
  (either build per-image query boxes or min-image the query), and replace the flagged assumption
  with verified logic + a periodic intersection test.
- **Redistribution/ghosting across seams**: confirm `MFTLGetOwnerProc` wraps positions on
  periodic axes (via `MPI_Cart` periodic neighbours) and that ghost layers form across seams
  with correctly shifted coordinates (unwrapped local frame — coordinate with the storage
  convention: MFTL2 keeps wrapped positions + min-image geometry, so ghosts across a seam are
  handled by min-image at use sites rather than by shifting stored coordinates; verify this holds
  for curvature/remesh stencils).

## Files/functions likely to change
`src/general/mftl_bvh.c`, `src/general/ensemble.c`/`communication.c` (owner wrap + periodic ghost
formation), plus periodic branches in the Stage 02/06/07/08–11 units. Test
`tests/test_periodic_endtoend.c`.

## Numerical/physical considerations
- **Minimum-image constraint**: period on each periodic axis must exceed twice the largest
  feature/edge/ghost-neighbourhood extent, so "nearest image" is unambiguous. Add a runtime
  assertion in periodic setup and document it (LeVeque/advection cases satisfy it).
- Conserved quantities (volume, mass) must be identical whether an object sits mid-domain or
  straddles a seam.

## MPI/PETSc considerations
- `MPI_Cart` periodic neighbours already provide the wrap for redistribution; ensure ownership
  lookup and ghost exchange use them on periodic axes only.
- Rank-independence must hold for periodic runs too (1/2/4/8), including the seam-crossing case.
- Downstream (Stage 15): multiflow front-tracking is currently non-periodic; enabling MFTL2
  periodicity must be opt-in per object and must not change non-periodic behavior.

## Possible failure modes
- A subsystem still reading raw coordinate differences across a seam → wrong geometry only when
  straddling; the straddling tests target each subsystem.
- Period too small vs feature size → ambiguous min-image; runtime assert.
- BVH broad-phase missing a cross-seam pair → periodic intersection test catches it.
- Ghosts across a seam with unshifted coordinates breaking a curvature/remesh stencil → verify
  the min-image-at-use-site convention covers 2-ring stencils.

## Unit/regression tests to add (`tests/test_periodic_endtoend.c`)
- **Seam invariance battery**: for X, Y, Z, XY, XYZ periodicity, place a sphere straddling the
  seam and assert volume/area/centroid(mod period)/inertia/curvature equal the mid-domain values
  (ties together Stages 02/06).
- **Periodic advection**: advect a sphere a full period along X (and diagonally) and back;
  volume and triangle count conserved; no "outside domain" errors; rank-independent on 1/2/4/8.
- **Periodic LeVeque**: single-vortex deformation with the drop crossing a periodic face;
  enclosed-volume drift within tolerance; returns toward initial shape at t=T.
- **Periodic intersection**: two objects overlapping only across a seam detected and quantified
  (ties to Stage 11); BVH `:535` case verified.
- Minimum-image assertion fires when period < 2×feature.

## Acceptance criteria
- All seam-straddling integrals/curvature/queries equal mid-domain values within tolerance, for
  every axis combination.
- Periodic advection/LeVeque conserve volume and are rank-independent; periodic intersection
  correct; BVH periodic assumption resolved.
- Runtime minimum-image constraint asserted.

# 04 — Object generators (cylinder, box, cube, face) to sphere parity

Status: NOT STARTED
Depends on: 02
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Bring MFTL2's built-in geometry generators to MFTL1 parity. MFTL2 today has only
`MFTLGenerateSphere`; add cylinder, box, cube, and single-face generators producing valid
half-edge meshes.

## Relevant existing code
- MFTL1: `src/generalfunctions/generateCylinder.c` (`MFTLGenerateCylinder(Object,dr,radius,L,
  ellipticity)`), `box.c` (`MFTLGenerateBox`), `sphere.c` (`MFTLGenerateCube`),
  `face.c` (`MFTLGenerateFace`), `CylinderModifier.c`.
- MFTL2: `src/general/sphere.c` (`MFTLGenerateSphere`, icosphere refinement; already calls
  `MFTLApplyPeriodicity` and builds half-edge connectivity via `MFTLRebuildTwins`/
  `MFTLFlattenConnectivity`) — the reference pattern to mirror.

## Proposed changes
Add generators following `sphere.c`'s lifecycle (fill `vertices`/triangle-vertex lists →
build half-edge topology → `MFTLComputeTriangleProperties` → integrity check):
- `MFTLGenerateCylinder(MFTL_Object*, double dr, double radius, double L, double ellipticity)`
  — capped cylinder; `dr` sets circumferential/longitudinal resolution.
- `MFTLGenerateBox(MFTL_Object*, const double lo[3], const double hi[3])` and
  `MFTLGenerateCube(MFTL_Object*, int refine)` — closed axis-aligned box, optionally refined.
- `MFTLGenerateFace(MFTL_Object*, const double v[4][3])` — a single quad→2-triangle open patch
  (note: open surface; volume undefined — flag as non-closed).
Prefer one new file per shape (`src/general/cylinder.c`, `box.c`, `face.c`) or a combined
`generators.c`; decls in a `include/mftl_geometry.h` "generators" section.

## Files/functions likely to change
New generator source file(s) under `src/general/`; `include/mftl_geometry.h`; test
`tests/test_generators.c` (+ `tests/CMakeLists.txt`).

## Numerical/physical considerations
- Closed generators (cylinder/box/cube) must satisfy Euler χ=2 and yield the analytic volume
  (cylinder πr²L + cap handling per MFTL1's convention; box ∏(hi−lo)).
- Consistent outward winding so `MFTLObjectVolume` is positive.
- `MFTLGenerateFace` is an **open** surface (χ≠2, no enclosed volume) — its tests check area
  and triangle count only.
- Reproduce MFTL1's cylinder parameterization (radius, length, ellipticity, `dr`) so meshes
  are comparable.

## MPI/PETSc considerations
Match `MFTLGenerateSphere`'s collective/serial semantics (it is collective, distributing the
generated mesh across the ensemble). Generators must produce identical global meshes
regardless of rank count. No PETSc.

## Possible failure modes
- Cap/seam vertices duplicated → non-manifold; ensure shared vertices are merged by id before
  `MFTLRebuildTwins`.
- Inconsistent winding on caps → negative/partial volume.
- Off-by-one in circumferential closure (last segment not joined to first).

## Unit/regression tests to add (`tests/test_generators.c`)
- Cube/box: closedness (χ=2), `MFTLCheckMeshIntegrity` pass, volume = analytic (tol 1e-12).
- Cylinder: volume within discretization tolerance of πr²L (+caps); refine → converges.
- Face: expected triangle/vertex counts and total area; integrity of an open patch.
- Rank-independence: identical global counts and volume on 1/2/4/8.
- Regression: cylinder volume vs MFTL1 `testCylinder.c`.

## Acceptance criteria
- Each closed generator passes integrity + Euler check and hits its analytic volume within
  tolerance; face generator yields the expected open-patch area/counts.
- Identical global mesh across rank counts.
- Matches MFTL1 cylinder volume within tolerance.

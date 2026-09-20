# 03 — Rigid-body transforms (scale, translate, rotate) + optional periodic wrap

Status: NOT STARTED
Depends on: 02
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Provide in-place object transforms — uniform/per-axis scale, translation, and rotation about a
centre — matching MFTL1's `MFTLScaleSurface`/`MFTLMoveBody`/`MFTLRotateBody`, plus an optional
periodic wrap so a translated body can re-enter the domain. Required by moveIBM and by the
multiflow transform wrapper.

## Relevant existing code
- MFTL1: `src/geometricfunctions/parallel/moveobject.c` (`MFTLMoveBody`, `MFTL_MoveVertex`,
  `MFTLRotateBody`, `MFTL_RotateVertex`), `MFTLScaleSurface`.
- multiflow caller: `src/general/mftlwrapper/transformMFTLObject.c` (Scale→Translate→Rotate,
  Euler→axis-angle), `moveibm/updatemoveibmftlbodies.c` (per-step move/rotate).
- MFTL2: `MFTL_Vertex.pos`, `MFTLApplyPeriodicity` (`mftl_geometry.h`),
  `MFTLComputeTriangleProperties` (recompute centres/normals/areas after motion).

## Proposed changes
New unit `src/general/transform.c` + decls in `include/mftl_geometry.h`:
- `MFTLScaleObject(MFTL_Object*, const double s[3])` — `pos[d]*=s[d]` for all local vertices.
- `MFTLTranslateObject(MFTL_Object*, const double d[3])` — `pos += d`.
- `MFTLRotateObject(MFTL_Object*, const double R[3][3], const double centre[3])` —
  `pos = centre + R·(pos−centre)`.
- All operate on **local** vertices only; ghosts refreshed via existing
  `MFTLUpdateGhostPositions`. After any transform, call `MFTLComputeTriangleProperties`.
- Optional `MFTLWrapObjectIntoDomain(MFTL_Object*)` applying `MFTLApplyPeriodicity` per vertex
  on periodic axes (off by default; unwrapped storage otherwise — see Stage 13).

## Files/functions likely to change
New `src/general/transform.c`; `include/mftl_geometry.h`; test `tests/test_transforms.c`.

## Numerical/physical considerations
- Rigid motion (rotate/translate) must leave **volume, area, and inertia eigenvalues**
  invariant; scale by `s` (isotropic) scales volume by `s³`, area by `s²`, inertia by `s⁵`.
- Rotation matrix orthonormality: assert `RᵀR≈I`, `det R≈1`; accept caller-supplied R (the
  Euler→axis-angle conversion stays in multiflow's wrapper).
- With unwrapped storage (Stage 13), do **not** wrap after every move; provide wrap only as an
  explicit call to bound coordinate magnitude for long-circulating bodies.

## MPI/PETSc considerations
Transforms are embarrassingly parallel over local vertices; no communication except the ghost
refresh afterwards. Rank-independence follows trivially. No PETSc.

## Possible failure modes
- Forgetting to recompute triangle properties → stale centres/normals break downstream owner
  assignment and integrals.
- Rotating ghosts as well as locals → double transform. Operate on locals; refresh ghosts.
- Non-orthonormal R → shape distortion; guard/assert.

## Unit/regression tests to add (`tests/test_transforms.c`)
- Volume invariant under a random rotation and a translation (tol 1e-9); area invariant.
- Isotropic scale by 2 → volume×8, area×4.
- **Round-trip**: rotate by θ then −θ about the same centre → vertices return to within 1e-12.
- **Rank-independence**: same post-transform volume/centroid on 1/2/4/8 ranks.
- Regression: reproduce MFTL1 `test1.c` (sphere→ellipsoid 2×1×1→rotate 30°z→move) volume/centre
  sequence and `testCylinder.c` move sequence.

## Acceptance criteria
- Rigid-motion invariants hold to 1e-9; scale laws exact to round-off.
- Round-trip error < 1e-12.
- Matches MFTL1 test1/testCylinder printed volume/centre values within tolerance.

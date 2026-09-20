# 02 — Object integral quantities (volume, centroid, area, moment of inertia)

Status: COMPLETE
Depends on: 01
Implemented by: Kimi Code
Reviewed by: Claude Code
Acceptance: PASSED

## Objective
Provide parallel, periodic-aware object-level integrals — enclosed volume, surface area,
area-weighted surface centroid, and moment-of-inertia tensor — matching MFTL1's definitions.
These are the ground truth for every downstream conservation test.

## Relevant existing code
- MFTL1: `src/geometricfunctions/parallel/objectvolume.c` (`VolumeFace`, `MFTLObjectVolume`),
  `objectcentre.c` (`CenterOfAreaFace`, `MFTLSurfaceCenter`),
  `momentofinertia.c` (`MFTLMomentOfInertia`).
- MFTL2: `MFTLGetTriangleArea`/`MFTLGetTriangleCenter`/`MFTLComputeTriangleProperties`
  (`include/mftl_geometry.h`, `src/general/…`); `MFTL_Triangle.center/area/normal`;
  owner guard `owner_proc`, ensemble comm `MFTL_Domain.comm_cart`; periodic primitives
  `MFTLGetMinimumImageVector` (`mftl_geometry.h`).

## Proposed changes
New unit `src/general/geometry_object.c` + decls in `include/mftl_geometry.h`:
- `double MFTLObjectVolume(MFTL_Object*)` — `⅙ Σ_owned v₁·(v₂×v₃)`, then `MPI_Allreduce(SUM)`
  over `obj->domain->comm_cart`.
- `double MFTLSurfaceCenter(MFTL_Object*, double cm[3])` — accumulate `Area·(v₁+v₂+v₃)` and
  `Area`; two `Allreduce`s; `cm/=3·area`; return total area.
- `void MFTLMomentOfInertia(MFTL_Object*, double I[3][3], const double center[3], double density)`
  — lumped-tetra sum about `center` (a=v₂−v₁, b=v₃−v₁, c=center−v₁, dV=|det|/6, cm=(v₁+v₂+v₃+
  center)/4, r=cm−center); three `Allreduce`s (one per row).
- **Periodic-aware form**: for each owned triangle, unwrap its 3 vertices into one contiguous
  image relative to `v₁` via `MFTLGetMinimumImageVector` before the triple product / tetra so
  a seam-straddling triangle integrates correctly; for whole-object integrals the per-triangle
  contribution is already local, so triangle-local unwrapping suffices (a single triangle is
  << half the domain).

## Files/functions likely to change
New `src/general/geometry_object.c`; `include/mftl_geometry.h`; `CMakeLists.txt` glob already
covers `src/*.c`. New test `tests/test_object_integrals.c` + `tests/CMakeLists.txt`.

## Numerical/physical considerations
- Volume needs a **closed, consistently-oriented** mesh; assert `MFTLCheckMeshIntegrity`
  first. Sign depends on triangle winding/normal consistency.
- Inertia is the MFTL1 **lumped** approximation (mass at each tetra centroid) about the volume
  centroid — reproduce it for comparability; add a cross-check against the analytic sphere
  tensor `2/5·M·R²`.
- Centroid is the **area-weighted surface** centroid (not the volume centroid); document the
  distinction where callers (moveIBM) expect a specific one — moveIBM uses `MFTLSurfaceCenter`
  then `MFTLObjectVolume` for mass and `MFTLMomentOfInertia` about that center.

## MPI/PETSc considerations
Owner guard `owner_proc==rank` prevents double-counting ghosts; reduce over `comm_cart` (per
object). Result must be bit-stable across rank counts up to reduction order (use tolerance,
not exact equality, for the rank-independence assertion). No PETSc.

## Possible failure modes
- Counting ghost triangles → double count. Guard strictly on `owner_proc`.
- Non-closed / inconsistent winding → wrong sign/'0' volume. Integrity precheck.
- Seam-straddling object without triangle-local unwrap → wildly wrong volume. Covered by the
  straddling test.
- Reduction-order float drift across ranks → keep tolerance realistic (~1e-9·|V|).

## Unit/regression tests to add (`tests/test_object_integrals.c`)
- Sphere r=1 refinement 4/5: volume→4/3π within tolerance decreasing with refinement;
  area→4π; centroid at origin (symmetry).
- Inertia of unit sphere → `2/5·M·R²` diagonal, off-diagonals ≈0 (limiting case).
- **Rank-independence**: identical volume/centroid/inertia on 1/2/4/8 ranks (tol 1e-9).
- **Periodic seam invariance**: translate the sphere so it straddles the X (then Y, Z, XY,
  XYZ) periodic face; volume/area/inertia unchanged; centroid equals the mid-domain centroid
  modulo period.
- **Regression vs MFTL1**: read `mftl/examples/test1.gts` (after Stage 05), compare volume to
  MFTL1's printed value from `test1.c`.

## Acceptance criteria
- All tests green on 1/2/4/8 ranks under ASan.
- Sphere volume/area/inertia match analytic values within refinement-appropriate tolerance.
- Seam-straddling integrals invariant to within 1e-9 of the mid-domain values.
- No double-counting: total triangle count used equals `total_global_triangles`.

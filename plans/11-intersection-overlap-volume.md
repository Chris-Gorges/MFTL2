# 11 — Object–object intersection + overlap-volume quantification

Status: NOT STARTED
Depends on: 02
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Add MFTL1's overlap-**volume** quantification (for collision response) on top of MFTL2's
existing BVH intersection *detection*, including the projection/tessellation method that yields
overlap volume, centre-of-volume, effective radius of curvature, and average surface normal —
plus periodic (cross-seam) support and a planar convex-hull helper.

## Relevant existing code
- MFTL1: `src/intersection/intersection.c` (Möller SAT `triBoxOverlap`, `BoxIntesectsTriangle`,
  `MFTL_BodyOverlap`/`_OneSide` — cross-proc triangle exchange, builds `AinB`/`BinA`);
  `overlap.c` (`MFTL_CalculateOverlap` → `struct MFTL_Overlap {Volume, AreaTriA/B,
  AverageSurfaceNorm, RealCentreOfVolume, EffectiveRadiusOfCurvature, ID_A, ID_B}`;
  `GS_OrthoNormalise`, `GetProjectedTriangle`, `AddProjectedGrad/Curv`, `TessilatedOverlap`,
  `OverlapFromVertices` prism volume `⅓·area·Σ(hB−hA)`, `FinaliseOverlap`); `MFTL_REL_TOLERANCE
  5e-7`. Convex hull: `src/geometricfunctions/serial/graham.c`.
- MFTL2: `src/general/mftl_bvh.c` (`MFTL_GenerateLBVH`, `MFTL_IntersectObjects` with callback,
  `MFTL_TriTriIntersect`, `MFTL_CreateSpatialGhosts`); note author flags at `mftl_bvh.c:190`
  ("brute force, needs optimization") and `:535` ("IS THIS CORRECT FOR PERIODIC DOMAINS?").

## Proposed changes
New unit `src/general/overlap.c` (+ `src/general/convex_hull.c`; decls in `include/mftl_bvh.h`
or a new `mftl_overlap.h`):
- `MFTLCalculateOverlap(objA, triA, objB, triB, const double vecAB[3], MFTL_Overlap* out)` —
  port the projection→2D tessellation→prism-volume pipeline: build an orthonormal frame from the
  A→B separation `vecAB`, project both triangles, fit in-plane gradient + radius of curvature,
  clip the two projected triangles (interior vertices + edge–edge intersections), CCW-order and
  fan-triangulate, integrate the enclosed prism volume, map centre-of-volume back to real space,
  set `AverageSurfaceNorm=0.5(N_A−N_B)` and the blended `EffectiveRadiusOfCurvature`.
- Drive it from `MFTL_IntersectObjects`: for each detected tri–tri pair, compute the overlap and
  accumulate per-pair `MFTL_Overlap` records.
- **Periodic**: apply a single `MFTLGetMinimumImageVector` shift of body B into A's frame (via
  `vecAB`) before projection, so a cross-seam pair is handled like a mid-domain pair.
- `MFTLConvexHull2D(points)` + polygon area/centroid helpers (Graham scan) — used by the
  tessellation ordering and available as public API.

## Files/functions likely to change
New `src/general/overlap.c`, `src/general/convex_hull.c`; `include/mftl_bvh.h`/new header
(`struct MFTL_Overlap`); hook in `mftl_bvh.c`. Test `tests/test_overlap.c` (+ extend
`test_intersection.c`).

## Numerical/physical considerations
- Overlap volume via prism integration `⅓·area·Σ(height_B−height_A)`; Heron areas with
  negative-radicand guards; `MFTL_REL_TOLERANCE 5e-7` slack in half-plane/edge tests; degenerate
  point culling at ~5e-15·norm.
- Effective radius blend (0.9 "smooth" reduced radius `1/(1/R_A+1/R_B)` + 0.1 actual) — reproduce
  MFTL1 weights for comparability.
- Symmetry: `overlap(A,B)` volume must equal `overlap(B,A)` (sign/orientation care).

## MPI/PETSc considerations
Objects A and B may live on different rank subsets. Mirror MFTL1's cross-proc triangle exchange:
gather candidate triangles (those whose owner==rank and whose AABB overlaps the other body's big
box) to a common communicator, compute overlaps, reduce the per-pair records. Reuse/extend
MFTL2's spatial-ghost mechanism (`MFTL_CreateSpatialGhosts`) — and address the flagged
"brute force" gather as an optimization note (not a correctness blocker). No PETSc.

## Possible failure modes
- Broad-phase false negatives across a seam if periodic images aren't tested → the periodic test
  catches this; also address the BVH `:535` periodic question (coordinate with Stage 13).
- Tessellation degeneracies (collinear/near-parallel edges) → the guarded 2D intersection paths.
- Asymmetric overlap volume A vs B → sign bug in prism integration.
- Cross-communicator gather when A,B on disjoint ranks → communicator union handling.

## Unit/regression tests to add (`tests/test_overlap.c`)
- Two unit spheres at centre distance d<2R: overlap volume vs the analytic lens volume within
  discretization tolerance; refine → converges.
- **Symmetry**: `overlap(A,B).Volume == overlap(B,A).Volume` (tol 1e-9).
- **Periodic**: two spheres overlapping only across the X seam give the same overlap volume as
  the identical pair placed mid-domain.
- Detection regression: mirror MFTL1 `test5` (sphere∩face), `test6` (sphere∩ellipsoid across a
  z-cut); `parallel_intersection_spheres` turned into an assertion (0 collisions before contact,
  >0 at/after).
- Rank-independence: overlap volume identical on 1/2/4 ranks.

## Acceptance criteria
- Two-sphere overlap volume matches the analytic lens within tolerance; symmetric; periodic ==
  mid-domain.
- Reproduces MFTL1 `test5`/`test6` detection; parallel intersection example asserts correctly.
- Rank-independent overlap volume.

# 11 — Object–object intersection + overlap-volume quantification

Status: COMPLETE
Depends on: 02
Implemented by: Kimi Code
Reviewed by: Claude Code (2026-09-21)
Acceptance: PASSED

> **Review note (accepted, Claude Code 2026-09-21):** Faithful MFTL1 overlap-volume port.
> Two-sphere lens converges to the analytic `π(4R+d)(2R−d)²/12` (ref4 0.26%, ref5 0.06%);
> sign correct (`⅓·area·Σ(hA−hB)` positive in the overlap region, far-side/non-overlap → V=0);
> `V(A,B)==V(B,A)` bit-identical; periodic seam == mid-domain bit-identical (bodies unwrapped
> into a common frame without mutating the mesh). Rank-independent by A-ownership partition
> (each pair once) + Allgatherv of the deterministic record list; identical on np 1/2/4/8.
> Uses outward normals (`AverageSurfaceNorm = ½(nA−nB)`); convex hull (Graham scan) sound.
> Release + ASan/UBSan/LSan clean on 1/2/4/8 (7/7); no unrelated regressions. **Carry-forward
> (Stage 13, not blocking):** the BVH/spatial-ghost *detection* path (`mftl_bvh.c:624`,
> `MFTLEnsembleIntersectObjects`) is NOT periodic-correct (the `IS THIS CORRECT FOR PERIODIC
> DOMAINS?` flag remains) — the overlap-*volume* driver handles periodicity itself and does
> not use the BVH, so the Stage-11 deliverable is unaffected; resolve detection periodicity in
> Stage 13. **Optional:** `EffectiveRadiusOfCurvature` is MFTL1's osculating heuristic (accurate
> only near the contact frame origin); the driver Allgatherv's all candidates to every rank
> (MFTL1 parity — a scalability note for very large bodies).

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

## As-implemented notes (2026-09, kimi-code)

- **Driver**: `MFTLEnsembleCalculateOverlap` does NOT reuse `MFTL_IntersectObjects`/spatial
  ghosts for the volume computation. Proximity-BVH pairs only touch within ~1e-6, which yields
  sliver volumes; the prism integral needs sheet pairs far apart in 3D that overlap in the
  projection plane. Instead the driver mirrors MFTL1 `MFTL_BodyOverlap`: owned triangles whose
  AABB meets the other body's (globally reduced) big box are Allgatherv'd to every rank
  (`MPI_BYTE` with BYTE counts/displacements), and each rank computes the pairs for the
  body-A candidates it owns (each pair exactly once). Records are Allgatherv'd back so all
  ranks return the identical, deterministically ordered list.
- **Periodic**: candidate positions are packed in a common unwrapped frame (each body unwrapped
  about its own surface centre, body B additionally shifted by the minimum image of
  `cmB - cmA`). The mesh is never mutated — an earlier draft shifted owned vertices in place,
  which mixed shifted/unshifted ghost vertices into candidate geometry at np>1. Seam-straddling
  pair == mid-domain pair to 2e-12, bit-identical across np 1/2/4/8.
- **Frame sign**: projection axis `w = +vecAB/|vecAB|` (A→B); the prism integrand
  `h_A - h_B >= 0` then selects exactly the overlap region and far-side cross-pairs are
  sign-rejected. Verified: two-sphere lens converges to the analytic value
  (ref4: 0.26%, ref5: 0.06%; bit-identical `V(A,B) == V(B,A)`).
- **Bug fixed during bring-up (restored MFTL1 semantics)**: the collinear branch of the 2D
  segment intersection must return only EXISTING segment endpoints (MFTL1 between-tests).
  Interpolating along the dominant axis mixes coordinates of different endpoints and injected
  off-edge garbage corners into clip polygons (~50% volume inflation for identical
  triangulations shifted along the projection axis, where corresponding edges are exactly
  collinear — the common case for sphere pairs).
- **Centre-of-volume**: the projected centre's out-of-plane component is stored in
  `RelHeight = -h` units (MFTL1 convention); the real-space reconstruction must negate it.
- **Effective radius of curvature**: MFTL1's `EffectiveRadiusOfCurvature_Actual` is always set
  equal to `_Smooth`, so the documented 0.9/0.1 blend reduces to the smooth reduced radius
  `1/(1/R_A + 1/R_B)`; this is reproduced. (The per-triangle osculating fit itself is MFTL1's
  heuristic `r²/(h·sign)` with absolute frame heights — meaningful only for contacts near the
  frame origin; tests assert only volume/area/centre/normal quantities.)
- Detection regression: the MFTL1 test5 geometry is a unit sphere at the origin vs the unit
  square `[0,1]^2` at z=0 (the sphere's equator pierces the square). A face fully inside the
  ball produces no triangle crossings by construction — an earlier test draft used such a face
  and correctly found 0 hits. Contact assertion now also checks detection counts
  (0 hits at d=2.05, >0 at d=1.9), plus `V=0`/`V≈lens`.
- Tests: `tests/test_overlap.c` (7 tests: analytic lens + convergence, symmetry, periodic seam,
  detection regressions, contact assertion, normal direction, 2D convex hull API), registered
  as `Overlap-np{1,2,4,8}` with the standard `lsan.supp` environment.

# 08 — Serial remeshing A: criteria, vertex fitting, feature protection, tangential smoothing

Status: NOT STARTED
Depends on: 01, 06 (+ LAPACK from 01)
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Extend MFTL2's existing edge operators to full MFTL1 serial-remeshing behavior: the four
remeshing criteria, curvature-adaptive target lengths, paraboloid/barycentric vertex fitting,
feature-edge protection, and quality-driven tangential smoothing. Volume-conserving smoothing
is Stage 09; parallel orchestration is Stage 10.

## Relevant existing code
- MFTL1: `src/remeshing/remeshing.c` (`MFTL_Remeshing` driver: order **split→collapse→flip→
  smooth**, per-iteration flag clearing); `edgesplit.c` (`MFTL_Refining`, `MFTL_EdgeSplit` —
  midpoint then fit; Botsch length `L=√(6ε/κ−3ε²)`); `edgecollapse.c`; `edgeflip.c` (Delaunay/
  valence/obscure-angle/quality gate stack); `vertexfitting.c` (`ParaboloidFit` via LAPACK
  `dgelsy_`; `BarycentricInterp`); `oneringcomputations.c` (`FeatureEdge_Check`,
  `MFTL_TriangleQuality` const 20.78460969); `tangentialrelaxation.c`
  (`MFTL_ImproveTriangleQuality`, `…CurvatureBased`); `MFTL_RemeshParameter` (`mftl.h`).
- MFTL2: `src/remeshing/remeshing_core.c` (`MFTLSplitEdge`, `MFTLCollapseEdge`, `MFTLFlipEdge`,
  `MFTLSwapEdge`, `MFTLSmoothTangential`, `MFTLDetectFeatures`, `SplitApproved`,
  `MFTLFinalizeRemeshing`, `MFTLInitRemeshParams`, `MFTL_RemeshParameters`).

## Proposed changes
Extend `remeshing_core.c` (+ `include/mftl_remeshing.h` params):
- Add the four `RemeshCriteria` variants: triangle-area, Botsch–Kobbelt standard, Botsch–Kobbelt
  curvature (uses Stage-06 curvature → adaptive `L`), Jiao. Extend `SplitApproved` + add
  collapse/flip approval predicates to match MFTL1 thresholds (`maxLength/minLength`,
  `min/maxAreaThreshold`, `maxAngleBetweenTris`, `epsilon`, quality thresholds).
- **Vertex fitting**: `MFTLParaboloidFit` (local frame from summed one-ring normals; gather ≥6
  neighbours; least-squares `z=Ax²+Bxy+Cy²+Dx+Ey+F` via LAPACK `dgelsy` under `MFTL_HAVE_LAPACK`,
  else fall back to midpoint) and `MFTLBarycentricInterp` (butterfly stencil). Reposition new/
  collapsed vertices unless the edge is a feature edge.
- **Feature protection**: dihedral-angle feature-edge check gating fit and flip.
- **Tangential smoothing**: extend `MFTLSmoothTangential` with the quality-driven and
  curvature-based variants; leave feature vertices fixed; iterate to `maxSmoothingIter` /
  quality window.
- Preserve the **split→collapse→flip→smooth** ordering and per-edge one-touch guards
  (`MFTL_FLAG_*` mirroring MFTL1's `RemeshingFlag[3]`).

## Files/functions likely to change
`src/remeshing/remeshing_core.c`, `include/mftl_remeshing.h`; new `src/remeshing/vertex_fit.c`
(paraboloid/barycentric); CMake LAPACK link (from Stage 01). Test `tests/test_remeshing.c`
(extend) + new `tests/test_remesh_quality.c`.

## Numerical/physical considerations
- Curvature-adaptive length depends on the **exact** Stage-06 convention (2/R); mismatch
  changes target resolution.
- ParaboloidFit needs ≥6 well-conditioned neighbours; `dgelsy` rcond ~1e-10; guard rank-deficient
  fits (fall back to midpoint/barycentric).
- Quality constant `20.78460969` (equilateral→1.0) reused in flip and smoothing gates.
- Volume is **not** exactly conserved by split/collapse/flip+tangential alone; track drift and
  bound it (Stage 09 restores volume). Feature edges must not be smoothed away.

## MPI/PETSc considerations
This stage is **serial** (single-rank / owned-region) remeshing; the two-pass parallel
orchestration and load balancing are Stage 10. Still, run the serial ops under np1 in the test
harness (as MFTL2's `test_remeshing` does). No PETSc.

## Possible failure modes
- LAPACK absent → paraboloid fit unavailable; ensure graceful midpoint fallback and a test for
  both paths.
- Over-aggressive collapse creating valence-3 or non-manifold edges → guard (≥2 neighbours per
  side, valence checks) and re-validate connectivity after each op (`ValidateMeshConnectivity`).
- Flipping across a feature edge → lost sharp features; dihedral gate must block it.
- Fit placing a vertex far off-surface on a coarse ring → clamp / fallback.

## Unit/regression tests to add
- Extend `test_remeshing.c`: after each split/collapse/flip, `MFTLCheckMeshIntegrity` +
  full connectivity re-validation; exact live/dead entity-count deltas (as existing tests do).
- `test_remesh_quality.c`: isotropic remesh of a refined sphere toward a target edge length →
  edge-length distribution converges to target ± tolerance; mean triangle quality improves
  toward 1.0; **volume drift bounded** (e.g. < 1% before Stage-09 recovery); closedness kept.
- Curvature-adaptive remesh puts more triangles in high-curvature regions (ellipsoid tips).
- Paraboloid vs midpoint fallback both keep the mesh valid (LAPACK-on and LAPACK-off builds).

## Acceptance criteria
- All operators keep a valid, closed, integrity-passing half-edge mesh (np1) under ASan.
- Isotropic remesh converges edge lengths to target and improves quality; bounded volume drift.
- Feature edges preserved; LAPACK-on and LAPACK-off builds both pass.

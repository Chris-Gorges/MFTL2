# 08 — Serial remeshing A: criteria, vertex fitting, feature protection, tangential smoothing

Status: COMPLETE
Depends on: 01, 06 (+ LAPACK from 01)
Implemented by: Kimi Code
Reviewed by: Claude Code
Acceptance: PASSED

> **Follow-up (resolved):** feature protection is now automatic. The remesh driver runs
> `MFTLDetectFeatures` every pass (after refreshing triangle normals; the detector sets
> AND clears flags to track the evolving geometry). Collapse is protected
> flag-independently: `CollapseApproved` rejects geometrically sharp edges
> (`mftlEdgeIsGeometricallySharp`) and blocks feature-line / hard-corner collapses, and
> the fit paths use `mftlEdgeIsFeatureEx` (flag OR geometric). Sharp features therefore
> survive even when the caller never pre-flags. Regression
> `test_collapse_geometric_gate_no_flags` (cube, no `MFTLDetectFeatures`, min-edge huge so
> only the geometric gate can protect) covers this. All gates share
> `feature_angle_threshold`.

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

## Follow-up (2026-09-22): paraboloid-fit stencil + timing corrected — MFTL1 porting defect

> **Reviewed & ACCEPTED (Claude Code, 2026-09-22).** A genuine MFTL1→MFTL2 porting defect in the
> paraboloid fit (surfaced by the Stage-13 review's LAPACK-on run) was root-caused and fixed in
> `src/remeshing/vertex_fit.c`. It was **not** a test-tolerance issue.
>
> **Defect.** The paraboloid neighbourhood was gathered as `one-ring(A) ∪ one-ring(B)` **with the
> two edge endpoints A,B explicitly excluded**, and it was gathered **after** `MFTLSplitEdge`
> (from the already-rewired topology). MFTL1 (`vertexfitting.c:FindNeighbourhoodVertices`,
> `edgesplit.c`) instead fits on the **intact** diamond, using a stencil **symmetric about the
> edge midpoint** (where the paraboloid is evaluated): the vertices of the two edge-adjacent
> triangles **including both endpoints**, plus the vertices of the up-to-four complementary
> triangles across the diamond's outer edges. Excluding the endpoints and gathering post-split
> left the fit with no data near the evaluation centre, so it extrapolated — mis-placing fitted
> vertices (radial error `5.03e-2` on a unit sphere, vs the `0.05` gate).
>
> **Fix.** `mftlGatherFitNeighbours` now reconstructs MFTL1's symmetric diamond+complementary
> stencil (endpoints included), falling back to the endpoint one-ring union only for degenerate
> diamonds (missing twin/complementary), mirroring MFTL1's `FindNeighbourhoodVertices` growth
> fallback. `MFTLSplitEdgeWithFit` gathers the stencil **and** the {u,w} frame from the **intact**
> edge **before** `MFTLSplitEdge`, then applies the fit to the new midpoint vertex — correct
> because the split only appends the new vertex and rewires half-edges, leaving every stencil
> vertex's index and position unchanged (MFTL1 `edgesplit.c` fits before the rewire, equivalently).
> `MFTLCollapseEdgeWithFit` already gathers the intact-edge stencil before the collapse and now
> benefits from the same corrected stencil (matches MFTL1 `edgecollapse.c`: midvertex at the
> midpoint, fit on the intact edge, then collapse). `mftlParaboloidFitAt` centre `c = midpoint`,
> frame `w = Σ one-ring normals` (from current geometry), design matrix and evaluation reproduce
> MFTL1.
>
> **Result (genuine, not overfitting).** Radial deviation `5.03e-2 → 5.46e-5` (~920× better) —
> the accuracy a 2nd-order fit of a smooth sphere is expected to achieve, and ~50× better than the
> linear barycentric fit (`3.05e-3`), the correct ordering. The test threshold (`0.05`) was
> **kept** (not loosened); `test_vertex_fitting_validity` also now destroys the ensemble **before**
> asserting so a failure can no longer produce a misleading LSan leak. Bounds safe
> (`MFTL_FIT_MAX_NBR = 96` ≫ the ≤18 the gather can append; every append is guarded).
> **Verified:** Release + ASan/UBSan/LSan, **LAPACK-on and LAPACK-off**, full suite **53/53** on
> 1/2/4/8; periodic paraboloid path (Stage 13) clean on seam-straddlers (np8 ASan). No public API
> change; non-paraboloid and non-periodic behaviour unaffected.

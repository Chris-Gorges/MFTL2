# 06 — Vertex normals, curvature (mean/Gaussian), roughness

Status: NOT STARTED
Depends on: 01 (uses existing one-ring traversal)
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Compute per-vertex normals, mean & Gaussian curvature (discrete cotangent Laplace–Beltrami),
and surface roughness — populating the `MFTL_Vertex` fields that already exist but are never
filled. Curvature feeds Botsch–Kobbelt curvature-adaptive remeshing (Stage 08).

## Relevant existing code
- MFTL1: `src/geometricfunctions/serial/vertexfunctions.c` (`MFTL_VertexNormal`,
  angle/edge-length-weighted `w=sin(α)/(L0·L1)`); `objectvertexnormal.c`
  (`MFTLObjectVertexNormals`, boundary exchange), `MFTL_OneRingVertexNormal`;
  `src/remeshing/oneringcomputations.c` (`MFTL_VertexLaplaceBeltrami` — cotangent scheme,
  mixed-Voronoi area, mean=2/R, gauss=2/R²; `MFTL_ObjectVertexCurvatures`);
  `src/remeshing/smoothing.c` (geometric-Laplacian & dihedral-angle roughness,
  `roughness_mean`/`roughness_variance`, valence ≤ 10 arrays).
- MFTL2: `MFTL_Vertex.normal/meancurvature/gaussiancurvature/area` (declared, MPI-packed in
  `communication.c`, never computed); one-ring traversal `MFTLGetVertexOneRing`,
  `MFTLGetVertexOneRingVertices/Halfedges`; `MFTLHe*` accessors.

## Proposed changes
New unit `src/general/curvature.c` (+ decls in `include/mftl_geometry.h`):
- `MFTLComputeVertexNormals(MFTL_Object*)` — angle/edge-weighted average of incident triangle
  normals via half-edge one-ring; store in `vertex.normal`.
- `MFTLComputeVertexCurvatures(MFTL_Object*)` — cotangent Laplace–Beltrami: mixed-Voronoi area
  `A_mix`, mean-curvature vector `0.5·Σ cot·(V−Vⱼ)/A_mix` → `meancurvature = |·|` (=2/R for a
  sphere), Gaussian `2·(2π − Σθ)/A_mix` (=2/R²). Store both + `vertex.area = A_mix`.
- `MFTLComputeRoughness(MFTL_Object*)` — geometric-Laplacian (`|V − Σ(1/d)Vⱼ/Σ(1/d)|`) and
  dihedral-angle (`1 − N_{T0}·N_{T1}`) roughness with per-vertex mean/variance.
- Boundary/ghost handling: ensure a full one-ring is available (existing `MFTLExtendGhostLayer`
  / neighbourhood) before computing at partition-boundary vertices; exchange contributions so a
  boundary vertex gets the same value as in serial.

## Files/functions likely to change
New `src/general/curvature.c`; `include/mftl_geometry.h`; test `tests/test_curvature.c`.

## Numerical/physical considerations
- **Convention is load-bearing**: mean=2/R, Gaussian=2/R² (0.125 Voronoi factor). Any change
  silently alters Stage-08 adaptive edge length `L=√(6ε/κ−3ε²)`.
- Obtuse-triangle Voronoi-area special cases (area/2 at obtuse vertex, area/4 otherwise) must
  be reproduced.
- Cotangent weights blow up on degenerate/near-zero-area triangles — guard with `MFTL_VERYSMALL`.
- Roughness arrays imply **valence ≤ 10**; assert or grow.

## MPI/PETSc considerations
Curvature is a one-ring quantity: a partition-boundary vertex needs its complete ring, which
may span ranks. Guarantee ghost coverage (≥1-ring, ideally 2-ring for the cotangent stencil)
and exchange so boundary-vertex curvature is **rank-independent** — mirror MFTL1's
`MFTLObjectVertexNormals` two-phase exchange using MFTL2's ghost machinery. No PETSc.

## Possible failure modes
- Incomplete one-ring at a boundary → wrong/zero curvature on seams; the rank-independence test
  catches this.
- Sign of mean-curvature vector (convex vs concave) inconsistent → store magnitude per MFTL1,
  but keep the vector direction available for smoothing (Stage 09).
- Degenerate triangles → NaN cotangents.

## Unit/regression tests to add (`tests/test_curvature.c`)
- Unit sphere (refinement 4/5): mean curvature → 2.0 and Gaussian → 2.0 at all vertices within
  discretization tolerance (tighter with refinement); **limiting case**.
- Flat patch (planar mesh): mean & Gaussian → 0.
- Cylinder lateral surface: mean → 1/R (=0.5/R·2 per convention), Gaussian → 0.
- **Rank-independence**: boundary-vertex curvature identical on 1/2/4/8 ranks (tol 1e-7,
  accumulated-geometry scale).
- Roughness of a smooth sphere ≈ 0; of a deliberately perturbed vertex > threshold.

## Acceptance criteria
- Sphere mean=2/R, Gaussian=2/R² within refinement tolerance; flat→0; cylinder as above.
- Boundary-vertex curvature rank-independent (tol 1e-7).
- Valence-cap respected; no NaNs on the test meshes.

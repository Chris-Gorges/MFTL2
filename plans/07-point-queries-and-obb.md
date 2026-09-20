# 07 — Point queries (point-in-object, point-triangle distance) + OBB tree

Status: NOT STARTED
Depends on: 01
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Provide point-in-object classification (ray casting), point-triangle distance, and an
oriented/axis-aligned bounding-box facility at MFTL1 parity. `IsPointInsideOfParticle` in
multiflow's moveIBM depends on these (cut-cell / volume-fraction tests).

## Relevant existing code
- MFTL1: `src/generalfunctions/point.c` (`MFTLPointInsideObject`/`…Local` — random ray,
  odd-crossing rule, broadcast ray direction, `MPI_Allreduce` of stab counts, retry on
  degenerate hits up to 100×; `MFTL_StabbedTriangles` OBB-tree traversal;
  `MFTL_RayCrossingBoxTest`, `MFTL_RayPlaneIntersection`, `MFTL_PointInTriangle`,
  `MFTLPointInsideObjectBox`; `MFTLGatherPoints`); `serial/pointtriangledistance.c`
  (`MFTLPointTriangleDistance`, 6-region Voronoi classification);
  `src/boundingbox/boundingbox.c` (`MFTLGenerateBoundingBox` OBB tree, `MFTLGenerateBigBoundingBox`),
  `moments.c` (covariance + Jacobi eigen).
- multiflow callers: `moveibm/ispointinsideofparticle.c` (builds ortho BBox + big BBox,
  `MFTLGatherPoints`, collective `MFTLPointInsideObject`), `MFTLPointInsideObjectBox` (4–6 uses).
- MFTL2: existing **BVH/LBVH** (`src/general/mftl_bvh.c`, `MFTL_GenerateLBVH`,
  `MFTL_IntersectObjects`, `MFTL_TriTriIntersect`) — reuse for ray/triangle acceleration
  instead of porting the OBB tree wholesale.

## Proposed changes
New unit `src/general/point_query.c` (+ decls in `include/mftl_geometry.h`):
- `MFTLPointTriangleDistance(const MFTL_Object*, int tri, const double p[3])` — port the
  6-region Voronoi method (self-contained per triangle).
- `MFTLPointInsideObject(MFTL_Object*, const double p[3])` — ray casting: broadcast a single
  random direction from rank 0, traverse the **BVH** (reuse `mftl_bvh.c` node AABB slab test)
  to gather candidate triangles, count crossings on owned triangles, `MPI_Allreduce(SUM)`,
  odd⇒inside; retry on degenerate (vertex/edge) hits; return −1 for on-surface. Periodic:
  test against the nearest image of the object (Stage 13 refines this).
- `MFTLPointInsideAABB(...)` — cheap containment with `MFTL_VERYSMALL` slack.
- OBB: provide `MFTLGenerateOBB(MFTL_Object*)` (covariance→Jacobi eigenvectors→rotation, extents
  from rotated vertices) **only if** a consumer needs an oriented box the BVH doesn't cover
  (moveIBM's ortho box). Prefer reusing the BVH AABBs where possible; document the decision.

## Files/functions likely to change
New `src/general/point_query.c`; optional OBB additions in/near `src/general/mftl_bvh.c`;
`include/mftl_geometry.h`/`mftl_bvh.h`; test `tests/test_point_query.c`.

## Numerical/physical considerations
- Ray length `50·|global_max−global_min|` (MFTL1) to guarantee exit; single shared random
  direction so all ranks agree; degenerate-hit retry to avoid grazing vertices/edges.
- Odd/even parity requires a **closed** surface; integrity precheck.
- Point-triangle distance is exact per triangle; watch the mirror/quadrant normalization.

## MPI/PETSc considerations
Point-in-object is **collective**: the object is distributed, each rank counts crossings on its
owned triangles, then `Allreduce`. The ray direction MUST be identical across ranks
(`MPI_Bcast`). Degeneracy flags must also be reduced so all ranks retry together. This exact
pattern is what multiflow's `ispointinsideofparticle.c` relies on (it also `MPI_Allgatherv`s
triangles for the IRL cut-cell path — keep an all-gather helper analogous to `MFTLGatherPoints`).
No PETSc directly, but results feed PETSc cut-cell volume fractions.

## Possible failure modes
- Different ray directions per rank → inconsistent parity → nondeterministic inside/outside.
- Grazing hits not flagged → miscount; ensure the `SameSide==2` degeneracy path exists.
- Open surface → undefined parity; precheck closedness.
- BVH traversal missing a triangle straddling a node boundary → missed crossing.

## Unit/regression tests to add (`tests/test_point_query.c`)
- Unit sphere: point at origin → inside; at (2,0,0) → outside; on surface → on-surface(−1);
  just inside/outside r=1 boundary correctly classified.
- Point-triangle distance vs analytic for vertex/edge/interior nearest-feature cases.
- **Rank-independence**: identical inside/outside verdict on 1/2/4/8 ranks for a battery of
  points (regression mirror of MFTL1 `test3.c` probe points; nearest-triangle mirror of
  `test4.c`).
- Degenerate ray (aimed through a vertex) still returns the correct verdict via retry.

## Acceptance criteria
- All probe points classified correctly and **identically across rank counts**.
- Point-triangle distances match analytic to 1e-12.
- Reproduces MFTL1 `test3`/`test4` classifications.

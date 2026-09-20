# 10 — Parallel remeshing + load balancing

Status: NOT STARTED
Depends on: 08, 09
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Finish the parallel-remeshing layer that is currently broken at HEAD, and add MFTL1's x-only
1-D load balancing with the two-pass boundary-shift scheme, so remeshing produces
**rank-independent** results and rebalances triangle load across ranks.

## Relevant existing code
- MFTL2 (broken WIP, `d248554…HEAD`): `src/remeshing/remeshing_parallel.c`
  (`MFTLRemeshEnsemble` duplicated vs `remeshing_core.c:1224`; `MFTLRemeshEnsembleShiftScheme`
  and `MFTLRemeshEnsembleDispersedCentering` **called but undefined**; `MFTLComputeCoarseningFactor`,
  `MFTLRemeshWithCoarsening`; `MFTL_Object**`/`*` mismatch at ~:45). `MFTLExtendGhostLayer`,
  `MFTLRedistributeListOfObjects`/`MFTLRedistributeObject` (`communication.c`).
- MFTL1: `src/remeshing/remeshing.c` (`MFTL_Remeshing` two-pass under LoadBalance;
  `MFTLRedistributeObjectLoadBalanced` — equal-triangle 1-D x-decomposition iterative optimizer;
  `ShiftOwnershipRanges` — shift x-boundaries by `max(0.2·minDX, 3·referenceLength)` for a second
  pass; `MFTL_RemeshBoundaryCleanup`).

## Proposed changes
- **Resolve the build defects first** (they are the reason HEAD doesn't compile — cherry-pick
  the sound parts): keep a single `MFTLRemeshEnsemble` (decide `void` vs `int` return; update
  the header decl), implement or remove `MFTLRemeshEnsembleShiftScheme` /
  `MFTLRemeshEnsembleDispersedCentering`, fix the `objects` pointer type (`MFTL_Object**`).
- **Parallel driver**: per remeshing iteration — ensure ≥1-ring ghosts (`MFTLExtendGhostLayer`)
  → run Stage-08 operators on owned edges (split→collapse→flip) → Stage-09 smoothing →
  `MFTLUpdateGhostPositions` → `MFTLRedistributeListOfObjects`.
- **Load balancing** `MFTLRedistributeObjectLoadBalanced`: x-only equal-triangle-count decomposition
  (`NoOfProcs=clamp(floor(total/MinNoTriangles),1,nproc)`), iterative slab-width optimizer
  (`DX += dampingfactor·(DX·ratio−DX)`, converge `max|1−ratio|<0.2`), retry with fewer procs if a
  slab < `Sizethreshold`.
- **Two-pass boundary shift** `MFTLShiftOwnershipRanges`: shift x-cuts so triangles previously on
  a proc boundary become interior for a second remeshing pass (they'd otherwise never be
  remeshed). Keep `Ycut=Zcut=1` (x-only) to avoid crossed boundaries.

## Files/functions likely to change
`src/remeshing/remeshing_parallel.c`, `remeshing_core.c`, `include/mftl_remeshing.h`;
`src/general/communication.c` (ownership-range shift helpers). Test `tests/test_remesh_parallel.c`.

## Numerical/physical considerations
- **Rank-independence is the headline invariant**: remeshing the same initial mesh on 1/2/4/8
  ranks must yield statistically identical results — same final volume (to Stage-09 tolerance),
  same triangle-count band, same edge-length distribution. Exact per-triangle identity is not
  expected (operation order differs), but conserved/aggregate quantities must match.
- Boundary triangles must actually get remeshed (the two-pass shift guarantees coverage).
- Conservation across redistribution: volume and (Stage 12) per-triangle data preserved.

## MPI/PETSc considerations
- Edge operations near a partition boundary need a consistent ghost ring; an edge split/collapse
  on a shared edge must be decided by a single owner to avoid divergent topology. Define the
  ownership/tie-break rule explicitly (e.g. lower `owner_proc` decides) and re-sync.
- Load-balance redistribution keys on triangle centre → owner; ensure the x-slab optimizer is
  computed identically on all ranks (broadcast the slab boundaries).
- This is where the multiflow dual-decomposition matters downstream (Stage 15): load balancing
  must still allow `AdaptOwnershiprangesToCartesianOverlapMesh` to realign afterwards.

## Possible failure modes
- Divergent topology when two ranks independently remesh a shared edge → the single-owner rule
  and re-validation must prevent it.
- Load-balance optimizer non-convergence (oscillating slab widths) → cap `maxBalanceIter`,
  fall back to previous decomposition.
- Ghost ring too thin for a collapse that removes a boundary vertex → require 2-ring near
  boundaries.
- Re-introducing the HEAD build breakage → CI must compile `remeshing_parallel.c` cleanly.

## Unit/regression tests to add (`tests/test_remesh_parallel.c`)
- Remesh a refined sphere on np1/2/4/8: assert equal final volume (Stage-09 tol), triangle count
  within a small band, and edge-length mean/variance within tolerance across rank counts
  (**rank-independence**).
- Load balance a deliberately imbalanced distribution → per-rank triangle counts converge to
  within `Sizethreshold` of equal; no rank starved.
- Boundary coverage: a feature straddling a proc boundary is remeshed after the two-pass shift.
- Full LeVeque parallel advect+remesh (mirror MFTL2 `parallel_advection_*`) conserves volume.

## Acceptance criteria
- `remeshing_parallel.c` compiles cleanly (HEAD breakage resolved); full CTest green on 1/2/4/8.
- Remeshed volume and aggregate mesh statistics rank-independent within tolerance.
- Load balancing converges and covers boundary triangles.

# 15 — MFTL1→MFTL2 swap behind the adapter + compatibility shim

Status: NOT STARTED
Depends on: 02–14
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Switch multiflow from MFTL1 to MFTL2 by reimplementing the Stage-14 adapter on MFTL2's API and
providing a thin `mftl.h`-compatible shim for the pure function-call sites — preserving the dual
MPI decomposition, the `sendnonlocal*` transpose bridge, and the opaque per-triangle data
contract. This is the final stage and depends on MFTL2 having reached parity (Stages 02–13).

## Relevant existing code
- Stage-14 adapter `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}` (the only place mesh
  access lives).
- multiflow coupling to preserve: `initializefronttracking.c` (MFTL `M[3]` + ownership ranges
  *adapted* to `Run->FTDomain` particle-box mesh), `moveibm/…AdaptOwnershiprangesToCartesianOverlapMesh`,
  `sendnonlocalcurvature.c`/`sendnonlocalvelocities.c`/… (transpose-table bridge between the MFTL
  and PETSc/DMPlex decompositions), `ispointinsideofparticle.c` (collective point-in-object +
  `MPI_Allgatherv` of triangles), `struct CellData` (opaque payload).
- MFTL2 parity API from Stages 02–13 (integrals, transforms, generators, I/O, curvature, point
  queries, remeshing, intersection/overlap, opaque data, periodicity).

## Proposed changes
- **Reimplement the adapter internals on MFTL2**: iteration → array loops
  (`t<n_local_triangles`); accessors → `MFTLGetTriangle*`/`vertices[].pos`/`.center/.normal/.area`;
  owner → `owner_proc`; payload → `MFTLGetTriangleData`/`data_size` (Stage 12); FT fields →
  `meancurvature`/`gaussiancurvature`/scalar. The physics files (rewritten in Stage 14) do not
  change again.
- **Compatibility shim** `mftl_compat.{h,c}` implementing the subset of `mftl.h` function
  signatures multiflow calls (the ~103 functions — many are pure math helpers `MFTL_VectorInProd`,
  `MFTL_ScaleVector`, `MFTL_PointPointDistance`, copied verbatim; the rest map to MFTL2 parity
  functions). This lets any residual function-call sites compile unchanged.
- **Dual-decomposition bridge**: implement `MFAdapterAdaptOwnershipToOverlapMesh(obj, overlapMesh)`
  mapping MFTL2's `MPI_Cart` ownership onto the `CartesianOverlapMesh` extents multiflow expects,
  so `sendnonlocal*` continues to reconcile the MFTL and PETSc partitions. This is the highest-risk
  item — validate the transpose bridge explicitly.
- **Opaque data contract**: `MFTLInitializeObjectData(obj, sizeof(struct CellData))` → MFTL2
  `data_size`; byte-exact transport (Stage 12). No pointers in `CellData` (already true).
- **Build wiring**: `#ifdef MFTL_ON` → link MFTL2 (`libMFTL.a`), point `-DMFTL_DIR` at MFTL2,
  reconcile include paths (`mftl.h` shim vs `mftl_*.h`). Handle MFTL2's per-object communicator
  vs MFTL1's `OBJECT_COMM_WORLD`.
- **Periodicity**: keep multiflow front-tracking non-periodic by default (its periodic branches
  are disabled today); expose MFTL2 periodicity as opt-in per object so existing runs are
  unchanged.

## Files/functions likely to change
`multiflow/src/general/mftlwrapper/mftl_adapter.c` (rewrite internals), new
`mftlwrapper/mftl_compat.{h,c}`; `multiflow/CMakeLists.txt` / build presets (`-DMFTL_DIR` →
MFTL2, LAPACK if paraboloid fit used); no changes to physics files beyond include paths.

## Numerical/physical considerations
- Results will **not** be bit-identical to MFTL1 (different mesh ordering, remeshing paths, float
  sums). Acceptance shifts to **physics-tolerance** validation, not golden-HDF5 diffs.
- Conserved quantities (volume, surface-scalar mass) must match MFTL1 within tight tolerance;
  curvature convention (2/R) must match so surface-tension forces are consistent.
- Expect (and measure) a **performance improvement** from the array/khash/BVH backend — one of
  the two original motivations.

## MPI/PETSc considerations
- **Dual decomposition is the crux**: MFTL2's `MPI_Cart` per-object grid must be adaptable to the
  fluid `CartesianOverlapMesh`/PETSc(DMForest) partition exactly as MFTL1's ownership ranges were,
  or the `sendnonlocal*` transpose exchanges will mis-route curvature/velocity/pressure and
  silently corrupt the front↔fluid coupling.
- `ispointinsideofparticle.c` collective ray test + triangle `MPI_Allgatherv` must map to MFTL2's
  collective point-in-object (Stage 07) and an all-gather helper.
- Validate on ≥4 ranks to exercise both decompositions and the bridge.

## Possible failure modes
- Ownership/overlap-mesh mismatch → mis-routed transpose exchanges → subtly wrong forces (no
  crash). The bridge test is essential.
- Opaque data size mismatch or a stray pointer in a payload → cross-rank corruption.
- Iteration-order-dependent multiflow code assuming uthash order → make such code order-independent.
- Communicator model mismatch (per-object comm vs global) → deadlocks/wrong reductions.
- `#ifdef MFTL_ON`-off build broken → keep the guard clean.

## Unit/regression tests to add
- multiflow FT and IBM cases (`FT`, `IBM`, `IBM-NP4`) pass on 1 and 4 ranks under MFTL2, judged by
  the **analytic** regressions from Stage 14 (oscillating-droplet frequency vs Lamb; rising-bubble
  terminal velocity; LeVeque shape error/volume drift) within physics tolerance — not golden diffs.
- **Dual-decomposition bridge test**: a front spanning a fluid-partition boundary gets correct
  non-local curvature/velocity after `sendnonlocal*` (compare a scalar field integral vs MFTL1).
- Opaque `CellData` round-trips through MFTL2 redistribution byte-exact; surface-scalar mass
  conserved.
- **Performance**: wall-clock of a representative FT step lower under MFTL2 than MFTL1 (report the
  speedup; the uthash→array motivation).
- `#ifdef MFTL_ON` off still compiles and runs the non-MFTL solver.

## Acceptance criteria
- multiflow runs FT + IBM on MFTL2 (1 and 4 ranks) meeting the analytic physics-tolerance
  regressions; conserved volume/mass match MFTL1 within tolerance.
- Dual-decomposition bridge verified (correct non-local exchange across a fluid-partition
  boundary); opaque data byte-exact; measured performance improvement over MFTL1.
- Clean build with `MFTL_ON` on (→ MFTL2) and off.

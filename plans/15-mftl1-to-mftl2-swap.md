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
- **Compatibility shim** `mftl_compat.{h,c}` reproducing the four MFTL1 headers multiflow
  includes — `mftl.h`, `mftl-functions.h`, `mftl-constants.h`, `mftl-genericio.h` — covering the
  **100 distinct `MFTL*` functions** actually called, the **~20 constants/macros** (`MFTL_NULL`,
  `MFTL_YES`/`NO`, `MFTL_VERYSMALL`, `MFTL_ERROR`, `MFTL_TRI_PACK_SIZE`, `MFTL_ORTHOGONAL`,
  `MFTL_ALIGNED`, `MFTL_ONESIXTH`, `MFTL_DEG_TO_RAD`, `MFTL_GHOST_VERTEX_FAIL`, …), and the memory
  helpers (`MFTL_FreeMemory` 27 uses, `MFTL_AllocateMemory`, `MFTL_DynamicallyArrayGrow` — thin
  wrappers over `free`/`malloc`, matched to MFTL2's allocator). Many are pure math helpers
  (`MFTL_VectorInProd`, `MFTL_ScaleVector`, `MFTL_PointPointDistance`, `MFTL_SQ`, tensor ops),
  copied verbatim; the rest map to MFTL2 parity functions. This lets residual function-call sites
  compile unchanged. **The distinct-value ids are for the compat surface** — but three families
  need real reimplementation, not passthrough (see "Structural representation mismatches").
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

## Structural representation mismatches (the non-passthrough work — added 2026-09-21 from the multiflow scan)
The compat surface is mostly 1:1, but three families are coupled to MFTL1's *representation* and
must be reimplemented, not shimmed. These are the real risk in this stage:
1. **`MFTL_Edge` → half-edges (biggest).** multiflow reaches `Triangle->Edge[j]` at **153 sites**
   across 8 front-tracking files and calls `MFTL_EdgeLength`, `MFTL_EdgeComplementaryTriangle`,
   `MFTL_EdgeNormal`, `MFTL_EdgeOpposingAngle`, `MFTL_OneRingVertexNormal`. MFTL1 has explicit
   undirected `MFTL_Edge` objects with `Vertex[2]`, `id`, and complementary-triangle links; MFTL2
   has only directed half-edges (`he_idx`/`twin_idx`/`next_idx`/`vertex_idx`), no standalone edge
   type. **Mitigation:** Stage 14's edge facade (`MFAdapterTriangleEdge`/`EdgeEndpoints`/
   `EdgeLength`/`EdgeAdjacentTriangles`/`EdgeId`) is the seam — back it here by choosing the
   canonical half-edge (min endpoint id) per undirected edge; complementary triangle = `twin`'s
   face; opposing angle from the two `next` vertices; a stable edge id from `min(he,twin)`. If
   Stage 14 shipped a too-thin facade, this is where the 8 files would have to reopen — so verify
   the facade covered every edge use before starting.
2. **`MFTL_BoundingBox`/OBB API → MFTL2 BVH/AABB.** ~30 sites across 6 files (moveibm
   `ispointinsideofparticle`, AMR `createadaptivitylabel`, marchingcubes `breakup`/`coalescence`,
   `indicator_irl`, `initializepboxmesh_ft`) use `MFTLGenerateBoundingBox`/`GenerateBigBoundingBox`/
   `ClearBoundingBox` + `MFTLPointInsideObjectBox(box, p)`. MFTL2 (Stage 07) provides
   `MFTLObjectAABB` + `MFTLPointInsideObject(obj, p)` — **different signature and no `MFTL_BoundingBox`
   struct**. The shim must present an `MFTL_BoundingBox`-shaped facade over MFTL2's AABB/BVH and
   bridge the point-in-box signature; and the Stage-07 **per-call LBVH rebuild** must be fixed
   (see efficiency guardrails) before `ispointinsideofparticle` relies on it.
3. **Manual triangle packing (`MFTL_TriangleToArray`/`MFTL_ArrayToTriangles`/`MFTL_TRI_PACK_SIZE`,
   10 uses).** `ispointinsideofparticle.c` and `reconstructfrontwithmarchingcubes.c` serialize
   triangles into flat arrays with MFTL1's byte layout for their own `MPI_Allgatherv`. MFTL2's
   packing (`PackedTriangleComplex`) differs → **these two sites must be rewritten** to MFTL2's
   pack API (or to the adapter's coordinate extraction), not shimmed byte-for-byte.
4. **N-ring neighbourhood parity.** multiflow uses 1-/2-/3-ring triangle & vertex lists,
   `MFTL_TrianglesWithinDistance`, and `MFTL_TriangleListFromVertex`. MFTL2 has the 1-ring; the
   2-/3-ring and within-distance variants must be composed from 1-ring traversal (Stage 06/13
   groundwork) — confirm they exist before the swap or add them to the shim.

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

### Efficiency guardrails (MUST — do not regress the reason MFTL2 exists)
The MFTL2 speedup is only realized if the physics hot loops run at array speed and the shim/
adapter add no per-element overhead. Preconditions and checks:
- **Adapter accessors stay `static inline`/zero-copy** (Stage 14): when the internals switch to
  MFTL2, they must resolve to direct `obj->triangles[t]`/`obj->vertices[...]` access — confirm at
  `-O3` that a representative hot loop is overhead-neutral vs direct access.
- **Compat shim must be thin.** Math helpers copied verbatim (identical cost); every other shim
  function must be an O(1) passthrough — audit for any that would turn an MFTL1 O(1) into an
  MFTL2 O(n) scan (none expected, but check, e.g. id→index lookups must use the `khash` maps,
  never a linear triangle scan).
- **Fix `MFTLPointInsideObject`'s per-call BVH rebuild before relying on it** (Stage-07 carry-
  forward): `moveibm/ispointinsideofparticle.c` issues many point queries against one particle;
  a per-query `MFTL_GenerateLBVH` is O(N_points·n log n) and would make the cut-cell coupling
  *slower* on MFTL2. Build/cache the BVH once (or accept a prebuilt BVH) so the query loop is
  O(N_points·log n).
- **Per-triangle `data` blob churn** (Stage-12): `CellData` is one `calloc`/`free` per triangle,
  and remeshing split/collapse churns them — same as MFTL1 (no regression), but if profiling
  flags it under heavy remeshing, switch to a contiguous `data_size`-strided pool (documented
  Stage-12 option). Access cost is unchanged either way.
- **Measure, don't assume.** Acceptance includes a wall-clock comparison of a representative FT
  timestep (front build + one remesh + one force/spread cycle) MFTL2 vs MFTL1 on ≥4 ranks; the
  MFTL2 path must be **no slower**, and ideally faster, than MFTL1 — a slower result means the
  adapter/shim is bottlenecking the backend and must be fixed before acceptance.

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
- **Edge-facade semantics wrong**: half-edge → undirected-edge mapping picks a non-canonical
  representative, or `EdgeComplementaryTriangle`/`EdgeOpposingAngle` return the wrong face/vertex →
  silently wrong front remeshing/curvature. Test the facade against MFTL1 on a known mesh.
- **Manual-packing rewrite drifts**: the rewritten `ispointinsideofparticle`/
  `reconstructfrontwithmarchingcubes` all-gathers must reproduce the same triangle set (order may
  differ) → assert the point-in-particle result matches MFTL1.
- `#ifdef MFTL_ON`-off build broken → keep the guard clean.

## Unit/regression tests to add
- multiflow FT and IBM cases (`FT`, `IBM`, `IBM-NP4`) pass on 1 and 4 ranks under MFTL2, judged by
  the **analytic** regressions from Stage 14 (oscillating-droplet frequency vs Lamb; rising-bubble
  terminal velocity; LeVeque shape error/volume drift) within physics tolerance — not golden diffs.
- **Edge-facade parity test**: on a fixed mesh, `MFAdapterEdgeLength`/`EdgeEndpoints`/
  `EdgeAdjacentTriangles`/`EdgeOpposingAngle` match MFTL1's `MFTL_Edge` results (the 153-site
  half-edge bridge — the highest structural risk).
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

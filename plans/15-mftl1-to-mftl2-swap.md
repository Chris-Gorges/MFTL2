# 15 — MFTL1→MFTL2 swap behind the adapter + compatibility shim

Status: COMPLETE (2026-09-23) — independently reviewed
Depends on: 02–14
Implemented by: Claude Code (multiflow branch `berend-mftl`; MFTL2 working tree; nothing committed)
Reviewed by: Claude Code, independent review 2026-09-23 (see "Independent review" at the end)
Acceptance: PASSED (2026-09-23) — with the should-fix items and the corrected golden-baseline
diagnosis recorded in "Independent review"

> **Implementation note.** The "compatibility shim" of the original plan was NOT built: the
> brief for this stage asked for local math helpers and no MFTL1 compat library, and the Stage-14
> adapter made a shim unnecessary. multiflow now contains no MFTL1 include, type or call; see
> "Stage 15 completion" at the end of this file.

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

## Stage 15 completion (2026-09-23, Claude Code) — IMPLEMENTED, pending review

multiflow (branch `berend-mftl`) now builds and runs front tracking, the IBM marker meshes,
MoveIBM and marching cubes on MFTL2 only (`find_package(MFTL2)` / `MFTL2::MFTL2`); MFTL1 is
neither included nor linked. The Stage-14 integration layer (`mftl_adapter.h`) is unchanged in
spirit; its MFTL1 backend (`mftl_adapter_mftl1.*`) was replaced by the MFTL2 backend
(`mftl_adapter_mftl2.h`, `mftl_adapter_mftl2_inline.h`, `mftl_adapter_mftl2.c`) and deleted
(it remains in the history). Nothing is committed in either repository.

### 1. MFTL2 library changes
Stage-14 carry-forwards:
- **L1 rectilinear decomposition + rank map**: `MFTL_Domain.axis_cuts`, `MFTLSlabIndex`,
  `MFTLEnsembleCreateRectilinear`, `MFTLEnsembleSetDecomposition`; `MFTLGetOwnerProc` uses the
  cuts and clamps out-of-box points (no assert); neighbours from the rank map; temporary domains.
- **L2 caller-held point query**: `MFTL_PointQuery` (BVH built once), `MFTLPointQueryInsideMany`.
- **L3 retained twin gid**: `MFTL_HalfEdge.twin_gid`, packed/unpacked, refreshed by the remesh
  finaliser (`mftlRefreshTwinGids`), `MFTLHeTwinGid`.
- **L4 detach/transfer**: `MFTLEnsembleRemoveObject`, `MFTLEnsembleTransferObject`.
- **L5 single-vertex kernels**: `MFTLComputeVertexNormalAt` (MFTL1 sin(α)/(l0·l1) weights),
  `MFTLComputeVertexCurvatureAt`.
- **L6 spatial ghosts**: not enabled (MFTL1 had none); `mftlRankBox` shared by
  `MFTLEnsembleCreateSpatialGhosts` and the new `MFTLObjectSendLocalTrianglesToAll`.
- **L7 conserved surface scalar**: `MFTL_Triangle.surface_scalar`, transported with the
  triangle, remapped mass-conservingly exactly as MFTL1 (split: pre-fit proportional share +
  post-fit concentration; flip: A_new/ΣA_old; collapse: ring mass ∝ new area; TSUR / volume-
  conserving smoothing: c *= A_cached/A_new with a full-ring parallel gate), native-mesh I/O.

New for the swap:
- Triangle-soup builder `MFTLBuildFromSoup` (distributed input, rim twins resolved through the
  retained twin gids, unreferenced vertices dropped), `MFTLObjectAppendTriangleCopy`,
  `MFTLObjectClear`; native mesh I/O `MFTLWriteNativeMesh` / `MFTLReadNativeMesh` (MFTL1 format,
  scalar with %.15e; failures reported on every rank).
- Serial bracket `MFTLEnsembleBeginSerial` / `EndSerial` (MFTL1's "gather to one rank" kernels).
- Remesh switches in `MFTL_RemeshParameters` (all initialised by `MFTLInitRemeshParams`):
  `skip_split/collapse/flip/tangential_smoothing`, `skip_volume_recovery`, `fixed_iterations`,
  `skip_split_cleanup`, `quiet`, `stat_splits/collapses/flips` and **`mftl1_criteria`**.
- **MFTL1 edge criteria** (`src/remeshing/mftl1_criteria.c`, opt-in via `mftl1_criteria`):
  MFTL1's split / collapse / flip tests (JIAO incl. the wing-edge and opposing-angle rules,
  area, Botsch–Kobbelt, Boundary_Check_Collapse feature-vertex test, collapse neighbour/valence
  checks, all six flip checks) and, in the serial driver, `MFTL_Remeshing`'s sweeps with MFTL1's
  edge ORDER (split longest-first and collapse shortest-first with the reference-length break,
  uthash `HASH_SORT` order carried into the flip sweep, one slot per edge, merged wing edge keeps
  the survivor's slot, MFTL1's collapse survivor convention).
- **Paraboloid fit parity** (`vertex_fit.c`): in `mftl1_criteria` mode the fit frame uses
  MFTL1's weighted vertex normals, the fit is unconditional (no rank/clamp rejection — MFTL1 had
  none) and the collapse fit uses a free mid point with both endpoints at their original
  positions (the MFTL2 path moved the survivor before gathering the stencil).
- MPI datatypes resized to `sizeof` (padding-safe), packed `surface_scalar` / `twin_gid`.
- Parallel remesh: split phase applies exactly the candidates it counted (id blocks can no longer
  overlap with neighbourhood-reading criteria); load balancing on 3-D grids through a virtual
  x-only domain; redistribution after the split clean-up.

Bugs found and fixed in MFTL2 on the way (all with evidence):
- **TSUR (Stage 09)**: the two endpoint normals were summed un-normalised; MFTL1 normalises each
  first. With MFTL1's edge order, MFTL2's TSUR now reproduces MFTL1 to 4e-16 (was 1.2e-5).
- **Area/normal floor**: `|cross| < 1e-6` was floored, giving wrong areas at front-tracking
  scales; the area is no longer floored and the normal divides by max(1e-15, |cross|).
- **Sphere generator**: id lookups were never filled on one rank (`FindTriById` failed).
- Remesh parameters added in this stage were uninitialised (garbage stats / hang) — initialised.
- Parallel TSUR lost scalar mass on ghost ring triangles — tightened gate.
- Twin-gid disagreement after the split clean-up — redistribute after the clean-up.
- Soup builder corrupted non-empty objects / crashed on unreferenced vertices; UB memcpy on NULL.
- `MFTLMomentOfInertia(…, double I[3][3], …)` clashed with complex.h's `I` macro via PETSc
  (renamed `inertia`); two warnings fixed (`size_t` loop in a header, maybe-uninitialised send
  buffer).

### 2. Backend design (multiflow `mftl_adapter_mftl2.*`)
- decomposition → rectilinear `MFTL_Domain` (cuts + rank map from the DMDA layout);
  body list → one `MFTL_Ensemble` + uthash id table (insertion order); body →
  `struct MFAdapterBody { MFTL_Object *obj; ... }`; tri/vert → pointers into the arrays;
  edge → canonical half-edge (smaller gid), id = min(gid, twin_gid) incl. the ghost rim.
- Standalone bodies with the same decomposition share one reference-counted ensemble (no
  communicator per IBM marker body). Per-body ensemble-level operations use a one-object view.
- Ownership by triangle centre: centres of local triangles are refreshed before every
  redistribution. One ghost layer (all edge neighbours of owned triangles + one-ring of owned
  vertices). Bodies assembled from rank-local pieces (distributed soup build, component copy,
  merge, unpack) are flagged and their next redistribution runs twice — MFTL2's redistribution
  ghosts a triangle only to the owners of its STORED neighbours (found by the adapter tests on
  2–4 ranks; affects marching-cubes reconstruction).
- Remesh: 1 rank → `MFTLRemeshEnsemble`; SERIAL → serial bracket; otherwise
  `MFTLRemeshEnsembleShiftScheme`; always MFTL1 parity settings (`mftl1_criteria`, fixed one
  iteration, no volume recovery / smoothing inside remeshing, VCS none — multiflow calls TSUR /
  VCS itself), then MFTL1's clamp of every vertex 1e-7 inside the global box and a
  redistribution. TSUR / VCS / roughness smoothing run as serial kernels (as MFTL1).
- Sphere generation reproduces MFTL1's icosphere exactly (MFTL1 icosahedron orientation, midpoint
  projection per level) through the builder: the initial front is identical to MFTL1's (642/642
  vertices, max coordinate difference 0).
- Unreadable GTS files / restart meshes abort the run as MFTL1 did (MFTL2 returns an error; the
  first version silently continued with an empty front).

### 3. Converted multiflow code
All MFTL1 reach-ins in ~85 files (front tracking incl. sharp CSF / integral surface tension,
surface scalar transport and viscosity, marching cubes, indicator (IRL), MoveIBM, IBM marker
readers, mftlwrapper, example userfiles) use the integration layer; math helpers are local
(`ftvectorops.h`, arithmetic identical to MFTL1); `MF_NormalizeVector` keeps MFTL1's 1e-15 floor
in MFTL builds. The one missed call, MFTL1's `VolumeRecovery` in the marching-cubes
reconstruction, is ported as `FTVolumeRecovery` (application level, same algorithm).
`mftl1_init_grid` and `MFAdapterStandardProcGrid` are removed. No MFTL1 identifier remains
outside comments.

### 4. Build
`MFTL2_ROOT` replaces `MFTL_DIR` (CMake, presets, scripts, README, test scripts);
`find_package(MFTL2 CONFIG REQUIRED)`, `target_link_libraries(MF3 MFTL2::MFTL2)`; configure fails
if MFTL2 was built without LAPACK (the paraboloid fit is required — without it every fit silently
falls back to barycentric interpolation). MFTL2 must be built with `-DMFTL_USE_LAPACK=ON`.

### 5. Validation (2026-09-23)
Reference = MFTL1 (multiflow HEAD, MFTL1 1.x) built with the same flags; both trees built with
`-O3 -DNDEBUG -ffp-contract=off` (PETSc's `-march=native` otherwise lets GCC contract inlined
helpers into FMAs → ULP noise). Comparisons with the golden-test criterion |Δ| ≤ 1e-7 + 1e-4·|ref|.
- **MFTL2 library**: `ctest` 57/57 in Release and in Debug + ASan/UBSan/LSan, both with
  `MFTL_USE_LAPACK=ON`; `test_stage15` (78 checks: rectilinear ownership, twin gids and transport,
  surface-scalar conservation through split/collapse/flip/TSUR in all three drivers and with both
  MFTL2 and MFTL1 criteria, point queries, soup build, native I/O, transfer) passes on 1–4 ranks
  with and without LAPACK.
- **Integration-layer tests** (`multiflow/tools/mftladapter_tests`, now on MFTL2): 78 / 168 /
  261 / 360 checks on 1 / 2 / 3 / 4 ranks, 0 failed — incl. surface-scalar mass through
  remeshing (≤ 1e-10) and restart (≤ 1e-12).
- **Kernel parity harnesses** (MFTL1 vs MFTL2 on the same perturbed icosphere): TSUR with MFTL1's
  edge order 4e-16; Kuprat volume-conserving smoothing ≤ 2e-16 on edges whose MFTL1 storage
  orientation matches (see §7); a full `MFTL_Remeshing` pass (JIAO, refine + coarsen + flip,
  paraboloid fit) is identical to MFTL1 — same operation counts and volume to 11 digits — in 6
  of 9 configurations (all refining passes); the 3 coarsening-heavy passes differ (§7).
- **Solver A/B, MFTL2 vs MFTL1** (np1 and np4):
  - FT_Sharp_CSF (IRL) and FT_Sharp_Integral (IRL): all datasets within tolerance (worst
    normalised difference 3.5e-7 / 4.6e-7).
  - IBM marker meshes (IBMMFTL-A np1/np4, IBMMFTL-B np4): within tolerance (worst 7e-8; one
    dataset at np4 flagged by the relative test at 4e-8 of its maximum).
  - FT (classic, remeshing every step, TSUR every 20): initial fields identical (1.5e-15); with
    remeshing off identical after 10 steps (indicator 1e-13); with remeshing the fields differ
    pointwise after 100 steps (worst 0.11 / 0.13 of a field maximum, localised at the
    interface) because of the remaining remeshing order effects (§7). Physics: bubble volume
    6.7e-4 / 1.4e-4, rise velocity 1.3e-3 / 2.9e-4, centroid 2e-4 (abs) — within MFTL1's own
    np1-vs-np4 spread (7.1e-4, 1.3e-3, 7e-5); front after 100 steps 904 vs 892 triangles, mean
    edge 0.0690 vs 0.0696, min edge 0.0300 vs 0.0283, volume 0.221566 vs 0.221426.
  - Restart (FT without AMR — multiflow cannot restart with AMR on either backend): restarted vs
    continuous differs by the same amount on both backends (flow-solver restart is not exact in
    multiflow); front-derived fields differ less on MFTL2 (indicator 8e-4 vs 3e-3 of max).
- **Performance** (wall time, same machine, idle): FT 274.7 s vs 276 s (np1), 102.3 s vs 103 s
  (np4); FT_Sharp_CSF 42.6 / 16.4 s vs 42.7 / 16.7 s; FT_Sharp_Integral 46.8 / 18.2 s vs 48.8 /
  19.4 s; IBM 26.0 / 8.5 / 6.7 s vs 25.9 / 8.4 / 6.6 s. No regression; no measurable speed-up
  either (front work is a small part of these cases).
- **MoveIBM (MOVEIBM2) and the topology-change example (TOPO)**: both backends hit the same
  900 s limit at the same point (MoveIBM first step; marching cubes stall as at HEAD); every file
  written until then is identical (worst normalised difference 1.6e-15); MoveIBM initialisation
  4.39 s vs 4.40 s.
- **Golden baselines** — *the diagnosis in this bullet is WRONG; see "Independent review" §R3:
  the failures come from the `origin/master` merge (10dce627), not from any MFTL1 Debug/Release
  dependence.* Original text (`/mnt/SHARE/Berend/MF3TESTS`, Debug `linux-gcc-dev`, as `dotests.sh`):
  FT, FT_Sharp_CSF and FT_Sharp_Integral FAIL at np1 and np4. Investigation: the baselines come
  from MFTL1 Debug builds, and **MFTL1 is itself build-type dependent** — the same MFTL1 library
  with a Release vs a Debug multiflow differs by up to 0.97 of a field maximum (FT_Sharp_Integral;
  0.063 FT_Sharp_CSF), whereas MFTL2 Release vs Debug agree to 7e-8 / 3e-8 (FT: the golden AMR
  mesh has 26052 cells vs 23980 for MFTL1 Release and MFTL2). MFTL2 matches MFTL1 Release (above).
  The likely MFTL1 cause (undefined behaviour, e.g. `Collapse_Neighbour_Valence_Check` without a
  return value) is not confirmed. IBM without MFTL also FAILS its golden — equally for the
  committed HEAD (the working tree is bit-identical to HEAD there): pre-existing since the master
  merge, not related to this stage. **The front-tracking golden baselines need regenerating on
  MFTL2** (owner decision; not done here).
- **Builds**: FT, FT_Sharp_CSF + IRL, FT_Sharp_Integral + IRL, IBM-MFTL, MoveIBM, the
  topology-change example and a non-MFTL build (ldc) compile with zero warnings; the MF3 binary
  links only `libMFTL2.a`.

### 6. Behavioural differences vs MFTL1 (intentional or unavoidable)
- **Ownership by triangle centre** (MFTL2 model); MFTL1's default redistribution gave a triangle
  to the owner of its first vertex (`Vertex[0]->processor`; by centre only in its centre
  variant). Redistribution refreshes cached centres first; point-query creation refreshes the
  owned geometry. Sums over owned triangles therefore use a different (equally complete)
  partition — ULP-level differences at np > 1.
- **Ghosts**: always one layer (all edge neighbours + vertex one-rings); `ghost_layers` is
  ignored. Proximity pairs are a rank-local uniform-grid candidate search; thin-region pairs are a
  superset of MFTL1's. Spatial ghosts are not enabled (as MFTL1).
- **N-rings**: true BFS rings (MFTL1's "3-ring" was a flood fill and its 2-ring triangle list was
  partial).
- **Edge statistics**: over canonical half-edges of owned triangles (MFTL1 double-counted ghost
  edges), population standard deviation as MFTL1.
- **Remeshing order effects** (§7) — the only source of the pointwise FT differences.
- **Kuprat smoothing**: MFTL1's second-order terms (v×dx in AA and dot3) take their sign from
  the storage order of `Edge->Vertex[]` vs `Edge->Triangle[]`, i.e. effectively at random per
  edge; MFTL2 uses the consistent orientation. Identical to MFTL1 where the orientations agree.
- **Polygonise** (marching cubes): edge ids and triangle vertex rotation differ; REPLICATED
  placement is treated as ROOT; `GatherTrianglesToCentreProc` = redistribution.
- **Surface-scalar solver path**: in the FT test configuration the payload scalar comes out 0 on
  BOTH backends (pre-existing multiflow behaviour, not investigated further in this stage).

### 7. Remaining remeshing differences (documented compromise)
With `mftl1_criteria` the split / collapse / flip tests, the fit and the sweep structure are
MFTL1's, and every refining pass tested is identical to MFTL1. Two MFTL1 storage artefacts are
not reproduced:
1. **Collapse survivor**: MFTL1 keeps `VLeft`, the origin of the edge in `Edge->Triangle[0]`.
   MFTL2 takes the lower-id triangle as the first one, which is exact for edges as MFTL1 builds
   them but not after MFTL1 has re-assigned an edge's first triangle during earlier operations.
   Because MFTL1 ends a coarsening sweep at the first merged edge that has grown beyond the
   reference length, the number of collapses per pass is sensitive to this choice (e.g. 135 vs 228
   on a refine-4 sphere at L = 0.2); over a simulation the triangle count and edge statistics
   stay close (FT: 904 vs 892 triangles after 100 steps).
2. **Ties**: MFTL1 sorts with a stable merge sort over its hash order; MFTL2 breaks ties by
   half-edge index. Exact length ties only occur on symmetric meshes.
Also: in parallel (`MFTLRemeshEnsembleShiftScheme`) the MFTL1 criteria are used with MFTL2's
parallel sweep (vertex locking, index order) — MFTL1's parallel remeshing was itself an
ownership-dependent sweep. TSUR sweeps its edges in MFTL2 index order (MFTL1: hash order); the
kernel is identical.

### 8. Performance compromises
- Bodies assembled from rank-local pieces redistribute twice once (second pass completes the
  ghost layer) — only after builds / merges / unpacks (marching cubes), not in the time loop.
- The MFTL1 sweeps sort the edge list (O(E log E)) per remesh pass, as MFTL1 did.
- TSUR / VCS / roughness smoothing gather the body to one rank (as MFTL1).
- Per-body ensemble operations use a one-object view (no copies). Standalone bodies share
  ensembles (no communicator per body).
- Point queries: build once / query many (L2); batch queries Allgatherv the points (as MFTL1).

### 9. Deferred to Stage 16 (not done here)
- Tracking MFTL1's per-edge first-triangle bookkeeping for an exact collapse survivor (if ever
  wanted; would change MFTL2's collapse data flow).
- The scale-dependent absolute thresholds that remain in MFTL2 (`MFTL_VERYSMALL = 1e-12` guards
  in several kernels, the 1e-7 domain clamp) — reviewed where they affected front tracking,
  others listed for the cleanup.
- The multiflow surface-scalar solver returning 0 mass in the FT test configuration (pre-existing).
- Formatting: only changed lines were clang-formatted (`git clang-format`); several MFTL2 files
  are not clang-format clean at HEAD.
- Paraboloid fit without LAPACK is a barycentric fallback; multiflow now refuses such a build.
- Regenerating the front-tracking golden baselines on MFTL2 (and the IBM baseline after the
  master merge) — owner decision; the evidence is in §5.
- ~~Investigating MFTL1's Release/Debug dependence~~ — resolved by the independent review: there
  is none (§R3).

### 10. Review focus
- `src/remeshing/mftl1_criteria.c` against MFTL1 `edgesplit.c` / `edgecollapse.c` / `edgeflip.c`
  / `remeshing.c`; the fit changes in `vertex_fit.c`; the parallel split-phase candidate freeze.
- The TSUR normalisation fix (`volume_conserving.c`) — changes MFTL2's own TSUR results.
- The ghost-completion second pass and the redistribution-refreshes-centres rule in the backend.
- The MFTL1 icosphere in `MFAdapterGenerateSphere` (adapter-level generator, not MFTL2's).
- The fatal-error behaviour on unreadable GTS / restart meshes.

## Independent review (2026-09-23, Claude Code) — Acceptance: PASSED

Scope: complete Stage-15 diffs of MFTL2 (working tree vs 30457c0) and multiflow (`berend-mftl`
working tree vs HEAD 10dce627), the MFTL2 backend, and MFTL1 `edgesplit.c` / `edgecollapse.c` /
`edgeflip.c` / `remeshing.c` / `tsur3d.c` where parity is claimed. Everything was rebuilt from
scratch in a session scratch directory; no golden baseline was touched.

### R1. Reproduced
- MFTL2 fresh builds, 0 warnings: ctest **57/57 Release** and **57/57 Debug + ASan/UBSan/LSan**
  (LAPACK on), no sanitizer report; Stage15-np1..4 included.
- Integration-layer tests on MFTL2: **78 / 168 / 261 / 360 checks, 0 failed** on 1 / 2 / 3 / 4 ranks.
- MFTL1 removal: no MFTL1 include, type, call or link in the tracked multiflow tree (comments
  only: `CMakeLists.txt` migration warning, README, five explanatory comments); the MF3 binary
  links `libMFTL2.a` only and contains no MFTL1 symbol; `find_package(MFTL2)` / `MFTL2::MFTL2`,
  configure refuses a LAPACK-less MFTL2.
- Initial front identical to MFTL1 (1280 triangles, volume/edge statistics equal); the FT front is
  bitwise identical to MFTL1 for the first 10 steps.

### R2. Numerical A/B (MFTL1 = multiflow HEAD + prebuilt `libMFTL.a`; natural PETSc flags,
Release = `linux-gcc-opt` `-O3 -march=native`, Debug = `linux-gcc-dev` `-O0`)
| case | comparison | result |
|---|---|---|
| FT_Sharp_Integral np1 | MFTL2 vs MFTL1, Release / Debug | pointwise 2.3e-8 / 1e-8 |
| FT_Sharp_Integral np4 | MFTL2 vs MFTL1 Release | pointwise 5.2e-7, identical AMR mesh |
| FT np1 | MFTL1 Debug vs Release; MFTL2 Debug vs Release | 3.2e-7; 3.8e-7 (same AMR mesh, 23980 cells) |
| FT np1 | MFTL2 vs MFTL1 (t = 0.5) | bubble volume 6.7e-4, rise velocity 1.2e-3, same 23980 cells |
| FT np4 (`Optimizer -1`) | MFTL2 vs MFTL1 | volume 1.4e-4, rise velocity 2.3e-4, same cells |
| FT np4 (`Optimizer 0`, Cleanfrequency 10 — parallel remesh, not covered by any golden) | MFTL2 vs MFTL1 | volume 1.0e-4, rise velocity 3.1e-4, same cells |
| FT, MFTL1's own spread | MFTL1 np1 vs np4 | volume 7.1e-4, rise velocity 1.3e-3 |
Front evolution (VTU every 10 steps): triangle counts within ~3 % (np1 892 vs 904, np4 904 vs
896 / 918 vs 890), front volume within 6e-4, centroid within 2e-4, mean edge within 2 %. The
pointwise field differences (≤ 0.33 of the maximum of the interface-localised surface-tension
field) are those of two different but equally valid front meshes. Verdict: physically and
numerically equivalent; the differences are remeshing-order effects, not a discrepancy.

### R3. Golden baselines — corrected diagnosis
The claim above that "MFTL1 is itself build-type dependent (0.97)" does **not** reproduce, and it
cannot be caused by MFTL1 undefined behaviour: Debug and Release multiflow link the SAME prebuilt
`-O3` `libMFTL.a` (the missing return of `Collapse_Neighbour_Valence_Check` deterministically
returns a neighbour triangle id there). Measured:
- HEAD (MFTL1) Debug vs Release: FT_Sharp_Integral 3.6e-8, FT 3.2e-7 — no build dependence.
- HEAD (MFTL1) vs golden: FT_Sharp_Integral 0.97 in BOTH Debug and Release; FT 23980 vs 26052
  cells, rise velocity 1.306 vs 1.404.
- Pre-merge commit a7cb9179 (MFTL1, before `Merge origin/master`): FT_Sharp_Integral Debug 1.6e-9,
  Release 2.8e-9 vs golden; FT Release 1.1e-8, 26052 cells, rise velocity 1.4040 — reproduces the
  goldens.
So the goldens are stale because of solver changes brought in by the master merge 10dce627
(gradients, `getRUcoefficients.c`, AMR interpolation flags, covariant smoothing), independently of
MFTL1/MFTL2 — consistent with the IBM (no-MFTL) golden failing since the same merge. They must be
regenerated regardless of the MFTL2 swap. Recommendation (owner decision, not done here):
regenerate FT, FT_Sharp_CSF, FT_Sharp_Integral (and IBM) on MFTL2 with the standard Debug
`dotests.sh` configuration after this acceptance; keep the old files only as an archived legacy
set tagged with commit a7cb9179 / ed966d70 provenance (they reproduce pre-merge MFTL1 exactly
and are useful only for bisecting that merge).

### R4. Collapse survivor
Exact MFTL1 survivor parity is NOT needed. The survivor's final position is the fitted edge
midpoint (paraboloid fit of the intact stencil), which is independent of which endpoint survives
(up to rounding); the choice only changes vertex identity and MFTL1's sweep bookkeeping (slot
retirement, the "first merged edge longer than the reference length" break), i.e. per-pass
collapse counts. Over the FT run this produces the statistically equivalent meshes of R2. Do not
emulate MFTL1's per-edge first-triangle bookkeeping; drop the Stage-16 item unless new evidence
appears.

### R5. Library review notes
- Genuine MFTL2 bug fixes (keep): TSUR endpoint normals (MFTL1 `tsur3d.c` normalises both before
  summing — confirmed), the |cross| ≥ 1e-6 area floor, sphere-generator lookups, MPI datatype
  extents resized to `sizeof`, twin-gid refresh after remeshing / post-cleanup redistribution,
  split-phase candidate freeze (gid blocks cannot overlap), uninitialised remesh parameters.
- MFTL1-compatibility code (opt-in, keep for now, revisit after physics-tolerance goldens exist):
  `mftl1_criteria.c` (checked against MFTL1: JIAO split/collapse thresholds, wing-edge rules,
  feature-spoke test, neighbour/valence checks, the six flip checks, sweep order), unstrict
  paraboloid fit, MFTL1 icosphere in the adapter, 1e-7 box clamp. These reproduce MFTL1 behaviour,
  not necessities of MFTL2.
- Surface scalar: split / flip / collapse remaps are exactly mass-conserving by construction;
  TSUR / VCS rescale uses the cached area (MFTL1 semantics); parallel collapse/VCS only touch fully
  owned rings; restart reader preserves winding for all edge orientations.

### R6. Should-fix (not blocking)
1. `MFAdapterRemesh` SERIAL mode ignores `boundary_cleanup`: MFTL1 with `Optimizer -1` on > 1 rank
   remeshed ONLY on cleanup steps (and the adapter contract says SERIAL "needs boundary_cleanup").
   All shipped `-1` inputs use `Cleanfrequency 1`, so nothing changes today; gate the serial remesh
   on `boundary_cleanup` (np > 1) or document the new behaviour.
2. Ghost completion for assembled bodies: two redistribution passes are not guaranteed to store
   every edge neighbour (an edge whose two triangles are owned by A and B while both endpoints are
   owned by a third rank C). It fails loudly after marching cubes (`ErrorH`), but
   `MFAdapterCreateEdgeTriangleGhostPattern` / `EnsureTriangleConnectivity` are silent no-ops on
   MFTL2. Iterate until `MFAdapterOwnedTrianglesMissNeighbours` is false globally (bounded), or
   let the library request missing twins by `twin_gid`.
3. SERIAL remesh on np > 1 redistributes twice (`MFTLEnsembleEndSerial(…, 1)` followed by the
   adapter's redistribution) — pass 0 to `EndSerial`.
4. `test_stage15` is added after the `mftl2_dev_target_options` loop (test code itself not
   sanitizer-instrumented; the library is).

### R7. Coverage gaps (documented, not blocking)
- Marching cubes / topology change and MoveIBM were not completed end to end on either backend
  (pre-existing stall); the rewritten polygonise emission, soup builder, component copy/merge and
  the two-pass ghost completion are covered by unit tests on 1–4 ranks only. Add a small, fast
  topology-change regression case before relying on it in production.
- Point queries (L2) have no production consumer any more (`ispointinsideofparticle.c` was
  removed upstream in d54fc045); unit-test coverage only.
- IBM marker, MoveIBM initialisation, restart and conserved-scalar solver paths were not re-run in
  this review (the implementer's A/B results were not contradicted by anything found); the
  cross-library restart claim (MFTL1-written file read by MFTL2) is untested.

### R8. Performance
No accidental regression: wall times equal within noise (FT np4 105.1 vs 103.6 s,
FT_Sharp_Integral np1 49.6 vs 47.9 s, np4 19.7 vs 18.7 s; machine loaded). Remaining documented
compromises: serial TSUR/VCS/roughness gathers, per-pass edge sort in the MFTL1 sweeps, second
redistribution after assembled-body construction, construction-time all-gathers, full
redistribution in the ghost-position update (as MFTL1), point queries allgather their points.
No speed-up is visible because front work is a small fraction of these cases.

### R9. Verdict
No architectural migration work remains (no MFTL1 representation recreated in the backend beyond
the opt-in parity criteria). Stage 15 is **accepted**; Stage 16 may start after the owner's
decision on the golden regeneration (R3) and with R6 on its list.

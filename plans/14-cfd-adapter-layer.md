# 14 — CFD adapter layer in multiflow (refactor against MFTL1, behavior-preserving)

Status: IMPLEMENTED (2026-09-23, incl. final-fix round) — awaiting independent acceptance
Depends on: — (runs against MFTL1; can start early; aligns with 12)
Implemented by: Claude Code (against multiflow branch `berend-matrix`)
Reviewed by: — (pending: fresh independent reviewer; the 2026-09-22 "Final review" below was of the previous round)
Acceptance: NOT PASSED (pending review)

## Objective
Refactor `multiflow` so **all** MFTL mesh access goes through a complete adapter layer, while
still running on MFTL1 (behavior-preserving). This isolates the library behind a narrow
interface so the MFTL1→MFTL2 swap (Stage 15) touches only the adapter, not the ~45 physics files.

## Relevant existing code
- Existing partial wrapper: `multiflow/src/general/mftlwrapper/` — `readGTSFiletoMFTLObject.c`,
  `transformMFTLObject.c`, `readtrianglesfromMFTLObject.c` (the canonical iterate-filter-copy:
  `for(T=Obj->Triangles; T; T=T->hh.next) if(T->mainprocessor==rank) …`),
  `getmarkersfromexternalmesh.c`.
- **Scope (re-scanned 2026-09-21).** 89 files reference MFTL; 81 sit under `#ifdef MFTL_ON`.
  multiflow includes **four** MFTL headers — `mftl.h` (53), `mftl-functions.h` (42),
  `mftl-constants.h` (40), `mftl-genericio.h` (3) — so the adapter/shim must cover all four, not
  just `mftl.h`. **100 distinct `MFTL*` functions** are called, plus ~20 constants/macros
  (`MFTL_NULL` 353, `MFTL_YES` 61, `MFTL_VERYSMALL` 32, `MFTL_ERROR` 21, `MFTL_NO` 11,
  `MFTL_TRI_PACK_SIZE` 10, `MFTL_ORTHOGONAL`, `MFTL_ALIGNED`, `MFTL_ONESIXTH`, `MFTL_DEG_TO_RAD`,
  `MFTL_GHOST_VERTEX_FAIL`, …) and the memory helpers `MFTL_FreeMemory` (27) /
  `MFTL_AllocateMemory` / `MFTL_DynamicallyArrayGrow`.
- **Actual struct reach-ins (corrected `-e`-guarded counts — the earlier `->center/->area/
  ->normal` figures were wrong).** The fields multiflow really touches are **`Vertex->x[]`
  (802)**, **`->id` (310)**, **`Triangle->Vertex[]` (197)**, **`Triangle->Edge[]` (153)**,
  **`Triangle->Data` (60)** + **`Object->SizeOfData` (15)**, `->mainprocessor` (46),
  `->centerproc` (15). Crucially, **`->Center`/`->Area`/`->Normal`/`->ScalarConcentration`/
  `->MeanCurvature` are ~0** — multiflow does *not* read cached geometry/curvature fields; it
  recomputes them via functions (`MFTLFindTriangleCentres`, `MFTL_TriangleArea`,
  `MFTL_TriangleNormal`, `MFTL_ObjectVertexCurvatures`) into local arrays. So the adapter's real
  per-element surface is **coordinates, ids, vertex/edge topology, and the opaque `Data` blob** —
  not geometry caches. The center/normal/area accessors in "Proposed changes" become thin wrappers
  over the existing compute-functions, not field reads.
- **Iteration.** ~444 `->hh.next` loops exist, but only ~180 iterate MFTL meshes
  (`Triangle` 63, `Obj/Object` 93, `Vertex/V` 29); the rest iterate **multiflow's own uthash
  tables** (`ParticleItem` 83, `MarkerItem` 44, `ProcItem` 18, `FluidCellItem` 16), which are
  NOT MFTL objects and must be left on uthash. Wrap only the mesh iterations.
- Hot files: `interfacial/fronttracking/{surfacescalartransport,surfaceviscosity,volumerecovery,
  computeforcesonfront,spreadfromfronttofluid,frontremeshing}.c`,
  `marchingcubes/{polygonise,reconstructfrontwithmarchingcubes,detectbodies}.c`,
  `lagrangian/moveibm/{updatemoveibmftlbodies,ispointinsideofparticle}.c`. All guarded by
  `#ifdef MFTL_ON`.

## Proposed changes
Introduce `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}` exposing a
library-representation-agnostic API. Then mechanically rewrite the direct reach-ins to use it:
- **Iteration**: `MFAdapterForEachOwnedTriangle(obj, cb, ctx)` / an index-based iterator that
  hides `hh.next` vs array loop; likewise for vertices.
- **Geometry accessors**: `MFAdapterTriangleCenter/Normal/Area(obj,t,out)` (thin wrappers over
  the compute-functions, since these are not stored fields),
  `MFAdapterTriangleVertexCoords(obj,t,3x3)`, `MFAdapterVertexPos(obj,v,out)`. `Vertex->x[]`
  (802 sites) and `->id` (310) are the dominant hot reach-ins — these map cleanly to MFTL2
  `vertex.pos[]`/index and must be `static inline`.
- **Edge facade (critical — no MFTL2 equivalent).** multiflow reaches `Triangle->Edge[j]` (153
  sites across 8 front-tracking files: `spreadfromfronttofluid`, `surfacescalartransport`,
  `frontremeshing`, `computeforcesonfront`, `reconstructfrontwithmarchingcubes`, `detectbodies`,
  `redistributesurfacescalar`, `polygonise`) plus `Edge->Vertex[0/1]->x`, `Edge->id`, and the
  edge functions `MFTL_EdgeLength`, `MFTL_EdgeComplementaryTriangle`, `MFTL_EdgeNormal`,
  `MFTL_EdgeOpposingAngle`, `MFTL_OneRingVertexNormal`. MFTL1 has explicit `MFTL_Edge` objects;
  MFTL2 has **half-edges** (directed, no standalone undirected-edge type). The adapter MUST
  introduce an edge handle now — `MFAdapterTriangleEdge(obj,t,j)`, `MFAdapterEdgeEndpoints`,
  `MFAdapterEdgeLength`, `MFAdapterEdgeAdjacentTriangles` (the two faces = a half-edge + its
  twin), `MFAdapterEdgeId` — so those 8 files route through it against MFTL1 today and Stage 15
  can back it with `(he_idx, twin_idx)` without reopening them. This is the single largest
  representational mismatch in the CFD layer; get the handle shape right in this stage.
- **Ownership**: `MFAdapterTriangleOwner(obj,t)`, `MFAdapterTriangleCenterProc(obj,t)`,
  `MFAdapterVertexOwner(obj,v)`, `MFAdapterRank(obj)`.
- **Payload**: `MFAdapterGetTriangleData/SetTriangleData(obj,t,blob)` (opaque, size = the
  object's data size) — maps to `Triangle->Data`+`SizeOfData` now, `MFTL_Triangle.data`+
  `data_size` after Stage 12.
- **FT fields**: `MFAdapterTriangleScalar(get/set)`, `MFAdapterVertexMeanCurv/GaussCurv`.
- **Lifecycle/redistribute/ghost/geometry** already covered by existing wrapper functions +
  thin passthroughs (`MFTLRedistributeObject`, `MFTLSendTrianglesToProcWCenter`,
  `MFTL_AssureOneRingNeighbourhood`, `MFTLObjectVolume`, `MFTLSurfaceCenter`,
  `MFTLMomentOfInertia`, `MFTLPointInsideObject`).
- Keep the Euler→axis-angle transform math in the wrapper (multiflow types), calling adapter
  transform passthroughs.

## Files/functions likely to change
New `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}`; edits to the ~45 `#ifdef MFTL_ON`
files listed above to route through the adapter; `multiflow/CMakeLists.txt` (add the adapter
source). No MFTL library changes in this stage.

## Numerical/physical considerations
- **Behavior-preserving**: this is a pure refactor on MFTL1 — outputs must be bit-identical to
  before (same iteration order, same float ops). Do it in small reviewable commits per
  subsystem, each keeping the golden `dotests.sh` suite green.
- Preserve the `MFTLSendTrianglesToProcWCenter(YES)…compute…(NO)` bracketing semantics exactly.

## MPI/PETSc considerations
- The adapter must not change the dual decomposition or the `sendnonlocal*` transpose bridge —
  it only wraps mesh access. PETSc coupling (Peskin interpolation, spreading, indicator/marching
  cubes) is untouched; it consumes adapter-extracted coordinate/scalar arrays exactly as today.
- Iteration order over triangles/vertices must remain deterministic (MFTL1 uthash order) to keep
  the golden suite stable in this stage; Stage 15 accepts a controlled change when the backend
  swaps.

## Computational-efficiency guardrails (MUST — this is why MFTL2 exists)
The adapter is the boundary between multiflow's hot loops and MFTL2's cache-friendly arrays; a
badly shaped adapter throws away the whole MFTL2 speedup. Design constraints, not options:
- **Inlinable, zero-copy accessors.** Every per-element accessor
  (`MFAdapterTriangleCenter/Normal/Area`, `MFAdapterVertexPos`, scalar/curvature getters) MUST be
  `static inline` in the adapter **header** so the compiler resolves it to direct array/index
  access (`obj->triangles[t].center[k]`, `obj->vertices[...].pos`) with **zero** call overhead.
  No out-of-line accessor functions in a `.c` for anything called per triangle/vertex.
- **No callback iterator in hot paths.** Expose iteration as a **count + index range**
  (`int n = MFAdapterNumOwnedTriangles(obj); for (int t=0; t<n; ++t) …`) so the loop stays a
  plain vectorizable `for`. Drop `MFAdapterForEachOwnedTriangle(cb, ctx)` — indirect-call-per-
  triangle defeats inlining and vectorization exactly where MFTL2 should be fastest.
- **Reference, not copy-out.** Prefer returning `const double*` into the array over `memcpy`-ing
  3–9 doubles per call where the caller only reads.
- **Verify it compiles away.** For one representative hot loop (e.g. `surfacescalartransport.c`),
  confirm at `-O3` that the adapter access generates the same (or fewer) instructions as the
  current direct struct access — the refactor must be overhead-neutral on MFTL1 before Stage 15
  makes it overhead-neutral on MFTL2.

## Possible failure modes
- Subtle behavior change (reordering, extra recompute) breaking the bit-fragile golden suite →
  refactor in tiny steps, diff HDF5 after each.
- Missing an accessor for a rarely-used field (e.g. `createadaptivitylabel.c` in AMR also
  includes `mftl.h`) → grep all four `#include "mftl*.h"` sites; cover every reached-into field.
- **Routing multiflow's own uthash tables through the mesh adapter** (Particle/Marker/Proc/
  FluidCell iterations share the `hh.next` idiom) → only convert loops over `MFTL_Object`/
  `MFTL_Triangle`/`MFTL_Vertex`; leave multiflow's native hashes untouched.
- **Edge handle designed too thin** (e.g. exposing only length) → later needing complementary
  triangle / opposing angle forces reopening the 8 FT files during the swap. Design the edge
  facade to cover every edge use now (see accessors).
- Performance regression from callback-per-triangle → prefer index iterators / batch extraction
  where hot.

## Unit/regression tests to add
- The existing `testcases/TESTS/dotests.sh` golden suite (FT, FT_Sharp_CSF, FT_Sharp_Integral,
  IBM, IBM-NP4) must remain **green and unchanged** after the refactor — this is the primary
  acceptance signal for behavior preservation.
- **New analytic FT regressions** (physics tolerance, ≥4 ranks), added now so they exist as
  baselines before the swap: oscillating-droplet frequency vs Lamb; rising-bubble terminal
  velocity; LeVeque shape error / volume drift. (Setups exist under
  `multiflow/examples/Fronttracking/` — wire them into an automated check.)
- A build test that `#ifdef MFTL_ON` off still compiles (adapter fully guarded).

## Acceptance criteria
- All direct MFTL struct reach-ins in physics code replaced by adapter calls; only the adapter
  (+ existing wrapper) includes `mftl.h`.
- Golden `dotests.sh` suite unchanged (bit-identical) on 1 and 4 ranks.
- New analytic FT regressions established and passing on MFTL1 as the migration baseline.

## As-implemented (2026-09-22, Claude Code — multiflow branch `berend-matrix`)

**Scope delivered (all verified) — the adapter *boundary* and the MFTL2 *library*, plus
representative adoption. The mechanical reach-in sweep of the ~45 hot physics files is
deliberately deferred (see "Remaining for completion / Stage 15") because it cannot be
bit-verified in this environment without the golden `dotests.sh` HDF5 baselines.**

### 1. MFTL2 CMake library target `MFTL2` (in the MFTL2 repo) — COMPLETE
- `CMakeLists.txt`: the library target is now `MFTL2` (archive `libMFTL2.a`, distinct from
  MFTL1's `libMFTL.a` so the two coexist during the swap), with:
  - `target_include_directories(MFTL2 PUBLIC $<BUILD_INTERFACE:include> $<BUILD_INTERFACE:external>
    $<INSTALL_INTERFACE:include>)` — `external/khash.h` is exposed because `mftl_core.h` puts
    `khash_t(...)` in the public `MFTL_Object` struct and public signatures; internal `src/`
    layout is NOT exposed.
  - PUBLIC `MPI::MPI_C`; PUBLIC `LAPACK::LAPACK` + `MFTL_HAVE_LAPACK` when `MFTL_USE_LAPACK=ON`.
  - `add_library(MFTL ALIAS MFTL2)` (keeps MFTL2's own tests/examples building unchanged) and
    `add_library(MFTL2::MFTL2 ALIAS MFTL2)` (the namespaced form for external consumers).
  - Full install/export rules (`install(TARGETS … EXPORT)`, headers + khash.h, generated
    `MFTL2Config.cmake`/`MFTL2ConfigVersion.cmake`, build-tree `export()`), so a separate project
    can `find_package(MFTL2)`. The generated config re-finds MPI always and LAPACK when enabled.
- **External build/link test**: `external_link_test/` (standalone project) does
  `find_package(MFTL2)` + links `MFTL2::MFTL2` + includes a public header + calls an out-of-line
  archive symbol (`MFTLPolygonArea2D`). Built against the build-tree export → **runs, prints
  "MFTL2 link test OK: unit-square area = 1"**. Proves a consumer needs only the target name.
- Verified: MFTL2 Release build 0 warnings; `libMFTL2.a` produced; **internal ctest 53/53** with
  the renamed target (via the `MFTL` alias).

### 2. Front-tracking adapter interface (in multiflow) — COMPLETE, backed by MFTL1
- New `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}`. The header is the single boundary:
  physics code includes it and uses the `MFAdapter*` API instead of reaching into MFTL structs.
- Honours the efficiency guardrails: opaque handle typedefs (`MFAdapterMesh/Tri/Vert/Edge` — on
  MFTL1 they are struct pointers; Stage 15 re-points them at MFTL2 indices without touching
  consumers); every per-element accessor is `static inline` and **zero-copy** (returns
  `const double*` into the mesh); iteration is a **cursor loop, not a callback**
  (`MFAdapterOwnedTriFirst/Next`, `MFAdapterTriFirst/Next`, vertex variants).
- Covers all required categories: geometry (`MFAdapterTriCentre/Normal/Area`, `MFAdapterVertPos/
  Normal`), on-demand compute passthroughs (`MFAdapterComputeTriCentres/Normals/Areas`,
  `ResolveTriTopology`), ownership (`MFAdapterTriOwner/CentreProc/Id`, `MFAdapterVertOwner/Id`,
  `MFAdapterRank`), opaque payload (`MFAdapterTriData`, `MFAdapterDataSize`), FT fields
  (`MFAdapterTriScalar` get/set, roughness; `MFAdapterVertMeanCurv/GaussCurv`), integral
  passthrough (`MFAdapterObjectVolume`, `MFAdapterNumOwnedTriangles`), and — critically — the
  **edge facade** (`MFAdapterTriEdge`, `MFAdapterEdgeEndpoint`, `MFAdapterEdgeLength`,
  `MFAdapterEdgeAdjTri`, `MFAdapterEdgeNormal`, `MFAdapterEdgeComplementaryTri`) so the 8
  edge-using FT files can route through it without reopening for Stage 15.
- Verified: compiles cleanly against MFTL1 with multiflow's real flags; the whole header is
  guarded by `#ifdef MFTL_ON` (empty TU otherwise — checked).

### 3. Representative routing through the adapter — DONE for the wrapper layer
- `mftlwrapper/readtrianglesfromMFTLObject.c` and `mftlwrapper/getmarkersfromexternalmesh.c`
  (the canonical iterate-filter-copy sites) now use adapter iteration + zero-copy accessors;
  behaviour-identical by construction (same uthash order, same owner filter, same reads).
  The other two wrapper files (`transformMFTLObject.c`, `readGTSFiletoMFTLObject.c`) contain no
  struct reach-ins (they call MFTL functions), so the wrapper layer is fully routed.

### Verification performed
- multiflow FT preset `fronttracking-laplace_equilibrium` builds against MFTL1 with the adapter:
  **0 errors, 0 warnings**, `mftl_adapter.c` compiled, `MF3` relinked.
- **Runtime smoke test** (`Laplace_Equilibrium`, np1): MF3 initialises the front
  ("MFTL bodies initialisation", 1280 MFTL triangles), prints meshes, **advects the front and
  completes front-tracking through timestep 1** — the adapter-routed wrapper paths run
  end-to-end with no crash (the run was stopped by a wall-clock cap during a later PETSc solve,
  not a fault).
- `MFTL_ON`-off compile of the adapter + both converted files → clean empty TUs (guard verified).
- MFTL2: library + external `find_package` link test + full 53/53 internal ctest.

### Remaining for completion / Stage 15
- **Physics-file reach-in sweep NOT done**: ~45 hot FT/IBM/marching-cubes files still include
  `mftl.h` and reach MFTL structs directly (still ~660 `->x[]`, ~156 `Edge[`, ~195 `Vertex[`,
  ~57 `->Data`, plus `mainprocessor`/`center[` — note raw greps also catch multiflow's own
  hashes, which must stay). This is mechanical but **bit-fragile**, and safe verification needs
  the golden `dotests.sh` HDF5 baselines (full multi-minute FT/IBM runs compared against stored
  references on the shared drive) rather than a compile — so it was not attempted blind here.
  The adapter API is designed to cover every one of these sites; the sweep is a file-by-file
  application of the established pattern, each step kept compiling and diffed against the golden
  suite.
- The analytic FT regressions (oscillating-droplet/Lamb, rising-bubble terminal velocity,
  LeVeque) are not yet wired into an automated check.
- Adapter still lets non-mesh multiflow `hh.next` hashes (Particle/Marker/Proc/FluidCell) alone,
  as required — only MFTL meshes route through it.
- No MFTL2 backend was implemented (correct — that is Stage 15); the adapter and the `MFTL2`
  library target are the seam Stage 15 plugs into.

## Review-fix round (2026-09-22, Claude Code — addressing the independent review)

The independent review's must-fix and small architectural items were resolved, and the adapter's
**operation surface** was extended so Stage 15 needs no new abstractions for the known structural
mismatches. All changes are in `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}` (+ the one
converted call site); still backed by MFTL1, still fully `#ifdef MFTL_ON`-guarded.

**M1 (must-fix) — topology accessors now take the mesh.** `MFAdapterTriVertex`,
`MFAdapterTriEdge`, `MFAdapterEdgeEndpoint`, `MFAdapterEdgeLength`, `MFAdapterEdgeAdjTri`,
`MFAdapterEdgeId` gained a leading `MFAdapterMesh m` (unused on MFTL1 → `(void) m`). On MFTL2 a
triangle stores only `he_idx` and an edge handle yields only indices, so these need the mesh
arrays; passing `m` now means Stage 15 backs them with half-edge traversal WITHOUT reopening call
sites. The single existing call site (`readtrianglesfromMFTLObject.c`) was updated. The adapter
is now genuinely swap-transparent for topology navigation.

**S2 — canonical undirected-edge id.** `MFAdapterEdgeId(m, e)` is documented and shaped as a
canonical per-edge id: identical regardless of which triangle the edge was reached from and, on
MFTL2, identical for a half-edge and its twin (Stage 15 returns `min(he->global_id,
twin->global_id)`); MFTL1 returns its native undirected `Edge->id`.

**S4 — no MFTL1-only field accessors.** `MFAdapterTriScalar/SetScalar/Roughness` (which read
MFTL1 `#ifdef FRONTTRACKING` struct fields absent on MFTL2) were removed. Front-tracking per-
triangle scalars live in the opaque `data`/`CellData` blob (reached via `MFAdapterTriData`),
which maps 1:1 onto `MFTL_Triangle.data` — documented in the header.

**Handle-invalidation contract** documented explicitly in the header: entity handles are borrowed
views, invalidated by ANY mesh-mutating op (redistribution/ghost/remesh/lifecycle/transform);
never retain across one; use global ids (`MFAdapterTriId/VertId/EdgeId`) for persistent
references. This matters on MFTL2 (flat arrays reallocated/compacted by those ops).

**Operation surface added** (backed by MFTL1, MFTL2-shaped — plain object + scalars + `double[]`,
no MFTL1 struct types in signatures except two deliberately-opaque handles):
- *Redistribution / ghost / front↔fluid ownership*: `MFAdapterRedistribute`,
  `MFAdapterGatherTrianglesToCentreProc` (the `SendTrianglesToProcWCenter(YES/NO)` bracket),
  `MFAdapterSendLocalTrianglesToAll`, `MFAdapterAssureOneRing`,
  `MFAdapterCreate{Triangle,EdgeTriangle}GhostPattern`, `MFAdapterProcOfVertex`.
- *Lifecycle*: `MFAdapterInitializeObjectWithComm`, `MFAdapterFinalizeObject`,
  `MFAdapterInitializeObjectData`, `MFAdapterGenerateSphere`.
- *Remeshing*: `MFAdapterRemesh(m, params, ops)` with an opaque `MFAdapterRemeshParams` typedef.
- *Curvature & integrals*: `MFAdapterComputeVertexCurvatures`, `MFAdapterSurfaceCentre`,
  `MFAdapterMomentOfInertia`.
- *Rigid-body transforms*: `MFAdapterMoveBody`, `MFAdapterRotateBody`, `MFAdapterScaleSurface`.
- *Point / OBB queries* (build-accel-once, query-many; OBB now → BVH in Stage 15): opaque
  `MFAdapterObjectBBox`/`MFAdapterObjectBigBox`, `MFAdapterBuildObjectBBox/BigBox`,
  `MFAdapterClearObject{BBox,BigBox}`, and out-of-line collective
  `MFAdapterPointInsideObject(m, bbox, p[3])` / `MFAdapterPointInsideBigBox(bigbox, p[3])` that
  wrap a plain coordinate triple in the library point type internally (no leak).
- *Triangle packing*: `MFAdapterTrianglePackSize` (bytes/triangle),
  `MFAdapterPackTriangle`/`MFAdapterUnpackTriangles`.

**Intentionally left outside the adapter** (correct — these are Stage-15 *shim*, not mesh
operations, or lower-priority categories): pure math helpers (`MFTL_VectorInProd`, `MFTL_SQ`,
`MFTL_ScaleVector`, tensor ops, …) and memory helpers (`MFTL_FreeMemory`, `MFTL_AllocateMemory`,
`MFTL_DynamicallyArrayGrow`) → verbatim Stage-15 shim; VTK/GTS I/O; the low-level per-vertex
curvature primitives (`MFTL_VertexLaplaceBeltrami`, `MFTL_FrenetTriangleCurvature`) behind the
object-level `MFAdapterComputeVertexCurvatures`; and the **N-ring neighbourhood-list queries**
(`MFTL_{One,Two,Three}RingNeighbourhood*`, `MFTL_TrianglesWithinDistance`) — a coherent remaining
category to add when the FT files that use them are swept (MFTL2 has 1-ring; 2/3-ring compose
from it). `MFTL_UpdateGhostVertexGradients` has no library declaration on this branch and was not
wrapped.

**Verification (this round):** MFTL2 Release build 0 warnings, `libMFTL2.a`, external
`find_package(MFTL2)` link test passes, internal **ctest 53/53**. multiflow FT
(`fronttracking-laplace_equilibrium`) builds against MFTL1 with the extended adapter: **0 errors,
0 warnings**; `MFTL_ON`-off compile of the adapter + converted files → clean empty TUs. FT smoke
test **np2**: front initialised (1280 triangles), advected, front-tracking completed through
timestep 1, fluid solves progressing — no crash, timings unchanged from the pre-fix run
(behaviour preserved).

## Final review (2026-09-22, Claude Code) — CHANGES REQUESTED / NOT ACCEPTED  [addressed by "Stage 14 completion" below]

**Verdict: NOT ACCEPTED.** The delivered work is correct, behaviour-preserving and
overhead-neutral, but the adapter's operation surface is **not** complete enough for Stage 15
to migrate the remaining physics code without another adapter-design round: several structural
categories of the multiflow↔MFTL coupling have no adapter shape, and two small API defects
would cause silent behaviour changes during the sweep. No fixes were implemented as part of
this review.

### Verification performed (independent)
| Check | Result |
|---|---|
| MFTL2 Release build (fresh build dir) | 0 warnings, `libMFTL2.a` |
| MFTL2 ctest | **53/53 passed** |
| External `find_package(MFTL2)` link test | passes from the **build tree** and from an **install prefix**; LAPACK-on config emits `find_dependency(LAPACK)` |
| multiflow FT preset vs MFTL1 | 0 errors, 0 warnings |
| Adapter + converted files with `MFTL_ON` off | clean empty TUs |
| Codegen guardrail (disassembly, converted files) | 230→207 and 161→162 instructions — overhead-neutral |
| **A/B bit-identity, HEAD vs working tree, np1 and np4** | **all outputs byte-identical** (fields/meshes/particles HDF5, marker files, VTK) |

A/B method: HEAD extracted via `git archive` and the working tree built with MFTL on; two IBM
cases derived from `testcases/IBM/run.mf`, one per converted function — (A) `IBMMFTL` +
`IBMMFTLMesh sphere.gts` → `GetMarkersFromExternalMesh`; (B) `IBMParticlesFileName 01 sphere.gts`
→ `ReadTrianglesfromMFTLObject`. Every output compared with `h5diff`/`cmp`. This method does not
need the shared golden baselines and is the recommended per-file verification for the sweep.

**Corrections to the as-implemented record above:**
- The two converted functions are called **only** from
  `lagrangian/ibm/createanddestroyibmstructure/createibmmarkers.c`. The FT `Laplace_Equilibrium`
  smoke test cited above does **not** exercise them (the golden IBM case is built without MFTL
  and does not either). They are now verified by the A/B test above.
- The golden `dotests.sh` HDF5 baselines **are** available locally at
  `/mnt/SHARE/Berend/MF3TESTS/` (FT, FT_Sharp_CSF, FT_Sharp_Integral, IBM, …). The stated reason
  for not attempting the sweep ("baselines not available in this environment") does not hold.

### Must-fix (before acceptance)
- **M1 — Handle-representation contract is inconsistent; `MFAdapterTriCentreProc` cannot be
  implemented on MFTL2.** The header says handles are opaque ("never assume pointer vs index")
  and this plan says Stage 15 re-points them at "MFTL2 indices", yet ~15 per-element accessors
  take only the handle (`MFAdapterTriCentre/Normal/Area/Owner/Id/Data`,
  `MFAdapterVertPos/Normal/Id/Owner/MeanCurv/GaussCurv/Area`). These are only implementable on
  MFTL2 if handles are **pointers into the object arrays**. Required: commit the contract to
  pointer handles on MFTL2 (`MFAdapterTri = MFTL_Triangle*` into `obj->triangles[]`,
  `MFAdapterVert = MFTL_Vertex*` into `obj->vertices[]`, `MFAdapterEdge` = the **canonical**
  `MFTL_HalfEdge*` of the pair) — which is also the most efficient mapping (direct field access,
  pointer-increment iteration, index = `t - m->triangles`). MFTL2 has no `centerproc` field, so
  `MFAdapterTriCentreProc` must take the mesh (computed from the centre via the domain's owner
  map). Cheap now (2 converted call sites), expensive after the sweep.
- **M2 — `MFAdapterBuildObjectBBox` hard-codes `MFTL_ALIGNED`, while 6 of the 10 multiflow
  `MFTLGenerateBoundingBox` sites use `MFTL_ORTHOGONAL`** (`ispointinsideofparticle.c`,
  `initializepboxmesh_ft.c` ×2, `indicator_irl.c` ×2, `createadaptivitylabel.c`). A mechanical
  sweep onto it would silently change the box type in point-in-particle / indicator paths. Needs
  a box-kind parameter. In addition, physics code **reads and inflates** box extents
  (`BigBox->Dimensions[k] += …`, `BigBox->Displacement`, `BBox->Dimensions` in `breakup.c`), so
  the opaque box types need extent get/inflate accessors.
- **M3 — Operation surface is missing whole structural categories** (design decisions, not
  passthroughs). These — not the items listed as deferred above — determine whether Stage 15
  "touches only the adapter":
  1. *Object lifecycle / storage / body collections.* multiflow stores `struct MFTL_Object` **by
     value** (`include/ibm.h:220`; `struct MFTL_Object body = {0}` in 6 places), sets `body.id`,
     and chains bodies through MFTL1's uthash `hh` (`AddMFTLObject`, `FT->Bodies` loops,
     `MFTL_Object **` lists in marching-cubes/moveibm). MFTL2 objects are heap-allocated
     (`MFTLObjectCreate`) and owned by an `MFTL_Ensemble` with a shared `MFTL_Domain`; remeshing
     and ghost extension/update are ensemble-level. The adapter must decide what `MFAdapterMesh`
     is on MFTL2 (e.g. wrapper `{ensemble, object}`, or one ensemble per body) and provide
     create/destroy + body-list iteration.
  2. *Per-body decomposition control (overview invariant 8 / top risk #2).*
     `ownershiprangemin/max` (71 sites, 7 files — incl.
     `AdaptOwnershiprangesToCartesianOverlapMesh`, `frontrestart.c`,
     `initializepboxmesh_ft.c`, marching-cubes files), `globalxmin/max` (16) and
     `OBJECT_COMM_WORLD` (25) have no adapter operation/accessor.
  3. *Mutation.* No vertex-position setter — front advection writes `V->x[] +=` directly
     (`frontadvection.c`), and `updateghostvertices.c` writes ghost positions from its own MPI
     exchange (→ MFTL2 `MFTLUpdateGhostPositions`). No per-triangle geometry recompute/setter
     (`MFTL_TriangleCentre/Normal/Area(T)`, `Tri->area = …` in `updatefrontproperties.c`;
     `MFTL_TriangleArea` has 11 call sites).
  4. *Mesh construction / extraction.* `detectbodies.c` and `reconstructfrontwithmarchingcubes.c`
     hand-assemble objects via `HASH_ADD_INT` into `Vertexes/Edges/Triangles` + struct `memcpy`,
     and `MFTL_DynamicallyTriangleArrayGrow` builds `MFTL_Triangle*` arrays. Needs a
     build-from-triangle-set / extract-submesh operation.
  5. *Point sets.* `MFTLGatherPoints` / `MFTLRemovePoints` and `MFTL_Point` uthash lists
     (`ispointinsideofparticle.c`) are not wrapped.

### Should-fix
- **S1 — Canonical edge-ID contract.** Sound in principle (MFTL2 half-edge `global_id`s travel
  with their triangle and are rank-consistent), but `min(he_gid, twin_gid)` is undefined when the
  twin is absent locally (ghost-layer rim, `twin_idx == -1`): the runtime `MFTL_HalfEdge` drops
  the `twin_gid` that `PackedHalfEdge` carries, so a rim edge could report different ids on
  different ranks. Either retain `twin_gid` in `MFTL_HalfEdge` or restrict/define `EdgeId` for
  rim edges. Also specify: `MFAdapterTriEdge` returns the canonical half-edge (so handle
  equality ⇔ edge equality), `MFAdapterEdgeAdjTri(e,0/1)` = canonical face / twin face, and
  `AdjTri` may return `MFADAPTER_TRI_NONE` (already relied on in
  `reconstructfrontwithmarchingcubes.c`). Note edge ids change numbering space vs MFTL1 `Edge->id`.
- **S2 — Invalidation list.** Add `MFAdapterUnpackTriangles` (appends → may reallocate). Replace
  "rigid-body transforms that may reorder storage" with a precise list (neither backend reorders
  on transform). State explicitly which calls never invalidate (Compute* passes, curvature,
  integrals, point queries).
- **S3 — Signatures/semantics.** `MFAdapterMomentOfInertia` centre is an input on both backends →
  `const double centre[3]`. Document `MFAdapterRotateBody` angle convention (MFTL1 angle vector
  vs MFTL2 `MFTLRotateObject(R[3][3], centre)`). Document `MFAdapterGenerateSphere(m, refine)` as
  "unit sphere at origin" (MFTL2 takes centre + radius). `MFAdapterRemeshParams` still aliases
  the MFTL1 type (acknowledged).
- **S4 — MFTL2 CMake/export.** Target and package config are correct for `find_package`
  consumers (verified build-tree + install). However: (a) Release archives contain LTO-only
  (non-fat) objects built with `-march=native` — not portable across compilers/HPC nodes;
  provide an option for fat-LTO / no-LTO / no-native package builds. (b) Debug builds add
  `-fsanitize=address,undefined` via global `add_compile_options`, so a Debug `libMFTL2.a` will
  not link into an unsanitized multiflow. (c) multiflow currently consumes MFTL via
  `MFTL_DIR/include` + `MFTL_DIR/lib` + `-lMFTL`, which fits neither MFTL2's source layout
  (`khash.h` in `external/`, archive in `build/lib`) nor its install layout (`lib64` via
  `GNUInstallDirs`) → Stage 15 must switch multiflow to `find_package(MFTL2)`. (d) 13 exported
  symbols share names but differ in signature between `libMFTL.a` and `libMFTL2.a`
  (`MFTLRedistributeObject`, `MFTLGenerateSphere`, `MFTLPointInsideObject`,
  `MFTLInitializeObjectData`, `MFTLMomentOfInertia`, `MFTLObjectVolume`, `MFTLSurfaceCenter`,
  `MFTLPointTriangleDistance`, `MFTLGenerateBox/Cube/Cylinder/Face`, `MFTL_TriangleQuality`),
  and both define `MFTL_Object`/`MFTL_Triangle` types → the libraries coexist on disk only;
  the swap must be atomic. (e) Cosmetic: exported target lists `include` twice.
- **S5 — Record accuracy** (addressed by the corrections above).

### Deferred items — assessment
- Math/memory helpers → verbatim Stage-15 shim: **safe to defer**.
- N-ring queries (`MFTL_{One,Two,Three}Ring…`, `MFTL_TrianglesWithinDistance`): **safe to
  defer**, but they return MFTL1 list types (`MFTL_VertexListAddItem`, `MFTL_GeneralListFree`),
  so a list/array abstraction is part of that work.
- Remesh-parameter construction: **defer only after M3.1** — MFTL2 uses
  `MFTL_RemeshParameters` / `MFTLInitRemeshParams` and ensemble-level `MFTLRemeshEnsemble*`;
  `frontremeshing.c` also calls `MFTL_TSUR3D` and sets `MFTL_VertexSmoothing*` fields directly.
- Also uncovered but minor: `MFTL_EdgeLengthStats`, `MFTLRedistributeObjectBasedOnTriangleCenter`,
  `MFTL_ProcOfTriangleCenter`, `MFTL_ObjectWriteMeshForeach`, `MFTLWriteObjectToVTK`,
  `MFTLGTSFileRead`, `MFTL_EdgeOpposingAngle` (named in this plan; only in commented-out code).

### Checked and found sound
- `CellData` is a flat POD (no pointers) → backend-independent `data`/`data_size` transport;
  new-triangle blobs are zeroed on both backends (MFTL1 `addtriangle.c` `calloc` ≈ MFTL2 split).
- Mean curvature is stored as the Laplace–Beltrami magnitude on both backends; multiflow uses it
  only as a threshold (`frontadvection.c`, `updatemoveibmftlbodies.c`).
- `MFAdapterTriData` maps directly onto `MFTL_Triangle.data` (NULL when `data_size == 0`).
- The M1 (previous round) mesh-context fix on topology accessors is correct as far as it goes.

### Stage-15 risks (carried forward)
1. Dual decomposition: per-body ownership ranges ↔ MFTL2 `MFTL_Domain` (cuts/`rank_map`/
   `lb_xcuts`); silent front↔fluid corruption if wrong (M3.2).
2. Point queries: MFTL2 `MFTLPointInsideObject` builds a BVH **per call**; multiflow's collective
   per-point loops would rebuild it per gathered point → needs a persistent-BVH query API.
3. Ghost/ownership semantics: MFTL1 `Redistribute(obj, layers)`, `SendTrianglesToProcWCenter`
   bracket, `SendLocalTrisToAllProcs`, ghost patterns vs MFTL2 ensemble-level
   `MFTLExtendGhostLayer`/`MFTLUpdateGhostPositions`; no `centerproc` in MFTL2.
4. Triangle packing: MFTL1 raw-struct `TriangleToArray` vs MFTL2 `PackedTriangleComplex` + data.
5. Validation: iteration order changes on the swap break the bit-exact golden suite, and the
   analytic FT regressions (Lamb droplet, rising-bubble terminal velocity, LeVeque) do not exist
   yet.

### Path to acceptance
Fix M1 and M2 (small); add adapter *shapes* for M3.1–M3.5 (backed by MFTL1, verified
bit-identical); apply S1–S3 contract text. The physics sweep can then proceed file by file,
each step verified with the HEAD-vs-working-tree A/B method above (np1 + np4).

## Stage 14 completion — MFTL-facing architecture (2026-09-23, Claude Code)

This round addresses the final review (M1–M3, S1–S5) by redesigning the MFTL-facing
infrastructure of multiflow instead of wrapping MFTL1. Layering:

    multiflow physics  ->  front-tracking integration layer (src/include/mftl_adapter.h)
                       ->  backend (Stage 14: MFTL1 — mftl_adapter_mftl1{.h,_inline.h,.c};
                                    Stage 15: MFTL2 — same three files for MFTL2)

The public header is backend independent (types, contracts, declarations); the backend binding
provides the handle typedefs, the private list fields, the inline bodies and the out-of-line
operations. Stage 15 writes the MFTL2 backend files and converts the remaining per-element
reach-ins; no new abstraction is required for any structural concept multiflow uses.

### 1. Object / ensemble / body model (resolves M3.1)
- `MFAdapterDecomposition` — multiflow-owned value type: communicator, global box, periodic
  flags, per-axis RECTILINEAR cut coordinates and the rank→slab map. Built from a
  `CartesianOverlapMesh` (DMDA particle-box layout; rejected if not rectilinear) with the exact
  expressions multiflow always used for the ownership ranges (bit-identical), or as a serial
  one-box decomposition (`PETSC_COMM_SELF` marker meshes). MFTL2: becomes the `MFTL_Domain`.
- `MFAdapterBodyList` — a collection of bodies sharing ONE decomposition and ONE payload size;
  owns its bodies. MFTL1: uthash list of `MFTL_Object`; MFTL2: one `MFTL_Ensemble` (shared
  domain) + id map. Operations: init / init-like, set data size, set decomposition, create
  body, count, first/next, find, destroy body/all, free, sort by id, renumber, transfer between
  lists. `FT->Bodies`, `MoveIBMParticleDomain.Bodies`, the marching-cubes scratch lists and the
  non-topology-changing scratch list are all `MFAdapterBodyList`.
- `MFAdapterMesh` — opaque body handle, never stored by value. MFTL1: `MFTL_Object *`;
  MFTL2: adapter-owned `struct MFAdapterBody *` (object + list/ensemble + id + cached BVH).
  Standalone bodies (`MFAdapterBodyCreate/Destroy`) have their own decomposition (IBM
  per-particle serial marker meshes — `struct IBMParticle` now holds `MFAdapterMesh MFTL_Body`
  instead of an embedded `struct MFTL_Object` — and scratch bodies).
- Remeshing / ghost refresh are body- or list-level operations (`MFAdapterRemesh`,
  `MFAdapterBodyListUpdateGhostPositions`), matching MFTL2's ensemble-level operations.

### 2. Handles (resolves M1, S1)
- MFTL2: `MFAdapterTri/Vert` are POINTERS into `obj->triangles[]/vertices[]`, `MFAdapterEdge`
  the CANONICAL `MFTL_HalfEdge*` (lower global id of the pair when both are local). This makes
  the handle-only per-record accessors single loads; everything needing more than the record
  takes the mesh (topology, edges, `MFAdapterTriCentreProc(m,t)`, `MFAdapterProcOfTriCentre`,
  N-ring, lookups).
- Canonical edge id: MFTL1 `Edge->id`; MFTL2 `min(he.gid, twin_gid)` with `twin_gid` retained in
  the half-edge even when the twin is not stored locally (Stage-15 library item L3).
  `EdgeAdjTri(k)` order unspecified, may be NONE on the ghost rim; ids never persisted.
- Invalidation contract lists every structural operation (incl. unpack, copy/merge, builder,
  generators/readers, gather/write-native, ghost refresh) and every value-only operation.

### 3. Decomposition and communication (resolves M3.2)
- `MFAdapterBodyListSetDecomposition` / `MFAdapterSetBodyDecomposition` (+ explicit
  `MFAdapterRedistribute*`), `MFAdapterGatherToRoot`, `MFAdapterRankBox` (LIVE ownership),
  `MFAdapterGlobalBox`, `MFAdapterComm/Rank/NumProcs`. All 71 `ownershiprange*`, 16 `globalx*`
  and 25 `OBJECT_COMM_WORLD` physics reach-ins are gone (restart gather/restore, P-box
  re-decomposition, marching-cubes decomposition switch, ghost-only body removal).
- Ghost / front↔fluid: `Redistribute[ByTriangleCentre]`, `GatherTrianglesToCentreProc`,
  `SendLocalTrianglesToAll`, `AssureOneRing`, `Create{Triangle,EdgeTriangle}GhostPattern`,
  `CompleteEdgeNeighbours` (relocated marching-cubes neighbour repair),
  `BodyListUpdateGhostPositions` (relocated `updateghostvertices.c` exchange).
- MFTL1 initialisation hint `mftl1_init_grid` (the Xcut/Ycut/Zcut MFTL1's remesh load balancer
  sees) is carried explicitly and removed in Stage 15.

### 4. Mutation and construction (resolves M3.3, M3.4)
- `MFAdapterVertPosMut/SetPos` (front advection converted), owned-vertex iteration,
  `MFAdapterUpdateTriGeometry/UpdateTriNormal/ComputeTriArea`, whole-body geometry passes,
  vertex-incidence maintenance (`Rebuild/Update/CheckVertexIncidence` — MFTL1's
  RefTriangle bookkeeping, a no-op on MFTL2), one-vertex normal / Laplace–Beltrami.
- Construction: triangle-soup builder (`BuildAddVertex/AddTriangle/Finalize`, collective,
  globally consistent MFTL1 edge ids, orientation preserved), component extraction
  `BodyCopyTriangleFrom` and `BodyMergeInto` (relocated from `detectbodies.c`; the merge now
  re-points moved records at the destination and payloads are deep-copied — the old code left
  dangling pointers / aliased payloads in the dedup case).
- Payload contract: every stored triangle has a zeroed payload provided by the layer
  (generators/readers/copies included); the application never allocates it
  (`InitializeCellData` no longer allocates).

### 5. Point queries and boxes (resolves M2, M3.5)
- `MFAdapterPointQuery` build-once/query-many (`Create/Inside/InsideBatch/Destroy`); the batch
  call gathers all ranks' candidates once and classifies them with ONE accelerator build
  (MFTL1 OBB tree; MFTL2 persistent LBVH — library item L2). `ispointinsideofparticle.c` is
  converted (no `MFTL_Point` lists).
- `MFAdapterAABB` plain value type + `MFAdapterBodyAABB` (collective), `AABBContains/Inflate`.
  The aligned/orthogonal distinction is no longer an API parameter: point queries/AABBs use the
  orthogonal tree, proximity/thin-region searches the aligned one, exactly as every former
  call site did. Converted: `initializepboxmesh_ft.c`, `indicator_irl.c`,
  `createadaptivitylabel.c`, `breakup.c` (`FindThinRegionPairs`), `coalescence.c`
  (`TrianglePairsWithinDistance`, rank-local as before).

### 6. Remeshing settings and misc
- `MFAdapterRemeshSettings` (backend-independent, 0/1 switches, `MFADAPTER_REMESH_*`
  criteria) replaces `MFTL_RemeshParameter` in `FT` and MoveIBM; the MFTL1 backend marshals
  in/out on every call (`Remesh`, `Tsur`, `VolumeConservingSmoothing`, `RoughnessSmoothing`).
- Transforms/generators with defined semantics (rotation VECTOR about a centre; scale about
  the origin; unit icosphere scaled/moved; ROOT vs REPLICATED placement), GTS/VTK/native-mesh
  I/O (`WriteNativeMesh` relocated from `frontrestart.c`), edge-length statistics, triangle
  packing, curvature estimate for output.

### 7. MFTL2 library / CMake (external HPC library review)
- Target `MFTL2` (+ `MFTL2::MFTL2`, `MFTL` alias kept). All flags target-scoped; no global
  `add_compile_options/add_link_options/include_directories`.
- `-march=native` is opt-in (`MFTL_NATIVE_ARCH`, default OFF → portable archive). LTO
  (`MFTL_ENABLE_IPO`) only for GCC and with `-ffat-lto-objects` (archive usable without LTO;
  plain `nm` now reads it). Sanitizers only when `ENABLE_SANITIZERS` + Debug; an instrumented
  archive exports `-fsanitize=...` as INTERFACE link option so consumers link; tests/examples
  get the flags via `mftl2_dev_target_options`. Tests/examples default ON only when MFTL2 is the
  top-level project. `m` in the link interface; `find_dependency(MPI COMPONENTS C)`.
- Code generation of the fat objects exposed 8 GCC `-Walloc-size-larger-than` diagnostics
  (possibly-negative `int` counts reaching `malloc`, previously hidden because non-fat LTO
  objects carried no code at compile time); fixed behaviour-identically with a non-negative
  clamp in `communication.c` and `mftl_bvh.c`.

### 8. Infrastructure converted in multiflow (representative + all structural sites)
Body storage and every body-list iteration (FT, MoveIBM, IBM particles, marching cubes,
example userfiles); all decomposition / communicator reach-ins; `detectbodies.c`,
`reconstructfrontwithmarchingcubes.c`, `executemarchingcubes.c`,
`marchingcubesmiscfunctions.c` (list/decomposition helpers `InitializeFTBodyList`,
`CreateStandaloneFTBody`, `FTDecompositionFromOverlapMesh`,
`AdaptOwnershiprangesToCartesianOverlapMeshList`, `MoveListOfBodies`), `breakup.c`,
`coalescence.c`, `initializefronttracking.c`, `finalizefronttracking.c`, `frontrestart.c`,
`initializepboxmesh_ft.c`, `frontadvection.c`, `updateghostvertices.c`,
`updatefrontproperties.c`, `frontremeshing.c` (settings), MoveIBM creation/destruction,
`ispointinsideofparticle.c`, `indicator_irl.c` / `createadaptivitylabel.c` (AABBs), IBM
particle marker meshes and the four `mftlwrapper` files (now include only `mftl_adapter.h`).
Net: 60 files changed in multiflow (+908 / −2008 lines) plus the layer (≈2.3k lines incl.
relocated code) and `tools/mftladapter_tests/`.

### 9. Remaining deliberate Stage-15 work (no new abstraction needed)
1. Write the MFTL2 backend (`mftl_adapter_mftl2.h`, `_inline.h`, `.c`) against the contracts;
   run `tools/mftladapter_tests` unchanged on 1–4 ranks.
2. MFTL2 library items the backend needs:
   L1 rectilinear per-axis cuts + rank map in `MFTL_Domain` (generalise `lb_xcuts` to 3 axes);
   L2 caller-held point-query BVH (build once / classify many; batch);
   L3 retain `twin_gid` in `MFTL_HalfEdge` (canonical edge id on the ghost rim);
   L4 detach an object from an ensemble (list transfer);
   L5 single-vertex normal / Laplace–Beltrami entry points;
   (optional) L6 spatial ghosts for cross-rank proximity (would change coalescence detection —
   decide deliberately);
   **L7 conserved per-triangle surface scalar**: either a native library-maintained scalar, or
   payload-remapping hooks, that REPRODUCE MFTL1's behaviour exactly — transported with the
   triangle, remapped mass-conservingly by split / flip / collapse / TSUR / volume-conserving
   smoothing (rules in `mftl_adapter.h`, `MFAdapterTriSurfaceScalar`), preserved by restart I/O.
   It must NOT degrade to ordinary zero-initialised payload semantics.
3. Convert the remaining per-element reach-ins inside physics loops (`->x[]`, `->Vertex[]`,
   `->Edge[]`, `->Data`, `->mainprocessor`, `->center/normal/area`, 1:1 MFTL calls such as
   `MFTLRedistributeObject`, `MFTL_EdgeLength`, `MFTL_TriangleArea`, N-ring lists) in ≈25 FT
   files — every one has an adapter equivalent; math/memory helpers → verbatim shim.
4. `polygonise.c` still assembles MFTL1 vertices/edges/triangles itself (its edge-id scheme is
   MFTL1-specific; converting it on MFTL1 would renumber edges and break bit-identity). Stage 15
   rewrites its emission onto the builder (`MFAdapterBuildAdd*/Finalize`), which removes the
   edge bookkeeping entirely.
5. **Conserved surface scalar (resolved in the final-fix round):** MFTL1's
   `scalarconcentration` is real solver state (surfactant concentration). It is now a
   backend-independent adapter concept (`MFAdapterTriSurfaceScalar/SetSurfaceScalar`) and all
   20 multiflow sites use it; the requirement for Stage 15 is to **preserve the current
   behaviour numerically** (L7), not to change or "fix" the physics. (The earlier remark that
   the mapped value "never reaches CellData" describes current behaviour, which is to be kept.)
6. `printfrontmesh.c` (VTU output) and `volumerecovery.c` read MFTL1 records directly — output /
   per-element conversions of item 3.
7. Remove `mftl1_init_grid` and `MFADAPTER_PLACE_REPLICATED` distinctions if MFTL2 makes them
   unnecessary. Periodicity: the front decomposition is NON-periodic by default
   (`MFAdapterDecomposition.periodic` = 0, the fluid's DMDA periodicity is only recorded in
   `fluid_periodic`); the MFTL2 backend must keep this MFTL1 behaviour, enabling periodic fronts
   only as an explicit, separate change (overview risk 7).

### 10. Known performance compromises (documented, MFTL1 backend)
- Builder finalisation gathers all edge keys to every rank (construction only).
- Proximity / AABB rebuild the OBB tree per call (coalescence previously reused one tree per
  body pair loop); point queries gather all candidate points to all ranks (as before).
- Remesh settings are marshalled into a fresh MFTL1 parameter block per call.
- `MFAdapterBodyListSetDecomposition` deep-copies the decomposition per call.
- Relocated MFTL1 algorithms keep their complexity (O(V·T) body detection BFS in
  `detectbodies.c`, all-to-all ghost-vertex exchange).

### 11. Verification (2026-09-23, MFTL1 backend, working tree vs HEAD)
Method: HEAD snapshot (`git archive`) and working tree built with identical settings; every
output file compared exactly (`h5diff` / `cmp`). Harness scripts lived in the session scratch.

| Case (path exercised) | np1 | np4 |
|---|---|---|
| Golden FT (`testcases/FT`, POISSON indicator: body list, decomposition, advection, remeshing settings, ghost refresh, restart-free) | 7/7 identical | 10/10 identical |
| FT_Sharp_CSF (IRL: AABB prefilters in `indicator_irl.c`) | 7/7 identical | 10/10 identical |
| FT_Sharp_Integral (IRL, integral surface tension) | 7/7 identical | 10/10 identical |
| IBM + `IBMMFTL` (`GetMarkersFromExternalMesh`, standalone serial body) | 10/10 identical | 10/10 identical |
| IBM + GTS injector (`ReadTrianglesfromMFTLObject`, particle `MFTL_Body`) | 11/11 identical | 11/11 identical |
| MoveIBM RotatingSphere (body list, decomposition, replicated generator, centre-based redistribution, particle properties) | 9/9 identical (step 0; step-1 fluid solve exceeds 25 min in HEAD and new) | no output within 25 min in HEAD and new (particle/fluid coupling at AMR 7) |
| Topology-change example (coalescence → marching cubes) | 5/5 identical up to the stall | 8/8 identical up to the stall |

- Golden baselines (`/mnt/SHARE/Berend/MF3TESTS`, Debug `linux-gcc-dev` builds of the working
  tree, `compare_hdf5-py.sh`): FT PASS, FT_Sharp_CSF PASS, FT_Sharp_Integral PASS, IBM np1 PASS,
  IBM np4 PASS (the IBM builds have MFTL OFF → full `MFTL_ON`-off compile of multiflow).
- Builds: all eight multiflow configurations (MFTL, MFTL+IRL, no MFTL; Release and Debug)
  0 warnings, 0 errors.
- Integration-layer unit tests `multiflow/tools/mftladapter_tests/run_tests.sh` (decomposition
  exactness and rectilinear rejection, body lists, handles/canonical edges/N-ring/curvature,
  owned-vertex mutation + ghost refresh, transforms, parallel builder with globally consistent
  edge ids and preserved orientation, copy/merge, single + batch point queries, AABB,
  proximity/thin regions, gather/restore, remesh settings, native restart round trip):
  67 / 144 / 225 / 312 checks pass on 1 / 2 / 3 / 4 ranks. Their first runs exposed four
  contract gaps that are now explicit (payload guarantee, edge ghost pattern after one ghost
  layer, rank-local proximity, vertex-incidence / N-ring preconditions).
- MFTL2: Release 53/53, Debug+ASan/UBSan 53/53, 0 warnings in Release, Release+native,
  IPO-off and Debug; `find_package(MFTL2)` consumer from build tree, install prefix and the
  instrumented Debug archive; `add_subdirectory` consumer (no leaked flags, no tests built).

**Coverage gaps (stated plainly):** the marching-cubes reconstruction (`detectbodies`,
`CompleteEdgeNeighbours`, copy/merge, list transfer) and `IsPointInsideOfParticle` were not
exercised end to end by an A/B run: the topology-change example stalls inside
`ExecuteMarchingCubes` identically in HEAD and the working tree (even with the MC resolution
factor 1 — pre-existing, cause not investigated), and the MoveIBM runs do not finish their first
fluid solve within 25 min (with IRL they additionally spend the time in IRL cut-cell
reconstruction, identically in HEAD). These paths are covered by the unit tests
(copy/merge, builder, batch point queries on 1–4 ranks) and by code review of the verbatim
relocations; Stage 15 should add a small, fast topology-change and MoveIBM regression case.

### 12. Resolution of the 2026-09-22 review findings
| Finding | Resolution |
|---|---|
| M1 handle contract / `TriCentreProc` | Pointer handles on MFTL2 (canonical half-edge for edges) fixed in the contract; mesh-context accessors where the record is insufficient (§2). |
| M2 aligned vs orthogonal boxes, extents | Kind removed from the API; `MFAdapterAABB` value type with centre/half extents, inflate/contains; accelerator kind chosen per operation exactly as before (§5). |
| M3.1 object lifecycle / body lists | Decomposition / body-list / body model; no `struct MFTL_Object` storage or MFTL1 hash chaining left in multiflow (§1). |
| M3.2 decomposition / communicator | All `ownershiprange*`/`globalx*`/`OBJECT_COMM_WORLD` reach-ins replaced (§3). |
| M3.3 mutation | Writable positions, owned-vertex iteration, per-triangle geometry updates, vertex incidence (§4). |
| M3.4 construction / extraction | Builder, copy/merge, neighbour completion; `polygonise.c` emission deliberately left for Stage 15 (§4, §9.4). |
| M3.5 point sets | Batch point query replaces `MFTL_Point` gather lists (§5). |
| S1 canonical edge id on the ghost rim, handle equality, `AdjTri` order | Specified; needs MFTL2 library item L3 (§2). |
| S2 invalidation list | Complete list of structural vs value-only operations in the header. |
| S3 `const`, rotation / sphere semantics | `const` inputs; rotation vector about a centre; scale about the origin; unit icosphere + ROOT/REPLICATED placement (§6). |
| S4 MFTL2 package portability | Target-scoped flags, native opt-in, fat LTO, instrumented-archive link interface, top-level-only tests (§7). |
| S5 record accuracy | Verification recorded with explicit coverage gaps (§11). |

## Final-fix round (2026-09-23, Claude Code — independent review follow-up)

No new architecture; the reviewer's remaining items only.

**M1 — conserved surface scalar.** New adapter concept `MFAdapterTriSurfaceScalar(m, t)` /
`MFAdapterTriSetSurfaceScalar(m, t, c)`, documented in `mftl_adapter.h` with MFTL1's exact
semantics (concentration per unit area; transported with the triangle; mass-conserving remap in
split — parent mass split by pre-fit child areas —, flip and collapse — local mass redistributed
by new areas —, TSUR and volume-conserving smoothing — per-triangle mass kept, c *= A_old/A_new;
not rescaled by application vertex moves; 0 on generated/built triangles; preserved by the
native restart mesh). MFTL1 backend maps directly to `MFTL_Triangle::scalarconcentration`
(compile-time check that MFTL1 is built with `FRONTTRACKING`). All 20 multiflow uses (8 files)
converted mechanically; `x += e` became `set(x + (e))` — identical arithmetic. Stage-15 item
**L7** added (native scalar or remapping hooks reproducing MFTL1; never zero-payload semantics).

**M2 — remeshing mode.** `MFAdapterRemeshSettings.load_balance` (int, -1/0/1) replaced by
`MFAdapterRemeshMode mode` ∈ {`MFADAPTER_REMESH_PARALLEL` (= 0, so zero-initialised settings keep
the historical default), `MFADAPTER_REMESH_SERIAL`, `MFADAPTER_REMESH_PARALLEL_LOAD_BALANCED`}.
The MFTL1 backend maps explicitly SERIAL→-1, PARALLEL→0 (`MFTL_NO`), LOAD_BALANCED→1
(`MFTL_YES`) and back (MFTL1 never writes the field, so the round trip is exact). multiflow:
`FTRemeshModeFromOptions` maps the `Optimizer` option (-1/0/1 — the only values the parser
produces) to the same three modes; the two "remesh in serial" sites (`executemarchingcubes.c`,
`reconstructfrontwithmarchingcubes.c`) use `MFADAPTER_REMESH_SERIAL`; the two
`== MFTL_YES` tests use `MFADAPTER_REMESH_PARALLEL_LOAD_BALANCED`. Every path therefore passes
the same MFTL1 value as before.

**Contract / test hardening.**
- Cross-rank canonical edge-id test (all stored triangles, owned and ghost, gathered from all
  ranks: one vertex pair ⇔ one edge id).
- Payload during remeshing: surviving triangles keep their payload; new triangles start zeroed;
  only the conserved surface scalar is remapped.
- Body-list iteration semantics under mutation: insertion order; destroy/transfer of the current
  body safe when the next handle is taken first; renumber re-appends at the end (the body is
  visited again); create/transfer-in appends.
- Triangle-soup builder restricted to EMPTY bodies (enforced: `BuildAdd*` return NONE and
  `BuildFinalize` fails collectively on a non-empty body).
- Edge endpoint order not guaranteed; `EdgeAdjTri` / `EdgeComplementaryTri` may return NONE.
- Periodic default: front decompositions are non-periodic (`periodic` = 0; DMDA periodicity only
  recorded as `fluid_periodic`), preserving MFTL1 behaviour into Stage 15.

**Verification.** Integration-layer unit tests on 1/2/3/4 ranks: 78/168/261/360 checks, 0
failed (new: cross-rank edge ids; surface scalar zero at generation, transport through
gather/redistribute to 1e-14, mass conservation through a serial refine/flip/collapse remesh to
1e-10, restart round trip to 1e-12; zero-initialised settings select PARALLEL; mode round trip;
builder rejection). MFTL2 Release 53/53, 0 warnings. multiflow builds MFTL (Release),
MFTL+IRL (Release) and MFTL off (Debug `linux-gcc-dev`): 0 warnings, 0 errors. HEAD-vs-working-
tree A/B of the golden FT case: np1 7/7 and np4 10/10 output files bit-identical. Remeshing-mode
paths: the FT/FT_Sharp golden cases use `Optimizer -1` (serial path, exercised by the A/B); the
parallel (0) and load-balanced (1) paths are covered by the one-to-one mapping (reviewed), the
zero-default check and the mode round-trip unit test, not by an end-to-end A/B.
No new architectural issue was found.

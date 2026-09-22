# 14 — CFD adapter layer in multiflow (refactor against MFTL1, behavior-preserving)

Status: NOT STARTED
Depends on: — (runs against MFTL1; can start early; aligns with 12)
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

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

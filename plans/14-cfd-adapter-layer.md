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
- Direct struct reach-ins to replace (counts across `multiflow/src/`): `->hh.next` (444),
  `->Triangles` (86), `->Vertexes` (76), `->Edges` (43), `->center[` (93), `->area` (86),
  `->normal[` (68), `Vertex[i]->x` (42/42/7), `->mainprocessor` (46), `->centerproc` (14),
  `Triangle->Data`/`->Data`/`->SizeOfData` (30/26/15), `->scalarconcentration` (30),
  `->meancurvature`/`->gaussiancurvature` (14/7). Hot files:
  `interfacial/fronttracking/{surfacescalartransport,surfaceviscosity,volumerecovery,
  computeforcesonfront,spreadfromfronttofluid}.c`, `lagrangian/moveibm/{updatemoveibmftlbodies,
  ispointinsideofparticle}.c`, `marchingcubes/polygonise.c`. All guarded by `#ifdef MFTL_ON`.

## Proposed changes
Introduce `multiflow/src/general/mftlwrapper/mftl_adapter.{h,c}` exposing a
library-representation-agnostic API. Then mechanically rewrite the direct reach-ins to use it:
- **Iteration**: `MFAdapterForEachOwnedTriangle(obj, cb, ctx)` / an index-based iterator that
  hides `hh.next` vs array loop; likewise for vertices.
- **Geometry accessors**: `MFAdapterTriangleCenter/Normal/Area(obj,t,out)`,
  `MFAdapterTriangleVertexCoords(obj,t,3x3)`, `MFAdapterVertexPos(obj,v,out)`.
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

## Possible failure modes
- Subtle behavior change (reordering, extra recompute) breaking the bit-fragile golden suite →
  refactor in tiny steps, diff HDF5 after each.
- Missing an accessor for a rarely-used field (e.g. `createadaptivitylabel.c` in AMR also
  includes `mftl.h`) → grep all `#include "mftl.h"` sites; cover every reached-into field.
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

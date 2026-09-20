# 12 — Opaque per-triangle Data payload + parallel transport

Status: NOT STARTED
Depends on: 01
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Reproduce MFTL1's per-triangle user-data mechanism (`void *Data` + `Object->SizeOfData`) with
**byte-exact** transport across redistribution and ghosting. This is the contract multiflow's
front-tracking relies on (`struct CellData`), and a prerequisite for the CFD migration (Stages
14–15).

## Relevant existing code
- MFTL1: `MFTLInitializeObjectData(Object, DataSize)` sets `SizeOfData`; per-triangle
  `MFTL_AllocateMemory(SizeOfData, &T->Data)` + `memcpy`; transport in
  `src/generalfunctions/scatterdataarray.c` (`MFTL_ScatterTriangleData`, `MPI_BYTE` of
  `SizeOfData`), `src/hashfunctions/triangletoarray.c` (`Data` bytes appended after 3 vertices),
  `arraytotriangles.c` (restore); driven by `MFTLSendTrianglesToProcWCenter(Obj, YES/NO)`.
- multiflow: `struct CellData` (`src/include/fronttracking.h`, flat POD: scalar, velocity[3],
  grad_scalar[3], stress_tensor[3][3], SFTForce[3], area history, curvature, …);
  `initializefronttracking.c:207` `MFTLInitializeObjectData(body, sizeof(struct CellData))`;
  dense `(struct CellData*)Triangle->Data` + `memcpy(...,SizeOfData)` in `surfacescalartransport.c`,
  `surfaceviscosity.c`, `spreadfromfronttofluid.c`.
- MFTL2: `MFTL_Triangle.data` (`void*`, with a TODO to hash it for cache), `MFTL_Object` (needs a
  `data_size` field); `MFTLPackSingleTriangle` (`communication.c:231`), `MFTLRedistribute*`,
  `MFTLUpdateGhostPositions`, `MFTL_ScatterTriangleData` equivalent (absent — the TODO at
  `communication.c:503` "incorporate void data pointer to triangle communication & ghost layer").

## Proposed changes
- Add `size_t data_size` to `MFTL_Object` and `MFTLInitializeObjectData(MFTL_Object*, size_t)`
  which allocates a `data_size` blob per triangle (contiguous pool preferred, per the existing
  TODO, for cache + simple transport).
- Extend the packed-triangle transport (`MFTLPackSingleTriangle` /
  `PackedTriangleComplex` / redistribution + `MFTLExtendGhostLayer` / `MFTLUpdateGhostPositions`)
  to append and restore `data_size` bytes as `MPI_BYTE`, mirroring MFTL1.
- Add `MFTLScatterTriangleData(MFTL_Ensemble*/MFTL_Object*)` — refresh ghost-triangle `data` from
  owners by global id, matching MFTL1's `MFTL_ScatterTriangleData` (used to update ghost payloads
  without moving ownership).
- Accessors `MFTLGetTriangleData(obj,t)` / `MFTLSetTriangleData(obj,t,src)` for the adapter
  (Stage 14).

## Files/functions likely to change
`include/mftl_core.h` (`MFTL_Object.data_size`), `src/general/object.c`
(`MFTLInitializeObjectData`, alloc/free), `src/general/communication.c`
(`MFTLPackSingleTriangle`, buffer sizing `MFTLCalcBufferCapacities`, redistribution, ghost
append/update, new `MFTLScatterTriangleData`), `include/mftl_communication.h`. Test
`tests/test_triangle_data.c`.

## Numerical/physical considerations
- The blob is **opaque and must stay a flat POD** — no embedded pointers (a pointer would be a
  valid address only on the origin rank). Document this contract loudly; multiflow's `CellData`
  already satisfies it.
- On edge split/collapse (Stage 08), user data must be redistributed physically-consistently
  (MFTL1 splits scalar mass by area ratio and asserts mass error < 1e-14) — provide a hook or
  document that data-carrying remeshing requires the caller's split/merge policy.

## MPI/PETSc considerations
- Transport uses `MPI_BYTE` of `data_size`; the receiver must know `data_size` (broadcast/known
  from `MFTLInitializeObjectData` collectively). Size mismatch across ranks silently corrupts.
- Ghost data staleness: after vertices move/redistribute, `MFTLScatterTriangleData` refreshes
  ghost payloads from owners (front-tracking calls this between compute phases). No PETSc here,
  but this feeds the PETSc-coupled spreading in multiflow.

## Possible failure modes
- Buffer capacity not accounting for `data_size` → truncated payload / overflow (ASan catches).
- Pointer inside the blob → cross-rank corruption with no error.
- Data not following a triangle through load-balance redistribution → mass loss; test explicitly.
- `data_size==0` (data unused) must be a no-op fast path.

## Unit/regression tests to add (`tests/test_triangle_data.c`)
- Attach a POD struct (id + scalar + vec3) to every triangle; redistribute across np2/4/8 →
  assert each triangle's blob is **byte-identical** to before (keyed by global id).
- Scalar-mass conservation: `Σ scalar·area` invariant across redistribution (tol 1e-14).
- Ghost consistency: after `MFTLScatterTriangleData`, ghost-triangle blobs equal their owners'
  (mirror MFTL1 `test14DataRedistribution.c` cases 1–4: data follows migration, ghost update,
  one-ring availability, scatter to ghosts).
- `data_size==0` path leaves transport unchanged (regression vs Stage 01 behavior).

## Acceptance criteria
- Per-triangle blob byte-identical after redistribution and ghost scatter, keyed by global id.
- Scalar mass conserved to 1e-14 across redistribution; ghost payloads match owners.
- No ASan errors; `data_size==0` is a no-op.

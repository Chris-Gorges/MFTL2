# 12 — Opaque per-triangle Data payload + parallel transport

Status: COMPLETE
Depends on: 01
Implemented by: Kimi Code CLI (2026-09-21)
Reviewed by: Claude Code (2026-09-21)
Acceptance: PASSED

> **Review note (accepted, Claude Code 2026-09-21):** Byte-exact transport verified across
> both redistribution routes (list + single-object), ghost-layer extension, spatial ghosting,
> and `MFTLScatterTriangleData`, on np 1/2/4/8 — Release + ASan/UBSan/LSan clean
> (`owned_found=1280/1280`, `|ΔM|=0`, 116/332 ghost payloads refreshed). The two-level radix
> permutation carries the payload with the *same* index as the triangle at both sort levels;
> ghost `data_cursor` advances unconditionally (skip-safe). Lifecycle correct: `new_triangle`
> zeroes split-product blobs, PATH-B compaction frees dead blobs exactly once (live blobs ride
> along by struct assignment), `Destroy` frees `[0,n)` — no leak/double-free/dangling
> (confirmed by an out-of-tree ASan/LSan remesh-with-data run: 5120→452 tris, clean; survivors
> keep data, split-products zeroed). Scatter is collective-safe with correct miss/dead
> (`found=0`, stale-copy) semantics; `data_size==0` is a true no-op. Per-blob heap deviation is
> sound. **Should-fix (not blocking; do before Stage 14/15):** (1) add a cheap collective
> `data_size` consistency check (Allreduce MIN/MAX per body) — the contract's "silent
> corruption" on a cross-rank mismatch is currently undetected; (2) add an in-repo
> remesh-with-`data_size>0` test — the split/PATH-B lifecycle has no CI ASan/LSan coverage
> (validated only out-of-tree here).

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

## Implementation notes / deviations (Kimi Code CLI, 2026-09-21)

- **Per-triangle heap blobs, not a contiguous pool.** The plan preferred one contiguous pool for
  cache and simple transport; the implementation keeps MFTL1's per-triangle
  `MFTL_AllocateMemory(SizeOfData, &T->Data)` shape (one `calloc` per triangle). Reason:
  `MFTLFinalizeRemeshing` PATH B compacts the live triangle array by struct assignment, so a
  `data` pointer simply rides along with the struct — a contiguous pool would need explicit
  offset remapping there. This is also exact parity with the MFTL1 layout that Stage 14's
  adapter assumes. Consequence: blob allocation/free is malloc-heavy at mesh build; fine at
  current scales.
- **Payload rides in a parallel `MPI_BYTE` stream, not inside the Packed structs.**
  `PackedTriangle`/`PackedTriangleComplex` are untouched (fixed stride keeps all the existing
  displacement arithmetic valid). Every transport site instead carries a second buffer in the
  identical element order (`all_data` in redistribution, `raw_send_data` in ghost-layer
  extension, `sendbuf_dataA/B` in pair ghosting) exchanged with `MPI_BYTE` counts derived from
  the already-exchanged triangle counts. A `NULL` blob where `data_size>0` is a hard
  `exit(EXIT_FAILURE)` at pack time rather than silent garbage.
- **Remeshing does not repartition user data.** Per "Numerical/physical considerations", the
  caller owns the split/merge policy: `new_triangle` gives split/collapse products a **zeroed**
  blob (documented in `remeshing_core.c`), dead triangles' blobs are freed in PATH B, and live
  triangles keep their payload unchanged. No area-ratio scalar splitting is implemented; MFTL1's
  1e-14 mass-error assertion belongs to multiflow's split/merge hooks, not this stage.
- **Mass-conservation tolerance is 1e-12 relative, not 1e-14.** Transport is byte-exact, but
  `Σ scalar·area` is a floating-point reduction whose order changes with the permutation and
  ghost partition; the 1e-14 figure in this plan refers to MFTL1's split/merge mass error.
- **`MFTLScatterTriangleData` is a two-stream request/reply** (int found-flags + `MPI_BYTE`
  blobs) mirroring `MFTLUpdateGhostPositions`, with an unconditional skip-consensus Allreduce
  (warns on stderr if any requested global id was found on no owner) and stale-copy semantics
  for ghosts whose owner is dead/missing. It early-outs collectively when no object in the
  ensemble has `data_size>0`.
- **Test 2 is a superset of the brief:** besides `MFTLRedistributeListOfObjects` it also routes
  through `MFTLRedistribute(ens)` (single-object → `MFTLRedistributeObject`) and byte-verifies
  all triangles against the pre-redistribution global snapshot — this covers the single-object
  path with data that the listed tests would otherwise not exercise.
- **Assumptions documented at the API:** `data_size` is collective (set once via
  `MFTLInitializeObjectData` on all ranks; re-init with a different nonzero size is an error),
  the blob is flat POD (no embedded pointers), and ghost-extension paths assume the ensemble's
  object list is identical on all ranks.
- Verified: Release `ctest` 49/49 and ASan/UBSan `ctest` 49/49, including the new
  `TriangleData-np1/2/4/8` tests in `tests/test_triangle_data.c`.

## Follow-up items (code review, 2026-09-21) — RESOLVED

- **Collective `data_size` consistency check** — `MFTLCheckTriangleDataSizeConsistency(obj)`
  (declared in `mftl_communication.h`): two tiny Allreduces (MIN/MAX, `MPI_UNSIGNED_LONG_LONG`)
  of `data_size` over `obj->domain->comm_cart`; on mismatch prints a loud FATAL stderr
  diagnostic (object gid, min/max/local sizes) and returns `MFTL_ERROR`. Called at the ENTRY
  of every triangle-data transport path, after the uniform early-outs (`nproc < 2` etc.) and
  before any packing/communication: `MFTLRedistributeListOfObjects` (per object),
  `MFTLRedistributeObject`, `MFTLExtendGhostLayer`, `MFTLEnsembleCreateSpatialGhosts`,
  `MFTLScatterTriangleData` (deliberately BEFORE its `any_data` early-out — that early-out is
  only rank-consistent when `data_size` agrees), and `MFTL_CreateSpatialGhosts` (both objects;
  returns `MFTL_ERROR` per its int convention). Negative test
  `test_data_size_mismatch_detected` (runs LAST in `tests/test_triangle_data.c`, np≥2): rank 0
  initializes `sizeof(TriData)`, others `+8`; all four entry points must return `MFTL_ERROR`
  uniformly on every rank (Allreduce-confirmed) with the mesh untouched; ensemble destroys
  cleanly.
- **Remeshing-with-data regression test** — `TestRemeshingTriangleData` in
  `tests/test_remeshing.c` (np1, under the existing LSAN suppression): builds a sphere via the
  ensemble/generator path, attaches a deterministic POD blob per triangle, snapshots
  (gid → bytes), then runs an explicit split×3 / in-place flip / collapse sequence with
  disjoint untouched quads, `MFTLFinalizeRemeshing(..., collapses_occurred=true)` (PATH B
  compaction), and one full serial `MFTLRemeshEnsemble` pass. Verifies after each: every
  triangle has a blob; snapshot gids are byte-identical (covers split-reused and flipped
  triangles); new split gids are fully zeroed (exactly 2 per interior split in the explicit
  sequence); no sanitizer findings across the blob alloc/free lifecycle.

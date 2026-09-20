# 05 — Mesh I/O: GTS reader + writer + PVTU

Status: NOT STARTED
Depends on: 01 (02 for volume-based regression)
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Give MFTL2 the ability to **load meshes from disk** (GTS) and write GTS/PVTU. This is the top
practical unblocker: MFTL2 can currently only synthesize spheres, so real geometries (and the
multiflow GTS workflow) cannot be evaluated on it.

## Relevant existing code
- MFTL1: `src/io/mftlio.c` (`MFTLGTSFileRead` — assigns sequential ids `id=i`; `MFTLGTSFileCheck`;
  GTS/VTK writers), `MFTLWriteObjectToGTS`, `MFTLWriteObjectToVTK`, `MFTLWritePVTUOutput`.
- multiflow callers: `src/general/mftlwrapper/readGTSFiletoMFTLObject.c`,
  `getmarkersfromexternalmesh.c` (both wrap `MFTLGTSFileRead`).
- MFTL2: `src/general/io.c` (`MFTLWriteVtk`, `MFTLWriteVtu`, already has "Smart Periodic
  Unwrapping"), topology builders `MFTLRebuildTwins`, `MFTLFlattenConnectivity` (used by
  `sphere.c` to turn a vertex+triangle-vertex list into half-edge connectivity).

## Proposed changes
In `src/general/io.c` (+ decls in `include/mftl_io.h`):
- `MFTLReadGTS(MFTL_Object*, const char* path)` — parse GTS (nV, nE, nT header; vertex lines;
  edge lines; triangle-as-edge-triple lines), populate `vertices[]` and a triangle→vertex
  list, then build half-edge topology via `MFTLRebuildTwins`/`MFTLFlattenConnectivity`, assign
  `global_id`s, `MFTLComputeTriangleProperties`, and integrity-check.
- `MFTLWriteGTS(MFTL_Object*, const char* path)` — inverse (derive edges from half-edges).
- `MFTLWritePvtu(MFTL_Ensemble*, const char* stem)` — parallel VTU collection (`.pvtu` + per-
  rank `.vtu`) reusing `MFTLWriteVtu`, with the periodic-unwrapping already in io.c.

## Files/functions likely to change
`src/general/io.c`, `include/mftl_io.h`; possibly small additions to topology helpers if GTS's
edge list must be reconciled with half-edge orientation. Test `tests/test_io.c` +
`tests/CMakeLists.txt`. Test fixtures: reuse `mftl/meshes/*.gts` and `mftl/examples/test1.gts`.

## Numerical/physical considerations
- GTS stores explicit edges; MFTL2 needs half-edges. The reader must derive consistent twin
  pairing and winding; a mis-oriented triangle flips a normal and corrupts volume sign.
- Preserve vertex coordinates bit-exactly (no reordering that changes float sums) so a
  read→write→read round-trip is stable.
- Enclosed volume of a read closed mesh must match MFTL1's `MFTLObjectVolume` on the same file.

## MPI/PETSc considerations
Decide read semantics: rank-0 reads then scatters (simplest, matches MFTL1 single-file GTS), or
collective. Match `MFTLGenerateSphere`'s post-generation distribution so a read mesh is
immediately redistributable. PVTU writer must gather per-rank piece filenames on rank 0. No
PETSc.

## Possible failure modes
- Non-manifold or open GTS input → `MFTLRebuildTwins` leaves boundary half-edges; handle or
  reject with a clear error (multiflow uses closed surfaces).
- Inconsistent edge/triangle winding in the file → normal/volume sign errors; canonicalize.
- Large meshes: ensure capacity growth (`MFTL_ENSURE_CAPACITY`) not O(n²) reallocs.
- Locale/whitespace parsing bugs in the GTS header.

## Unit/regression tests to add (`tests/test_io.c`)
- Read `mftl/meshes/sphere_locally_refined.gts` / `disco.gts`: integrity pass, twin symmetry,
  expected V/E/T counts.
- **Round-trip**: read→`MFTLWriteGTS`→read; identical counts and volume (tol 1e-12).
- **Volume regression**: read `mftl/examples/test1.gts`; volume equals MFTL1's value (tol 1e-9).
- Parallel read on np2/np4 distributes triangles (each rank owns >0) and global volume matches
  the serial read (rank-independence).
- PVTU: write then confirm the `.pvtu` references N `.vtu` pieces and reloads in a viewer
  (smoke: file exists, well-formed XML, correct piece count).

## Acceptance criteria
- Reads MFTL1 GTS fixtures into a valid, closed, integrity-passing half-edge mesh.
- Round-trip preserves counts and volume to 1e-12; volume matches MFTL1 to 1e-9.
- Parallel read is rank-independent in global volume; PVTU output is well-formed.

# 01 — Build stabilization & test-harness hardening

Status: NOT STARTED
Depends on: —
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
Establish a green, buildable MFTL2 trunk from commit `1327579`, fix the defects that break
HEAD, harden the test harness so failures are actually detectable, and introduce the optional
LAPACK/BLAS build dependency required later (Stage 08). This stage unblocks all others.

## Relevant existing code
- Header cycle: `include/mftl_core.h:20` (`#include "mftl_remeshing.h"`),
  `include/mftl_remeshing.h:13` (`#include "mftl_communication.h"`),
  `include/mftl_communication.h:17` (`#include "mftl_core.h"`) +
  `include/mftl_communication.h:172` (`MFTLGetOwnerProc` dereferencing `MFTL_Domain`).
  `mftl_core.h` needs only two enums from remeshing.h: `MFTL_EnsembleType` (field at
  `mftl_core.h:251`) and `MFTL_RemeshAlgorithm` (`:252`).
- Ghost-layer bug: `src/general/communication.c:MFTLExtendGhostLayer` (~:2113) +
  `MFTLAppendGhostVertex`/`MFTLAppendGhostTriangle` (~:2041/:2068). The np4 test fails with
  `Half-edge references missing vertex GID` after a 2-step extension.
- Harness gaps: `tests/test_remeshing.c:517` (`return 0`), `tests/test_mpi.c` (Send/Recv
  stub), `tests/test_mpi_stress.c` (triangle-count-only invariant).
- Build: `CMakeLists.txt` (only `find_package(MPI REQUIRED)`), `build_all.py`,
  `run_local_tests.py`, `tests/CMakeLists.txt`.

## Proposed changes
1. **Break the header cycle** (choose the least-invasive): move `MFTL_EnsembleType` and
   `MFTL_RemeshAlgorithm` enum definitions into `include/mftl_constants.h`, and remove
   `#include "mftl_remeshing.h"` from `mftl_core.h`. (Fallback: type the two `MFTL_Object`
   fields as `int` — less clean, loses enum typing.)
2. **Fix `MFTLExtendGhostLayer`** so appended ghost half-edges' `vertex_idx` (and
   `twin_idx`/`next_idx`/`triangle_idx`) are remapped from global ids to the appended ghost
   entities' local indices, for arbitrary step counts. Verify with the existing np4 test +
   an added 3-step case.
3. **Harness hardening**: `test_remeshing.c` `main` → `return (g_tests_failed>0)`; replace
   `test_mpi.c` stub with a real `MFTLUpdateGhostPositions` round-trip assertion; add a
   *volume*-conservation invariant to `test_mpi_stress.c` (sum signed tet volumes, Allreduce,
   compare to initial within tol) alongside the triangle-count check.
4. **CMake LAPACK/BLAS (optional)**: `option(MFTL_USE_LAPACK …)`; `find_package(LAPACK)`; when
   found, `target_link_libraries(MFTL PUBLIC LAPACK::LAPACK)` and define `MFTL_HAVE_LAPACK`.
   No use yet (Stage 08 consumes it). Keep the default build LAPACK-free.

## Files/functions likely to change
`include/mftl_core.h`, `include/mftl_constants.h`, `include/mftl_remeshing.h`,
`include/mftl_communication.h`; `src/general/communication.c`
(`MFTLExtendGhostLayer`, ghost append helpers); `tests/test_remeshing.c`, `tests/test_mpi.c`,
`tests/test_mpi_stress.c`, `tests/CMakeLists.txt`; `CMakeLists.txt`.

## Numerical/physical considerations
None new. The added volume invariant in the stress test uses the Stage-02 formula (⅙
triple-product); until Stage 02 lands, compute it inline in the test or gate this sub-item on
Stage 02. Tolerance for the count/volume check ~1e-9 (exact) — the mesh is rigid-advected, so
volume must be conserved to round-off.

## MPI/PETSc considerations
Ghost-layer remapping is the crux: half-edge indices are rank-local, so any global-id→local
remap on the receiver must use the ghost append order deterministically. Confirm behavior on
1/2/4/8 ranks. No PETSc in this stage.

## Possible failure modes
- Moving the enums breaks other includers → grep for `MFTL_EnsembleType`/`MFTL_RemeshAlgorithm`
  and adjust includes.
- Ghost remap fixed for 2 steps but not N → test 1,2,3,5 steps and assert monotonic ghost
  growth (as the existing test does at `:130`).
- LAPACK not present on a target machine → optional flag must degrade gracefully (Stage 08
  paraboloid fit falls back to midpoint when `MFTL_HAVE_LAPACK` undefined).

## Unit/regression tests to add
- Re-enable exit-code failure in `test_remeshing`; confirm a deliberately broken op turns the
  CTest red.
- `test_mpi` real ghost-position sync assertion on np2.
- `test_ghost_layers` extended: assert no dangling half-edge→vertex references after 1/2/3/5
  steps on np4 (the previously-failing case must pass).
- `test_mpi_stress` volume-conservation assertion on np8.

## Acceptance criteria
- `python3 build_all.py` (Release) and `--debug` (ASan) both succeed with **0 warnings** on the
  library.
- Full `ctest` green on 1, 2, 4, and 8 ranks, including the fixed ghost-layer and stress tests.
- `MFTL_USE_LAPACK=ON` configures and links when LAPACK present; default build unaffected.
- `git` history untouched beyond the new working branch (no commits pushed).

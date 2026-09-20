# 00 — Overview: reimplementing MFTL1 functionality in MFTL2

Status: NOT STARTED
Depends on: —
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Purpose

Bring MFTL2 to **full functional parity with MFTL1** (including Cartesian periodicity) and
migrate the `multiflow` CFD solver onto it, in independently-reviewable stages that each
leave the tree compiling and testable. This file is the shared design contract; every stage
`01…15` refers back to the invariants, conventions, and validation strategy defined here.

## The two codebases

- **MFTL1** — `/home/berend/git/Code/mftl`, ~28k LOC / 90 files. Data model: three `uthash`
  tables per `MFTL_Object` (`Vertexes`, `Edges`, `Triangles`), pointer connectivity
  (`Edge->Triangle[2]`, `Triangle->Vertex[3]`, `Vertex->RefTriangle`). Mature remeshing,
  intersection/overlap, point queries, generators, GTS/VTK/PVTU I/O. Slow (uthash +
  pointer-chasing across 58 files); **no periodicity**.
- **MFTL2** — `/home/berend/git/Code/MFTL2`, ~10k LOC. Data model: **half-edge** over
  flattened arrays (`vertices[]`, `triangles[]`, `half_edges[]`, index references
  `he_idx`/`twin_idx`/`next_idx`/`vertex_idx`), `khash` id→index maps, ghosts stored after
  locals. MPI-Cartesian decomposition (`MPI_Cart_create` with `periods[]`), BVH/LBVH
  intersection, CTest suite, and **periodicity primitives** (`MFTLGetMinimumImageVector`,
  `MFTLApplyPeriodicity`, `MFTL_Domain.is_periodic[3]`/`domain_half_size[3]`). Purpose-built
  for the two things MFTL1 lacks (speed + periodicity), but incomplete.

### Data-model translation (applies to every stage)

| MFTL1 idiom | MFTL2 equivalent |
|---|---|
| `for(T=Obj->Triangles; T; T=T->hh.next)` | `for(int t=0;t<obj->n_triangles;t++)` (locals: `t<n_local_triangles`) |
| `Triangle->Vertex[i]->x[d]` | `MFTLGetTriangleVertices(obj,t,v)` → `obj->vertices[v[i]].pos[d]` |
| `Triangle->center/normal/area` | `MFTL_Triangle.center/normal/area` (via `MFTLComputeTriangleProperties`) |
| `Triangle->mainprocessor` / `centerproc` | `MFTL_Triangle.owner_proc` (+ ghost flag `is_ghost`) |
| `Edge->Triangle[k]`, one-ring by pointer | half-edge traversal (`MFTLHeTwin/Next/…`, `MFTLGetVertexOneRing*`) |
| `HASH_FIND_INT(Obj->Vertexes,&id,..)` | `khash` id→index map (`vertex_map` etc.) |
| entity `id` (sequential counter) | `global_id` (per-entity) + array index |
| `Triangle->Data` + `Object->SizeOfData` | `MFTL_Triangle.data` (see Stage 12) |
| `Object->globalxmin/max`, `ownershiprange*` | `MFTL_Domain.global_min/max`, `MPI_Cart` grid |

## Build baseline

Work branches from **commit `1327579`** — the last commit that compiles cleanly and passes
the suite under AddressSanitizer. HEAD (`f4795fe`) and the two preceding commits do **not**
build:
1. **Circular header include**: `mftl_core.h`→`mftl_remeshing.h`→`mftl_communication.h`→
   `mftl_core.h`; a `static inline` (`MFTLGetOwnerProc`) dereferences `MFTL_Domain` before it
   is defined.
2. **Incomplete parallel-remeshing WIP** in `remeshing_parallel.c`: `MFTLRemeshEnsemble`
   defined twice (conflicting `void`/`int`), `MFTLRemeshEnsembleShiftScheme` and
   `MFTLRemeshEnsembleDispersedCentering` called but never implemented, `MFTL_Object**`
   assigned to `MFTL_Object*`.

The `d248554…HEAD` commits are treated as a source to finish deliberately in **Stage 10**,
not as a working base. Stage 01 establishes the green trunk.

## Cross-cutting invariants (enforced by tests in every stage)

1. **Closed, consistently-oriented 2-manifold.** Euler characteristic (V−E+F=2 per body)
   preserved by all remeshing operators. `MFTLCheckMeshIntegrity` must pass after every
   topological operation.
2. **Volume** = `⅙ Σ_tri v₁·(v₂×v₃)` about the coordinate origin (MFTL1
   `parallel/objectvolume.c:VolumeFace`). This is the ground truth for all conservation
   checks. Requires closed, consistently-oriented surface.
3. **Surface centroid** = area-weighted vertex-sum ÷ (3·total area) (MFTL1
   `objectcentre.c`). **Moment of inertia** = lumped tetra approximation about the volume
   centroid, uniform density (MFTL1 `momentofinertia.c`) — **not** the exact
   tetrahedron-covariance formula; reproduce the lumped form for bit-comparability, and add
   an exact-formula cross-check test.
4. **Curvature convention**: for a sphere of radius R, mean curvature stored as **2/R** and
   Gaussian as **2/R²** (cotangent Laplace–Beltrami with mixed-Voronoi area, 0.125 factor;
   MFTL1 `oneringcomputations.c:MFTL_VertexLaplaceBeltrami`). Deviating changes the
   Botsch–Kobbelt adaptive edge length `L=√(6ε/κ−3ε²)`.
5. **Parallel rank-independence**: integral quantities sum over owned entities only
   (`owner_proc==rank`) then `MPI_Allreduce(MPI_SUM)`; ray-cast direction is `MPI_Bcast` from
   rank 0; tolerance is `MPI_Allreduce(MPI_MIN)`. A result must be identical (to tolerance)
   on 1/2/4/8 ranks.
6. **Mesh-adaptive global tolerance**: `MFTL_TOLERANCE` = (smallest triangle's shortest edge)
   × 1e-5, min-reduced across ranks (MFTL1 `tolerance.c`). Companion constants: `MFTL_SMALL
   1e-7`, `MFTL_VERYSMALL 1e-15`. Overlap uses its own `MFTL_REL_TOLERANCE 5e-7`.
7. **Opaque per-triangle `Data`** is a flat POD, transported byte-exact as `MPI_BYTE` of
   `SizeOfData` (multiflow's `struct CellData`, `src/include/fronttracking.h`). No embedded
   pointers — Stage 12.
8. **Dual MPI decomposition** (migration): MFTL's per-object MPI-Cartesian grid is
   independent of multiflow's PETSc/DMPlex(DMForest/p4est) fluid partition; they are *adapted*
   to line up with the `CartesianOverlapMesh` particle-box grid but do not coincide, and the
   `sendnonlocal*.c` transpose-table exchanges reconcile them. Must be preserved — Stage 15.

## Numerical fragilities to design around (from MFTL1)

- Manual 3×3 adjugate inversions (vertex fitting, overlap Gram–Schmidt).
- Jacobi eigensolver (500-sweep cap) for OBB covariance.
- **LAPACK `dgelsy`** dependency for the paraboloid least-squares fit (Stage 08) — MFTL2
  currently links only MPI; introduce LAPACK/BLAS as an optional CMake dependency in Stage 01.
- Heron's-formula areas with negative-radicand guards (overlap).
- Hard-coded triangle-quality constant `20.78460969` (equilateral → 1.0).
- Roughness arrays sized 10 ⇒ **valence ≤ 10** cap.
- Load balancing assumes **x-only 1-D** cuts (`Ycut=Zcut=1`) to avoid crossed boundaries.

## Validation strategy

**Harness**: mimic MFTL2 `tests/test_utils.h`: a test is `int test_x(void)` returning 0/1;
`RUN_TEST(test_x)`; `TEST_ASSERT`, `TEST_ASSERT_EQUAL_INT`, `TEST_ASSERT_EQUAL_DOUBLE(exp,
act,tol)`; `TEST_REPORT()`; `main` returns `g_tests_failed>0`. MPI tests wrap `MPI_Init/
Finalize`, register with `add_test(NAME … COMMAND mpirun -np N $<TARGET_FILE:…>)` (MPICH —
**no** `--oversubscribe`), and self-skip if `nproc` ≠ N. Build via `python3 build_all.py
[--debug]`; run via `python3 run_local_tests.py` or `ctest`.

**Every stage's tests span**: rank-independence (1/2/4/8), conservation (volume/mass),
symmetry (A∩B==B∩A; symmetric-shape centroid at origin), limiting cases (sphere curvature
2/R; flat patch→0; single-triangle known area), and regression vs MFTL1 example outputs
(`test1/8/testCylinder` volumes; `test14` data redistribution; `test5/6` intersection;
LeVeque).

**Harness gaps to fix in Stage 01** (found during inspection): `test_remeshing.c:517` `main`
returns 0 unconditionally (can't fail CI); `test_mpi.c` is a hand-rolled Send/Recv stub, not
a real `MFTLUpdateGhostPositions` test; `test_mpi_stress.c` conserves triangle *count*, not
*volume*.

**multiflow validation**: the existing `testcases/TESTS/dotests.sh` golden-HDF5 comparison is
bit-fragile and will not survive the rewrite (mesh ordering, remeshing, float paths change).
Stages 14–15 therefore add **physics-tolerance** regressions on ≥4 ranks: LeVeque single-vortex
shape error (field reverses at t=T ⇒ sphere returns to initial shape; measure enclosed-volume
drift + Hausdorff/L1 shape error), oscillating-droplet frequency vs Lamb, rising-bubble
terminal velocity — plus explicit coverage of the opaque `CellData` byte-transport and the
currently-disabled periodic/clamping advection behavior.

## Stage list & dependency order

`01` build/harness → everything.
`02` integrals → `03` transforms, `04` generators, `09` volume-conserving smoothing, `11` overlap.
`05` I/O (⚠ top unblocker). `06` curvature → `08`. `07` point queries + OBB.
`{01,06,LAPACK}` → `08` serial remeshing A → `09` volume-conserving smoothing → `10` parallel remeshing.
`12` opaque Data transport (front-tracking prereq).
`{02,06,07,10,11}` → `13` periodicity completion.
`14` CFD adapter (early, vs MFTL1) → `15` MFTL1→MFTL2 swap (depends on 02–14).

**Critical path to a usable, faster, periodic library**: 01→02→05→06→08→09→10→13.
**Migration path**: 14 (early, low-risk) then 15 (last).

## Top technical risks

1. Parallel remeshing (10) — least mature, currently broken.
2. Dual decomposition + `sendnonlocal*` bridge (15) — silent front↔fluid corruption if wrong.
3. Opaque `Data` byte-transport (12) — pointer/size mismatch corrupts across ranks silently.
4. Curvature/tolerance semantics (06/01) — subtle remeshing-behavior drift.
5. LAPACK dependency (08) — new build dependency.
6. Validation is being *added*, not ported — physics tolerances must be meaningful yet stable.
7. multiflow periodicity gaps — enabling MFTL2 periodicity must not change existing runs.

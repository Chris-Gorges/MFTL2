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

**Decision (recorded 2026-09-20, Stage 01 review): the project consciously accepts the
current HEAD code commit `f4795fe` as the baseline.** The original plan — branch from
`1327579`, treating `d248554…f4795fe` as discardable WIP — was rejected. Stage 01 instead
fixed the defects that broke HEAD directly on top of it and established the green trunk
there (0 build warnings; full ctest green on 1/2/4/8 ranks in Release and ASan/UBSan
builds). `1327579` is of historical interest only. The parallel-remeshing WIP in
`d248554…f4795fe` is still to be **finished deliberately in Stage 10**: the Stage-01
touches of `remeshing_parallel.c` are mechanical build fixes only (dispatcher renamed to
`MFTLRemeshEnsembleParallel` to resolve the duplicate/conflicting `MFTLRemeshEnsemble`;
prototypes added for the not-yet-implemented Stage-10 kernels; one `MFTL_Object**` type
fix). No parallel-remeshing logic was implemented.

Historical record — HEAD `f4795fe` did not build for two reasons, both fixed in Stage 01:
1. **Circular header include**: `mftl_core.h`→`mftl_remeshing.h`→`mftl_communication.h`→
   `mftl_core.h`; a `static inline` (`MFTLGetOwnerProc`) dereferences `MFTL_Domain` before it
   is defined. Fixed by moving `MFTL_EnsembleType`/`MFTL_RemeshAlgorithm` into
   `mftl_constants.h` and dropping the remeshing include from `mftl_core.h`.
2. **Incomplete parallel-remeshing WIP** in `remeshing_parallel.c`: `MFTLRemeshEnsemble`
   defined twice (conflicting `void`/`int`), `MFTLRemeshEnsembleShiftScheme` and
   `MFTLRemeshEnsembleDispersedCentering` called but never implemented, `MFTL_Object**`
   assigned to `MFTL_Object*`.

## Cross-cutting invariants (enforced by tests in every stage)

1. **Closed, consistently-oriented 2-manifold.** Euler characteristic (V−E+F=2 per body)
   preserved by all remeshing operators. `MFTLCheckMeshIntegrity` must pass after every
   topological operation.
2. **Volume** = `⅙ Σ_tri v₁·(v₂×v₃)` about the coordinate origin (MFTL1
   `parallel/objectvolume.c:VolumeFace`). This is the ground truth for all conservation
   checks. Requires closed, consistently-oriented surface.
   **✅ Winding convention (RESOLVED 2026-09-21, before Stages 11/13):** MFTL2 standardizes
   on **OUTWARD winding for every closed mesh** — outward-pointing triangle/vertex normals
   and **positive signed volume (`V > 0`)**. This is the physical/CFD convention (surface
   tension `F = σκn̂` and the indicator inside/outside test both assume outward normals),
   the standard geometry convention, and what the Stage-04 generators and the canonicalized
   GTS reader already produce. The inward-wound sphere was an **accident** of
   `generate_icosahedron`'s hard-coded triangle table, not a deliberate design. Mixed
   winding is explicitly *not* supported: it would force every direction-dependent
   consumer — Stage-11 overlap, Stage-13 periodic image tests, Stage-14/15 coupling — to
   carry a per-object sign.
   **APPLIED (2026-09-21):** `MFTLGenerateSphere` now canonicalizes to outward winding
   (`canonicalize_outward_winding` in `src/general/sphere.c`: signed-volume check, flip
   only if negative — the GTS-reader approach; idempotent). All 9 pinned sign assertions
   in `test_object_integrals.c` / `test_transforms.c` / `test_remesh_parallel.c` now
   assert `V > 0`. Full Release + ASan/UBSan suites green on 1/2/4/8 ranks; conservation,
   curvature (mean stored as LB *magnitude*), point queries (ray parity) and remeshing
   are unchanged in magnitude, as predicted below.
   **REVIEWED & ACCEPTED (2026-09-21, Claude Code):** `reverse_triangle_winding` verified
   topologically correct (reverses each 3-cycle, `MFTLRebuildTwins` re-pairs); measured
   sphere volume `V = +4.1797214693` (outward); all winding-agnostic consumers confirmed
   unchanged. **Stage 11 is cleared to begin** on the single outward convention (no
   per-object sign handling needed).

   *Why this is safe / small:* everything downstream today is winding-**agnostic** and keeps
   working after the flip — signed-volume conservation (`MFTLVolumeRecovery` is
   winding-invariant: `V` and the ring normal flip together), curvature (mean stored as the
   Laplace–Beltrami *magnitude*, Gaussian via the angle sum), point-in-object (ray parity),
   remeshing/smoothing (geometric), periodicity (minimum-image), and intersection *detection*
   (`MFTL_Orient3D` predicate). No consumer currently *relies* on the sphere being inward;
   only test sign-assertions pin it. That is exactly why Stages 02–10 all pass with
   magnitude/`fabs` checks.

   **Adjustment set (APPLIED 2026-09-21, before Stage 11):**
   - **Production (1 file):** `src/general/sphere.c` — make the generated sphere outward.
     Preferred: after `generate_icosahedron` + refinement, canonicalize to `V > 0` by flipping
     every triangle if the signed volume is negative (reuse the exact approach the GTS reader
     already uses, `src/general/io.c:690`), or equivalently swap two indices in each row of the
     20-entry `triangles_data` table. `tessellate_sphere` inherits the base orientation, so it
     needs no change.
   - **Tests — flip the pinned inward-sign assertions to `V > 0` (8 sites, 3 files):**
     `tests/test_object_integrals.c` (lines 88, 217, 278; the `fabs(fabs(V)-V_true)` magnitude
     checks at 92/113 stay valid but can be simplified), `tests/test_transforms.c` (lines 85,
     140, 207; the `fabs` invariance checks at 96/106/144/226 stay valid),
     `tests/test_remesh_parallel.c` (lines 229, 377, 436, and change `fabs(-v1 - analytic)` →
     `fabs(v1 - analytic)` at 230).
   - **Tests — no change (already winding-agnostic):** `test_volume_smoothing.c` (signed
     differences + `fabs`; `VolumeRecovery` target sign flips harmlessly), `test_remesh_quality.c`
     (uses `fabs(MFTLObjectVolume)`), `test_generators.c` / `test_io.c` (already assert `V > 0`),
     `test_curvature.c`, `test_point_query.c`.
   - **Examples — no change:** none assert a volume sign; `parallel_intersection_spheres`
     uses winding-agnostic BVH detection. The only visible effect is that written VTK/VTU
     sphere normals now point outward (cosmetic/correct).
   - **Verification:** rebuild Release + ASan/UBSan and run the full suite on 1/2/4/8 — the
     sphere must now report `V > 0`, and volume conservation / curvature / point-in-object /
     remeshing results must be unchanged in magnitude.

   **Stage impact:** only Stages 02, 03 and 10 have test-sign edits; Stages 05/06/07/08/09
   need no change. Stages 11 (overlap normal `½(N_A−N_B)`, effective radius, "which side"),
   13 (periodic image orientation) and 14/15 (front-tracking outward normals) are then built
   directly on the single outward convention with no per-object sign handling.
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

**Progress (2026-09-21):** Stages **01–10 COMPLETE / Acceptance PASSED** (implemented by
Kimi Code, reviewed by Claude Code; Release + ASan/UBSan clean — incl. LAPACK/OpenBLAS
for Stage 08's paraboloid fit — full suite green on 1/2/4/8 ranks). Stages 11–15 NOT
STARTED. The **winding convention is now RESOLVED** (invariant #2): MFTL2 standardizes on
**outward winding / `V > 0`** for all closed meshes. A small prerequisite commit — flip the
sphere generator to outward (`src/general/sphere.c`) + flip 8 pinned inward-sign assertions
in `test_object_integrals.c` / `test_transforms.c` / `test_remesh_parallel.c` — must land
**before Stage 11** so its overlap-normal logic is built on the single convention. With that
prerequisite applied, **Stage 11 can safely begin** (all other consumers are winding-agnostic;
no rework of Stages 05–10 needed).

**Stage 10** (parallel remeshing + x-only load balancing) is accepted: MFTL1-style two-pass
shift-scheme + centering (gather-to-rank-0) dispatcher, single-owner interior gates,
disjoint per-rank GID blocks, deterministic slab optimizer. Volume rank-independent
(drift ≤ ~7e-11 on 1/2/4/8); Release + ASan/UBSan clean; no regressions. It **resolved the
Stage-09 carry-forwards**: serial edge-local smoothers now fail loudly on distributed
meshes and the driver uses parallel-safe `*Parallel` variants gated to the interior rule;
`roughness_mean/variance` now cross ranks (`PackedVertex`/`GhostUpdatePayload`).
**Stage 10 carry-forward:** (a) **RESOLVED (2026-09-21, reviewed by Claude Code):** MFTL1's
`MFTL_RemeshBoundaryCleanup` is now implemented (`params->boundary_cleanup`, default on →
`MFTLRemeshEnsembleDispersedCentering` gather→serial-kernel→scatter, correctly forcing
`lb_nprocs = 0` during the gather). It fixes both the boundary-coverage gap the winding flip
exposed and the rank-count-dependent triangle count: the remeshed count is now
decomposition-independent (1414/1436/1414/1436 on np1/2/4/8, ~1.5% spread vs the previous
~13%) with exact boundary coverage (`hard==0` long edges on all rank counts). Residual
**should-fix (not blocking):** the cleanup gathers all triangles to rank 0 each remesh call
(MFTL1's default behaviour, togglable) — a rank-0 serialization/memory cost at large mesh ×
rank counts; consider amortizing or a distributed variant later. (b) transient
non-neighbour-owner ghosts during LB passes are kept stale until the next redistribution
(negligible volume impact, no topology corruption).

**Stage 09** (volume-conserving smoothing) is accepted: TSUR3D / Kuprat / VolumeRecovery /
roughness smoothing, MFTL1-consistent, full remesh cycle conserves volume to ~6e-12 (the
driver always ends with `VolumeRecovery`). For **Stage 13**: TSUR's `[global_min,global_max]`
endpoint clamp is periodic-unaware. (Mean-curvature-flow `LaplacianSmoothing`, which needs
the signed Laplace–Beltrami vector, was out of Stage-09 scope and not implemented.)

The two Stage 07/08 follow-up items are
now **RESOLVED** (reviewed by Claude Code; Release + ASan/UBSan clean incl. LAPACK):
(a) Stage 07 — `MFTLPointInsideObject` builds the LBVH ONCE before the retry loop and
reuses it (regression `test_bvh_built_once` asserts builds == queries across a forced
retry); (b) Stage 08 — the remesh driver auto-runs `MFTLDetectFeatures` every pass, and
collapse is now feature-protected flag-independently (a geometric sharp-edge gate in
`CollapseApproved` + feature-line/hard-corner gating; the fit paths use the geometric
`mftlEdgeIsFeatureEx`), so sharp features survive even without caller pre-flagging
(regression `test_collapse_geometric_gate_no_flags`). Geometric feature detection is a
no-op on smooth surfaces and MFTL1-consistent on coarse curved ones.

**Stage 10 is accepted** (parallel remeshing + x-only load balancing + two-pass boundary
shift; full details and acceptance record in `plans/10`). Carry-forward items for
**Stages 11–15**: (a) load balancing requires an x-only decomposition (`cuts={nproc,1,1}`);
other grids fall back to interior-only remeshing with a rank-0 note. (b) The WIP
freeze-band/coarsening scaffolding from HEAD was removed rather than completed — its
constraint system conflicts with the approved MFTL1 design (recorded in plan 10). (c)
Ghost contract change: `MFTLUpdateGhostPositions` no longer fabricates zero payloads or
dies on a missing owned GID — it reports `found == 0` for entities killed by parallel
remesh ops and skips ghosts whose owners are temporarily non-neighbouring (transient
slab-spanning edges); both refresh at the next redistribution. (d) `VolumeRecovery` is
parallel-exact (corner-weighted area, secant slope, consensus loop exits) but still
serial-bit-identical at np1. (e) Deferred: per-object centroid ranks for the centering
kernel; TSUR's periodic-unaware endpoint clamp (Stage 13). **Update (2026-09-21, winding
prerequisite):** MFTL1's `MFTL_RemeshBoundaryCleanup` is now implemented (the
gather→serial-kernel→scatter path in `MFTLRemeshEnsembleDispersedCentering`, driven by
`params->boundary_cleanup`, default on) — it covers the sliver edges that straddle both
the old and the shifted cut, which the two-pass shift provably cannot reach. The same
change fixed two latent defects in the centering path: the temp single-partition
communicator now uses `MPI_Comm_dup` (a 1-cell `MPI_Cart_create` handed `MPI_COMM_NULL`
to non-zero ranks), it clears any active `lb_*` slab ownership, and it syncs GID
counters between gather and serial kernel (rank 0 could otherwise mint colliding GIDs).

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

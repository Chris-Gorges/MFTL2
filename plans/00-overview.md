# 00 — Overview: reimplementing MFTL1 functionality in MFTL2

Status: IN PROGRESS — Stages 01–13 accepted; 14 IMPLEMENTED (awaiting independent acceptance); 15 ACCEPTED (independent review 2026-09-23); 16 (generalized surfaces) **ACCEPTED / COMPLETE (2026-09-25)** — 16A–16E ACCEPTED (independent reviews 2026-09-24/25), 16F final validation ACCEPTED (2026-09-25, `plans/16` §28); 17 (cleanup + manual) planned; 18 (performance) to be planned
Depends on: —
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Purpose

Bring MFTL2 to **full functional parity with MFTL1** (including Cartesian periodicity) and
migrate the `multiflow` CFD solver onto it, in independently-reviewable stages that each
leave the tree compiling and testable. This file is the shared design contract; every stage
`01…15` refers back to the invariants, conventions, and validation strategy defined here.

## Design and migration principles

**MFTL1 defines required numerical behaviour, not required implementation.** MFTL2
implementations should exploit the MFTL2 data structures, ownership model, and communication
design, and should not reproduce MFTL1 implementation patterns unless necessary for numerical
correctness or compatibility.

**During the initial migration, correctness and numerical parity take precedence over
optimization.** Known performance compromises — including serial fallbacks, unnecessary global
gathers, temporary representation conversions, redundant rebuilds, or other scalability
limitations — must be documented explicitly so they can be revisited during subsequent profiling
and optimization. Such compromises are acceptable when they provide a clear, tested path to
correctness, but they must not become implicit architectural requirements.

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
| `Triangle->Data` + `Object->SizeOfData` | `MFTL_Triangle.data` + `MFTL_Object.data_size` (Stage 12, implemented) |
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
   pointers — Stage 12 (implemented: `MFTL_Object.data_size`, `MFTLInitializeObjectData`,
   parallel `MPI_BYTE` streams alongside every triangle transport, `MFTLScatterTriangleData`).
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

**Progress (2026-09-22):** Stages **01–13 COMPLETE / Acceptance PASSED** (implemented by
Kimi Code, reviewed by Claude Code; Release + ASan/UBSan clean — incl. LAPACK/OpenBLAS
for Stage 08's paraboloid fit — full suite green on 1/2/4/8 ranks). **Stage 13 ACCEPTED
(2026-09-22, Claude Code):** end-to-end periodicity across integrals, curvature, point
queries, remeshing and intersection, plus the runtime minimum-image constraint — default
(LAPACK-off) build Release **53/53** and ASan/UBSan/LSan **53/53** on 1/2/4/8, independently
reproduced; non-periodic behaviour verified **bit-identical** (LAPACK-off `RemeshQuality`
deviation `3.051e-03` byte-matches the Stage-10 baseline; every periodic edit reduces to the
raw formula when no axis is periodic). Its verification exposed and fixed a **redistribution
heap-corruption bug** (both `MFTLRedistribute*` variants): stale ghost-vertex copies with
wrong owner labels (systematic for periodic seam-straddlers — wrapped storage makes their
owners non-neighbour ranks) could write one gid as both local and ghost, overflowing the
rebuilt vertex array; the receiver now classifies ownership only from sender-authoritative
copies with hash-based dedup (`write_count==n_unique`, exactly-once ownership), and non-ghost
vertices without a non-ghost incident triangle are shipped explicitly. Fix stress-verified:
wide seam-straddler `PeriodicEndToEnd` np8 ASan ×15 clean, non-periodic `RemeshParallel`/
`TriangleData` np8 ASan clean. **Carry-forward RESOLVED (2026-09-22):** the LAPACK-on
`test_vertex_fitting_validity` failure (`5.032e-02` vs `0.05`) was a genuine Stage-08 port
bug, not a tolerance issue: the paraboloid fit stencil (`mftlGatherFitNeighbours`) excluded
the edge endpoints and used the asymmetric one-ring union, while MFTL1's `ParaboloidFit`
fits the symmetric diamond stencil (both endpoints + adjacent/complementary triangle
vertices) gathered from the intact edge BEFORE the split. Restored to MFTL1 parity in
`vertex_fit.c` (split path gathers stencil/frame pre-split); deviation is now `5.5e-05`
(barycentric: `3.1e-03`) and the tolerance stays `0.05`. The test now also destroys the
ensemble before asserting (no misleading LSan leak on failure). Verified: Release +
ASan/UBSan/LSan 53/53 in BOTH LAPACK-off and LAPACK-on builds. **Stage 14 IMPLEMENTED (2026-09-23, incl. final-fix round)** — redesigned
MFTL-facing architecture, awaiting independent acceptance (see the Stage 14 paragraph below and
`plans/14`, "Stage 14 completion"). **Stage 15 IMPLEMENTED (2026-09-23)** on the owner's
instruction while the Stage-14 review is still pending — multiflow runs on MFTL2 only;
**independently reviewed 2026-09-23, Acceptance: PASSED** (see the Stage 15 paragraph below and
`plans/15`, "Independent review").

**Stage 14** (front-tracking integration layer in multiflow, branch `berend-matrix`) —
**IMPLEMENTED, pending review.** The 2026-09-22 review (M1–M3, S1–S5) is addressed by a
redesign rather than a wrapper: `src/include/mftl_adapter.h` is a backend-independent contract
(multiflow physics → integration layer → backend), bound to MFTL1 today through
`mftl_adapter_mftl1{.h,_inline.h,.c}`; Stage 15 supplies the same three files for MFTL2.
- *Object model:* `MFAdapterDecomposition` (multiflow-owned rectilinear rank boxes built from
  the DMDA overlap mesh → MFTL2 `MFTL_Domain`), `MFAdapterBodyList` (bodies sharing one
  decomposition/payload size → one `MFTL_Ensemble`), `MFAdapterMesh` (opaque body; standalone
  bodies for serial IBM marker meshes). multiflow no longer stores `struct MFTL_Object` or
  chains bodies through MFTL1 hashes.
- *Handles:* MFTL2 tri/vert handles are pointers into the object arrays, edges the canonical
  half-edge; mesh-context accessors where the record is insufficient (`TriCentreProc`);
  canonical undirected edge id incl. the ghost rim; explicit invalidation rules.
- *Decomposition/communication, mutation/construction, point queries, remeshing:* all 71
  `ownershiprange*`, 16 `globalx*`, 25 `OBJECT_COMM_WORLD` reach-ins removed; vertex writes via
  `MFAdapterVertPosMut`; builder + component copy/merge + neighbour completion + ghost-position
  refresh as layer operations; build-once/query-many batch point queries and plain AABBs
  (aligned/orthogonal no longer an API choice); backend-independent remesh settings.
- *MFTL2 CMake:* target-scoped flags; portable by default (`-march=native` opt-in, GCC fat-LTO
  objects), sanitizers only for Debug dev builds with an exported INTERFACE link flag,
  tests/examples only when top-level; `find_package(MFTL2)` verified from build tree, install
  prefix and an instrumented Debug archive, `add_subdirectory` verified; 53/53 Release and
  Debug+ASan/UBSan; two latent `-Walloc-size-larger-than` sites fixed behaviour-identically.
- *Verification (MFTL1 backend):* all golden baselines pass with the working tree (FT, FT_Sharp_CSF,
  FT_Sharp_Integral, IBM np1/np4 without MFTL); HEAD-vs-working-tree A/B bit-identical on
  np1 and np4 for FT, FT_Sharp_CSF, FT_Sharp_Integral and both IBM-MFTL marker paths (MoveIBM
  body creation also identical); new integration-layer unit tests
  (`multiflow/tools/mftladapter_tests`) 748 checks pass on 1–4 ranks. *Coverage gap:* the
  marching-cubes reconstruction and `IsPointInsideOfParticle` could not be A/B-tested end to
  end (the topology-change example stalls in `ExecuteMarchingCubes` identically at HEAD; MoveIBM
  runs exceed 25 min per step) — covered by unit tests and review only.

**Explicit Stage-15 carry-forwards (from Stage 14):**
1. MFTL2 backend files + MFTL2 library items: L1 rectilinear per-axis cuts/rank map in
   `MFTL_Domain`; L2 caller-held point-query BVH (batch); L3 retain `twin_gid` in
   `MFTL_HalfEdge`; L4 detach an object from an ensemble; L5 single-vertex normal /
   Laplace–Beltrami; optional L6 spatial ghosts for cross-rank proximity (behaviour change).
2. Convert the remaining per-element reach-ins in ≈25 FT files (each has an adapter
   equivalent) and move math/memory helpers to a shim; `polygonise.c` onto the builder;
   `printfrontmesh.c` output.
3. **L7 — conserved surface scalar (PRESERVE current behaviour):** MFTL1's `scalarconcentration`
   is real solver state, now the adapter concept `MFAdapterTriSurfaceScalar` (all 20 multiflow
   sites converted). MFTL2 needs either a native library-maintained triangle scalar or
   payload-remapping hooks that reproduce MFTL1 exactly: transport with the triangle,
   mass-conserving remap in split/flip/collapse/TSUR/volume-conserving smoothing, restart I/O.
   Not ordinary zero-initialised payload semantics; not a physics change.
4. Dual decomposition stays top risk #2 (now an explicit, tested value type); MFTL1-only
   `mftl1_init_grid` / replicated placement to be dropped. Periodic default = MFTL1 behaviour:
   the front decomposition is non-periodic (`periodic` = 0); enabling periodic fronts is a
   separate, explicit change (risk 7). Remeshing mode is an explicit enum
   (serial / parallel / parallel + load balancing).
5. Build/link: switch multiflow to `find_package(MFTL2)`/`MFTL2::MFTL2`; MFTL1 and MFTL2 export
   same-named symbols, so the swap is atomic (never link both).
6. Validation: iteration order changes break bit-exact goldens → the analytic FT regressions
   (Lamb droplet, rising bubble, LeVeque) must exist before the swap; the topology-change
   example is too slow for routine A/B at full resolution.

**Stage 15** (MFTL1 → MFTL2 swap, multiflow branch `berend-mftl`) — **ACCEPTED (independent
review 2026-09-23, `plans/15` §R1–R9); nothing committed.** multiflow links only `MFTL2::MFTL2` (`find_package(MFTL2)`,
`MFTL2_ROOT`; MFTL2 must be built with `MFTL_USE_LAPACK=ON` — enforced at configure time); the
MFTL1 backend of the integration layer is deleted and no MFTL1 identifier remains outside
comments (math helpers are local, `ftvectorops.h`).
- *Library:* Stage-14 carry-forwards L1–L5 and L7 implemented (L6 spatial ghosts deliberately not
  enabled — MFTL1 behaviour); soup builder, triangle copy/clear, native mesh I/O, serial bracket,
  remesh switches; **MFTL1 edge criteria and sweep order** (`mftl1_criteria.c`, opt-in) and a
  paraboloid-fit parity mode. Bugs fixed with evidence: TSUR endpoint normals not normalised
  (Stage 09 port; now 4e-16 vs MFTL1 with MFTL1's order), area/normal floor at front scales,
  sphere-generator lookups, parallel split-phase gid blocks with neighbourhood-reading criteria,
  soup-builder corruption, uninitialised remesh parameters, parallel TSUR scalar gate,
  post-cleanup twin gids, `I` parameter vs complex.h.
- *Validation:* MFTL2 ctest 57/57 Release and ASan/UBSan (LAPACK on); Stage-15 library test 78
  checks on 1–4 ranks (LAPACK on and off); integration-layer tests 867 checks on 1–4 ranks;
  multiflow vs MFTL1: FT_Sharp_CSF, FT_Sharp_Integral and IBM marker cases within the golden
  tolerance on 1 and 4 ranks; FT (remeshing) within MFTL1's own np1-vs-np4 spread for volume,
  rise velocity and centroid; initial front identical; the golden baselines fail — *corrected by the review:* not because of any MFTL1 Debug/Release
  dependence (there is none: HEAD-MFTL1 Debug vs Release 3.6e-8 / 3.2e-7) but because of the
  `origin/master` merge 10dce627 (pre-merge a7cb9179 on MFTL1 reproduces the goldens to 1e-8;
  HEAD on MFTL1 fails them exactly as MFTL2 does) — baselines to be regenerated on MFTL2 (owner
  decision); one full remesh pass identical to MFTL1 in
  all refining configurations tested; wall time equal to MFTL1 (FT np1 274.7 s vs 276 s).
- *Limitations / compromises:* collapse survivor and tie order follow MFTL1's convention only
  for edges as MFTL1 builds them (per-pass collapse counts can differ in coarsening-heavy passes);
  MFTL2 owns triangles by centre and always keeps one ghost layer; MFTL1's random-sign
  second-order terms in Kuprat smoothing are not reproduced; the parallel remesh uses MFTL1's
  criteria in MFTL2's sweep. Full list in `plans/15` §6–§8.
- *Review (2026-09-23):* MFTL2 ≡ MFTL1 physically (FT volume/rise velocity within MFTL1's own
  np1-vs-np4 spread on 1 and 4 ranks, serial and parallel remesh; FT_Sharp_Integral ≤ 5e-7);
  exact collapse-survivor parity NOT needed; should-fix items (SERIAL mode ignores
  boundary_cleanup, guaranteed ghost completion for assembled bodies, redundant redistribution,
  test_stage15 instrumentation) and coverage gaps (marching cubes / MoveIBM end to end) listed in
  `plans/15` §R6–R7 — now Stage 17 (renumbered), except R6.2 (guaranteed ghost completion),
  which is folded into Stage 16A.
- *Deferred to Stage 17:* remaining scale-dependent absolute thresholds, the pre-existing zero-mass result of
  multiflow's surface-scalar solver in the FT test configuration, repository-wide formatting.

**Stage 16 (generalized surfaces: periodic and open triangulated objects) — ACCEPTED / COMPLETE
(2026-09-25; final validation 16F, `plans/16` §28):
16A ACCEPTED (independent review 2026-09-24); §3.6 revision 3.2 INDEPENDENTLY REVIEWED AND FROZEN
(2026-09-24, errata E1–E8); 16B ACCEPTED (independent review 2026-09-24, with fixes to
winding validity, periodic transforms and the generated-test build; `plans/16` §20, §21); 16C
ACCEPTED (independent review 2026-09-24, after one fix; `plans/16` §22, §23); 16D ACCEPTED
(independent review 2026-09-25, after fixes R16D-1 to R16D-5; `plans/16` §24, §25); 16E
ACCEPTED (independent review 2026-09-25, after corrections C16E-1 to C16E-6; `plans/16` §26,
§27; gap C16E-7 closed in 16F); 16F ACCEPTED (2026-09-25, `plans/16` §28)**
(`plans/16-generalized-surfaces.md`).
One surface model for both
capabilities and their combination. Substages, each gated (closed non-periodic behaviour
bit-identical to Stage 15 at every gate):
- *16A topology/invariants:* three strict half-edge states — LOCAL, REMOTE (MPI rim,
  `twin_gid ≥ 0` naming a live twin), BOUNDARY (`twin_gid = −2 − tag`); no unknown state;
  maintained explicitly by every operator (incl. `glue_wings`, copy/commit ASSEMBLE/CUT) and
  validated collectively at quiescent points; remote-twin discovery (global mode with any
  authoritative holder before ownership exists; after redistribution geometric candidates
  with the global maximum edge length + margin, then an always-executed global fallback) giving guaranteed ghost completion (every edge of an owned
  triangle LOCAL or BOUNDARY; complete owned fans); manifold/orientable builder validation;
  distributed boundary loops with persistent tags.
- *16B periodic image representation:* persistent per-vertex per-axis image counters
  (`img[3]`, continuous coordinate `pos + img·L` evaluated relative to reference points; the
  only structural change — `PackedVertex` and native I/O v2), every vertex update a
  minimum-image displacement from an anchor, homological winding (cycle sums of edge lattice
  shifts) with canonicalisation after every assembly event, no object-extent restriction (only
  per-edge < half period); phase-boundary sets with DEFINED / UNDEFINED / UNSUPPORTED status:
  separating iff closed (or wall-closed), net flux/homology Φ(S) = 0 and consistent
  orientation; evaluation along an admissible axis in a canonical relative gauge, otherwise
  UNSUPPORTED; class-guarded remesh drivers.
- *16C kernels + remeshing:* local-frame kernels, boundary conormal/tangent geometry, curvature
  validity flags, boundary curve (represented spline or application projector), corners,
  augmented (virtual-vertex) link condition for collapses in serial and parallel.
- *16D multiflow integration:* explicit `PeriodicFronts` / `OpenSurface` / phase sets,
  enforced capability matrix, adapter class checks (no bypass), per-body remesh criteria
  (closed bodies → Stage-15 parity periodic or not; open → generalized; closed generalized
  only by opt-in), periodic `FTDomain`, conversion of the
  131 raw-position sites, marching cubes in continuous coordinates, IBM open bodies only for
  force coupling or with explicit mass properties. Review fixes (§25): corner-threshold
  initialisation in `MFTLInitRemeshParams`; generalized bodies always remeshed with the two-pass
  shift scheme; capability re-check of the bodies created by marching cubes; periodic marching
  cubes at factor 1 only (factor > 1 rejected); iterated volume recovery after a periodic
  reconstruction. D5 and D6 run and passed.
- *16E boundary models:* surface-scalar NEUMANN/DIRICHLET loop BCs; surface tension REJECT
  (default) / PINNED (reaction on a declared support), CSF only when pinned on a wall closure;
  contact lines reserved for a later stage. Implemented 2026-09-25 (§26): per-tag
  `SurfaceScalarBC`, `SurfaceTensionBoundary … PINNED STATIC` (static supports; IBM/DEM supports
  UNSUPPORTED in Stage 16), `PhaseSetMember` + `WallClosure` (Poisson indicator with Neumann
  closure faces, wall-closed volume recovery, CSF/IRL non-periodic only), extended capability
  matrix. Review corrections (§27): Frenet surface tension on a pinned open surface no longer needs
  the indicator (single-phase fluid; the indicator/CSF/volume rows keep the wall closure); the
  Dirichlet distance is the normal distance h_T/3 (the midpoint distance was first order on skewed
  boundary triangles) with the sliver floor max(δ_e, 0.1 L_e) (the frozen 0.1 h_T could never
  act); multiflow CMake composes `CMAKE_C_FLAGS` instead of replacing it (sanitizers now really
  instrument multiflow) and `FT_SCALAR_TRANSPORT` is the option `MF3_FT_SCALAR_TRANSPORT`; the
  MFTL2 parallel remesh clamp keeps declared-open bodies in the closed box (pinned rims stay exactly
  on their wall); the validation is versioned in `multiflow/testcases/FT_Stage16E`.
- *16F validation:* invariant fuzzing with exactly-once ownership, link-condition pinch/ear/loop
  cases, dyadic seam/corner translation invariance, ribbon/film/slab phase sets, Gauss–Bonnet,
  pinned-cap force balance, Dirichlet diffusion, scalar conservation, mixed ensembles, restart on
  a different rank count, geometry scales 1e-4/1e4, np 1/2/4/8 incl. 1- and 2-slab periodic
  decompositions. **Executed and accepted 2026-09-25 (§28):** the phase set is re-classified after every
  geometry change a phase-dependent module relies on (C16E-7: advection, remeshing, volume
  recovery; a wall-closed cap pushed through its wall stops before the indicator uses the invalid
  set); minimal fixes: unchecked volume / inertia also reject open bodies (F16F-1), the abort
  message is collective (F16F-2), suite scripts (F16F-3); all D5/D6/marching-cubes/restart/A-B inputs
  versioned (`multiflow/testcases/FT_Stage16D`, reproducing the accepted 16D outputs bit for bit),
  new wrapping-topology test `test_stage16f`; closed legacy A/B bit-identical; periodic
  seam/edge/corner translation exact to 1e-15; 1 000-step invariant fuzz on 1/2/4/8 ranks;
  genuinely instrumented multiflow ASan/UBSan clean; capability matrix audited, every requirement
  classified (§28.13, §28.14). **Stage-16 summary: §28.15.**

**Stage 17 (code-quality/documentation cleanup + MFTL2 manual; formerly Stage 16, renumbered
2026-09-23) — PLANNED, NOT STARTED.** Runs **only after Stage 16 is complete and accepted**,
when the code, public API (incl. the generalized surface model) and multiflow coupling are
stable. Purpose: a repository-wide, **behaviour-preserving** cleanup —
(17A) comments/file-header standardization and terminology, (17B) internal naming on
lowerCamelCase (public API, types, macros, struct fields, packed layouts frozen; conventional
short math/index names kept), (17C) structural cleanup in four individually gated sub-steps —
17C1 directory/file moves, 17C2 large-file splits, 17C3 de-duplication/shared helpers
(bit-identical; never combined with the `communication.c` move/split), 17C4 public/private
header separation (only declarations verified unconsumed by multiflow, examples, tests and
external `find_package(MFTL2)` consumers may move) — and (17D) a new LaTeX manual at
`doc/MFTL2.tex` conceptually modelled on MFTL1's `mftl-doc/MFTL.tex` (prose, equations,
figures, compilable examples) but describing the actual post-Stage-16
half-edge/array/`khash`/periodic/open-surface MFTL2, with a documented `MFTL2.pdf` build
target, closed by (17E) a full consistency + regression gate (any repo-wide formatting is one
isolated final commit). No intended numerical change; no public-API
change without explicit approval; behaviour/perf fixes surfaced during cleanup (linear-search→hash,
boundary-cleanup distributed variant, point-query BVH cache, paraboloid tolerance) are
**documented and deferred** to Stage 18, not performed. Also carries the Stage-15 review
should-fix items R6.1, R6.3, R6.4 and the R7 coverage gaps. Details in
`plans/17-code-quality-documentation.md`.

**Stage 18 (performance) — to be planned** after Stage 17: the deferred performance items
(point-query BVH cache, distributed boundary cleanup, serial TSUR/VCS gathers, per-pass edge
sort, construction all-gathers, full redistribution in ghost-position updates). The **winding convention is RESOLVED and APPLIED** (invariant #2): MFTL2
standardizes on **outward winding / `V > 0`** for all closed meshes; the sphere generator
and the affected sign-pinned tests have been updated and independently reviewed.

**Stage 11** (object–object intersection + overlap-volume) is accepted: faithful MFTL1
overlap-volume port — two-sphere lens converges to analytic (ref5 0.06%), sign correct
(far-side/non-overlap → V=0), `V(A,B)==V(B,A)` bit-identical, rank-independent (A-ownership
partition + gathered deterministic list), periodic seam == mid-domain bit-identical (bodies
unwrapped into a common frame without mutating the mesh; uses outward normals). Release +
ASan/UBSan/LSan clean on 1/2/4/8 (7/7); no unrelated regressions. **Carry-forward (Stage 13):
RESOLVED (2026-09-22, implemented):** the BVH/spatial-ghost *detection* path
(`MFTLEnsembleIntersectObjects`, `mftl_bvh.c:535`) is
now periodic-correct — spatial-ghost interval overlap and BVH pair traversal use
periodic-aware interval overlap, and `MFTL_TriTriIntersect`/`Proximity` materialize a
common per-triangle frame when periodic (verified by `PeriodicEndToEnd` seam intersection).

**Stage 12** (opaque per-triangle `data` transport) is accepted: byte-exact `MPI_BYTE`
transport across both redistribution routes, ghost extension, spatial ghosting and
`MFTLScatterTriangleData`; per-triangle heap blobs with a correct alloc/free lifecycle
(split→zeroed blob, PATH-B compaction frees dead-only, `Destroy` frees all); `data_size==0`
no-op. Release + ASan/UBSan/LSan clean on 1/2/4/8; no regressions. **Carry-forwards RESOLVED
(2026-09-21):** (a) `MFTLCheckTriangleDataSizeConsistency` — a cheap collective MIN/MAX
Allreduce of `data_size` over `comm_cart` at the entry of every triangle-data transport path
(both redistribution routes, ghost-layer extension, ensemble + pair spatial ghosting, scatter);
a cross-rank mismatch now fails loudly with `MFTL_ERROR` before anything is communicated or
modified (negative test: deliberate 56-vs-64-byte mismatch refused uniformly by all four
entry points on all ranks, mesh untouched). (b) Remesh-with-data regression test
(`TestRemeshingTriangleData` in `test_remeshing.c`, np1, under LSAN): explicit
split×3/flip/collapse + PATH-B compaction sequence plus one full serial
`MFTLRemeshEnsemble` pass with data attached — surviving gids byte-identical, new split
gids fully zeroed, no leaks/double-frees.

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
driver always ends with `VolumeRecovery`). For **Stage 13** — **RESOLVED (2026-09-22,
implemented):** TSUR's `[global_min,global_max]` endpoint clamp is now periodic-aware (the
clamp is skipped on periodic axes; corner unwraps, ring areas, projection midpoint and tetra
legs all use the minimum image). (Mean-curvature-flow `LaplacianSmoothing`, which needs
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
kernel; TSUR's periodic-unaware endpoint clamp (Stage 13 — RESOLVED 2026-09-22, clamp
skipped on periodic axes). **Update (2026-09-21, winding
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
`14` CFD adapter (early, vs MFTL1) → `15` MFTL1→MFTL2 swap (depends on 02–14; both implemented 2026-09-23; 15 accepted 2026-09-23, 14 awaiting independent acceptance).
`{13,15}` (accepted) → `16` generalized surfaces (periodic + open) → `17` code-quality/documentation cleanup + MFTL2 manual → `18` performance.

**Critical path to a usable, faster, periodic library**: 01→02→05→06→08→09→10→13.
**Migration path**: 14 (early, low-risk) then 15 (last).
**Capability path**: 16 (periodic and open triangulated surfaces, after the migration).
**Polish path**: 17 (after 16 — behaviour-preserving cleanup + manual), then 18 (performance).

## Top technical risks

1. Parallel remeshing (10) — least mature, currently broken.
2. Dual decomposition + `sendnonlocal*` bridge (15) — silent front↔fluid corruption if wrong.
3. Opaque `Data` byte-transport (12) — pointer/size mismatch corrupts across ranks silently.
4. Curvature/tolerance semantics (06/01) — subtle remeshing-behavior drift.
5. LAPACK dependency (08) — new build dependency.
6. Validation is being *added*, not ported — physics tolerances must be meaningful yet stable.
7. multiflow periodicity gaps — enabling MFTL2 periodicity must not change existing runs.

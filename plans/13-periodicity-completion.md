# 13 — Periodicity completion end-to-end

Status: COMPLETE
Depends on: 02, 06, 07, 10, 11
Implemented by: Kimi Code
Reviewed by: Claude Code (2026-09-22)
Acceptance: PASSED

## Objective
Ensure Cartesian periodicity (X/Y/Z, any combination) is correct across **every** subsystem —
integrals, curvature, point queries, remeshing, and intersection — not just the primitives.
Resolve the outstanding periodic-correctness question in the BVH and add end-to-end periodic
validation.

## Relevant existing code
- MFTL2 primitives (correct, tested): `MFTLGetMinimumImageVector`, `MFTLApplyPeriodicity`,
  `MFTL_Domain.is_periodic[3]`/`domain_half_size[3]`; `MPI_Cart_create` with `periods[]`
  (`src/general/ensemble.c`); `test_internals.c` (`test_apply_periodicity`, `test_min_image_vector`,
  `test_edge_length`). Open item: `src/general/mftl_bvh.c:535`
  "ATTENTION: IS THIS CORRECT FOR PERIODIC DOMAINS?".
- Consumers wired in earlier stages: 02 (integrals unwrap per triangle), 06 (curvature one-ring),
  07 (point-in-object nearest image), 08–10 (remesh edge length / midpoint via min-image), 11
  (overlap min-image shift of B into A).

## Proposed changes
- **Audit + wire**: verify each stage's periodic path is exercised, and fill any gaps:
  - Integrals (02): per-triangle unwrap relative to `v₁`.
  - Curvature (06): one-ring computed on min-image-unwrapped neighbours across seams.
  - Point-in-object (07): ray-cast against the nearest periodic image of the object; periodic
    ray re-entry or image selection.
  - Remeshing (08–10): edge length/midpoint/fit use `MFTLGetMinimumImageVector`; new/moved
    vertices optionally `MFTLApplyPeriodicity`-wrapped; disable any wall-clamp on periodic axes.
  - Intersection (11): min-image shift of B into A's frame; broad phase tests periodic images.
- **BVH periodic correctness** (`mftl_bvh.c:535`): make LBVH node bounds / traversal periodic-aware
  (either build per-image query boxes or min-image the query), and replace the flagged assumption
  with verified logic + a periodic intersection test.
- **Redistribution/ghosting across seams**: confirm `MFTLGetOwnerProc` wraps positions on
  periodic axes (via `MPI_Cart` periodic neighbours) and that ghost layers form across seams
  with correctly shifted coordinates (unwrapped local frame — coordinate with the storage
  convention: MFTL2 keeps wrapped positions + min-image geometry, so ghosts across a seam are
  handled by min-image at use sites rather than by shifting stored coordinates; verify this holds
  for curvature/remesh stencils).

## Files/functions likely to change
`src/general/mftl_bvh.c`, `src/general/ensemble.c`/`communication.c` (owner wrap + periodic ghost
formation), plus periodic branches in the Stage 02/06/07/08–11 units. Test
`tests/test_periodic_endtoend.c`.

## Numerical/physical considerations
- **Minimum-image constraint**: period on each periodic axis must exceed twice the largest
  feature/edge/ghost-neighbourhood extent, so "nearest image" is unambiguous. Add a runtime
  assertion in periodic setup and document it (LeVeque/advection cases satisfy it).
- Conserved quantities (volume, mass) must be identical whether an object sits mid-domain or
  straddles a seam.

## MPI/PETSc considerations
- `MPI_Cart` periodic neighbours already provide the wrap for redistribution; ensure ownership
  lookup and ghost exchange use them on periodic axes only.
- Rank-independence must hold for periodic runs too (1/2/4/8), including the seam-crossing case.
- Downstream (Stage 15): multiflow front-tracking is currently non-periodic; enabling MFTL2
  periodicity must be opt-in per object and must not change non-periodic behavior.

## Possible failure modes
- A subsystem still reading raw coordinate differences across a seam → wrong geometry only when
  straddling; the straddling tests target each subsystem.
- Period too small vs feature size → ambiguous min-image; runtime assert.
- BVH broad-phase missing a cross-seam pair → periodic intersection test catches it.
- Ghosts across a seam with unshifted coordinates breaking a curvature/remesh stencil → verify
  the min-image-at-use-site convention covers 2-ring stencils.

## Unit/regression tests to add (`tests/test_periodic_endtoend.c`)
- **Seam invariance battery**: for X, Y, Z, XY, XYZ periodicity, place a sphere straddling the
  seam and assert volume/area/centroid(mod period)/inertia/curvature equal the mid-domain values
  (ties together Stages 02/06).
- **Periodic advection**: advect a sphere a full period along X (and diagonally) and back;
  volume and triangle count conserved; no "outside domain" errors; rank-independent on 1/2/4/8.
- **Periodic LeVeque**: single-vortex deformation with the drop crossing a periodic face;
  enclosed-volume drift within tolerance; returns toward initial shape at t=T.
- **Periodic intersection**: two objects overlapping only across a seam detected and quantified
  (ties to Stage 11); BVH `:535` case verified.
- Minimum-image assertion fires when period < 2×feature.

## Acceptance criteria
- All seam-straddling integrals/curvature/queries equal mid-domain values within tolerance, for
  every axis combination.
- Periodic advection/LeVeque conserve volume and are rank-independent; periodic intersection
  correct; BVH periodic assumption resolved.
- Runtime minimum-image constraint asserted.

## Implementation summary (2026-09-22)

Convention confirmed and applied everywhere: vertices are stored **wrapped** in-domain;
seam-straddling triangles are "stretched" in raw coordinates; all geometry uses the
minimum image at use sites. New helpers `MFTLDomainAnyPeriodic` and `MFTLUnwrapToAnchor`
(`include/mftl_geometry.h`) are exact copies / raw differences on non-periodic domains, so
**non-periodic behavior is bit-identical** (branches on `MFTLDomainAnyPeriodic` only where the
periodic-safe formula differs in floating point).

Fix areas:
- **Curvature (06)**: `mftlTriangleCorners` and the mean-curvature sum unwrap one-ring corners
  to the anchor vertex; `MFTLComputeRoughness` geometric-Laplacian unwraps neighbours
  (`src/general/curvature.c`).
- **Volume-conserving smoothing (09)**: min-image edges in TSUR3D corner unwraps, ring areas,
  projection midpoint, tetra legs; the `[global_min,global_max]` endpoint clamp now skips
  periodic axes; Kuprat endpoint data unwraps to an anchor; `MFTLVolumeRecovery` re-wraps after
  the normal shift; edge-VCS uses min-image edge vectors and re-wraps after moves
  (`src/remeshing/volume_conserving.c`).
- **Remeshing core (08–10)**: split/collapse midpoints and tangential-smoothing ring sums are
  seam-safe with `MFTLApplyPeriodicity` applied to new/moved vertices; `MFTL_TriangleQuality`,
  unit normals and the flip dihedral gate use minimum-image vectors
  (`src/remeshing/remeshing_core.c`); barycentric/paraboloid fits unwrap to the anchor and
  re-wrap fitted positions (`src/remeshing/vertex_fit.c`).
- **Point-in-object (07)**: on periodic domains `MFTLPointInsideObject` temporarily unwraps all
  local vertices to a collectively chosen anchor image (restored before return) and re-images
  the probe by integer periods; the non-periodic path is untouched
  (`src/general/point_query.c`).
- **BVH / intersection (11)**: the `mftl_bvh.c:535` open item is resolved by design — LBVH node
  bounds are built in the stretched embedding (conservative superset of the true periodic box),
  and periodicity is handled at the query level: spatial-ghost interval overlap and BVH pair
  traversal use periodic-aware 1-D interval overlap; `MFTL_TriTriIntersect`/`Proximity`
  materialize a common per-triangle frame (unwrap to own v0, integer-period shift of B toward
  A's centroid) when periodic. `MFTLEnsembleIntersectObjects` now fails loudly if
  `MFTL_CreateSpatialGhosts` fails (`src/general/mftl_bvh.c`).
- **Minimum-image constraint (runtime)**: new collective check
  `MFTLCheckMinimumImageConstraint(const MFTL_Object*)` (`src/general/geometry_object.c`,
  declared in `include/mftl_geometry.h`) — errors when the longest minimum-image owned edge
  reaches half a period, or when an unwrapped body's extent reaches half a period on any
  periodic axis; rank-0 diagnostics, no assert. Wired into `MFTLGenerateSphere` (returns
  `MFTL_ERROR`) and `MFTLGenerateCube` (loud warning; void return).

Tests: `tests/test_periodic_endtoend.c` registered as `PeriodicEndToEnd-np{1,2,4,8}` — seam
invariance battery (X/Y/Z/XY/XYZ; volume/area/inertia/centroid mod period/curvature/point
probes seam == mid), periodic advection (X and XYZ, forth and back, volume/triangle-count
conserved), periodic LeVeque single vortex on [0,1]³ with the drop crossing the seam (returns
to initial shape; relerr 4.5e-9, max displacement 2.4e-9), periodic remesh across the seam
(small R=0.10 centering-path case AND a wide R=0.7 shift-scheme seam-straddler — see the
follow-up below), periodic intersection + overlap across a seam, and the minimum-image
constraint firing.
Verification: Release `ctest` 53/53; ASan/UBSan/LSan `ctest` 53/53. Regression probes:
commenting out the TSUR corner unwrap breaks the remesh curvature gate; disabling the common
frame in `MFTL_TriTriIntersect` produces phantom seam hits (200 vs 60).

Deviations from the proposed test plan:
- ~~Periodic-remesh test uses R=0.10 instead of R=0.7~~ **RESOLVED (2026-09-22):** the wide
  seam-straddler corruption was root-caused and fixed (see below); the remesh test now covers
  BOTH the small centering-path sphere (R=0.10) and a wide shift-scheme seam-straddler
  (R=0.7 in [-2,2]³, spanning slabs {0,1,6,7} at np8).
- Intersection probes are asymmetric by design (a mirror-symmetric config produces
  near-degenerate `Orient3D` flips); overlap volume across a seam is asserted to 1e-6 relative
  (observed 2.7e-8, a near-degenerate duplicate-point branch in the Stage-11 prism machinery
  flipping under frame translation); the symmetric `test_overlap.c` case still pins 1e-9.
- `MFTLTranslateObject` does **not** wrap positions; callers use `MFTLWrapObjectIntoDomain`
  (the test does so explicitly).

### Follow-up (2026-09-22): redistribution heap corruption on wide periodic seam-straddlers

**Defect:** remeshing a wide periodic seam-straddler (R=0.7 in [-2,2]³, x-periodic, cuts
{8,1,1}) heap-corrupted at np8 inside the Stage-10 shift scheme (`free(): corrupted unsorted
chunks`), preceded by "110 ghost(s) have non-neighbour owners". Non-periodic identical
geometry was fine.

**Root cause** (`src/general/communication.c`, vertex reconstruction in
`MFTLRedistributeListOfObjects` / `MFTLRedistributeObject`): redistribution ships a vertex
copy with every non-ghost triangle touching it, labeled with an owner recomputed from the
SENDER's stored position. Wrapped storage stretches seam triangles across the whole domain
in raw coordinates, so their ghost vertices are owned by ranks 2+ slabs away — never
refreshed (the Stage-10 contract keeps non-neighbour-owner ghosts stale). When the owner
moves such a vertex across the seam wrap (a full-period jump), stale ghost copies carry a
wrong slab label. The receiver classified copies purely by that label with a
consecutive-duplicate dedup, so one gid could be written BOTH as local (fresh copy) and as
ghost (stale copy) — overflowing the `n_unique`-sized vertex array (ASan: heap-buffer-overflow
write of size 4 at the pass-2 ghost write) and clobbering `vertex_lookup` onto the ghost slot.

**Fix (no shift-scheme redesign):**
1. Senders now pack the TRUTHFUL sender-side `is_ghost` in `PackedVertex` (previously
   hardcoded FALSE; no receiver read it before), marking ownership-authoritative copies.
2. Receiver vertex reconstruction: a gid becomes LOCAL only when an authoritative
   (sender-non-ghost) copy labeled with this rank arrives; dedup is hash-based via
   `vertex_lookup` (so `write_count == n_unique` always, no overflow); ghost data prefers
   authoritative (freshest) copies. Stale copies can no longer create phantom locals or
   double ownership (the unique non-ghost holder always ships the fresh copy to the new
   owner — either with an incident triangle or via 3).
3. Authority completeness: non-ghost vertices with NO non-ghost incident triangle
   (transient near partition boundaries; observed routinely in periodic LeVeque, 1-6 per
   redistribute — they left vertices ownerless under rule 2 and broke volume conservation)
   are now explicitly sent to their recomputed owner.
4. Triangle reconstruction dedup converted to the same hash-based scheme (defensive;
   triangle copies are single-source so labels cannot conflict).

**Verification:** repro `/tmp/repro_remesh_r07.c` (periodic seam, np 1/2/4/8) and the
non-periodic mid-domain control (np 2/4/8) all complete with rc=0 under ASan/UBSan;
`test_periodic_remesh_wide_seam` pins the R=0.7 case (volume seam vs mid ~2e-8 apart,
rank-independent np1/2/4/8); periodic LeVeque volume drift back to 4.5e-9 with exactly-once
vertex ownership verified per step during debugging.

---

## Independent review (Claude Code, 2026-09-22) — ACCEPTED

**Verdict: Stage 13 is accepted.** No must-fix issues. Reviewed the full working-tree diff
(Stages 11–12 were already accepted; this review isolated the Stage-13 logic), rebuilt
Release + ASan/UBSan (both LAPACK-on and the default LAPACK-off), ran the full suite on
1/2/4/8 ranks, and stress-ran the wide seam-straddler.

**Builds:** 0 warnings, Release + ASan/UBSan, LAPACK on and off.

**Suite (default LAPACK-off build — the configuration `build_all.py` produces and the
implementer's "53/53" refers to):** Release **53/53**, ASan/UBSan/LSan **53/53**, clean on
1/2/4/8. Confirmed independently.

**One failure appears ONLY with `-DMFTL_USE_LAPACK=ON`** — `RemeshQuality-np1`
(`test_vertex_fitting_validity`): the paraboloid-fit radial deviation is `5.032e-02`, just
over the hard-coded `0.05`, and the failing `TEST_ASSERT` returns before
`MFTLEnsembleDestroy`, so LSan then reports the ensemble as leaked. **This is pre-existing and
NOT a Stage-13 regression:** a HEAD (Stage-10, `8a58cb4`) worktree built with LAPACK on
produces the byte-identical `5.032e-02` failure and the same leak. It is a should-fix on the
test, independent of this stage (see below).

**Non-periodic behaviour is bit-identical (verified, not just claimed):** the current tree's
LAPACK-off `test_vertex_fitting_validity` deviation is `3.051e-03`, byte-identical to the
HEAD baseline; non-periodic `RemeshParallel` volume drift (~1e-11) matches Stage 10. Every
Stage-13 edit either replaces a raw `p2−p1` with `MFTLGetMinimumImageVector` (reduces to the
raw difference when `is_periodic` is false), replaces a raw copy with `MFTLUnwrapToAnchor`
(exact copy when non-periodic), branches on `MFTLDomainAnyPeriodic` keeping the old code on
the non-periodic arm, or calls `MFTLApplyPeriodicity` (a no-op with no periodic axes).
Confirmed across `vertex_fit.c`, `curvature.c`, `volume_conserving.c` (TSUR clamp),
`remeshing_core.c` (23 min-image swaps + 3 gated branches) and `mftl_bvh.c`.

**Periodicity correctness:**
- Min-image/unwrap helpers are per-axis correct for X/Y/Z/any combination; `MFTLUnwrapToAnchor`
  uses `round`, consistent with the single-correction `MFTLGetMinimumImageVector` under the
  documented constraint.
- Integrals/point-query use a **collective anchor** (lowest-index owned-triangle v0 on the
  min rank that owns anything, broadcast) so every rank unwraps into the SAME image →
  rank-independent; seam-invariance battery confirms seam == mid to 1e-9 (V/A/inertia/centroid)
  and 1e-8 (curvature) for X/Y/Z/XY/XYZ.
- `MFTLPointInsideObject` temporarily unwraps ALL vertices to the anchor, builds the BVH on the
  unwrapped mesh, re-images the probe, and **restores on the single return path** (no early
  return leaves the mesh stretched; `saved` freed) — verified leak-free.
- BVH: node bounds in the stretched embedding + periodic 1-D interval overlap (shifts
  s∈{−1,0,1}, conservative superset) + common per-triangle frame in
  `MFTL_TriTriIntersect`/`Proximity`; combined with the A-ownership partition this prevents
  missing AND duplicate/phantom cross-seam pairs. Non-periodic interval test is bit-identical.
- Minimum-image constraint check is collective (two parts: longest owned min-image edge ≥
  smallest half-period; unwrapped body extent ≥ half-period), returns SUCCESS immediately on
  non-periodic domains, and is wired non-intrusively (sphere → `MFTL_ERROR`; cube → warning).

**Redistribution heap-corruption fix — reviewed in depth, sound:**
- Sender packs the **truthful `is_ghost`** (was hard-coded FALSE); owner labels are recomputed
  from the sender's fresh position (`MFTLGetOwnerProc`).
- Receiver dedup is **hash-based via `vertex_lookup`**, so `write_count == n_unique` always and
  the `n_unique`-sized array cannot overflow; PASS-1(locals)-then-PASS-2(ghosts) with the
  lookup check guarantees no gid is written both local and ghost.
- **Only authoritative (sender-non-ghost) copies with `owner_proc==my_rank` create locals**, so
  stale non-neighbour ghost copies (systematic for periodic seam-straddlers) can no longer
  fabricate phantom locals or double ownership → **exactly-once ownership**. Stale ghosts stay
  safe (become ghosts, refreshed next redistribute).
- **Orphan/authority-completeness:** non-ghost vertices with no non-ghost incident triangle are
  shipped explicitly to their recomputed owner (fixes the ownerless-vertex volume-conservation
  break seen in periodic LeVeque). Both `MFTLRedistributeObject` and
  `MFTLRedistributeListOfObjects` carry the fix; triangle reconstruction uses the same scheme.
- **Stress:** wide seam-straddler (R=0.7, x-periodic) `PeriodicEndToEnd` on np8 under ASan ×15
  — all clean; non-periodic `RemeshParallel` np8 ×8 and `TriangleData` np8 ×4 under ASan —
  all clean (no regression from the `is_ghost` change).

**Test quality:** the tests are genuinely discriminating, not just exercising code — the 1e-9
seam==mid tolerances on a truly straddling sphere would explode on any raw-coordinate leak;
the wide-seam remesh, advection, LeVeque, seam intersection and constraint-firing cases all
assert meaningful invariants. The implementer's manual sensitivity probes (removing the TSUR
corner unwrap breaks the curvature gate; disabling the common frame yields 200 vs 60 phantom
hits) corroborate.

**Should-fix (not blocking; pre-existing, independent of Stage 13):**
- ~~`test_vertex_fitting_validity` fails under `MFTL_USE_LAPACK=ON` (`5.032e-02` vs `0.05`)~~
  **RESOLVED (2026-09-22):** genuine Stage-08 port bug — the fit stencil excluded the edge
  endpoints and was asymmetric around the evaluation point; restored MFTL1's symmetric
  diamond stencil, gathered from the intact edge pre-split (`vertex_fit.c`). Deviation now
  `5.5e-05`; tolerance unchanged; the test destroys the ensemble before asserting.
  Independently reviewed & accepted (full write-up in `plans/08`, "Follow-up 2026-09-22"):
  Release + ASan/UBSan/LSan, LAPACK-on **and** LAPACK-off, full suite **53/53** on 1/2/4/8;
  periodic paraboloid path clean on seam-straddlers. No public-API or non-periodic change.

**Optional improvements:**
- Consolidate the duplicated collective-anchor helpers (`MFTLObjectUnwrapAnchor` in
  `geometry_object.c` vs `mftlPQUnwrapAnchor` in `point_query.c`) and the two `unwrapToAnchor`
  spellings (header `MFTLUnwrapToAnchor` + file-local `mftlUnwrapToAnchor`).
- The ~500-line receive-side reconstruction is duplicated between the single-object and
  list-object redistribute paths — a future fix must touch both; factor into one helper.
- `MFTL_ASSERT(write_count == n_unique)` compiles out in Release; a hard runtime guard would be
  cheap insurance on the heap-safety invariant.
- Perf (acceptable under the overview's correctness-first policy): point-query unwraps all
  vertices + mallocs per call; linear `target_obj` search in redistribution (already TODO'd).

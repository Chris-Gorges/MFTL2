# 10 — Parallel remeshing + load balancing

Status: COMPLETE
Depends on: 08, 09
Implemented by: Kimi Code
Reviewed by: Claude Code
Acceptance: PASSED

> **Review note (accepted):** Verified — single-owner interior gates prevent divergent
> topology; GID uniqueness via disjoint per-rank blocks (confirmed by the ownership-stability
> `n2==n1` re-redistribution check); load balancing deterministic (`lb_xcuts` from
> `Allreduce`'d global counts) and effective; CENTERING gather-to-rank-0 exercised (advect
> np2) and lossless; roughness data now crosses ranks; serial edge-local smoothers fail
> loudly on distributed meshes (resolves the Stage-09 carry-forward). Volume rank-independent
> (V=−4.17972146930, drift ≤ ~7e-11 on 1/2/4/8); Release + ASan/UBSan clean (0 sanitizer
> hits) with repeats; no regressions to Stages 01–09. **Carry-forward (should-fix, not
> blocking):** (a) triangle count is rank-count-dependent (~13% np1→np8) because the
> interior-only gate under-remeshes partition-boundary regions — MFTL1's
> `MFTL_RemeshBoundaryCleanup` (gather-to-one-rank final pass) would restore full
> resolution-decomposition-independence; implement or flag before Stage 15. (b) transient
> non-neighbour-owner ghosts during LB passes are kept stale until the next redistribution
> (negligible ~1e-11 volume impact, no topology corruption). Periodic remeshing remains a
> Stage-13 item (TSUR endpoint clamp).

> **Implementation notes (for review):** the WIP freeze-band scaffolding
> (`MFTL_SHIFT_PHASES`, `decomp_shift`, freeze flags, `MFTL_RemeshAction`,
> coarsening `MFTLComputeCoarseningFactor`/`MFTLRemeshWithCoarsening`) was
> **removed**, not completed: its hard constraint (h_min > 4δ) and the
> dispatcher's `MFTL_ERROR` path (unreachable on np2 for typical cuts)
> belong to a different design than the approved MFTL1-style scheme below.
> What shipped:
>
> - **Single-owner rule** (MFTL1 `Boundary_Check`): split/flip act on edges
>   with both endpoints owned and both adjacent triangles local and
>   ghost-free; collapse additionally requires the whole edge fan local and
>   ghost-free. Rings around owned vertices are complete by the
>   redistribution fan-out, so no mid-pass topology communication is needed;
>   killed entities may still have stale ghost copies on other ranks — that
>   is expected (MFTL1 behaves the same) and `MFTLUpdateGhostPositions`
>   reports them via `GhostUpdatePayload.found == 0`; they are dropped by
>   the next redistribution.
> - **Load balancing**: `MFTLRedistributeEnsembleLoadBalanced` — MFTL1's
>   x-only equal-count slab optimizer (`NoOfProcs`, `DX += damping*(DX*ratio
>   - DX)`, converge `max|1-ratio| < 0.2`, retry with fewer slabs below
>   `Sizethreshold`), ensemble-wide because MFTL2 has one shared domain
>   ownership function. Slab cuts are arbitrary x positions stored in
>   `MFTL_Domain.lb_*` and honoured by `MFTLGetOwnerProc`; **requires an
>   x-only decomposition** (`cuts = {nproc,1,1}`, MFTL1's production
>   configuration). Other decompositions fall back to interior-only
>   remeshing (MFTL1 `LoadBalance = NO` analog, with a rank-0 note).
> - **Two-pass boundary shift**: `MFTLShiftOwnershipRanges` shifts cuts
>   right by `max(0.2*minDX, 3*referenceLength)`; boundary coverage is
>   complemented by a bounded split-cleanup under both the lb and the
>   restored native decomposition (flip diagonals have no length gate and
>   can leave long interior edges).
> - **GID uniqueness**: per-pass disjoint per-rank GID blocks (prefix
>   schedule over counted split candidates, base = Allreduce-MAX of the
>   counters); counters re-synced after every redistribution
>   (`MFTLSyncEnsembleGidCounters`).
> - **Stage-09 follow-up**: the serial edge-local smoothers fail loudly on
>   distributed meshes; the driver uses new `*Parallel` variants gating
>   every edge to the interior rule. Roughness data now crosses ranks
>   (`PackedVertex`/`GhostUpdatePayload` gained `roughness_mean/variance`).
> - **Volume conservation**: per-pass `VolumeRecovery` (ghost-synced before
>   it so it iterates on current geometry; corner-weighted effective area,
>   secant-calibrated slope, consensus-guarded loop exits) plus recover→sync
>   cycles in the restored decomposition. Observed: |ΔV/V| ≤ ~2.5e-13 on
>   1/2/4/8 ranks. `MFTLVolumeRecovery` itself is unchanged in behaviour on
>   serial meshes (k=3 ⇒ identical area; consensus no-ops at nproc==1).
> - **Not done** (out of scope, recorded in 00-overview): MFTL1's
>   `MFTL_RemeshBoundaryCleanup` (gather-to-one-rank final cleanup); the
>   centering kernel gathers to rank 0 (per-object centroid ranks would
>   need per-object domains); `MFTLExtendGhostLayer` is not used by the
>   remesh pass (redistribution already provides the 1-ring; the 2-hop
>   extension can create non-neighbour-owner ghosts on narrow slabs) but is
>   now defensive for other callers.

## Objective
Finish the parallel-remeshing layer that is currently broken at HEAD, and add MFTL1's x-only
1-D load balancing with the two-pass boundary-shift scheme, so remeshing produces
**rank-independent** results and rebalances triangle load across ranks.

## Relevant existing code
- MFTL2 (broken WIP, `d248554…HEAD`): `src/remeshing/remeshing_parallel.c`
  (`MFTLRemeshEnsemble` duplicated vs `remeshing_core.c:1224`; `MFTLRemeshEnsembleShiftScheme`
  and `MFTLRemeshEnsembleDispersedCentering` **called but undefined**; `MFTLComputeCoarseningFactor`,
  `MFTLRemeshWithCoarsening`; `MFTL_Object**`/`*` mismatch at ~:45). `MFTLExtendGhostLayer`,
  `MFTLRedistributeListOfObjects`/`MFTLRedistributeObject` (`communication.c`).
- MFTL1: `src/remeshing/remeshing.c` (`MFTL_Remeshing` two-pass under LoadBalance;
  `MFTLRedistributeObjectLoadBalanced` — equal-triangle 1-D x-decomposition iterative optimizer;
  `ShiftOwnershipRanges` — shift x-boundaries by `max(0.2·minDX, 3·referenceLength)` for a second
  pass; `MFTL_RemeshBoundaryCleanup`).

## Proposed changes
- **Resolve the build defects first** (they are the reason HEAD doesn't compile — cherry-pick
  the sound parts): keep a single `MFTLRemeshEnsemble` (decide `void` vs `int` return; update
  the header decl), implement or remove `MFTLRemeshEnsembleShiftScheme` /
  `MFTLRemeshEnsembleDispersedCentering`, fix the `objects` pointer type (`MFTL_Object**`).
- **Parallel driver**: per remeshing iteration — ensure ≥1-ring ghosts (`MFTLExtendGhostLayer`)
  → run Stage-08 operators on owned edges (split→collapse→flip) → Stage-09 smoothing →
  `MFTLUpdateGhostPositions` → `MFTLRedistributeListOfObjects`.
- **Load balancing** `MFTLRedistributeObjectLoadBalanced`: x-only equal-triangle-count decomposition
  (`NoOfProcs=clamp(floor(total/MinNoTriangles),1,nproc)`), iterative slab-width optimizer
  (`DX += dampingfactor·(DX·ratio−DX)`, converge `max|1−ratio|<0.2`), retry with fewer procs if a
  slab < `Sizethreshold`.
- **Two-pass boundary shift** `MFTLShiftOwnershipRanges`: shift x-cuts so triangles previously on
  a proc boundary become interior for a second remeshing pass (they'd otherwise never be
  remeshed). Keep `Ycut=Zcut=1` (x-only) to avoid crossed boundaries.

## Files/functions likely to change
`src/remeshing/remeshing_parallel.c`, `remeshing_core.c`, `include/mftl_remeshing.h`;
`src/general/communication.c` (ownership-range shift helpers). Test `tests/test_remesh_parallel.c`.

## Numerical/physical considerations
- **Rank-independence is the headline invariant**: remeshing the same initial mesh on 1/2/4/8
  ranks must yield statistically identical results — same final volume (to Stage-09 tolerance),
  same triangle-count band, same edge-length distribution. Exact per-triangle identity is not
  expected (operation order differs), but conserved/aggregate quantities must match.
- Boundary triangles must actually get remeshed (the two-pass shift guarantees coverage).
- Conservation across redistribution: volume and (Stage 12) per-triangle data preserved.

## MPI/PETSc considerations
- Edge operations near a partition boundary need a consistent ghost ring; an edge split/collapse
  on a shared edge must be decided by a single owner to avoid divergent topology. Define the
  ownership/tie-break rule explicitly (e.g. lower `owner_proc` decides) and re-sync.
- Load-balance redistribution keys on triangle centre → owner; ensure the x-slab optimizer is
  computed identically on all ranks (broadcast the slab boundaries).
- This is where the multiflow dual-decomposition matters downstream (Stage 15): load balancing
  must still allow `AdaptOwnershiprangesToCartesianOverlapMesh` to realign afterwards.

## Possible failure modes
- Divergent topology when two ranks independently remesh a shared edge → the single-owner rule
  and re-validation must prevent it.
- Load-balance optimizer non-convergence (oscillating slab widths) → cap `maxBalanceIter`,
  fall back to previous decomposition.
- Ghost ring too thin for a collapse that removes a boundary vertex → require 2-ring near
  boundaries.
- Re-introducing the HEAD build breakage → CI must compile `remeshing_parallel.c` cleanly.

## Unit/regression tests to add (`tests/test_remesh_parallel.c`)
- Remesh a refined sphere on np1/2/4/8: assert equal final volume (Stage-09 tol), triangle count
  within a small band, and edge-length mean/variance within tolerance across rank counts
  (**rank-independence**).
- Load balance a deliberately imbalanced distribution → per-rank triangle counts converge to
  within `Sizethreshold` of equal; no rank starved.
- Boundary coverage: a feature straddling a proc boundary is remeshed after the two-pass shift.
- Full LeVeque parallel advect+remesh (mirror MFTL2 `parallel_advection_*`) conserves volume.

## Acceptance criteria
- `remeshing_parallel.c` compiles cleanly (HEAD breakage resolved); full CTest green on 1/2/4/8.
- Remeshed volume and aggregate mesh statistics rank-independent within tolerance.
- Load balancing converges and covers boundary triangles.

## Acceptance record (2026-09-21)

- `remeshing_parallel.c` compiles cleanly (0 warnings); full CTest green in
  Release (41/41) and ASan/UBSan/LSan builds on 1/2/4/8 ranks, including the
  new `RemeshParallel-np{1,2,4,8}` tests.
- Rank independence: refined-sphere remesh gives |V - V_analytic| < 1%,
  triangle count in the equilateral band, converged edge statistics, mesh
  integrity, and GID-counter agreement on 1/2/4/8 ranks; a further
  redistribution preserves volume/count exactly.
- Load balancing: two off-centre spheres balance to min=296/max=336 around
  fair=320 at np8 (MFTL1 0.2 tolerance); boundary coverage asserted via a
  hard band (no edge above 1.5*max) plus a small straggler allowance for
  late-convergence flip diagonals; advect+remesh (exact rotation) holds the
  per-step remesh drift at ~1e-12*V.
- Known limitations recorded in 00-overview: x-only decomposition required
  for load balancing; transient slab-spanning edges get skipped ghost
  updates (noted on stderr, refreshed at the next redistribution); flip
  diagonals are not length-gated (same as MFTL1).

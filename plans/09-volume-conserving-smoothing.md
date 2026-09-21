# 09 — Volume-conserving smoothing (TSUR3D, Kuprat, VolumeRecovery, roughness smoothing)

Status: COMPLETE
Depends on: 02, 08
Implemented by: Kimi Code
Reviewed by: Claude Code
Acceptance: PASSED

> **Review note (accepted):** TSUR3D / Kuprat Alg-4 / VolumeRecovery / roughness smoothing
> all implemented and MFTL1-consistent; conservation verified (TSUR local drift ~4e-15,
> Kuprat first-order + `VolumeRecovery` → ~1e-13, full remesh cycle ~6e-12 ≤ 1e-8). The
> remesh driver now always ends with `VolumeRecovery` against the pre-remesh volume +
> ghost refresh. Feature protection reuses the Stage-08 geometric dihedral check.
> Release + ASan/UBSan clean, suite green on 1/2/4/8. **Scope/deferrals:** the edge-local
> smoothers run on the serial (rank-0-local / ghost-complete) workflow only — fully
> parallel edge-local smoothing is Stage 10 (they currently lack an owner-guard/precondition
> assertion, so must not be called on a distributed mesh); TSUR's `[global_min,global_max]`
> clamp is periodic-unaware → Stage 13; `roughness_mean` ghost transport → Stage 10. The
> mean-curvature-flow `LaplacianSmoothing` (needs the signed Laplace–Beltrami vector) was
> out of scope and not implemented.

## Objective
Add the volume-preserving smoothing machinery that makes remeshing conservative: Jiao's TSUR3D
null-space smoothing, Kuprat's Algorithm-4 volume-conserving Laplacian, global `VolumeRecovery`,
and roughness-triggered smoothing. Essential for front tracking (mass/volume conservation).

## Relevant existing code
- MFTL1: `src/remeshing/tsur3d.c` (`MFTL_TSUR3D`, `MFTL_EdgeTSUR` — Jiao JCP 2004: area-weighted
  endpoint normals, common volume-conserving height `h=(V0+V1)/(V0/h0+V1/h1)`);
  `volumeconservation.c` (`VolumeRecovery` — global normal shift `h=(V−V_old)/Area`;
  `VolumeConservingSmoothing`/`MFTL_EdgeVolumeConservingSmoothing` — Kuprat 2001 Algorithm 4,
  exact local one-ring volume correction); `smoothing.c` (`Object_Roughness_Smoothing`,
  geometric-Laplacian & dihedral roughness, valence ≤ 10); `MFTL_LaplacianSmoothing`
  (mean-curvature flow + `VolumeRecovery`).
- MFTL2: Stage-08 tangential smoothing, Stage-06 curvature/roughness, Stage-02
  `MFTLObjectVolume`/area; one-ring traversal.

## Proposed changes
New unit `src/remeshing/volume_conserving.c` (+ decls in `include/mftl_remeshing.h`):
- `MFTLVolumeRecovery(MFTL_Object*, double target_volume)` — compute current volume+area, shift
  every vertex along its one-ring normal by `h=(V−target)/Area`; report relative volume error.
- `MFTLEdgeVolumeConservingSmoothing(...)` / object driver — Kuprat Algorithm 4: per-endpoint
  Laplacian target, area-vector `A_i`, normal-direction correction `h` cancelling first-order
  volume change; move `x_i += dx_i + h·n`.
- `MFTLEdgeTSUR(...)` / `MFTLTSUR3D(...)` — Jiao null-space smoothing per the documented 8-step
  algorithm (project endpoints to a common tangent plane, solve volume-conserving height).
- `MFTLRoughnessSmoothing(MFTL_Object*, params)` — per-edge trigger on
  `roughness_mean`/geometric-Laplacian thresholds, dispatch to Kuprat or TSUR.

## Files/functions likely to change
New `src/remeshing/volume_conserving.c`; `include/mftl_remeshing.h` (params:
`LaplacianSmoothingUnderRelaxation`, `maxSmoothingLaplacianIter`, `maxRoughness`,
`maxGeometricLaplacian`, `MFTL_RoughnessSmoothing`); driver hook in `remeshing_core.c`. Test
`tests/test_volume_smoothing.c`.

## Numerical/physical considerations
- **Invariant**: TSUR and Kuprat conserve *local one-ring* volume to first order; `VolumeRecovery`
  restores *global* volume exactly (uniform normal shift). Combined with Stage-08 operators, a
  full remesh cycle should conserve total volume to a tight tolerance.
- NaN guards (TSUR returns error on degenerate normals); negative-radicand guards.
- Under-relaxation ω (`LaplacianSmoothingUnderRelaxation`) controls stability vs speed.
- Valence ≤ 10 arrays; mean-curvature flow step `h=5e-5·referenceLength` (MFTL1) — small and
  explicit; watch stability over many iterations.

## MPI/PETSc considerations
MFTL1 collapses ownership to the vertex-richest proc for serial-equivalent smoothing, then
restores. In MFTL2, either (a) mirror that (gather to one rank, smooth, redistribute) for exact
serial equivalence, or (b) do a parallel edge-local version with sufficient ghost rings and
verify rank-independence. Prefer (a) initially for correctness, (b) as an optimization noted in
Stage 10. Volume must be reduced globally (Stage 02). No PETSc.

## Possible failure modes
- Volume "conserved locally" but global drift accumulates → always finish a remesh cycle with
  `VolumeRecovery` against the pre-remesh volume.
- Kuprat `AA` denominator near zero (flat/degenerate ring) → guard, skip.
- Roughness smoothing oscillating → cap iterations, monitor roughness decreasing.
- Parallel edge-local smoothing double-moving shared vertices → owner-guarded updates.

## Unit/regression tests to add (`tests/test_volume_smoothing.c`)
- Perturb a sphere's vertices (noise), record volume, run Kuprat/TSUR smoothing →
  **volume conserved** to ≤1e-8·V and roughness reduced monotonically.
- `VolumeRecovery`: after tangential-only smoothing that drifts volume, recover to target within
  1e-10·V.
- One-ring local volume conservation: single-edge TSUR/Kuprat step changes local one-ring volume
  by < first-order tolerance.
- Rank-independence: smoothed volume identical on 1/2/4/8 (via the gather-smooth-scatter path).

## Acceptance criteria
- Full remesh + smoothing cycle conserves total volume to ≤1e-8·V; `VolumeRecovery` restores to
  ≤1e-10·V.
- Roughness monotonically reduced on a perturbed sphere; no NaNs.
- Rank-independent smoothed volume.

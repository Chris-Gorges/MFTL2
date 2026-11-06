# TSUR3D and a standalone volume correction

**Status: implemented** (2026-11-05) as `src/correction/`. Theory:
`src/correction/README.md`; measurements: `docs/analysis.md` §15 and
`studies/leveque_remeshing/PAPER_NOTES.md` section 10.

Differences from this design:

- Settings are two sections, `[volume]` and `[undulation]`, not one `[correction]`.
- The example's `-field rotation` case was added: TSUR3D is only meaningful where the
  mesh resolves the interface.

Not done:

- a local volume correction; why it is wanted is in `docs/analysis.md` §15.1 item 5;
- moving the cross-rank edges in TSUR3D.

The original design follows.

## Findings

- MFTL2 already has a volume correction, but only inside `MFTLEnsembleReconstructBodies`
  (`src/topology_change/topo_driver.c`, `displacement_field` + the Newton block after the remesh).
  It is NOT callable on its own.
- How it works: a uniform offset `d` along the area-weighted unit vertex normals. A rim vertex on a wall
  loses its wall-normal component. The step is Newton's, `d = (V_ref - V) / (sum_t S_t . mean(u))`, run
  for 3 fixed steps (1e-9 relative from a 15 % deficit).
- mftl `VolumeRecovery` uses the same displacement field and one step with `d = (V - V_old) / A`.
  - The area replaces the exact first variation, so the step is off by O((kappa h)^2) and O(kappa d).
  - There is no iteration and no wall handling, and it redistributes afterwards.
  - With an absolute target called every step, its residual does not accumulate. For one-shot large
    deficits it converges only linearly (0.03 per step).
- TSUR3D (de Sousa et al., JCP 198, 2004; mftl `tsur3d.c`), per edge a-b:
  - m is the midpoint, so h_a = -h_b.
  - With P_i = n . S_i (S_i the fan's vector area, which depends only on its link):
    `V_i / h_i = P_i / 3`.
  - mftl's `h = (V1+V2)/(V1/h1+V2/h2)` is therefore exactly `h = (h_a P_a + h_b P_b)/(P_a + P_b)`,
    the projected-area-weighted mean height. That form has no 0/0, and the move is bounded by
    |h_a| = |h_b|.
  - Volume is exact: both moves are along the same n, so the cross term det(n, n, .) is 0, and P_a does
    not depend on b's position along n.

## Design

- New module `src/correction/` and `include/mftl_correction.h` (layer 3). Its only MPI lives in
  `correction_comm.c`, guarded by a confinement CTest. It has no dependency on the remesher, so Phong
  projection cannot revert it.
- `MFTLEnsembleRemoveUndulations(ens, params, stats)` (collective) and
  `MFTLRemoveUndulationsObject` (local; moves only rank-private vertices).
  - Params: `sweeps` (1), `min_angle` (0 = every edge; edge kink `|B.n| <= tan * |B_tan|` is
    skipped), `relaxation` (1).
  - Improvements over mftl:
    - the P-weighted h, refusing P_i <= 0;
    - a fold guard on every fan triangle;
    - minimum-image geometry and `MFTLApplyPeriodicity` instead of the domain clamp;
    - BOUNDARY, FEATURE and CORNER vertices stay fixed;
    - no gather onto one rank.
  - Distribution, round 0: every rank processes edges whose ends are owned and rank-private, meaning
    every fan triangle is owned and unreplicated. No sync is needed.
  - Distribution, then colored rounds: per-axis rank colours are parity, plus a third colour for the
    last cell of an odd periodic axis. Same-colour ranks are never 26-neighbours. In its round, a rank
    processes the remaining edges with both ends owned. Each round is followed by
    `MFTLUpdateGhostPositions`, and one MAX Allreduce skips empty colours.
  - The result is exact global volume at any np. Edges whose ends have different owners are never
    moved, and are counted.
  - Fields:
    - CONSERVATIVE_DENSITY is rescaled per triangle by `A_start/A_end`. Normal motion keeps the amount
      per triangle, the rescale is path-independent, and it is bitwise identical on every holder,
      ghosts included.
    - Vertex fields are unchanged, since the material point is the same.
- `MFTL_Object::reference_volume` (NaN = unset):
  - It is carried in the redistribution body header (7 -> 9 ints, memcpy of the double bits), and
    asserted equal on holders.
  - Topology change gives each output body `V_body * R_cluster / V_cluster` (a MAX-allreduce of the
    input bodies' references). Without this, a merged body keeping id A would be shrunk back to A's
    volume.
- `MFTLEnsembleSetReferenceVolumes(ens)` measures them with `MFTLEnsembleBodyIntegrals`.
- `MFTLEnsembleCorrectVolumes(ens, params{iterations=3, rim_project}, stats)`:
  - It uses the integrals' axis/reference, then Newton with one `MFTLEnsembleReduceBodies` (2 sums per
    body) per step, with a ghost sync after each step.
  - Fields get the same rescale.
  - `displacement_field` moves to `src/geometry/body_integrals.c` as `MFTLBodyVolumeDisplacement`,
    shared with topology change.
- Settings section `correction` with the parameters above.
- LeVeque example flags: `-volcorrect off|remesh|both`, `-undulation N`, and `-noise eps`. The last is
  a per-vertex normal velocity perturbation hashed from (gid, step), a model of subgrid fluctuations.

## Tests (`tests/test_correction.c`, np 1/2/4/8/27)

- One TSUR3D edge conserves volume to round-off. A sweep conserves volume to 1e-13 and reduces the rms
  noise on a noisy sphere.
- A noisy periodic sheet goes to its mean height.
- A body on the seam.
- Conservative densities: amount per triangle and in total; ghosts agree with owners.
- Non-vacuity: band edges are moved in coloured rounds at np >= 2.
- Volume: a shrunken sphere is restored to 1e-12 in 3 steps and stays spherical. An open body on a
  wall. The reference survives a redistribution onto a rank that never held the body. Topology change
  conserves the summed reference.

## Studies

- LeVeque:
  - correction after remesh only vs after advection and remesh;
  - Newton vs the mftl area step (as an archived experiment patch);
  - TSUR3D with and without `-noise`.
- Metrics: d_rms, E_H, N_H against the exact reference.

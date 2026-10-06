# 02 — Rims that do not shrink

Status: IMPLEMENTED in `317e767` (branch `christian`, 2026-10-03), with these deviations:
- **The curve is the average of the two circumcircle arcs, not centripetal Catmull–Rom.** Measured on a unit disk refined and coarsened 8 times: chord r = 0.977..0.988, Catmull–Rom 0.994..0.998 (O(h²) on uneven spacing), arcs exactly 1 to 6e-16.
- **Rim splits gained a fold guard** that falls back to the chord midpoint.
- **`MFTLSplitEdge` and `MFTLCollapseEdge` take the parameter block.**
- **The travelling-wave topology CTest moved to 96×24.** At 64×16 it passed on a chaotic point. The remesher's folds at creases are recorded as an open issue (analysis §9.8).
Replaces: the boundary tags and boundary curves of MFTL2 `main` (`plans/16` §2.6, §4.3 there)
Independent of `01`.

## 1. Purpose

On an open surface the remesher must keep the rim (the physical boundary) where it is:
- a split of a rim edge must place its new vertex on the rim curve;
- a collapse along the rim must keep the survivor on it;
- smoothing must move rim vertices along it.

The same must hold after a topology change reconstructs the surface.

## 2. Findings that decide it

- **Splits use the chord midpoint** (`midpoint_periodic`, `remesh_internal.h:327-337`, called at `remesh_ops.c:230`). A collapse between two rim vertices uses the same midpoint (`collapse_target_position`, `remesh_ops.c:342-360`).
  - **Planar walls:** the midpoint of two points on a planar wall is on that wall to the bit. So rims on planar walls never leave their wall; the Rayleigh–Taylor example relies on this (`rayleigh_taylor_open_surface.c:101-121`).
  - **Curved rims:** the midpoint shortens and flattens a curved rim *within* its plane, by O(h²κ) per operation, accumulating over passes. This affects a contact line curving along a side wall, the steep wave profiles at the channel walls, and the jet's nozzle circle.
- **The smoother moves a rim vertex along its ridge tangent** (`ridge_displacement`, `remesh_smooth.c:324`). On a curved rim that step also leaves the curve, by O(step²κ). `test_a_regular_ridge_is_a_fixed_point` covers only the symmetric case.
- **Applications repair this after every remesh.** The jet projects the lip back onto the nozzle circle (`project_lip`, `liquid_jet_open_surface.c:791-800`, called at `:1322`, `:1375`) "because a split of a lip edge puts its midpoint on the chord". Rayleigh–Taylor snaps the rim back onto its walls (`snap_to_domain`, `:648`, called at `:681`, `:717`, `:725`).
  - This works, but the remesher's approval tests (stretch, fold-over) judge positions that are then moved.
  - Every application has to re-implement it.

**What boundary tags would add.** None of this needs them. A rim on a wall is identified by its position, as the examples already do. Tags add an identity only for boundary parts that position cannot tell apart:
- two rims on one wall plane needing different physics;
- free rims away from the walls (topology change refuses these today);
- rims on moving supports.

## 3. Decision

1. **Curve-following placement on rims**, using centripetal Catmull–Rom. Used for rim splits, rim–rim collapses and rim smoothing.
2. **An optional, position-based constraint hook** in `MFTL_RemeshParameters`, applied wherever the library places or moves a rim vertex, including after a topology change.
3. **No boundary tags now** (§8).

### Why centripetal Catmull–Rom

- **It interpolates the rim's own vertices,** so nothing has to be stored (no tangents or normals, unlike Mmg's Bézier ridges).
- **It cannot overshoot.** Centripetal parameterisation (α = ½) is the only Catmull–Rom parameterisation without cusps or self-intersections within a segment ([Yuksel, Schaefer, Keyser 2011](https://www.cemyuksel.com/research/catmullrom_param/catmullrom_cad.pdf)). The uniform variant used on `main` (`boundary_remesh.c`, `mftlCR`) overshoots when neighbouring rim edges differ in length.
- **It keeps planar rims exactly in their plane** when evaluated as a weighted sum of differences from `a`. The coordinate normal to a planar wall has zero differences, so it reproduces today's guarantee to the bit.
- **Its error at the midpoint is O(h⁴)**, against O(h²) for the chord.

## 4. Design

### 4.1 The curve point (`src/remeshing/`, local)

For a rim half-edge a→b with rim neighbours p = pred(a) and s = succ(b):

1. Work in a's minimum-image frame: `P0 = a + d(a,p)`, `P1 = a`, `P2 = a + d(a,b)`, `P3 = P2 + d(b,s)`.
2. Knots: `t_{i+1} = t_i + |P_{i+1} − P_i|^{1/2}`.
3. Evaluate with Barry–Goldman's pyramidal form at `t = (t1 + t2)/2`, written as `a + Σ w_i (P_i − a)`.

Rules:
- **Breakpoints.** If a or b carries `MFTL_FLAG_CORNER`, or a neighbour is missing, the missing control point is the linear extrapolation (`P0 = 2P1 − P2`, `P3 = 2P2 − P1`). A straight rim then stays exactly straight, as on `main`.
- **Locality.** Rim operations already run only on unfrozen owned entities with complete fans, so p and s are in the one-rings of a and b. If either is unavailable, fall back to the chord midpoint and count it in the stats (which should read 0).
- **Wrapping.** The result is wrapped with `MFTLApplyPeriodicity`, exactly as today.

### 4.2 Where the curve point is used

- **Split** of a rim edge (`remesh_ops.c:230`): the curve point instead of the midpoint. Interior edges are unchanged.
- **Collapse target** when both a and b are rim vertices (`remesh_ops.c:359`). The rules that keep a pinned vertex in place stay as they are. The fold-over and stretch approval clauses then evaluate the real target.
- **Smoothing** of a rim (ridge) vertex v. Replace the tangent step with an arc-length-equalising move along the two segments (pred, v) and (v, succ):
  - evaluate the curve parameter that splits the polyline length between pred and succ equally;
  - evaluate the curve there;
  - keep the existing step clamp.

  On a regular polygon this is the identity, so `test_a_regular_ridge_is_a_fixed_point` still holds. Interior feature ridges (a dihedral feature, not a rim) can use the same code path. Decide by measurement whether to extend it to them.

### 4.3 The constraint hook

```c
/* Moves pos onto the constraint of a rim vertex. Must be a pure function of its
 * arguments and identical on every rank. Return MFTL_SUCCESS, or MFTL_ERROR to
 * refuse (the operation is then refused). */
typedef int (*MFTL_RimProjector)(const MFTL_Domain *dom, double pos[3], void *user);

/* in MFTL_RemeshParameters: */
MFTL_RimProjector rim_project;   /* MFTL_NULL: the curve alone */
void *rim_project_user;
```

- **Applied** after every rim placement (split, collapse target) and every rim smoothing move, *before* the approval clauses see the position.
- **Agreement.** A function pointer cannot be hashed, so `MFTL_REMESH_AGREED_FIELDS` (`remesh_comm.c:308`) gains a `rim_project != NULL` flag. Being the same function on every rank is a documented contract.
- **Topology change** (`topo_driver.c`) applies it to every reconstructed rim vertex before the volume correction, through `MFTL_TopologyOptions::remesh`, which already carries the remesh parameters. Marching cubes already places rim vertices exactly on wall planes; the hook restores a non-planar constraint such as the nozzle circle.
- **Settings.** No new key: the hook is code, not configuration.

### 4.4 Examples

- `liquid_jet_open_surface.c`: `project_lip` becomes the hook.
- `rayleigh_taylor_open_surface.c`: the snap becomes the hook. Its owner clamp stays if it is needed for `MFTLGetOwnerProc` (`:643`).
- `travelling_waves_open_surface.c`: needs nothing new; it gains wave-height accuracy at the walls.
- Every example keeps `--verify`. The "rim vertex on its constraint" check then becomes a library property instead of a repair.

## 5. Files to modify

| File | Change |
|---|---|
| `src/remeshing/remesh_internal.h` | curve-point helper next to `midpoint_periodic` |
| `src/remeshing/remesh_ops.c` | split (`:230`), collapse target (`:342-360`) |
| `src/remeshing/remesh_smooth.c` | rim smoothing along the curve (`ridge_displacement`, `:324`) |
| `include/mftl_remeshing.h`, `src/remeshing/remesh_comm.c` | hook fields and Doxygen; agreed-fields flag |
| `src/topology_change/topo_driver.c` | apply the hook to reconstructed rim vertices |
| `examples/liquid_jet_open_surface.c`, `examples/rayleigh_taylor_open_surface.c` | hook replaces the post-remesh projection |
| `tests/test_remeshing.c` (or a new `tests/test_rim_curves.c`) | tests of §6 |
| `src/remeshing/README.md` | the curve, why centripetal, the hook contract |
| `docs/analysis.md` | measured rim drift before and after; example metrics |

## 6. Tests

- **Circular rim.** A disk patch with a circular rim, remeshed 20 times at a fixed target length:
  - chord midpoint (baseline): measure and record the radius drift;
  - curve: drift far below the baseline;
  - curve plus hook: drift 0 to the bit.
- **Uneven spacing (1:10 edge ratio).** The new vertex stays inside the control polygon's convex hull. Uniform Catmull–Rom fails this; keep it as the regression.
- **Planar rim.** A rim on a wall plane keeps the wall coordinate bit-exact through 50 passes.
- **Straight rim and corners.** A straight rim stays straight; corners never move.
- **Distributed.** np 1, 2, 4, 8 on the four `test_*_open*` runs, with `--require-rim-cut` so that the rim actually crosses ranks.

## 7. Gate

- **Closed surfaces:** all digests bit-identical, since no rim means no code path changes.
- **Open surfaces:** the results change by design, so show they are not worse, by attribution:
  - wave height at the walls (travelling waves);
  - rim exactly on its constraint (all three);
  - volume balance;
  - Euler characteristic;
  - minimum and mean quality;
  - remesh time (interleaved A/B).
- Zero warnings in both manuals.

## 8. Considered and deferred

### Boundary tags (an MMG-style integer identity per boundary edge)

- **Storage, if ever needed:** half-edge `flags` bits 16–31, zero bytes.
  - Those bits are free: the flag registry is `mftl_topology.h:53-66`, and every library clear is a single-bit mask.
  - They already travel on the wire in `MFTL_PackedTriangleComplex.he_flags`.
  - Splits already copy the parent's flags to both halves (`remesh_ops.c:246-268`).
  - Only `glue_rim` (`remesh_ops.c:547`) would need to pass the removed rim edge's tag on.
- **Loop quantities** (length, line force, winding Σj): per-(body, tag) holder sums, never `main`'s all-gather of every boundary vertex.

**Revive when** a boundary part must be identified without its position:
- two rims on one wall with different conditions;
- free rims off the walls;
- rims on moving IBM supports;
- per-loop boundary conditions in a coupled solver.

### Per-tag curve modes (REPRESENTED, CONSTRAINED, PINNED per tag, as on `main`)

Revive with tags. Until then, one position-based hook covers every example.

## 9. Related, out of scope

Interior splits also use the chord midpoint. That is the same shrinkage mechanism on closed surfaces, and the cause of measured volume loss in the LeVeque runs (`docs/analysis.md` §4).

The analogous fix is a curved midpoint, for example a normal-based Hermite or PN-triangle midpoint, or `main`'s paraboloid fit. It deserves its own plan and measurement.

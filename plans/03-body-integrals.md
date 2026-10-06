# 03 — Per-body integrals

Status: IMPLEMENTED in `de6dd8d` (branch `christian`, 2026-10-03), with these deviations:
- **The rule for open bodies is extended.** It now prefers an axis with rims on exactly one wall, measured from that wall, then a rim-free axis from its lower wall.
- **Two cases added:** closed sheets across periodic axes (detected by the total area vector) and closed tubes that wrap an axis.
- **The status enum is `MFTL_INTEGRAL_*`**, and the struct gains `reference` and `lowest_holder`.
- **No error from rule violations:** the extents are exact to 3L/4.
- **Topology change keeps its own per-cluster axis choice** but shares the kernels bit-identically.
- **The serial LeVeque example keeps its own formula.**
Replaces: the phase-boundary sets of MFTL2 `main` (`plans/16` §3.6 there)
Depends on: `01-periodic-extent-limitation.md` (centre of a closed body on a periodic axis; the rule that makes it exact)

## 1. Purpose

A front-tracking or IBM solver needs, per body, its area, the volume it encloses and that volume's centre, for closed bodies and for open surfaces closed by domain walls. Later it will also need an inside/outside test.

`christian` computes these today only:
- inside the topology module (`MFTLTopoRimWalls`, `MFTLTopoChooseFlux`, `MFTLTopoObjectVolume`, `topo_cluster.c:416-495`);
- by hand in every example that needs them (`travelling_waves_open_surface.c:931-983`, `rayleigh_taylor_open_surface.c:842-910`, `liquid_jet_open_surface.c:984-1041`, and the LeVeque examples).

The plan turns this into one library function.

## 2. Findings that decide it

- **One rule covers every example.** Integrate the flux of F = (x_k − ref) e_k along a **non-periodic** axis k whose **far wall carries no rim**, with ref on the **near wall**. The divergence of F is 1. The near-wall patch has F = 0, and the side walls have F·n = 0. So the closed-surface integral reduces to the open surface alone.
  - travelling waves: z about the bottom wall;
  - Rayleigh–Taylor: z about the bottom wall;
  - jet: x about the nozzle wall.
- **Closed bodies need any axis about their own centre,** with the nearest-image lift. Under rule 01 that is exact.
- **What main's phase sets add on top of these sums is validation of a composition:**
  - consistency of the orientation signs;
  - nesting of bodies inside bodies;
  - inferring the exterior phase;
  - wall patches bounded by nested loops;
  - centres of regions made of several components.

  That is a guard on a general solver's configuration, not geometry the library must own. Compositions are one line of application arithmetic:
  - the matrix phase is `V_box − Σ V_b`;
  - a slab between two films wrapping x and y is `σ₁V₁ + σ₂V₂ (mod V_cell)`.

  Main also re-gathers every triangle of the set and re-classifies on every query, which is its measured 11.5 % hotspot with 27 periodic bubbles.

## 3. Decision

One collective call, `MFTLEnsembleBodyIntegrals`.

- Per body it returns area, volume and centroid, each with a status: DEFINED, UNDEFINED or UNSUPPORTED.
- It costs **one holder reduction**, plus a first one on periodic domains for the closed-body centre (from 01).
- Quadrature on triangles is exact.
- **No classification machinery.**
- **Inside/outside** is designed (§4.6) but deferred until a consumer exists.

## 4. Design

### 4.1 API (`include/mftl_ensemble.h`; status enum shared, e.g. in `mftl_geometry.h`)

```c
typedef enum { MFTL_DEFINED = 0, MFTL_UNDEFINED = 1, MFTL_UNSUPPORTED = 2 } MFTL_IntegralStatus;

typedef struct MFTL_BodyIntegrals {
  double area;                       /* always DEFINED                                 */
  double volume;                     /* signed: the side the normals do NOT point into */
  double centroid[3];                /* wrapped into the domain on periodic axes        */
  MFTL_IntegralStatus volume_status;
  MFTL_IntegralStatus centroid_status[3];
  int axis;                          /* flux axis used, -1 if none                      */
  int closed;                        /* 1 if the body has no rim                        */
} MFTL_BodyIntegrals;

/* out: [n_local_objects], by LOCAL SLOT (the rule for per-body arrays that never cross a rank). */
int MFTLEnsembleBodyIntegrals(MFTL_Ensemble *ens, MFTL_BodyIntegrals *out);   /* collective */
```

Following the rule for where a geometry function lives:
- the per-triangle kernel (local) goes in `src/geometry/`;
- the object loop and the reductions go in `src/ensemble/`.

### 4.2 Rules

**Closed body** (no `MFTL_FLAG_BOUNDARY` anywhere, decided from the reduced rim count):
- Axis k = 2, or any axis; the volume does not depend on it.
- ref is the body centre from 01's extent helper on periodic axes, and `global_min` on non-periodic axes.
- Each triangle's k-coordinate is the nearest image to ref.
- If 01 reports a violation, every status is UNSUPPORTED and the call returns `MFTL_ERROR`.
- Each centroid component j uses its own axis: `C_j = ref_j + (1/2V) ∫ y_j² n_j dA`.

**Open body:**
- Every rim edge must lie on one non-periodic wall: both endpoints within `1e-8·L_k` of the wall plane, the tolerance `MFTLTopoRimWalls` uses (`topo_cluster.c:428`). If not, `volume_status = UNDEFINED`.
- Axis: the first non-periodic k whose **far** (max) wall carries no rim, with ref the **near** (min) wall. This is the `MFTLTopoChooseFlux` rule, `topo_cluster.c:455-463`. It is extended symmetrically: if the rim sits on the max wall and the min wall is rim-free (a drop hanging from the ceiling), use ref on the max wall. F = (x_k − x_max) e_k still has divergence 1 and vanishes on that wall.
- None available: UNSUPPORTED. Example: a liquid bridge with rims on both walls of every non-periodic axis.
- Centroid along k: `ref + (1/2V) ∫ y_k² n_k dA`.
- Lateral centroid j ≠ k: `ref_j + (1/V) ∫ y_k y_j n_k dA`. This is the field y_k y_j e_k, whose divergence is y_j; the wall patch and side walls again contribute nothing. y_j uses the nearest image about the body centre on a periodic j.
- A centroid along an axis the body **wraps** (01's `wraps[j]`, e.g. x for travelling waves): UNDEFINED.

**Sign.** The volume is reported signed: positive is the region on the side the normals do not point into. That is outward winding for closed bodies and is fixed by the input for open ones. An application wanting the other phase uses `V_box − V`.

### 4.3 Exact quadrature (per owned triangle, linear integrands)

- `∫_T y n_k dA = (A n_k) · (y0 + y1 + y2)/3`
- `∫_T f g n_k dA = (A n_k)/12 · (Σ f_i g_i + (Σ f_i)(Σ g_i))`, which gives y² and y_k·y_j.

Slots per body: area, the V numerator, and the three moment sums. These are SUM slots in one `MFTLEnsembleReduceBodies` call. The rim-on-wall flags (6 walls) and "has a rim" are MAX slots in the first reduction, together with 01's 18 extent slots.

**Bit-identity with the topology module.** `MFTLTopoObjectVolume` (`topo_cluster.c:467-495`) uses the per-triangle centroid wrapped to the cluster centre, times the area vector component. The shared kernel keeps exactly that arithmetic for the volume, so the topology census stays bit-identical. Second moments are a separate accumulation.

### 4.4 Deduplication

- **Topology change:** `MFTLTopoRimWalls`, `MFTLTopoChooseFlux` and the volume part of `MFTLTopoObjectVolume` call the shared kernel and rule. The cluster-level combination (several bodies, one cluster) stays in `topo_driver.c`.
- **Examples** switch `S_VOLUME` (and centroid if reported) to the API, after proving the values identical. Bit-identical where the arithmetic order allows, otherwise ≤ 1e-14 relative, recorded.

### 4.5 Recipes in the README (no library machinery)

- **Liquid around N bubbles in a fully periodic box:** `V_cell − Σ V_b`. The centre is UNDEFINED (it wraps).
- **Slab between two films wrapping x and y:** each film is open-free and closed on the torus. The application integrates both along z about a common ref: `σ₁V₁ + σ₂V₂` reduced mod V_cell. This is the one case where the library gives the sums and the application gives the composition.
- **Gas above a wall-closed interface:** `V_box − V`.

### 4.6 Inside/outside (designed, deferred)

- **Ray parity** along the evaluation axis: any axis for a closed body, toward the far wall (rim-free) for a wall-closed open body.
- **Exactness** comes from an **adaptive exact orient2d plus Simulation of Simplicity** for edge and vertex hits. The existing `orient2d` (`bvh_predicates.c:915`) is the plain determinant and must stay separate. The new one is compiled with `-ffp-contract=off` like the rest of `bvh_predicates.c`.
- **Local fast path** when the ray segment to its decisive crossing lies inside the rank box shrunk by the longest edge.
- **Otherwise** a march along the axis over `comm_graph` that counts crossings from owned triangles only (exactly once).
- **Revive when** IBM coupling or indicator seeding needs it.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_geometry.h`, `src/geometry/body_integrals.c` (new) | status enum, per-triangle kernel, rule helpers (local) |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_integrals.c` (new) | `MFTLEnsembleBodyIntegrals` (loop and reductions) |
| `src/topology_change/topo_cluster.c`, `topo_driver.c`, `topo_internal.h` | use the shared kernel and rule |
| `examples/{travelling_waves,rayleigh_taylor,liquid_jet}_open_surface.c`, `examples/leveque_remeshing_*.c` | use the API after the identity check |
| `tests/test_body_integrals.c` (new), `tests/CMakeLists.txt` | tests of §6 |
| `src/geometry/README.md` | rules, quadrature, recipes |
| `CLAUDE.md` | add a row to the "Where a geometry function lives" table (bodies: local kernel / `MFTLEnsembleBodyIntegrals`) |
| `docs/analysis.md` | cost per call against the hand-written sums |

## 6. Tests

- **Closed bodies:** sphere, ellipsoid and a seam-straddling sphere (< L/2). Volume and centroid against analytic values, converging at second order in h, at np 1, 2, 4, 8, 27 with `agreed()`. Every rank returns bit-identical values (holder fold order).
- **Open bodies:**
  - a flat sheet at height h over the bottom wall, with rims on four walls: V = h·A_wall exactly;
  - a spherical cap on a wall: analytic cap volume and centre;
  - each example's current hand-written volume is reproduced (§4.4).
- **Statuses:**
  - a rim off the walls gives UNDEFINED;
  - rims on both walls of every non-periodic axis gives UNSUPPORTED;
  - the travelling-wave centroid along x gives UNDEFINED;
  - an over-long body (01) gives `MFTL_ERROR` on every rank.
- **Non-vacuity:** at least one body per case is held by two or more ranks (counted and asserted).

## 7. Gate

- All digests bit-identical (new call; topology volume bit-identical through the shared kernel).
- Topology census tests at np 1 … 27 unchanged.
- Zero warnings in both manuals.
- An interleaved A/B of `MFTLEnsembleBodyIntegrals` against one geometry pass (expect about one streaming pass plus one or two small reductions).

## 8. Considered and deferred

### Phase-boundary sets (`main`, `plans/16` §3.6 rev. 3.2)

- **Main's version:** members with σ, wall closures with nested-loop patches, the Φ(Ŝ) = 0 criterion, nesting through lines, exterior-phase inference, placement.
- **If revived,** keep main's reviewed mathematics and its 72-case model (`tools/phase_set_prototype/p36.py`) as the oracle, but restructure it:
  - classification is topological, so cache it and invalidate it only on topology change or a membership edit;
  - evaluate by per-member holder sums, never an all-gather of triangles;
  - cast lines only for classification and inside queries.
- **Revive when** a solver must validate arbitrary user-declared phase compositions, infer the matrix phase automatically, or report centres of multi-component regions.

### Generalized or fast winding numbers (Jacobson 2013, Barill 2018) for inside/outside

They are not defined for periodic or wall-closed sets without lattice sums, and they are approximate (fast) or O(N) per query (exact). Not adopted.

# 05 — Curvature and surface tension

Status: PLANNED (2026-10-05, branch `christian`)
Depends on: `04-attached-fields.md` (σ as a field, outputs as fields; both are optional, since raw arrays work without 04)
Used by: `06` (the force that gets spread), `07` (curvature-adaptive sizing), `09` (Helfrich bending, Marangoni with surfactant)

## 1. Purpose

Surface tension is the one force every front-tracking solver needs, and the library has no curvature and no
force. Each solver re-derives both on the raw arrays. multiflow does this in `computeforcesonfront`, in
FT_Sharp_CSF and in FT_Sharp_Integral. Each solver then also re-derives the partition argument for why its
neighbours are present.

The library knows that argument, and can deliver the force on OWNED entities with no communication at all.

## 2. Findings that decide it

- **The vertex-complete halo makes a per-vertex force free of communication.** Every triangle incident to an
  owned vertex is present locally. That is what `MFTLComputeVertexNormals` relies on (CLAUDE.md,
  "`MFTLComputeVertexNormals` needs a VERTEX-COMPLETE halo"). A force that is a sum over incident triangles is
  therefore exact on owned vertices, by the same scatter loop (`vertex_geometry.c`).
- **Every owned triangle has all three edge neighbours** (`inv_partition_is_sound`). So a per-triangle force
  that reads the neighbour across each edge is exact on owned triangles, again with no message.
- **The discrete area gradient is the force that conserves what physics conserves.**
  - Take `E = Σ_T σ_T A_T`. Then `F_v = −∂E/∂x_v = −Σ_{T∋v} σ_T · ½ n_T × (x_k − x_j)`, with (v, j, k)
    counter-clockwise.
  - E is invariant under translation and rotation, so `ΣF = 0` and `Σ x×F = 0` hold to round-off on a closed
    body.
  - For constant σ this is `−σ κ n A_mixed` with the cotangent mean-curvature normal. So force and curvature
    are consistent by construction, not by tuning.
- **Tryggvason's edge form is the physical line integral.** It is `F_T = σ ∮ m ds = ∫(∇_sσ − σκn) dA`, with
  m = t × n the outward conormal. Its discrete form `Σ_e σ_e (b − a) × n_e` telescopes to `ΣF = 0` if n_e and
  σ_e are symmetric across the edge. It conserves force, but NOT angular momentum exactly, because the force
  is applied at centroids. Both forms are wanted: the vertex form for vertex markers, the edge form for
  centroid markers.
- **Marangoni comes for free in both forms.**
  - In the edge form σ_e varies along the boundary.
  - In the vertex form σ_T varies per triangle, and the variation of Σσ_T A_T with σ held fixed gives the
    tangential term.
  - No separate gradient operator is needed for constant-per-element σ.
- **The cotangent curvature is not pointwise convergent on irregular meshes** (Xu 2004; Wardetzky et al. 2007).
  - It converges in a weak sense, which is what the area-gradient force needs. A pointwise κ for diagnostics,
    hybrid formulations or sizing needs a fit.
  - A quadric fit to the 2-ring would need a ghost layer that does not exist (CLAUDE.md "Absent: topological
    ghost-layer growth").
  - A fit to the **1-ring positions plus the 1-ring normals** has 3 equations per neighbour for 5 unknowns,
    and needs only exact ghost normals. `MFTLEnsembleUpdateVertexNormals` already provides those, in one
    exchange.
- **Sign and scale convention.**
  - `κ = ∇_s·n = κ1 + κ2`, i.e. **2/R for a sphere with outward normals**. This is main's convention (stored
    2/R), and the Young–Laplace jump is `Δp = σκ`.
  - Force: `F = −σκ n dA` points inward on a sphere.
  - Stated once in `mftl_geometry.h` and asserted by a test.

## 3. Decision

Two curvature methods and two force forms, all local kernels in `src/geometry/`, each with an ensemble
wrapper in `src/ensemble/` that adds exactly the collective its result needs (the placement rule):

| | one object, LOCAL | every object, COLLECTIVE |
|---|---|---|
| curvature | `MFTLComputeVertexCurvature` | `MFTLEnsembleComputeCurvature` (+ the ghost normal exchange it needs, + optional ghost exchange of the result) |
| surface tension | `MFTLComputeSurfaceTension` | `MFTLEnsembleSurfaceTension` (+ an optional per-body diagnostic reduction) |

Balanced-force coupling to the grid is the solver's business: it needs the solver's own pressure-gradient
operator. The library ships the building blocks and a README recipe (§4.6).

## 4. Design

### 4.1 API (`include/mftl_geometry.h`, `include/mftl_ensemble.h`)

```c
typedef enum { MFTL_CURVATURE_COTAN = 0, MFTL_CURVATURE_QUADRIC = 1 } MFTL_CurvatureMethod;

/* LOCAL. Exact on owned vertices [0, verts.n_local); ghost values unspecified.
   kappa: [verts.n] mean-curvature SUM (2/R on a sphere); gauss, curv_normal: nullable.
   QUADRIC needs vertex normals exact on ghosts (MFTLEnsembleUpdateVertexNormals). */
int MFTLComputeVertexCurvature(const MFTL_Object *obj, MFTL_CurvatureMethod method, double *kappa, double *gauss, double *curv_normal);

typedef enum { MFTL_ST_VERTEX_AREA_GRADIENT = 0, MFTL_ST_TRIANGLE_EDGE = 1 } MFTL_SurfaceTensionForm;
typedef enum { MFTL_EDGE_NORMAL_TRIANGLES = 0, MFTL_EDGE_NORMAL_VERTICES = 1 } MFTL_EdgeNormalMode;
typedef enum { MFTL_RIM_FORCE_INCLUDE = 0, MFTL_RIM_FORCE_WALL_TANGENTIAL = 1, MFTL_RIM_FORCE_EXCLUDE = 2 } MFTL_RimForceMode;

typedef struct MFTL_SurfaceTensionParams {
  MFTL_SurfaceTensionForm form;
  double sigma;              /* used when sigma_field < 0                                    */
  int sigma_field;           /* attached field id (vertex OR triangle, 1 component), or -1   */
  MFTL_EdgeNormalMode edge_normal;
  MFTL_RimForceMode rim;
} MFTL_SurfaceTensionParams;

/* LOCAL. force: [3*verts.n_local] (vertex form) or [3*tris.n_local] (edge form), INTEGRATED force per
   entity (not a density): what 06 spreads. */
int MFTLComputeSurfaceTension(const MFTL_Object *obj, const MFTL_SurfaceTensionParams *p, double *force);

typedef struct MFTL_SurfaceTensionStats {   /* per body, by LOCAL SLOT; reduced over holders */
  double net_force[3], net_torque[3];       /* about the body centroid; ~0 for a closed body  */
  double energy;                            /* Σ σ_T A_T                                      */
  int lowest_holder;
} MFTL_SurfaceTensionStats;

/* COLLECTIVE. Writes force into an attached field (or the caller's per-slot arrays);
   stats nullable: one holder reduction when asked, none otherwise. */
int MFTLEnsembleComputeCurvature(MFTL_Ensemble *ens, MFTL_CurvatureMethod method, int kappa_field, int gauss_field);
int MFTLEnsembleSurfaceTension(MFTL_Ensemble *ens, const MFTL_SurfaceTensionParams *p, int force_field, MFTL_SurfaceTensionStats *stats);
```

### 4.2 Vertex area-gradient form (default)

- **Loop.** Scatter over ALL triangles `[0, tris.n)`, owned and ghost. For corner v of T, with (v, j, k)
  counter-clockwise, add `−½ σ_T n_T^raw × (x_k − x_j)` to `F_v`.
  - `n_T^raw` is the unit normal: the edge vectors come from minimum-image corners (`MFTLGetTriangleCorners`).
  - Skip corners with `v ≥ n_local`.
  - This is the same per-corner scatter as the area-weighted vertex normal, reading the same corners. So the
    two can fuse into one pass when both are wanted. That fusion is a measured option, not the default.
- **σ_T:**
  - constant σ;
  - or a triangle field;
  - or the mean of a vertex field's three corner values.
- **Rim vertex** (`MFTL_FLAG_BOUNDARY`): the open fan contributes an in-plane pull along the rim conormal. That
  pull IS the line force of the contact line.
  - `INCLUDE` leaves it.
  - `WALL_TANGENTIAL` removes the component along the normal of the wall the vertex lies on (the
    `1e-8·L_k` rule of `MFTLBodyRimWalls` / `displacement_field`, `topo_driver.c:238`): the wall reacts
    to it.
  - `EXCLUDE` drops rim vertices entirely.

### 4.3 Tryggvason edge form

- **Edge contribution.** For owned triangle t and half-edge `3t+k` from a to b (counter-clockwise), take the
  edge vector `MFTLGetEdgeVector(obj, 3t+k)` (minimum image). Then:
  - `n_e = normalise(n_t + n_nb)`, with `nb = tris.neighbors[3t+k]` (`TRIANGLES` mode);
  - or `normalise(n_a + n_b)` from exact vertex normals (`VERTICES` mode);
  - fallback `n_t` when the sum vanishes, which is a fold.
- **σ_e** is the mean over the two triangles, or over the two endpoint vertices. It is symmetric, so the two
  sides cancel.
- `F_t += σ_e e × n_e`.
- **Rim edge** (no neighbour): `n_e = n_t`. The rim modes act as in §4.2.
- Requires `tris.normal` current over `tris.n`, i.e. the geometry pass covers ghosts. That is already the
  documented per-timestep order.

### 4.4 Curvature

- **`COTAN`** (Meyer et al. 2003): `K_v = ∇_v A / A_mixed`, `κ_v = K_v · n_v`, Gaussian = angle deficit /
  A_mixed. The same scatter as §4.2 plus one accumulation of A_mixed. Rim vertices are reported but flagged
  (one-sided).
- **`QUADRIC`**, the normal-augmented 1-ring fit:
  - Local frame (t1, t2, n_v). Unknowns `h(u, w) = ½(a u² + 2b uw + c w²) + d u + e w`.
  - Per neighbour j: one height equation and two slope equations from n_j.
  - Weighted least squares through a 5×5 normal-equation Cholesky in registers (no LAPACK).
  - `κ = (a + c)` corrected for the slope terms at the origin; Gaussian `ac − b²`.
  - Fallback to `COTAN` when the system is rank-deficient: fewer than 3 neighbours, or a rim fan with
    collinear neighbours. Counted.
- **Accuracy is measured, not claimed** (§6): rates on a sphere and an ellipsoid, at refine 2 … 6, recorded in
  `docs/analysis.md`.

### 4.5 Length-unit independence

- Every degeneracy test is relative to the triangle's own edges, the `fold_area_vector` rule (CLAUDE.md "NO
  TEST MAY COMPARE A DIMENSIONAL QUANTITY AGAINST AN ABSOLUTE CONSTANT").
- Normalisation uses `inv = (mag > 0.0) ? 1.0/mag : 0.0` (the unit-normal rule).
- So a body scaled by 2⁻²⁰ gives curvature × 2²⁰ and force × 2⁻²⁰ **bit for bit**.

### 4.6 README recipe: balanced force (Shin & Juric 2005; review Popinet 2018)

Spreading `σκn δ` directly produces parasitic currents. The hybrid formulation avoids them, using only
library outputs and two calls to the [coupler's spreading interface](../src/coupling/README.md):
1. Spread the vertex force F (§4.2) and the area-weighted normal `G_v = n_v A_v` to the grid.
2. On the grid, `κ_H = (F·G) / (σ G·G)`.
3. Apply `σ κ_H ∇_h I`, where `∇_h` is **the solver's own** pressure-gradient operator.

The library cannot do step 3, and the README says why.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_geometry.h` | enums, params, the two local functions, convention block |
| `src/geometry/curvature.c` (new) | §4.4 |
| `src/geometry/surface_tension.c` (new) | §4.2, §4.3 |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_surface.c` (new) | wrappers: normals + ghost normal exchange for QUADRIC; optional holder reduction for stats (9 SUM + 1 MIN slots) |
| `tests/test_curvature.c`, `tests/test_surface_tension.c` (new), `tests/CMakeLists.txt` | §6 |
| `src/geometry/README.md` | derivations, convention, rim modes, the §4.6 recipe |
| `CLAUDE.md` | placement table: curvature and surface tension rows; "What exists today" |

## 6. Tests

- **Curvature.**
  - Spheres R = 1 and R = 2⁻²⁰, refine 2 … 6:
    - QUADRIC max error converges at second order;
    - COTAN mean error converges;
    - both rates recorded.
  - Ellipsoid against analytic κ(x).
  - A flat patch gives exactly 0.
  - The Gaussian curvature integral is 4π (Gauss–Bonnet) to round-off on a closed sphere. The angle deficit is
    exact in this.
- **Conservation.** On a closed sphere at np 1, 2, 4, 8, 27, with the verdicts `agreed()`:
  - |ΣF| ≤ 1e-13·σA for both forms;
  - |Σ x×F| ≤ 1e-13·σAR for the area-gradient form.
- **Laplace.** `−Σ_v F_v·n_v / Σ_v A_v → 2σ/R` (the pressure jump), with a measured rate.
- **Marangoni.** Flat sheet, σ linear in x:
  - the normal force is 0;
  - the total tangential force is exactly `Δσ · W`;
  - the edge form and the vertex form agree in total.
- **Partition independence.** Owned forces by global id equal the np = 1 forces, quantised at `llround(x·1e12)`,
  because the scatter order can change the last bit.
- **Unit independence.** 2⁻²⁰ scaling bit for bit (§4.5).
- **Rims.**
  - A flat sheet across four walls: zero interior force, and the rim force is exactly 0 under
    `WALL_TANGENTIAL`.
  - A spherical cap on a wall: interior force plus rim force balance (Young–Laplace closure).
- **Non-vacuity.** Counted owned vertices whose 1-ring contains a ghost must be > 0 at np ≥ 2.

## 7. Gate

- New functions only: all digests are unchanged.
- **Interleaved A/B.**
  - The area-gradient pass is measured against `MFTLComputeVertexNormals` (expected: the same order, one
    scatter).
  - The QUADRIC cost per vertex is measured.
  - The fused normals+force option is measured, and kept only if it wins.
- Zero warnings in both manuals.

## 8. Considered and deferred

- **Higher-order fits** (jets, Cazals–Pouget; cubic fits): revive if QUADRIC's measured accuracy limits a
  Laplace-pressure test.
- **Grid-based curvature** (height functions): a VOF-side tool, not the library's.
- **Fusing force into `MFTLEnsembleUpdate`**: rejected on the same grounds as §1.6 of `docs/analysis.md`.
  `MFTLEnsembleUpdate` is divide-bound already, and the force is not needed every time the quality is.

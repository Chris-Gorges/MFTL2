# 07 — Point queries, volume correction, sizing field

Status: PLANNED (2026-10-05, branch `christian`)
Depends on:
- `03-body-integrals.md` (volumes, rim walls, the deferred ray-parity design §4.6);
- `04-attached-fields.md` (sizing field);
- `05-curvature-and-surface-tension.md` (curvature for adaptive sizing);
- `06-point-cloud-coupling.md` (the solver points and their exchange).

Three independent milestones (D1, D2, D3), in that order of value. Each can land alone.

## 1. Purpose

Three things every FT or IBM solver needs next to forces and interpolation:

| | Need | Today |
|---|---|---|
| D1 | the indicator (phase / solid fraction) and the signed distance **at the solver's points** | only a lattice window (`MFTLRasterizeSignedDistance`); inside test deferred (`plans/03` §4.6) |
| D2 | restoring each body's volume after advection, which loses volume through interpolation error | the Newton correction exists, private to topology change (`topo_driver.c` ~583–640, `displacement_field` :238, `project_rim` :210) |
| D3 | a target edge length that differs per body (a polydisperse spray) and per vertex (curvature) | one global `target_length`, checksummed on the wire (`remesh_comm.c:311`) |

## 2. Findings that decide it

- **The signed distance of a point near the surface is a closest-point query.** That is already exact and
  local:
  - `MFTLClosestPointOnTriangle` reports which feature was hit;
  - the angle-weighted pseudo-normal makes the sign exact at edges and vertices (Bærentzen–Aanæs 2005);
  - `MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE)` is in place for it.

  What is missing is only the bridge to the solver's points, which 06 builds.
- **Owned triangles suffice, and the merge makes the answer np-independent.**
  - Plan 06's halo gives a triangle's owner every solver point within `band + max edge` of its box.
  - So every (point, triangle) pair within the band is examined by at least one rank.
  - Each rank produces a candidate φ from its owned triangles only. The solver side folds the candidates with
    `MFTLSdfMerge`, which is commutative and associative, so the result is **bitwise identical at every np**.
    That is the property topology change already relies on.
- **The far field needs a sign, not a distance, and the sign is a parity.**
  - Outside a body's bounding box the answer is "outside", free.
  - Inside it, a ray-crossing count with exact predicates (adaptive orient2d plus Simulation of Simplicity)
    is the only exact local rule. Generalised winding numbers are approximate or O(N) per query, and are
    undefined for periodic or wall-closed sets (`plans/03` §8).
  - Distributed form: crossings counted from OWNED triangles in each rank box along the ray, then summed
    along the Cartesian row by one exclusive scan per row. That is O(cuts[k]) latency, never O(P) memory.
- **Volume correction is already solved and must not be re-derived.**
  - Topology change applies a fixed number (3) of Newton steps on a uniform normal displacement, with the
    exact first variation `Σ_t S_t · mean(u)`.
  - Rim vertices keep the wall-normal component of their displacement at zero, and are projected back onto
    the constraint.
  - The fixed count is deliberate: a stopping test on a reduced double is two ranks disagreeing about
    another collective (CLAUDE.md, topology invariants).
  - Extract the kernel; keep topology's per-cluster combination. This is exactly what `plans/03` did with
    the flux kernels.
- **A sizing field is one mechanism for both needs.**
  - Per-body sizing is a constant per body.
  - Curvature sizing is `L = √(6ε/κ − 3ε²)` (Botsch–Kobbelt, the ε-chord rule).
  - Both are a per-vertex INTENSIVE field from `04`, and the split interpolates it for free.
  - The edge bounds become `min(L_a, L_b)·{4/5, 4/3}`.
  - The global bounds are read in nine places across four files (`remesh_ops.c` 3, `remesh_driver.c` 2,
    `remesh_parallel.c` 2, `remesh_comm.c` 2). Those, and the frozen-band rule `h > 4·delta`, are the
    complete list of sites to change.

## 3. Decision

- **D1:**
  - `MFTLCouplerSignedDistance`, the narrow-band φ at the solver's points through plan 06;
  - `MFTLCouplerIndicator`, a smoothed Heaviside from it, plus a per-body box classification;
  - the exact far-field parity as the second step of D1.
- **D2:** `MFTLEnsembleCorrectVolumes` over a shared local kernel. Topology change calls the kernel
  bit-identically.
- **D3:** `MFTL_RemeshParameters.sizing_field`, with -1 meaning off (digest-neutral), plus two helpers that
  fill it.

## 4. Design

### 4.1 D1 — signed distance and indicator at solver points (`src/coupling/`, `include/mftl_coupling.h`)

```c
/* Collective. phi[j] for every solver point within `band` of some surface (others MFTL_SDF_UNKNOWN);
   body_gid (nullable) names the nearest body. Requires angle-weighted vertex normals exact on ghosts. */
int MFTLCouplerSignedDistance(MFTL_Coupler *c, MFTL_Ensemble *ens, double band, double *phi, ptrdiff_t stride, int *body_gid);

/* Collective. H in [0,1] per solver point: 1 inside any closed body, smoothed over |phi| < eps
   (H = ½(1 − φ/ε − sin(πφ/ε)/π)); outside the band the sign comes from 4.1.2. */
int MFTLCouplerIndicator(MFTL_Coupler *c, MFTL_Ensemble *ens, double eps, double *H, ptrdiff_t stride);
```

**4.1.1 Band.**
- On the MFTL side, bin the owned triangles by AABB, inflated by the band, into the coupler's bins.
- For each band point in an occupied bin, take the closest owned triangle and sign it by the pseudo-normal
  of the feature hit.
- The candidates go back over the reverse graph and the solver side folds them with `MFTLSdfMerge`.
- The band is `≥ ε + support`, so an indicator computed here is consistent with the force spread in 06.

**4.1.2 Far field** (second step of D1):
- A point outside every closed body's reduced bounding box is outside.
- Otherwise, cast along the non-periodic axis of `plans/03` §4.6 (any axis for a closed body). Each rank
  counts crossings of its OWNED triangles with the ray segment inside its box, using exact orient2d plus
  SoS. These are new predicates, compiled with `-ffp-contract=off` like `bvh_predicates.c`; the plain
  `orient2d` stays separate.
- An `MPI_Exscan` on the row communicator of the Cartesian grid sums the counts along the ray direction.
- Periodic axes are not ray axes. A body that wraps every axis has no exterior, and that is reported as
  UNSUPPORTED, consistent with `MFTLEnsembleBodyIntegrals`.

### 4.2 D2 — volume correction (`src/geometry/` kernel, `src/ensemble/` wrapper)

```c
typedef struct MFTL_VolumeCorrectionParams {
  int n_newton;                       /* FIXED step count, default 3; never a tolerance          */
  MFTL_RimProjector rim_project;      /* same contract as MFTL_RemeshParameters::rim_project       */
  void *rim_user;
} MFTL_VolumeCorrectionParams;

/* LOCAL kernels, shared with topology change, bit-identical arithmetic:
   d(V)/dd for a uniform normal displacement, and applying displacement d to owned vertices. */
double MFTLVolumeDisplacementDerivative(MFTL_Object *obj);
void   MFTLApplyNormalDisplacement(MFTL_Object *obj, double d, const MFTL_VolumeCorrectionParams *p);

/* COLLECTIVE: per body, by LOCAL SLOT, the target volume (the same on every holder; Debug-checked by a
   MIN/MAX reduction). n_newton x (one holder reduction + MFTLUpdateGhostPositions). */
int MFTLEnsembleCorrectVolumes(MFTL_Ensemble *ens, const double *target_by_slot, const MFTL_VolumeCorrectionParams *p, double *residual_by_slot);
```

- Volumes come from the `MFTLBodyFluxVolume` kernel under the axis rule of `plans/03`, so open bodies closed
  by walls are corrected too.
- A body whose volume is UNDEFINED or UNSUPPORTED is left alone and reported.
- `topo_driver.c` keeps its per-cluster loop and calls the two kernels. **Gate:** the topology census is
  bit-identical.

### 4.3 D3 — sizing field (`src/remeshing/`)

- `MFTL_RemeshParameters` gains `int sizing_field` (-1 = off) and `double sizing_gradation` (default 0.3).
  The field id rides in the parameter checksum (`remesh_comm.c:311`).
- **Edge test.** `L_e = min(L_a, L_b)`; split if `ℓ² > (4/3 L_e)²`, collapse if `ℓ² < (4/5 L_e)²`.
  - The split vertex inherits L by 04's INTENSIVE rule.
  - The collapse-approval stretch bound uses the same `L_e`.
- **Smoother.** The area-weighted tangential target is weighted by `1/L²` (density-adapted, Botsch–Kobbelt
  adaptive remeshing).
- **Distributed drivers:**
  - `count_hidden_edges` (`remesh_parallel.c:351`) uses `L_e`.
  - The frozen-band rule `h > 4·delta` and the coarsening factors use the reduced MAXIMUM of L over the body,
    which is conservative.
  - `MFTLRemeshChooseStrategy` is unchanged: it measures the radius, not L.
- **Gradation.** Before a pass, a few Jacobi sweeps clamp `|L_a − L_b| ≤ g·ℓ_ab`, with one
  `MFTLUpdateGhostFields` per sweep. Without it, adjacent targets that differ by more than the 4/5…4/3 window
  make split and collapse oscillate, which is the "stretch bound" failure CLAUDE.md records.
- **Off path.** `sizing_field == −1` selects the existing code by one branch hoisted per pass. Digests are
  bit-identical.
- **Helpers (collective):**
  - `MFTLEnsembleSizingPerBody(ens, field, length_of_gid)`: the callback is evaluated per local body. It must
    be a pure function of the gid.
  - `MFTLEnsembleSizingFromCurvature(ens, field, eps, L_min, L_max)`: uses 05's curvature.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_coupling.h`, `src/coupling/coupling_sdf.c` (new), `coupling_comm.c` | D1 |
| `src/bvh/bvh_predicates.c` (or a new `src/geometry/exact_predicates.c` with the same compile flag), `CMakeLists.txt` | exact orient2d + SoS |
| `include/mftl_geometry.h`, `src/geometry/volume_correction.c` (new) | D2 kernels |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_integrals.c` | `MFTLEnsembleCorrectVolumes` |
| `src/topology_change/topo_driver.c` | call the shared kernels |
| `include/mftl_remeshing.h`, `src/remeshing/remesh_ops.c`, `remesh_driver.c`, `remesh_parallel.c`, `remesh_smooth.c`, `remesh_comm.c` | D3 |
| `tests/test_point_queries.c`, `tests/test_volume_correction.c`, `tests/test_sizing.c` (new) | §6 |
| READMEs of coupling, geometry and remeshing; `CLAUDE.md` | rules, recipes, the new conventions |

## 6. Tests

- **D1.**
  - Sphere and torus: `|φ − φ_exact| ≤ C h²` in the band.
  - The sign is exact at every band point, including points equidistant to an edge and to a vertex (the
    pseudo-normal cases).
  - The result is bitwise identical at np 1, 2, 4, 8, 27.
  - `Σ (1 − H) ΔV → V` at second order.
  - Far-field parity: the exact count at points on the ray through a vertex and along an edge (SoS),
    np-independent, and the periodic UNSUPPORTED case.
  - Non-vacuity: band points whose candidates came from ≥ 2 ranks, counted, must be > 0.
- **D2.**
  - A sphere perturbed by 5 % reaches its target to 1e-12 in 3 steps.
  - The Rayleigh–Taylor open surface: volume restored, rim still on its walls.
  - The topology census is bit-identical before and after the refactor.
  - Holders disagreeing on a target are reported in Debug.
- **D3.**
  - Two spheres with radius ratio 10 and per-body L ∝ R: each reaches its own mean edge length within 10 %,
    on the serial, spanning and dispersed paths.
  - Curvature sizing on an ellipsoid: the edge length follows `√(1/κ)` to within the window.
  - No oscillation: the op count decays to 0.
  - `sizing_field = −1` gives bit-identical digests.

## 7. Gate

- All digests are bit-identical with the features off: D3 off, and D2 only changes `topo_driver.c` through
  bit-identical kernels.
- Measured and recorded in `docs/analysis.md`:
  - D1 cost against `MFTLCouplerInterpolate`;
  - D2 cost against one geometry pass;
  - D3 off-path neutrality by interleaved A/B.
- Zero warnings in both manuals.

## 8. Considered and deferred

- **Exact cut-cell volume fractions** (clipping triangles against cells, IRL-like): needs cell geometry.
  Revive with a rectilinear-solver consumer.
- **Fast or generalised winding numbers:** see `plans/03` §8.
- **Anisotropic (metric-tensor) remeshing:** revive for thin films, where an isotropic L wastes resolution
  along the film.
- **A volume correction weighted by curvature** rather than uniform: uniform is what topology change
  validated. Revive if a measured shape drift is attributed to it.

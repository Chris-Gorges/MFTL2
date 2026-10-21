# 09 — Surface PDE operators, membranes, contact-line hooks

Status: PLANNED (2026-10-05, branch `christian`)
Depends on: the implemented [optional fields](../src/object/README.md) (unknown storage; membrane reference semantics belong to F3), `05-curvature-and-surface-tension.md` (curvature, the area-gradient force, Marangoni with σ(Γ))

Four milestones (F1 operators + solver, F2 interfacial rheology recipes, F3 membranes, F4 contact-line
hooks), plus one recorded non-decision (continuous collision).

## 1. Purpose

The physics that lives ON the interface:
- **insoluble and soluble surfactant transport:** an advection–diffusion equation for Γ on a moving surface;
- **surface viscosity** (Boussinesq–Scriven);
- **elastic membranes** of capsules and red blood cells, the flagship IBM case;
- **contact lines** where an open surface meets a wall.

multiflow has `surfacescalartransport.c`, `surfaceviscosity.c` and `redistributesurfacescalar.c`, each
hand-written on MFTL1's pointer mesh.

All four rest on a handful of discrete operators and one communication pattern the library does not have
yet.

## 2. Findings that decide it

- **P1 finite elements on triangles are the right discretisation.** With linear hat functions φ_v:
  - lumped mass `M_v = Σ_{T∋v} A_T/3`;
  - stiffness `K = ` the cotangent Laplace–Beltrami;
  - surface gradient: constant per triangle;
  - divergence: its adjoint.

  Applied to the coordinate functions, K gives `K x = ∇A`: exactly the area gradient of 05. So surfactant
  diffusion, curvature and surface tension share one consistent operator family. That consistency is the
  discrete form of the identity Δ_s x = −κn.
- **A matvec on owned vertices is local** after one ghost refresh of the input, by the vertex-complete halo.
  The implemented `MFTLUpdateGhostFields` provides that refresh.
- **Membrane bending is NOT 1-ring local, and that decides a new primitive.**
  - The dihedral-angle energy of edge (j,k) involves the two triangles at the edge. For an owned vertex v,
    the edges opposite v in its fan have their outer triangle in the 2-RING, which the halo does not contain.
  - The fix is not a second ghost layer (absent, and expensive). It is to compute each edge's force ONCE, on
    a rank holding both its triangles: every owned triangle has all edge neighbours. Then **send the
    contributions that landed on ghost vertices back to their owners.**
  - `MFTLAccumulateGhostFields` is that reverse exchange: ghost → owner, summed in ascending source rank, so
    it is deterministic. It is the transpose of `MFTLUpdateGhostFields`, over the same schedule.
  - Any per-edge or per-triangle scatter force needs it, so it is general.
- **One edge, one computation.** An edge is computed by the rank owning the LOWER-global-id triangle of the
  pair. That is decidable locally (both triangles are present) and partition-independent.
- **A stopping test on a reduced double is a deadlock risk** (CLAUDE.md, the volume correction rule).
  - An iterative solver must therefore iterate a count every rank agrees on.
  - `MPI_Allreduce` returns identical bits on every rank in practice, but the library's own reductions
    guarantee it by construction. So the residual norm is reduced with `MFTLEnsembleReduceBodies` at GLOBAL
    scope (fixed fold order), never with an ad-hoc `MPI_Allreduce`.
  - A global scalar per iteration is O(1), not O(B): the whole ensemble solves as one block-diagonal system.
- **Membranes are not remeshed.** Their reference shape is per triangle and remeshing would need a reference
  remap. Capsule codes keep the mesh fixed, which is standard in the literature (Krüger et al. 2011;
  Guckenberger et al. 2016).
  - The plan says so.
  - F3 must refuse remeshing a body carrying a membrane reference field in both Release and Debug;
    the current registry has no membrane-specific transfer policy or guard.
- **The contact angle is a geometric quantity at the rim.** The library knows it exactly; the model that
  turns it into a velocity or a force (Cox–Voinov, Navier slip, hysteresis) is solver physics.
  - `MFTL_RimProjector` (plan 02) already lets the application place rim vertices.
  - What is missing is the measurement.

## 3. Decision

- **F1.**
  - Local P1 operators.
  - The ghost → owner accumulation.
  - A preconditioned-CG ensemble solver for `(αM + βK) x = b`, with a deterministic stopping rule.
- **F2.** README recipes plus one force kernel (surface viscosity). Surfactant transport is a recipe over F1
  and the existing field storage, not a black box: Lagrangian or ALE, soluble or not, is the solver's choice.
- **F3.**
  - Skalak / neo-Hookean in-plane forces and dihedral bending (Helfrich-cotangent as an option).
  - Global area and volume penalty terms.
  - All `−∂E/∂x`, with exact gradients.
- **F4.** `MFTLRimContactAngles`.

## 4. Design

### 4.1 F1 — operators (`include/mftl_geometry.h`, `src/geometry/surface_operators.c` new; collectives in `src/ensemble/ensemble_surface.c`)

```c
/* LOCAL; exact on owned vertices given ghost inputs current. ncomp-interleaved arrays like fields. */
void MFTLSurfaceLumpedMass(const MFTL_Object *obj, double *m);                                        /* [verts.n_local]       */
void MFTLSurfaceStiffnessApply(const MFTL_Object *obj, const double *u, int ncomp, double *Ku);       /* u over verts.n         */
void MFTLSurfaceGradient(const MFTL_Object *obj, const double *u, int ncomp, double *grad);           /* [3*ncomp*tris.n_local] */
void MFTLSurfaceDivergenceApply(const MFTL_Object *obj, const double *w, int ncomp, double *divw);    /* adjoint of gradient    */

/* COLLECTIVE, the transpose of MFTLUpdateGhostFields: ghost partial sums are added into the owners
   (ascending source rank), then the ghost copies are zeroed. */
int MFTLAccumulateGhostFields(MFTL_Ensemble *ens, const int *ids, int nfields);

/* COLLECTIVE: solves (alpha M + beta K) x = b per vertex field, all bodies at once, Jacobi-preconditioned CG.
   Residual norms reduced with a fixed fold order; stops on the reduced value (bit-identical everywhere)
   or at max_iter. Reports the iteration count (identical on all ranks; asserted). */
int MFTLEnsembleSurfaceSolve(MFTL_Ensemble *ens, double alpha, double beta, int b_field, int x_field, double rtol, int max_iter, int *iters);
```

### 4.2 F2 — recipes and the surface-viscosity force

- **Insoluble surfactant, Lagrangian mesh** (vertices move with the full fluid velocity): the mass per
  element is conserved, so the application applies `Γ^{n+1} = Γ^n A^n / A^{n+1}` per triangle during
  physical motion. Registering `MFTL_FIELD_CONSERVATIVE_DENSITY` does not perform this update; its
  [local remeshing transfer](../src/remeshing/README.md) preserves `sum(A * Γ)` during remeshing.
  Diffusion is implicit: `(M + Δt D K) Γ = M Γ*` via `MFTLEnsembleSurfaceSolve`. These P1 operators use
  vertex unknowns, so F2 still needs an explicit conservative triangle/vertex conversion or a conservative
  vertex-field policy with its nodal measure. Neither is supplied by the current field API.
- **ALE mesh** (normal velocity plus tangential smoothing): adds `∇_s·(Γ u_t)`. Advection operator deferred
  (§8). The README states that the Lagrangian form is the supported one.
- **Equation of state.** Langmuir `σ = σ₀ (1 + β ln(1 − Γ/Γ∞))` gives a σ field, which 05 turns into
  Marangoni plus capillary forces.
- **Soluble surfactant.** The bulk–interface flux comes from the solver's bulk concentration through
  `MFTLCouplerInterpolate` (06), and goes back as a sink through `MFTLCouplerSpread`.
- **Surface viscosity:**
  - `MFTLSurfaceViscousForce(obj, vel_field, mu_s, kappa_s, force)`.
  - The P1 velocity gradient per owned triangle gives the Boussinesq–Scriven stress
    `τ = (κ_s − μ_s)(∇_s·u) P + μ_s (∇_s u P + P ∇_s uᵀ)`.
  - The force on the vertices is the weak form `F_v = −Σ_T A_T τ_T : ∇_s φ_v`.
  - Local over owned triangles, with ghost contributions sent back by `MFTLAccumulateGhostFields`.

### 4.3 F3 — membranes (`src/geometry/membrane.c` new)

```c
typedef struct MFTL_MembraneParams {
  int model;                  /* MFTL_MEMBRANE_SKALAK | MFTL_MEMBRANE_NEO_HOOKEAN                    */
  double shear_modulus, area_dilation_modulus;      /* Gs, C (Skalak)                                  */
  int bending_model;          /* MFTL_BENDING_DIHEDRAL | MFTL_BENDING_HELFRICH_COTAN                   */
  double bending_modulus, spontaneous_curvature;
  double area_penalty, volume_penalty;              /* global constraints, 0 = off                       */
  int reference_field;        /* persistent triangle reference data; representation defined by F3      */
} MFTL_MembraneParams;

int MFTLEnsembleSetMembraneReference(MFTL_Ensemble *ens, int reference_field);               /* collective: store current shape */
int MFTLEnsembleMembraneForce(MFTL_Ensemble *ens, const MFTL_MembraneParams *p, int force_field);  /* collective */
```

- **In-plane.** Per owned triangle:
  - map the current and reference triangles to 2D;
  - deformation gradient → principal stretches → the strain energy density (Skalak 1973 / neo-Hookean;
    Charrier et al. 1989);
  - nodal forces `−∂(A₀W)/∂x`, analytic per linear element.
  - Contributions to ghost corners are accumulated back.
- **Bending.**
  - Dihedral: per edge, computed once by the owner of the lower-gid triangle; the four vertices receive the
    exact gradient of `k_b (1 − cos(θ − θ₀))`.
  - Helfrich: built from 05's cotangent curvature: `H` at owned vertices, then a ghost refresh, then the
    force; documented as the more expensive option.
  - Guckenberger et al. 2016 compare the two; the README cites their recommendation and our measurement.
- **Constraints.** Global area and volume from `MFTLEnsembleBodyIntegrals` (holder reduction), then local
  penalty forces.
- **Refusal.** F3 must add a Release and Debug guard covering all public remeshing entry points for a
  body carrying a `reference_field`; ordinary field interpolation is not a reference-shape remap.

### 4.4 F4 — contact-line hooks (`src/geometry/rim_geometry.c` new or `remesh_rim.c`'s neighbour)

```c
typedef struct MFTL_RimVertexInfo {
  int vertex;                 /* owned rim vertex index                                     */
  int wall_axis, wall_side;   /* the wall it lies on (the 1e-8 L rule), -1 if none          */
  double conormal[3];         /* in-surface, pointing out of the surface across the rim     */
  double angle;               /* between the surface and the wall, measured through the side the normals do NOT point into */
  double length;              /* half the incident rim edge lengths: the line-integral weight */
} MFTL_RimVertexInfo;
int MFTLRimContactAngles(const MFTL_Object *obj, MFTL_RimVertexInfo *out, int *n_out);   /* LOCAL, owned rim vertices */
```

The angle convention matches the volume sign of `plans/03`: the reported phase is the side the normals do not
point into. A spherical cap of contact angle θ on a wall reports θ.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_geometry.h`, `src/geometry/surface_operators.c`, `membrane.c`, `rim_geometry.c` (new) | F1 kernels, F3 kernels, F4 |
| `include/mftl_communication.h`, `src/communication/ghost_sync.c` | `MFTLAccumulateGhostFields` |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_surface.c` | solver, membrane and viscosity collectives |
| `src/remeshing/remesh_driver.c` | the membrane-reference refusal |
| `tests/test_surface_operators.c`, `tests/test_membrane.c`, `tests/test_rim_angles.c` (new) | §6 |
| `src/geometry/README.md` | operator derivations, surfactant and viscosity recipes, the bending comparison |
| `CLAUDE.md` | the reverse-accumulation pattern, the one-edge-one-rank rule, the not-remeshed rule |

## 6. Tests

- **F1.**
  - `K x = ∇A` against 05's area gradient, to round-off.
  - The sphere Laplacian eigenvalues `l(l+1)/R²` for spherical harmonics converge at second order.
  - Implicit diffusion conserves `Σ M Γ` to 1e-14.
  - CG: the iteration count is identical on every rank; the result equals np = 1 to 1e-12.
  - `MFTLAccumulateGhostFields`: the owners' sums equal the np = 1 scatter by gid, quantised.
  - Non-vacuity: ghost contributions sent, counted, must be > 0.
- **F2.**
  - A flat sheet with linear Γ and linear σ(Γ): the Marangoni total is exact.
  - Surface viscosity: a rigid rotation gives zero force; a uniform dilation gives the analytic force.
- **F3.**
  - The finite-difference energy gradient matches the analytic force to 1e-7 relative (central differences,
    random vertex).
  - The reference shape gives zero force.
  - Net force and torque are zero on a closed capsule.
  - An inflated neo-Hookean sphere matches the analytic tension–stretch law.
  - Dihedral bending of a cylinder against the analytic `k_b/R²` energy density.
  - Partition independence at np 1, 2, 4, 8, 27.
- **F4.**
  - A spherical cap on a wall reports θ to O(h²).
  - A flat sheet perpendicular to its walls reports 90° exactly.

## 7. Gate

- New functions only: digests unchanged.
- The cost per membrane force evaluation against one geometry pass, recorded.
- Zero warnings in both manuals.

## 8. Considered and deferred

- **ALE surface advection operator** (SUPG or finite volume on the dual mesh): revive with a solver that
  moves the mesh by normal velocity only.
- **Remeshing membranes with reference remap:** revive when a capsule deformation exceeds what a fixed mesh
  resolves.
- **Contact-line models** (Cox–Voinov, hysteresis, slip): physics owned by the solver. The library measures.
- **Continuous collision detection** (deferred issue 16).
  - When: the solver's CFL step lets surfaces close a gap larger than the contact envelope in one step.
  - How: an optional previous-position vertex field gives each triangle its swept
    volume. Then swept AABBs in the LBVH (`MFTL_RefitLBVH` over the union of both positions), followed by
    exact vertex–face and edge–edge CCD (Brochu, Edwards & Bridson 2012; Wang 2015).
  - Not now: nothing has measured a tunnelled contact in a real run. The pinned miss
    (`test_tunnelling_between_two_timesteps_is_a_known_miss`) stays the record.

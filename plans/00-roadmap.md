# 00 — Roadmap: what MFTL2 is still missing

Status: 2026-11-13, branch `christian` at `d4f2b6a`; §3 D (known deferred issues) added 2026-11-14. This file
replaces `plans/01`–`09`. It is the single
source for detailed implementation plans: write one file per item, `plans/NN-<item>.md`, and cite the item id
used here (A1, B3, D20, ...).

Sources reviewed:
- `christian`: the headers, the module READMEs, the "Known deferred issues" in `CLAUDE.md`, and plans 01–09.
- MFTL2 `origin/main` (Berend, `c0b8de0`): the MFTL1-parity port. Its plans 00–18 were deleted in `5fc00ef`;
  read them at `6188067`.
- multiflow `origin/master` (`5a734d15`): front tracking (FT), static IBM and MoveIBM. They call MFTL2 `main`
  through `src/include/mftl_adapter.h`.
- MFTL1 (`~/git/mftl`, `89fd8aa`).

## 1. What a solver asks of the library

multiflow is the reference consumer. The list below comes from its call sites: about 1 450 adapter calls, of
which about 1 300 are in FT, 100 in MoveIBM and 5 in static IBM.

**Front tracking** (`src/interfacial/fronttracking/solvefronttracking.c`). Each step:
1. Integrates pressure, viscous and surface-tension forces per body.
2. Advects the vertices. The velocity is Peskin-interpolated across the dual decomposition (`sendnonlocal*.c`).
3. Transports surfactant, remeshes, and recovers the volume along the normals.
4. Spreads the element surface-tension forces and the indicator gradient, and solves a Poisson equation for the
   indicator. The alternative is the **sharp** path (CSF, integral surface tension, IRL indicator), which works
   as follows:
   - every front triangle within 15 dx goes to every fluid rank (`MFAdapterSendLocalTrianglesToAll`);
   - IRL cuts the fluid cells to get volume fractions;
   - curvature is a paraboloid fitted to a 2-ring;
   - the signed distance comes from a nearest-*vertex* sweep plus a Monge patch (`integralsurfacetension/distance.c`,
     `ftclosestpoint.h`).
5. Runs the marching-cubes topology change.

**Static IBM** (`src/lagrangian/ibm/`).
- The library reads a GTS file and returns triangle centroids as markers (`getmarkersfromexternalmesh.c`).
  Nothing else.
- Direct forcing, MLS weight renormalisation and the covariant smoothing of the spread source (the "smooth IBM",
  `addibmsource.c`) are solver arithmetic on the fluid grid.
- Inside tests are analytic spheres (`iscellinparticle.c`).

**MoveIBM** (`src/lagrangian/moveibm/`), rigid bodies.
- Once, at setup, the library computes volume, centre and inertia.
- Every step it translates and rotates every body incrementally (`updatemoveibmftlbodies.c`), redistributes it, and
  supplies triangle centroids as markers.
- The implicit forcing assembles a marker system in PETSc from the interpolation and spreading weights.
- The internal-mass (fictitious fluid) correction **all-gathers every triangle of a particle to every rank** and
  clips the cells by the triangle planes with IRL (`gathermoveibmparticleplanes.c`). That is O(triangles) per rank
  and correct for convex bodies only.

**No multiflow physics calls** point-in-object, triangle pairs within a distance, or thin-region pairs; only
adapter unit tests do. Overlap volume is not in the adapter at all. Particle contact is DEM's own, on its own
shapes (spheres, multi-spheres, superquadrics with GJK).

Berend measured MFTL2's share of realistic multiflow runs at ≤ 2.1 %, and 11.5 % with 27 periodic bubbles
(`6188067:plans/00-overview.md`, Stage 18B). So the library's single-rank speed is not what limits these runs.
The O(N) gathers that the solver performs around the library are.

**Conclusion.** Transfer (the coupler), remeshing, topology change, volume correction, fields and body integrals
exist in `christian`. Five things are missing:
- the force on the front;
- distance and inside information at the solver's points;
- rigid-body motion;
- mesh input;
- restart.

## 2. Verdict on what `main` and MFTL1 have and `christian` lacks

**Needed** (scheduled in §3):

| Capability | Where it exists | Item |
|---|---|---|
| Signed curvature and surface-tension force | MFTL1 Laplace–Beltrami; multiflow Frenet forces | A1 |
| Inside/outside and distance at points | `main` `MFTLPointQuery*`; MFTL1 ray casting | A2, new design |
| Translate/rotate a body; inertia tensor | `main` transforms, `MFTLMomentOfInertia` | A3 |
| GTS reader | `main`, MFTL1 | A4 (plus STL) |
| Restart mesh | `main` `MFTL*NativeMesh` | A5 |
| Removing or inserting a body mid-run | `main` `MFTLEnsembleRemoveObject` / `TransferObject` | B2, narrower |

**Not ported.** Do not reopen these without new evidence:

| `main` / MFTL1 | Why not |
|---|---|
| Per-vertex periodic image counters, homological winding | The extent rule (narrower than L/2, measured to 3L/4) is exact for every body a solver has. `MFTLEnsembleCheckPeriodicExtents` checks it. |
| Phase-boundary sets | Per-body integrals plus caller arithmetic cover every example (C9). |
| Boundary tags, loops, boundary-curve objects | A rim is identified by its position. The rim curve is two circular arcs. `MFTL_RimProjector` takes application constraints. |
| MFTL1 remesh criteria and sweep parity; Kuprat, Laplacian and roughness smoothing; rank-0 boundary cleanup; x-only slab balancing; LAPACK fit | Superseded, with measurements (`docs/analysis.md` §4, §5, §14). Distributed without a gather. |
| Opaque per-triangle byte payloads | Typed registered fields with transfer policies. The conservative density is MFTL1's surface scalar. |
| `twin_gid`, pointer handles, the edge facade, per-entity accessors, single-vertex kernels | Implicit indexing makes edge identity `3t+k`. Kernels work on arrays, not on one entity per call. |
| 2-/3-ring neighbourhoods; topological ghost growth | Curvature comes from a 1-ring position-plus-normal fit (A1). Edge forces use reverse accumulation (B4). |
| Rectilinear cuts that match the fluid decomposition | The coupler decouples the two decompositions. Cuts matter only for load balance (C5). |
| Collective per-point ray casting with random rays and retries | Costs collectives per query and has no fixed direction. Replaced by A2's exact predicates. |
| OBB tree, covariance boxes, Graham scan | Replaced by the LBVH. |
| Overlap volume of two bodies (`MFTL_CalculateOverlap`) | No consumer. Gap and penetration come from B3; the overlap-volume model is C7. |
| Box, cylinder and plane generators | `MFTLBuildMeshFromLevelSet`, or `MFTLBuildIndexedMesh` with `MFTLPublishGeneratedBodies`; CAD comes through A4. |
| Mesh-adaptive global tolerance `MFTL_TOLERANCE` | Every test is relative to the entity's own size. |
| Serial bracket; sending all triangles to all ranks | Exactly the O(N) per rank the library exists to avoid. |
| GTS and native writers | Output is VTU (B7); restart is A5. |

## 3. Missing work

Priorities:
- **A** blocks FT or IBM in production;
- **B** is needed for coverage or scale;
- **C** waits for a consumer or a measurement;
- **D** is a known limit with no scheduled fix.

### A1 — Surface tension and signed curvature (FT)

**Need.** The force the coupler spreads. Today every solver re-derives curvature and force.

**Have.**
- unsigned Meyer curvature, used for sizing (`MFTLComputeVertexCurvatures`);
- area- and angle-weighted vertex normals;
- the vertex-complete halo;
- `MFTL_SPREAD_INTEGRATED`.

**Build.**
- **Vertex area-gradient force** (the default):
  - `F_v = −Σ_{T∋v} σ_T ½ n_T × (x_k − x_j)`, one scatter, exact on owned vertices without a message;
  - ΣF = 0 and Σx×F = 0 to round-off;
  - Marangoni for free, from a per-triangle σ.
- **Tryggvason edge form** for centroid markers.
- **Signed mean curvature**, two methods:
  - cotangent;
  - a 1-ring position-plus-normal quadric fit, which needs exact ghost normals (`MFTLEnsembleUpdateVertexNormals`).

  2/R on a sphere with outward normals.
- **Rim modes:** include, wall-tangential (the wall reacts), exclude.
- **Optional:** per-body net force, torque and energy, by one holder reduction.
- **README recipe, not code:** the balanced-force hybrid of Shin & Juric, applied with the solver's own gradient
  operator.

**Gate.**
- |ΣF| and |Σx×F| ≤ 1e-13 relative, at np 1–27;
- a measured convergence rate for the Laplace pressure;
- the 2⁻²⁰ scaling reproduces the result bit for bit;
- a flat Marangoni sheet is exact.

**Design:** `d4f2b6a:plans/05-curvature-and-surface-tension.md` §4 still holds.

### A2 — Distance and inside at the solver's points (the main IBM gap; FT sharp models)

**Need.**
- **IBM:** which cells lie inside a body (fictitious fluid, the internal-mass integral ∫ρu dV, cell tagging,
  solid fraction, AMR labels), for any shape, without gathering the body.
- **Ghost-cell or cut-cell IBM, sharp FT surface tension, curvature extension:** the signed distance and the
  closest surface point near the front.
- **IRL-like indicator, MoveIBM fictitious cells:** the volume fraction of a cell.

**Have.**
- an exact closest point with its feature (`MFTLClosestPointOnTriangle`);
- angle-weighted pseudo-normals;
- a local lattice rasteriser with a commutative merge (`MFTLRasterizeSignedDistance`, `MFTLSdfMerge`);
- the coupler's solver-grid registration and its sparse discovery.

**Build, in three steps:**

1. **Band query.** For every registered solver point within `band` of a surface, return:
   - φ;
   - the body gid;
   - the closest triangle's gid and the barycentric coordinates there, so a solver can carry any vertex field
     (curvature, velocity) to the point.

   φ is computed from OWNED triangles only and merged with `MFTLSdfMerge`, so it is bitwise identical at every np.
   The points travel to the owners of the triangles; triangles never travel to every rank.
2. **Sign everywhere.**
   - A point outside every body's reduced box is outside, at no cost.
   - Inside a box: exact ray parity along a non-periodic axis. It uses adaptive orient2d plus Simulation of
     Simplicity (new predicates, compiled with `-ffp-contract=off`), counts crossings from owned triangles, and
     sums them with an exclusive scan along the Cartesian row.
   - A body that wraps every axis is UNSUPPORTED, as in `MFTLEnsembleBodyIntegrals`.
   - Weigh against: band signs plus a solver-side fill, which is what multiflow's IRL path does. It is cheaper
     where most points are far from every body.
3. **Cut-cell moments** for hexahedral cells, exact for the polyhedral surface, and local to the cell. A crossed
   cell's corners lie in the band, so step 1 signs them. Then:
   - `V(body ∩ cell) = ∫ n_z (z − z_lo) dA` over the triangle pieces clipped to the cell, plus
     `(z_hi − z_lo) · A_in(top face)`;
   - `A_in` is the same identity one dimension down, ending at corner signs.

   This is `MFTLBodyFluxVolume`'s identity restricted to a cell. It replaces MoveIBM's convex plane clipping and
   its all-gather.

**Constraint** (the owner's decision): A2 is not part of the coupler's transfer scope, because the coupler's
nearest-surface queries were removed on purpose. A2 may reuse the solver-grid registration and the window
machinery. The detailed plan decides where it lives.

**Gate.**
- φ is O(h²) accurate, and the sign is exact on a sphere and a torus, including edge and vertex ties;
- the results are bitwise identical at np 1–27;
- Σ(1−H)ΔV → V;
- non-vacuity: band points answered by two or more ranks are counted.

**Design:** `d4f2b6a:plans/07-point-queries-volume-sizing.md` D1 and `plans/03-body-integrals.md` §4.6.

### A3 — Rigid bodies (MoveIBM; also per-bubble loads in FT)

**Need.** Moving every rigid body every step; mass properties; the hydrodynamic load per body.

**Have.**
- per-body area, volume and centroid through holder reductions (`MFTLEnsembleBodyIntegrals`);
- persistent vertex fields;
- `MFTLRedistribute`.

**Build.**
- **Inertia tensor:** second moments by the divergence theorem. It is exact for the polyhedron, and is the same
  kernel family as `MFTLBodyFluxMoments`. MFTL1's lumped tetrahedron formula is not reproduced.
- **A pose, not increments.**
  - Store the body-frame coordinates once, as a persistent 3-component vertex field.
  - Each step, set `x = c + R(q) x_ref` and wrap.
  - Composing incremental rotations, as MoveIBM and old plan 08 do, drifts. The pose does not.
  - Every holder computes the same bits, so owned and ghost copies stay identical without a sync.
- **Loads:** Σ F and Σ (x − c) × F over a body's markers (any coupler marker kind), through
  `MFTLEnsembleReduceBodies`.
- **Not built: the rigid-body ODE integrator.** MoveIBM owns its quaternion integration, prescribed DOFs and
  implicit coupling.

**Gate.**
- the inertia is exact for a box mesh, converges at O(h²) for curved bodies, and is right across a seam;
- after a motion, ghosts equal their owners bitwise;
- the loads are np-independent;
- the volume is invariant.

**Design:** `d4f2b6a:plans/08-ibm-bodies-io-bindings.md` E1, without the Newton–Euler helper.

### A4 — Mesh input

**Need.** STL for CAD and segmented IBM geometries; GTS for multiflow's existing inputs. OBJ costs almost nothing
extra.

**Build.** One pipeline:
1. parse on rank 0;
2. weld vertices: bitwise, by a radix sort of the bit patterns, or with an explicit tolerance;
3. drop degenerate facets, and count them;
4. `MFTLBuildIndexedMesh`;
5. reject non-manifold edges and inconsistent winding, with a message, on every rank;
6. optionally canonicalise to outward winding;
7. `MFTLPublishGeneratedBodies`, then `MFTLRedistribute`.

Scale, translate and rotate on read. The reader inherits the rank-0 generation peak (D20). A
parallel STL read is the later fix.

**Gate.**
- round trips;
- a bad file is rejected on every rank;
- `inv_partition_is_sound` holds after the scatter.

**Design:** old plan 08 E2.

### A5 — Checkpoint and restart

**Need.** Runs of several days. multiflow already restarts FT and MoveIBM.

**Build.**
- **M1, same np, bit-exact continuation.** Each rank writes its pools as they are, with MPI-IO, in a documented
  and versioned raw layout (no HDF5). That covers:
  - owned and ghost entities, in pool order;
  - flags, `partition_settled`, and the id counters;
  - the field registry and the field values.
- **M2, any np.** Read the owned entities, rendezvous by global id, rebuild the twins, then one `MFTLRedistribute`.

**Gate.**
- N steps equal N/2 steps + restart + N/2 steps by digest, with remeshing, fields and topology change active;
- M2 reproduces the census at np 2, 3 and 8 from a checkpoint written at np 4.

**Design:** old plan 08 E3.

### B1 — A caller-supplied sizing field for the remesher

**Need.**
- IBM markers must keep a spacing near the LOCAL fluid h, which varies with AMR.
- Polydisperse sprays want a target length per body.

`MFTLCouplerSurfaceSizing` already computes the per-vertex length the fluid asks for. But the remesher takes one
global target, so today the job-wide minimum is used.

**Build.** A third `MFTL_RemeshSizing` mode that takes `verts.sizing` as given instead of computing it.
- The carrying exists already: split and collapse interpolation, compaction, the migrator, the ghost sync, and the
  edge judges.
- Optionally the minimum of the given field and the curvature field.
- A gradation clamp, if oscillation is measured.

**Gate.**
- the uniform and curvature digests are unchanged;
- two spheres with per-body targets each reach their own mean edge length, on the serial, spanning and dispersed
  paths.

**Design:** the rest of old plan 07 D3.

### B2 — Bodies entering and leaving mid-run

**Need.** Droplets and particles leaving through an outflow; IBM particle injection; deletion by the user.

**Build.**
- **Collective removal by global id.** Every holder destroys its object, and the id is retired, as topology change
  already does.
- **Insertion:** check that generate-or-read, publish and redistribute work on a running, redistributed ensemble
  (registered fields, settled bits, reference volumes), and document it.

**Gate.**
- the census and `inv_partition_is_sound` hold at np 1–27;
- a removed id is never reused.

### B3 — Proximity between bodies, and between a body and a wall

**Need.**
- lubrication and collision models for resolved non-spherical particles;
- the film thickness between approaching droplets, as a physical coalescence criterion.

Nothing in multiflow uses either today, so keep it small.

**Have.** The distributed broad phase, the LBVH, an exact intersection test, the triangle–triangle distance to rounding
with closest points, envelopes, the pair filter, and the spatial halo.

**Build.**
- **Per body pair within an envelope:** the minimum gap, with witness points and a normal. Ties break
  deterministically by triangle gid. The owner of the A-side reports the pair once, and it is reduced per pair.
- **Per body and wall:** the same quantities from the owned vertices, with one holder MIN.
- **Penetration depth of A into B:** the maximum over A's vertices of −φ_B, using A2's local kernels on the halo
  patch.

**Gate.** The result is tested as a SET against the answer before partitioning, like `test_bvh_ghosting`, with a
count of cross-rank pairs as the non-vacuity measure.

### B4 — Surface PDE operators (surfactant)

**Need.**
- implicit surfactant diffusion with Marangoni (multiflow `surfacescalartransport.c`);
- any edge or triangle scatter whose contributions land on ghost vertices.

**Build.**
- **Local operators:** P1 lumped mass, cotangent stiffness apply, surface gradient and divergence.
- **`MFTLAccumulateGhostFields`:** ghost to owner, in ascending source rank; the transpose of
  `MFTLUpdateGhostFields`.
- **A Jacobi-preconditioned CG** for `(αM + βK)x = b` over the whole ensemble, with a deterministic stopping rule.
- **A conservative triangle ↔ vertex conversion** for the conservative-density field.
- **Recipes:** Lagrangian insoluble surfactant; Langmuir σ(Γ), which feeds A1.

**Design:** old plan 09 F1 and the F2 recipes.

### B5 — Coupler efficiency

These come from old plan 06. None of them is measured yet.
1. Reply to known peers in the directory lookup: one discovery barrier fewer.
2. Search every released fetch buffer for a compatible layout.
3. A sparse peer → index lookup instead of linear scans.
4. Reuse released binding storage.

Then measure the synchronisation (two validation reductions per prepared fetch, plus the discovery barriers) before
merging agreements or adding persistent requests.

Later:
- per-region resolution bounds;
- resolution-dependent bins;
- incremental grid updates.

### B6 — Folds at creases (D25)

- Add a fold test against the edge neighbours' normals in flip and collapse, and a fold test in the smoother.
- Restricted to rim-adjacent triangles, it leaves the closed-surface digests unchanged.
- This issue is why the travelling-wave topology test runs at 96×24.

### B7 — VTU output of fields

`MFTLWriteVtu` writes positions, owner, body id and global id, but no registered field. Add an option to write
registered vertex and triangle fields, plus caller arrays, as point and cell data: curvature, surfactant, forces.
Owned entities only, as now.

### C — Only with a consumer or a measurement

| Id | Item | Revive when |
|---|---|---|
| C1 | Contact angle and conormal at rim vertices (old 09 F4) | a contact-line model runs on an open surface |
| C2 | Surface viscosity force, Boussinesq–Scriven (old 09 F2); needs B4 | a case needs it |
| C3 | Membranes: Skalak or neo-Hookean, dihedral bending, area and volume penalties; a body with a reference shape is never remeshed (old 09 F3) | capsules or red blood cells are simulated |
| C4 | Local volume correction (D29); local topology reconstruction (D22); interpenetrating bodies (D28); detection cost (D23) | a production case measures the shape drift or the topology cost |
| C5 | A surface-adaptive decomposition: weighted rectilinear cuts through the `MFTL_OwnerMap` seam | idle ranks are measured with a single interface (RT, a film, a jet), where a uniform grid can leave most ranks without triangles |
| C6 | Continuous collision detection (D16) | a tunnelled contact is measured in a real run |
| C7 | Overlap-volume contact model (MFTL1) | a volume-based contact model for triangulated particles is chosen |
| C8 | Fortran bindings (old 08 E4) | a Fortran consumer exists |
| C9 | Phase sets (`main`) | user-declared multi-body phase compositions must be validated |

### D — Known deferred issues

Known limits with no scheduled fix, measured where possible. Moved here from the local `CLAUDE.md` on 2026-11-14.
The numbers are the ones code comments cite: "deferred issue N" in older text is DN here. D6 (a persistent
redistribution workspace, `docs/analysis.md` §2.3) is done; D8 is retired. Correctness defects, most severe first,
are listed in `docs/analysis.md` §9.

**Performance questions that need the cluster**
- **D1 Morton renumbering** (`MFTLObjectReorderMorton`) has no measured benefit on Apple silicon. It is kept because
  it makes one index space possible and the LBVH needs it. Measure on an x86 node with 1–2 MiB L2 per core before
  putting it in a per-timestep loop.
- **D4 The update kernel is bound by the FP divide/sqrt unit.** It needs both `|n|` and `n/|n|`. The only lever left
  is a hardware `rsqrt` estimate plus a Newton step: unmeasured, not bit-reproducible, microarchitecture-specific.
- **D9 The two I/O file layouts** have only been compared on a laptop SSD, where a shared file cannot win. Open: P in
  the thousands on Lustre or GPFS. The shared-file backend passes `MPI_INFO_NULL` (no collective-buffering hints).
- **D10 Cluster scaling covers only the advected, redistributed timestep.** Measured in October 2026 at commit
  `ac631b7` on 36-core nodes (`scaling/`): strong scaling of 110 592 bodies reaches 0.82 efficiency at P = 288 against
  P = 36; weak scaling with 1000 bodies per rank reaches 0.90. Not yet written up in `docs/analysis.md`, whose §9
  still says "nothing above four ranks". Still unmeasured above four real ranks: remeshing (D11), the holder route's
  P ≈ 45 crossover, neighbourhood against `comm_world` collectives at large P, I/O (D9) and Morton (D1).
- **D11 The spanning remesh's cost at scale.** A pass does one serial call's work whatever `parallel_phase_budget`
  is (a cap, default 8). Over the LeVeque period ONCE takes 1.9–2.7 placements and 2.8–3.7 migrations per call at
  np = 2–8, NESTED 5.5–7.4 and 6.2–8.3. Open: does migration cost stay proportional to the band at P in the
  hundreds, and which mode is cheaper there (`scaling/launch_remesh_*.sh`).
- **D14 Nothing in `src/bvh/` is profiled in time** beyond `orient3d_predicate_profile`. Known cliff: the broad-phase
  cell is `2*max_half + max_envelope`, so one domain-spanning body collapses the grid to a single bucket. A two-level
  grid removes it.
- **D18 Coarsening at P in the hundreds.** A coarse destination lies outside the 26-neighbourhood, so
  `MFTLRedistributeWithOwners` and `MFTLRemeshSyncGhostIds` take their global paths (O(nproc)). Redistribution's
  32-bit `MFTL_DEDUP_BITS` reverts to a sort above 32 distinct destinations, which np ≤ 27 cannot test. The
  single-rank gather is exempt from the second.
- **D21 `compute_owners`** is about 30 % of the redistribution phase with the body skip on, and the skip cannot touch
  it, because it produces the verdict. Design if it is ever built: a per-body slack (the minimum distance of any
  entity to its cell's planes, stored on `MFTL_Object` next to `partition_settled`) against a caller-supplied motion
  bound `dt * v_max` proves "nothing crossed a plane" without a scan. It must ship off, be selected by one hoisted
  branch per body, have a Debug verifier, join `check_object_lists_agree`'s checksum, and be disabled when
  `maps != MFTL_NULL`. Fusing the classification into `MFTLComputeTriangleGeometry` was rejected.

**Caches and fusions not built, because the reference loop redistributes every step**
- **D5 The ghost-exchange schedule** of `MFTLUpdateGhostPositions` is rebuilt every step. An exact cached push
  (`docs/analysis.md` §8.2) would make a sync one `MPI_Neighbor_alltoallv`; it pays only for a solver that
  redistributes less than once per step.
- **D7 The geometry pass after a redistribution is bound by first touch.** The lever is fusing it into the
  redistribution's rebuild, which would put geometry into a module that transmits none.
- **D12 The BVH leaf-leaf narrow phase is batched but not vectorised.** `flush_batch` (`src/bvh/bvh_traverse.c`)
  receives up to 64 independent tests. Start with `MFTL_TriTriProximity` (244 ns/pair against 31 for
  `MFTL_TriTriIntersect`).
- **D13 The spatial-halo overlap list is rebuilt every call.** If a solver redistributes every N steps, cache it on
  `MFTL_SpatialHalo`, not on `MFTL_Domain`. A halo that survives a redistribution is not wanted: it is stale exactly
  when ownership changed.

**Structural and accounting limits**
- **D2 The half-edge triple invariant is checked only in tests** (`inv_implicit_indexing`, part of
  `inv_mesh_is_sound`), not by `MFTLCheckMeshIntegrity` or `MFTLObjectSetCounts`.
- **D3 `tris.verts` / `tris.neighbors` are redundant state** whose benefit (4× less traffic) is arithmetic, not
  measured. Re-examine when a consumer that isolates them exists.
- **D15 Remeshing renumbers global ids at the end of every pass, and has to**: redistribution and the ghost sync size
  their arrival tables from `next_*_gid`. So pool indices and global ids are both unstable across a remesh.
- **D19 `MFTLUpdateGhostPositions` has no global fallback.** A ghost whose owner is not a graph neighbour is skipped,
  counted (`g_n_orphaned`), reported on stderr, asserted in Debug and returned as `MFTL_ERROR`. It is safe only
  because both remeshing drivers end with a plain `MFTLRedistribute` back to the fine partition.
- **D20 Generation is rank-0 serial.** `MFTLGenerateSphere` / `MFTLGenerateSphereLocal` build every body whole on
  rank 0: `3440 * 4^refine + 128` bytes per body (140 B/triangle + 64 B/vertex, from the pool layouts in
  `src/object/object.c`). `parallel_advection_scaling` prints it and refuses above `--max-gen-mb`. The fix is a
  distributed generator or reader (A4 inherits the peak); batched generation is not (O(N²/K), `docs/analysis.md`
  §6.1). Bodies rank 0 keeps a sliver of hold full blocks; `MFTLRedistributeSetShrinkSlack` trims them (off by
  default: −8 % capacity for about 2 % of the phase). Regression test
  `test_generation_capacity_does_not_stay_resident_on_rank_0`. Measure in RSS, never in a byte counter.
- **D24 `n_bodies_global` counts retired ids**, and the `N_bodies >= nproc` guard in `dispersed_box_min`
  (`remesh_dispersed.c`) reads it as a body count. The fix, when it matters: a live count on `MFTL_Ensemble`, known
  exactly in `MFTLEnsembleReconstructBodies`.

**Physics, contact and coupling limits**
- **D16 Contact detection is discrete-time** (C6). `test_tunnelling_between_two_timesteps_is_a_known_miss` asserts
  the miss.
- **D17 A pair's contact envelope is `MAX(tol_A, tol_B)`**, so a µm droplet against a m body is judged at the larger
  scale (the only safe direction for the broad phase). Escape: `proximity_factor = 0` and the physical gap as
  `proximity_floor`. `test_two_length_scales_share_the_larger_envelope`.
- **D22 A topology change re-triangulates every body in its cluster, everywhere** (C4). Rayleigh–Taylor, np = 4:
  3.46 s in 80 topology calls against 2.11 s in 80 remeshing passes. Skipping the reconstructions that change no
  topology was measured WORSE: the erosion of sub-delta features is a load-bearing regulariser (`docs/analysis.md`
  §13.3).
- **D23 Detection queries the whole ensemble on every call**, most of which find nothing (0.16 of 0.43 s on the jet;
  C4). Levers, unmeasured: a per-primitive normal cache, or querying only bodies whose curvature or thickness moved.
- **D25 The remesher can fold a triangle against all its neighbours at a crease** (B6). It is why
  `test_travelling_waves_topology` runs at 96×24.
- **D26 The coupler is scoped to velocity and surface-load transfer** (efficiency items in B5). A broad SELECTED
  provider certificate can still over-request; no speedup of the refactored coupler has been measured.
- **D27 A point location's default ownership tiles** (up to 64 samples) bridge a hole thinner than a tile inside one
  registered cloud: it is certified empty rather than unprovided. `MFTL_ProviderRegions` make it exact.
- **D28 Bodies that already interpenetrate are reconstructed wrongly**, and nothing reports it (C4,
  `docs/analysis.md` §13.5).
- **D29 The volume correction is global**, a uniform offset against losses that arise locally (C4,
  `docs/analysis.md` §15.1 and §15.4). `collapse_preserves_volume` removes most of the loss at the source; the
  compensated-flip and flip-guard variants stay experiments and are not wanted in the library.

## 4. Integration with multiflow

multiflow's adapter is shaped by MFTL1:
- one pointer handle per entity, and per-entity accessors in hot loops;
- opaque triangle payloads;
- gather-to-root and send-all-triangles;
- an edge facade built on `twin_gid`.

Implementing that contract on `christian` would bring back everything §2 rejects. Instead, port multiflow one
subsystem at a time onto the native API, and keep the adapter only where one call maps onto one call:

| multiflow | `christian` |
|---|---|
| `peskininterpolation.c`, `sendnonlocal*.c`, `spreadfromfronttofluid.c` | the coupler: interpolate, spread, quadrature markers |
| `CellData` payload, surface scalar | registered fields; the conservative density |
| `computeforcesonfront.c`, Frenet surface tension, LB curvature | A1, plus the A3 loads |
| IRL indicator, `distance.c`, `SendLocalTrianglesToAll` | A2 |
| `frontremeshing.c`, TSUR, `volumerecovery.c` | `MFTLRemeshEnsemble`, `MFTLEnsembleRemoveUndulations`, `MFTLEnsembleCorrectVolumes` |
| `marchingcubes/` | `MFTLEnsembleApplyTopologyChanges` |
| MoveIBM transforms, mass properties, fictitious cells | A3; A2 step 3 |
| GTS input, `frontrestart.c` | A4, A5 |

Validate physically, not against golden files, on four ranks or more:
- the LeVeque shape error;
- the Lamb oscillation frequency;
- the rising-bubble terminal velocity;
- the MoveIBM settling velocity.

## 5. Rules every item follows

- New functions only. With a feature off, every digest is unchanged (interleaved A/B where an off path exists).
- Collective verdicts go through `agreed()`. Tests run at np 1, 2, 4, 8 and 27, and every distributed test has a
  non-vacuity count.
- The 2⁻²⁰ length-unit test passes bit for bit. No dimensional quantity is compared against an absolute constant.
- No per-rank state or message is sized by the number of bodies or ranks in the job. Per-body collectives go through
  `MFTLEnsembleReduceBodies` or `MFTLEnsembleExscanBodies`.
- A module's MPI lives in one `*_comm.c`, guarded by a confinement test.
- Each new option is one `KEY(...)` line in `src/settings/settings_keys.c`.
- Costs are recorded in `docs/analysis.md` as new sections, with stable numbering. Both manuals build with zero
  warnings.

## Appendix — what the removed plans became

| Old file | Status | Now in |
|---|---|---|
| 01 periodic extent | done, `85cec62` | `src/geometry/README.md`, `MFTLEnsembleCheckPeriodicExtents` |
| 02 rim curves | done, `317e767` | `src/remeshing/README.md` |
| 03 body integrals | done, `de6dd8d`; inside/outside → A2 | `src/geometry/README.md` |
| (04) optional fields | done, `82ff621` | `src/object/README.md` |
| 05 curvature and surface tension | open | A1 |
| 06 coupler improvements | open | B5 |
| 07 point queries, volume correction, sizing | D2 done and curvature sizing done, `de10103`; D1 → A2; sizing field → B1 | `src/correction/`, `src/remeshing/README.md` |
| 07 TSUR3D and volume correction | done, `de10103`; local correction and cross-rank edges → C4 | `src/correction/README.md` |
| 08 IBM bodies, I/O, Fortran | open | A3, A4, A5, C8 |
| 09 surface PDEs, membranes, contact lines | open | B4, C1–C3 |

Full texts: `git show d4f2b6a:plans/<file>`. Berend's plans 00–18: `git show 6188067:plans/<file>`.

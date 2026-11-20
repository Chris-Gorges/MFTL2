# Roadmap

What MFTL does not do yet, what is deliberately not planned, and the known limits of what it does. Status
2026-11-20. Item ids (A1, B3, D20, ...) are stable: source comments cite them as "roadmap item DN". A detailed plan
for an item, when one is written, is a separate `plans/NN-<item>.md` that cites its id and is deleted once the item
is done and its theory is in the module README.

Priorities:
- **A** blocks production use in a front-tracking or immersed-boundary solver;
- **B** is needed for coverage or scale;
- **C** waits for a consumer or a measurement;
- **D** is a known limit with no scheduled fix.

## A — Blocking production use

| Id | Item | Plan |
|---|---|---|
| A2 | **Distance and inside/outside at the solver's points.** Which cells lie inside a body, and the signed distance and closest surface point near the front, for any shape and without gathering the body. | Three steps: a band query (φ, body, closest triangle and barycentrics) from owned triangles only, bitwise np-independent; an exact sign everywhere (ray parity with exact predicates, summed along the process row); exact cut-cell volume moments. |
| A3 | **Rigid bodies.** Moving every rigid body every step; mass properties; the load per body. | An exact inertia tensor by the divergence theorem; motion as a pose (`x = c + R(q) x_ref` from a stored body-frame field), not composed increments, so it never drifts and ghosts stay bitwise equal to owners; per-body force and torque through the holder reduction. The rigid-body integrator stays with the solver. |
| A4 | **Mesh input.** STL, GTS and OBJ. | One pipeline: parse on rank 0, weld, drop degenerate facets, `MFTLBuildIndexedMesh`, reject non-manifold input on every rank, publish and redistribute. Inherits the rank-0 generation peak (D20). |
| A5 | **Checkpoint and restart.** | First the same rank count, bit-exact (each rank writes its pools in a documented, versioned raw layout with MPI-IO); then any rank count, through one redistribution. Gate: N steps equal N/2 + restart + N/2 by digest, with remeshing, fields and topology change active. |

## B — Coverage and scale

| Id | Item |
|---|---|
| B1 | **A caller-supplied sizing field for the remesher**, so markers can follow a locally refined fluid grid, or a per-body target length. A third `MFTL_RemeshSizing` mode that takes `verts.sizing` as given; the carrying already exists. |
| B2 | **Bodies entering and leaving mid-run**: collective removal by global id (the id is retired), and documented insertion into a running, redistributed ensemble. |
| B3 | **Proximity between bodies, and between a body and a wall**: the minimum gap with witness points and normal per body pair within an envelope, and penetration depth. Tested as a set against the unpartitioned answer. |
| B4 | **Surface PDE operators** (surfactant transport): P1 mass, cotangent stiffness, surface gradient and divergence, a ghost-to-owner accumulation, and a deterministic preconditioned CG over the ensemble. |
| B5 | **Coupler efficiency**: fewer discovery barriers, buffer and binding reuse, sparse peer lookup; then measure the synchronisation before changing the protocol. |
| B6 | **Folds at creases** (D25): a fold test against the edge neighbours' normals in flip, collapse and smoothing, restricted to rim-adjacent triangles. |
| B7 | **VTU output of registered fields** as point and cell data (owned entities only, as now). |

## C — When a consumer or a measurement asks for it

| Id | Item | Revive when |
|---|---|---|
| C1 | Contact angle and conormal at rim vertices. The surface-tension rim modes build the rim's conormal from a one-sided normal, which is first order (a cap's wall force 1.4 % off at 24 rings) | a contact-line model runs on an open surface |
| C2 | Surface viscosity (Boussinesq–Scriven); needs B4 | a case needs it |
| C3 | Membranes: hyperelastic in-plane forces, dihedral bending, area and volume penalties; a body with a reference shape is never remeshed | capsules or red blood cells are simulated |
| C4 | Local volume correction (D29), local topology reconstruction (D22), interpenetrating bodies (D28), detection cost (D23) | a production case measures the shape drift or the topology cost |
| C5 | A surface-adaptive decomposition (weighted rectilinear cuts through the `MFTL_OwnerMap` seam) | idle ranks are measured with a single large interface, where a uniform grid leaves most ranks without triangles |
| C6 | Continuous (swept) collision detection (D16) | a tunnelled contact is measured in a real run |
| C7 | An overlap-volume contact model | a volume-based contact model for triangulated particles is chosen |
| C8 | Fortran bindings | a Fortran consumer exists |
| C9 | Declared multi-body phase sets | user-declared phase compositions must be validated |

## Deliberately not planned

Capabilities that earlier front-tracking libraries offer and MFTL replaces or leaves out on purpose:

| Capability | Why not |
|---|---|
| Per-vertex periodic image counters, homological winding | The periodic extent rule (a body wraps an axis or is narrower than L/2, measured to 3L/4) covers every body a solver has; `MFTLEnsembleCheckPeriodicExtents` checks it. |
| Boundary tags, loops and boundary-curve objects | A rim is identified by its position; `MFTL_RimProjector` takes application constraints. |
| Opaque per-triangle byte payloads | Typed registered fields with transfer policies. |
| Pointer handles, an edge facade, per-entity accessors in hot loops | Implicit indexing makes edge identity `3t+k`; kernels work on arrays. |
| 2- and 3-ring neighbourhoods; topological ghost-layer growth | Curvature from a 1-ring position-plus-normal fit (`MFTL_CURVATURE_FIT`) with second-order fitted normals; edge forces by reverse accumulation (B4). |
| A process grid matched to the fluid decomposition | The coupler decouples the two decompositions; cuts matter only for load balance (C5). |
| Per-point ray casting with random rays and retries | Replaced by A2's exact predicates. |
| Gathering a body to one rank, or sending every triangle to every rank | Exactly the O(N) per rank the library exists to avoid. |
| A global absolute geometric tolerance | Every test is relative to the entity's own size. |

## D — Known limits

Limits with no scheduled fix, measured where possible. Some are rules: the obvious fix is wrong.

**Performance questions that need a cluster**
- **D1** Morton renumbering (`MFTLObjectReorderMorton`) has no measured benefit on Apple silicon. Kept for the single
  index space and the LBVH; measure on an x86 node with 1–2 MiB L2 per core before using it every timestep.
- **D4** The geometry update kernel is bound by the FP divide/sqrt unit. The only lever left, a hardware `rsqrt`
  estimate plus a Newton step, costs bit reproducibility and is unmeasured.
- **D9** The two VTU file layouts (a file per rank, one shared MPI-IO file) have been compared only on a laptop SSD.
  Open: thousands of ranks on a parallel filesystem.
- **D10** Cluster scaling covers only the advected, redistributed timestep ([`scaling/README.md`](../scaling/README.md#measured-results)):
  0.82 strong and 0.90 weak efficiency at 288 against 36 ranks. Unmeasured beyond a few ranks: remeshing (D11), the
  per-body reduction route against `comm_world` at large P, I/O (D9) and Morton (D1).
- **D11** The spanning remesh's cost at scale: whether migration cost stays proportional to the frozen band at
  hundreds of ranks, and which spanning mode is cheaper there.
- **D14** `src/bvh/` is not profiled in time beyond the predicate. Known cliff: one domain-spanning body collapses
  the broad-phase grid to a single bucket; a two-level grid removes it.
- **D18** Coarsening at hundreds of ranks: a coarse destination lies outside the 26-neighbourhood, so two exchanges
  take their O(nproc) global paths.
- **D21** Owner classification is about 30 % of the redistribution phase and cannot be skipped. A per-body slack
  against a caller-supplied motion bound would prove "nothing crossed a plane" without a scan; if built, it ships
  off, with a Debug verifier.

**Caches and fusions not built, because the reference loop redistributes every step**
- **D5** The ghost-exchange schedule of `MFTLUpdateGhostPositions` is rebuilt every step; a cached schedule pays
  only for a solver that redistributes less often.
- **D7** The geometry pass after a redistribution is bound by first touch; fusing it into the rebuild would put
  geometry into a module that transmits none.
- **D12** The BVH leaf–leaf narrow phase is batched but not vectorised.
- **D13** The spatial-halo overlap list is rebuilt every call. A cache belongs on `MFTL_SpatialHalo`; a halo that
  survives a redistribution is not wanted, because it is stale exactly when ownership changed.

**Structural and accounting limits**
- **D2** The half-edge triple invariant is checked in tests only, not by `MFTLCheckMeshIntegrity`.
- **D3** `tris.verts` / `tris.neighbors` are redundant state whose benefit (4× less traffic) is arithmetic, not
  measured.
- **D15** Remeshing renumbers global ids at the end of every pass, and has to; pool indices and global ids are both
  unstable across a remesh.
- **D19** `MFTLUpdateGhostPositions` has no global fallback: a ghost whose owner is not a graph neighbour is
  counted, reported and returned as `MFTL_ERROR`. Safe because both remeshing drivers end with a redistribution
  back to the fine partition.
- **D20** Generation is rank-0 serial: `3440 * 4^refine + 128` bytes per body on rank 0. The fix is a distributed
  generator or reader; batched generation is not (O(N²/K)).
- **D24** `n_bodies_global` counts retired ids, and one guard in the dispersed remesher reads it as a body count.

**Physics, contact and coupling limits**
- **D16** Contact detection is discrete-time (C6); `test_tunnelling_between_two_timesteps_is_a_known_miss` asserts
  the miss.
- **D17** A pair's contact envelope is `MAX(tol_A, tol_B)`, so a small droplet against a large body is judged at the
  larger scale. Escape: `proximity_factor = 0` with the physical gap as `proximity_floor`.
- **D22** A topology change re-triangulates every body in its cluster (C4). Skipping reconstructions that change no
  topology was measured worse: the re-sampling is a regulariser.
- **D23** Topology detection queries the whole ensemble on every call, most of which find nothing (C4).
- **D25** The remesher can fold a triangle against all its neighbours at a crease (B6).
- **D26** The coupler is scoped to velocity and surface-load transfer; no speedup of it has been measured (B5).
- **D27** A point set's default ownership tiles (up to 64 samples) can bridge a hole thinner than a tile;
  `MFTL_ProviderRegions` make it exact.
- **D28** Bodies that already interpenetrate are reconstructed wrongly by topology change, and nothing reports it
  (C4).
- **D29** The volume correction is global, a uniform offset against losses that arise locally (C4).
  `collapse_preserves_volume` removes most of the loss at the source.

D6 and D8 are retired.

## Rules every item follows

- New functions only. With a feature off, every digest is unchanged.
- Collective verdicts are agreed across ranks. Tests run at np 1, 2, 4, 8 and 27, and every distributed test has a
  non-vacuity count.
- The 2⁻²⁰ length-unit test passes bit for bit; no dimensional quantity is compared against an absolute constant.
- No per-rank state or message is sized by the number of bodies or ranks in the job. Per-body collectives go through
  `MFTLEnsembleReduceBodies` or `MFTLEnsembleExscanBodies`.
- A module's MPI lives in one `*_comm.c`, guarded by a confinement test.
- Each new option is one `KEY(...)` line in `src/settings/settings_keys.c`.
- Both manuals build with zero warnings.

# 08 — IBM bodies, mesh input, checkpoint/restart, Fortran bindings

Status: PLANNED (2026-10-05, branch `christian`)
Depends on: `03-body-integrals.md` (moment kernels, reference frames), `04-attached-fields.md` (checkpointed state), the [coupler's spreading interface](../src/coupling/README.md) (the IBM force path; the Fortran API wraps it)

Four milestones (E1 loads and rigid motion, E2 mesh reader, E3 checkpoint/restart, E4 Fortran). Each lands
alone.

## 1. Purpose

What separates "usable from a test program" from "usable in a production CFD run":

| | Need | Today |
|---|---|---|
| E1 | moving rigid IBM particles: hydrodynamic force and torque per body, inertia tensor, applying the rigid motion | none |
| E2 | importing a CAD or segmented geometry (STL, OBJ; GTS for multiflow) | "Absent: a mesh READER of any format" (CLAUDE.md); only spheres and level sets |
| E3 | restarting a run that took days, at the same or another rank count | VTU writer only; nothing reads |
| E4 | calling the library from Fortran, which most CFD codes are written in | none |

## 2. Findings that decide it

- **A per-body sum of marker forces is a holder reduction**, the same mechanism as the quality statistics.
  - Owned markers contribute once.
  - `MFTLEnsembleReduceBodies` folds the contributions in source-rank order, so every holder gets the same
    bits.
  - Torque needs a reference point identical on all holders: the body centroid from
    `MFTLEnsembleBodyIntegrals` (already reduced), or a solver-given centre.
- **The inertia tensor of a closed triangulated body is exact** by the divergence theorem (Mirtich 1996;
  Eberly 2002): per-triangle polynomial integrals of x_i x_j, i.e. the second moments.
  - `MFTLBodyFluxMoments` already integrates first moments with the nearest-image reference of `plans/01`;
    second moments are the same kernel family.
  - Open bodies get UNSUPPORTED, consistent with `plans/03`.
- **A rigid motion needs no ghost sync if every holder applies bit-identical arithmetic.**
  - `x' = c + t + R(q)(x − c)` with the minimum image about c, then wrap.
  - c, t and q are the same bits on every holder (they come from reduced state). Applied to owned AND ghost
    vertices, ghosts stay exact copies.
  - The same argument as the remesher's "every ghost is frozen on its owner".
- **A reader is a generator with a file as input.** The generation path already exists and is right:
  1. build the mesh on rank 0 (`MFTLGenerateSphereLocal` pattern);
  2. publish (`MFTLPublishGeneratedBodies`, two collectives for N bodies);
  3. `MFTLRedistribute` scatters it.

  It inherits the rank-0 generation peak (deferred issue 20). That is stated, not hidden.
- **Vertex welding needs no hash map.** STL repeats each vertex once per facet, with bit-identical floats. A
  stable LSD radix sort on the three 32-bit patterns (`MFTLSortPairs32`, three passes) groups them, and a
  linear scan assigns indices. OBJ and GTS are already indexed. Tolerance welding (snapping to a grid of
  `tol`) is an explicit option, never silent.
- **multiflow reads `.gts`** (`readGTSFiletoMFTLObject.c`, `IBMMFTLMesh sphere.gts`; `origin/main` plan 14). So
  GTS is in scope as the migration path, not as legacy.
- **A restart at the same np should be BIT-EXACT, and that is cheap.**
  - Write each rank's pools as they are: owned and ghost, pool order, flags, `partition_settled`/
    `partition_dirty`, id counters, the field registry and the field values.
  - Read them back, rank for rank.
  - "N steps" against "N/2 + restart + N/2" then compares by digest: the only test that proves nothing was
    forgotten.
- **A restart at a different np** cannot reuse the pool layout. It needs owned entities by global id, a
  rendezvous and then a redistribution. That is a different mechanism with a different cost.
- **Fortran must not mirror C structs.**
  - Opaque handles (`type(c_ptr)`) plus accessor functions survive every layout change. Field arrays come
    back as `c_ptr` + size → `c_f_pointer`, so they are zero-copy.
  - `MPI_Comm` crosses as `MPI_Fint` (`MPI_Comm_f2c`).
  - C `bool` crosses as `logical(c_bool)`.

## 3. Decision

- **E1.**
  - `MFTLEnsembleBodyLoads`.
  - `MFTLEnsembleBodyInertia`.
  - `MFTLEnsembleApplyRigidMotion`.
  - A pure-arithmetic Newton–Euler helper (no MPI): the solver integrates the rigid-body ODE however it likes.
- **E2.** `MFTLReadSurfaceMesh` for STL (binary and ASCII), OBJ and GTS, rank-0 read, through the generation
  path.
- **E3.**
  - Milestone 1: same-np restart, bit-exact.
  - Milestone 2: any-np restart by global-id rendezvous plus `MFTLRedistribute`.
  - MPI-IO raw binary with a documented layout. No HDF5 dependency.
- **E4.** A `mftl` Fortran module (ISO_C_BINDING) over the step, field, coupler and I/O API. Built only when
  CMake finds a Fortran compiler.

## 4. Design

### 4.1 E1 — loads, inertia, rigid motion (`include/mftl_ensemble.h`, `src/ensemble/ensemble_bodies.c` new, kernels in `src/geometry/body_integrals.c`)

```c
typedef struct MFTL_BodyLoads { double force[3], torque[3], ref[3]; int lowest_holder; } MFTL_BodyLoads;
/* Collective. Per owned marker (vertices or centroids of every local body, concatenated in slot order,
   or an attached field), summed over holders. ref_by_slot NULL = body centroid (MFTLEnsembleBodyIntegrals). */
int MFTLEnsembleBodyLoads(MFTL_Ensemble *ens, MFTL_MarkerKind kind, int force_field, const double *force, const double *ref_by_slot, MFTL_BodyLoads *out);

typedef struct MFTL_BodyInertia { double volume, centroid[3], inertia[6]; /* xx yy zz xy xz yz, about centroid, unit density */
                                  MFTL_IntegralStatus status; } MFTL_BodyInertia;
int MFTLEnsembleBodyInertia(MFTL_Ensemble *ens, MFTL_BodyInertia *out);      /* collective; 10 SUM slots */

typedef struct MFTL_RigidMotion { double centre[3], translation[3], quaternion[4]; } MFTL_RigidMotion;
/* Collective only for the Debug agreement check; arithmetic is local. All vertices, owned and ghost;
   clears MFTL_DERIVED_GEOMETRY; rotates verts.normal (no recompute needed). by_slot: [n_local_objects]. */
int MFTLEnsembleApplyRigidMotion(MFTL_Ensemble *ens, const MFTL_RigidMotion *by_slot);

/* Pure arithmetic, no MPI: one explicit Newton–Euler step (Strang-split quaternion update). */
void MFTLRigidBodyStep(double mass, const double inertia_body[6], const MFTL_BodyLoads *loads, double dt,
                       double x[3], double v[3], double q[4], double omega[3]);
```

- **Reading the loads.** For IBM direct forcing the marker force is minus the force applied to the fluid, so
  the hydrodynamic load is `−Σ F_i` over the body's markers.
  - The internal-fluid correction `ρ_f d/dt ∫_V u dV` (Uhlmann 2005; Kempe & Fröhlich 2012) is a volume
    integral over the solver's grid. That is solver-side, and the README gives the recipe using 07's
    indicator.
- **Periodic bodies.** The torque arm and the inertia use the nearest image about `ref` and the centroid
  respectively, under rule 01. A violation gives UNSUPPORTED, never a wrong number.
- **Order inside a step.**
  1. `ApplyRigidMotion`.
  2. Geometry.
  3. `MFTLRedistribute`, as with any other motion.

  The ghosts are already right, so no ghost sync is needed before the geometry pass. That is the one place
  this order differs from the documented loop, and the README says why.

### 4.2 E2 — reader (`include/mftl_io.h`, `src/io/mesh_reader.c` new)

```c
typedef enum { MFTL_MESH_STL = 0, MFTL_MESH_OBJ = 1, MFTL_MESH_GTS = 2, MFTL_MESH_AUTO = 3 } MFTL_MeshFormat;
typedef struct MFTL_MeshReadOptions {
  MFTL_MeshFormat format;
  double scale, translate[3], rotate_quaternion[4];   /* applied at read                                   */
  double weld_tolerance;                              /* 0 = bitwise welding only                          */
  bool canonicalise_outward;                          /* flip if the signed volume is negative (closed only) */
} MFTL_MeshReadOptions;
/* Collective. Rank 0 reads and builds; the object is published and must then go through MFTLRedistribute.
   Returns the new body's global id (the handle that survives), MFTL_ERROR on every rank on failure. */
int MFTLReadSurfaceMesh(MFTL_Ensemble *ens, const char *path, const MFTL_MeshReadOptions *opt);
```

**Pipeline on rank 0:**
1. Parse.
2. Weld: bitwise radix sort, or a snapped key when a tolerance is given.
3. Drop degenerate facets, counted.
4. `MFTLBuildIndexedMesh`.
5. Validate: non-manifold edges and inconsistent winding make it fail with a message naming the first
   offending facet. Then `MFTLMarkMeshBoundary`.
6. Optional outward canonicalisation by signed volume.

`MFTLPublishGeneratedBodies` then publishes the body. Every failure is broadcast, so all ranks return
`MFTL_ERROR` together: no rank is left waiting in the next collective.

### 4.3 E3 — checkpoint and restart (`include/mftl_io.h`, `src/io/checkpoint.c` new)

```c
int MFTLCheckpointWrite(MFTL_Ensemble *ens, const char *path);                       /* collective */
MFTL_Ensemble *MFTLCheckpointRead(MPI_Comm comm, const char *path, int *status);      /* collective */
```

**File layout**, little-endian, versioned, documented in `src/io/README.md`:
1. A header: magic, version, domain box, periodicity, the `cuts` written with, `n_bodies_global`, the
   redistribution switches, the field registry (name, entity, ncomp, kind) and the resolved settings text
   for provenance.
2. A rank table: byte offsets per writing rank, from one `MPI_Exscan` over byte counts at checkpoint time.
   That is not per step.
3. Per rank, per local body:
   - the object header (gid, id counters, global counts, flags);
   - `verts`: rec, pos and fields;
   - `tris`: rec, he and fields.

   Derived arrays are rebuilt, never written.

**M1, same np.**
- Each rank reads its own block with `MPI_File_read_at_all`.
- The pools are restored slot for slot, including ghosts and the settled/dirty bits.
- `MFTLObjectBuildDerived`.
- The continuation is bit-identical.

**M2, different np.**
- Each new rank reads a contiguous share of the old blocks, keeping OWNED entities only.
- Entities are sent to a rendezvous rank by `global_id mod P'` (the gid-hashed directory). There, each
  triangle is joined to its three vertices by sorting on vertex gid.
- Each triangle then goes to the owner of its centroid, which receives its vertices with it.
- Twins are rebuilt with `MFTLRebuildTwins`.
- One `MFTLRedistribute` builds the halo.
- Cost: O(entities / P') per rank, with three all-to-all rounds over NBX-discovered partners (the `06` NBX
  helper, moved to `src/communication/` when its second user appears).

### 4.4 E4 — Fortran (`fortran/mftl.F90`, `src/fortran/mftl_fortran.c` new, `examples/fortran_coupled_step.F90`)

- **C side.** Thin `MFTL*F` wrappers only where a C signature cannot be called directly:
  - `MPI_Comm` ↔ `MPI_Fint`;
  - strings (null-termination);
  - a field returned as (pointer, length).

  They are exported as `MFTL*`, so the symbol rule holds.
- **Fortran side.**
  - `module mftl` with `bind(C)` interfaces.
  - Handle types `type(mftl_ensemble)` and `type(mftl_coupler)` wrapping `c_ptr`.
  - Convenience routines `mftl_field_pointer(ens, slot, id, array)`, which apply `c_f_pointer` with the
    right shape.
- **Coverage.**
  - Ensemble and settings.
  - Generation and reading.
  - The step functions: update, normals, redistribute, remesh, topology.
  - Fields, curvature and surface tension.
  - The coupler.
  - Loads and rigid motion.
  - VTU, checkpoint.

  The local hot-loop accessors are NOT wrapped: per-entity calls across languages are the anti-pattern.
  Fortran gets whole arrays.
- **CMake.** `include(CheckLanguage)` / `check_language(Fortran)`. When found: build the module and the
  example, and add one CTest that runs the Fortran example and compares its printed digest with the C
  twin's.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_bodies.c` (new) | E1 collectives |
| `include/mftl_geometry.h`, `src/geometry/body_integrals.c` | second-moment kernel, rigid-motion kernel, `MFTLRigidBodyStep` |
| `include/mftl_io.h`, `src/io/mesh_reader.c`, `src/io/checkpoint.c` (new) | E2, E3 |
| `include/mftl_types.h` | the version constant for the checkpoint format |
| `fortran/mftl.F90`, `src/fortran/mftl_fortran.c`, `examples/fortran_coupled_step.F90` (new), `CMakeLists.txt` | E4 |
| `tests/test_body_loads.c`, `tests/test_mesh_reader.c`, `tests/test_checkpoint.c` (new), `tests/data/` (a small STL, OBJ and GTS) | §6 |
| `src/io/README.md`, `src/ensemble/README.md`, `CLAUDE.md` | the "Absent" list loses "a mesh READER"; conventions for rigid motion and restart |

## 6. Tests

- **E1.**
  - Uniform traction on a sphere gives the exact net force. A rotational force field gives the analytic
    torque.
  - Inertia of a sphere, an ellipsoid and a box mesh against analytic values: exact for the box,
    second-order convergent for the curved bodies, and correct on a periodic seam.
  - Rigid motion: ghost positions equal owner positions bitwise; `MFTLCheckMeshIntegrity` passes; volume is
    invariant to 1e-14; n rotations by 2π/n return to the start within n ulps × R.
  - np 1, 2, 4, 8, 27, `agreed()`.
  - Non-vacuity: bodies with ≥ 2 holders, counted, must be > 0.
- **E2.**
  - Round trip: an icosphere written as STL, OBJ and GTS reads back with the same area, volume, Euler
    characteristic and vertex count.
  - Duplicate vertices are welded; a non-manifold input and a flipped facet are rejected on every rank.
  - The scatter at np 1 … 8 passes `inv_partition_is_sound`.
- **E3.**
  - M1: N steps against N/2 + checkpoint + read + N/2 gives identical digests (mesh and partition) with
    fields, remeshing and topology change all active, at np 2 and 4.
  - M2: writing at np 4 and reading at np 2, 3 and 8 gives an identical census and identical field values by
    gid.
- **E4.** The Fortran example's digest equals the C twin's.

## 7. Gate

- New functions only: digests unchanged.
- Measured and recorded:
  - read time and rank-0 peak memory for a 10⁶-facet STL;
  - checkpoint write and read bandwidth (laptop numbers labelled as such, as `docs/analysis.md` §3 does).
- Zero warnings in both manuals.

## 8. Considered and deferred

- **Parallel STL read** (fixed 50-byte records, so a slab read is trivial) with distributed welding by a
  distributed sort. It removes the rank-0 peak. Revive when a CAD body's facet count makes rank 0 the limit.
  It is the same fix deferred issue 20 wants for generation.
- **HDF5 / ADIOS2 checkpoints:** a dependency for features (compression, introspection) nobody has asked
  for. The raw layout is documented and versioned.
- **Collision response** (soft-sphere DEM, lubrication): detection exists (`MFTLEnsembleIntersectObjects`). The
  force model is physics the solver owns.
- **Deformable-body restart from a different MFTL version:** the format version refuses; no migration tool
  until a second version exists.

# 04 — Attached fields: per-vertex and per-triangle state that survives everything

Status: PLANNED (2026-10-05, branch `christian`)
Depends on: —
Used by: `05` (σ field, curvature/force outputs), `06` (marker state), `07` (sizing field), `08` (checkpoint), `09` (surface PDE unknowns, membrane reference shape)

## 1. Purpose

A solver keeps **state on the front**: surfactant concentration Γ, temperature, marker velocity of the previous
stage, a membrane's reference shape, a per-vertex sizing length. Today that state has nowhere to live:
- pools hold only `rec/pos/normal` and `rec/he/area/normal/center/verts/neighbors` (`mftl_types.h:132-182`);
- the wire records `pack_vertex` / `pack_triangle` (`redistribute.c:444-500`) carry identity and position only;
- pool indices AND global ids are unstable across a remesh (deferred issue 15), and an `MFTL_Object*` is invalid
  after a redistribution. A caller-side array keyed by either goes stale silently.

This is the gap every later physics plan stands on.

## 2. Findings that decide it

- **Two ownership questions are already answered and must not be re-asked.** The field travels with its entity. So:
  - it is owned where the entity is owned (prefix);
  - it is ghosted where the entity is ghosted;
  - it is reordered where the entity is reordered.

  Every mechanism this needs already exists for `pos`. A parallel caller-side array would need its own sync,
  staleness flag and index mapping — exactly the "second index space" trap of invariant 3.
- **The single-allocation rule extends to fields at zero cost.** `MFTLLayoutAdd` cuts any number of
  sub-arrays out of one block (`vertex_layout_compute` / `triangle_layout_compute`, `object.c:48,160`).
  - A field is one more sub-array, so growth, `MFTLObjectShrinkToFit`, `MFTLObjectReleasePools` and the
    "last addressable element" ASan test cover it automatically.
- **Redistribution has a free payload slot.** It sends three payloads through `MFTLExchangeStart`
  (`redistribute.c:1956`); `MFTL_MAX_EXCHANGE_PAYLOADS` is 4.
  - Field values go as the fourth payload, in record order, so no index rides on the wire.
  - With no fields registered the call stays at three payloads, which makes it bit-identical to today.
- **A field changes message lengths, so it is COLLECTIVE STATE** (CLAUDE.md, the skip-switch rule).
  - Two ranks with different registries deadlock or corrupt with no diagnostic.
  - So the registry joins the Debug checksum in `check_object_lists_agree` (`redistribute.c:1707`).
- **Remeshing operators are the only place a field must be *invented*,** and what to invent depends on the
  physics, which the library cannot know. Hence a `kind`, not a callback:
  - a callback would run inside the operator loops;
  - it could not be checksummed;
  - it would make the distributed remesher's determinism depend on user code.
- **Topology change rebuilds the surface from a lattice and has no "old entity" to copy from.** But its merge
  `MFTLSdfMerge` (smaller |φ|, tie towards negative) is commutative and associative, and the field can ride
  with the winning φ.
  - The existing distributed merge in `topo_comm.c` (`alltoallv_doubles(…, stride, …)`) carries it with a
    larger stride.
  - So the transferred field is np-independent for the same reason the surface is.

## 3. Decision

- Typed double fields, registered collectively on the ensemble, before the first body is added.
- They are stored as SoA sub-arrays of the pool block.
- They are transported by redistribution, Morton reorder, every remeshing operator, compaction and topology
  change, with semantics chosen by a `kind`.
- Add one collective ghost exchange for any subset of fields.
- **No opaque byte blob** (main's Stage 12): it cannot be interpolated on a split, and integers up to 2⁵³ fit
  exactly in a `COPY` double.

## 4. Design

### 4.1 API (`include/mftl_object.h` for the accessor, `include/mftl_ensemble.h` for the registry, `include/mftl_communication.h` for the exchange)

```c
typedef enum { MFTL_FIELD_VERTEX = 0, MFTL_FIELD_TRIANGLE = 1 } MFTL_FieldEntity;

typedef enum {
  MFTL_FIELD_INTENSIVE = 0,  /* a point value: velocity, temperature, sizing length          */
  MFTL_FIELD_EXTENSIVE = 1,  /* a DENSITY whose integral over the body is conserved: Γ, mass */
  MFTL_FIELD_COPY      = 2   /* a label: reference data, integer ids; children inherit       */
} MFTL_FieldKind;

#define MFTL_MAX_FIELDS 32          /* a field mask is one uint32_t                         */
#define MFTL_MAX_FIELD_DOUBLES 64   /* summed ncomp per entity class; bounds the wire        */

/* Collective. Before the first MFTLEnsembleAddObject(); MFTL_ERROR afterwards.
   Returns the field id (dense, in registration order), identical on every rank. */
int MFTLEnsembleRegisterField(MFTL_Ensemble *ens, const char *name, MFTL_FieldEntity entity, int ncomp, MFTL_FieldKind kind);
int MFTLEnsembleFindField(const MFTL_Ensemble *ens, const char *name);   /* -1 if absent */

/* [ncomp * capacity] doubles, component-interleaved per entity like verts.pos.
   Valid until the next call that may resize or rebuild the pool (same rule as verts.pos). */
static inline double *MFTLObjectField(MFTL_Object *obj, int field_id);
static inline int MFTLObjectFieldComponents(const MFTL_Object *obj, int field_id);

/* Collective: ghost copies of the fields in `mask` := their owners' values. */
int MFTLUpdateGhostFields(MFTL_Ensemble *ens, uint32_t mask);
```

Registry storage:
- `MFTL_FieldRegistry` lives on `MFTL_Ensemble`: name, entity, ncomp, kind, and offset per entity class.
- Each object holds a borrowed `const MFTL_FieldRegistry *fields`, set by `MFTLEnsembleAddObject` exactly
  like `domain`, and by every place that creates an object on arrival (redistribution, topology change).
- An object created before `AddObject` (the generators) is re-laid by one `object.c` call at `AddObject`
  time. That stays inside the only file allowed to write a pool pointer.
- New entities start at 0.0, except where §4.3 says otherwise.

### 4.2 Transport (bit-exact except where §4.3 interpolates)

| Event | Where | What happens to fields |
|---|---|---|
| growth / shrink / release | `object.c` layout | sub-arrays of the same block; one `capacity` |
| Morton reorder | `MFTLObjectReorderMorton` | permuted with the entity |
| redistribution | `redistribute.c` pack / unpack | 4th payload `double[ncomp_total]` per moving entity, record order; ghosts re-received carry the owner's values |
| skipped body | `MFTLRedistributeSetSkipUnchangedBodies` | untouched (the pool is untouched) |
| compaction | `MFTLRemeshCompactObject` | moved with the entity |
| remesh migration | `MFTLRedistributeWithOwners` | same as redistribution |
| ghost refresh | `MFTLUpdateGhostFields` | owner value copied; reuses the request/reply schedule of `ghost_sync.c` (`ghost_sync_vec3`, `ghost_field_read/write` generalised from 3 to `ncomp`) |
| spatial-halo shadow objects | `spatial_halo.c` | none; shadow objects carry no fields (contact needs none) |

### 4.3 Remeshing semantics (`remesh_ops.c`: `MFTLSplitEdge` :228, `MFTLCollapseEdge` :663, `MFTLFlipEdge` :895; `remesh_smooth.c`)

Notation: edge (a, b), new vertex m at parameter s along it.
- s is ½ for a chord split. For a rim-curve split (plan 02) it is the projection of the placed point onto the
  chord.
- A_T is the triangle area.
- For EXTENSIVE fields, an "amount" is density × area: dual area for a vertex, A_T for a triangle.

| Operator | Vertex INTENSIVE | Vertex EXTENSIVE | Triangle INTENSIVE | Triangle EXTENSIVE | COPY |
|---|---|---|---|---|---|
| split | `(1−s) f_a + s f_b` | same as intensive, then renormalised (§4.5) | children inherit parent | child density = parent amount / Σ child areas (**exact local conservation**) | from the endpoint / parent with the lower global id |
| collapse a→b at point p | linear interpolation of f_a, f_b at p | same, then renormalised | fan keeps its values | removed triangles' amounts distributed over the survivor's fan ∝ new area (**exact**) | survivor's value |
| flip | unchanged | unchanged | area-weighted mean of the pair | pair amount split ∝ new areas (**exact**) | lower global id of the pair |
| tangential smoothing by δ | `f += ∇_s f · δ` (P1 gradient over the complete 1-ring, from pre-sweep values) | same, then renormalised | unchanged | density × A_old / A_new (**exact**) | unchanged |

Why it is safe in parallel:
- The freeze discipline gives every operator an owned, unfrozen, complete neighbourhood (CLAUDE.md,
  `MFTL_FLAG_FROZEN`). So every rule above reads only entities the acting rank owns or holds exactly.
- The "lower global id" tie-breaks make a COPY outcome partition-independent.

The guidance this implies, recorded in the README: **store conserved quantities on TRIANGLES.**
- Triangle EXTENSIVE fields are conserved exactly and locally by every operator.
- Vertex EXTENSIVE fields are conserved only up to the per-body renormalisation.

### 4.4 Topology change (`signed_distance.c`, `topo_reconstruct.c`, `topo_comm.c`)

1. **Rasterise.** `MFTL_SdfGrid` gains an optional payload: `n_payload` doubles per node.
   - When `MFTLRasterizeSignedDistance` writes a candidate φ, it also writes the field at the closest point:
     - a vertex field is P1-interpolated with the barycentrics `MFTLClosestPointOnTriangle` already
       computes;
     - a triangle field takes the triangle's value.
   - The merge compares φ only and copies the payload of the winner, so ties are resolved exactly as now.
2. **Exchange.** The existing `alltoallv_doubles` call carries `1 + n_payload` doubles per node (the `stride`
   argument).
3. **Extract.** A marching-cubes vertex on lattice edge (n0, n1) at parameter t gets `(1−t) f0 + t f1`, as
   its position does. A new triangle field is the mean of its corners' transferred values.
4. **Conserve.** EXTENSIVE fields are renormalised per CLUSTER: Σ amount in = Σ amount out. This is a
   holder reduction folded into the reduction the volume census already takes per cluster.

### 4.5 Renormalisation of EXTENSIVE fields

- One holder reduction (`MFTLEnsembleReduceBodies`, SUM) before and after a remesh pass gives a per-body
  factor `M_before / M_after`, applied to owned and ghost values alike. The factor is bit-identical on every
  holder, so ghosts stay exact.
- Fused into one reduction of `2 × n_extensive` slots.
- Skipped entirely, with no reduction, when no EXTENSIVE vertex field is registered. Triangle fields are
  already exact.

### 4.6 Output

`MFTL_VtkOptions` gains `write_fields` (default on): registered fields of OWNED entities go out as
PointData/CellData. The owned-only rule of `MFTLWriteVtu` applies unchanged.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_types.h` | `MFTL_FieldRegistry`; `fields` pointer on `MFTL_Object`; registry on `MFTL_Ensemble` |
| `include/mftl_object.h`, `src/object/object.c` | layout with field sub-arrays, accessor, re-layout at `AddObject`, reorder permutes fields |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble.c` | `MFTLEnsembleRegisterField`, `MFTLEnsembleFindField`, propagate the pointer |
| `src/communication/redistribute.c`, `comm_internal.h` | 4th payload; checksum includes the registry |
| `include/mftl_communication.h`, `src/communication/ghost_sync.c` | `MFTLUpdateGhostFields` over the generalised `ghost_field` path |
| `src/remeshing/remesh_ops.c`, `remesh_smooth.c`, `remesh_compact.c`, `remesh_driver.c` | §4.3 rules; §4.5 renormalisation call |
| `src/geometry/signed_distance.c`, `include/mftl_geometry.h` | SDF payload |
| `src/topology_change/topo_reconstruct.c`, `topo_comm.c`, `topo_driver.c` | §4.4 |
| `src/io/vtu_writer.c`, `include/mftl_io.h` | `write_fields` |
| `tests/test_fields.c` (new), `tests/CMakeLists.txt` | §6 |
| `src/object/README.md`, `src/remeshing/README.md` | the semantics table and the "conserved on triangles" guidance |
| `CLAUDE.md` | one bullet under "What exists today"; the registry as collective state in Conventions |

## 6. Tests

- **Layout.** After every growth, ASan writes the last addressable element of every field array (the
  `test_object.c` pattern).
- **Reorder / redistribute.**
  - Set `f = g(global_id)`. After Morton reorder and after twenty redistributions of moving bodies,
    `f == g(gid)` holds bitwise at np 1, 2, 4, 8, 27.
  - Every ghost equals its owner bitwise, before and after `MFTLUpdateGhostFields`.
- **Skip path.** With fields registered, `test_skipping_reproduces_the_full_rebuild` passes; its digest is
  extended over the field arrays.
- **Remesh.**
  - A linear INTENSIVE vertex field `f = a·x + b` is exact to rounding after split and after midpoint
    collapse.
  - Triangle EXTENSIVE total conserved to 1e-13 relative through serial, spanning and dispersed remeshing at
    np 1, 2, 4, 8.
  - Vertex EXTENSIVE total conserved to 1e-13 with renormalisation.
  - Smoothing with the gradient update gives a lower field error than without (measured, recorded).
- **Topology change.**
  - A constant field stays constant exactly.
  - A linear field has error O(dx²).
  - The cluster total of an EXTENSIVE field is conserved to 1e-12.
  - Census plus quantised field values equal the rank-0 `MPI_COMM_SELF` rerun (the `test_topology_change.c`
    pattern).
- **Collective consistency.** A Debug build with deliberately different registries on two ranks reports
  instead of hanging.
- **Non-vacuity.** Counted: field doubles sent by redistribution and by `MFTLUpdateGhostFields` must be > 0
  at np ≥ 2.

## 7. Gate

- **No field registered.**
  - Every digest is bit-identical: `leveque_remeshing_parallel -digest`,
    `parallel_advection_redistribute_list --digest`, the open-surface and topology examples.
  - Message counts and byte counts in `MFTLRedistributeGetStats` are identical.
- **Interleaved A/B of the redistribution phase.**
  - With no fields: neutral.
  - With one 3-component vertex field: the cost is measured and recorded. The expected order is the 24 B per
    MOVING vertex on the wire.
- Both manuals build with zero warnings.

## 8. Considered and deferred

- **Opaque per-triangle blob** (main Stage 12): no interpolation semantics. `COPY` covers its uses.
- **Exact local conservation for vertex fields** (supermesh / overlap remap): the right tool for a
  high-accuracy surfactant solver. Revive when the renormalisation's global character measurably biases a
  result.
- **Fields on edges or half-edges**: no consumer.
- **Registering after bodies exist**: possible (re-lay every local object), but a mid-run registration is not
  a known use. Revive with a consumer.

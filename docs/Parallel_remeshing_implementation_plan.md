# Parallel Remeshing for MFTL: Complete Implementation Specification

---

## 0. AS-BUILT NOTE (2026-08-17) — read before implementing anything below

**Both strategies are implemented.** The spanning / shift-freeze path (§5) is
`src/remeshing/remesh_parallel.c`; the dispersed / object-centering path (§4) is
`src/remeshing/remesh_dispersed.c`; the dispatch (§2, §3) is `MFTLRemeshChooseStrategy`, resolved from a
measured body radius. Tested at np = 1, 2, 4, 8 by `tests/test_remeshing_parallel.c` and
`tests/test_remeshing_dispersed.c`.

### §12 checklist, audited

| Item | Status |
|---|---|
| `MFTL_EnsembleType` + promote/demote radii on `MFTL_Ensemble` | **Deviated.** The strategy is `MFTL_RemeshParameters.parallel_strategy`, resolved per call by `MFTLRemeshChooseStrategy`. Putting it on the ensemble would freeze a decision that the bodies invalidate by growing; re-measuring every call costs two Allreduces and removes the hysteresis machinery of §6 entirely. |
| `decomp_shift[3]` on `MFTL_Domain`, inside `MFTLGetOwnerProc` | **Reverted** — see above. `MFTLRemeshOwnerProc` + `MFTL_OwnerMap`. |
| `MFTLDomainSetShiftVec` with the 8-phase table | Done, as `MFTLRemeshPhaseShift`. |
| `MFTL_FLAG_FROZEN` | Done. `FROZEN_NEXT` and `NEVER_INTERIOR` are not needed — see rows 5.4 and 7.4 below. |
| Freeze marking during ghost construction | **Deviated** — derived locally and exactly. Row 5.3 below. |
| `MFTLDilateFrozenBand` | Done, defaulted off. Row 5.4. |
| Gate split / collapse / flip / smoothing | Done, and the gate runs BEFORE the approval. Row 7.4. |
| 64-bit rank-encoded GIDs | **Rejected with cause.** Row 5.6. |
| `MFTL_WorkList` | **Not implemented.** Measured instead: on the LeVeque case the remesh is 96–99 % of the timestep, but the phase count (i.e. migrations) is what varies with the budget and the rank count, not the sweep cost. A work list shortens each sweep; nothing measured says the sweeps are the problem. Revisit with a profile that isolates them. |
| Replace `calloc(nproc, …)` in redistribution | **Not done, deliberately.** Pre-existing property of the production redistribution; rewriting a 1500-line function's indexing to fix a memory constant is the change most likely to break what this work was asked to protect. Belongs on redistribution's own schedule with a measurement at P in the thousands. |
| `MFTLEnsembleCenterObjects` | Done, as `MFTLRemeshMigrateToRanks`. |
| `MFTLEnsembleRebalance` with hysteresis | **Not implemented; superseded.** It exists to stop one large body forcing the expensive strategy on many small ones, by keeping two ensembles. Nothing else in MFTL_6 supports a two-ensemble pattern (one spatial halo, one BVH, one object list on the wire), and AUTO already re-measures every call so there is no state to thrash. What a caller genuinely needs to do this themselves is the per-body measurement, and that is now public: `MFTLRemeshMeasureBodies`. The remaining gap is a MIXED ensemble, which resolves to SPANNING — correct, not optimal. |
| `MFTLComputeCoarseningFactor` + coarsening path | **Formula done** as `MFTLRemeshCoarseningFactor`, and the driver now prints it in the `h <= 4*delta` warning so the diagnostic §9 asks for is actionable. The sub-communicator path is NOT implemented. |
| Assertions of §9 | Done as warnings plus `MFTL_RemeshStats.converged`; row 9 below. Two NOT in the plan were added after they bit: a refusal on a degenerate length band, and a Debug check that all ranks were handed the same band. |
| Compaction before redistribution | Done. |
| Geometry (incl. centres) before migration | Done, over the FULL triangle count, with a Debug check in `MFTLRemeshMigrateShifted`. |
| `MFTLDetectFeatures` every pass | Done — twice per iteration, see `MFTLRemeshObject`. |
| Periodic / non-periodic handling for shifted planes | Done, and it is the one place a plain `(int)` cast is wrong; `test_a_sphere_on_the_periodic_seam_remeshes_everywhere`. |
| Unit test: Euler characteristic **and edge-length histogram** vs serial | Done. Euler was already there; the histogram is `test_the_edge_length_distribution_matches_the_serial_answer`. Measured total variation against a CONVERGED serial reference: 0.0000 at np = 1, then 0.028, 0.027, 0.044 at np = 2, 4, 8. |

**Still NOT implemented, deliberately:** §4.4 Regime C sub-communicator coarsening (a second communication
subsystem, judged the riskiest part for the least certain benefit), §6 ensemble promotion/demotion
(superseded — see the table), and §7 work lists (no measurement says the sweeps are the bottleneck).

### One thing the plan does not mention and should have

The length band has to be the SAME on every rank, and nothing in the algorithm communicates it. The natural
way to derive a target — `MFTLMeanEdgeLength(obj)` — is LOCAL, and the generators are COLLECTIVE and
already distribute, so a rank owning no part of the body measures the empty set and gets 0.0. Its band is
then `[0, 0]`, every edge it is ever sent is too long, and it refines without bound. Measured while writing
`examples/leveque_remeshing_parallel.c`: 1280 triangles became 148 287 on the rank that started empty, in
eight phases of one timestep, still climbing when the run was killed. No assertion fired and no invariant
broke — every one of those triangles was valid.

Closed three ways: `MFTLEnsembleMeanEdgeLength` (the collective form), a refusal on a degenerate band in
both drivers, and a Debug `MFTLRemeshParamsAgree` check. Guarded by
`test_the_edge_length_band_is_a_collective_measurement`.

### The one structural decision this plan does not contain

**All the remeshing module's MPI is in `src/remeshing/remesh_comm.c`, and nothing outside it is allowed to
call MPI.** Checked by `test_remeshing_mpi_is_confined`, a CMake source scan, because no runtime test can
see the property.

The first implementation did the opposite, following this plan's checklist (§12) literally: the shifted
decomposition was expressed by teaching `MFTLGetOwnerProc` a per-axis offset, and the id exchange by adding
a third payload to the per-timestep ghost sync. Both were correct and tested. Both put work used a few
times per hundred timesteps inside loops that run every timestep — `MFTLGetOwnerProc` **measured at 1.70×**
(1.79 → 3.05 ns/entity; 42 → 89 instructions), and it is the body of `compute_owners`.

Both are reverted. The replacement is ONE seam: `MFTL_OwnerMap` in `mftl_communication.h`, a per-body pair
of destination arrays that redistribution uses instead of the geometric rule. MFTL_NULL on the production
path, one predicted branch per body, hoisted out of both loops. It is also exactly the hook §4 needs —
this plan's earlier note called it "an owner-override hook in `compute_owners`" — and it had to be an
ownership ARRAY rather than a smarter ownership function, because the dispersed rule is not a plane test at
all.

### Where the plan is wrong for MFTL_6

Nine things, wrong because MFTL_6 is a rewrite of MFTL_5 rather than a refactor, so the hooks assumed here
do not exist:

| § | The plan says | What was built, and why |
|---|---|---|
| 5.2 | Put `decomp_shift[3]` on `MFTL_Domain` and subtract it inside `MFTLGetOwnerProc` | **Reverted.** The field is gone from `MFTL_Domain` and the shift is a local of the driver, passed to `MFTLRemeshOwnerProc` (`src/remeshing/remesh_comm.h`) and delivered to redistribution as an `MFTL_OwnerMap`. See above for the measurement. |
| 5.3 | Set `MFTL_FLAG_FROZEN` inside `MFTLPackSingleTriangle` / `MFTLEnsembleCreateSpatialGhosts` | Those do not exist. The halo falls out of `collect_destinations` sending each owned triangle to the owners of its corners *and* edge neighbours — which makes the classification a **purely local** pass: `corner >= verts.n_local \|\| neighbour >= tris.n_local`. No hook, no communication, and EXACT rather than a k-ring approximation. `MFTLRemeshMarkFrozen`. |
| 5.4 | Dilate the band by 1 ring for safety | **Not needed, and defaulted off.** An unfrozen vertex has every incident triangle owned and unreplicated, and the halo is vertex-complete, so every fan an approval predicate walks is already complete. Kept as `freeze_dilation_rings` so the claim can be tested against a conservative alternative. |
| 5.6 | 64-bit rank-encoded GIDs; remove all `MPI_Exscan` | **Would break redistribution.** `redistribute.c` and `ghost_sync.c` size their arrival tables from `next_*_gid`, i.e. they require `inv_global_ids_are_dense`; rank-encoded ids are sparse by construction. Built instead: 32-bit, one `MPI_Exscan` per phase for uniqueness, one re-densification at the end plus `MFTLRemeshSyncGhostIds`. In the DISPERSED path even that is unnecessary — one rank owns a whole body, so its own counter is already globally unique for it, and the only agreement needed is one `Allreduce(MAX)` of the counts. |
| 5.8 | A per-phase ghost position sync is implied | **Removed.** Every ghost in a rank's pool is provably frozen *on its owner*, so the copies are still exact when the sweeps finish. One collective per phase saved. |
| 5.8 | 8 phases, converge after 2 clean | Default budget **32**; converge on `ops == 0 && hidden == 0`, else on a full table of silent placements. Measured: 4 phases at np = 1, 6 at np = 2, 19 at np = 4, 17 at np = 8 for a sphere going 1280 → 3830 triangles. One sweep of the table is not enough. |
| 7.4 | `NEVER_INTERIOR` certificate | Replaced by `count_hidden_edges` — out-of-band edges with a frozen endpoint. Less state, and it needs no fan walk, which matters: the first attempt counted "approved then gate-refused" and reported 3 permanently blocked operations on a fully converged np = 4 mesh, because a rank's one-ring around a *ghost* vertex is incomplete. |
| 9 | `MFTL_ASSERT` on `h_min > 4*delta` | Warns once on rank 0 instead. The scheme degrades rather than breaks, `MFTL_RemeshStats.converged` reports it, and `freeze_dilation_rings` is a supported knob that widens delta — a parameter the API invites you to try must not abort the process. |
| 4.2 | `MFTLObjectIsFullyLocal` after centering is the safety check | **Vacuous, and quietly so.** An owner map names one destination for every entity of a body, so redistribution puts the body wholly on that rank *whatever its size* — a domain-spanning jet would arrive complete on one rank, report itself perfectly local, and remesh correctly, having serialised the largest object in the simulation onto one rank's memory. The admission test is therefore on the measured RADIUS and is taken BEFORE the migration, so the fallback to the spanning driver never inherits a half-migrated ensemble. |

### Also required and not anticipated here

`remesh_new_vertex` / `remesh_new_triangle` broke the ownership prefix (they appended past the ghosts *and*
incremented `n_local`), and `MFTLRemeshCompactObject` had to grow an ownership partition — a swap of the
two trailing blocks — to put it back. So the claim in `mftl_remeshing.h` that "nothing here needs to
change" was false for the allocation helpers.

`MFTLRemeshMeasureBodies` needs the CIRCULAR MEAN, not the arithmetic one. A body straddling a periodic
seam has half its vertices at each end, so an arithmetic mean puts its centroid in the middle of the domain
and its radius at half the box — after which it is classified SPANNING and takes the phase schedule it does
not need, forever, silently. Guarded by `test_a_seam_straddling_droplet_is_not_measured_as_domain_sized`.

### §7.5 — assessed, not changed

`redistribute_core` does allocate six arrays of size `nproc` (`is_neighbor`, two `mftl_int_buffer` lists,
two/three `MFTL_SendBuffer` arrays, `touched`, `rank_bit`). At P = 10⁴ that is real. It was left alone on
purpose: it is a pre-existing property of the production redistribution, the brief for this work was to
avoid disturbing that path, and rewriting a 1500-line function's indexing to fix a memory constant is
exactly the kind of change that breaks what it was asked to protect. It belongs on the redistribution's own
schedule, with a measurement at P in the thousands to justify it.

---

## 1. Architectural Overview

This document specifies a scalable, production-grade parallel remeshing system for triangulated surface meshes within the MFTL front-tracking framework. The central design principles are:

- **The ensemble is the unit of parallelization policy.** An `MFTL_Ensemble` is not merely a bag of objects — it is a bag of objects that share the same domain decomposition assumption and therefore the same remeshing strategy. Never mix objects of fundamentally different scale inside one ensemble.
- **Two ensemble types cover all practical multiphase CFD scenarios.** A dispersed ensemble (many small bubbles/droplets) and a spanning ensemble (large interfaces, jets, films). Each type has exactly one remeshing code path.
- **All inter-rank communication is strictly 26-neighbor.** The decomposition geometry guarantees this. Global collectives are limited to one `MPI_Allreduce` per remeshing pass for convergence detection.
- **The serial remeshing kernel is never modified.** Parallelism is enforced by a frozen-entity mask. The serial kernel operates on the interior; the parallel machinery handles only the boundaries.

---

## 2. Ensemble Type Definition

```c
typedef enum {
    MFTL_ENSEMBLE_DISPERSED = 0,   /* many small objects, R_obj << h       */
    MFTL_ENSEMBLE_SPANNING  = 1    /* few large objects,  R_obj >> h       */
} MFTL_EnsembleType;

/* Add to MFTL_Ensemble: */
MFTL_EnsembleType type;
double            promote_radius;  /* objects above this → move to spanning */
double            demote_radius;   /* objects below this → move to dispersed */
```

**Creation contract:**
- `MFTL_ENSEMBLE_DISPERSED`: caller asserts that typical object radius $R_\text{obj} < 0.25 \cdot h_\text{min}$. The remeshing algorithm guarantees zero cross-rank communication for objects satisfying this.
- `MFTL_ENSEMBLE_SPANNING`: caller asserts that $h_\text{min} > 4\delta$ where $\delta \approx (k_\text{ghost} + k_\text{dilate} + 1) \cdot L_\text{max}$. If violated, the coarsening path (Section 4.4) must be used.

The dispatch is a single switch:

```c
int MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
    switch (ens->type) {
        case MFTL_ENSEMBLE_DISPERSED:
            return MFTLRemeshEnsembleDispersed(ens, params);
        case MFTL_ENSEMBLE_SPANNING:
            return MFTLRemeshEnsembleSpanning(ens, params);
    }
    return MFTL_ERROR;
}
```

---

## 3. The Four Parallel Remeshing Regimes

Every configuration of object count $N_\text{obj}$, rank count $P$, object radius $R_\text{obj}$, and rank box width $h$ falls into one of four regimes. The optimal strategy differs per regime.

| Regime | Condition | Physical example | Optimal strategy |
|--------|-----------|------------------|-----------------|
| **A — Dense** | $N_\text{obj} \gg P$, $R_\text{obj} \ll h$ | Bubbly flow, spray | Dispersed ensemble: object-centering + serial kernel |
| **B — Matched** | $N_\text{obj} \sim P$, $R_\text{obj} \sim h$ | Moderate bubble count | Dispersed or spanning depending on $R_\text{obj}/h$ |
| **C — Sparse** | $N_\text{obj} \ll P$ | Few large drops at high rank count | Coarsen to $P_\text{remesh}$ ranks (Section 4.4), then Regime A or D |
| **D — Large-object** | $R_\text{obj} \gg h$ | Jet, film, large deforming interface | Spanning ensemble: spatial shift scheme |

**How to detect the regime at runtime:**

```c
MFTL_Regime MFTLDetectRegime(const MFTL_Ensemble *ens,
                               const MFTL_RemeshParameters *params)
{
    double h_min   = MFTLDomainGetHMin(&ens->domain);
    double delta   = (params->k_ghost + params->k_dilate + 1)
                   * params->max_edge_length;
    double r_max   = MFTLEnsembleGetMaxObjectRadius(ens);  /* Allreduce inside */
    int    n_obj_g = MFTLEnsembleGetGlobalObjectCount(ens);

    if (r_max > 0.4 * h_min)    return MFTL_REGIME_D;   /* spanning          */
    if (n_obj_g < ens->domain.world_size / 4)
                                return MFTL_REGIME_C;   /* sparse            */
    return (n_obj_g > ens->domain.world_size)
         ? MFTL_REGIME_A : MFTL_REGIME_B;
}
```

---

## 4. Dispersed Ensemble: Object-Centering Strategy

### 4.1 Core Idea

For objects smaller than the rank box, a simpler alternative to the shift scheme exists: **move the objects to make them fully local, rather than moving the cut planes.** After centering, every object is wholly owned by one rank. The serial kernel runs on 100% of each object. No frozen entities. No phase scheduling.

### 4.2 Object-Centering Migration

```c
int MFTLRemeshEnsembleDispersed(MFTL_Ensemble *ens,
                                  MFTL_RemeshParameters *params)
{
    /* Step 1: migrate all triangles of each object to the rank
       whose box contains the object's centroid.
       Since R_obj < h/2, all triangles are on the centroid rank
       or one of its 26 neighbors → strictly local communication.     */
    MFTLEnsembleCenterObjects(ens);          /* 26-neighbor migration  */

    /* Step 2: every object is now fully local. No ghost layer needed
       for remeshing (ghosts are still built for advection separately). */
    int local_ops = 0;
    for (int o = 0; o < ens->num_objects; ++o)
    {
        MFTL_Object *obj = ens->objects[o];
        if (!MFTLObjectIsFullyLocal(obj)) continue;  /* safety: skip any
                                                        still-split obj  */
        local_ops += MFTLRemeshObjectSerial(obj, params);
        MFTLFinalizeRemeshing(obj);
    }

    /* Step 3: one Allreduce for convergence */
    int global_ops = 0;
    MPI_Allreduce(&local_ops, &global_ops, 1, MPI_INT,
                  MPI_SUM, ens->domain.comm_cart);
    return global_ops;
}

static inline int MFTLObjectIsFullyLocal(const MFTL_Object *obj)
{
    return (obj->n_ghost_triangles == 0);
}
```

`MFTLEnsembleCenterObjects` iterates over all objects, computes the centroid of each, determines the owner rank from `MFTLGetOwnerProc`, and packs triangles for migration if owner != `my_rank`. Uses the existing `MPI_Neighbor_alltoallv` path.

### 4.3 Smoothing in the Dispersed Ensemble

Because objects are fully local, smoothing has no special parallel constraints. Run Laplacian or Taubin smoothing directly inside `MFTLRemeshObjectSerial` after topology operations:

```
Per object, per pass:
  1. Edge splits      (all edges, no frozen check needed)
  2. Edge collapses
  3. Edge flips
  4. N_smooth iterations of Laplacian/Taubin smoothing
  5. Feature re-detection (normals changed by smoothing)
```

No frozen-entity gating is needed for any of these operations.

### 4.4 Regime C: Coarsening Before Dispersed Remeshing

When $N_\text{obj} \ll P$, many ranks hold zero objects. The coarsening factor $G$ groups $G$ flow ranks into one remesh rank:

$$G > \left(\frac{4\delta}{h_\text{min}}\right)^3$$

Choose $G$ as a perfect cube ($8, 27, 64, \ldots$) so grouped boxes remain roughly cubic.

```c
int MFTLComputeCoarseningFactor(const MFTL_Ensemble *ens,
                                  const MFTL_RemeshParameters *params)
{
    double delta = (params->k_ghost + params->k_dilate + 1)
                 * params->max_edge_length;
    double h_min = MFTLDomainGetHMin(&ens->domain);
    double ratio = 4.0 * delta / h_min;
    if (ratio <= 1.0) return 1;
    int cbrt_G = (int)ceil(ratio);
    return cbrt_G * cbrt_G * cbrt_G;
}

int MFTLRemeshEnsembleAdaptive(MFTL_Ensemble *ens,
                                 MFTL_RemeshParameters *params)
{
    int G = MFTLComputeCoarseningFactor(ens, params);
    if (G == 1)
        return MFTLRemeshEnsemble(ens, params);

    /* Threshold: if coarsening is extreme, just gather to root */
    if (G >= ens->domain.world_size / 4)
        return MFTLRemeshEnsembleSerialAllGather(ens, params);

    int flow_rank  = ens->domain.my_rank;
    int group_id   = flow_rank / G;
    int group_pos  = flow_rank % G;

    MPI_Comm group_comm;
    MPI_Comm_split(ens->domain.comm_cart, group_id, group_pos, &group_comm);

    MFTL_Ensemble *coarse_ens = NULL;
    if (group_pos == 0)
        coarse_ens = MFTLEnsembleGatherFromGroup(ens, G, group_comm);
    else
        MFTLEnsembleSendToGroupMaster(ens, group_comm);

    MPI_Comm remesh_comm = MPI_COMM_NULL;
    MPI_Comm_split(ens->domain.comm_cart,
                   (group_pos == 0) ? 0 : MPI_UNDEFINED,
                   group_id, &remesh_comm);

    int result = 0;
    if (remesh_comm != MPI_COMM_NULL) {
        MFTLDomainCoarsen(&coarse_ens->domain, G);
        coarse_ens->domain.comm_cart = remesh_comm;
        result = MFTLRemeshEnsemble(coarse_ens, params);
        MPI_Comm_free(&remesh_comm);
    }

    if (group_pos == 0)
        MFTLEnsembleScatterToGroup(coarse_ens, ens, G, group_comm);
    else
        MFTLEnsembleReceiveFromGroupMaster(ens, group_comm);

    MPI_Comm_free(&group_comm);
    if (coarse_ens) MFTLEnsembleFree(coarse_ens);

    MFTLEnsembleCreateSpatialGhosts(ens, params->max_edge_length);
    return result;
}
```

Group gather/scatter communication is within-group (contiguous rank slabs), so it remains 26-neighbor. The gather/scatter cost is $O(N_\text{tri})$ amortized over many timesteps between remesh events — negligible in practice.

---

## 5. Spanning Ensemble: Spatial Shift Scheme

### 5.1 The Geometric Guarantee

Let $h_k = \text{domain\_size}_k / \text{cuts}_k$ be the box width in axis $k$. Let $\delta$ be the geometric width of the frozen band:

$$\delta \approx (k_\text{ghost} + k_\text{dilate} + 1) \cdot L_\text{max}$$

**Hard requirement** (assert at ensemble creation, abort with a clear error if violated):

$$h_\text{min} > 4\delta$$

Define **Phase A** as cut planes at canonical positions and **Phase B** as all cut planes shifted by $h_k/2$ in every axis simultaneously.

A point frozen in Phase A has distance $\geq h/2 - \delta$ to the nearest Phase B plane. It is interior (unfrozen) in Phase B if and only if:

$$\frac{h}{2} - \delta > \delta \quad\Longleftrightarrow\quad \delta < \frac{h}{4}$$

This is the per-axis guarantee. In 3D, a single $(h/2, h/2, h/2)$ diagonal shift handles all **slab conflicts** (entity near one cut plane only). Two additional conflict types require more phases:

| Conflict type | Measure | Phases needed |
|---|---|---|
| Slab: near 1 axis | $O(\delta/h)$ | 2 — phases $(0,0,0)$ and $(1,1,1)$ |
| Tube: near 2 axes, mixed parity | $O((\delta/h)^2)$ | 4 — the even-parity OA set |
| Corner: near 3 axes, mixed parity | $O((\delta/h)^3)$ | 8 — all of $\{0,h/2\}^3$ |

The 4-phase even orthogonal-array schedule $S_\text{even} = \{(0,0,0),(0,1,1),(1,0,1),(1,1,0)\}$ (in units of $h/2$) covers all tubes. The full 8-phase schedule covers all corners. **In practice**, phases 3–8 operate on $\lesssim 1\%$ of entities; their serial kernel exits almost immediately. Typical convergence is 2–4 passes.

### 5.2 Decomposition Shift

Add to `MFTL_Domain`:

```c
double decomp_shift[3];   /* 0 or h_k/2 per axis */
```

```c
static const double PHASES[8][3] = {
    /* Phase 0,1: workhorses — cover all slabs */
    {0,0,0}, {1,1,1},
    /* Phase 2,3: complete even OA — cover all tubes */
    {0,1,1}, {1,0,1},
    /* Phase 4,5,6,7: odd set — cover all corners */
    {1,1,0}, {1,0,0}, {0,1,0}, {0,0,1}
};

void MFTLDomainSetShiftVec(MFTL_Domain *d, const double phase[3])
{
    for (int k = 0; k < 3; ++k) {
        double h = d->domain_size[k] / (double)d->cuts[k];
        d->decomp_shift[k] = phase[k] * 0.5 * h;
    }
}

static inline int MFTLGetOwnerProc(const MFTL_Domain *d, const double pos[3])
{
    int c[3];
    for (int k = 0; k < 3; ++k) {
        double s = (pos[k] - d->global_min[k] - d->decomp_shift[k])
                 * d->grid_factors[k];
        int i = (int)floor(s);
        if (d->is_periodic[k])
            i = ((i % d->cuts[k]) + d->cuts[k]) % d->cuts[k];
        else
            i = (i < 0) ? 0 : (i >= d->cuts[k] ? d->cuts[k]-1 : i);
        c[k] = i;
    }
    return d->rank_map[c[0] + c[1]*d->cuts[0]
                            + c[2]*d->cuts[0]*d->cuts[1]];
}
```

Between any two phases, each plane moves by at most $h/2$ per axis, so every migrating entity moves to one of the 26 graph neighbors. The `MPI_Neighbor_alltoallv` path is guaranteed for all phase transitions.

Also update `MFTLEnsembleCreateSpatialGhosts` Phase 1: neighbor box bounds must apply `decomp_shift` when reconstructing neighbor boxes.

### 5.3 Freeze Marking During Ghost Construction

The frozen set is defined operationally: **whatever was packed and sent as a ghost is frozen.** This is the cleanest possible definition and requires no separate geometric pass.

Inside `MFTLPackSingleTriangle` / `MFTLPackSingleVertex` call sites in `MFTLEnsembleCreateSpatialGhosts` and `MFTLExtendGhostLayer`:

```c
/* When packing owned entity as a ghost for a neighbor rank: */
obj->triangles[tri_idx].flags |= MFTL_FLAG_FROZEN;
for (int j = 0; j < 3; ++j)
    obj->vertices[obj->flat_tri_verts[3*tri_idx + j]].flags |= MFTL_FLAG_FROZEN;
```

Clear `MFTL_FLAG_FROZEN` at the start of each pass (it is recomputed from scratch from the new ghost layer after each migration):

```c
for (int i = 0; i < obj->n_vertices;  ++i) obj->vertices[i].flags  &= ~MFTL_FLAG_FROZEN;
for (int i = 0; i < obj->n_triangles; ++i) obj->triangles[i].flags &= ~MFTL_FLAG_FROZEN;
```

### 5.4 Frozen Band Dilation

The ghost layer marks entities within $k_\text{ghost}$ rings. One additional dilation step adds a safety margin covering smoothing's 1-ring read and collapse's 2-ring footprint. Use a two-flag trick to avoid in-sweep propagation:

```c
void MFTLDilateFrozenBand(MFTL_Object *obj, int rings)
{
    for (int r = 0; r < rings; ++r) {
        for (int i = 0; i < obj->n_half_edges; ++i) {
            MFTL_HalfEdge *he = &obj->half_edges[i];
            if (he->flags & MFTL_FLAG_DEAD) continue;
            int vb = he->vertex_idx;
            int va = obj->half_edges[
                         obj->half_edges[he->next_idx].next_idx
                     ].vertex_idx;
            if (obj->vertices[va].flags & MFTL_FLAG_FROZEN)
                obj->vertices[vb].flags |= MFTL_FLAG_FROZEN_NEXT;
            if (obj->vertices[vb].flags & MFTL_FLAG_FROZEN)
                obj->vertices[va].flags |= MFTL_FLAG_FROZEN_NEXT;
        }
        for (int i = 0; i < obj->n_vertices; ++i)
            if (obj->vertices[i].flags & MFTL_FLAG_FROZEN_NEXT)
                obj->vertices[i].flags = (obj->vertices[i].flags
                                         & ~MFTL_FLAG_FROZEN_NEXT)
                                         | MFTL_FLAG_FROZEN;
    }
}
```

Call with `rings = 1` after ghost construction. Use `k_ghost >= 2` to ensure no unfrozen vertex can circulate into the ghost-layer rim (where `twin_idx == MFTL_ERROR` does not imply a physical boundary).

### 5.5 Gating the Serial Kernel

Define:

```c
#define MFTL_FLAG_IMMUTABLE  (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED | MFTL_FLAG_FROZEN)
```

Rules per operation:

- **Split**: both endpoints non-frozen, non-ghost. New vertices are interior by construction.
- **Collapse**: both endpoints non-frozen, non-ghost. Their 1-rings are then guaranteed local-complete by the $k=2$ ghost halo.
- **Flip**: all four vertices $v_a, v_b, v_c, v_d$ non-frozen.
- **Smoothing**: skip if `v->flags & MFTL_FLAG_FROZEN || v->is_ghost`. *Reading* frozen/ghost neighbor positions is correct and required.

### 5.6 GID Generation Without Synchronization

The existing `next_vertex_gid++` is unsafe in parallel. Use rank-encoded 64-bit GIDs:

```c
/* 24-bit rank (supports 16M ranks) | 40-bit local counter (1T entities/rank) */
static inline int64_t MFTLNewGid(int rank, int64_t *counter)
{
    return ((int64_t)rank << 40) | (*counter)++;
}
```

Per-rank counters never reset across passes or timesteps, guaranteeing uniqueness. The originating rank is recoverable from any GID for debugging. **Do not use 32-bit GIDs** — front-tracking at scale exhausts them.

### 5.7 Correctness Contract

Three invariants that make rank-to-rank negotiation unnecessary:

1. **Exclusive geometric ownership.** After migration, every entity has exactly one rank permitted to modify it.
2. **Frozen = replicated.** An owned entity is frozen iff a copy exists as a ghost on some neighbor. Ops require the entire modified stencil to be unfrozen → no rank ever modifies anything another rank has a copy of. Conflicts are impossible by construction.
3. **Ghosts are throwaway.** Interior remeshing may make neighbor ghosts stale (e.g., a collapse deletes a triangle whose third vertex is frozen on a neighbor). This is safe: ghosts are read-only and rebuilt from scratch after every migration. **Never attempt to incrementally patch ghost topology across a remeshing pass.**

### 5.8 The Complete Driver

```c
int MFTLRemeshEnsembleSpanning(MFTL_Ensemble *ens,
                                 MFTL_RemeshParameters *params)
{
    MFTL_Domain *dom = &ens->domain;
    int clean_streak = 0;

    /* Runtime certificate: track entities never seen as interior */
    MFTLMarkAllNeverInterior(ens);

    for (int p = 0; p < 8; ++p)
    {
        /* 1. Set decomposition shift for this phase */
        MFTLDomainSetShiftVec(dom, PHASES[p]);

        /* 2. Migrate band entities to new owners; wipe old ghosts.
              Compaction MUST have run at end of previous pass.
              Triangle centers MUST be up to date before this call. */
        MFTLRedistributeListOfObjects(ens);

        /* 3. Build k=2 ghost halo; marks MFTL_FLAG_FROZEN on packed entities */
        MFTLEnsembleCreateSpatialGhosts(ens, params->max_edge_length);

        /* 4. Dilate frozen band by 1 ring (safety margin) */
        for (int o = 0; o < ens->num_objects; ++o)
            MFTLDilateFrozenBand(ens->objects[o], 1);

        /* 5. Update geometry: normals, areas, centers.
              MUST happen before feature detection.
              MUST happen before migration (centers drive ownership). */
        MFTLEnsembleUpdateGeometry(ens);
        MFTLDetectFeatures(ens);

        /* 6. Clear NEVER_INTERIOR flag on entities now unfrozen */
        MFTLClearNeverInteriorOnUnfrozen(ens);

        /* 7. Serial kernel with frozen gating; includes FinalizeRemeshing */
        int local_ops     = MFTLRemeshEnsembleInterior(ens, params);
        int local_pending = MFTLCountPendingOpsOnNeverInterior(ens, params);

        /* 8. Convergence: both ops AND pending work must be zero */
        int g[2], l[2] = {local_ops, local_pending};
        MPI_Allreduce(l, g, 2, MPI_INT, MPI_SUM, dom->comm_cart);

        if (g[0] == 0 && g[1] == 0) {
            clean_streak++;
            if (clean_streak == 2) break;   /* both parities clean: converged */
        } else {
            clean_streak = 0;
        }
    }

    /* Restore canonical decomposition for flow solver */
    MFTLDomainSetShiftVec(dom, PHASES[0]);   /* (0,0,0) */
    MFTLRedistributeListOfObjects(ens);
    MFTLEnsembleCreateSpatialGhosts(ens, params->max_edge_length);
    MFTLEnsembleUpdateGlobalCounts(ens);
    return MFTL_SUCCESS;
}
```

**Critical ordering notes:**
- `MFTLFinalizeRemeshing` (compaction) must run at the end of `MFTLRemeshEnsembleInterior`, before the next call to `MFTLRedistributeListOfObjects`. The redistribution iterates raw arrays and does not skip `MFTL_FLAG_DEAD` entries.
- Triangle centers must be recomputed after splits/collapses/smoothing and before migration. This is a correctness requirement, not a nicety — redistribution decides ownership from `tri->center`.
- Feature detection must run after geometry update (reads triangle normals) and before the serial kernel (reads feature flags).

### 5.9 Smoothing Integration in the Spanning Ensemble

Smoothing is placed after topology operations within `MFTLRemeshEnsembleInterior`, still inside the per-pass loop, before `MFTLFinalizeRemeshing`:

```
Per pass (inside MFTLRemeshEnsembleInterior):
  for each object:
    1. Edge splits:    stencil fully non-frozen/non-ghost
    2. Edge collapses: stencil fully non-frozen/non-ghost
    3. Edge flips:     all 4 diamond vertices non-frozen/non-ghost
    4. Vertex smoothing (N_smooth = 5–20 iterations):
         for each vertex v:
           if (v->flags & MFTL_FLAG_FROZEN || v->is_ghost) continue;
           /* Reading frozen/ghost neighbor positions is CORRECT.     */
           /* Feature vertices: tangential displacement only.         */
           MFTLSmoothVertex(v, obj);
    5. Recompute normals/areas for modified triangles
    6. MFTLFinalizeRemeshing(obj)  ← compaction before next migration
```

Because smoothing reads frozen/ghost positions without modifying them, and because the $k=2$ ghost halo guarantees that the 1-ring of every unfrozen vertex is complete, smoothing is fully correct across partition boundaries without any additional communication.

**Volume conservation**: use Taubin $\lambda/\mu$ smoothing (two-pass with $\lambda > 0$ and $\mu < 0$) or apply a global volume correction factor after smoothing. Pure Laplacian shrinks closed surfaces.

---

## 6. Object Lifecycle: Promotion and Demotion

Objects change scale over time (coalescence, atomization). Handle this with a single rebalance call before each remesh event:

```c
void MFTLEnsembleRebalance(MFTL_Ensemble *dispersed,
                             MFTL_Ensemble *spanning,
                             const MFTL_RemeshParameters *params)
{
    double h_min = MFTLDomainGetHMin(&dispersed->domain);

    /* Hysteresis prevents thrashing at the boundary */
    double promote_r = dispersed->promote_radius;  /* e.g., 0.40 * h_min */
    double demote_r  = spanning->demote_radius;    /* e.g., 0.20 * h_min */

    /* Check dispersed objects for growth */
    for (int i = dispersed->num_objects - 1; i >= 0; --i)
        if (MFTLObjectGetRadius(dispersed->objects[i]) > promote_r)
            MFTLMoveObjectBetweenEnsembles(dispersed, spanning, i);

    /* Check spanning objects for fragmentation */
    for (int i = spanning->num_objects - 1; i >= 0; --i)
        if (MFTLObjectGetRadius(spanning->objects[i]) < demote_r)
            MFTLMoveObjectBetweenEnsembles(spanning, dispersed, i);
}
```

`MFTLMoveObjectBetweenEnsembles` is a pointer swap — triangles do not move at this point. The triangle migration happens automatically in the next redistribution step of the receiving ensemble. Call `MFTLEnsembleRebalance` once per remesh event, before dispatching to `MFTLRemeshEnsemble`.

---

## 7. Data Structure Optimization: Efficient Looping

The naive implementation iterates over all $N$ triangles/edges every pass. As remeshing converges, 90%+ of entities are already within bounds but still scanned. This section replaces the full-array sweep with a **work-list** architecture.

### 7.1 Work-List Per Object

```c
typedef struct {
    int  *edge_indices;   /* half-edge indices of candidate edges   */
    int   n_edges;
    int   capacity;
    /* Separate lists for split and collapse candidates             */
    int  *split_candidates;
    int   n_split;
    int  *collapse_candidates;
    int   n_collapse;
    /* Vertex list for smoothing pass                              */
    int  *dirty_vertices;
    int   n_dirty;
} MFTL_WorkList;
```

**Initialization** (start of each remesh call, or after migration when ghost layer is rebuilt):

```c
void MFTLWorkListInit(MFTL_WorkList *wl, const MFTL_Object *obj)
{
    wl->n_split = wl->n_collapse = wl->n_dirty = 0;
    for (int i = 0; i < obj->n_half_edges; ++i) {
        MFTL_HalfEdge *he = &obj->half_edges[i];
        if (he->flags & MFTL_FLAG_DEAD)   continue;
        if (he->flags & MFTL_FLAG_FROZEN) continue;
        if (he->is_ghost)                  continue;
        double len = MFTLEdgeLength(obj, i);
        if (len > params->max_edge_length) MFTLWorkListAddSplit(wl, i);
        if (len < params->min_edge_length) MFTLWorkListAddCollapse(wl, i);
    }
    for (int i = 0; i < obj->n_vertices; ++i) {
        if (obj->vertices[i].flags & MFTL_FLAG_FROZEN) continue;
        if (obj->vertices[i].is_ghost)                  continue;
        MFTLWorkListAddDirty(wl, i);
    }
}
```

**After each topology operation**, only the modified neighborhood is re-evaluated:

```c
/* After MFTLSplitEdge(obj, edge_idx, &new_vert_idx): */
void MFTLWorkListAfterSplit(MFTL_WorkList *wl, MFTL_Object *obj,
                              int orig_edge, int new_vert)
{
    /* The 4 new edges emanating from new_vert need evaluation */
    int he = obj->vertices[new_vert].outgoing_half_edge;
    do {
        if (!(obj->half_edges[he].flags & MFTL_FLAG_FROZEN))
            MFTLWorkListAddSplit(wl, he);
        he = obj->half_edges[obj->half_edges[he].twin_idx].next_idx;
    } while (he != obj->vertices[new_vert].outgoing_half_edge);
    MFTLWorkListAddDirty(wl, new_vert);
}

/* After MFTLCollapseEdge(obj, edge_idx, &surviving_vert): */
void MFTLWorkListAfterCollapse(MFTL_WorkList *wl, MFTL_Object *obj,
                                  int surviving_vert)
{
    /* Re-evaluate all edges in 1-ring of surviving vertex */
    int he = obj->vertices[surviving_vert].outgoing_half_edge;
    do {
        if (!(obj->half_edges[he].flags & MFTL_FLAG_FROZEN))
        {
            double len = MFTLEdgeLength(obj, he);
            if (len > params->max_edge_length) MFTLWorkListAddSplit(wl, he);
            if (len < params->min_edge_length) MFTLWorkListAddCollapse(wl, he);
        }
        he = obj->half_edges[obj->half_edges[he].twin_idx].next_idx;
    } while (he != obj->vertices[surviving_vert].outgoing_half_edge);
    MFTLWorkListAddDirty(wl, surviving_vert);
}
```

An edge that is checked but passes all criteria (within target length, no flip needed) is simply not re-added to the work list. It will only re-enter if a neighboring operation changes its length or valence. **When the work list is empty, the pass is done — no Allreduce needed for the local convergence check.** Only the global convergence Allreduce (checking whether any rank still has pending work) remains.

### 7.2 Priority Ordering Within the Work List

Rather than processing edges in insertion order, sort for maximum effectiveness:

```c
/* Sort split candidates descending by length (process longest first) */
/* Sort collapse candidates ascending by length (process shortest first) */
/* Both maximise the "useful work per operation" ratio.                 */
qsort(wl->split_candidates,    wl->n_split,    sizeof(int), MFTLCmpEdgeLenDesc);
qsort(wl->collapse_candidates, wl->n_collapse, sizeof(int), MFTLCmpEdgeLenAsc);
```

This is especially important in the first pass where many edges are out of bounds simultaneously — it prevents a long edge being re-evaluated multiple times as its neighbors are fixed.

### 7.3 Avoiding Dead-Slot Iteration After Compaction

After `MFTLFinalizeRemeshing` compacts the arrays, all active triangles and vertices are in contiguous slots `[0, n_active)`. The work list is rebuilt from scratch after compaction (using the `MFTLWorkListInit` path). **No dead-slot overhead between compaction and the next pass**.

The pattern is:

```
End of pass:
  topology ops complete → compaction → rebuild work list for next pass
```

The work list never holds indices into dead slots because it is rebuilt immediately after compaction. Dead-slot scanning is therefore impossible by construction.

### 7.4 The NEVER_INTERIOR Certificate Bookkeeping

```c
/* Set at the beginning of the first spanning-ensemble remesh call:      */
for each entity: entity->flags |= MFTL_FLAG_NEVER_INTERIOR;

/* Clear when the entity is seen as unfrozen in some phase:              */
void MFTLClearNeverInteriorOnUnfrozen(MFTL_Ensemble *ens)
{
    for each object obj in ens:
        for each vertex v in obj:
            if (!(v->flags & MFTL_FLAG_FROZEN) && !v->is_ghost)
                v->flags &= ~MFTL_FLAG_NEVER_INTERIOR;
        for each triangle t in obj:
            if (!(t->flags & MFTL_FLAG_FROZEN) && !t->is_ghost)
                t->flags &= ~MFTL_FLAG_NEVER_INTERIOR;
}

/* Count candidate ops rejected solely because the stencil is frozen
   AND the stencil has never been interior (a true stuck entity):        */
int MFTLCountPendingOpsOnNeverInterior(const MFTL_Ensemble *ens,
                                         const MFTL_RemeshParameters *params)
{
    int pending = 0;
    for each object obj in ens:
        for each edge e in obj->split_candidates + collapse_candidates:
            if (MFTLStencilAllNeverInterior(obj, e))
                pending++;
    return pending;
}
```

`pending == 0` globally is the convergence certificate. It cannot be spoofed by an entity that simply doesn't need any operation (it's already in spec) — such entities have `NEVER_INTERIOR` cleared after their first unfrozen pass.

### 7.5 O(P) Allocation Optimization

`MFTLRedistributeListOfObjects` currently uses `calloc(nproc, ...)` for send buffers. At $10^4+$ ranks this wastes memory and initialization time since at most 26 destinations are ever used during remeshing passes.

```c
/* Use neighbor-slot indexing instead of global-rank indexing: */
int  neighbor_ranks[26];
int  n_neighbors;
/* buffers indexed [0..n_neighbors), not [0..world_size) */
```

The 26 neighbor ranks are fixed for the duration of a remeshing call (the decomposition shift moves planes but not the neighbor graph). Compute once at the start of the remeshing call and reuse.

---

## 8. Full Remesh Call Sequence

The following shows where every component plugs in relative to the broader MFTL timestep loop:

```
Each timestep (or every N_ts timesteps):

  1. MFTLEnsembleRebalance(dispersed_ens, spanning_ens, params)
        → pointer swaps only, no communication

  2. [Dispersed ensemble]
     MFTLRemeshEnsembleAdaptive(dispersed_ens, params)
        → MFTLComputeCoarseningFactor
        → if G>1: coarsen subcommunicator, gather, remesh, scatter
        → MFTLEnsembleCenterObjects           (26-neighbor migration)
        → MFTLRemeshObjectSerial per object:
              work-list init
              split → collapse → flip → smooth(N iters) → feature detect
              MFTLFinalizeRemeshing
        → MPI_Allreduce (convergence)

  3. [Spanning ensemble]
     MFTLRemeshEnsembleSpanning(spanning_ens, params)
        → for each phase p in 0..7:
              MFTLDomainSetShiftVec
              MFTLRedistributeListOfObjects    (26-neighbor, band only)
              MFTLEnsembleCreateSpatialGhosts  (k=2, marks FROZEN)
              MFTLDilateFrozenBand             (rings=1)
              MFTLEnsembleUpdateGeometry
              MFTLDetectFeatures
              MFTLClearNeverInteriorOnUnfrozen
              MFTLRemeshEnsembleInterior:
                  work-list init
                  split → collapse → flip → smooth(N iters, frozen-gated)
                  recompute centers/normals
                  MFTLFinalizeRemeshing
              MPI_Allreduce(ops, pending)
              if converged for 2 consecutive phases: break
        → restore phase (0,0,0), redistribute, rebuild ghosts

  4. Advection step (existing code, unchanged)

  5. Rebuild flow-solver ghost data (existing code, unchanged)
```

---

## 9. Required Assertions and Failure Modes

These must fire with clear diagnostic messages, not silent degradation:

```c
/* At spanning ensemble creation: */
MFTL_ASSERT(h_min > 4.0 * delta,
    "h_min=%.4f <= 4*delta=%.4f. "
    "Increase rank box size or reduce k_ghost/L_max. "
    "Use MFTLComputeCoarseningFactor to find required G.",
    h_min, 4.0*delta);

/* At dispersed ensemble creation: */
MFTL_ASSERT(typical_radius < 0.25 * h_min,
    "Typical object radius %.4f >= 0.25*h_min=%.4f. "
    "This object should be in a spanning ensemble.",
    typical_radius, 0.25*h_min);

/* After 8 phases of spanning remesh with pending > 0: */
MFTL_ASSERT(pending == 0,
    "Spanning remesh did not converge after 8 phases. "
    "pending=%d entities permanently frozen. "
    "h_min=%.4f, 4*delta=%.4f. Constraint violated at runtime.",
    pending_g, h_min, 4.0*delta);

/* Before every migration: */
MFTL_ASSERT(MFTLAllObjectsCompacted(ens),
    "MFTLFinalizeRemeshing must be called before redistribution.");

/* Before geometry update: */
MFTL_ASSERT(MFTLAllCentersUpToDate(ens),
    "Triangle centers must be recomputed after remeshing "
    "before migration (ownership is determined by center).");
```

---

## 10. Scalability Cost Model

With band fraction $\beta = O(\delta/h) \ll 1$, $N$ local owned triangles, and $P$ ranks:

$$T_\text{pass} = \underbrace{O(\beta N)}_{\text{migration, 26-nbr}} + \underbrace{O(\beta N)}_{\text{ghost build}} + \underbrace{O(N_\text{active})}_{\text{serial kernel (work-list)}} + \underbrace{O(\log P)}_{\text{1× Allreduce}}$$

Where $N_\text{active}$ shrinks each pass: $N_\text{active}^{(p+1)} \leq \epsilon \cdot N_\text{active}^{(p)}$ for some $\epsilon < 1$ depending on mesh quality. The work-list ensures the serial kernel cost tracks actual remaining work, not total mesh size.

**Weak scaling** (fixed work per rank as $P$ grows): $\beta = O(\delta/h)$ is constant since $h$ is fixed per rank. Communication and ghost-build are $O(\beta N) = O(1)$ per rank. Only the Allreduce grows as $O(\log P)$. The scheme is **asymptotically optimal** for this problem class.

**Strong scaling** (fixed total work, growing $P$): $N \sim N_\text{total}/P$ shrinks, $h \sim h_\text{total}/P^{1/3}$ also shrinks. The constraint $h > 4\delta$ eventually fails → use the coarsening path from Section 4.4 to limit $P_\text{remesh}$.

---

## 11. API Summary

```c
/* Ensemble lifecycle */
MFTL_Ensemble *MFTLEnsembleCreate(MFTL_EnsembleType type,
                                    MFTL_Domain *domain,
                                    double promote_radius,
                                    double demote_radius);
void           MFTLEnsembleDestroy(MFTL_Ensemble *ens);
void           MFTLEnsembleRebalance(MFTL_Ensemble *dispersed,
                                      MFTL_Ensemble *spanning,
                                      const MFTL_RemeshParameters *params);

/* Top-level dispatch */
int  MFTLRemeshEnsemble(MFTL_Ensemble *ens,
                          MFTL_RemeshParameters *params);
int  MFTLRemeshEnsembleAdaptive(MFTL_Ensemble *ens,   /* handles coarsening  */
                                  MFTL_RemeshParameters *params);

/* Dispersed path (called by dispatch) */
int  MFTLRemeshEnsembleDispersed(MFTL_Ensemble *ens,
                                    MFTL_RemeshParameters *params);
void MFTLEnsembleCenterObjects(MFTL_Ensemble *ens);

/* Spanning path (called by dispatch) */
int  MFTLRemeshEnsembleSpanning(MFTL_Ensemble *ens,
                                   MFTL_RemeshParameters *params);
void MFTLDomainSetShiftVec(MFTL_Domain *d, const double phase[3]);
void MFTLDilateFrozenBand(MFTL_Object *obj, int rings);
int  MFTLRemeshEnsembleInterior(MFTL_Ensemble *ens,    /* serial kernel, gated */
                                   MFTL_RemeshParameters *params);
int  MFTLCountPendingOpsOnNeverInterior(const MFTL_Ensemble *ens,
                                          const MFTL_RemeshParameters *params);

/* Work-list management */
void MFTLWorkListInit(MFTL_WorkList *wl, const MFTL_Object *obj,
                       const MFTL_RemeshParameters *params);
void MFTLWorkListAfterSplit(MFTL_WorkList *wl, MFTL_Object *obj,
                              int orig_edge, int new_vert);
void MFTLWorkListAfterCollapse(MFTL_WorkList *wl, MFTL_Object *obj,
                                  int surviving_vert,
                                  const MFTL_RemeshParameters *params);
void MFTLWorkListAfterFlip(MFTL_WorkList *wl, MFTL_Object *obj,
                              int edge_idx);

/* GID generation */
int64_t MFTLNewGid(int rank, int64_t *counter);

/* Coarsening */
int MFTLComputeCoarseningFactor(const MFTL_Ensemble *ens,
                                   const MFTL_RemeshParameters *params);
```

---

## 12. Implementation Checklist for the Coding Agent

- [ ] Add `MFTL_EnsembleType` field and promote/demote radii to `MFTL_Ensemble`
- [ ] Add `decomp_shift[3]` to `MFTL_Domain`; update `MFTLGetOwnerProc` and neighbor-bounds computation in ghost construction
- [ ] Implement `MFTLDomainSetShiftVec` with the 8-phase table
- [ ] Add `MFTL_FLAG_FROZEN`, `MFTL_FLAG_FROZEN_NEXT`, `MFTL_FLAG_NEVER_INTERIOR` to flag definitions; update `MFTL_FLAG_IMMUTABLE`
- [ ] Insert freeze marking into `MFTLPackSingleTriangle` / `MFTLPackSingleVertex`; add frozen-clear at pass start
- [ ] Implement `MFTLDilateFrozenBand` with two-flag sweep
- [ ] Gate `CollapseApproved`, `SplitApproved`, `FlipApproved`, and smoothing loop on `MFTL_FLAG_IMMUTABLE`
- [ ] Switch GIDs to 64-bit rank-encoded scheme; remove all `MPI_Exscan` per-pass ID allocation
- [ ] Implement `MFTL_WorkList` with init, split/collapse/flip update functions, priority sort
- [ ] Replace `calloc(nproc,...)` send buffers in redistribution with 26-slot neighbor-indexed buffers
- [ ] Implement `MFTLEnsembleCenterObjects` for dispersed remeshing
- [ ] Implement `MFTLEnsembleRebalance` with hysteresis radii
- [ ] Implement `MFTLComputeCoarseningFactor` and the subcommunicator coarsening path
- [ ] Add all required assertions (Section 9) with diagnostic messages
- [ ] Ensure `MFTLFinalizeRemeshing` is called at end of each pass before redistribution
- [ ] Ensure `MFTLEnsembleUpdateGeometry` (including triangle centers) is called after topology ops and before migration
- [ ] Ensure `MFTLDetectFeatures` is called after geometry update, every pass
- [ ] Update periodic and non-periodic boundary handling in `MFTLGetOwnerProc` for shifted planes
- [ ] Unit test: sphere on $2\times2\times2$ ranks — compare Euler characteristic $V - E + F = 2$ and edge-length histogram against serial result after each phase
# 01 — Periodic extent limitation: state it, check it

Status: IMPLEMENTED in `85cec62` (branch `christian`, 2026-10-03), with these deviations:
- **Four frames, not three** (cuts at 0, L/2, L/4, 3L/4). The 0 and L/2 frames are computed and compared exactly as before, so every extent the old code got right is bit-identical, and the topology digests are unchanged. Exact for every compact body below 3L/4; "covers" is decided at 3L/4, by size, never by placement.
- **No refusal inside the remesher or the topology change.** Neither is wrong for a body in [L/2, 3L/4): the remesher's spanning tests already remesh a 0.7L sphere, and the topology seam test uses a 0.525L dumbbell. The rule is enforced by `MFTLEnsembleCheckPeriodicExtents`, which the application calls.
- **The silent failure fixed** was a skipped volume correction (`volume_before = 0`), not a wrong one. Clusters with no flux axis are now counted in `n_clusters_unmeasured`.
- **The spatial halo was left on its own two-frame code.** It is conservative (it over-sends) and was not deduplicated.
Replaces: the per-vertex image counters of MFTL2 `main` (`plans/16` §3 there)
Used by: `03-body-integrals.md` (the centre of a closed body on a periodic axis)

## 1. Purpose

Per-body quantities on a periodic axis need every vertex lifted into one consistent image. Volume, centroid, the topology-change volume correction and the remesher's body measurement all depend on that.

`christian` does this with heuristics:
- a two-frame extent (`topo_cluster.c:189-259`, `spatial_halo.c:182-232`);
- a circular-mean centroid (`remesh_comm.c:509`);
- a per-triangle nearest image to the centre (`topo_cluster.c:483-491`).

These are exact for every body narrower than L/2. **The plan makes that the documented rule and checks it**, so a body that breaks it produces an error instead of a wrong number.

## 2. Findings that decide it

- **No open-surface example needs an unwrapped coordinate.**
  - Rayleigh–Taylor and the liquid jet run on fully non-periodic domains (`rayleigh_taylor_open_surface.c:1131`, `liquid_jet_open_surface.c:1173`).
  - The travelling-wave surface wraps the periodic x axis, but its volume is a flux along the non-periodic z axis about the bottom wall (`travelling_waves_open_surface.c:931-983`).
- **Where `christian` is silently wrong today:** a compact body with extent E in [L/2, L) that straddles both 0 and L/2.
  - In both frames the triangles' bounds then span the axis (`hi − lo ≥ L`), so `full[k]` is set and the body is treated as wrapping (`topo_cluster.c:243-253`).
  - The volume correction then uses a wrong flux axis or none.
  - No error is raised.
- **Two frames do measure exactly any body with E < L/2.** Its gap is wider than L/2, so it contains the cut point of at least one of the two frames. That is why everything works for small bodies.
- **Image counters would remove the limitation.** That would cost:
  - 6 B per vertex as an int16 column (about +1.7 % of 172 B per triangle);
  - +8 B on every migrating vertex record;
  - an obligation on every position writer (application advection, split, collapse, smoother, topology change);
  - a distributed canonicalisation.

  Their only gains are compact bodies of at least L/2, multi-period trajectories (an application gets these per body from the minimum image of its centroid displacement each step), and rigid transforms across the seam (`christian` has none).

## 3. The rule

> **On every periodic axis, a body either wraps the axis (it closes on itself only after going around the period), or its extent along that axis is below L/2.**
>
> Every edge shorter than L/2 on every periodic axis is already required.

Under the rule:
- the nearest image about a body's centre is exact for every vertex and triangle;
- "wraps the axis" is equivalent to "covers the whole axis", so classification needs no topology computation.

## 4. Design

### 4.1 One extent helper, three frames (`src/geometry/`, local, no MPI)

Each triangle contributes its minimum-image corners (`MFTLGetTriangleCorners`) in three frames per periodic axis. The cut points are at `global_min + {0, L/3, 2L/3}`. In each frame the triangle is shifted rigidly so that corner 0 lies in that frame's window. The contribution is `lo` and `−hi`, so one MIN reduction carries both.

The resolve step picks the frame with the smallest `hi − lo` and returns:
- the centre and half-width;
- `wraps[k]` when every frame reports `hi − lo ≥ L`;
- `violates[k]` when the body does not wrap and its smallest extent is ≥ L/2 − margin.

The margin is a fraction of the longest edge, so it is relative and never an absolute constant.

**Why three frames.** An open arc longer than L/3 contains one of three equally spaced cut points. So any compact body with E < 2L/3 is measured exactly. Bodies grow continuously, so a body cannot reach L/2 without passing through [L/2, 2L/3), where the check sees it. Two frames miss exactly the straddle-both case above. More than three buys nothing under the rule.

**Payload.** 18 MIN slots per body (3 frames × 3 axes × lo/−hi) on the periodic axes, against 12 today.

Suggested API, local:

- `MFTLPeriodicExtentInit(double acc[18])`
- `MFTLPeriodicExtentAddTriangle(const MFTL_Object *obj, int t, double acc[18])`
- `MFTLPeriodicExtentResolve(const MFTL_Domain *dom, const double acc[18], double centre[3], double half[3], int wraps[3], int violates[3])`

### 4.2 The three consumers

1. **Topology clusters.** `MFTLTopoLocalExtents` / `MFTLTopoResolveExtents` (`topo_cluster.c:189-259`) move to the helper. The cluster record gains `violates[3]`. `topo_driver.c:340-349` returns `MFTL_ERROR` on every rank when any reconstructed cluster violates the rule, which is decided from the reduced values and is therefore identical everywhere.
2. **Remesher measurement.** `MFTLRemeshMeasureBodies` (`remesh_comm.c:388`, run on every remesh call) adds the 18 slots to its existing first reduction. On violation the remesh returns `MFTL_ERROR` collectively. The circular-mean centroid stays; it is only used to choose the strategy and the dispersed home rank.
3. **Spatial halo** (cleanup, optional). Its own two-frame code (`spatial_halo.c:182-232`, `:422`) moves to the helper. It needs only a conservative extent, so this is deduplication, not a correctness change. It must stay bit-identical or be gated separately.

### 4.3 A public check

```c
typedef struct MFTL_PeriodicExtentReport {
  int n_violations;        /* bodies breaking the rule, counted once (lowest holder) */
  int first_body_gid;      /* smallest offending gid, -1 if none                    */
  int first_axis;          /* its axis                                              */
  double first_extent;     /* its measured extent                                   */
} MFTL_PeriodicExtentReport;

int MFTLEnsembleCheckPeriodicExtents(MFTL_Ensemble *ens, MFTL_PeriodicExtentReport *report);  /* collective */
```

- One `MFTLEnsembleReduceBodies` call (MIN slots, `AUTO` scope, holder route).
- Bodies are counted exactly once through the lowest holder (the existing `lowest_holder` rule).
- The report is identical on every rank.
- Intended uses: output intervals, after setup, and in Debug inside `MFTLRedistribute`'s existing checksum path.
- It returns `MFTL_SUCCESS` when the report is clean.
- With no periodic axis it returns immediately on every rank and makes no MPI call. That is safe because periodicity is a domain property.

### 4.4 MPI confinement

- The helper is local.
- The reductions stay in files that already make them: `remesh_comm.c`, `topo_comm.c` / `topo_driver.c` (check against `test_topology_change_mpi_is_confined`), and the ensemble module for the public check.

## 5. Files to modify

| File | Change |
|---|---|
| `include/mftl_geometry.h` | helper declarations and Doxygen (`@param`/`@return`, zero-warning manual) |
| `src/geometry/periodic_extent.c` (new) | helper implementation |
| `src/topology_change/topo_cluster.c`, `topo_internal.h`, `topo_driver.c` | use the helper; `violates[3]`; collective error |
| `src/remeshing/remesh_comm.c` | 18 slots in `MFTLRemeshMeasureBodies`; collective error |
| `src/communication/spatial_halo.c` | optional dedup onto the helper |
| `include/mftl_ensemble.h`, `src/ensemble/ensemble_reduce.c` (or a new `ensemble_check.c`) | `MFTLEnsembleCheckPeriodicExtents` |
| `tests/test_periodic_extents.c` (new), `tests/CMakeLists.txt` | tests of §6 at np 1, 2, 4, 8, 27 |
| `src/geometry/README.md` | the rule, the three-frame argument, what breaks without it |
| `CLAUDE.md` | invariant entry next to the minimum-image rule |
| `docs/analysis.md` | one paragraph: why counters were not adopted, and their design if ever needed (§7) |

## 6. Tests

- **Detection at every placement.** An ellipsoid grows from 0.40 L to 0.60 L along a periodic axis, placed at offsets that straddle 0, L/3, L/2 and 2L/3 and none of them. It is flagged exactly when it crosses L/2, at np 1, 2, 4, 8, 27, with `agreed()` verdicts. The straddle-both offsets fail under today's two frames; keep them as the regression.
- **No false alarm.** A 0.49 L body at 64 offsets passes. A seam-straddling sphere passes.
- **Wrapping is not a violation.** A film wrapping x and y in a fully periodic box gives `wraps = {1, 1, 0}` and no violation. The travelling-wave ribbon gives `wraps[x] = 1`.
- **Non-vacuity.** The report states how many bodies were held by two or more ranks.
- **Topology change and remesh.** A coalescence producing an over-long body returns `MFTL_ERROR` on every rank and does not hang.

## 7. Gate

- `leveque_remeshing_parallel -digest`, `parallel_advection_redistribute_list --digest`, and the open-surface and topology examples: **bit-identical**. The check only reads; topology extents are bit-identical for every body that obeys the rule.
- Interleaved remesh A/B within noise (18 more MIN slots on an existing reduction).
- `build_docs.py --strict` with zero warnings.

## 8. Considered and deferred

### Per-vertex image counters

The design, if a case ever needs it, is recorded so that nobody re-derives it:

- **Storage.** `int16_t img[3*capacity]` in the vertex pool block, present only on periodic domains, plus a per-body `int64 img_base[3]` carried in the redistribution body header.
- **Maintenance.**
  - The application wraps through a counting `MFTLVertexWrap`.
  - Ghosts follow by the minimum-image jump on receipt, so the ghost sync carries no extra bytes.
  - Library writers place vertices from an anchor.
- **Canonicalisation.** A distributed weighted union-find with lattice offsets, reusing the label rounds of `topo_split_bodies.c`.
- **Invariant.** "Compact along k ⇒ no cut edge along k".
- **Prior art.** LAMMPS (3×10-bit packed), HOOMD-blue (int32×3), GROMACS (on-demand graph traversal, no storage) and CGAL (per-simplex offsets). Winding needs no counters at all: it is the cycle sum of the per-edge minimum-image wraps.

**Revive when** a required case has a compact body of at least L/2 on a periodic axis:
- unit-cell slug flow with bubbles longer than half the cell;
- strongly stretched drops in a short periodic shear box;
- rigid bodies transformed across many periods.

### Exact topological wrap classification (union-find over edges)

Not needed under the rule, because full coverage then equals wrapping. **Revive with the counters.**

/*
 * @Author: Christian Gorges
 * @Date: 2026-06-17 11:03:52
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2026-06-17 11:03:52
 */

// Stage 10 — parallel remeshing + load balancing (MFTL1 parity).
//
// Design (approved in plans/10-parallel-remeshing-load-balancing.md):
//
//   * Single-owner rule (MFTL1 Boundary_Check): every edge operator acts
//     ONLY on parallel-interior edges — both endpoints owned by this rank
//     and (split/flip) both adjacent triangles local and ghost-free,
//     (collapse) the entire edge fan local and ghost-free. Rings around
//     owned vertices are complete by the ghost-layer invariant (redistribute
//     ships every triangle touching an owned vertex to its owner), so
//     approvals and fits are exactly local and NO mid-pass topology
//     communication is needed. Boundary edges are remeshed by the second
//     pass after the ownership ranges are shifted (MFTL1's two-pass scheme).
//
//   * Load balancing (MFTL1 MFTLRedistributeObjectLoadBalanced): x-only
//     1-D slab decomposition with arbitrary (sub-cell) cut positions, stored
//     in MFTL_Domain.lb_* and honoured by MFTLGetOwnerProc. The slab-width
//     optimizer runs on Allreduced counts, so all ranks compute identical
//     cuts. Requires an x-only Cartesian grid (cuts[1]==cuts[2]==1) so that
//     adjacent slabs are comm_graph neighbours (ghost updates stay valid).
//
//   * Two-pass boundary shift (MFTL1 ShiftOwnershipRanges): pass 1 remeshes
//     on the balanced slabs; pass 2 shifts the cuts right by
//     max(0.2*min slab width, 3*reference_length) so former boundary
//     triangles become interior and get remeshed.
//
//   * GID uniqueness: split phases mint entities from per-rank disjoint GID
//     blocks (prefix-scheduled over candidate counts, base = global counter
//     maximum). Counters are re-synchronized (Allreduce MAX) after every
//     redistribution.
//
//   * Stage-09 follow-up: the serial edge-local smoothers fail loudly on
//     distributed meshes; the parallel driver uses the *Parallel variants,
//     which gate every edge to the parallel-interior rule above.

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

/* ══════════════════════════════════════════════════════════════════════
 * MFTLEnsembleGetTypicalRadius
 *
 * Returns the maximum bounding-box width (largest dimension) across all
 * objects in the ensemble, reduced globally with MPI_MAX so every rank
 * sees the same value. Used by MFTLSelectRemeshAlgorithm.
 * ══════════════════════════════════════════════════════════════════════ */
double MFTLEnsembleGetTypicalRadius(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL && "MFTLEnsembleGetTypicalRadius: ensemble is NULL\n");

  MFTL_Domain *dom = &ens->domain;
  double local_max = 0.0;

  MFTL_Object **object_base = ens->objects;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = object_base[o];
    MFTL_ASSERT(obj != MFTL_NULL && "MFTLEnsembleGetTypicalRadius: object is NULL\n");

    double bbox_min[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
    double bbox_max[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};

    MFTL_Vertex *vertex_base = obj->vertices;

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &vertex_base[i];

      if (v->is_ghost) continue;

      for (int j = 0; j < 3; j++)
      {
        bbox_min[j] = MFTL_MIN(bbox_min[j], v->pos[j]);
        bbox_max[j] = MFTL_MAX(bbox_max[j], v->pos[j]);
      }
    }

    /* Largest extent in any single axis (periodicity-aware). */
    double obj_width[3];
    MFTLGetMinimumImageVector(dom, obj_width, bbox_min, bbox_max);
    local_max = MFTL_MAX(MFTL_MAX(local_max, obj_width[0]), MFTL_MAX(obj_width[1], obj_width[2]));
  }

  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, dom->comm_graph);

  return global_max;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLSelectRemeshAlgorithm
 *
 * SPANNING ensemble            -> MFTL_ALGO_SHIFT_SCHEME
 * DISPERSED, W < 0.5 * h_min   -> MFTL_ALGO_CENTERING (gather + serial kernel)
 * DISPERSED otherwise          -> MFTL_ALGO_SHIFT_SCHEME
 *
 * Collective: issues one MPI_Allreduce (inside MFTLEnsembleGetTypicalRadius).
 * ══════════════════════════════════════════════════════════════════════ */
MFTL_RemeshAlgorithm MFTLSelectRemeshAlgorithm(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(params != MFTL_NULL);

  if (ens->type == MFTL_ENSEMBLE_SPANNING)
  {
    ens->last_algorithm = MFTL_ALGO_SHIFT_SCHEME;
    return MFTL_ALGO_SHIFT_SCHEME;
  }

  double h_min = MFTLDomainGetHMin(&ens->domain);
  double w_typical = MFTLEnsembleGetTypicalRadius(ens);
  double threshold = MFTL_CENTERING_WIDTH_FACTOR * h_min;

  if (w_typical < threshold)
    ens->last_algorithm = MFTL_ALGO_CENTERING;
  else
    ens->last_algorithm = MFTL_ALGO_SHIFT_SCHEME;

  return ens->last_algorithm;
}

/* ══════════════════════════════════════════════════════════════════════
 * GID counter synchronization
 * ══════════════════════════════════════════════════════════════════════ */

void MFTLSyncEnsembleGidCounters(MFTL_Ensemble *ens)
{
  MFTL_Domain *dom = &ens->domain;
  if (dom->nproc < 2) return;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    int vals[3] = {obj->next_vertex_gid, obj->next_triangle_gid, obj->next_halfedge_gid};
    int maxv[3];
    MPI_Allreduce(vals, maxv, 3, MPI_INT, MPI_MAX, dom->comm_cart);
    obj->next_vertex_gid = maxv[0];
    obj->next_triangle_gid = maxv[1];
    obj->next_halfedge_gid = maxv[2];
  }
}

/* ══════════════════════════════════════════════════════════════════════
 * Parallel-interior gates (MFTL1 Boundary_Check parity)
 * ══════════════════════════════════════════════════════════════════════ */

// Both endpoints owned and both adjacent triangles local and ghost-free
// (MFTL1 edgesplit.c:215-219 / edgeflip.c:142-147). Rings around owned
// vertices are complete by the ghost-layer invariant, so decisions on such
// edges are exactly local.
static int mftlEdgeInterior(MFTL_Object *obj, int he_idx)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return 0;
  if (he->twin_idx == MFTL_ERROR) return 0;
  MFTL_HalfEdge *tw = &obj->half_edges[he->twin_idx];
  if (tw->flags & MFTL_FLAG_DEAD) return 0;

  if (obj->triangles[he->triangle_idx].is_ghost) return 0;
  if (obj->triangles[tw->triangle_idx].is_ghost) return 0;

  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  if (obj->vertices[he->vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[he_next].vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[he_prev].vertex_idx].is_ghost) return 0;

  int tw_next = tw->next_idx;
  int tw_prev = obj->half_edges[tw_next].next_idx;
  if (obj->vertices[tw->vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[tw_next].vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[tw_prev].vertex_idx].is_ghost) return 0;

  return 1;
}

// Triangle-locality of one triangle (all three vertices owned).
static int mftlTriFullyLocal(MFTL_Object *obj, int tri_idx)
{
  MFTL_Triangle *tri = &obj->triangles[tri_idx];
  if (tri->is_ghost) return 0;
  if (tri->flags & MFTL_FLAG_DEAD) return 0;

  MFTL_HalfEdge *he = &obj->half_edges[tri->he_idx];
  for (int k = 0; k < 3; ++k)
  {
    if (obj->vertices[he->vertex_idx].is_ghost) return 0;
    if (obj->vertices[he->vertex_idx].flags & MFTL_FLAG_DEAD) return 0;
    he = &obj->half_edges[he->next_idx];
  }
  return 1;
}

// The full edge fan (all triangles incident to either endpoint) local and
// ghost-free (MFTL1 edgecollapse.c:348-374 link-condition gate).
static int mftlFanInterior(MFTL_Object *obj, int he_idx)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return 0;
  if (he->twin_idx == MFTL_ERROR) return 0;
  MFTL_HalfEdge *tw = &obj->half_edges[he->twin_idx];
  if (tw->flags & MFTL_FLAG_DEAD) return 0;

  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v_a = obj->half_edges[he_prev].vertex_idx;
  int v_b = he->vertex_idx;

  if (obj->vertices[v_a].is_ghost || obj->vertices[v_b].is_ghost) return 0;
  if (obj->vertices[v_a].flags & MFTL_FLAG_DEAD) return 0;
  if (obj->vertices[v_b].flags & MFTL_FLAG_DEAD) return 0;

  MFTL_Triangle *ring[64];
  int n_a = MFTLGetVertexOneRing(obj, &obj->vertices[v_a], ring, 64, MFTL_FALSE);
  for (int i = 0; i < n_a; ++i)
    if (!mftlTriFullyLocal(obj, (int) (ring[i] - obj->triangles))) return 0;

  int n_b = MFTLGetVertexOneRing(obj, &obj->vertices[v_b], ring, 64, MFTL_FALSE);
  for (int i = 0; i < n_b; ++i)
    if (!mftlTriFullyLocal(obj, (int) (ring[i] - obj->triangles))) return 0;

  return 1;
}


/* ══════════════════════════════════════════════════════════════════════
 * Split phase with disjoint per-rank GID blocks
 *
 * Every rank counts its split candidates (parallel-interior gate +
 * criterion), the counts and current GID counters are Allgathered, and each
 * rank receives a disjoint GID block per object:
 *
 *   base  = max over ranks of the object's next_*_gid (all existing GIDs)
 *   start = base + prefix_sum(my candidate count) * per-split consumption
 *
 * A split consumes 1 vertex, 2 triangle and 6 half-edge GIDs, so the
 * candidate count bounds consumption exactly. The predicate is read-only
 * and deterministic, so the counting pass sees the same candidates as the
 * apply pass.
 * ══════════════════════════════════════════════════════════════════════ */
static long mftlParallelSplitPhase(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_Domain *dom = &ens->domain;
  int nobj = ens->num_objects;
  int nproc = dom->nproc;
  long splits = 0;

  // 1. Count candidates per object (serial order: highest half-edge first).
  int *cand = (int *) calloc((size_t) nobj, sizeof(int));
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int j = obj->n_half_edges - 1; j >= 0; --j)
    {
      if (!mftlEdgeInterior(obj, j)) continue;
      if (!MFTLSplitApproved(obj, j, params)) continue;
      cand[o]++;
    }
  }

  // 2. Exchange candidate counts + current counters in one Allgather.
  int *sendbuf = (int *) malloc((size_t) 4 * nobj * sizeof(int));
  int *recvbuf = (int *) malloc((size_t) 4 * nobj * nproc * sizeof(int));
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    sendbuf[4 * o + 0] = cand[o];
    sendbuf[4 * o + 1] = obj->next_vertex_gid;
    sendbuf[4 * o + 2] = obj->next_triangle_gid;
    sendbuf[4 * o + 3] = obj->next_halfedge_gid;
  }
  MPI_Allgather(sendbuf, 4 * nobj, MPI_INT, recvbuf, 4 * nobj, MPI_INT, dom->comm_cart);
  free(sendbuf);

  // 3. Assign disjoint blocks and apply.
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];

    int base_v = 0, base_t = 0, base_h = 0;
    for (int r = 0; r < nproc; ++r)
    {
      const int *rec = &recvbuf[(size_t) r * 4 * nobj + 4 * o];
      base_v = MFTL_MAX(base_v, rec[1]);
      base_t = MFTL_MAX(base_t, rec[2]);
      base_h = MFTL_MAX(base_h, rec[3]);
    }

    long off = 0;  // prefix of candidate counts over ranks < me
    for (int r = 0; r < dom->rank; ++r)
      off += recvbuf[(size_t) r * 4 * nobj + 4 * o + 0];

    // Consumption per split: 1 vertex, 2 triangles, 6 half-edges. The +1
    // keeps blocks disjoint even when a rank uses its full bound.
    obj->next_vertex_gid = base_v + (int) (off * 1) + 1;
    obj->next_triangle_gid = base_t + (int) (off * 2) + 1;
    obj->next_halfedge_gid = base_h + (int) (off * 6) + 1;

    for (int j = obj->n_half_edges - 1; j >= 0; --j)
    {
      if (!mftlEdgeInterior(obj, j)) continue;
      if (!MFTLSplitApproved(obj, j, params)) continue;
      MFTLSplitEdgeWithFit(obj, j, params);
      splits++;
    }
  }

  free(recvbuf);
  free(cand);
  return splits;
}

/* ══════════════════════════════════════════════════════════════════════
 * One parallel remesh pass on the current (redistributed) decomposition
 *
 * Precondition: every object is redistributed per the active ownership
 * (native or lb slabs) and the ghost layer covers the one-ring.
 *
 * Sequence per object: geometry refresh + feature detection ->
 * split (GID blocks) -> collapse -> flip -> tangential smoothing ->
 * VISITED cleanup -> finalize (compaction; also drops dead entities before
 * the next redistribution, which would otherwise ship them). Then the
 * Stage-09 parallel-safe smoothers and a per-pass global VolumeRecovery
 * against the pre-remesh volume, and a ghost position refresh.
 *
 * Returns the LOCAL operation count (caller reduces globally).
 * ══════════════════════════════════════════════════════════════════════ */
static long mftlParallelRemeshPass(MFTL_Ensemble *ens, MFTL_RemeshParameters *params, const double *start_volume)
{
  long ops = 0;
  int nobj = ens->num_objects;

  bool *collapses_occurred = (bool *) calloc((size_t) nobj, sizeof(bool));
  int *snap_v = (int *) malloc((size_t) nobj * sizeof(int));
  int *snap_t = (int *) malloc((size_t) nobj * sizeof(int));
  int *snap_h = (int *) malloc((size_t) nobj * sizeof(int));

  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    snap_v[o] = obj->n_vertices;
    snap_t[o] = obj->n_triangles;
    snap_h[o] = obj->n_half_edges;
  }

  // 1. Geometry refresh (identical conditions to the serial driver).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    if (obj->n_vertices == 0) continue;

    MFTLComputeTriangleProperties(obj);
    MFTLDetectFeatures(obj, params->feature_angle_threshold);
    if (params->vertex_fit != MFTL_FIT_NONE || params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE ||
        params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
    {
      MFTLComputeVertexNormals(obj);
      if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE || params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
        MFTLComputeVertexCurvatures(obj);
    }
  }

  // 2. SPLIT (collective block scheduling inside).
  ops += mftlParallelSplitPhase(ens, params);

  // 3. COLLAPSE (parallel-interior fan gate + Stage-08 approval).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int j = 0; j < obj->n_half_edges; ++j)
    {
      if (!mftlFanInterior(obj, j)) continue;
      if (!CollapseApproved(obj, j, params)) continue;
      MFTLCollapseEdgeWithFit(obj, j, params);
      collapses_occurred[o] = MFTL_TRUE;
      ops++;
    }
  }

  // 4. FLIP (parallel-interior gate + Stage-08 gate stack).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int j = 0; j < obj->n_half_edges; ++j)
    {
      MFTL_HalfEdge *he = &obj->half_edges[j];
      if (he->flags & MFTL_FLAG_DEAD) continue;
      if (he->twin_idx == MFTL_ERROR) continue;
      if (j > he->twin_idx) continue;
      if (he->flags & MFTL_FLAG_FEATURE) continue;
      if (!mftlEdgeInterior(obj, j)) continue;
      if (!FlipApprovedWithParams(obj, j, params)) continue;
      MFTLFlipEdge(obj, j);
      ops++;
    }
  }

  // 5. Tangential smoothing (ghost-safe since Stage 10; mirrors the serial
  //    driver, which runs it after the edge operators).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    if (obj->n_vertices == 0) continue;
    MFTLSmoothTangentialWithParams(obj, params);
  }

  // 6. VISITED cleanup + finalize (compaction; dead entities must not
  //    reach the next redistribution).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int j = 0; j < obj->n_half_edges; ++j)
      obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
    for (int j = 0; j < obj->n_vertices; ++j)
      obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;

    MFTLFinalizeRemeshing(obj, snap_v[o], snap_t[o], snap_h[o], collapses_occurred[o]);
  }

  // 7. Stage-09 volume-conserving stages (parallel-safe variants). No
  //    emptiness guard: MFTLComputeRoughness (inside the roughness driver)
  //    is collective and must be reached by every rank.
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    MFTLComputeTriangleProperties(obj);
    if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_KUPRAT)
      MFTLVolumeConservingSmoothingParallel(obj, params);
    else if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_TSUR)
      MFTLTSUR3DParallel(obj, params);
    if (params->roughness_smoothing) MFTLRoughnessSmoothingParallel(obj, params);
  }

  // 8. Per-pass global volume recovery against the pre-remesh volume
  //    (collective per object) + geometry refresh. Ghost positions are
  //    refreshed FIRST: owned triangles with ghost corners contribute to
  //    the volume sum, so recovery must iterate against current geometry
  //    (a ghost sync afterwards only re-stales corners by the final,
  //    converged ~1e-12 shift, which is negligible).
  MFTLUpdateGhostPositions(ens, NULL);
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    MFTLVolumeRecovery(obj, start_volume[o]);
    MFTLComputeTriangleProperties(obj);
  }

  // 9. Ghost position refresh for the next pass / decomposition.
  MFTLUpdateGhostPositions(ens, NULL);

  free(collapses_occurred);
  free(snap_v);
  free(snap_t);
  free(snap_h);
  return ops;
}

/* ══════════════════════════════════════════════════════════════════════
 * Ensemble-wide owned bounding box (Allreduce MIN/MAX over owned vertices)
 * ══════════════════════════════════════════════════════════════════════ */
static void mftlEnsembleOwnedBBox(MFTL_Ensemble *ens, double out_min[3], double out_max[3])
{
  double gmin[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double gmax[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &obj->vertices[i];
      if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD)) continue;
      for (int k = 0; k < 3; ++k)
      {
        gmin[k] = MFTL_MIN(gmin[k], v->pos[k]);
        gmax[k] = MFTL_MAX(gmax[k], v->pos[k]);
      }
    }
  }

  MPI_Allreduce(gmin, out_min, 3, MPI_DOUBLE, MPI_MIN, ens->domain.comm_cart);
  MPI_Allreduce(gmax, out_max, 3, MPI_DOUBLE, MPI_MAX, ens->domain.comm_cart);
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRedistributeEnsembleLoadBalanced
 *
 * MFTL1 MFTLRedistributeObjectLoadBalanced, ensemble-wide (MFTL2 has one
 * shared domain ownership function, so all objects balance together —
 * see the Stage-10 review notes). Computes the x-only slab decomposition
 * (domain->lb_*), activates it, redistributes, and synchronizes the GID
 * counters. Returns the number of active slabs (0 = nothing to balance).
 *
 * Requires cuts[1] == cuts[2] == 1 (x-only): adjacent slabs must be
 * comm_graph neighbours for the ghost machinery to stay valid.
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRedistributeEnsembleLoadBalanced(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_Domain *dom = &ens->domain;

  if (dom->cuts[1] != 1 || dom->cuts[2] != 1)
  {
    if (dom->rank == 0)
      fprintf(stderr,
              "MFTLRedistributeEnsembleLoadBalanced: requires an x-only decomposition (cuts = {nproc,1,1}); "
              "got {%d,%d,%d}. Load balancing skipped.\n",
              dom->cuts[0], dom->cuts[1], dom->cuts[2]);
    return 0;
  }

  // Triangle centres must be current for the slab counting.
  for (int o = 0; o < ens->num_objects; ++o)
    MFTLComputeTriangleProperties(ens->objects[o]);

  // 1. Global live triangle count.
  long local_total = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int t = 0; t < obj->n_triangles; ++t)
      if (!obj->triangles[t].is_ghost && !(obj->triangles[t].flags & MFTL_FLAG_DEAD)) local_total++;
  }
  long total = 0;
  MPI_Allreduce(&local_total, &total, 1, MPI_LONG, MPI_SUM, dom->comm_cart);

  if (total == 0)
  {
    dom->lb_nprocs = 0;  // nothing to balance: keep native ownership
    return 0;
  }

  int nproc = dom->nproc;
  int N = (params->min_triangles_per_proc > 0) ? MFTL_MAX(MFTL_MIN((int) floor((double) total / (double) params->min_triangles_per_proc), nproc), 1) : nproc;

  // 2. Active window: owned bounding box +/- reference length (MFTL1).
  double bmin[3], bmax[3];
  mftlEnsembleOwnedBBox(ens, bmin, bmax);
  double wxmin = bmin[0] - params->reference_length;
  double wxmax = bmax[0] + params->reference_length;
  if (!(wxmax > wxmin)) wxmax = wxmin + MFTL_MAX(2.0 * params->reference_length, 1.0);

  double *xcuts = (double *) malloc((size_t) (N + 1) * sizeof(double));

  if (N < 2)
  {
    // Everything to rank 0 (MFTL1 NoOfProcs < 2 branch).
    xcuts[0] = wxmin;
    xcuts[1] = wxmax;
    N = 1;
  }
  else
  {
    // 3. Iterative slab-width optimizer (MFTL1 remeshing.c:556-640).
    double *DX = (double *) malloc((size_t) N * sizeof(double));
    double *prefix = (double *) malloc((size_t) N * sizeof(double));
    int *counts = (int *) malloc((size_t) N * sizeof(int));
    int *counts_g = (int *) malloc((size_t) N * sizeof(int));

    int size_flag = 0;
    int iterations = 0;
    double balance_ratio = 0.0;

    while (!size_flag)
    {
      double width = (wxmax - wxmin) / (double) N;
      for (int i = 0; i < N; ++i) DX[i] = width;

      long target_l = (long) ceil((double) total / (double) N);
      int flag = 0;

      while (!flag && iterations < params->max_balance_iter)
      {
        double acc = 0.0;
        for (int i = 0; i < N; ++i)
        {
          acc += DX[i];
          prefix[i] = acc;
          counts[i] = 0;
        }

        for (int o = 0; o < ens->num_objects; ++o)
        {
          MFTL_Object *obj = ens->objects[o];
          for (int t = 0; t < obj->n_triangles; ++t)
          {
            MFTL_Triangle *tri = &obj->triangles[t];
            if (tri->is_ghost || (tri->flags & MFTL_FLAG_DEAD)) continue;
            double x = tri->center[0];
            for (int i = 0; i < N; ++i)
            {
              if (x < wxmin + prefix[i])
              {
                counts[i]++;
                break;
              }
            }
          }
        }

        MPI_Allreduce(counts, counts_g, N, MPI_INT, MPI_SUM, dom->comm_cart);

        balance_ratio = 0.0;
        for (int i = 0; i < N; ++i)
        {
          double ratio = (counts_g[i] > 0) ? ((double) target_l / (double) counts_g[i]) : 1.0e6;
          balance_ratio = MFTL_MAX(balance_ratio, fabs(1.0 - ratio));
        }

        if (balance_ratio < 0.2)
        {
          flag = 1;
        }
        else
        {
          for (int i = 0; i < N; ++i)
          {
            double ratio = (counts_g[i] > 0) ? ((double) target_l / (double) counts_g[i]) : 1.0e6;
            DX[i] += params->damping_factor * (DX[i] * ratio - DX[i]);
            if (DX[i] < 0.0) DX[i] = 0.0;  // monotone cut guard
          }
        }
        iterations++;
      }

      // Retry with fewer slabs when a slab is narrower than the threshold.
      int check = 0;
      for (int i = 0; i < N; ++i)
        if (DX[i] < params->size_threshold) check = 1;

      if (check && N > 1)
      {
        N--;
        DX = (double *) realloc(DX, (size_t) N * sizeof(double));
        prefix = (double *) realloc(prefix, (size_t) N * sizeof(double));
        counts = (int *) realloc(counts, (size_t) N * sizeof(int));
        counts_g = (int *) realloc(counts_g, (size_t) N * sizeof(int));
      }
      else
      {
        size_flag = 1;
        // Persist the converged widths.
        double acc = wxmin;
        for (int i = 0; i < N; ++i)
        {
          xcuts[i] = acc;
          acc += DX[i];
        }
        xcuts[N] = wxmax;
      }
    }

    free(DX);
    free(prefix);
    free(counts);
    free(counts_g);
  }

  // 5. Activate the slab ownership and redistribute.
  free(dom->lb_xcuts);
  dom->lb_xcuts = xcuts;
  dom->lb_nprocs = N;
  dom->lb_xmin = wxmin;
  dom->lb_xmax = wxmax;

  MFTLRedistributeListOfObjects(ens);
  MFTLSyncEnsembleGidCounters(ens);

  return N;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLShiftOwnershipRanges
 *
 * MFTL1 ShiftOwnershipRanges: translate the x-cuts right by
 * shift = max(0.2 * min slab width, 3 * reference_length) and redistribute,
 * so triangles that were near a slab boundary in pass 1 become interior
 * and get remeshed in pass 2. Returns MFTL_ERROR when there is nothing to
 * shift (fewer than 2 active slabs — single-rank remeshing).
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLShiftOwnershipRanges(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_Domain *dom = &ens->domain;
  int N = dom->lb_nprocs;
  if (N < 2) return MFTL_ERROR;

  double dxmin = MFTL_DBL_MAX;
  for (int i = 0; i < N; ++i) dxmin = MFTL_MIN(dxmin, dom->lb_xcuts[i + 1] - dom->lb_xcuts[i]);
  dxmin = MFTL_MAX(0.2 * dxmin, 3.0 * params->reference_length);

  // Recompute the active window from the CURRENT mesh (MFTL1 recomputes the
  // bounding box after the pass-1 remeshing).
  double bmin[3], bmax[3];
  mftlEnsembleOwnedBBox(ens, bmin, bmax);
  double wxmin = bmin[0] - params->reference_length;
  double wxmax = bmax[0] + params->reference_length;
  if (!(wxmax > wxmin)) wxmax = wxmin + MFTL_MAX(2.0 * params->reference_length, 1.0);

  double *xcuts = (double *) malloc((size_t) (N + 1) * sizeof(double));
  xcuts[0] = wxmin;
  xcuts[1] = dom->lb_xcuts[1] + dxmin;  // MFTL1: max[0] += DXmin
  for (int n = 2; n < N; ++n) xcuts[n] = xcuts[n - 1] + dxmin;
  xcuts[N] = wxmax;

  // Monotonicity guard (pathologically wide shifts collapse middle slabs).
  for (int n = 1; n < N; ++n)
  {
    if (xcuts[n] > wxmax) xcuts[n] = wxmax;
    if (xcuts[n] < xcuts[n - 1]) xcuts[n] = xcuts[n - 1];
  }

  free(dom->lb_xcuts);
  dom->lb_xcuts = xcuts;
  dom->lb_xmin = wxmin;
  dom->lb_xmax = wxmax;
  // lb_nprocs unchanged: same number of slabs, shifted cuts.

  MFTLRedistributeListOfObjects(ens);
  MFTLSyncEnsembleGidCounters(ens);

  return MFTL_SUCCESS;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRemeshEnsembleShiftScheme
 *
 * MFTL1 MFTL_Remeshing (LoadBalance = YES), adapted to MFTL2:
 *
 *   per iteration:
 *     pass 1: x-only load-balanced slabs -> redistribute -> extend ghosts
 *             -> split/collapse/flip (parallel-interior gates) -> Stage-09
 *             smoothers -> VolumeRecovery -> ghost refresh
 *     pass 2: shift the slab cuts -> same pipeline (boundary coverage)
 *   convergence: global op count < 10 (serial semantics), failsafe 15
 *   iterations for run-to-convergence mode
 *
 * Afterwards: restore the native decomposition, final redistribution, GID
 * counter sync, and the MFTL1 end-of-remesh domain clamp (non-periodic
 * axes).
 *
 * On a non-x-only decomposition the shift scheme is unavailable (slab
 * owners would not be comm_graph neighbours); the driver then remeshes on
 * the native decomposition with parallel-interior gates only — correct,
 * but triangles touching a partition boundary are not remeshed (the MFTL1
 * LoadBalance = NO behaviour).
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRemeshEnsembleShiftScheme(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_Domain *dom = &ens->domain;
  int nobj = ens->num_objects;
  int x_only = (dom->cuts[1] == 1 && dom->cuts[2] == 1);

  // Snapshot pre-remesh volumes (collective per object).
  double *start_volume = (double *) malloc((size_t) nobj * sizeof(double));
  for (int o = 0; o < nobj; ++o)
    start_volume[o] = MFTLObjectVolume(ens->objects[o]);

  int do_two_pass = (params->load_balance != 0) && x_only && dom->nproc > 1;

  if (!x_only && dom->rank == 0 && dom->nproc > 1)
  {
    printf("MFTLRemeshEnsembleShiftScheme: non-x-only decomposition {%d,%d,%d}; "
           "using interior-only remeshing without load balancing or boundary shift "
           "(create the ensemble with cuts = {nproc,1,1} for MFTL1-style load balancing).\n",
           dom->cuts[0], dom->cuts[1], dom->cuts[2]);
  }

  int iter = 0;
  long total_ops = 0;

  while (params->iterations == MFTL_ERROR || iter < params->iterations)
  {
    long local_ops = 0;

    if (do_two_pass)
    {
      // Pass 1: load-balanced slabs. NOTE: no MFTLExtendGhostLayer here —
      // redistribution already gives every owned vertex its complete
      // one-ring (all triangles incident to an owned vertex ship to their
      // owner), which is exactly MFTL1's MFTL_AssureOneRingNeighbourhood
      // requirement. Extending to a 2-hop ring would create ghosts whose
      // owners sit two slabs away on narrow decompositions (the expansion
      // reaches ~2 edge-lengths), breaking the graph-neighbour invariant
      // of the ghost machinery.
      MFTLRedistributeEnsembleLoadBalanced(ens, params);
      local_ops += mftlParallelRemeshPass(ens, params, start_volume);

      // Pass 2: shifted cuts so former boundary triangles become interior.
      if (MFTLShiftOwnershipRanges(ens, params) != MFTL_ERROR)
      {
        local_ops += mftlParallelRemeshPass(ens, params, start_volume);
      }
    }
    else
    {
      // Native decomposition, interior-only (no load balance, no shift).
      dom->lb_nprocs = 0;
      MFTLRedistributeListOfObjects(ens);
      MFTLSyncEnsembleGidCounters(ens);
      local_ops += mftlParallelRemeshPass(ens, params, start_volume);
    }

    MPI_Allreduce(&local_ops, &total_ops, 1, MPI_LONG, MPI_SUM, dom->comm_cart);
    if (total_ops < 10) break;
    if (iter >= 15 && params->iterations == MFTL_ERROR) break;
    iter++;
  }

  // Split cleanup helper: the last passes can leave interior edges above
  // max_edge_length (flip diagonals carry no length gate, and the collapse
  // fit can overshoot its clamp on coarse pockets). The split criterion
  // approves exactly those edges, so a few bounded cleanup passes enforce
  // the invariant "no interior edge longer than max".
  // An edge can be boundary-edged in BOTH lb passes (its band never moved)
  // yet interior under the native cuts — so the cleanup runs twice: under
  // the lb decomposition and again after restoring the native one.
#define MFTL_SPLIT_CLEANUP()                                          \
  do                                                                  \
  {                                                                   \
    int *snap_v = (int *) malloc((size_t) nobj * sizeof(int));        \
    int *snap_t = (int *) malloc((size_t) nobj * sizeof(int));        \
    int *snap_h = (int *) malloc((size_t) nobj * sizeof(int));        \
    for (int clean = 0; clean < 4; ++clean)                           \
    {                                                                 \
      for (int o = 0; o < nobj; ++o)                                  \
      {                                                               \
        snap_v[o] = ens->objects[o]->n_vertices;                      \
        snap_t[o] = ens->objects[o]->n_triangles;                     \
        snap_h[o] = ens->objects[o]->n_half_edges;                    \
      }                                                               \
      long local_fix = mftlParallelSplitPhase(ens, params);           \
      long total_fix = 0;                                             \
      MPI_Allreduce(&local_fix, &total_fix, 1, MPI_LONG, MPI_SUM, dom->comm_cart); \
      for (int o = 0; o < nobj; ++o)                                  \
      {                                                               \
        MFTL_Object *obj = ens->objects[o];                           \
        for (int j = 0; j < obj->n_half_edges; ++j)                   \
          obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;             \
        for (int j = 0; j < obj->n_vertices; ++j)                     \
          obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;               \
        MFTLFinalizeRemeshing(obj, snap_v[o], snap_t[o], snap_h[o], MFTL_FALSE); \
        MFTLComputeTriangleProperties(obj);                           \
      }                                                               \
      if (total_fix == 0) break;                                      \
    }                                                                 \
    free(snap_v);                                                     \
    free(snap_t);                                                     \
    free(snap_h);                                                     \
  } while (0)

  MFTL_SPLIT_CLEANUP();

  // Restore the native decomposition and re-sync everything.
  dom->lb_nprocs = 0;
  MFTLRedistributeListOfObjects(ens);
  MFTLSyncEnsembleGidCounters(ens);
  MFTLUpdateGhostPositions(ens, NULL);
  MFTL_SPLIT_CLEANUP();

  // MFTL1 end-of-remesh clamp into the domain box (non-periodic axes only;
  // periodic axes are wrapped by the advection instead).
  for (int o = 0; o < nobj; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &obj->vertices[i];
      if (v->is_ghost) continue;
      for (int k = 0; k < 3; ++k)
      {
        if (dom->is_periodic[k]) continue;
        v->pos[k] = MFTL_MAX(dom->global_min[k] + MFTL_VERYSMALL, MFTL_MIN(v->pos[k], dom->global_max[k] - MFTL_VERYSMALL));
      }
    }
  }

  for (int o = 0; o < nobj; ++o)
    MFTLComputeTriangleProperties(ens->objects[o]);
  MFTLUpdateGhostPositions(ens, NULL);

  // Final volume recovery in the RESTORED decomposition, as recover->sync
  // cycles: a ghost refresh after a recovery re-stales ghost corners by
  // that recovery's total shift, partially undoing the correction (the
  // local response is corner-weighted; the post-sync response involves the
  // neighbours' shifts too). Each cycle removes the previous jump
  // geometrically, so a few cheap cycles converge the post-sync volume to
  // the target; converged cycles exit on their first check.
  for (int cycle = 0; cycle < 5; ++cycle)
  {
    MFTLUpdateGhostPositions(ens, NULL);
    for (int o = 0; o < nobj; ++o)
    {
      MFTL_Object *obj = ens->objects[o];
      MFTLVolumeRecovery(obj, start_volume[o]);
      MFTLComputeTriangleProperties(obj);
    }
  }
  MFTLUpdateGhostPositions(ens, NULL);

  free(start_volume);
  return MFTL_SUCCESS;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRemeshEnsembleDispersedCentering
 *
 * Gather the whole ensemble to rank 0 through a temporary single-partition
 * domain, run the ungated serial remesh kernel there (ghost-free, exactly
 * the serial workflow), then restore the native domain and redistribute.
 * All ranks execute both redistributions and the serial kernel (whose
 * collectives run on the temporary communicator) — empty objects on
 * non-zero ranks are collective-safe.
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRemeshEnsembleDispersedCentering(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_Domain *dom = &ens->domain;

  MFTL_Domain saved = *dom;

  MFTL_Domain tmp = *dom;
  tmp.cuts[0] = 1;
  tmp.cuts[1] = 1;
  tmp.cuts[2] = 1;
  for (int k = 0; k < 3; ++k) tmp.grid_factors[k] = 1.0 / tmp.domain_size[k];

  tmp.rank_map = (int *) malloc(sizeof(int));
  tmp.rank_map[0] = 0;

  int periods[3] = {(int) tmp.is_periodic[0], (int) tmp.is_periodic[1], (int) tmp.is_periodic[2]};
  MPI_Cart_create(tmp.comm_world, 3, tmp.cuts, periods, 0, &tmp.comm_cart);
  tmp.num_neighbors = 0;
  tmp.neighbors = NULL;
  MPI_Dist_graph_create_adjacent(tmp.comm_world, 0, NULL, MPI_UNWEIGHTED, 0, NULL, MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &tmp.comm_graph);

  // Objects reference &ens->domain; swap in the single-partition domain.
  *dom = tmp;

  MFTLRedistributeListOfObjects(ens);  // everything -> rank 0
  MFTLRemeshEnsemble(ens, params);     // serial kernel (collective-safe on empty objects)

  // Restore the native domain and redistribute back.
  *dom = saved;
  free(tmp.rank_map);
  MPI_Comm_free(&tmp.comm_cart);
  MPI_Comm_free(&tmp.comm_graph);

  MFTLRedistributeListOfObjects(ens);
  MFTLSyncEnsembleGidCounters(ens);

  return MFTL_SUCCESS;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRemeshEnsembleParallel
 *
 * Top-level parallel remesh dispatcher (collective on comm_cart):
 *   nproc == 1  -> serial kernel
 *   CENTERING   -> gather to rank 0 + serial kernel + scatter (small,
 *                  dispersed objects; works on any decomposition)
 *   SHIFT_SCHEME-> MFTL1 x-only load-balanced two-pass remeshing (full
 *                  behaviour on x-only decompositions, interior-only
 *                  fallback otherwise)
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(params != MFTL_NULL);

  if (ens->num_objects == 0) return MFTL_SUCCESS;

  MFTL_Domain *dom = &ens->domain;
  if (dom->nproc < 2)
  {
    MFTLRemeshEnsemble(ens, params);
    return MFTL_SUCCESS;
  }

  MFTL_RemeshAlgorithm algo = MFTLSelectRemeshAlgorithm(ens, params);

  switch (algo)
  {
    case MFTL_ALGO_CENTERING:
      return MFTLRemeshEnsembleDispersedCentering(ens, params);

    case MFTL_ALGO_SHIFT_SCHEME:
      return MFTLRemeshEnsembleShiftScheme(ens, params);

    default:
      fprintf(stderr, "MFTLRemeshEnsembleParallel: unknown algorithm %d\n", (int) algo);
      return MFTL_ERROR;
  }
}

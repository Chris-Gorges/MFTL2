/**
 * @file remesh_project.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_remeshing
 *
 * @brief Back-projection of smoothed vertices onto the Phong surface of the
 *        mesh a remeshing kernel call received (#MFTL_REMESH_PROJECT_PHONG).
 *
 * ## Why the reference is the call's input, and why it is curved
 *
 * Botsch & Kobbelt project smoothed vertices back onto the ORIGINAL surface,
 * kept as a mesh. In a graphics remesher that is a fixed, usually finer mesh. In
 * front tracking it is the previous step's mesh at the same resolution, and its
 * flat triangles are chords: every call puts the smoothed vertices on them and
 * the next call starts from there (the LeVeque period lost 12.5 % of its volume
 * at refinement 4 that way, against 3.2 % without projection). Its Phong
 * surface passes through the same vertices with their normals and is fourth-order
 * accurate on a sphere, so the projection re-lifts what this call's collapses
 * and flips cut off instead (docs/analysis.md §14.9). A reference rebuilt from the
 * CURRENT mesh does not: it has no record of what the call removed.
 *
 * ## Finding the closest triangle without a search structure
 *
 * A kernel call does not renumber vertex global ids, so a vertex the call
 * received is found in the copy by binary search on its id; a vertex a split
 * created is found through a neighbour. From there a walk over the copy's edge
 * neighbours reaches the triangle whose closest point is nearest, usually in
 * zero or one step, and each sweep after the first starts from the triangle the
 * previous one found. Neither needs a grid, and both are local, so the pass
 * needs no communication: a moved vertex is unfrozen, its fan is owned, and
 * ghost corners of a reference triangle contribute no lift.
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_vectoroperations.h"
#include "remesh_internal.h"

/** @brief Steps of the walk to the closest reference triangle. */
#define MFTL_REF_WALK_MAX 32

/** @brief Triangles of one reference vertex fan the walk examines at most. */
#define MFTL_REF_FAN_MAX 64

void MFTLRemeshReferenceBuild(MFTL_RemeshReference *ref, MFTL_Object *obj)
{
  MFTL_ASSERT(ref != MFTL_NULL && obj != MFTL_NULL);
  memset(ref, 0, sizeof(*ref));
  ref->domain = obj->domain;

  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  const int n_v = obj->verts.n;
  const int n_t = obj->tris.n;
  ref->n_v = n_v;
  ref->n_t = n_t;

  ref->pos = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_v, 1) * 3 * sizeof(double), "remesh reference positions");
  ref->nrm = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_v, 1) * 3 * sizeof(double), "remesh reference normals");
  ref->tv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_t, 1) * 3 * sizeof(int), "remesh reference corners");
  ref->tn = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_t, 1) * 3 * sizeof(int), "remesh reference neighbours");
  ref->vtri = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_v, 1) * sizeof(int), "remesh reference vertex triangles");
  ref->gid = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_v, 1) * sizeof(int), "remesh reference ids");
  ref->gid_v = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_v, 1) * sizeof(int), "remesh reference id vertices");
  ref->bytes = (double) n_v * (6 * sizeof(double) + 3 * sizeof(int)) + (double) n_t * 6 * sizeof(int);

  /* Angle weights. Max's weights are exact on a sphere and make the lift
   * fourth-order on an IRREGULAR sphere mesh, where angle weights leave it
   * second-order (test_projected_smoothing_stays_on_the_sphere measures both
   * orders' constants), but over the LeVeque period they were worse: d_rms
   * 0.194 against 0.136 % R0 and E_H 0.29 against 0.15 at refinement 4,
   * equal at 5. They weight short edges heavily, which a graded,
   * stretched mesh has plenty of (docs/analysis.md §14.10). */
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE);
  memcpy(ref->pos, obj->verts.pos, (size_t) n_v * 3 * sizeof(double));
  memcpy(ref->nrm, obj->verts.normal, (size_t) n_v * 3 * sizeof(double));
  for (int v = 0; v < n_v; ++v)
  {
    /* A crease has no normal, a rim vertex's fan is a half, a ghost's is
     * partial: such a corner gets no lift, so Phong degrades to the flat
     * triangle towards it rather than bending with a wrong normal. */
    const int f = obj->verts.rec[v].flags;
    const bool no_lift = (f & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER | MFTL_FLAG_BOUNDARY | MFTL_FLAG_DEAD)) || v >= obj->verts.n_local;
    if (no_lift)
      for (int k = 0; k < 3; ++k) ref->nrm[3 * v + k] = 0.0;
    ref->vtri[v] = -1;
    ref->gid[v] = obj->verts.rec[v].global_id;
    ref->gid_v[v] = v;
  }

  for (int t = 0; t < n_t; ++t)
  {
    const bool dead = (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) != 0;
    for (int k = 0; k < 3; ++k)
    {
      ref->tv[3 * t + k] = dead ? -1 : obj->tris.verts[3 * t + k];
      ref->tn[3 * t + k] = dead ? MFTL_ERROR : obj->tris.neighbors[3 * t + k];
      if (!dead) ref->vtri[obj->tris.verts[3 * t + k]] = t;
    }
  }

  MFTLSortPairs32(ref->gid, ref->gid_v, n_v);
}

void MFTLRemeshReferenceFree(MFTL_RemeshReference *ref)
{
  if (ref == MFTL_NULL) return;
  free(ref->pos);
  free(ref->nrm);
  free(ref->tv);
  free(ref->tn);
  free(ref->vtri);
  free(ref->gid);
  free(ref->gid_v);
  memset(ref, 0, sizeof(*ref));
}

/**
 * @brief The reference copy of the vertex with global id @p gid, or -1.
 * @param ref The reference.
 * @param gid A vertex global id.
 * @return Its reference index, or -1 if the call created it.
 */
static inline int ref_vertex_of_gid(const MFTL_RemeshReference *ref, int gid)
{
  const int i = MFTLSortedIndexOf(ref->gid, ref->n_v, gid);
  return (i < 0) ? -1 : ref->gid_v[i];
}

/**
 * @brief Closest point of reference triangle @p t to @p q, in coordinates
 *        relative to @p q (minimum image per corner, so seam-safe).
 * @param      ref     The reference.
 * @param      t       A live reference triangle.
 * @param      q       The query point.
 * @param[out] rel     The corners relative to @p q.
 * @param[out] foot    The closest point relative to @p q.
 * @param[out] feature Its feature.
 * @return The squared distance.
 */
static double ref_closest(const MFTL_RemeshReference *ref, int t, const double q[3], double rel[3][3], double foot[3], MFTL_TriangleFeature *feature)
{
  for (int k = 0; k < 3; ++k) MFTLGetMinimumImageVector(ref->domain, rel[k], q, &ref->pos[3 * ref->tv[3 * t + k]]);
  const double origin[3] = {0.0, 0.0, 0.0};
  return MFTLClosestPointOnTriangle(origin, rel, foot, feature);
}

/** @brief The best reference triangle seen so far during a walk. */
typedef struct
{
  int t;  ///< The triangle.
  double d_sq;  ///< Squared distance.
  double rel[3][3];  ///< Its corners relative to the query.
  double foot[3];  ///< Its closest point relative to the query.
  MFTL_TriangleFeature feature;  ///< Which part of it is closest.
} walk_best;

/**
 * @brief Tries triangle @p t; keeps it in @p best if it is closer.
 * @param         ref  The reference.
 * @param         t    Candidate, may be MFTL_ERROR or dead.
 * @param         q    The query point.
 * @param[in,out] best The best so far.
 * @return `true` if @p t became the best.
 */
static bool walk_try(const MFTL_RemeshReference *ref, int t, const double q[3], walk_best *best)
{
  if (t < 0 || ref->tv[3 * t] < 0) return MFTL_FALSE;
  walk_best c;
  c.t = t;
  c.d_sq = ref_closest(ref, t, q, c.rel, c.foot, &c.feature);
  if (!(c.d_sq < best->d_sq)) return MFTL_FALSE;
  *best = c;
  return MFTL_TRUE;
}

/**
 * @brief Corner index of reference vertex @p r in triangle @p t, or -1.
 * @param ref The reference.
 * @param t   Triangle.
 * @param r   Vertex.
 * @return 0, 1, 2 or -1.
 */
static inline int corner_of(const MFTL_RemeshReference *ref, int t, int r)
{
  for (int k = 0; k < 3; ++k)
    if (ref->tv[3 * t + k] == r) return k;
  return -1;
}

/**
 * @brief Examines every triangle around reference vertex @p r, starting from
 *        @p t0 and turning both ways until the fan closes or ends.
 * @param         ref  The reference.
 * @param         t0   A triangle with corner @p r.
 * @param         r    The vertex.
 * @param         q    The query point.
 * @param[in,out] best The best so far.
 */
static void walk_fan(const MFTL_RemeshReference *ref, int t0, int r, const double q[3], walk_best *best)
{
  for (int dir = 0; dir < 2; ++dir)
  {
    int t = t0;
    bool closed = MFTL_FALSE;
    for (int s = 0; s < MFTL_REF_FAN_MAX; ++s)
    {
      const int k = corner_of(ref, t, r);
      if (k < 0) break;
      /* dir 0: across the edge ending at r; dir 1: across the edge starting at it. */
      const int next = ref->tn[3 * t + (dir == 0 ? (k + 2) % 3 : k)];
      if (next == t0) closed = MFTL_TRUE;
      if (next < 0 || next == t0) break;
      walk_try(ref, next, q, best);
      t = next;
    }
    if (closed) break;  // one turn saw the whole fan; an open one needs the other way too
  }
}

int MFTLRemeshReferenceProject(const MFTL_RemeshReference *ref, const MFTL_Object *obj, int v, int *hint, double max_d_sq, double p[3])
{
  /* ── Where to start ─────────────────────────────────────────────────── */
  int start = *hint;
  if (start < 0 || start >= ref->n_t || ref->tv[3 * start] < 0)
  {
    start = -1;
    const int r = ref_vertex_of_gid(ref, obj->verts.rec[v].global_id);
    if (r >= 0) start = ref->vtri[r];
    if (start < 0)
    {
      /* Created by this call (a split): through a neighbour the call received. */
      int ring[MFTL_REMESH_RING_MAX];
      const int n = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_REMESH_RING_MAX, MFTL_FALSE);
      for (int i = 0; i < n && start < 0; ++i)
      {
        const int rw = ref_vertex_of_gid(ref, obj->verts.rec[ring[i]].global_id);
        if (rw >= 0) start = ref->vtri[rw];
      }
    }
    if (start < 0) return MFTL_ERROR;
  }

  /* ── Walk to the closest triangle ───────────────────────────────────── */
  walk_best best;
  best.t = start;
  best.d_sq = ref_closest(ref, start, p, best.rel, best.foot, &best.feature);

  for (int step = 0; step < MFTL_REF_WALK_MAX && best.feature != MFTL_TRI_FEATURE_FACE; ++step)
  {
    const int t = best.t;
    bool moved = MFTL_FALSE;
    if (best.feature >= MFTL_TRI_FEATURE_EDGE0)
    {
      moved = walk_try(ref, ref->tn[3 * t + (best.feature - MFTL_TRI_FEATURE_EDGE0)], p, &best);
    }
    else
    {
      /* Closest to a corner: the two triangles across its edges first, then,
       * if neither is closer, the whole fan -- the corner may be the answer. */
      const int k = best.feature - MFTL_TRI_FEATURE_VERTEX0;
      const bool m1 = walk_try(ref, ref->tn[3 * t + k], p, &best);
      const bool m2 = walk_try(ref, ref->tn[3 * t + (k + 2) % 3], p, &best);
      moved = m1 || m2;
      if (!moved)
      {
        walk_fan(ref, t, ref->tv[3 * t + k], p, &best);
        break;
      }
    }
    if (!moved) break;
  }

  if (!(best.d_sq <= max_d_sq)) return MFTL_ERROR;

  /* ── The Phong lift over the closest point ──────────────────────────── */
  const double(*x)[3] = best.rel;
  double e1[3], e2[3], r[3];
  for (int c = 0; c < 3; ++c)
  {
    e1[c] = x[1][c] - x[0][c];
    e2[c] = x[2][c] - x[0][c];
    r[c] = best.foot[c] - x[0][c];
  }
  const double d11 = MFTLVectorInProd(e1, e1), d12 = MFTLVectorInProd(e1, e2), d22 = MFTLVectorInProd(e2, e2);
  const double r1 = MFTLVectorInProd(r, e1), r2 = MFTLVectorInProd(r, e2);
  const double den = d11 * d22 - d12 * d12;
  if (!(den > 0.0)) return MFTL_ERROR;

  const double u1 = (d22 * r1 - d12 * r2) / den;
  const double u2 = (d11 * r2 - d12 * r1) / den;
  const double u[3] = {1.0 - u1 - u2, u1, u2};

  double f[3] = {best.foot[0], best.foot[1], best.foot[2]};
  for (int i = 0; i < 3; ++i)
  {
    const double *n = &ref->nrm[3 * ref->tv[3 * best.t + i]];
    const double hgt = (best.foot[0] - x[i][0]) * n[0] + (best.foot[1] - x[i][1]) * n[1] + (best.foot[2] - x[i][2]) * n[2];
    for (int c = 0; c < 3; ++c) f[c] -= 0.5 * u[i] * hgt * n[c];
  }

  for (int c = 0; c < 3; ++c) p[c] += f[c];
  MFTLApplyPeriodicity(ref->domain, p);
  *hint = best.t;
  return MFTL_SUCCESS;
}

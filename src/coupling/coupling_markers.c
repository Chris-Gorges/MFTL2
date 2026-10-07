/**
 * @file coupling_markers.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief Markers from the surface, the surface-generation snapshot a binding
 *        is checked against, and the local half of the sampling check.
 *
 * A vertex marker's weight is its dual area \f$\sum_{T\ni v} A_T/3\f$, summed
 * over EVERY triangle in the pool, ghosts included: the canonical halo is
 * vertex-complete, so an owned vertex sees all of its fan and the weights of
 * all owned vertices sum to the owned area exactly (each triangle's area is
 * split three ways, every corner counted by its owner). Positions are copied
 * and wrapped; nothing in the pool is written.
 */

#include "coupling_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

uint64_t MFTLCplPositionsChecksum(const double *pos, int64_t n) { return cpl_fnv(CPL_FNV_INIT, pos, 3 * (size_t) n * sizeof(double)); }

static void snapshot_bodies(const MFTL_Ensemble *ens, cpl_markers *m)
{
  m->n_bodies = ens->n_local_objects;
  m->bodies = (cpl_body_snap *) MFTLMallocOrDie((size_t) MFTL_MAX(m->n_bodies, 1) * sizeof(cpl_body_snap), "marker body snapshot");
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    m->bodies[o].gid = obj->global_id;
    m->bodies[o].generation = obj->generation;
    m->bodies[o].n_local_v = obj->verts.n_local;
    m->bodies[o].n_local_t = obj->tris.n_local;
  }
}

bool MFTLCplBodiesCurrent(const MFTL_Ensemble *ens, const cpl_body_snap *b, int n)
{
  if (ens->n_local_objects != n) return false;
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    if (obj->global_id != b[o].gid || obj->generation != b[o].generation || obj->verts.n_local != b[o].n_local_v || obj->tris.n_local != b[o].n_local_t)
      return false;
  }
  return true;
}

int MFTLCplMarkersGather(const MFTL_Ensemble *ens, const MFTL_MarkerSpec *spec, cpl_markers *m)
{
  memset(m, 0, sizeof *m);
  if (spec == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const MFTL_Domain *dom = &ens->domain;

  if (spec->kind == MFTL_MARKERS_USER)
  {
    if (spec->n_user < 0 || (spec->n_user > 0 && spec->user_pos == MFTL_NULL)) return MFTL_COUPLING_INVALID;
    m->n = spec->n_user;
    m->pos = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(m->n, 1) * sizeof(double), "markers");
    m->ref = (MFTL_MarkerRef *) MFTLMallocOrDie((size_t) MFTL_MAX(m->n, 1) * sizeof(MFTL_MarkerRef), "markers");
    int bad = 0;
    for (int64_t i = 0; i < m->n; ++i)
    {
      double x[3] = {spec->user_pos[3 * i], spec->user_pos[3 * i + 1], spec->user_pos[3 * i + 2]};
      if (!isfinite(x[0]) || !isfinite(x[1]) || !isfinite(x[2])) bad = 1;
      MFTLApplyPeriodicity(dom, x);
      memcpy(&m->pos[3 * i], x, sizeof x);
      m->ref[i].key = spec->user_keys ? spec->user_keys[i] : (uint64_t) i;
      m->ref[i].body_gid = -1;
      m->ref[i].slot = -1;
      m->ref[i].entity = (int) i;
      m->ref[i].weight = spec->user_weight ? spec->user_weight[i] : 1.0;
    }
    m->n_bodies = 0;
    m->bodies = MFTL_NULL;
    return bad ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK;
  }

  if (spec->kind != MFTL_MARKERS_VERTICES && spec->kind != MFTL_MARKERS_CENTROIDS) return MFTL_COUPLING_INVALID;

  int64_t total = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    total += (spec->kind == MFTL_MARKERS_VERTICES) ? obj->verts.n_local : obj->tris.n_local;
  }
  m->n = total;
  m->pos = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "markers");
  m->ref = (MFTL_MarkerRef *) MFTLMallocOrDie((size_t) MFTL_MAX(total, 1) * sizeof(MFTL_MarkerRef), "markers");
  snapshot_bodies(ens, m);

  int64_t at = 0;
  double *dual = MFTL_NULL;
  int dual_cap = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    const uint64_t body = (uint64_t) (uint32_t) obj->global_id << 32;
    if (spec->kind == MFTL_MARKERS_VERTICES)
    {
      const int nl = obj->verts.n_local;
      if (nl > dual_cap)
      {
        dual_cap = MFTLGrowCapacity(dual_cap, nl);
        dual = (double *) MFTLReallocOrDie(dual, (size_t) dual_cap * sizeof(double), "dual areas");
      }
      for (int v = 0; v < nl; ++v) dual[v] = 0.0;
      for (int t = 0; t < obj->tris.n; ++t)
      {
        int vv[3];
        MFTLGetTriangleVertices(obj, t, vv);
        double area = 0.0;
        MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, MFTL_NULL);
        const double third = area * MFTL_ONE_THIRD;
        for (int c = 0; c < 3; ++c)
          if (vv[c] >= 0 && vv[c] < nl) dual[vv[c]] += third;
      }
      for (int v = 0; v < nl; ++v, ++at)
      {
        double x[3] = {obj->verts.pos[3 * v], obj->verts.pos[3 * v + 1], obj->verts.pos[3 * v + 2]};
        MFTLApplyPeriodicity(dom, x);
        memcpy(&m->pos[3 * at], x, sizeof x);
        m->ref[at].key = body | (uint64_t) (uint32_t) obj->verts.rec[v].global_id;
        m->ref[at].body_gid = obj->global_id;
        m->ref[at].slot = o;
        m->ref[at].entity = v;
        m->ref[at].weight = dual[v];
      }
    }
    else
    {
      for (int t = 0; t < obj->tris.n_local; ++t, ++at)
      {
        double area = 0.0, c[3];
        MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, c);
        memcpy(&m->pos[3 * at], c, sizeof c);
        m->ref[at].key = body | (uint64_t) (uint32_t) obj->tris.rec[t].global_id;
        m->ref[at].body_gid = obj->global_id;
        m->ref[at].slot = o;
        m->ref[at].entity = t;
        m->ref[at].weight = area;
      }
    }
  }
  free(dual);
  return MFTL_COUPLING_OK;
}

void MFTLCplMarkersFree(cpl_markers *m)
{
  free(m->pos);
  free(m->ref);
  free(m->bodies);
  memset(m, 0, sizeof *m);
}

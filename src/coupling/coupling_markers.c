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
#include <stdio.h>
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
  m->bodies = (cpl_body_snap *) MFTLCplAlloc((size_t) MFTL_MAX(m->n_bodies, 1) * sizeof(cpl_body_snap), "marker body snapshot");
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

void MFTLCplQuadraturePoint(int m, int q, double *l1, double *l2)
{
  /* Row j of the subdivision holds m - j upward sub-triangles and m - j - 1
   * downward ones, interleaved: up(0,j), down(0,j), up(1,j), ..., up(m-1-j,j).
   * An upward one (i,j) has corners (i,j), (i+1,j), (i,j+1) in units of 1/m of
   * (lambda1, lambda2); a downward one (i+1,j), (i,j+1), (i+1,j+1). */
  int j = 0, base = 0;
  while (q >= base + 2 * (m - j) - 1)
  {
    base += 2 * (m - j) - 1;
    ++j;
  }
  const int r = q - base, i = r / 2;
  const double third = (r % 2 == 0) ? 1.0 / 3.0 : 2.0 / 3.0;
  *l1 = ((double) i + third) / (double) m;
  *l2 = ((double) j + third) / (double) m;
}

int MFTLCplMarkersGather(const MFTL_Ensemble *ens, const MFTL_MarkerSpec *spec, const int *levels, cpl_markers *m)
{
  memset(m, 0, sizeof *m);
  if (spec == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const MFTL_Domain *dom = &ens->domain;

  if (spec->kind == MFTL_MARKERS_USER)
  {
    if (spec->n_user < 0 || (spec->n_user > 0 && spec->user_pos == MFTL_NULL)) return MFTL_COUPLING_INVALID;
    m->n = spec->n_user;
    m->pos = (double *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(m->n, 1) * sizeof(double), "markers");
    m->ref = (MFTL_MarkerRef *) MFTLCplAlloc((size_t) MFTL_MAX(m->n, 1) * sizeof(MFTL_MarkerRef), "markers");
    int bad = 0;
    for (int64_t i = 0; i < m->n; ++i)
    {
      double x[3] = {spec->user_pos[3 * i], spec->user_pos[3 * i + 1], spec->user_pos[3 * i + 2]};
      if (!isfinite(x[0]) || !isfinite(x[1]) || !isfinite(x[2])) bad = 1;
      cpl_wrap(dom, x);
      memcpy(&m->pos[3 * i], x, sizeof x);
      m->ref[i].key = spec->user_keys ? spec->user_keys[i] : (uint64_t) i;
      m->ref[i].body_gid = -1;
      m->ref[i].slot = -1;
      m->ref[i].entity = (int) i;
      m->ref[i].sub = 0;
      m->ref[i].weight = spec->user_weight ? spec->user_weight[i] : 1.0;
    }
    m->n_bodies = 0;
    m->bodies = MFTL_NULL;
    return bad ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK;
  }

  if (spec->kind != MFTL_MARKERS_VERTICES && spec->kind != MFTL_MARKERS_CENTROIDS && spec->kind != MFTL_MARKERS_QUADRATURE) return MFTL_COUPLING_INVALID;
  if (spec->kind == MFTL_MARKERS_QUADRATURE && levels == MFTL_NULL) return MFTL_COUPLING_INVALID;

  int64_t total = 0;
  {
    int64_t tri = 0;
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      const MFTL_Object *obj = ens->local_objects[o];
      if (spec->kind == MFTL_MARKERS_QUADRATURE)
        for (int t = 0; t < obj->tris.n_local; ++t, ++tri) total += (int64_t) levels[tri] * levels[tri];
      else
        total += (spec->kind == MFTL_MARKERS_VERTICES) ? obj->verts.n_local : obj->tris.n_local;
    }
  }
  m->n = total;
  m->pos = (double *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "markers");
  m->ref = (MFTL_MarkerRef *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(MFTL_MarkerRef), "markers");
  if (spec->kind == MFTL_MARKERS_QUADRATURE) m->bary = (double *) MFTLCplAlloc(2 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "markers");
  snapshot_bodies(ens, m);

  int64_t at = 0, tri_seen = 0;
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
        dual = (double *) MFTLCplRealloc(dual, (size_t) dual_cap * sizeof(double), "dual areas");
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
        cpl_wrap(dom, x);
        memcpy(&m->pos[3 * at], x, sizeof x);
        m->ref[at].key = body | (uint64_t) (uint32_t) obj->verts.rec[v].global_id;
        m->ref[at].body_gid = obj->global_id;
        m->ref[at].slot = o;
        m->ref[at].entity = v;
        m->ref[at].sub = 0;
        m->ref[at].weight = dual[v];
      }
    }
    else if (spec->kind == MFTL_MARKERS_QUADRATURE)
    {
      for (int t = 0; t < obj->tris.n_local; ++t)
      {
        const int lv = levels[tri_seen++];
        double area = 0.0, cr[3][3];
        MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, MFTL_NULL);
        MFTLGetTriangleCorners(obj, t, cr);
        const double w = area / ((double) lv * (double) lv);
        if (!(w > 0.0) || !isfinite(w)) return MFTL_COUPLING_INVALID;
        for (int q = 0; q < lv * lv; ++q, ++at)
        {
          double l1, l2, x[3];
          MFTLCplQuadraturePoint(lv, q, &l1, &l2);
          const double l0 = 1.0 - l1 - l2;
          for (int k = 0; k < 3; ++k) x[k] = l0 * cr[0][k] + l1 * cr[1][k] + l2 * cr[2][k];
          cpl_wrap(dom, x);
          memcpy(&m->pos[3 * at], x, sizeof x);
          m->bary[2 * at] = l1;
          m->bary[2 * at + 1] = l2;
          m->ref[at].key = body | (uint64_t) (uint32_t) obj->tris.rec[t].global_id;
          m->ref[at].body_gid = obj->global_id;
          m->ref[at].slot = o;
          m->ref[at].entity = t;
          m->ref[at].sub = q;
          m->ref[at].weight = w;
        }
      }
    }
    else
    {
      for (int t = 0; t < obj->tris.n_local; ++t, ++at)
      {
        double area = 0.0, c[3];
        MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, c);
        cpl_wrap(dom, c);
        memcpy(&m->pos[3 * at], c, sizeof c);
        m->ref[at].key = body | (uint64_t) (uint32_t) obj->tris.rec[t].global_id;
        m->ref[at].body_gid = obj->global_id;
        m->ref[at].slot = o;
        m->ref[at].entity = t;
        m->ref[at].sub = 0;
        m->ref[at].weight = area;
      }
    }
  }
  MFTLCplFree(dual);
  return MFTL_COUPLING_OK;
}

void MFTLCplMarkersFree(cpl_markers *m)
{
  MFTLCplFree(m->pos);
  MFTLCplFree(m->ref);
  MFTLCplFree(m->bary);
  MFTLCplFree(m->bodies);
  memset(m, 0, sizeof *m);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Quadrature points and their vertices
 *
 * A point of an owned triangle contributes to its three corners, which may be
 * owned here or ghosts held for the triangle. The ghost schedule is built once
 * per binding: each rank names, to each owner, the (body, vertex) ids it holds
 * as ghosts -- sorted, so the owner's reply order is the request order and no
 * id travels again -- and the owner resolves them to its own vertex slots.
 * Afterwards a reverse exchange completes per-vertex sums and a forward one
 * delivers owned values to the ghost copies: fixed peers, fixed sizes.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One ghost corner while the table is assembled. */
typedef struct
{
  int owner;
  int body;
  int vid;
  int64_t occ;  ///< which corner occurrence (3 q + c)
} qghost_t;

static int cmp_qghost(const void *a, const void *b)
{
  const qghost_t *x = (const qghost_t *) a, *y = (const qghost_t *) b;
  if (x->owner != y->owner) return (x->owner > y->owner) - (x->owner < y->owner);
  if (x->body != y->body) return (x->body > y->body) - (x->body < y->body);
  if (x->vid != y->vid) return (x->vid > y->vid) - (x->vid < y->vid);
  return (x->occ > y->occ) - (x->occ < y->occ);
}

/** @brief (vertex id, slot) of owned vertices, sorted by id: an owner's lookup. */
typedef struct
{
  int gid, slot;
} qvid_t;

static int cmp_qvid(const void *a, const void *b)
{
  const qvid_t *x = (const qvid_t *) a, *y = (const qvid_t *) b;
  return (x->gid > y->gid) - (x->gid < y->gid);
}

void MFTLCplQuadFree(cpl_quad **pq)
{
  if (pq == MFTL_NULL || *pq == MFTL_NULL) return;
  cpl_quad *q = *pq;
  MFTLCplFree(q->corner);
  MFTLCplFree(q->bary);
  MFTLCplFree(q->owner_rank);
  MFTLCplFree(q->owner_off);
  MFTLCplFree(q->req_rank);
  MFTLCplFree(q->req_off);
  MFTLCplFree(q->req_vt);
  MFTLCplFree(q);
  *pq = MFTL_NULL;
}

int MFTLCplQuadBuild(MPI_Comm comm, const MFTL_Ensemble *ens, const cpl_markers *m, cpl_quad **out, long *bytes, long *messages)
{
  int status = MFTL_COUPLING_OK;
  cpl_quad *q = (cpl_quad *) MFTLCplCalloc(1, sizeof(cpl_quad), "quadrature");
  const int nb = ens->n_local_objects;
  int64_t *body_off = (int64_t *) MFTLCplAlloc((size_t) (nb + 1) * sizeof(int64_t), "quadrature");
  body_off[0] = 0;
  for (int o = 0; o < nb; ++o) body_off[o + 1] = body_off[o] + ens->local_objects[o]->verts.n_local;
  q->n_own = body_off[nb];
  q->corner = (int64_t *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(m->n, 1) * sizeof(int64_t), "quadrature");
  q->bary = (double *) MFTLCplAlloc(2 * (size_t) MFTL_MAX(m->n, 1) * sizeof(double), "quadrature");
  if (m->n > 0) memcpy(q->bary, m->bary, 2 * (size_t) m->n * sizeof(double));

  /* ── corners: owned ones directly, ghost ones collected ── */
  int64_t ng = 0, cap = 64;
  qghost_t *gh = (qghost_t *) MFTLCplAlloc((size_t) cap * sizeof(qghost_t), "quadrature ghosts");
  for (int64_t i = 0; i < m->n; ++i)
  {
    const MFTL_Object *obj = ens->local_objects[m->ref[i].slot];
    int v[3];
    MFTLGetTriangleVertices(obj, m->ref[i].entity, v);
    for (int c = 0; c < 3; ++c)
    {
      if (v[c] < obj->verts.n_local)
      {
        q->corner[3 * i + c] = body_off[m->ref[i].slot] + v[c];
        continue;
      }
      if (ng == cap)
      {
        cap *= 2;
        gh = (qghost_t *) MFTLCplRealloc(gh, (size_t) cap * sizeof(qghost_t), "quadrature ghosts");
      }
      gh[ng].owner = obj->verts.rec[v[c]].owner_proc;
      gh[ng].body = obj->global_id;
      gh[ng].vid = obj->verts.rec[v[c]].global_id;
      gh[ng].occ = 3 * i + c;
      ++ng;
    }
  }
  if (ng > 1) qsort(gh, (size_t) ng, sizeof(qghost_t), cmp_qghost);
  /* distinct ghosts -> table entries n_own + u; owners grouped */
  int64_t u = -1;
  int cap_own = 0;
  for (int64_t i = 0; i < ng; ++i)
  {
    if (i == 0 || gh[i].owner != gh[i - 1].owner || gh[i].body != gh[i - 1].body || gh[i].vid != gh[i - 1].vid)
    {
      ++u;
      if (i == 0 || gh[i].owner != gh[i - 1].owner)
      {
        if (q->n_owner == cap_own)
        {
          cap_own = MFTLGrowCapacity(cap_own, q->n_owner + 1);
          q->owner_rank = (int *) MFTLCplRealloc(q->owner_rank, (size_t) cap_own * sizeof(int), "quadrature owners");
          q->owner_off = (int64_t *) MFTLCplRealloc(q->owner_off, (size_t) (cap_own + 1) * sizeof(int64_t), "quadrature owners");
        }
        q->owner_rank[q->n_owner] = gh[i].owner;
        q->owner_off[q->n_owner] = u;
        q->n_owner++;
      }
    }
    q->corner[gh[i].occ] = q->n_own + u;
  }
  q->n_ghost = u + 1;
  if (q->owner_off == MFTL_NULL) q->owner_off = (int64_t *) MFTLCplAlloc(sizeof(int64_t), "quadrature owners");
  q->owner_off[q->n_owner] = q->n_ghost;

  /* ── requests: per owner, the (body, vertex) ids of its ghosts, in table order ── */
  unsigned char **sb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(q->n_owner, 1) * sizeof(unsigned char *), "quadrature requests");
  size_t *sz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(q->n_owner, 1) * sizeof(size_t), "quadrature requests");
  {
    int64_t i = 0;
    for (int o = 0; o < q->n_owner; ++o)
    {
      cpl_buf b = {MFTL_NULL, 0, 0};
      for (; i < ng && gh[i].owner == q->owner_rank[o]; ++i)
      {
        if (i > 0 && gh[i].owner == gh[i - 1].owner && gh[i].body == gh[i - 1].body && gh[i].vid == gh[i - 1].vid) continue;
        const int32_t rec[2] = {gh[i].body, gh[i].vid};
        MFTLCplBufPut(&b, rec, sizeof rec);
      }
      sb[o] = b.data;
      sz[o] = b.n;
    }
  }
  MFTLCplFree(gh);
  cpl_msg *in = MFTL_NULL;
  int n_in = 0;
  MFTLCplNbx(comm, CPL_TAG_QUAD_SETUP, q->n_owner, q->owner_rank, sb, sz, &in, &n_in, bytes, messages);
  for (int o = 0; o < q->n_owner; ++o) MFTLCplFree(sb[o]);
  MFTLCplFree(sb);
  MFTLCplFree(sz);

  /* ── owner side: resolve every requested id to an owned vertex-table index ── */
  q->n_req = n_in;
  q->req_rank = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n_in, 1) * sizeof(int), "quadrature requesters");
  q->req_off = (int64_t *) MFTLCplAlloc((size_t) (n_in + 1) * sizeof(int64_t), "quadrature requesters");
  int64_t total = 0;
  for (int r = 0; r < n_in; ++r) total += (int64_t) (in[r].bytes / (2 * sizeof(int32_t)));
  q->req_vt = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(int64_t), "quadrature requesters");
  qvid_t **lookup = (qvid_t **) MFTLCplCalloc((size_t) MFTL_MAX(nb, 1), sizeof(qvid_t *), "quadrature lookup");
  int64_t at = 0;
  for (int r = 0; r < n_in; ++r)
  {
    q->req_rank[r] = in[r].peer;
    q->req_off[r] = at;
    const int64_t nrec = (int64_t) (in[r].bytes / (2 * sizeof(int32_t)));
    for (int64_t e = 0; e < nrec; ++e)
    {
      int32_t rec[2];
      memcpy(rec, in[r].data + (size_t) e * sizeof rec, sizeof rec);
      int slot = -1;
      for (int lo = 0, hi = nb; lo < hi;)
      {
        const int mid = lo + (hi - lo) / 2;
        const int gid = ens->local_objects[mid]->global_id;
        if (gid == rec[0])
        {
          slot = mid;
          break;
        }
        if (gid < rec[0])
          lo = mid + 1;
        else
          hi = mid;
      }
      int64_t vt = -1;
      if (slot >= 0)
      {
        const MFTL_Object *obj = ens->local_objects[slot];
        if (lookup[slot] == MFTL_NULL)
        {
          lookup[slot] = (qvid_t *) MFTLCplAlloc((size_t) MFTL_MAX(obj->verts.n_local, 1) * sizeof(qvid_t), "quadrature lookup");
          for (int v = 0; v < obj->verts.n_local; ++v)
          {
            lookup[slot][v].gid = obj->verts.rec[v].global_id;
            lookup[slot][v].slot = v;
          }
          if (obj->verts.n_local > 1) qsort(lookup[slot], (size_t) obj->verts.n_local, sizeof(qvid_t), cmp_qvid);
        }
        const qvid_t key = {rec[1], 0};
        const qvid_t *hit = (const qvid_t *) bsearch(&key, lookup[slot], (size_t) obj->verts.n_local, sizeof(qvid_t), cmp_qvid);
        if (hit) vt = body_off[slot] + hit->slot;
      }
      if (vt < 0)
      {
        fprintf(stderr, "MFTL coupling: rank holds body %d vertex %d as a ghost owned here, but it is not owned here\n", rec[0], rec[1]);
        status = MFTL_COUPLING_INVALID;
        vt = 0;
      }
      q->req_vt[at++] = vt;
    }
  }
  q->req_off[n_in] = at;
  for (int o = 0; o < nb; ++o) MFTLCplFree(lookup[o]);
  MFTLCplFree(lookup);
  MFTLCplMsgsFree(in, n_in);
  MFTLCplFree(in);
  MFTLCplFree(body_off);
  *out = q;
  return status;
}

void MFTLCplQuadAccumulate(MPI_Comm comm, const cpl_quad *q, int64_t n_markers, const MFTL_MarkerRef *ref, int ncomp, const double *val, double *acc,
                           long *bytes, long *messages)
{
  const int nc1 = ncomp + 1;
  const int64_t ntab = q->n_own + q->n_ghost;
  double *tab = (double *) MFTLCplCalloc((size_t) MFTL_MAX(ntab, 1) * (size_t) nc1, sizeof(double), "quadrature sums");
  for (int64_t i = 0; i < n_markers; ++i)
  {
    const double l1 = q->bary[2 * i], l2 = q->bary[2 * i + 1];
    const double lam[3] = {1.0 - l1 - l2, l1, l2};
    for (int c = 0; c < 3; ++c)
    {
      const double wl = ref[i].weight * lam[c];
      double *t = &tab[q->corner[3 * i + c] * nc1];
      for (int k = 0; k < ncomp; ++k) t[k] += wl * val[i * ncomp + k];
      t[ncomp] += wl;
    }
  }
  /* reverse: ghost sums to their owners; owners add requesters' in rank order */
  const int64_t nrecv = q->req_off[q->n_req] * nc1;
  double *rbuf = (double *) MFTLCplAlloc((size_t) MFTL_MAX(nrecv, 1) * sizeof(double), "quadrature receive");
  cpl_sched s;
  memset(&s, 0, sizeof s);
  for (int r = 0; r < q->n_req; ++r)
  {
    const int64_t n = (q->req_off[r + 1] - q->req_off[r]) * nc1;
    if (n > 0) MFTLCplSchedRecv(comm, &s, CPL_TAG_QUAD, q->req_rank[r], rbuf + q->req_off[r] * nc1, (size_t) n * sizeof(double));
  }
  for (int o = 0; o < q->n_owner; ++o)
  {
    const int64_t n = (q->owner_off[o + 1] - q->owner_off[o]) * nc1;
    if (n > 0)
    {
      MFTLCplSchedSend(comm, &s, CPL_TAG_QUAD, q->owner_rank[o], tab + (q->n_own + q->owner_off[o]) * nc1, (size_t) n * sizeof(double), messages);
      if (bytes) *bytes += (long) (n * (int64_t) sizeof(double));
    }
  }
  MFTLCplSchedWait(&s);
  MFTLCplSchedFree(&s);
  if (bytes) *bytes += (long) (nrecv * (int64_t) sizeof(double));
  for (int64_t e = 0; e < q->req_off[q->n_req]; ++e)
    for (int k = 0; k < nc1; ++k) tab[q->req_vt[e] * nc1 + k] += rbuf[e * nc1 + k];
  if (q->n_own > 0) memcpy(acc, tab, (size_t) q->n_own * (size_t) nc1 * sizeof(double));
  MFTLCplFree(rbuf);
  MFTLCplFree(tab);
}

void MFTLCplQuadInterpolate(MPI_Comm comm, const cpl_quad *q, int64_t n_markers, const MFTL_MarkerRef *ref, bool weighted, int ncomp, const double *g,
                            double *out, long *bytes, long *messages)
{
  const int64_t ntab = q->n_own + q->n_ghost;
  double *tab = (double *) MFTLCplAlloc((size_t) MFTL_MAX(ntab, 1) * (size_t) MFTL_MAX(ncomp, 1) * sizeof(double), "quadrature values");
  if (q->n_own > 0) memcpy(tab, g, (size_t) q->n_own * (size_t) ncomp * sizeof(double));
  /* forward: owners send the requested vertices' values; ghosts arrive in table order */
  const int64_t nsend = q->req_off[q->n_req] * ncomp;
  double *sbuf = (double *) MFTLCplAlloc((size_t) MFTL_MAX(nsend, 1) * sizeof(double), "quadrature send");
  for (int64_t e = 0; e < q->req_off[q->n_req]; ++e)
    for (int k = 0; k < ncomp; ++k) sbuf[e * ncomp + k] = g[q->req_vt[e] * ncomp + k];
  cpl_sched s;
  memset(&s, 0, sizeof s);
  for (int o = 0; o < q->n_owner; ++o)
  {
    const int64_t n = (q->owner_off[o + 1] - q->owner_off[o]) * ncomp;
    if (n > 0) MFTLCplSchedRecv(comm, &s, CPL_TAG_QUAD, q->owner_rank[o], tab + (q->n_own + q->owner_off[o]) * ncomp, (size_t) n * sizeof(double));
  }
  for (int r = 0; r < q->n_req; ++r)
  {
    const int64_t n = (q->req_off[r + 1] - q->req_off[r]) * ncomp;
    if (n > 0)
    {
      MFTLCplSchedSend(comm, &s, CPL_TAG_QUAD, q->req_rank[r], sbuf + q->req_off[r] * ncomp, (size_t) n * sizeof(double), messages);
      if (bytes) *bytes += (long) (2 * n * (int64_t) sizeof(double));
    }
  }
  MFTLCplSchedWait(&s);
  MFTLCplSchedFree(&s);
  for (int64_t i = 0; i < n_markers; ++i)
  {
    const double l1 = q->bary[2 * i], l2 = q->bary[2 * i + 1];
    const double lam[3] = {1.0 - l1 - l2, l1, l2};
    const double w = weighted ? ref[i].weight : 1.0;
    for (int k = 0; k < ncomp; ++k)
    {
      double v = 0.0;
      for (int c = 0; c < 3; ++c) v += lam[c] * tab[q->corner[3 * i + c] * ncomp + k];
      out[i * ncomp + k] = w * v;
    }
  }
  MFTLCplFree(sbuf);
  MFTLCplFree(tab);
}

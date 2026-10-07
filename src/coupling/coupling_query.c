/**
 * @file coupling_query.c
 * @author Christian Gorges
 * @date 2026-10-07
 * @ingroup mftl_coupling
 *
 * @brief Nearest-surface candidates: their total order and merge, and the
 *        local geometry a query needs -- a triangle in its centroid's frame and
 *        partition-independent vertex pseudo-normals. No MPI.
 *
 * A query's result at a sample is the least of every candidate any rank
 * computed for it. The least of a SET is one record however the set is split
 * and merged only when the order is total and every field takes part in it:
 * with a scalar \f$\min|\phi|\f$ two triangles at the same distance would leave
 * the body, the feature and even the sign to whichever arrived first. So the
 * order compares the distance, its sign, the body, the triangle, the feature
 * and the offset, in that order, and two candidates compare equal only when
 * they are the same record.
 */

#include "coupling_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The candidate's order
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLSurfaceCandidateReset(MFTL_SurfaceCandidate *c)
{
  if (c == MFTL_NULL) return;
  memset(c, 0, sizeof *c);
  c->distance = MFTL_DBL_MAX;
  c->triangle = -1;
  c->body = -1;
}

/** @brief A candidate takes part in the order only if it says it is valid and is finite. */
static bool cand_valid(const MFTL_SurfaceCandidate *c)
{
  return c->valid == 1 && isfinite(c->distance) && isfinite(c->offset[0]) && isfinite(c->offset[1]) && isfinite(c->offset[2]);
}

/** @brief Finite doubles by value, -0 before +0: equal only when the bits are. */
static int cmp_signed(double a, double b)
{
  if (a < b) return -1;
  if (a > b) return 1;
  const int sa = signbit(a) ? 1 : 0, sb = signbit(b) ? 1 : 0;
  return sb - sa;
}

int MFTLSurfaceCandidateCompare(const MFTL_SurfaceCandidate *a, const MFTL_SurfaceCandidate *b)
{
  const bool va = cand_valid(a), vb = cand_valid(b);
  if (va != vb) return va ? -1 : 1;
  if (!va) return 0;
  const double ma = fabs(a->distance), mb = fabs(b->distance);
  if (ma != mb) return (ma < mb) ? -1 : 1;
  const int sa = signbit(a->distance) ? 1 : 0, sb = signbit(b->distance) ? 1 : 0;
  if (sa != sb) return sa ? -1 : 1; /* inside first; -0 before +0 */
  if (a->body != b->body) return (a->body < b->body) ? -1 : 1;
  if (a->triangle != b->triangle) return (a->triangle < b->triangle) ? -1 : 1;
  if (a->feature != b->feature) return (a->feature < b->feature) ? -1 : 1;
  for (int k = 0; k < 3; ++k)
  {
    const int c = cmp_signed(a->offset[k], b->offset[k]);
    if (c != 0) return c;
  }
  return 0;
}

void MFTLSurfaceCandidateMerge(MFTL_SurfaceCandidate *acc, const MFTL_SurfaceCandidate *c)
{
  if (acc == MFTL_NULL || c == MFTL_NULL) return;
  if (MFTLSurfaceCandidateCompare(c, acc) < 0) *acc = *c;
  if (!cand_valid(acc)) MFTLSurfaceCandidateReset(acc);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Local geometry
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLCplTriangleCentroidFrame(const MFTL_Domain *d, const MFTL_Object *obj, int t, double T[3][3], double c[3])
{
  double area = 0.0;
  MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, c);
  cpl_wrap(d, c);
  MFTLGetTriangleCorners(obj, t, T);
  for (int k = 0; k < 3; ++k)
  {
    if (!d->is_periodic[k]) continue;
    const double c0 = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
    const double n = nearbyint((c[k] - c0) * d->inv_domain_size[k]);
    if (n == 0.0) continue;
    for (int i = 0; i < 3; ++i) T[i][k] += n * d->domain_size[k];
  }
}

void MFTLCplFaceNormal(const double T[3][3], double n[3])
{
  const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
  const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const double mag = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
  for (int k = 0; k < 3; ++k) n[k] *= inv;
}

/** @brief One triangle's contribution to one owned vertex's pseudo-normal. */
typedef struct
{
  int64_t vt;
  int64_t gid;
  double v[3];
} pn_t;

static int cmp_pn(const void *a, const void *b)
{
  const pn_t *x = (const pn_t *) a, *y = (const pn_t *) b;
  if (x->vt != y->vt) return (x->vt > y->vt) - (x->vt < y->vt);
  return (x->gid > y->gid) - (x->gid < y->gid);
}

void MFTLCplVertexPseudoNormals(const MFTL_Ensemble *ens, double *g)
{
  int64_t n_own = 0, n_contrib = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    n_own += ens->local_objects[o]->verts.n_local;
    n_contrib += 3 * (int64_t) ens->local_objects[o]->tris.n;
  }
  for (int64_t i = 0; i < 3 * n_own; ++i) g[i] = 0.0;
  pn_t *pn = (pn_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_contrib, 1) * sizeof(pn_t), "pseudo-normals");
  int64_t np = 0, base = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n; ++t)
    {
      int vv[3];
      double T[3][3], nf[3];
      MFTLGetTriangleVertices(obj, t, vv);
      MFTLGetTriangleCorners(obj, t, T);
      MFTLCplFaceNormal(T, nf);
      for (int c = 0; c < 3; ++c)
      {
        if (vv[c] < 0 || vv[c] >= obj->verts.n_local) continue;
        const double *a = T[c], *b = T[(c + 1) % 3], *e = T[(c + 2) % 3];
        const double e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {e[0] - a[0], e[1] - a[1], e[2] - a[2]};
        const double cx[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        const double ang = atan2(sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]), e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]);
        pn[np].vt = base + vv[c];
        pn[np].gid = obj->tris.rec[t].global_id;
        for (int k = 0; k < 3; ++k) pn[np].v[k] = ang * nf[k];
        ++np;
      }
    }
    base += obj->verts.n_local;
  }
  /* the fan in ascending triangle id: the same sum on every partition */
  if (np > 1) qsort(pn, (size_t) np, sizeof(pn_t), cmp_pn);
  for (int64_t i = 0; i < np; ++i)
    for (int k = 0; k < 3; ++k) g[3 * pn[i].vt + k] += pn[i].v[k];
  for (int64_t v = 0; v < n_own; ++v)
  {
    double *x = &g[3 * v];
    const double mag = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
    for (int k = 0; k < 3; ++k) x[k] *= inv;
  }
  free(pn);
}

/** @file remesh_fields.c
 * @brief Local conservative interpolation and swept-edge material transfer.
 */
#include "mftl_remeshing.h"
#include "remesh_fields.h"
#include "mftl_memory.h"
#include "mftl_vectoroperations.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>

/** @cond INTERNAL */
static double area_points(const MFTL_Object *obj, const double *a, const double *b, const double *c)
{
  double u[3], v[3];
  MFTLGetMinimumImageVector(obj->domain, u, a, b);
  MFTLGetMinimumImageVector(obj->domain, v, a, c);
  const double x = u[1] * v[2] - u[2] * v[1], y = u[2] * v[0] - u[0] * v[2], z = u[0] * v[1] - u[1] * v[0];
  return 0.5 * sqrt(x * x + y * y + z * z);
}

double MFTLRemeshFieldArea(const MFTL_Object *obj, const int corners[3], const double point[3])
{
  const double *p[3];
  for (int k = 0; k < 3; ++k) p[k] = corners[k] < 0 ? point : obj->verts.pos + 3 * corners[k];
  return area_points(obj, p[0], p[1], p[2]);
}

double MFTLRemeshFieldEdgeParameter(const MFTL_Object *obj, int a, int b, const double point[3])
{
  double edge[3], delta[3], dot = 0.0, norm = 0.0;
  MFTLGetMinimumImageVector(obj->domain, edge, obj->verts.pos + 3 * a, obj->verts.pos + 3 * b);
  MFTLGetMinimumImageVector(obj->domain, delta, obj->verts.pos + 3 * a, point);
  for (int k = 0; k < 3; ++k)
  {
    dot += delta[k] * edge[k];
    norm += edge[k] * edge[k];
  }
  return norm > 0.0 ? fmax(0.0, fmin(1.0, dot / norm)) : 0.5;
}

void MFTLRemeshFieldVertex(MFTL_Object *obj, int a, int b, int dest, double fraction)
{
  if (!MFTLFieldsStride(obj, 0)) return;
  for (int f = 0; f < obj->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
    if (d->entity != MFTL_FIELD_VERTEX || d->mode == MFTL_FIELD_RECOMPUTED) continue;
    double *v = obj->field_data[f];
    for (int c = 0; c < d->ncomp; ++c)
      v[(size_t) dest * d->ncomp + c] = (1.0 - fraction) * v[(size_t) a * d->ncomp + c] + fraction * v[(size_t) b * d->ncomp + c];
  }
}

double *MFTLRemeshFieldPrepare(const MFTL_Object *obj, const int *old_ids, const double *old_area, int nold, const int (*corners)[3],
                             const double point[3], const double *new_area, int nnew, const int *parent, bool keep_survivors)
{
  const int stride = MFTLFieldsStride(obj, 1);
  MFTL_ASSERT(stride && nold <= 2 * MFTL_REMESH_RING_MAX && nnew <= 2 * MFTL_REMESH_RING_MAX);
  double total_old_area = 0.0, total_new_area = 0.0;
  for (int i = 0; i < nold; ++i)
  {
    if (!isfinite(old_area[i]) || old_area[i] < 0.0) return NULL;
    total_old_area += old_area[i];
  }
  for (int j = 0; j < nnew; ++j)
  {
    if (!isfinite(new_area[j]) || (keep_survivors ? new_area[j] < 0.0 : new_area[j] <= 0.0)) return NULL;
    total_new_area += new_area[j];
  }
  if (!(total_new_area > 0.0) || !isfinite(total_new_area) || !isfinite(total_old_area)) return NULL;

  bool conservative = false;
  for (int f = 0; f < obj->fields->n; ++f) conservative |= obj->fields->entries[f].mode == MFTL_FIELD_CONSERVATIVE_DENSITY;
  int donor[2 * MFTL_REMESH_RING_MAX];
  if (conservative && !parent)
  {
    /* Search only the edit's old patch. Corners are in one periodic image;
     * candidate centres are brought into each donor's image before projection. */
    double triangles[2 * MFTL_REMESH_RING_MAX][3][3];
    for (int i = 0; i < nold; ++i) MFTLGetTriangleCorners(obj, old_ids[i], triangles[i]);
    for (int j = 0; j < nnew; ++j)
    {
      if (new_area[j] == 0.0) continue;
      const double *p0 = corners[j][0] < 0 ? point : obj->verts.pos + 3 * corners[j][0];
      double center[3] = {p0[0], p0[1], p0[2]};
      for (int k = 1; k < 3; ++k)
      {
        const double *p = corners[j][k] < 0 ? point : obj->verts.pos + 3 * corners[j][k];
        double delta[3];
        MFTLGetMinimumImageVector(obj->domain, delta, p0, p);
        for (int d = 0; d < 3; ++d) center[d] += delta[d] / 3.0;
      }
      double best = DBL_MAX;
      int gid = INT_MAX;
      donor[j] = -1;
      for (int i = 0; i < nold; ++i)
      {
        double delta[3], p[3], q[3];
        MFTLGetMinimumImageVector(obj->domain, delta, triangles[i][0], center);
        for (int k = 0; k < 3; ++k) p[k] = triangles[i][0][k] + delta[k];
        const double distance = MFTLClosestPointOnTriangle(p, triangles[i], q, NULL);
        if (!isfinite(distance)) return NULL;
        const int candidate_gid = obj->tris.rec[old_ids[i]].global_id;
        /* Squared-distance round-off at an old shared edge must not decide
         * between discontinuous donor values after a periodic translation. */
        const double tie = 64 * DBL_EPSILON * fmax(old_area[i], donor[j] < 0 ? 0.0 : old_area[donor[j]]);
        if (donor[j] < 0 || distance < best - tie || (fabs(distance - best) <= tie && candidate_gid < gid))
        {
          best = distance;
          gid = candidate_gid;
          donor[j] = i;
        }
      }
      if (donor[j] < 0) return NULL;
    }
  }

  double *out = MFTLMallocOrDie((size_t) nnew * stride * sizeof(double), "local field transfer");
  int offset = 0;
  for (int f = 0; f < obj->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
    if (d->entity != MFTL_FIELD_TRIANGLE || d->mode == MFTL_FIELD_RECOMPUTED) continue;
    const double *v = obj->field_data[f];
    for (int c = 0; c < d->ncomp; ++c)
    {
      double old[2 * MFTL_REMESH_RING_MAX], total = 0.0, removed = 0.0;
      for (int i = 0; i < nold; ++i)
      {
        old[i] = v[(size_t) old_ids[i] * d->ncomp + c];
        if (!isfinite(old[i]) || (d->mode == MFTL_FIELD_CONSERVATIVE_DENSITY && old[i] < 0.0)) goto invalid;
        total += old[i] * old_area[i];
        if (keep_survivors && new_area[i] == 0.0) removed += old[i] * old_area[i];
      }
      if (!isfinite(total)) goto invalid;
      /* A single patch correction retains the interpolated profile. Scale the
       * predictor first to avoid overflowing A*rho before normalization. */
      double max_predictor = 0.0, predicted = 0.0;
      if (d->mode == MFTL_FIELD_CONSERVATIVE_DENSITY && !parent)
      {
        for (int j = 0; j < nnew; ++j)
          if (new_area[j] > 0.0) max_predictor = fmax(max_predictor, old[donor[j]]);
        if (max_predictor > 0.0)
          for (int j = 0; j < nnew; ++j)
            if (new_area[j] > 0.0) predicted += (old[donor[j]] / max_predictor) * new_area[j];
      }
      for (int j = 0; j < nnew; ++j)
      {
        if (new_area[j] == 0.0) continue;
        double value;
        if (parent)
        {
          const int p = parent[j];
          double children = 0.0;
          for (int k = 0; k < nnew; ++k)
            if (parent[k] == p) children += new_area[k];
          value = old[p];
          if (d->mode == MFTL_FIELD_CONSERVATIVE_DENSITY) value *= old_area[p] / children;
        }
        else if (d->mode == MFTL_FIELD_INTERPOLATED)
          value = total_old_area > 0.0 ? total / total_old_area : old[0];
        else if (predicted > 0.0)
          value = (total * (((old[donor[j]] / max_predictor) * new_area[j]) / predicted)) / new_area[j];
        else if (keep_survivors)
          value = old[j] * (old_area[j] / new_area[j]) + removed / total_new_area;
        else
          value = total / total_new_area;
        if (!isfinite(value) || (d->mode == MFTL_FIELD_CONSERVATIVE_DENSITY && value < 0.0)) goto invalid;
        out[(size_t) j * stride + offset + c] = value;
      }
    }
    offset += d->ncomp;
  }
  return out;
invalid:
  free(out);
  return NULL;
}

void MFTLRemeshFieldCommit(MFTL_Object *obj, const int *new_ids, const double *new_area, int nnew, double *values)
{
  const int stride = MFTLFieldsStride(obj, 1);
  for (int j = 0; j < nnew; ++j)
    if (new_area[j] > 0.0) MFTLFieldsRecord(obj, MFTL_FIELD_TRIANGLE, new_ids[j], values + (size_t) j * stride, true);
  free(values);
}

/* ClosestPoint already classifies degenerate triangles as edges or vertices. */
static void barycentric(const double tri[3][3], const double q[3], MFTL_TriangleFeature feature, double w[3])
{
  w[0] = w[1] = w[2] = 0.0;
  if (feature >= MFTL_TRI_FEATURE_VERTEX0 && feature <= MFTL_TRI_FEATURE_VERTEX2)
  {
    w[feature - MFTL_TRI_FEATURE_VERTEX0] = 1.0;
    return;
  }
  if (feature >= MFTL_TRI_FEATURE_EDGE0)
  {
    const int a = feature - MFTL_TRI_FEATURE_EDGE0, b = (a + 1) % 3;
    double dot = 0.0, norm = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double e = tri[b][k] - tri[a][k];
      dot += (q[k] - tri[a][k]) * e;
      norm += e * e;
    }
    const double t = norm > 0.0 ? fmax(0.0, fmin(1.0, dot / norm)) : 0.0;
    w[a] = 1.0 - t;
    w[b] = t;
    return;
  }
  double aa = 0.0, ab = 0.0, bb = 0.0, aq = 0.0, bq = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    const double a = tri[1][k] - tri[0][k], b = tri[2][k] - tri[0][k], p = q[k] - tri[0][k];
    aa += a * a;
    ab += a * b;
    bb += b * b;
    aq += a * p;
    bq += b * p;
  }
  const double det = aa * bb - ab * ab;
  if (!(det > 0.0))
  {
    w[0] = 1.0;
    return;
  }
  w[1] = fmax(0.0, (bb * aq - ab * bq) / det);
  w[2] = fmax(0.0, (aa * bq - ab * aq) / det);
  w[0] = fmax(0.0, 1.0 - w[1] - w[2]);
  const double sum = w[0] + w[1] + w[2];
  for (int k = 0; k < 3; ++k) w[k] /= sum;
}

typedef struct
{
  double old_area, new_area, normal[3];
} FieldSmoothTriangle;

typedef struct
{
  int from, to;
  double area;
} FieldSmoothEdge;

/* Stage every conservative component, so even a bad final component cannot
 * leave earlier ones modified. All edge fluxes read the old sweep. */
static bool smooth_conservative(const MFTL_Object *obj, const double *new_pos, int width, double **result, unsigned char **changed)
{
  const int nt = obj->tris.n;
  FieldSmoothTriangle *geometry = MFTLCallocOrDie((size_t) nt, sizeof(*geometry), "field smoothing geometry");
  FieldSmoothEdge *edges = MFTLMallocOrDie(((size_t) nt * 3 / 2 + 1) * sizeof(*edges), "field smoothing edges");
  unsigned char *moved = MFTLCallocOrDie((size_t) nt, sizeof(*moved), "field smoothing patch");
  double *out = MFTLMallocOrDie((size_t) nt * width * sizeof(double), "field smoothing values");
  double *outgoing = MFTLCallocOrDie((size_t) nt, sizeof(double), "field smoothing outgoing");
  double *incoming = MFTLCallocOrDie((size_t) nt, sizeof(double), "field smoothing incoming");
  bool valid = false, fallback = false;
  int ne = 0;
  for (int t = 0; t < nt; ++t)
  {
    if (tri_is_dead(obj, t)) continue;
    int v[3];
    double p[3][3], a[3], b[3];
    MFTLGetTriangleVertices(obj, t, v);
    for (int k = 0; k < 3; ++k) moved[t] |= memcmp(obj->verts.pos + 3 * v[k], new_pos + 3 * v[k], 3 * sizeof(double)) != 0;
    if (!moved[t]) continue;
    /* A moved vertex must have its entire writable fan here and unreplicated. */
    if (MFTLTriangleIsFrozen(obj, t)) goto done;
    MFTLGetTriangleCorners(obj, t, p);
    for (int k = 0; k < 3; ++k)
    {
      a[k] = p[1][k] - p[0][k];
      b[k] = p[2][k] - p[0][k];
    }
    MFTLVectorCrossVector(geometry[t].normal, a, b);
    const double length = MFTLVectorNorm(geometry[t].normal);
    geometry[t].old_area = 0.5 * length;
    if (length > 0.0)
      for (int k = 0; k < 3; ++k) geometry[t].normal[k] /= length;
    double next_normal[3];
    MFTLGetMinimumImageVector(obj->domain, a, new_pos + 3 * v[0], new_pos + 3 * v[1]);
    MFTLGetMinimumImageVector(obj->domain, b, new_pos + 3 * v[0], new_pos + 3 * v[2]);
    MFTLVectorCrossVector(next_normal, a, b);
    geometry[t].new_area = 0.5 * MFTLVectorNorm(next_normal);
    if (!(geometry[t].new_area > 0.0) || !isfinite(geometry[t].new_area) || !isfinite(length)) goto done;
    if (length > 0.0 && !(MFTLVectorInProd(geometry[t].normal, next_normal) > 0.0)) goto done;
  }
  for (int h = 0; h < 3 * nt; ++h)
  {
    const int twin = obj->tris.he[h].twin_idx, left = MFTLHeTriIdx(h);
    if (twin < 0 || h > twin || tri_is_dead(obj, left)) continue;
    const int right = MFTLHeTriIdx(twin);
    const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
    if (memcmp(obj->verts.pos + 3 * a, new_pos + 3 * a, 3 * sizeof(double)) == 0 &&
        memcmp(obj->verts.pos + 3 * b, new_pos + 3 * b, 3 * sizeof(double)) == 0) continue;
    if (!moved[left] || !moved[right] || tri_is_dead(obj, right)) goto done;
    double normal[3], edge[3], da[3], db[3], end[3], c1[3], c2[3];
    for (int k = 0; k < 3; ++k) normal[k] = geometry[left].normal[k] + geometry[right].normal[k];
    const double length = MFTLVectorNorm(normal);
    if (!(length > 1e-12) || !(geometry[left].old_area > 0.0) || !(geometry[right].old_area > 0.0))
    {
      fallback = true;
      break;
    }
    MFTLGetMinimumImageVector(obj->domain, edge, obj->verts.pos + 3 * a, obj->verts.pos + 3 * b);
    MFTLGetMinimumImageVector(obj->domain, da, obj->verts.pos + 3 * a, new_pos + 3 * a);
    MFTLGetMinimumImageVector(obj->domain, db, obj->verts.pos + 3 * b, new_pos + 3 * b);
    for (int k = 0; k < 3; ++k) end[k] = edge[k] + db[k];
    MFTLVectorCrossVector(c1, edge, db);
    MFTLVectorCrossVector(c2, end, da);
    double swept = 0.0;
    /* Signed area of (a,b,b_new,a_new). Positive means left loses area.
     * Reverse orientation changes the sign; the shared value is computed once. */
    for (int k = 0; k < 3; ++k) swept += 0.5 * (c1[k] + c2[k]) * (normal[k] / length);
    if (!isfinite(swept)) goto done;
    if (swept == 0.0) continue;
    edges[ne++] = (FieldSmoothEdge) {swept > 0.0 ? left : right, swept > 0.0 ? right : left, fabs(swept)};
  }
  if (fallback) ne = 0;  /* Per-triangle old/new area transfer for this sweep. */
  int offset = 0;
  for (int f = 0; f < obj->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
    if (d->mode != MFTL_FIELD_CONSERVATIVE_DENSITY) continue;
    const double *data = obj->field_data[f];
    for (int c = 0; c < d->ncomp; ++c)
    {
      memset(outgoing, 0, (size_t) nt * sizeof(double));
      memset(incoming, 0, (size_t) nt * sizeof(double));
      for (int t = 0; t < nt; ++t)
        if (moved[t])
        {
          const double rho = data[(size_t) t * d->ncomp + c];
          if (!(rho >= 0.0) || !isfinite(rho * geometry[t].old_area)) goto done;
        }
      for (int e = 0; e < ne; ++e)
      {
        const int from = edges[e].from;
        const double flux = edges[e].area * data[(size_t) from * d->ncomp + c];
        outgoing[from] += flux;
        if (!isfinite(outgoing[from])) goto done;
      }
      for (int e = 0; e < ne; ++e)
      {
        const int from = edges[e].from;
        const double mass = geometry[from].old_area * data[(size_t) from * d->ncomp + c];
        const double limit = outgoing[from] > mass ? mass / outgoing[from] : 1.0;
        incoming[edges[e].to] += limit * (edges[e].area * data[(size_t) from * d->ncomp + c]);
      }
      for (int t = 0; t < nt; ++t)
        if (moved[t])
        {
          const double mass = geometry[t].old_area * data[(size_t) t * d->ncomp + c];
          const double next = (fmax(0.0, mass - outgoing[t]) + incoming[t]) / geometry[t].new_area;
          if (!isfinite(next)) goto done;
          out[(size_t) t * width + offset + c] = next;
        }
    }
    offset += d->ncomp;
  }
  valid = true;
done:
  free(geometry);
  free(edges);
  free(outgoing);
  free(incoming);
  if (!valid)
  {
    free(out);
    free(moved);
    return false;
  }
  *result = out;
  *changed = moved;
  return true;
}

bool MFTLRemeshFieldSmooth(MFTL_Object *obj, const double *new_pos)
{
  if (!MFTLFieldsStride(obj, 0) && !MFTLFieldsStride(obj, 1)) return true;
  const int nv = obj->verts.n, nt = obj->tris.n;
  if (memcmp(obj->verts.pos, new_pos, (size_t) nv * 3 * sizeof(double)) == 0) return true;
  double *conserved = NULL;
  unsigned char *changed = NULL;
  int width = 0;
  for (int f = 0; f < obj->fields->n; ++f)
    if (obj->fields->entries[f].mode == MFTL_FIELD_CONSERVATIVE_DENSITY) width += obj->fields->entries[f].ncomp;
  if (width && !smooth_conservative(obj, new_pos, width, &conserved, &changed)) return false;
  if (MFTLFieldsStride(obj, 0))
  {
    int *donors = MFTLMallocOrDie((size_t) nv * 3 * sizeof(int), "smoothing field donors");
    double *weights = MFTLCallocOrDie((size_t) nv * 3, sizeof(double), "smoothing field weights");
    for (int v = 0; v < nv; ++v)
    {
      for (int k = 0; k < 3; ++k) donors[3 * v + k] = v;
      weights[3 * v] = 1.0;
      if (memcmp(obj->verts.pos + 3 * v, new_pos + 3 * v, 3 * sizeof(double)) == 0) continue;
      int fan[MFTL_REMESH_RING_MAX];
      const int n = MFTLGetVertexOneRingHalfedges(obj, v, fan, MFTL_REMESH_RING_MAX, false);
      double best = DBL_MAX;
      int best_gid = INT_MAX;
      for (int j = 0; j < n; ++j)
      {
        const int t = MFTLHeTriIdx(fan[j]);
        double tri[3][3], delta[3], p[3], q[3];
        MFTL_TriangleFeature feature;
        MFTLGetTriangleCorners(obj, t, tri);
        MFTLGetMinimumImageVector(obj->domain, delta, tri[0], new_pos + 3 * v);
        for (int k = 0; k < 3; ++k) p[k] = tri[0][k] + delta[k];
        const double distance = MFTLClosestPointOnTriangle(p, tri, q, &feature);
        if (distance < best || (distance == best && obj->tris.rec[t].global_id < best_gid))
        {
          best = distance;
          best_gid = obj->tris.rec[t].global_id;
          MFTLGetTriangleVertices(obj, t, donors + 3 * v);
          barycentric(tri, q, feature, weights + 3 * v);
        }
      }
    }
    double *out = MFTLMallocOrDie((size_t) nv * MFTLFieldsWidth(obj, sizeof(double)), "smoothing field values");
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
      if (d->entity != MFTL_FIELD_VERTEX || d->mode == MFTL_FIELD_RECOMPUTED) continue;
      double *data = obj->field_data[f];
      memcpy(out, data, (size_t) nv * d->ncomp * sizeof(double));
      for (int v = 0; v < nv; ++v)
        if (memcmp(obj->verts.pos + 3 * v, new_pos + 3 * v, 3 * sizeof(double)) != 0)
          for (int c = 0; c < d->ncomp; ++c)
          {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) value += weights[3 * v + k] * data[(size_t) donors[3 * v + k] * d->ncomp + c];
            out[(size_t) v * d->ncomp + c] = value;
          }
      memcpy(data, out, (size_t) nv * d->ncomp * sizeof(double));
    }
    free(out);
    free(donors);
    free(weights);
  }
  if (conserved)
  {
    int offset = 0;
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
      if (d->mode != MFTL_FIELD_CONSERVATIVE_DENSITY) continue;
      for (int t = 0; t < nt; ++t)
        if (changed[t])
          memcpy(obj->field_data[f] + (size_t) t * d->ncomp, conserved + (size_t) t * width + offset, (size_t) d->ncomp * sizeof(double));
      offset += d->ncomp;
    }
    free(conserved);
    free(changed);
  }
  return true;
}
/** @endcond */

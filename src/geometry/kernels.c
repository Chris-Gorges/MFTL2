/*
 * MFTL2 geometry — local-frame kernels for closed, open, seam-crossing and wrapping surfaces
 * (plans/16 §4.1, §4.2).
 *
 * Every kernel evaluates in a local frame: minimum-image differences from an anchor vertex
 * (every edge is shorter than half a period, §3.5), so seam-crossing, wrapping and open
 * surfaces are handled alike, and non-periodic arithmetic reduces to plain differences.
 *
 *  - MFTLVertexDelta / MFTLEdgeVector / MFTLVertexContinuous / MFTLTriangleUnwrapped;
 *  - MFTLBoundaryEdgeGeometry: length, tangent t, in-surface outward conormal nu = t x n and the
 *    continuous midpoint of a BOUNDARY half-edge;
 *  - fan quantities: MFTLVertexAngleSum, MFTLVertexTurningAngle (geodesic turning angle
 *    pi - sum of corner angles at a boundary vertex), MFTLBoundaryNeighbours (loop neighbours);
 *  - periodic proximity: MFTLFindThinRegionPairs and MFTLFindProximityPairs (spatial hashing of
 *    wrapped positions on a periodic cell grid, minimum-image distances).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"

void MFTLVertexDelta(const MFTL_Object* obj, const MFTL_Vertex* a, const MFTL_Vertex* b, double d[3]) { MFTLGetMinimumImageVector(obj->domain, d, a->pos, b->pos); }

void MFTLEdgeVector(const MFTL_Object* obj, const MFTL_HalfEdge* he, double d[3])
{
  MFTLGetMinimumImageVector(obj->domain, d, MFTLHeOrigin(obj, he)->pos, MFTLHeTarget(obj, he)->pos);
}

void MFTLVertexContinuous(const MFTL_Object* obj, const MFTL_Vertex* v, double X[3])
{
  for (int k = 0; k < 3; ++k) X[k] = v->pos[k] + (obj->domain->is_periodic[k] ? (double) v->img[k] * obj->domain->domain_size[k] : 0.0);
}

void MFTLTriangleUnwrapped(const MFTL_Object* obj, const MFTL_Triangle* tri, double P[3][3])
{
  MFTL_Vertex* v[3];
  MFTLGetTriangleVertices(obj, tri, v);
  memcpy(P[0], v[0]->pos, 3 * sizeof(double));
  for (int c = 1; c < 3; ++c)
  {
    double d[3];
    MFTLGetMinimumImageVector(obj->domain, d, v[0]->pos, v[c]->pos);
    for (int k = 0; k < 3; ++k) P[c][k] = P[0][k] + d[k];
  }
}

int MFTLBoundaryEdgeGeometry(const MFTL_Object* obj, int he_idx, double* len, double t[3], double nu[3], double mid_X[3])
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (!MFTLHeIsBoundary(he)) return MFTL_ERROR;
  const MFTL_Vertex* a = MFTLHeOrigin(obj, he);
  double d[3];
  MFTLGetMinimumImageVector(obj->domain, d, a->pos, MFTLHeTarget(obj, he)->pos);
  const double l = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (!(l > 0.0)) return MFTL_ERROR;
  // unit normal of the half-edge's triangle (local frame)
  double P[3][3];
  MFTLTriangleUnwrapped(obj, &obj->triangles[he->triangle_idx], P);
  double e1[3] = {P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]};
  double e2[3] = {P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  const double nl = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (!(nl > 0.0)) return MFTL_ERROR;
  for (int k = 0; k < 3; ++k)
  {
    t[k] = d[k] / l;
    n[k] /= nl;
  }
  // in-surface outward conormal: the triangle lies to the left of t (CCW about n)
  nu[0] = t[1] * n[2] - t[2] * n[1];
  nu[1] = t[2] * n[0] - t[0] * n[2];
  nu[2] = t[0] * n[1] - t[1] * n[0];
  *len = l;
  if (mid_X)
  {
    MFTLVertexContinuous(obj, a, mid_X);
    for (int k = 0; k < 3; ++k) mid_X[k] += 0.5 * d[k];
  }
  return MFTL_SUCCESS;
}

/* For the surface-scalar Dirichlet condition (plans/16 §6): the in-surface distance delta_e from the centroid of the boundary
   half-edge's triangle to the edge midpoint, and the triangle height h_T = 2 A_T / L_e over the
   edge, both in the triangle's local frame. Geometry only: the Dirichlet floor is the
   application's choice. */
int MFTLBoundaryEdgeCentroidDistance(const MFTL_Object* obj, int he_idx, double* delta, double* h_T)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (!MFTLHeIsBoundary(he)) return MFTL_ERROR;
  const MFTL_Vertex* a = MFTLHeOrigin(obj, he);
  double P[3][3], pa[3], d[3];
  MFTLTriangleUnwrapped(obj, &obj->triangles[he->triangle_idx], P);
  // the edge endpoints in the same frame as the triangle corners (P[0] is the stored position of
  // the triangle's first corner)
  MFTLGetMinimumImageVector(obj->domain, pa, P[0], a->pos);
  for (int k = 0; k < 3; ++k) pa[k] += P[0][k];
  MFTLGetMinimumImageVector(obj->domain, d, a->pos, MFTLHeTarget(obj, he)->pos);
  const double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  double e1[3] = {P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]};
  double e2[3] = {P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  const double twoA = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (!(L > 0.0) || !(twoA > 0.0)) return MFTL_ERROR;
  double r[3];
  for (int k = 0; k < 3; ++k) r[k] = (P[0][k] + P[1][k] + P[2][k]) / 3.0 - (pa[k] + 0.5 * d[k]);
  *delta = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  *h_T = twoA / L;
  return MFTL_SUCCESS;
}

/* Sum of the corner angles at v over its stored fan. Returns the ring class (MFTL_RING_*); the
   sum is valid for CLOSED and OPEN_COMPLETE. */
int MFTLVertexAngleSum(const MFTL_Object* obj, int v, double* sum)
{
  int ring[256], nr = 0;
  const int cls = MFTLVertexRing(obj, v, ring, 256, &nr);
  *sum = 0.0;
  if (cls == MFTL_RING_INCOMPLETE || nr > 256) return MFTL_RING_INCOMPLETE;
  const double* pv = obj->vertices[v].pos;
  for (int i = 0; i < nr; ++i)
  {
    const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];
    const int x = h->vertex_idx, y = obj->half_edges[h->next_idx].vertex_idx;
    double dx[3], dy[3];
    MFTLGetMinimumImageVector(obj->domain, dx, pv, obj->vertices[x].pos);
    MFTLGetMinimumImageVector(obj->domain, dy, pv, obj->vertices[y].pos);
    const double lx = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
    const double ly = sqrt(dy[0] * dy[0] + dy[1] * dy[1] + dy[2] * dy[2]);
    if (!(lx * ly > 0.0)) continue;
    double c = (dx[0] * dy[0] + dx[1] * dy[1] + dx[2] * dy[2]) / (lx * ly);
    c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
    *sum += acos(c);
  }
  return cls;
}

int MFTLVertexTurningAngle(const MFTL_Object* obj, int v, double* theta)
{
  double sum;
  if (MFTLVertexAngleSum(obj, v, &sum) != MFTL_RING_OPEN_COMPLETE) return MFTL_ERROR;
  *theta = MFTL_PI - sum;
  return MFTL_SUCCESS;
}

/* Loop neighbours of a boundary vertex: next = target of v's outgoing BOUNDARY half-edge, prev =
   origin of its incoming BOUNDARY half-edge (-1 when that part of the fan is not stored).
   MFTL_ERROR when no outgoing BOUNDARY half-edge of v is stored. */
int MFTLBoundaryNeighbours(const MFTL_Object* obj, int v, int* prev, int* next)
{
  *prev = *next = -1;
  const MFTL_Vertex* vx = &obj->vertices[v];
  if (vx->out_he_idx < 0 || vx->out_he_idx >= obj->n_half_edges) return MFTL_ERROR;
  // the outgoing BOUNDARY half-edge: the out_he convention, else the end of the next(twin) walk
  int h = vx->out_he_idx, steps = 0;
  const int cap = 4 * obj->n_half_edges + 8;
  while (!MFTLHeIsBoundary(&obj->half_edges[h]))
  {
    const MFTL_HalfEdge* hh = &obj->half_edges[h];
    if (!MFTLHeIsLocal(hh)) return MFTL_ERROR;  // REMOTE: the fan end is not stored
    h = obj->half_edges[hh->twin_idx].next_idx;
    if (h == vx->out_he_idx || ++steps > cap) return MFTL_ERROR;  // closed fan: interior vertex
  }
  *next = obj->half_edges[h].vertex_idx;
  // rewind (twin(prev)) to the incoming BOUNDARY half-edge
  int curr = h;
  steps = 0;
  while (1)
  {
    const int p = obj->half_edges[obj->half_edges[curr].next_idx].next_idx;  // incoming to v
    const MFTL_HalfEdge* hp = &obj->half_edges[p];
    if (MFTLHeIsBoundary(hp))
    {
      *prev = obj->half_edges[obj->half_edges[hp->next_idx].next_idx].vertex_idx;  // origin of p
      return MFTL_SUCCESS;
    }
    if (!MFTLHeIsLocal(hp)) return MFTL_SUCCESS;  // not stored: prev unknown
    curr = hp->twin_idx;
    if (curr == h || ++steps > cap) return MFTL_SUCCESS;
  }
}

/* ------------------------------------------------------------------------------------------
 * periodic proximity (spatial hashing on a periodic cell grid, minimum-image distances)
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  long key;
  int tri;
} mftlCellRec;

static int mftlCmpCell(const void* a, const void* b)
{
  const mftlCellRec* x = (const mftlCellRec*) a;
  const mftlCellRec* y = (const mftlCellRec*) b;
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  return (x->tri > y->tri) - (x->tri < y->tri);
}

typedef struct
{
  const MFTL_Domain* dom;
  double h;  // cell size
  long nc[3];  // cells per periodic axis (0 = non-periodic: unbounded index)
} mftlGrid;

static void mftlGridInit(mftlGrid* g, const MFTL_Domain* dom, double h)
{
  g->dom = dom;
  g->h = h;
  for (int k = 0; k < 3; ++k)
  {
    g->nc[k] = 0;
    if (dom->is_periodic[k])
    {
      long n = (long) floor(dom->domain_size[k] / h);
      g->nc[k] = n < 1 ? 1 : n;
    }
  }
}

static void mftlGridCell(const mftlGrid* g, const double x[3], long c[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (g->nc[k] > 0)
    {
      const double L = g->dom->domain_size[k];
      double u = (x[k] - g->dom->global_min[k]) / L;
      u -= floor(u);
      long i = (long) floor(u * (double) g->nc[k]);
      if (i >= g->nc[k]) i = g->nc[k] - 1;
      if (i < 0) i = 0;
      c[k] = i;
    }
    else
      c[k] = (long) floor(x[k] / g->h);
  }
}

static long mftlGridKey(const long c[3]) { return ((c[0] & 0x1FFFFF) << 42) | ((c[1] & 0x1FFFFF) << 21) | (c[2] & 0x1FFFFF); }

/* the (deduplicated) neighbour cells within one cell of c */
static int mftlGridNeighbours(const mftlGrid* g, const long c[3], long* keys)
{
  int n = 0;
  long d[3];
  for (d[0] = -1; d[0] <= 1; ++d[0])
    for (d[1] = -1; d[1] <= 1; ++d[1])
      for (d[2] = -1; d[2] <= 1; ++d[2])
      {
        long q[3];
        for (int k = 0; k < 3; ++k)
        {
          q[k] = c[k] + d[k];
          if (g->nc[k] > 0) q[k] = ((q[k] % g->nc[k]) + g->nc[k]) % g->nc[k];
        }
        const long key = mftlGridKey(q);
        int dup = 0;
        for (int i = 0; i < n; ++i)
          if (keys[i] == key) dup = 1;
        if (!dup) keys[n++] = key;
      }
  return n;
}

static void mftlPushPair(int** pairs, int* n, int* cap, int a, int b)
{
  if (*n == *cap)
  {
    *cap = *cap ? 2 * *cap : 256;
    *pairs = realloc(*pairs, (size_t) (2 * *cap) * sizeof(int));
  }
  (*pairs)[2 * *n] = a;
  (*pairs)[2 * *n + 1] = b;
  (*n)++;
}

/* Thin-region pairs (breakup detection): pairs of stored triangles with opposing normals, centre
   distance (minimum image) below threshold and no shared vertex. pairs = (gid, gid) with
   gid0 < gid1 in stored order; rank-local. Returns the pair count. */
int MFTLFindThinRegionPairs(MFTL_Object* obj, double threshold, int** pairs, int* npairs)
{
  *pairs = MFTL_NULL;
  *npairs = 0;
  const int n = obj->n_triangles;
  if (n == 0 || !(threshold > 0.0)) return 0;
  mftlGrid g;
  mftlGridInit(&g, obj->domain, threshold);
  mftlCellRec* cells = malloc((size_t) n * sizeof(mftlCellRec));
  for (int t = 0; t < n; ++t)
  {
    long c[3];
    mftlGridCell(&g, obj->triangles[t].center, c);
    cells[t].key = mftlGridKey(c);
    cells[t].tri = t;
  }
  qsort(cells, (size_t) n, sizeof(mftlCellRec), mftlCmpCell);
  int cap = 0;
  long keys[27];
  for (int t0 = 0; t0 < n; ++t0)
  {
    const MFTL_Triangle* T0 = &obj->triangles[t0];
    if (T0->flags & MFTL_FLAG_DEAD) continue;
    long c0[3];
    mftlGridCell(&g, T0->center, c0);
    const int nk = mftlGridNeighbours(&g, c0, keys);
    MFTL_Vertex* v0[3];
    MFTLGetTriangleVertices(obj, T0, v0);
    for (int q = 0; q < nk; ++q)
    {
      int lo = 0, hi = n;
      while (lo < hi)
      {
        int mid = (lo + hi) / 2;
        if (cells[mid].key < keys[q])
          lo = mid + 1;
        else
          hi = mid;
      }
      for (int k = lo; k < n && cells[k].key == keys[q]; ++k)
      {
        const int t1 = cells[k].tri;
        if (t1 <= t0) continue;
        const MFTL_Triangle* T1 = &obj->triangles[t1];
        if (T1->flags & MFTL_FLAG_DEAD) continue;
        const double dot = T0->normal[0] * T1->normal[0] + T0->normal[1] * T1->normal[1] + T0->normal[2] * T1->normal[2];
        if (!(dot < 0.0)) continue;
        double d[3];
        MFTLGetMinimumImageVector(obj->domain, d, T0->center, T1->center);
        if (!(sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) < threshold)) continue;
        MFTL_Vertex* v1[3];
        MFTLGetTriangleVertices(obj, T1, v1);
        int shared = 0;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            if (v0[a] == v1[b]) shared = 1;
        if (!shared) mftlPushPair(pairs, npairs, &cap, T0->global_id, T1->global_id);
      }
    }
  }
  free(cells);
  return *npairs;
}

/* per-triangle local-frame box: centre (corner 0 frame, wrapped) and half extents */
static void mftlTriBox(const MFTL_Object* obj, int t, double c[3], double e[3])
{
  double P[3][3];
  MFTLTriangleUnwrapped(obj, &obj->triangles[t], P);
  for (int k = 0; k < 3; ++k)
  {
    double lo = fmin(P[0][k], fmin(P[1][k], P[2][k])), hi = fmax(P[0][k], fmax(P[1][k], P[2][k]));
    c[k] = 0.5 * (lo + hi);
    e[k] = 0.5 * (hi - lo);
  }
}

/* Proximity pairs (coalescence detection): (stored triangle of o0, stored triangle of o1) whose
   local-frame bounding boxes are closer than `distance` in the minimum image (per-axis
   separation <= e0 + e1 + distance). pairs = (gid in o0, gid in o1); rank-local. */
int MFTLFindProximityPairs(MFTL_Object* o0, MFTL_Object* o1, double distance, int** pairs, int* npairs)
{
  *pairs = MFTL_NULL;
  *npairs = 0;
  const int n1 = o1->n_triangles, n0 = o0->n_triangles;
  if (n0 == 0 || n1 == 0) return 0;
  const MFTL_Domain* dom = o0->domain;
  double emax = 0.0;
  double(*c1)[3] = malloc((size_t) n1 * sizeof(*c1));
  double(*e1)[3] = malloc((size_t) n1 * sizeof(*e1));
  for (int t = 0; t < n1; ++t)
  {
    mftlTriBox(o1, t, c1[t], e1[t]);
    for (int k = 0; k < 3; ++k) emax = fmax(emax, e1[t][k]);
  }
  double emax0 = 0.0;
  double(*c0)[3] = malloc((size_t) n0 * sizeof(*c0));
  double(*e0)[3] = malloc((size_t) n0 * sizeof(*e0));
  for (int t = 0; t < n0; ++t)
  {
    mftlTriBox(o0, t, c0[t], e0[t]);
    for (int k = 0; k < 3; ++k) emax0 = fmax(emax0, e0[t][k]);
  }
  // cell size >= the largest possible centre separation of a pair: e0 + e1 + distance
  const double h = fmax(emax + emax0 + distance, 1e-300);
  mftlGrid g;
  mftlGridInit(&g, dom, h);
  mftlCellRec* cells = malloc((size_t) n1 * sizeof(mftlCellRec));
  for (int t = 0; t < n1; ++t)
  {
    long c[3];
    mftlGridCell(&g, c1[t], c);
    cells[t].key = mftlGridKey(c);
    cells[t].tri = t;
  }
  qsort(cells, (size_t) n1, sizeof(mftlCellRec), mftlCmpCell);
  int cap = 0;
  long keys[27];
  for (int s = 0; s < n0; ++s)
  {
    if (o0->triangles[s].flags & MFTL_FLAG_DEAD) continue;
    long c[3];
    mftlGridCell(&g, c0[s], c);
    const int nk = mftlGridNeighbours(&g, c, keys);
    for (int q = 0; q < nk; ++q)
    {
      int lo = 0, hi = n1;
      while (lo < hi)
      {
        int mid = (lo + hi) / 2;
        if (cells[mid].key < keys[q])
          lo = mid + 1;
        else
          hi = mid;
      }
      for (int k = lo; k < n1 && cells[k].key == keys[q]; ++k)
      {
        const int t = cells[k].tri;
        if (o1->triangles[t].flags & MFTL_FLAG_DEAD) continue;
        double d[3];
        MFTLGetMinimumImageVector(dom, d, c0[s], c1[t]);
        int ov = 1;
        for (int a = 0; a < 3; ++a)
          if (fabs(d[a]) > e0[s][a] + e1[t][a] + distance) ov = 0;
        if (ov) mftlPushPair(pairs, npairs, &cap, o0->triangles[s].global_id, o1->triangles[t].global_id);
      }
    }
  }
  free(cells);
  free(c0);
  free(e0);
  free(c1);
  free(e1);
  return *npairs;
}

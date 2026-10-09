/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Nearest-surface queries (plan 06, C3, for plan 07): the candidate's
 *        total order, and the distributed query against brute force and against
 *        itself at another process count, bit for bit.
 *
 * Determinism of a merge is a property of the ORDER, so the order is tested on
 * its own first: antisymmetric, transitive, equal only for identical records,
 * with every tie the geometry can produce (equal distances of opposite sign,
 * -0 and +0, two bodies, two triangles of one body, two features of one
 * triangle), and merges in random orders and random tree shapes agreeing with a
 * sort by the test's own comparator. Then the query:
 *
 *  - against brute force over the whole surface gathered on every rank, with
 *    the test's own pseudo-normals: distances, signs, bodies, triangles, on a
 *    lattice and a scattered point cloud, spheres across a periodic seam;
 *  - against a rerun on rank 0 alone (MPI_COMM_SELF): the merged records, bit
 *    for bit, at np = 2, 4, 8, 27 -- on spheres, and on a cube whose corners
 *    and edges are lattice samples, where exact ties are everywhere;
 *  - non-vacuity: samples whose nearest triangle another rank owns;
 *  - two ensembles queried into one array equal one ensemble holding both;
 *  - a query wider than its window, a moved surface, a keyed window, bad
 *    widths and duplicate fields are agreed refusals.
 */

#include "test_coupling_irregular.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. The order
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The specification, written out again: valid, |d|, sign (inside first), body, triangle, feature, offsets (-0 first). */
static int spec_cmp(const MFTL_SurfaceCandidate *a, const MFTL_SurfaceCandidate *b)
{
  const int va = a->valid == 1 && isfinite(a->distance) && isfinite(a->offset[0]) && isfinite(a->offset[1]) && isfinite(a->offset[2]);
  const int vb = b->valid == 1 && isfinite(b->distance) && isfinite(b->offset[0]) && isfinite(b->offset[1]) && isfinite(b->offset[2]);
  if (va != vb) return vb - va;
  if (!va) return 0;
  if (fabs(a->distance) != fabs(b->distance)) return fabs(a->distance) < fabs(b->distance) ? -1 : 1;
  if (!!signbit(a->distance) != !!signbit(b->distance)) return signbit(a->distance) ? -1 : 1;
  if (a->body != b->body) return a->body < b->body ? -1 : 1;
  if (a->triangle != b->triangle) return a->triangle < b->triangle ? -1 : 1;
  if (a->feature != b->feature) return a->feature < b->feature ? -1 : 1;
  for (int k = 0; k < 3; ++k)
  {
    if (a->offset[k] != b->offset[k]) return a->offset[k] < b->offset[k] ? -1 : 1;
    if (!!signbit(a->offset[k]) != !!signbit(b->offset[k])) return signbit(a->offset[k]) ? -1 : 1;
  }
  return 0;
}

static int same_record(const MFTL_SurfaceCandidate *a, const MFTL_SurfaceCandidate *b)
{
  if (a->valid != b->valid || a->body != b->body || a->triangle != b->triangle || a->feature != b->feature) return 0;
  if (memcmp(&a->distance, &b->distance, sizeof(double)) != 0) return 0;
  return memcmp(a->offset, b->offset, sizeof a->offset) == 0;
}

static int cmp_sort(const void *x, const void *y) { return spec_cmp((const MFTL_SurfaceCandidate *) x, (const MFTL_SurfaceCandidate *) y); }

static int sgn(int v) { return (v > 0) - (v < 0); }

/**
 * A pool built to collide: distances of +-0.5, +-0.0, 0.25, NaN and infinity;
 * two bodies, two triangles, three features, offsets with signed zeros. The
 * library's order agrees with the specification on every pair, is
 * antisymmetric and equal only for identical records; merging the pool in 300
 * random orders and 300 random tree shapes gives the specification's minimum
 * every time; an all-unknown merge is the canonical unknown.
 */
static int test_the_merge_is_a_total_order(void)
{
  enum
  {
    N = 400
  };
  MFTL_SurfaceCandidate *pool = (MFTL_SurfaceCandidate *) calloc(N, sizeof(MFTL_SurfaceCandidate));
  const double ds[7] = {0.5, -0.5, 0.0, -0.0, 0.25, NAN, INFINITY}, os[5] = {0.1, -0.1, 0.0, -0.0, 0.2};
  uint64_t s = 17;
  for (int i = 0; i < N; ++i)
  {
    MFTL_SurfaceCandidate *c = &pool[i];
    MFTLSurfaceCandidateReset(c);
    c->valid = (adv_next(&s) % 9u) != 0;
    c->distance = ds[adv_next(&s) % 7u];
    c->body = (int) (adv_next(&s) % 2u);
    c->triangle = 3 + 4 * (int64_t) (adv_next(&s) % 2u);
    c->feature = (int) (adv_next(&s) % 3u) * 2;
    for (int k = 0; k < 3; ++k) c->offset[k] = os[adv_next(&s) % 5u];
  }
  int bad = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
    {
      const int a = MFTLSurfaceCandidateCompare(&pool[i], &pool[j]), b = MFTLSurfaceCandidateCompare(&pool[j], &pool[i]);
      if (sgn(a) != sgn(spec_cmp(&pool[i], &pool[j]))) ++bad;
      if (sgn(a) != -sgn(b)) ++bad;
      const int valid = spec_cmp(&pool[i], &pool[i]) == 0 && pool[i].valid == 1 && isfinite(pool[i].distance);
      if (a == 0 && valid && !same_record(&pool[i], &pool[j])) ++bad;
    }
  TEST_ASSERT(bad == 0);
  /* the specification's minimum */
  MFTL_SurfaceCandidate *sorted = (MFTL_SurfaceCandidate *) malloc(N * sizeof(MFTL_SurfaceCandidate));
  memcpy(sorted, pool, N * sizeof(MFTL_SurfaceCandidate));
  qsort(sorted, N, sizeof(MFTL_SurfaceCandidate), cmp_sort);
  const MFTL_SurfaceCandidate want = sorted[0];
  int perm[N];
  for (int rep = 0; rep < 300; ++rep)
  {
    for (int i = 0; i < N; ++i) perm[i] = i;
    for (int i = N - 1; i > 0; --i)
    {
      const int j = (int) (adv_next(&s) % (uint64_t) (i + 1)), t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
    }
    MFTL_SurfaceCandidate acc;
    MFTLSurfaceCandidateReset(&acc);
    for (int i = 0; i < N; ++i) MFTLSurfaceCandidateMerge(&acc, &pool[perm[i]]);
    if (!same_record(&acc, &want)) ++bad;
    /* a random tree: repeatedly merge two random survivors */
    MFTL_SurfaceCandidate work[N];
    int n = N;
    for (int i = 0; i < N; ++i) work[i] = pool[perm[i]];
    while (n > 1)
    {
      const int a = (int) (adv_next(&s) % (uint64_t) n);
      int b = (int) (adv_next(&s) % (uint64_t) (n - 1));
      if (b >= a) ++b;
      MFTLSurfaceCandidateMerge(&work[a], &work[b]);
      work[b] = work[--n];
      if (a == n) continue;
    }
    if (!same_record(&work[0], &want)) ++bad;
  }
  TEST_ASSERT(bad == 0);
  /* the named ties */
  MFTL_SurfaceCandidate p, m;
  MFTLSurfaceCandidateReset(&p);
  p.valid = 1;
  p.distance = 0.0;
  p.body = 0;
  p.triangle = 0;
  m = p;
  m.distance = -0.0;
  m.body = 9;
  TEST_ASSERT(MFTLSurfaceCandidateCompare(&m, &p) < 0); /* -0 first, whatever the body */
  m.distance = -0.25;
  p.distance = 0.25;
  TEST_ASSERT(MFTLSurfaceCandidateCompare(&m, &p) < 0); /* inside first at equal distance */
  p.distance = NAN;
  m.distance = 1e300;
  TEST_ASSERT(MFTLSurfaceCandidateCompare(&m, &p) < 0); /* NaN never wins */
  MFTL_SurfaceCandidate u1, u2, r;
  MFTLSurfaceCandidateReset(&r);
  u1 = p;
  u2 = p;
  u2.offset[1] = INFINITY;
  u2.distance = 0.1;
  MFTLSurfaceCandidateMerge(&u1, &u2);
  TEST_ASSERT(same_record(&u1, &r)); /* unknown merged with unknown: the canonical unknown */
  if (g_rank == 0) printf("    total order: %d pairs, 600 merge orders agree with the specification\n", N * N);
  free(pool);
  free(sorted);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. The distributed query
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Every owned triangle of the job, on every rank: body, id, owner, corners (centroid frame), vertex ids. */
typedef struct
{
  int n;
  double *T;  ///< [9n]
  int *body, *owner;
  int64_t *gid;
  int *vid;  ///< [3n]
  int *nbr;  ///< [3n] the triangle across edge e (corner e -> e+1), -1 at a rim
} all_tris;

static void gather_tris(const MFTL_Ensemble *ens, all_tris *A)
{
  /* per triangle: 9 corners, body, gid, owner, 3 vertex ids, as doubles */
  enum
  {
    W = 15
  };
  int mine = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) mine += ens->local_objects[o]->tris.n_local;
  double *buf = (double *) malloc((size_t) (mine > 0 ? mine : 1) * W * sizeof(double));
  int at = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t, ++at)
    {
      double T[3][3];
      int vv[3];
      MFTLGetTriangleCorners(obj, t, T);
      MFTLGetTriangleVertices(obj, t, vv);
      for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) buf[W * at + 3 * i + k] = T[i][k];
      buf[W * at + 9] = obj->global_id;
      buf[W * at + 10] = (double) obj->tris.rec[t].global_id;
      buf[W * at + 11] = g_rank;
      for (int i = 0; i < 3; ++i) buf[W * at + 12 + i] = obj->verts.rec[vv[i]].global_id;
    }
  }
  int n_tot = 0;
  double *all = allgather_doubles(buf, mine, W, &n_tot);
  free(buf);
  A->n = n_tot;
  A->T = (double *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * 9 * sizeof(double));
  A->body = (int *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * sizeof(int));
  A->owner = (int *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * sizeof(int));
  A->gid = (int64_t *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * sizeof(int64_t));
  A->vid = (int *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * 3 * sizeof(int));
  for (int i = 0; i < n_tot; ++i)
  {
    memcpy(&A->T[9 * i], &all[W * i], 9 * sizeof(double));
    A->body[i] = (int) all[W * i + 9];
    A->gid[i] = (int64_t) all[W * i + 10];
    A->owner[i] = (int) all[W * i + 11];
    for (int k = 0; k < 3; ++k) A->vid[3 * i + k] = (int) all[W * i + 12 + k];
  }
  free(all);
  A->nbr = (int *) malloc((size_t) (n_tot > 0 ? n_tot : 1) * 3 * sizeof(int));
  for (int i = 0; i < n_tot; ++i)
    for (int e = 0; e < 3; ++e)
    {
      A->nbr[3 * i + e] = -1;
      const int va = A->vid[3 * i + e], vb = A->vid[3 * i + (e + 1) % 3];
      for (int j = 0; j < n_tot && A->nbr[3 * i + e] < 0; ++j)
      {
        if (j == i || A->body[j] != A->body[i]) continue;
        for (int c = 0; c < 3; ++c)
          if (A->vid[3 * j + c] == vb && A->vid[3 * j + (c + 1) % 3] == va) A->nbr[3 * i + e] = j;
      }
    }
}

static void free_tris(all_tris *A)
{
  free(A->T);
  free(A->body);
  free(A->owner);
  free(A->gid);
  free(A->vid);
  free(A->nbr);
}

static void unit_normal(const double *T, double n[3])
{
  const double e1[3] = {T[3] - T[0], T[4] - T[1], T[5] - T[2]}, e2[3] = {T[6] - T[0], T[7] - T[1], T[8] - T[2]};
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const double m = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (int k = 0; k < 3; ++k) n[k] = (m > 0) ? n[k] / m : 0.0;
}

/** @brief The reference at one sample: the least candidate over every triangle, and the runner-up's distance. */
typedef struct
{
  MFTL_SurfaceCandidate best;
  double second;  ///< |d| of the best candidate from ANOTHER triangle
  double dot_rel;  ///< |dot| / |d| of the best (how clear its sign is)
} ref_result;

static void reference_at(const MFTL_Domain *d, const all_tris *A, const double *vn, const int *vn_key_body, const int *vn_key_vid, int n_vn, const double X[3],
                         double width, ref_result *R)
{
  MFTLSurfaceCandidateReset(&R->best);
  R->second = INFINITY;
  R->dot_rel = 0.0;
  for (int i = 0; i < A->n; ++i)
  {
    double T[3][3];
    for (int a = 0; a < 3; ++a)
      for (int k = 0; k < 3; ++k) T[a][k] = A->T[9 * i + 3 * a + k];
    /* shift the triangle to the sample's image */
    for (int k = 0; k < 3; ++k)
      if (d->is_periodic[k])
      {
        const double c = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
        const double sh = d->domain_size[k] * nearbyint((X[k] - c) / d->domain_size[k]);
        for (int a = 0; a < 3; ++a) T[a][k] += sh;
      }
    double q[3];
    MFTL_TriangleFeature f;
    const double d2 = MFTLClosestPointOnTriangle(X, T, q, &f);
    if (d2 > width * width) continue;
    double n[3], Tf[9];
    for (int a = 0; a < 3; ++a)
      for (int k = 0; k < 3; ++k) Tf[3 * a + k] = T[a][k];
    unit_normal(Tf, n);
    if (f >= MFTL_TRI_FEATURE_EDGE0)
    {
      const int j = A->nbr[3 * i + (f - MFTL_TRI_FEATURE_EDGE0)];
      if (j >= 0)
      {
        double nt[3];
        unit_normal(&A->T[9 * j], nt);
        for (int k = 0; k < 3; ++k) n[k] += nt[k];
      }
    }
    else if (f >= MFTL_TRI_FEATURE_VERTEX0)
    {
      const int v = A->vid[3 * i + (f - MFTL_TRI_FEATURE_VERTEX0)];
      for (int j = 0; j < n_vn; ++j)
        if (vn_key_body[j] == A->body[i] && vn_key_vid[j] == v) memcpy(n, &vn[3 * j], sizeof n);
    }
    MFTL_SurfaceCandidate c;
    MFTLSurfaceCandidateReset(&c);
    for (int k = 0; k < 3; ++k) c.offset[k] = X[k] - q[k];
    const double dist = sqrt(d2), dot = c.offset[0] * n[0] + c.offset[1] * n[1] + c.offset[2] * n[2];
    c.distance = (dot < 0) ? -dist : dist;
    c.valid = 1;
    c.body = A->body[i];
    c.triangle = A->gid[i];
    c.feature = (int) f;
    if (spec_cmp(&c, &R->best) < 0)
    {
      if (R->best.valid && (R->best.triangle != c.triangle || R->best.body != c.body)) R->second = fmin(R->second, fabs(R->best.distance));
      R->best = c;
      R->dot_rel = (dist > 0) ? fabs(dot) / dist : 0.0;
    }
    else if (c.triangle != R->best.triangle || c.body != R->best.body)
      R->second = fmin(R->second, dist);
  }
}

/** @brief Angle-weighted vertex normals over the gathered triangles: (body, vertex id) -> normal. */
static int vertex_normals(const all_tris *A, int **kb, int **kv, double **vn)
{
  int *b = (int *) malloc((size_t) (3 * A->n + 1) * sizeof(int)), *v = (int *) malloc((size_t) (3 * A->n + 1) * sizeof(int));
  double *n = (double *) calloc((size_t) (3 * A->n + 1) * 3, sizeof(double));
  int m = 0;
  for (int i = 0; i < A->n; ++i)
  {
    double nf[3];
    unit_normal(&A->T[9 * i], nf);
    for (int c = 0; c < 3; ++c)
    {
      int at = -1;
      for (int j = 0; j < m; ++j)
        if (b[j] == A->body[i] && v[j] == A->vid[3 * i + c]) at = j;
      if (at < 0)
      {
        at = m++;
        b[at] = A->body[i];
        v[at] = A->vid[3 * i + c];
      }
      const double *P = &A->T[9 * i];
      const double *a = &P[3 * c], *bb = &P[3 * ((c + 1) % 3)], *e = &P[3 * ((c + 2) % 3)];
      const double e1[3] = {bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]}, e2[3] = {e[0] - a[0], e[1] - a[1], e[2] - a[2]};
      const double cr = sqrt(pow(e1[1] * e2[2] - e1[2] * e2[1], 2) + pow(e1[2] * e2[0] - e1[0] * e2[2], 2) + pow(e1[0] * e2[1] - e1[1] * e2[0], 2));
      const double ang = atan2(cr, e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]);
      for (int k = 0; k < 3; ++k) n[3 * at + k] += ang * nf[k];
    }
  }
  for (int j = 0; j < m; ++j)
  {
    const double mag = sqrt(n[3 * j] * n[3 * j] + n[3 * j + 1] * n[3 * j + 1] + n[3 * j + 2] * n[3 * j + 2]);
    for (int k = 0; k < 3; ++k) n[3 * j + k] /= mag;
  }
  *kb = b;
  *kv = v;
  *vn = n;
  return m;
}

/** @brief The query scene: spheres in a box periodic in x and z, one across the x seam. */
static MFTL_Ensemble *query_scene(MPI_Comm comm, bool two_bodies_second_only)
{
  int np;
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {true, false, true};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(comm, cuts, lo, hi, per, 4);
  if (!two_bodies_second_only)
  {
    MFTL_Object *a = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(ens, a);
    double ca[3] = {0.97, 0.45, 0.52};
    MFTLGenerateSphere(a, ca, 0.21, 2);
  }
  MFTL_Object *b = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, b);
  double cb[3] = {0.43, 0.5, 0.27};
  MFTLGenerateSphere(b, cb, 0.17, 2);
  /* Needles, the shapes where a vertex's sign needs its whole fan. On a
   * convex surface any one face normal at a vertex gives the right sign
   * wherever the vertex is the nearest point, so a sphere cannot tell a
   * pseudo-normal from a face normal. Outward spikes (1.6 R) put samples
   * beside a sharp tip, nearest to it, on the far side of one face's plane;
   * inward needles (0.3 R) put INSIDE samples nearest to their tip. */
  for (int v = 0; v < b->verts.n; ++v)
  {
    const double f = (v % 23 == 0) ? 0.3 : ((v % 23 == 11) ? 1.6 : 1.0);
    for (int k = 0; k < 3; ++k) b->verts.pos[3 * v + k] = cb[k] + f * (b->verts.pos[3 * v + k] - cb[k]);
  }
  MFTLObjectMarkChanged(b);
  MFTLComputeTriangleGeometry(b, b->tris.n);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  return ens;
}

/** @brief A lattice location of N^3 cell centres in [0,1]^3, owned by z-slabs of @p comm, and its result array. */
typedef struct
{
  int loc, N;
  int64_t lo[3], hi[3];
  MFTL_SurfaceCandidate *res;
  MFTL_SurfaceCandidate *ptr[1];
} qlat;

static void qlat_add(MFTL_SolverGrid *g, MPI_Comm comm, int N, qlat *Q)
{
  int r, np;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &np);
  memset(Q, 0, sizeof *Q);
  Q->N = N;
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "cells";
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = N;
    ld.lattice.spacing[k] = 1.0 / N;
    ld.lattice.origin[k] = 0.5 / N;
  }
  Q->loc = MFTLSolverGridAddLocation(g, &ld);
  for (int k = 0; k < 2; ++k)
  {
    Q->lo[k] = 0;
    Q->hi[k] = N;
  }
  Q->lo[2] = (int64_t) N * r / np;
  Q->hi[2] = (int64_t) N * (r + 1) / np;
  const int64_t n = (int64_t) N * N * (Q->hi[2] - Q->lo[2]);
  Q->res = (MFTL_SurfaceCandidate *) malloc((size_t) (n > 0 ? n : 1) * sizeof(MFTL_SurfaceCandidate));
  for (int64_t i = 0; i < n; ++i) MFTLSurfaceCandidateReset(&Q->res[i]);
  Q->ptr[0] = Q->res;
  if (n > 0)
  {
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = Q->loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = Q->lo[k];
      bd.owned_hi[k] = Q->hi[k];
      bd.array_lo[k] = Q->lo[k];
      bd.array_dims[k] = Q->hi[k] - Q->lo[k];
    }
    MFTLSolverGridAddBlock(g, &bd);
  }
}

static MFTL_QueryField qlat_field(qlat *Q)
{
  MFTL_QueryField f = {Q->loc, Q->ptr, 1};
  return f;
}

static int run_query(MFTL_Coupler *c, int n_loc, const int *locs, double width, int nf, const MFTL_QueryField *F)
{
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestQuery(c, n_loc, locs, width, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerQueryNearest(c, w, width, nf, F);
  MFTLCouplerReleaseWindow(&w);
  return st;
}

/**
 * The query against brute force over the whole surface, on a lattice and on a
 * scattered cloud, with spheres across the periodic seam: every sample within
 * the width -- and only those -- receives the nearest triangle, its distance,
 * its sign (wherever the sign is not a rounding question) and its body.
 */
static int test_the_query_equals_brute_force(void)
{
  MFTL_Ensemble *ens = query_scene(MPI_COMM_WORLD, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  qlat Q;
  qlat_add(g, MPI_COMM_WORLD, 24, &Q);
  pts_block P;
  add_points(g, "cloud", 14, 0.6, OWN_SCATTERED, false, &P);
  MFTL_SurfaceCandidate *pres = (MFTL_SurfaceCandidate *) malloc((size_t) (P.n > 0 ? P.n : 1) * sizeof(MFTL_SurfaceCandidate));
  for (int64_t i = 0; i < P.n; ++i) MFTLSurfaceCandidateReset(&pres[i]);
  MFTL_SurfaceCandidate *pptr[1] = {pres};
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double width = 2.5 / 24.0;
  const int locs[2] = {Q.loc, P.loc};
  MFTL_QueryField F[2] = {qlat_field(&Q), {P.loc, pptr, 1}};
  TEST_ASSERT(agreed(run_query(c, 2, locs, width, 2, F) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  all_tris A;
  gather_tris(ens, &A);
  int *kb, *kv;
  double *vn;
  const int nvn = vertex_normals(&A, &kb, &kv, &vn);
  long counts[8] = {0, 0, 0, 0, 0, 0, 0, 0}; /* checked, valid, distance bad, sign bad, triangle bad, cross-rank, vertex nearest outside, inside */
  double worst = 0.0;
  for (int pass = 0; pass < 2; ++pass)
  {
    const int64_t n = (pass == 0) ? (int64_t) Q.N * Q.N * (Q.hi[2] - Q.lo[2]) : P.n;
    for (int64_t i = 0; i < n; ++i)
    {
      double X[3];
      const MFTL_SurfaceCandidate *got;
      if (pass == 0)
      {
        const int64_t ix = i % Q.N, iy = (i / Q.N) % Q.N, iz = Q.lo[2] + i / ((int64_t) Q.N * Q.N);
        X[0] = (ix + 0.5) / Q.N;
        X[1] = (iy + 0.5) / Q.N;
        X[2] = (iz + 0.5) / Q.N;
        got = &Q.res[i];
      }
      else
      {
        memcpy(X, &P.pos[3 * i], sizeof X);
        got = &pres[i];
      }
      ref_result R;
      reference_at(&ens->domain, &A, vn, kb, kv, nvn, X, width, &R);
      ++counts[0];
      if (R.best.valid != got->valid)
      {
        /* a sample on the edge of the width by a rounding error may go either way */
        if (!(R.best.valid && fabs(fabs(R.best.distance) - width) <= 1e-12) && !(got->valid && fabs(fabs(got->distance) - width) <= 1e-12)) ++counts[2];
        continue;
      }
      if (!got->valid) continue;
      ++counts[1];
      worst = fmax(worst, fabs(fabs(got->distance) - fabs(R.best.distance)));
      if (fabs(fabs(got->distance) - fabs(R.best.distance)) > 1e-13) ++counts[2];
      if (R.dot_rel > 1e-9 && !!signbit(got->distance) != !!signbit(R.best.distance)) ++counts[3];
      if (R.second - fabs(R.best.distance) > 1e-12 && (got->triangle != R.best.triangle || got->body != R.best.body)) ++counts[4];
      for (int t = 0; t < A.n; ++t)
        if (A.gid[t] == got->triangle && A.body[t] == got->body && A.owner[t] != g_rank) ++counts[5];
      if (got->feature >= MFTL_TRI_FEATURE_VERTEX0 && got->feature < MFTL_TRI_FEATURE_EDGE0) ++counts[(got->distance < 0.0) ? 7 : 6];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, counts, 8, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0)
    printf("    query vs brute force: %ld samples, %ld within the width (nearest a vertex: %ld outside, %ld inside), distance %.1e; wrong: validity/distance "
           "%ld, sign %ld, triangle %ld; cross-rank %ld\n",
           counts[0], counts[1], counts[6], counts[7], worst, counts[2], counts[3], counts[4], counts[5]);
  TEST_ASSERT(agreed(counts[2] == 0 && counts[3] == 0 && counts[4] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(counts[1] > 1000 && counts[6] > 20 && counts[7] > 5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || counts[5] > 0, MPI_COMM_WORLD));
  free(kb);
  free(kv);
  free(vn);
  free_tris(&A);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  free(Q.res);
  free(pres);
  free_points(&P);
  return 0;
}

/** @brief The query's lattice results of a scene, gathered to rank 0 in key order (every rank calls). */
static MFTL_SurfaceCandidate *gather_lattice(const qlat *Q, MPI_Comm comm)
{
  int r, np;
  MPI_Comm_rank(comm, &r);
  MPI_Comm_size(comm, &np);
  const int64_t n = (int64_t) Q->N * Q->N * (Q->hi[2] - Q->lo[2]);
  const int bytes = (int) (n * (int64_t) sizeof(MFTL_SurfaceCandidate));
  int *cnt = (int *) malloc((size_t) np * sizeof(int)), *dsp = (int *) malloc((size_t) np * sizeof(int));
  MPI_Gather(&bytes, 1, MPI_INT, cnt, 1, MPI_INT, 0, comm);
  int tot = 0;
  if (r == 0)
    for (int i = 0; i < np; ++i)
    {
      dsp[i] = tot;
      tot += cnt[i];
    }
  MFTL_SurfaceCandidate *all = (MFTL_SurfaceCandidate *) malloc((size_t) (tot > 0 ? tot : 1));
  MPI_Gatherv(Q->res, bytes, MPI_BYTE, all, cnt, dsp, MPI_BYTE, 0, comm);
  free(cnt);
  free(dsp);
  return all; /* z-slabs in rank order: key order */
}

/** @brief The cube [lo, hi]^3 as 12 triangles, wound outward, on rank 0 of @p ens; published and spread. */
static void add_cube(MFTL_Ensemble *ens, double lo, double hi)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  int r;
  MPI_Comm_rank(ens->domain.comm_world, &r);
  if (r == 0)
  {
    double pos[24];
    for (int i = 0; i < 8; ++i)
      for (int k = 0; k < 3; ++k) pos[3 * i + k] = ((i >> k) & 1) ? hi : lo;
    /* faces as quads (outward, counter-clockwise from outside), split on a diagonal */
    const int quad[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    int tri[36], t = 0;
    for (int f = 0; f < 6; ++f)
    {
      tri[t++] = quad[f][0];
      tri[t++] = quad[f][1];
      tri[t++] = quad[f][2];
      tri[t++] = quad[f][0];
      tri[t++] = quad[f][2];
      tri[t++] = quad[f][3];
    }
    MFTLBuildIndexedMesh(obj, pos, 8, tri, 12);
  }
  MFTLPublishGeneratedBodies(ens);
}

/** @brief A scene on @p comm: spheres (shape 0) or a cube with corners on lattice samples (shape 1); queried; rank 0 gets the results. */
static MFTL_SurfaceCandidate *scene_results(MPI_Comm comm, int shape, int N, double width, int64_t *n_out, int *status)
{
  MFTL_Ensemble *ens;
  if (shape == 0)
    ens = query_scene(comm, false);
  else
  {
    int np;
    MPI_Comm_size(comm, &np);
    int cuts[3];
    test_grid_for(np, cuts);
    double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    bool per[3] = {false, false, false};
    ens = MFTLEnsembleCreate(comm, cuts, lo, hi, per, 4);
    /* corners on cell centres: (k + 0.5) / N */
    add_cube(ens, 5.5 / N, 12.5 / N);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    MFTLRedistribute(ens);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(comm);
  qlat Q;
  qlat_add(g, comm, N, &Q);
  MFTL_Coupler *c = MFTL_NULL;
  *status = MFTLCouplerCreate(ens, g, MFTL_NULL, &c);
  if (*status == MFTL_COUPLING_OK)
  {
    MFTL_QueryField F = qlat_field(&Q);
    *status = run_query(c, 1, &Q.loc, width, 1, &F);
  }
  MFTL_SurfaceCandidate *all = gather_lattice(&Q, comm);
  *n_out = (int64_t) N * N * N;
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  free(Q.res);
  return all;
}

/**
 * The merged records do not depend on the process count, bit for bit: the
 * distributed query's results, gathered, against the same scene built and
 * queried on rank 0 alone. Two scenes: spheres (one across the seam), and a
 * cube whose corners and edges lie on lattice samples -- samples on its faces,
 * edges and corners (distance exactly zero), samples equidistant from two or
 * three faces, diagonals shared by two triangles: the ties the order exists for.
 * Inside the cube every sign is negative, outside positive.
 */
static int test_the_merged_records_are_partition_independent(void)
{
  for (int shape = 0; shape < 2; ++shape)
  {
    const int N = (shape == 0) ? 20 : 16; /* the cube's lattice: a power of two, so its corners ARE samples, to the bit */
    const double width = (shape == 0) ? 2.0 / N : 3.0 / N;
    int64_t n = 0;
    int st = 0, st_ref = MFTL_COUPLING_OK;
    MFTL_SurfaceCandidate *dist = scene_results(MPI_COMM_WORLD, shape, N, width, &n, &st);
    TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int ok = 1;
    long n_valid = 0, n_zero = 0, n_ties = 0, sign_bad = 0;
    if (g_rank == 0)
    {
      int64_t n1 = 0;
      MFTL_SurfaceCandidate *ref = scene_results(MPI_COMM_SELF, shape, N, width, &n1, &st_ref);
      if (st_ref != MFTL_COUPLING_OK || n1 != n) ok = 0;
      for (int64_t i = 0; i < n && ok; ++i)
      {
        if (!same_record(&dist[i], &ref[i])) ok = 0;
        if (!dist[i].valid) continue;
        ++n_valid;
        if (dist[i].distance == 0.0) ++n_zero;
        if (dist[i].feature != MFTL_TRI_FEATURE_FACE) ++n_ties;
        if (shape == 1)
        {
          const int64_t ix = i % N, iy = (i / N) % N, iz = i / ((int64_t) N * N);
          const int inside = ix > 5 && ix < 12 && iy > 5 && iy < 12 && iz > 5 && iz < 12;
          const int on = !inside && ix >= 5 && ix <= 12 && iy >= 5 && iy <= 12 && iz >= 5 && iz <= 12;
          if (inside && !(dist[i].distance < 0.0)) ++sign_bad;
          if (!inside && !on && !(dist[i].distance > 0.0)) ++sign_bad;
          if (on && dist[i].distance != 0.0) ++sign_bad;
        }
      }
      free(ref);
      printf("    %s: %lld samples, %ld within the width (%ld at distance 0, %ld on edges or corners) -- %s; signs %s\n", shape ? "cube" : "spheres",
             (long long) n, n_valid, n_zero, n_ties, ok ? "bit-identical to rank 0 alone" : "DIFFERENT", sign_bad ? "WRONG" : "right");
    }
    TEST_ASSERT(agreed(g_rank != 0 || (ok && sign_bad == 0 && n_valid > 100 && (shape == 0 || (n_zero > 50 && n_ties > 50))), MPI_COMM_WORLD));
    free(dist);
  }
  return 0;
}

/**
 * Several ensembles, one answer: two couplers -- one per sphere's ensemble --
 * querying into the same arrays give exactly what one ensemble holding both
 * spheres gives. The body ids differ between the two set-ups (each ensemble
 * numbers its own), so the comparison is of distance, sign and offset.
 */
static int test_two_ensembles_query_into_one_array(void)
{
  MFTL_Ensemble *both = query_scene(MPI_COMM_WORLD, false), *only_b = query_scene(MPI_COMM_WORLD, true);
  /* the first sphere alone: rebuild the scene with only body 0 */
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {true, false, true};
  MFTL_Ensemble *only_a = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
  {
    MFTL_Object *a = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(only_a, a);
    double ca[3] = {0.97, 0.45, 0.52};
    MFTLGenerateSphere(a, ca, 0.21, 2);
    for (int o = 0; o < only_a->n_local_objects; ++o) MFTLComputeTriangleGeometry(only_a->local_objects[o], only_a->local_objects[o]->tris.n);
    MFTLRedistribute(only_a);
    for (int o = 0; o < only_a->n_local_objects; ++o) MFTLComputeTriangleGeometry(only_a->local_objects[o], only_a->local_objects[o]->tris.n);
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  qlat Q1, Q2;
  qlat_add(g, MPI_COMM_WORLD, 20, &Q1);
  Q2 = Q1;
  const int64_t n = (int64_t) Q1.N * Q1.N * (Q1.hi[2] - Q1.lo[2]);
  Q2.res = (MFTL_SurfaceCandidate *) malloc((size_t) (n > 0 ? n : 1) * sizeof(MFTL_SurfaceCandidate));
  for (int64_t i = 0; i < n; ++i) MFTLSurfaceCandidateReset(&Q2.res[i]);
  Q2.ptr[0] = Q2.res;
  const double width = 2.0 / 20.0;
  MFTL_Coupler *cb = MFTL_NULL, *ca = MFTL_NULL, *cboth = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(both, g, MFTL_NULL, &cboth) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerCreate(only_a, g, MFTL_NULL, &ca) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerCreate(only_b, g, MFTL_NULL, &cb) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_QueryField F1 = qlat_field(&Q1), F2 = qlat_field(&Q2);
  TEST_ASSERT(agreed(run_query(cboth, 1, &Q1.loc, width, 1, &F1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(run_query(cb, 1, &Q2.loc, width, 1, &F2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(run_query(ca, 1, &Q2.loc, width, 1, &F2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  long bad = 0, valid = 0;
  for (int64_t i = 0; i < n; ++i)
  {
    const MFTL_SurfaceCandidate *a = &Q1.res[i], *b = &Q2.res[i];
    if (a->valid != b->valid) ++bad;
    if (!a->valid) continue;
    ++valid;
    if (memcmp(&a->distance, &b->distance, sizeof(double)) != 0 || memcmp(a->offset, b->offset, sizeof a->offset) != 0 || a->triangle != b->triangle ||
        a->feature != b->feature)
      ++bad;
  }
  long sums[2] = {bad, valid};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("    two ensembles into one array: %ld samples within the width, %ld differ from one ensemble holding both\n", sums[1], sums[0]);
  TEST_ASSERT(agreed(sums[0] == 0 && sums[1] > 500, MPI_COMM_WORLD));
  MFTLCouplerDestroy(&ca);
  MFTLCouplerDestroy(&cb);
  MFTLCouplerDestroy(&cboth);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&both);
  MFTLEnsembleDestroy(&only_a);
  MFTLEnsembleDestroy(&only_b);
  free(Q1.res);
  free(Q2.res);
  return 0;
}

/**
 * What a query refuses, on every rank: a width wider than the window certifies,
 * a surface moved out of it since, a window of named samples, a width that is
 * not a positive number, the same location twice, a location the window does
 * not hold.
 */
static int test_query_refusals(void)
{
  MFTL_Ensemble *ens = query_scene(MPI_COMM_WORLD, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  qlat Q;
  qlat_add(g, MPI_COMM_WORLD, 16, &Q);
  pts_block P;
  add_points(g, "cloud", 8, 0.3, OWN_SCATTERED, false, &P);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double width = 1.0 / 16.0;
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestQuery(c, 1, &Q.loc, width, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_QueryField F = qlat_field(&Q);
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, width, 1, &F) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  if (g_nproc > 1)
  {
    /* Two independently prepared windows are different collective handles,
     * even when their present geometry happens to give identical schedules. */
    MFTL_CouplingWindow *w2 = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestQuery(c, 1, &Q.loc, width, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c, req, &w2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    const int mixed = MFTLCouplerQueryNearest(c, g_rank == 0 ? w2 : w, width, 1, &F);
    MFTLCouplerReleaseWindow(&w2);
    TEST_ASSERT(agreed(mixed == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  }
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, 8.0 * width, 1, &F) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, 0.0, 1, &F) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, NAN, 1, &F) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTL_QueryField FF[2] = {F, F};
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, width, 2, FF) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTL_SurfaceCandidate *pr = (MFTL_SurfaceCandidate *) calloc((size_t) (P.n > 0 ? P.n : 1), sizeof(MFTL_SurfaceCandidate));
  MFTL_SurfaceCandidate *pp[1] = {pr};
  MFTL_QueryField Fp = {P.loc, pp, 1};
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, width, 1, &Fp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD)); /* not in the window */
  TEST_ASSERT(agreed(MFTLCouplerRequestQuery(c, 1, &Q.loc, -1.0, &req) == MFTL_COUPLING_INVALID && req == MFTL_NULL, MPI_COMM_WORLD));
  /* move the surface by a third of the box and announce it */
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v) obj->verts.pos[3 * v + 1] += 0.3;
    MFTLObjectMarkChanged(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, width, 1, &F) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  MFTLCouplerReleaseWindow(&w);
  /* a keyed window */
  uint64_t key = 0;
  int owner = 0; /* the z-slab holding iz = 0: rank 0 owns none at np = 27 */
  while ((int64_t) Q.N * (owner + 1) / g_nproc == 0) ++owner;
  MFTL_SampleList L = {Q.loc, g_rank == 0 ? 1 : 0, &key, &owner};
  st = MFTLCouplerRequestSamples(c, 1, &L, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQueryNearest(c, w, width, 1, &F) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerReleaseWindow(&w);
  free(pr);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  free(Q.res);
  free_points(&P);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling: nearest-surface queries (np = %d) ===\n", g_nproc);

  RUN_TEST(test_the_merge_is_a_total_order);
  RUN_TEST(test_the_query_equals_brute_force);
  RUN_TEST(test_the_merged_records_are_partition_independent);
  RUN_TEST(test_two_ensembles_query_into_one_array);
  RUN_TEST(test_query_refusals);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}

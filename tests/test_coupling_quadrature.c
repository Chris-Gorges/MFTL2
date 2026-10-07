/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Surface quadrature markers: the points, their weights, the level
 *        choice, and the vertex <-> point conversions as an exact adjoint pair.
 *
 * The references are the triangles themselves -- areas and corners read from
 * the pools, a quadratic integrated exactly by the edge-midpoint rule -- and
 * the same conversions recomputed on one rank alone, which shares the
 * arithmetic but none of the ghost schedule. A conversion that dropped a ghost
 * corner's share, or counted it twice, still reproduces constants on a closed
 * sphere only if both errors cancel exactly, which the power identity and the
 * one-rank comparison rule out.
 */

#include "test_coupling_irregular.h"

/** @brief A sphere scattered over the ranks, on a periodic unit box. */
static MFTL_Ensemble *quad_sphere(MPI_Comm comm, double r, int refine)
{
  int np;
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {true, true, true};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(comm, cuts, lo, hi, per, 4);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.47, 0.52, 0.5};
  MFTLGenerateSphere(obj, c, r, refine);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  return ens;
}

/** @brief A uniform periodic cell lattice of N^3 on the unit box. */
static adv_lat quad_lattice(int64_t N)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = N;
    A.h[k] = 1.0 / (double) N;
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = true;
  }
  return A;
}

/** @brief A coupler with a QUADRATURE window and binding over one lattice location. */
typedef struct
{
  MFTL_SolverGrid *g;
  int loc;
  adv_blocks B;
  MFTL_Coupler *c;
  MFTL_CouplingWindow *w;
  MFTL_CouplingBinding *b;
  adv_lat A;
} qsetup;

static int quad_setup(MFTL_Ensemble *ens, MPI_Comm comm, const adv_lat *A, const MFTL_MarkerSpec *ms, const MFTL_CouplerParams *P, qsetup *x)
{
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  memset(x, 0, sizeof *x);
  x->A = *A;
  x->g = MFTLSolverGridCreate(comm);
  x->loc = adv_add_location(x->g, "cells", A, MFTL_BOUNDARY_TRUNCATE);
  adv_decompose_on(rank, np, x->g, x->loc, A, 2 * np + 1, 2468u, 1, 1, false, MFTL_NULL, &x->B);
  int st = MFTLCouplerCreate(ens, x->g, P, &x->c);
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerRequestSupport(x->c, ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(x->c, req, &x->w);
  MFTLCouplerReleaseRequest(&req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(x->c, ms, x->w, &x->b);
  return st;
}

static void quad_teardown(qsetup *x)
{
  MFTLCouplerReleaseBinding(&x->b);
  MFTLCouplerReleaseWindow(&x->w);
  MFTLCouplerDestroy(&x->c);
  adv_free_blocks(&x->B);
  MFTLSolverGridDestroy(&x->g);
}

static MFTL_MarkerSpec quad_spec(int level)
{
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_QUADRATURE;
  ms.quadrature_level = level;
  return ms;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The points
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief At a fixed level m every owned triangle carries exactly m^2 points,
 *        numbered 0..m^2-1, of weight A/m^2 each, inside the triangle; at m = 1
 *        the point is the centroid. A quadratic integrated by the points
 *        converges at 1/m^2 to its exact integral over the facets (edge-midpoint
 *        rule), and a linear one is exact at every level.
 */
static int test_points_and_weights(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 2);
  adv_lat A = quad_lattice(12);
  double err_quad[4];
  const int levels[4] = {1, 2, 4, 8};
  int ok = 1;
  for (int li = 0; li < 4; ++li)
  {
    const int m = levels[li];
    MFTL_MarkerSpec ms = quad_spec(m);
    qsetup x;
    TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const int64_t n = MFTLCouplerBindingMarkerCount(x.b);
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
    const double *pos = MFTLCouplerBindingPositions(x.b);
    int64_t n_tri = 0;
    for (int o = 0; o < ens->n_local_objects; ++o) n_tri += ens->local_objects[o]->tris.n_local;
    if (n != n_tri * m * m) ok = 0;
    /* per triangle: the run of m^2 points, sub 0..m^2-1, weights summing to the area */
    double s_lin[2] = {0.0, 0.0}, s_quad[2] = {0.0, 0.0};
    for (int64_t i = 0; i < n && ok; i += (int64_t) m * m)
    {
      const MFTL_Object *obj = ens->local_objects[ref[i].slot];
      const int t = ref[i].entity;
      double area = 0.0, cr[3][3];
      MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, MFTL_NULL);
      MFTLGetTriangleCorners(obj, t, cr);
      double wsum = 0.0;
      for (int q = 0; q < m * m; ++q)
      {
        const MFTL_MarkerRef *r = &ref[i + q];
        if (r->entity != t || r->slot != ref[i].slot || r->sub != q || r->key != ref[i].key) ok = 0;
        if (fabs(r->weight - area / (double) (m * m)) > 1e-15 * area) ok = 0;
        wsum += r->weight;
        /* inside the (unwrapped) triangle: positive barycentric coordinates by areas */
        double dv[3], p[3];
        MFTLGetMinimumImageVector(&ens->domain, dv, cr[0], &pos[3 * (i + q)]);
        for (int k = 0; k < 3; ++k) p[k] = cr[0][k] + dv[k];
        double e1[3], e2[3], ep[3];
        for (int k = 0; k < 3; ++k)
        {
          e1[k] = cr[1][k] - cr[0][k];
          e2[k] = cr[2][k] - cr[0][k];
          ep[k] = p[k] - cr[0][k];
        }
        /* solve ep = a e1 + b e2 in the triangle's plane (least squares) */
        const double g11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], g12 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
        const double g22 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
        const double r1 = ep[0] * e1[0] + ep[1] * e1[1] + ep[2] * e1[2], r2 = ep[0] * e2[0] + ep[1] * e2[1] + ep[2] * e2[2];
        const double det = g11 * g22 - g12 * g12, a = (r1 * g22 - r2 * g12) / det, bb = (r2 * g11 - r1 * g12) / det;
        if (!(a > 0.0 && bb > 0.0 && a + bb < 1.0)) ok = 0;
        if (m == 1 && (fabs(a - 1.0 / 3.0) > 1e-12 || fabs(bb - 1.0 / 3.0) > 1e-12)) ok = 0;
        /* integrands in the frame of corner 0: linear x, quadratic x^2 */
        s_lin[0] += r->weight * p[0];
        s_quad[0] += r->weight * p[0] * p[0];
      }
      if (fabs(wsum - area) > 1e-14 * area) ok = 0;
      /* exact over the flat triangle: linear by the centroid, quadratic by the edge midpoints */
      s_lin[1] += area * (cr[0][0] + cr[1][0] + cr[2][0]) / 3.0;
      double mq = 0.0;
      for (int e = 0; e < 3; ++e)
      {
        const double mx = 0.5 * (cr[e][0] + cr[(e + 1) % 3][0]);
        mq += mx * mx;
      }
      s_quad[1] += area * mq / 3.0;
    }
    MPI_Allreduce(MPI_IN_PLACE, s_lin, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s_quad, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (fabs(s_lin[0] - s_lin[1]) > 1e-13 * fabs(s_lin[1])) ok = 0;
    err_quad[li] = fabs(s_quad[0] - s_quad[1]);
    quad_teardown(&x);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* second order: halving the sub-triangles quarters the error */
  for (int li = 1; li < 4; ++li)
  {
    const double ratio = err_quad[li - 1] / err_quad[li];
    if (g_rank == 0 && !(ratio > 3.9 && ratio < 4.1)) printf("\n  level %d -> %d: error ratio %.4f", levels[li - 1], levels[li], ratio);
    TEST_ASSERT(ratio > 3.9 && ratio < 4.1);
  }
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The conversions
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The dual areas of the owned vertices, from a VERTICES binding in the same window. */
static double *dual_areas(qsetup *x, int64_t *n)
{
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  /* a vertex window of its own: the quadrature window need not reach every vertex's support */
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  MFTLCouplerRequestSupport(x->c, &ms, &ws, &req);
  MFTLCouplerPrepareWindow(x->c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  MFTLCouplerBind(x->c, &ms, w, &b);
  *n = MFTLCouplerBindingMarkerCount(b);
  double *a = (double *) malloc((size_t) (*n > 0 ? *n : 1) * sizeof(double));
  const MFTL_MarkerRef *r = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < *n; ++i) a[i] = r[i].weight;
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  return a;
}

/**
 * @brief MFTLCouplerQuadratureToVertices() and ...FromVertices() are an exact
 *        adjoint pair over the whole distributed surface: the power identity
 *        holds to round-off with random fields, the total load is conserved,
 *        constants project exactly, the lumped mass IS the dual area, and the
 *        density and integrated forms agree.
 */
static int test_the_conversions_are_an_adjoint_pair(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.31, 3);
  adv_lat A = quad_lattice(14);
  for (int variant = 0; variant < 2; ++variant)
  {
    MFTL_MarkerSpec ms = quad_spec(variant == 0 ? 3 : 0); /* fixed and adaptive */
    qsetup x;
    TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const int64_t nq = MFTLCouplerBindingMarkerCount(x.b), nv = MFTLCouplerQuadratureVertexCount(x.b);
    int64_t nv2 = 0;
    double *dual = dual_areas(&x, &nv2);
    TEST_ASSERT(agreed(nv == nv2, MPI_COMM_WORLD));
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
    const int nc = 3;
    double *Uq = (double *) malloc((size_t) (nq > 0 ? nq * nc : 1) * sizeof(double)), *Fq = (double *) malloc((size_t) (nq > 0 ? nq * nc : 1) * sizeof(double));
    double *Uv = (double *) malloc((size_t) (nv > 0 ? nv * nc : 1) * sizeof(double)), *Fv = (double *) malloc((size_t) (nv > 0 ? nv * nc : 1) * sizeof(double));
    double *Fq2 = (double *) malloc((size_t) (nq > 0 ? nq * nc : 1) * sizeof(double));
    for (int64_t i = 0; i < nq; ++i)
      for (int k = 0; k < nc; ++k) Uq[i * nc + k] = hash01(ref[i].key * 131u + (uint64_t) ref[i].sub * 7u + (uint64_t) k);
    /* vertex loads keyed by the vertex's global id, so they do not depend on the partition */
    {
      int64_t v = 0;
      for (int o = 0; o < ens->n_local_objects; ++o)
        for (int a = 0; a < ens->local_objects[o]->verts.n_local; ++a, ++v)
          for (int k = 0; k < nc; ++k) Fv[v * nc + k] = hash01((uint64_t) ens->local_objects[o]->verts.rec[a].global_id * 17u + (uint64_t) k + 5u);
    }
    TEST_ASSERT(agreed(MFTLCouplerQuadratureToVertices(x.c, x.b, nc, Uq, Uv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, nc, Fv, MFTL_SPREAD_INTEGRATED, Fq) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double s[6] = {0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < nq * nc; ++i)
    {
      s[0] += Uq[i] * Fq[i];
      s[1] += fabs(Uq[i] * Fq[i]);
      s[4] += Fq[i];
    }
    for (int64_t i = 0; i < nv * nc; ++i)
    {
      s[2] += Uv[i] * Fv[i];
      s[3] += fabs(Uv[i] * Fv[i]);
      s[5] += Fv[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, s, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0 && !(fabs(s[0] - s[2]) <= 1e-13 * fmax(s[1], s[3]))) printf("\n  power: %.17g vs %.17g", s[0], s[2]);
    TEST_ASSERT(fabs(s[0] - s[2]) <= 1e-13 * fmax(s[1], s[3]));
    TEST_ASSERT(fabs(s[4] - s[5]) <= 1e-13 * fmax(1.0, fabs(s[5])));

    /* constants project exactly; the lumped mass is the dual area */
    for (int64_t i = 0; i < nq * nc; ++i) Uq[i] = 2.5;
    TEST_ASSERT(agreed(MFTLCouplerQuadratureToVertices(x.c, x.b, nc, Uq, Uv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int cst = 1;
    for (int64_t i = 0; i < nv * nc; ++i)
      if (fabs(Uv[i] - 2.5) > 4e-16 * 2.5 * 4.0) cst = 0;
    TEST_ASSERT(agreed(cst, MPI_COMM_WORLD));
    /* integrated loads equal to the dual area are a density of one: every point load is its weight */
    for (int64_t v = 0; v < nv; ++v)
      for (int k = 0; k < nc; ++k) Fv[v * nc + k] = dual[v];
    TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, nc, Fv, MFTL_SPREAD_INTEGRATED, Fq) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int mass_ok = 1;
    for (int64_t i = 0; i < nq; ++i)
      for (int k = 0; k < nc; ++k)
        if (fabs(Fq[i * nc + k] - ref[i].weight) > 1e-13 * ref[i].weight) mass_ok = 0;
    TEST_ASSERT(agreed(mass_ok, MPI_COMM_WORLD));
    /* the density form: g directly */
    for (int64_t v = 0; v < nv * nc; ++v) Fv[v] = 1.0;
    TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, nc, Fv, MFTL_SPREAD_DENSITY, Fq2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int dens_ok = 1;
    for (int64_t i = 0; i < nq * nc; ++i)
      if (fabs(Fq2[i] - Fq[i]) > 1e-13 * fabs(Fq[i])) dens_ok = 0;
    TEST_ASSERT(agreed(dens_ok, MPI_COMM_WORLD));
    free(Uq);
    free(Fq);
    free(Fq2);
    free(Uv);
    free(Fv);
    free(dual);
    quad_teardown(&x);
  }
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Projects a deterministic point field on @p comm; returns (vertex gid, value) pairs. */
static int project_on(MPI_Comm comm, int64_t *n, int **vid, double **val)
{
  MFTL_Ensemble *ens = quad_sphere(comm, 0.29, 3);
  adv_lat A = quad_lattice(13);
  MFTL_MarkerSpec ms = quad_spec(0);
  qsetup x;
  int st = quad_setup(ens, comm, &A, &ms, MFTL_NULL, &x);
  const int64_t nq = MFTLCouplerBindingMarkerCount(x.b);
  *n = MFTLCouplerQuadratureVertexCount(x.b);
  *vid = (int *) malloc((size_t) (*n > 0 ? *n : 1) * sizeof(int));
  *val = (double *) malloc((size_t) (*n > 0 ? *n : 1) * sizeof(double));
  if (st == MFTL_COUPLING_OK)
  {
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
    double *Uq = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double));
    for (int64_t i = 0; i < nq; ++i) Uq[i] = hash01(ref[i].key * 977u + (uint64_t) ref[i].sub);
    st = MFTLCouplerQuadratureToVertices(x.c, x.b, 1, Uq, *val);
    int64_t v = 0;
    for (int o = 0; o < ens->n_local_objects; ++o)
      for (int a = 0; a < ens->local_objects[o]->verts.n_local; ++a, ++v) (*vid)[v] = ens->local_objects[o]->verts.rec[a].global_id;
    free(Uq);
  }
  quad_teardown(&x);
  MFTLEnsembleDestroy(&ens);
  return st;
}

/**
 * @brief The projected vertex values do not depend on the partition beyond
 *        round-off: the same surface on one rank alone, with no ghost schedule
 *        at all, gives every vertex the same value. A ghost corner's share lost
 *        or doubled shows up at exactly the vertices on rank boundaries.
 */
static int test_the_projection_does_not_depend_on_the_partition(void)
{
  int64_t n = 0;
  int *vid = MFTL_NULL;
  double *val = MFTL_NULL;
  TEST_ASSERT(agreed(project_on(MPI_COMM_WORLD, &n, &vid, &val) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *mine = (double *) malloc((size_t) (n > 0 ? 2 * n : 1) * sizeof(double));
  for (int64_t i = 0; i < n; ++i)
  {
    mine[2 * i] = (double) vid[i];
    mine[2 * i + 1] = val[i];
  }
  int tot = 0;
  double *all = allgather_doubles(mine, (int) n, 2, &tot);
  int ok = 1;
  if (g_rank == 0)
  {
    int64_t ns = 0;
    int *vs = MFTL_NULL;
    double *xs = MFTL_NULL;
    ok = project_on(MPI_COMM_SELF, &ns, &vs, &xs) == MFTL_COUPLING_OK && ns == tot;
    double worst = 0.0;
    for (int i = 0; i < tot && ok; ++i)
    {
      int64_t j = -1;
      for (int64_t q = 0; q < ns; ++q)
        if (vs[q] == (int) all[2 * i]) j = q;
      if (j < 0)
        ok = 0;
      else
        worst = fmax(worst, fabs(all[2 * i + 1] - xs[j]));
    }
    if (worst > 1e-14) ok = 0;
    if (!ok) printf("\n  projection vs one rank: worst %.3e (%lld vs %d vertices)", worst, (long long) ns, tot);
    free(vs);
    free(xs);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  free(all);
  free(mine);
  free(vid);
  free(val);
  return 0;
}

/** @brief A smooth test function of an unwrapped position. */
static double smooth_f(const double p[3]) { return sin(3.1 * p[0]) + 0.7 * cos(2.3 * p[1] - 0.4) + p[2] * p[2]; }

/**
 * @brief The projection is what its definition says, vertex by vertex: the
 *        test recomputes \f$\sum_q w_q \lambda_{qv} f(x_q) / \sum_q w_q
 *        \lambda_{qv}\f$ from the GEOMETRY -- barycentric coordinates solved from
 *        each point's position in its triangle, corners named by their global
 *        ids, every rank's contributions gathered -- and compares. A swapped
 *        coordinate pair, a corner mapped to the wrong vertex or a ghost share
 *        lost would all keep constants, masses and the adjoint identity intact;
 *        this is what sees them.
 */
static int test_the_projection_matches_its_definition(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 2);
  adv_lat A = quad_lattice(18);
  MFTL_MarkerSpec ms = quad_spec(0);
  qsetup x;
  TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nq = MFTLCouplerBindingMarkerCount(x.b), nv = MFTLCouplerQuadratureVertexCount(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  const double *pos = MFTLCouplerBindingPositions(x.b);
  double *Uq = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double)), *Uv = (double *) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(double));
  /* contributions (vertex gid, w lambda f, w lambda), three per point */
  double *con = (double *) malloc((size_t) (nq > 0 ? 9 * nq : 1) * sizeof(double));
  for (int64_t i = 0; i < nq; ++i)
  {
    const MFTL_Object *obj = ens->local_objects[ref[i].slot];
    double cr[3][3], dv[3], p[3], e1[3], e2[3], ep[3];
    int vv[3];
    MFTLGetTriangleCorners(obj, ref[i].entity, cr);
    MFTLGetTriangleVertices(obj, ref[i].entity, vv);
    MFTLGetMinimumImageVector(&ens->domain, dv, cr[0], &pos[3 * i]);
    for (int k = 0; k < 3; ++k)
    {
      p[k] = cr[0][k] + dv[k];
      e1[k] = cr[1][k] - cr[0][k];
      e2[k] = cr[2][k] - cr[0][k];
      ep[k] = p[k] - cr[0][k];
    }
    const double g11 = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], g12 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
    const double g22 = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
    const double r1 = ep[0] * e1[0] + ep[1] * e1[1] + ep[2] * e1[2], r2 = ep[0] * e2[0] + ep[1] * e2[1] + ep[2] * e2[2];
    const double det = g11 * g22 - g12 * g12, l1 = (r1 * g22 - r2 * g12) / det, l2 = (r2 * g11 - r1 * g12) / det;
    const double lam[3] = {1.0 - l1 - l2, l1, l2};
    /* the field at the point, in a frame every rank agrees on: the wrapped position */
    Uq[i] = smooth_f(&pos[3 * i]);
    for (int c = 0; c < 3; ++c)
    {
      con[9 * i + 3 * c] = (double) obj->verts.rec[vv[c]].global_id;
      con[9 * i + 3 * c + 1] = ref[i].weight * lam[c] * Uq[i];
      con[9 * i + 3 * c + 2] = ref[i].weight * lam[c];
    }
  }
  TEST_ASSERT(agreed(MFTLCouplerQuadratureToVertices(x.c, x.b, 1, Uq, Uv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int tot = 0;
  double *all = allgather_doubles(con, (int) (3 * nq), 3, &tot);
  int max_vid = 0;
  for (int i = 0; i < tot; ++i) max_vid = (int) fmax(max_vid, all[3 * i]);
  double *num = (double *) calloc((size_t) max_vid + 1, sizeof(double)), *den = (double *) calloc((size_t) max_vid + 1, sizeof(double));
  for (int i = 0; i < tot; ++i)
  {
    num[(int) all[3 * i]] += all[3 * i + 1];
    den[(int) all[3 * i]] += all[3 * i + 2];
  }
  double worst = 0.0;
  int64_t v = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
    for (int a = 0; a < ens->local_objects[o]->verts.n_local; ++a, ++v)
    {
      const int gid = ens->local_objects[o]->verts.rec[a].global_id;
      worst = fmax(worst, fabs(Uv[v] - num[gid] / den[gid]));
    }
  if (worst > 1e-12) printf("\n  rank %d: projection off its definition by %.3e", g_rank, worst);
  TEST_ASSERT(agreed(worst <= 1e-12, MPI_COMM_WORLD));
  free(num);
  free(den);
  free(all);
  free(con);
  free(Uq);
  free(Uv);
  quad_teardown(&x);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Through the whole coupler: interpolate an affine fluid field at the
 *        points (exact, LINEAR rows), project to the vertices, convert vertex
 *        forces to point loads and spread them. The fluid receives the vertex
 *        force exactly, and the power the vertices see equals the power the
 *        fluid sees: \f$u \cdot W^T F_q = U_v \cdot F_v\f$.
 */
static int test_quadrature_through_the_coupler(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 2);
  adv_lat A = quad_lattice(20);
  MFTL_MarkerSpec ms = quad_spec(0);
  qsetup x;
  TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nq = MFTLCouplerBindingMarkerCount(x.b), nv = MFTLCouplerQuadratureVertexCount(x.b);
  rnd_ctx rc = {{20, 20, 20}, 42u};
  adv_fill(&x.B, &A, 0, f_random, &rc, false);
  MFTL_SolverField F = adv_field(&x.B, x.loc, 0, 1, false);
  double *Uq = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double)), *Fq = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double));
  double *Uv = (double *) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(double)), *Fv = (double *) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, Uq) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQuadratureToVertices(x.c, x.b, 1, Uq, Uv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  {
    int64_t v = 0;
    for (int o = 0; o < ens->n_local_objects; ++o)
      for (int a = 0; a < ens->local_objects[o]->verts.n_local; ++a, ++v) Fv[v] = 0.5 + hash01((uint64_t) ens->local_objects[o]->verts.rec[a].global_id);
  }
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 1, Fv, MFTL_SPREAD_INTEGRATED, Fq) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* the force: spread into a zeroed copy of the field */
  adv_blocks T = x.B;
  double *pt[ADV_MAX_BOXES];
  for (int i = 0; i < x.B.n; ++i)
  {
    T.a[i] = (double *) calloc((size_t) x.B.n_elem[i], sizeof(double));
    pt[i] = T.a[i];
  }
  MFTL_SolverField Ft = F;
  Ft.data = x.B.n ? pt : MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &Ft, Fq, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double s[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < x.B.n; ++i)
    for (int64_t e = 0; e < x.B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&x.B, i, e, idx);
      bool own = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < x.B.lo[i][k] || idx[k] >= x.B.hi[i][k]) own = false;
      if (!own) continue;
      s[0] += T.a[i][e];
      s[1] += x.B.a[i][e] * T.a[i][e];
      s[2] += fabs(x.B.a[i][e] * T.a[i][e]);
    }
  for (int64_t v = 0; v < nv; ++v)
  {
    s[3] += Fv[v];
    s[4] += Uv[v] * Fv[v];
  }
  MPI_Allreduce(MPI_IN_PLACE, s, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(fabs(s[0] - s[3]) <= 1e-13 * fabs(s[3]));
  TEST_ASSERT(fabs(s[1] - s[4]) <= 1e-13 * s[2]);
  for (int i = 0; i < x.B.n; ++i) free(T.a[i]);
  free(Uq);
  free(Fq);
  free(Uv);
  free(Fv);
  quad_teardown(&x);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The levels
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Adaptive levels follow the fluid: on a coarse surface (refine 1) in a
 *        fine lattice every triangle's sub-triangles are within the ratio of the
 *        spacing, and no level is larger than needed. A cap below the need is
 *        BAD_SAMPLING under an ERROR policy, accepted under REPORT and when the
 *        binding is for interpolation only.
 */
static int test_adaptive_levels_follow_the_fluid(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 1);
  adv_lat A = quad_lattice(24);
  const double h = A.h[0];
  for (int r = 0; r < 2; ++r)
  {
    MFTL_MarkerSpec ms = quad_spec(0);
    ms.quadrature_ratio = (r == 0) ? 1.0 : 0.5;
    qsetup x;
    TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const int64_t n = MFTLCouplerBindingMarkerCount(x.b);
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
    int ok = 1, max_m = 0;
    for (int64_t i = 0; i < n;)
    {
      int64_t j = i;
      while (j < n && ref[j].key == ref[i].key && ref[j].slot == ref[i].slot) ++j;
      const int m = (int) llround(sqrt((double) (j - i)));
      if ((int64_t) m * m != j - i) ok = 0;
      double cr[3][3], lmax = 0.0;
      MFTLGetTriangleCorners(ens->local_objects[ref[i].slot], ref[i].entity, cr);
      for (int e = 0; e < 3; ++e)
      {
        double l2 = 0.0;
        for (int k = 0; k < 3; ++k) l2 += (cr[(e + 1) % 3][k] - cr[e][k]) * (cr[(e + 1) % 3][k] - cr[e][k]);
        lmax = fmax(lmax, sqrt(l2));
      }
      /* resolved, and minimal */
      if (lmax / m > ms.quadrature_ratio * h * (1.0 + 1e-12)) ok = 0;
      if (m > 1 && lmax / (m - 1) <= ms.quadrature_ratio * h) ok = 0;
      max_m = (m > max_m) ? m : max_m;
      i = j;
    }
    TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(max_m >= 3, MPI_COMM_WORLD)); /* non-vacuity: the surface really is coarser than the fluid */
    quad_teardown(&x);
  }
  /* a cap below the need */
  for (int pol = 0; pol < 3; ++pol)
  {
    MFTL_CouplerParams P;
    MFTLCouplerDefaultParams(&P);
    P.sampling = (pol == 1) ? MFTL_SAMPLING_REPORT : MFTL_SAMPLING_ERROR;
    MFTL_MarkerSpec ms = quad_spec(0);
    ms.quadrature_max_level = 2;
    ms.interpolation_only = (pol == 2);
    qsetup x;
    const int st = quad_setup(ens, MPI_COMM_WORLD, &A, &ms, &P, &x);
    TEST_ASSERT(agreed(pol == 0 ? st == MFTL_COUPLING_BAD_SAMPLING && x.b == MFTL_NULL : st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    quad_teardown(&x);
  }
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Lifetimes and misuse
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A stage binding of a QUADRATURE binding converts with its own (stage)
 *        weights through the base's schedule; the base outlives it; a moved
 *        surface stales both conversions on every rank; a VERTICES binding,
 *        zero components, missing arrays and ranks that disagree on the
 *        component count are refused before any message is sent.
 */
static int test_quadrature_lifetimes_and_misuse(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 2);
  adv_lat A = quad_lattice(16);
  MFTL_MarkerSpec ms = quad_spec(2);
  qsetup x;
  TEST_ASSERT(agreed(quad_setup(ens, MPI_COMM_WORLD, &A, &ms, MFTL_NULL, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nq = MFTLCouplerBindingMarkerCount(x.b), nv = MFTLCouplerQuadratureVertexCount(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  /* stage: same positions, doubled weights -- projections of constants unchanged, point loads doubled */
  double *wt = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double));
  for (int64_t i = 0; i < nq; ++i) wt[i] = 2.0 * ref[i].weight;
  MFTL_StageGeometry sg = {MFTLCouplerBindingPositions(x.b), wt};
  MFTL_CouplingBinding *bs = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(x.c, x.b, &sg, &bs) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *Fv = (double *) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(double));
  double *F1 = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double)), *F2 = (double *) malloc((size_t) (nq > 0 ? nq : 1) * sizeof(double));
  for (int64_t v = 0; v < nv; ++v) Fv[v] = 1.0 + 0.1 * (double) (v % 7);
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 1, Fv, MFTL_SPREAD_DENSITY, F1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, bs, 1, Fv, MFTL_SPREAD_DENSITY, F2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int dbl = 1;
  for (int64_t i = 0; i < nq; ++i)
    if (F2[i] != 2.0 * F1[i]) dbl = 0;
  TEST_ASSERT(agreed(dbl, MPI_COMM_WORLD));
  /* integrated loads: the mass doubles too, so the point loads are the same */
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 1, Fv, MFTL_SPREAD_INTEGRATED, F1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, bs, 1, Fv, MFTL_SPREAD_INTEGRATED, F2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst = 0.0;
  for (int64_t i = 0; i < nq; ++i) worst = fmax(worst, fabs(F2[i] - F1[i]) / fabs(F1[i]));
  TEST_ASSERT(agreed(worst <= 1e-15, MPI_COMM_WORLD));
  /* the base cannot go while its stage binding lives */
  TEST_ASSERT(agreed(MFTLCouplerReleaseBinding(&x.b) == MFTL_ERROR && x.b != MFTL_NULL, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bs);

  /* misuse */
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 0, Fv, MFTL_SPREAD_INTEGRATED, F1) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 1, Fv, (MFTL_SpreadForm) 7, F1) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  {
    /* a missing input array on the ranks that have points: refused on every rank */
    const int some = anywhere(nq > 0, MPI_COMM_WORLD);
    const int st = MFTLCouplerQuadratureToVertices(x.c, x.b, 1, MFTL_NULL, Fv);
    TEST_ASSERT(agreed(st == (some ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK), MPI_COMM_WORLD));
  }
  if (g_nproc > 1)
  {
    /* ranks disagreeing on the component count: refused everywhere, nothing hangs */
    const int nc = (g_rank == 0) ? 2 : 1;
    double *bq = (double *) malloc((size_t) (nq > 0 ? 2 * nq : 1) * sizeof(double)), *bv = (double *) malloc((size_t) (nv > 0 ? 2 * nv : 1) * sizeof(double));
    for (int64_t i = 0; i < 2 * nq; ++i) bq[i] = 1.0;
    TEST_ASSERT(agreed(MFTLCouplerQuadratureToVertices(x.c, x.b, nc, bq, bv) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    free(bq);
    free(bv);
  }
  {
    MFTL_MarkerSpec vs;
    memset(&vs, 0, sizeof vs);
    vs.kind = MFTL_MARKERS_VERTICES;
    MFTL_CouplingBinding *bv = MFTL_NULL;
    MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    MFTL_CouplingWindow *wv = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(x.c, &vs, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(x.c, req, &wv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    TEST_ASSERT(agreed(MFTLCouplerBind(x.c, &vs, wv, &bv) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerQuadratureVertexCount(bv) == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, bv, 1, Fv, MFTL_SPREAD_INTEGRATED, F1) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLCouplerReleaseBinding(&bv);
    MFTLCouplerReleaseWindow(&wv);
    /* ranks binding different kinds are refused, not diverging into different exchanges */
    MFTL_MarkerSpec mixed = (g_rank == 0) ? ms : vs;
    MFTL_CouplingBinding *bm = MFTL_NULL;
    if (g_nproc > 1) TEST_ASSERT(agreed(MFTLCouplerBind(x.c, &mixed, x.w, &bm) == MFTL_COUPLING_INVALID && bm == MFTL_NULL, MPI_COMM_WORLD));
  }
  /* the surface moved: stale, everywhere */
  MFTLEnsembleMarkChanged(ens);
  TEST_ASSERT(agreed(MFTLCouplerQuadratureFromVertices(x.c, x.b, 1, Fv, MFTL_SPREAD_INTEGRATED, F1) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  free(wt);
  free(Fv);
  free(F1);
  free(F2);
  quad_teardown(&x);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Sizing: what the fluid asks of the surface
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief MFTLCouplerSurfaceSizing() gives every owned vertex (3/4) theta ratio
 *        h on a uniform lattice, and remeshing a surface far coarser than the
 *        fluid to the job-wide minimum brings MFTLCouplerSurfaceSpacing() within
 *        theta of the sampling bound -- the loop the plan describes, closed.
 */
static int test_sizing_lets_a_remesh_meet_the_sampling_bound(void)
{
  MFTL_Ensemble *ens = quad_sphere(MPI_COMM_WORLD, 0.3, 1);
  adv_lat A = quad_lattice(24);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc + 1, 11u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_SpacingResult sp;
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSpacing(c, &sp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(sp.max_ratio > 2.0); /* non-vacuity: far too coarse to begin with */
  const double theta = 0.9;
  int64_t nv = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) nv += ens->local_objects[o]->verts.n_local;
  double *sizes = (double *) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(double));
  double gmin = 0.0;
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSizing(c, theta, sizes, &gmin) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double expect = 0.75 * theta * A.h[0];
  int ok = fabs(gmin - expect) <= 1e-15 * expect;
  for (int64_t v = 0; v < nv; ++v)
    if (fabs(sizes[v] - expect) > 1e-15 * expect) ok = 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* misuse */
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSizing(c, 0.0, sizes, &gmin) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSizing(c, theta, sizes, MFTL_NULL) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  /* remesh to it, then measure again */
  MFTL_RemeshParameters rp;
  /* a sevenfold refinement in one shot: 8 iterations leave the serial remesher at 1.04 h
   * (theta 0.9), 16 converge to 0.87 h -- a remesh that has not converged is not sized */
  MFTLInitRemeshParams(&rp, MFTL_REMESH_ISOTROPIC, gmin, 16, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  rp.parallel_phase_budget = 16;
  MFTL_RemeshStats rs;
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &rp, &rs) == MFTL_SUCCESS, MPI_COMM_WORLD));
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSpacing(c, &sp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  if (g_rank == 0 && !(sp.max_ratio <= theta * 1.05)) printf("\n  after the remesh: max ratio %.4f (theta %.2f)", sp.max_ratio, theta);
  TEST_ASSERT(sp.max_ratio <= theta * 1.05 && sp.n_offending == 0);
  free(sizes);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling quadrature (np = %d) ===\n", g_nproc);

  RUN_TEST(test_points_and_weights);
  RUN_TEST(test_the_conversions_are_an_adjoint_pair);
  RUN_TEST(test_the_projection_does_not_depend_on_the_partition);
  RUN_TEST(test_the_projection_matches_its_definition);
  RUN_TEST(test_quadrature_through_the_coupler);
  RUN_TEST(test_adaptive_levels_follow_the_fluid);
  RUN_TEST(test_quadrature_lifetimes_and_misuse);
  RUN_TEST(test_sizing_lets_a_remesh_meet_the_sampling_bound);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}

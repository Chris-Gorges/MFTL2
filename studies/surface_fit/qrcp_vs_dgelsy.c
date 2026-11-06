/**
 * @file qrcp_vs_dgelsy.c
 * @brief MFTLLeastSquaresQRCP() against LAPACK's dgelsy on the least-squares
 *        problems the paraboloid placement solves: speed, agreement, and the
 *        accuracy of the placed point against the exact sphere.
 *
 * Not part of the build (the library does not link LAPACK). Build and run with
 * run_benchmarks.sh, which links Accelerate on macOS and -llapack elsewhere.
 *
 * Problems: every edge of an icosphere of radius R (refinement 3, 4, 5), with
 * the stencil, frame, weights and filter of src/remeshing/remesh_fit.c
 * reproduced here, in four forms:
 *
 *   scaled, weighted, two-ring      what the library solves (positions; + normals)
 *   unscaled, unweighted, mftl's 8  what the old mftl solved, at R = 1, 1e-2, 1e-4, 1e-6
 *
 * plus two hand-built rank-deficient stencils.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "mftl_core.h"

extern void dgelsy_(const int *m, const int *n, const int *nrhs, double *a, const int *lda, double *b, const int *ldb, int *jpvt, const double *rcond,
                    int *rank, double *work, const int *lwork, int *info);

#define MAX_ROWS 192
#define MAX_STENCIL 64

typedef struct
{
  int m;  // rows
  double A[6 * MAX_ROWS];  // column-major, columns 1 x y x^2 xy y^2
  double b[MAX_ROWS];
  double h;  // length unit of the coordinates (1 for unscaled)
  double z_true;  // exact height of the sphere above the midpoint along the frame normal, in coordinate units
} problem;

static double now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static void max_normal(const MFTL_Object *obj, int v, double n[3])
{
  n[0] = n[1] = n[2] = 0.0;
  const int start = obj->verts.rec[v].out_he_idx;
  const double *p = MFTLVertexPos(obj, v);
  int h = start;
  for (int g = 0; g < 64; ++g)
  {
    double d1[3], d2[3], c[3];
    for (int k = 0; k < 3; ++k)
    {
      d1[k] = MFTLVertexPos(obj, MFTLHeTarget(obj, h))[k] - p[k];
      d2[k] = MFTLVertexPos(obj, MFTLHeTarget(obj, MFTLHeNextIdx(h)))[k] - p[k];
    }
    MFTLVectorCrossVector(c, d1, d2);
    const double w = MFTLVectorNormSq(d1) * MFTLVectorNormSq(d2);
    for (int k = 0; k < 3; ++k) n[k] += c[k] / w;
    h = obj->tris.he[MFTLHePrevIdx(h)].twin_idx;
    if (h == start) break;
  }
  MFTLNormalizeVector(n);
}

static int add_unique(int *list, int n, int v)
{
  for (int i = 0; i < n; ++i)
    if (list[i] == v) return n;
  if (n < MAX_STENCIL) list[n++] = v;
  return n;
}

static double wendland(double q) { return (q < 1.0) ? pow(1.0 - q, 4) * (1.0 + 4.0 * q) : 0.0; }

/* The frame of remesh_fit.c, and the exact sphere height along its normal. */
static void frame(const MFTL_Object *obj, int h, double R, double m[3], double ex[3], double ey[3], double nz[3], double *z_true, double *hl)
{
  const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
  const double *pa = MFTLVertexPos(obj, a), *pb = MFTLVertexPos(obj, b);
  double na[3], nb[3], ab[3];
  max_normal(obj, a, na);
  max_normal(obj, b, nb);
  for (int k = 0; k < 3; ++k)
  {
    nz[k] = na[k] + nb[k];
    m[k] = 0.5 * (pa[k] + pb[k]);
    ab[k] = pb[k] - pa[k];
  }
  MFTLNormalizeVector(nz);
  const double al = MFTLVectorInProd(ab, nz);
  for (int k = 0; k < 3; ++k) ex[k] = ab[k] - al * nz[k];
  MFTLNormalizeVector(ex);
  MFTLVectorCrossVector(ey, nz, ex);
  /* |m + t nz| = R, the positive root. */
  const double mn = MFTLVectorInProd(m, nz), mm = MFTLVectorInProd(m, m);
  *z_true = -mn + sqrt(mn * mn - mm + R * R);

  double sum = 0.0;
  int cnt = 0;
  for (int e = 0; e < 2; ++e)
  {
    const int v = e ? b : a;
    int ring[64];
    const int nr = MFTLGetVertexOneRingVertices(obj, v, ring, 64, MFTL_FALSE);
    for (int i = 0; i < nr; ++i)
    {
      double d[3];
      for (int k = 0; k < 3; ++k) d[k] = MFTLVertexPos(obj, ring[i])[k] - MFTLVertexPos(obj, v)[k];
      sum += MFTLVectorNorm(d);
      cnt++;
    }
  }
  *hl = sum / cnt;
}

/* The library's problem: two-ring, scaled by the local edge length, Wendland weights, normal filter. */
static int build_library(const MFTL_Object *obj, int h, double R, bool normals, double rho, problem *P)
{
  double m[3], ex[3], ey[3], nz[3], zt, hl;
  frame(obj, h, R, m, ex, ey, nz, &zt, &hl);
  const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
  int st[MAX_STENCIL], ns = 0;
  ns = add_unique(st, ns, a);
  ns = add_unique(st, ns, b);
  for (int e = 0; e < 2; ++e)
  {
    int ring[64];
    const int nr = MFTLGetVertexOneRingVertices(obj, e ? b : a, ring, 64, MFTL_FALSE);
    for (int i = 0; i < nr; ++i) ns = add_unique(st, ns, ring[i]);
  }
  const int n1 = ns;
  for (int i = 2; i < n1; ++i)
  {
    int ring[64];
    const int nr = MFTLGetVertexOneRingVertices(obj, st[i], ring, 64, MFTL_FALSE);
    for (int j = 0; j < nr; ++j) ns = add_unique(st, ns, ring[j]);
  }
  double pts[3 * MAX_STENCIL], sl[2 * MAX_STENCIL], w[MAX_STENCIL];
  int n = 0;
  for (int i = 0; i < ns; ++i)
  {
    double d[3], nv[3];
    for (int k = 0; k < 3; ++k) d[k] = (MFTLVertexPos(obj, st[i])[k] - m[k]) / hl;
    const double wi = wendland(MFTLVectorNorm(d) / rho);
    if (wi <= 0.0) continue;
    max_normal(obj, st[i], nv);
    const double nzl = MFTLVectorInProd(nv, nz);
    if (nzl < 0.5) continue;
    pts[3 * n] = MFTLVectorInProd(d, ex);
    pts[3 * n + 1] = MFTLVectorInProd(d, ey);
    pts[3 * n + 2] = MFTLVectorInProd(d, nz);
    sl[2 * n] = -MFTLVectorInProd(nv, ex) / nzl;
    sl[2 * n + 1] = -MFTLVectorInProd(nv, ey) / nzl;
    w[n++] = wi;
  }
  /* Rows as MFTLFitHeightParaboloid() assembles them. */
  int r = 0;
  const int rows = n * (normals ? 3 : 1);
  for (int j = 0; j < n; ++j)
  {
    const double s = sqrt(w[j]), x = pts[3 * j], y = pts[3 * j + 1];
    const double row[3][7] = {{s, s * x, s * y, s * x * x, s * x * y, s * y * y, s * pts[3 * j + 2]},
                              {0, s, 0, 2 * s * x, s * y, 0, s * sl[2 * j]},
                              {0, 0, s, 0, s * x, 2 * s * y, s * sl[2 * j + 1]}};
    for (int q = 0; q < (normals ? 3 : 1); ++q)
    {
      for (int c = 0; c < 6; ++c) P->A[c * rows + r] = row[q][c];
      P->b[r] = row[q][6];
      ++r;
    }
  }
  P->m = rows;
  P->h = hl;
  P->z_true = zt / hl;
  return n;
}

/* mftl's problem: a, b, the two apexes, the far vertices across the four wing
 * edges; unweighted; coordinates in the physical unit (not scaled). */
static int build_mftl(const MFTL_Object *obj, int h, double R, bool scaled, problem *P)
{
  double m[3], ex[3], ey[3], nz[3], zt, hl;
  frame(obj, h, R, m, ex, ey, nz, &zt, &hl);
  const double u = scaled ? hl : 1.0;
  int st[16], ns = 0;
  const int ht = obj->tris.he[h].twin_idx;
  const int wing[4] = {MFTLHeNextIdx(h), MFTLHePrevIdx(h), MFTLHeNextIdx(ht), MFTLHePrevIdx(ht)};
  ns = add_unique(st, ns, MFTLHeOrigin(obj, h));
  ns = add_unique(st, ns, MFTLHeTarget(obj, h));
  ns = add_unique(st, ns, MFTLHeTarget(obj, MFTLHeNextIdx(h)));
  ns = add_unique(st, ns, MFTLHeTarget(obj, MFTLHeNextIdx(ht)));
  for (int k = 0; k < 4; ++k)
  {
    const int tw = obj->tris.he[wing[k]].twin_idx;
    ns = add_unique(st, ns, MFTLHeTarget(obj, MFTLHeNextIdx(tw)));  // the far corner across the wing edge
  }
  for (int i = 0; i < ns; ++i)
  {
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = (MFTLVertexPos(obj, st[i])[k] - m[k]) / u;
    const double x = MFTLVectorInProd(d, ex), y = MFTLVectorInProd(d, ey), z = MFTLVectorInProd(d, nz);
    const double row[6] = {1, x, y, x * x, x * y, y * y};
    for (int c = 0; c < 6; ++c) P->A[c * ns + i] = row[c];
    P->b[i] = z;
  }
  P->m = ns;
  P->h = u;
  P->z_true = zt / u;
  return ns;
}

static void solve_qrcp(const problem *P, double tau, double x[6], int *rank)
{
  double A[6 * MAX_ROWS], b[MAX_ROWS];
  memcpy(A, P->A, sizeof(double) * 6 * P->m);
  memcpy(b, P->b, sizeof(double) * P->m);
  *rank = MFTLLeastSquaresQRCP(P->m, 6, 3, A, b, tau, x, NULL);
}

static double g_work[4096];
static int g_lwork = 0;

static void solve_dgelsy(const problem *P, double rcond, double x[6], int *rank)
{
  double A[6 * MAX_ROWS], b[MAX_ROWS];
  memcpy(A, P->A, sizeof(double) * 6 * P->m);
  memcpy(b, P->b, sizeof(double) * P->m);
  int m = P->m, n = 6, nrhs = 1, lda = P->m, ldb = (P->m > 6 ? P->m : 6), jpvt[6] = {0, 0, 0, 0, 0, 0}, info = 0;
  dgelsy_(&m, &n, &nrhs, A, &lda, b, &ldb, jpvt, &rcond, rank, g_work, &g_lwork, &info);
  for (int i = 0; i < 6; ++i) x[i] = b[i];
}

static void workspace_query(void)
{
  int m = MAX_ROWS, n = 6, nrhs = 1, lda = MAX_ROWS, ldb = MAX_ROWS, jpvt[6] = {0}, rank, info, lw = -1;
  double A[6 * MAX_ROWS] = {0}, b[MAX_ROWS] = {0}, rc = 1e-10, q;
  dgelsy_(&m, &n, &nrhs, A, &lda, b, &ldb, jpvt, &rc, &rank, &q, &lw, &info);
  g_lwork = (int) q;
  if (g_lwork > 4096) g_lwork = 4096;
}

typedef struct
{
  double err_sum, err_max;  // |z - z_true| / |z_true|
  int n, rank_def;
} acc_t;

static void acc_add(acc_t *a, double z, double zt, int rank)
{
  const double e = fabs(z - zt) / fabs(zt);
  a->err_sum += e;
  if (e > a->err_max) a->err_max = e;
  a->n++;
  if (rank < 6) a->rank_def++;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  workspace_query();
  printf("dgelsy optimal workspace: %d doubles\n\n", g_lwork);

  /* ── 1. Speed and agreement on the library's problems ───────────────── */
  printf("1. The library's problems (two-ring, scaled, Wendland radius 3), icosphere R = 1\n");
  printf("%-4s %-9s %6s %5s | %10s %10s %7s | %12s %10s | %12s %12s\n", "ref", "rows", "edges", "m", "QRCP ns", "dgelsy ns", "ratio", "max |dx|/|x|",
         "rank diff", "QRCP err", "dgelsy err");
  for (int ref = 3; ref <= 5; ++ref)
  {
    for (int normals = 0; normals < 2; ++normals)
    {
      int cuts[3] = {1, 1, 1};
      double lo[3] = {-2, -2, -2}, hi[3] = {2, 2, 2}, c[3] = {0, 0, 0};
      bool per[3] = {0, 0, 0};
      MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, per, 1);
      MFTL_Object *obj = MFTLObjectCreate(0);
      MFTLEnsembleAddObject(ens, obj);
      MFTLGenerateSphere(obj, c, 1.0, ref);
      MFTLObjectBuildDerived(obj);

      const int n_he = 3 * obj->tris.n;
      problem *probs = malloc(sizeof(problem) * (size_t) n_he);
      int np = 0;
      double msum = 0;
      for (int h = 0; h < n_he; ++h)
      {
        if (obj->tris.he[h].twin_idx < h) continue;
        build_library(obj, h, 1.0, normals, 3.0, &probs[np]);
        msum += probs[np].m;
        np++;
      }

      acc_t aq = {0}, ad = {0};
      double maxdiff = 0.0;
      int rankdiff = 0;
      for (int i = 0; i < np; ++i)
      {
        double xq[6], xd[6];
        int rq, rd;
        solve_qrcp(&probs[i], 1e-3, xq, &rq);
        solve_dgelsy(&probs[i], 1e-3, xd, &rd);
        double nd = 0, nx = 0;
        for (int k = 0; k < 6; ++k)
        {
          nd += (xq[k] - xd[k]) * (xq[k] - xd[k]);
          nx += xd[k] * xd[k];
        }
        if (sqrt(nd / nx) > maxdiff) maxdiff = sqrt(nd / nx);
        if (rq != rd) rankdiff++;
        acc_add(&aq, xq[0], probs[i].z_true, rq);
        acc_add(&ad, xd[0], probs[i].z_true, rd);
      }

      const int reps = (ref == 5) ? 20 : 100;
      double xs[6];
      int rk;
      double sink = 0.0;
      double t0 = now();
      for (int r = 0; r < reps; ++r)
        for (int i = 0; i < np; ++i)
        {
          solve_qrcp(&probs[i], 1e-3, xs, &rk);
          sink += xs[0];
        }
      const double tq = (now() - t0) / ((double) reps * np);
      t0 = now();
      for (int r = 0; r < reps; ++r)
        for (int i = 0; i < np; ++i)
        {
          solve_dgelsy(&probs[i], 1e-3, xs, &rk);
          sink += xs[0];
        }
      const double td = (now() - t0) / ((double) reps * np);
      printf("%-4d %-9s %6d %5.1f | %10.1f %10.1f %7.2f | %12.2e %10d | %12.3e %12.3e   (%g)\n", ref, normals ? "pos+norm" : "pos", np, msum / np, 1e9 * tq,
             1e9 * td, td / tq, maxdiff, rankdiff, aq.err_sum / aq.n, ad.err_sum / ad.n, sink * 0.0);
      free(probs);
      MFTLEnsembleDestroy(&ens);
    }
  }
  printf("   err: mean |z0 - z_sphere| / z_sphere at the edge midpoint (the sagitta's relative error)\n\n");

  /* ── 1b. Accuracy against the Wendland radius ─────────────────────────── */
  printf("1b. Mean relative sagitta error on the exact sphere against the Wendland radius (QRCP)\n");
  printf("%-4s | %-23s | %-23s | %-23s | %-23s | %s\n", "ref", "radius 2  pos / +norm", "radius 2.5", "radius 3", "radius 4", "mftl 8 pts, scaled");
  for (int ref = 3; ref <= 5; ++ref)
  {
    int cuts[3] = {1, 1, 1};
    double lo[3] = {-2, -2, -2}, hi[3] = {2, 2, 2}, c[3] = {0, 0, 0};
    bool per[3] = {0, 0, 0};
    MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, per, 1);
    MFTL_Object *obj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(ens, obj);
    MFTLGenerateSphere(obj, c, 1.0, ref);
    MFTLObjectBuildDerived(obj);
    printf("%-4d |", ref);
    const double rhos[4] = {2.0, 2.5, 3.0, 4.0};
    for (int ir = 0; ir < 4; ++ir)
    {
      acc_t ap = {0}, an = {0};
      for (int h = 0; h < 3 * obj->tris.n; ++h)
      {
        if (obj->tris.he[h].twin_idx < h) continue;
        problem P;
        double x[6];
        int rk;
        build_library(obj, h, 1.0, false, rhos[ir], &P);
        solve_qrcp(&P, 1e-3, x, &rk);
        acc_add(&ap, x[0], P.z_true, rk);
        build_library(obj, h, 1.0, true, rhos[ir], &P);
        solve_qrcp(&P, 1e-3, x, &rk);
        acc_add(&an, x[0], P.z_true, rk);
      }
      printf(" %10.3e %10.3e |", ap.err_sum / ap.n, an.err_sum / an.n);
    }
    acc_t am = {0};
    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (obj->tris.he[h].twin_idx < h) continue;
      problem P;
      double x[6];
      int rk;
      build_mftl(obj, h, 1.0, true, &P);
      solve_qrcp(&P, 1e-3, x, &rk);
      acc_add(&am, x[0], P.z_true, rk);
    }
    printf(" %10.3e\n", am.err_sum / am.n);
    MFTLEnsembleDestroy(&ens);
  }
  printf("\n");

  /* ── 2. Length unit: mftl's unscaled problem against the scaled one ──── */
  printf("2. Placement error against the sphere's sagitta at refinement 4, sphere radius R\n");
  printf("%-8s | %-28s | %-28s | %-28s | %-28s\n", "R", "mftl: 8 pts, unscaled, dgelsy", "mftl's 8 pts, scaled, QRCP", "scaled two-ring, dgelsy",
         "scaled two-ring, QRCP");
  printf("%-8s | %9s %9s %8s | %9s %9s %8s | %9s %9s %8s | %9s %9s %8s\n", "", "mean err", "max err", "rank<6", "mean err", "max err", "rank<6", "mean err",
         "max err", "rank<6", "mean err", "max err", "rank<6");
  const double radii[6] = {1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
  for (int ir = 0; ir < 6; ++ir)
  {
    const double R = radii[ir];
    int cuts[3] = {1, 1, 1};
    double lo[3] = {-2 * R, -2 * R, -2 * R}, hi[3] = {2 * R, 2 * R, 2 * R}, c[3] = {0, 0, 0};
    bool per[3] = {0, 0, 0};
    MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, per, 1);
    MFTL_Object *obj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(ens, obj);
    MFTLGenerateSphere(obj, c, R, 4);
    MFTLObjectBuildDerived(obj);
    acc_t am = {0}, ams = {0}, ads = {0}, aqs = {0};
    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (obj->tris.he[h].twin_idx < h) continue;
      problem P;
      double x[6];
      int rk;
      build_mftl(obj, h, R, false, &P);
      solve_dgelsy(&P, 1e-10, x, &rk);  // mftl's call
      acc_add(&am, x[0], P.z_true, rk);
      build_mftl(obj, h, R, true, &P);
      solve_qrcp(&P, 1e-3, x, &rk);
      acc_add(&ams, x[0], P.z_true, rk);
      build_library(obj, h, R, false, 3.0, &P);
      solve_dgelsy(&P, 1e-3, x, &rk);
      acc_add(&ads, x[0], P.z_true, rk);
      solve_qrcp(&P, 1e-3, x, &rk);
      acc_add(&aqs, x[0], P.z_true, rk);
    }
    printf("%-8.0e | %9.2e %9.2e %8d | %9.2e %9.2e %8d | %9.2e %9.2e %8d | %9.2e %9.2e %8d\n", R, am.err_sum / am.n, am.err_max, am.rank_def,
           ams.err_sum / ams.n, ams.err_max, ams.rank_def, ads.err_sum / ads.n, ads.err_max, ads.rank_def, aqs.err_sum / aqs.n, aqs.err_max, aqs.rank_def);
    MFTLEnsembleDestroy(&ens);
  }
  printf("   (the columns are ordered 1, x, y, x^2, xy, y^2 in every form; mftl's own order only permutes them)\n\n");

  /* ── 3. Rank-deficient stencils: basic (grouped pivoting) vs minimum norm ─ */
  printf("3. Rank-deficient stencils (scaled coordinates)\n");
  {
    /* (a) 8 samples on the unit circle around the origin, all at height z = 1:
     *     1, x^2 and y^2 are dependent (x^2 + y^2 = 1), so the height at the
     *     origin is not determined by the data alone. */
    problem P;
    P.m = 8;
    for (int i = 0; i < 8; ++i)
    {
      const double t = 2 * M_PI * i / 8.0, x = cos(t), y = sin(t);
      const double row[6] = {1, x, y, x * x, x * y, y * y};
      for (int c = 0; c < 6; ++c) P.A[c * 8 + i] = row[c];
      P.b[i] = 1.0;
    }
    double xq[6], xd[6];
    int rq, rd;
    solve_qrcp(&P, 1e-3, xq, &rq);
    solve_dgelsy(&P, 1e-3, xd, &rd);
    printf("   (a) ring of 8 at z = 1:   QRCP rank %d  c0 = %.6f  (c3, c4, c5) = (%.3f, %.3f, %.3f)\n", rq, xq[0], xq[3], xq[4], xq[5]);
    printf("                             dgelsy rank %d  c0 = %.6f  (c3, c4, c5) = (%.3f, %.3f, %.3f)\n", rd, xd[0], xd[3], xd[4], xd[5]);

    /* (b) A strip: samples along x only (y = 0): y, xy, y^2 undetermined. */
    P.m = 7;
    for (int i = 0; i < 7; ++i)
    {
      const double x = -1.5 + 0.5 * i, y = 0.0;
      const double row[6] = {1, x, y, x * x, x * y, y * y};
      for (int c = 0; c < 6; ++c) P.A[c * 7 + i] = row[c];
      P.b[i] = 0.2 + 0.1 * x - 0.3 * x * x;
    }
    solve_qrcp(&P, 1e-3, xq, &rq);
    solve_dgelsy(&P, 1e-3, xd, &rd);
    printf("   (b) strip along x, z = 0.2 + 0.1x - 0.3x^2:  QRCP rank %d c0 = %.6f   dgelsy rank %d c0 = %.6f\n", rq, xq[0], rd, xd[0]);
  }

  MPI_Finalize();
  return 0;
}

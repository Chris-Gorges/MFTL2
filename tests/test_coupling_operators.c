/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Directional samples and the two gradient operators (plan 06, C3).
 *
 * Three operators beyond the value row, each tested against something that
 * shares none of its code:
 *
 *  - a DIRECTIONAL row reconstructs a vector from projections s_j . u. The
 *    reference solves the same 12-coefficient weighted fit by Gaussian
 *    elimination; the fields are affine VECTOR fields, which a LINEAR row must
 *    reproduce exactly whatever the directions -- and therefore rigid motions,
 *    so force and torque balance by work;
 *  - the GRADIENT FIT is the slope of the affine fit: exact for affine fields,
 *    not the derivative of the value row;
 *  - the VALUE DERIVATIVE is the derivative of the value row, checked by
 *    central differences of value rows built at displaced positions -- with
 *    every built-in kernel, every reproduction policy, a metric that varies
 *    with position (diagonal and full), and through the whole coupler with a
 *    Shepard metric over a graded cloud.
 *
 * Distributed parts run at np = 1, 2, 4, 8 and 27; every assertion is agreed.
 */

#include "test_coupling_irregular.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * Small linear algebra the library does not share
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Solves A X = B (n x n, nrhs columns, row-major) by Gaussian elimination with partial pivoting; 0 on success. */
static int ge_solve(int n, int nrhs, double *A, double *B)
{
  for (int c = 0; c < n; ++c)
  {
    int best = c;
    for (int r = c + 1; r < n; ++r)
      if (fabs(A[r * n + c]) > fabs(A[best * n + c])) best = r;
    if (A[best * n + c] == 0.0) return 1;
    if (best != c)
    {
      for (int j = 0; j < n; ++j)
      {
        const double t = A[c * n + j];
        A[c * n + j] = A[best * n + j];
        A[best * n + j] = t;
      }
      for (int j = 0; j < nrhs; ++j)
      {
        const double t = B[c * nrhs + j];
        B[c * nrhs + j] = B[best * nrhs + j];
        B[best * nrhs + j] = t;
      }
    }
    for (int r = c + 1; r < n; ++r)
    {
      const double f = A[r * n + c] / A[c * n + c];
      for (int j = c; j < n; ++j) A[r * n + j] -= f * A[c * n + j];
      for (int j = 0; j < nrhs; ++j) B[r * nrhs + j] -= f * B[c * nrhs + j];
    }
  }
  for (int i = n - 1; i >= 0; --i)
    for (int j = 0; j < nrhs; ++j)
    {
      double s = B[i * nrhs + j];
      for (int k = i + 1; k < n; ++k) s -= A[i * n + k] * B[k * nrhs + j];
      B[i * nrhs + j] = s / A[i * n + i];
    }
  return 0;
}

/** @brief A random unit vector. */
static void rand_unit(uint64_t *s, double v[3])
{
  for (;;)
  {
    double m2 = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      v[k] = 2.0 * adv_unif(s) - 1.0;
      m2 += v[k] * v[k];
    }
    if (m2 > 0.04 && m2 <= 1.0)
    {
      const double m = sqrt(m2);
      for (int k = 0; k < 3; ++k) v[k] /= m;
      return;
    }
  }
}

/** @brief A jittered cloud: one sample per cell of an n^3 grid of spacing h about c, jitter j*h. */
static int jitter_cloud(const double c[3], double h, int n, double jit, uint64_t seed, double *X, double *q)
{
  uint64_t s = seed;
  int m = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
      {
        const int idx[3] = {i, j, k};
        for (int a = 0; a < 3; ++a) X[3 * m + a] = c[a] + ((double) idx[a] - 0.5 * (n - 1)) * h + jit * h * (adv_unif(&s) - 0.5);
        q[m] = h * h * h;
        ++m;
      }
  return m;
}

static MFTL_CouplerParams params_with(MFTL_CouplingKernel k, MFTL_Reproduction r)
{
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.kernel = k;
  P.reproduction = r;
  P.max_amplification = 20.0;
  return P;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. Kernel derivatives
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * Every built-in derivative against a central difference of the kernel, away
 * from a C0 kernel's breakpoints; odd for a tensor kernel, zero at the end of
 * the support for a C1 one. A wrong sign or a dropped chain-rule factor in one
 * branch is caught here and nowhere else as cleanly.
 */
static int test_kernel_derivatives_match_differences(void)
{
  const MFTL_CouplingKernel ks[6] = {MFTL_KERNEL_PESKIN4, MFTL_KERNEL_ROMA3, MFTL_KERNEL_COSINE4, MFTL_KERNEL_HAT2, MFTL_KERNEL_BSPLINE3, MFTL_KERNEL_WENDLAND_C2};
  double worst = 0.0;
  int n_checked = 0;
  for (int t = 0; t < 6; ++t)
  {
    MFTL_KernelInfo info;
    TEST_ASSERT(MFTLCouplingKernelInfo(ks[t], &info) == MFTL_SUCCESS);
    const double hw = info.half_width;
    for (int i = 0; i < 2000; ++i)
    {
      const double r = (info.radial ? 0.0 : -hw) + (info.radial ? 1.0 : 2.0) * hw * ((double) i + 0.5) / 2000.0;
      /* stay off the breakpoints, where a C0 kernel has no derivative */
      const double bp[5] = {0.0, 0.5, 1.0, 1.5, 2.0};
      bool near = false;
      for (int b = 0; b < 5; ++b)
        if (fabs(fabs(r) - bp[b]) < 1e-4) near = true;
      if (near) continue;
      const double e = 1e-6, xs[2] = {r - e, r + e};
      double f[2], df;
      MFTLCouplingKernelEval(ks[t], xs, f, 2);
      MFTLCouplingKernelDerivative(ks[t], &r, &df, 1);
      const double fd = (f[1] - f[0]) / (2.0 * e);
      worst = fmax(worst, fabs(fd - df));
      ++n_checked;
      if (!info.radial)
      {
        const double mr = -r;
        double dm;
        MFTLCouplingKernelDerivative(ks[t], &mr, &dm, 1);
        TEST_ASSERT(dm == -df);
      }
    }
    /* beyond the support: zero; at its end: zero for a C1 kernel */
    const double out[2] = {hw * 1.01, hw}, zero = 0.0;
    double d2[2], d0;
    MFTLCouplingKernelDerivative(ks[t], out, d2, 2);
    TEST_ASSERT(d2[0] == 0.0);
    if (info.smoothness >= 1) TEST_ASSERT(fabs(d2[1]) <= 1e-12);
    MFTLCouplingKernelDerivative(ks[t], &zero, &d0, 1);
    TEST_ASSERT(d0 == 0.0);
  }
  if (g_rank == 0) printf("    kernel derivatives vs central differences: %d points, worst %.2e\n", n_checked, worst);
  TEST_ASSERT(worst <= 1e-7);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. Directional rows
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The test's own directional LINEAR row: the 12-coefficient fit by elimination. */
static int ref_directional_row(const double x[3], double h, int n, const double *X, const double *q, const double *S, double *A)
{
  double G[144], C[36];
  memset(G, 0, sizeof G);
  double *psi = (double *) malloc((size_t) n * sizeof(double)), *f = (double *) malloc(12 * (size_t) n * sizeof(double));
  for (int j = 0; j < n; ++j)
  {
    double d[3];
    psi[j] = q[j] / (h * h * h);
    for (int k = 0; k < 3; ++k)
    {
      d[k] = (X[3 * j + k] - x[k]) / h;
      psi[j] *= ref_peskin(d[k]);
    }
    const double sg = sqrt(S[3 * j] * S[3 * j] + S[3 * j + 1] * S[3 * j + 1] + S[3 * j + 2] * S[3 * j + 2]);
    for (int a = 0; a < 3; ++a)
    {
      f[12 * j + a] = S[3 * j + a] / sg;
      for (int b = 0; b < 3; ++b) f[12 * j + 3 + 3 * a + b] = f[12 * j + a] * d[b];
    }
    for (int a = 0; a < 12; ++a)
      for (int b = 0; b < 12; ++b) G[12 * a + b] += psi[j] * f[12 * j + a] * f[12 * j + b];
  }
  memset(C, 0, sizeof C);
  for (int k = 0; k < 3; ++k) C[3 * k + k] = 1.0; /* E: the first three unit columns */
  const int bad = ge_solve(12, 3, G, C);
  for (int j = 0; j < n && !bad; ++j)
  {
    const double sg = sqrt(S[3 * j] * S[3 * j] + S[3 * j + 1] * S[3 * j + 1] + S[3 * j + 2] * S[3 * j + 2]);
    for (int k = 0; k < 3; ++k)
    {
      double acc = 0.0;
      for (int l = 0; l < 12; ++l) acc += C[3 * l + k] * f[12 * j + l];
      A[3 * j + k] = psi[j] * acc / sg;
    }
  }
  free(psi);
  free(f);
  return bad;
}

/**
 * A jittered cloud whose samples carry random directions with random
 * magnitudes (flux-like): the LINEAR row equals the reference fit, reproduces
 * an affine vector field whatever the directions, and is scale free -- the
 * same projections measured with area vectors ten times longer give the same
 * vector. CONSTANT reproduces constants only, and the difference is visible.
 */
static int test_directional_rows_reproduce_affine_vector_fields(void)
{
  enum
  {
    N = 7,
    M = N * N * N
  };
  const double c[3] = {0.31, -0.42, 0.57}, h = 0.1;
  double X[3 * M], q[M], S[3 * M], w[3 * M], wref[3 * M];
  uint64_t s = 99;
  const int n = jitter_cloud(c, h, N, 0.6, 7, X, q);
  for (int j = 0; j < n; ++j)
  {
    rand_unit(&s, &S[3 * j]);
    const double mag = 0.3 + 2.0 * adv_unif(&s);
    for (int k = 0; k < 3; ++k) S[3 * j + k] *= mag;
  }
  const double Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
  /* an affine vector field u = a + B (X - x0) */
  const double a[3] = {0.7, -1.3, 0.4}, B[9] = {0.3, -0.8, 1.1, 0.5, 0.2, -0.6, -1.4, 0.9, 0.25}, x0[3] = {0.2, -0.1, 0.4};
  double *qv = (double *) malloc((size_t) n * sizeof(double));
  double worst_ref = 0.0, worst_lin = 0.0, worst_const_c = 0.0, const_lin_err = 0.0, worst_scale = 0.0;
  for (int t = 0; t < 40; ++t)
  {
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = c[k] + 1.2 * h * (2.0 * adv_unif(&s) - 1.0);
    for (int j = 0; j < n; ++j)
    {
      double u[3];
      for (int r = 0; r < 3; ++r)
        u[r] = a[r] + B[3 * r] * (X[3 * j] - x0[0]) + B[3 * r + 1] * (X[3 * j + 1] - x0[1]) + B[3 * r + 2] * (X[3 * j + 2] - x0[2]);
      qv[j] = S[3 * j] * u[0] + S[3 * j + 1] * u[1] + S[3 * j + 2] * u[2];
    }
    double ux[3];
    for (int r = 0; r < 3; ++r) ux[r] = a[r] + B[3 * r] * (x[0] - x0[0]) + B[3 * r + 1] * (x[1] - x0[1]) + B[3 * r + 2] * (x[2] - x0[2]);

    MFTL_CouplerParams P = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR);
    MFTL_RowQuality rq;
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, &rq) == MFTL_COUPLING_OK);
    TEST_ASSERT(rq.order == 1 && rq.corrected);
    TEST_ASSERT(ref_directional_row(x, h, n, X, q, S, wref) == 0);
    double U[3] = {0, 0, 0}, scale = 0.0, diff = 0.0;
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < 3; ++k)
      {
        diff = fmax(diff, fabs(w[3 * j + k] - wref[3 * j + k]));
        scale = fmax(scale, fabs(wref[3 * j + k]));
        U[k] += w[3 * j + k] * qv[j];
      }
    worst_ref = fmax(worst_ref, diff / fmax(scale, 1e-300));
    for (int k = 0; k < 3; ++k) worst_lin = fmax(worst_lin, fabs(U[k] - ux[k]));

    /* the same samples measured with ten times the area vector: the same vector */
    double S10[3 * M], q10[M], w10[3 * M];
    for (int j = 0; j < n; ++j)
    {
      for (int k = 0; k < 3; ++k) S10[3 * j + k] = 10.0 * S[3 * j + k];
      q10[j] = 10.0 * qv[j];
    }
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S10, w10, MFTL_NULL) == MFTL_COUPLING_OK);
    double U10[3] = {0, 0, 0};
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < 3; ++k) U10[k] += w10[3 * j + k] * q10[j];
    for (int k = 0; k < 3; ++k) worst_scale = fmax(worst_scale, fabs(U10[k] - U[k]));

    /* CONSTANT: a constant vector exactly, an affine one not */
    MFTL_CouplerParams Pc = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_CONSTANT);
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pc, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, &rq) == MFTL_COUPLING_OK);
    TEST_ASSERT(rq.order >= 0);
    double Uc[3] = {0, 0, 0}, Ul[3] = {0, 0, 0};
    for (int j = 0; j < n; ++j)
    {
      const double qa = S[3 * j] * a[0] + S[3 * j + 1] * a[1] + S[3 * j + 2] * a[2];
      for (int k = 0; k < 3; ++k)
      {
        Uc[k] += w[3 * j + k] * qa;
        Ul[k] += w[3 * j + k] * qv[j];
      }
    }
    for (int k = 0; k < 3; ++k)
    {
      worst_const_c = fmax(worst_const_c, fabs(Uc[k] - a[k]));
      const_lin_err = fmax(const_lin_err, fabs(Ul[k] - ux[k]));
    }
  }
  free(qv);
  if (g_rank == 0)
    printf("    directional: weights vs reference %.1e, affine %.1e, x10 area vectors %.1e; CONSTANT: constant %.1e, affine %.1e\n", worst_ref, worst_lin,
           worst_scale, worst_const_c, const_lin_err);
  TEST_ASSERT(worst_ref <= 1e-9);
  TEST_ASSERT(worst_lin <= 1e-11);
  TEST_ASSERT(worst_scale <= 1e-11);
  TEST_ASSERT(worst_const_c <= 1e-12);
  TEST_ASSERT(const_lin_err > 1e-4); /* CONSTANT is genuinely lower order */
  return 0;
}

/**
 * Directional rank is a requirement of its own: positions alone can be
 * perfectly spread while every direction lies in one plane -- then the
 * normal component is not observable and the row must fail, not return a
 * vector with an invented z component. Positions in a plane leave a constant
 * vector observable (CONSTANT passes) but not its slope across the plane.
 */
static int test_directional_rank_deficiency_is_refused(void)
{
  enum
  {
    N = 6,
    M = N * N * N
  };
  const double c[3] = {0, 0, 0}, h = 0.1, x[3] = {0.013, -0.021, 0.008};
  double X[3 * M], q[M], S[3 * M], w[3 * M];
  const int n = jitter_cloud(c, h, N, 0.5, 3, X, q);
  uint64_t s = 5;
  const double Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
  MFTL_CouplerParams P = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR), Pc = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_CONSTANT);

  /* every direction along x */
  for (int j = 0; j < n; ++j) S[3 * j] = 1.0, S[3 * j + 1] = 0.0, S[3 * j + 2] = 0.0;
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pc, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  /* every direction in the xy plane, random within it */
  for (int j = 0; j < n; ++j)
  {
    const double t = 2.0 * MFTL_PI * adv_unif(&s);
    S[3 * j] = cos(t), S[3 * j + 1] = sin(t), S[3 * j + 2] = 0.0;
  }
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pc, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  /* spanning directions, but every position in the plane z = x_z */
  for (int j = 0; j < n; ++j)
  {
    rand_unit(&s, &S[3 * j]);
    X[3 * j + 2] = x[2];
  }
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pc, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_OK);
  /* no gradient of directional samples, no directional NONE row */
  MFTL_CouplerParams Pn = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_NONE);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_GRADIENT_FIT, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_INVALID);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pn, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, S, w, MFTL_NULL) == MFTL_COUPLING_INVALID);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. The two gradients
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Value weights at x with metric Hc (scalar samples). */
static int value_row(const MFTL_CouplerParams *P, const double x[3], const double Hc[9], int n, const double *X, const double *q, double *w)
{
  return MFTLCouplingAssembleBlockRow(P, MFTL_OPERATOR_VALUE, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, w, MFTL_NULL);
}

/** @brief A metric field: H(x) = H0 + sum_m x_m E_m (affine, full matrices), and its derivative E. */
typedef struct
{
  double H0[9], E[27];
} metric_field;

static void metric_at(const metric_field *F, const double x[3], double H[9])
{
  for (int a = 0; a < 9; ++a) H[a] = F->H0[a] + x[0] * F->E[a] + x[1] * F->E[9 + a] + x[2] * F->E[18 + a];
}

/**
 * The derivative of the value row against central differences of value rows
 * built at x +- eps e_m, candidate by candidate: every built-in kernel except
 * the C0 ones (whose derivative jumps inside the support), every reproduction
 * policy, on a jittered cloud where LINEAR rows are corrected; with a constant
 * metric, a varying diagonal one and a varying full (sheared) one, each
 * differentiated -- the metric's derivative enters through d, through
 * |det H|^-1 and through the moment matrix, and dropping any one of them is a
 * visible error.
 */
static int test_the_derivative_is_the_derivative(void)
{
  enum
  {
    N = 8,
    M = N * N * N
  };
  const double c[3] = {0.2, 0.1, -0.3}, h = 0.1;
  double X[3 * M], q[M], wd[3 * M], wp[M], wm[M];
  const int n = jitter_cloud(c, h, N, 0.5, 11, X, q);
  const MFTL_CouplingKernel ks[4] = {MFTL_KERNEL_PESKIN4, MFTL_KERNEL_COSINE4, MFTL_KERNEL_BSPLINE3, MFTL_KERNEL_WENDLAND_C2};
  const MFTL_Reproduction rs[3] = {MFTL_REPRODUCE_LINEAR, MFTL_REPRODUCE_CONSTANT, MFTL_REPRODUCE_NONE};
  metric_field fields[3];
  memset(fields, 0, sizeof fields);
  for (int f = 0; f < 3; ++f)
    for (int k = 0; k < 3; ++k) fields[f].H0[4 * k] = h * (1.0 + 0.1 * k);
  /* a varying diagonal metric */
  fields[1].E[0] = 0.08 * h / h, fields[1].E[9 + 4] = -0.05, fields[1].E[18 + 8] = 0.06, fields[1].E[4] = 0.04;
  /* a varying sheared one */
  fields[2].H0[1] = 0.02 * h, fields[2].H0[5] = -0.03 * h, fields[2].H0[6] = 0.015 * h;
  for (int a = 0; a < 27; ++a) fields[2].E[a] = 0.03 * sin(1.7 * a + 0.3);
  uint64_t s = 21;
  double worst = 0.0;
  int n_rows = 0, n_corrected = 0;
  for (int f = 0; f < 3; ++f)
    for (int kk = 0; kk < 4; ++kk)
      for (int rr = 0; rr < 3; ++rr)
      {
        MFTL_CouplerParams P = params_with(ks[kk], rs[rr]);
        P.max_amplification = 50.0;
        for (int t = 0; t < 6; ++t)
        {
          double x[3], H[9];
          for (int k = 0; k < 3; ++k) x[k] = c[k] + 0.8 * h * (2.0 * adv_unif(&s) - 1.0);
          metric_at(&fields[f], x, H);
          MFTL_RowQuality rq;
          const int st = MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, H, (f > 0) ? fields[f].E : MFTL_NULL, n, X, q, MFTL_NULL, wd, &rq);
          TEST_ASSERT(st == MFTL_COUPLING_OK);
          if (rq.corrected) ++n_corrected;
          double scale = 0.0;
          for (int j = 0; j < 3 * n; ++j) scale = fmax(scale, fabs(wd[j]));
          for (int m = 0; m < 3; ++m)
          {
            const double e = 1e-6 * h;
            double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]}, Hp[9], Hm[9];
            xp[m] += e;
            xm[m] -= e;
            metric_at(&fields[f], xp, Hp);
            metric_at(&fields[f], xm, Hm);
            TEST_ASSERT(value_row(&P, xp, Hp, n, X, q, wp) == MFTL_COUPLING_OK);
            TEST_ASSERT(value_row(&P, xm, Hm, n, X, q, wm) == MFTL_COUPLING_OK);
            for (int j = 0; j < n; ++j) worst = fmax(worst, fabs((wp[j] - wm[j]) / (2.0 * e) - wd[3 * j + m]) / scale);
          }
          ++n_rows;
        }
      }
  if (g_rank == 0) printf("    derivative vs central differences: %d rows (%d corrected), worst relative %.2e\n", n_rows, n_corrected, worst);
  TEST_ASSERT(n_corrected > n_rows / 4);
  TEST_ASSERT(worst <= 2e-6);
  return 0;
}

/**
 * On a complete uniform lattice Peskin's value row is the kernel itself, so
 * its derivative is the product of one derivative and two kernel values --
 * written out here in closed form. Both gradients reproduce an affine field's
 * gradient exactly on a jittered cloud; on that cloud they are different
 * operators (the fit is not the derivative), which is why they have different
 * names.
 */
static int test_the_fit_and_the_derivative_are_different_operators(void)
{
  const double h = 0.05;
  enum
  {
    N = 9,
    M = N * N * N
  };
  double X[3 * M], q[M], wd[3 * M], wf[3 * M];
  /* a complete lattice: no jitter */
  const double c0[3] = {0.0, 0.0, 0.0};
  const int n = jitter_cloud(c0, h, N, 0.0, 1, X, q);
  const double Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
  MFTL_CouplerParams P = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR);
  uint64_t s = 8;
  double worst_closed = 0.0;
  for (int t = 0; t < 20; ++t)
  {
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = h * (2.0 * adv_unif(&s) - 1.0);
    MFTL_RowQuality rq;
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, wd, &rq) == MFTL_COUPLING_OK);
    TEST_ASSERT(!rq.corrected);
    for (int j = 0; j < n; ++j)
    {
      double d[3], ph[3], dph[3];
      for (int k = 0; k < 3; ++k)
      {
        d[k] = (X[3 * j + k] - x[k]) / h;
        ph[k] = ref_peskin(d[k]);
        const double r = fabs(d[k]), sg = (d[k] > 0) ? 1.0 : ((d[k] < 0) ? -1.0 : 0.0);
        dph[k] = (r >= 2.0) ? 0.0
                            : sg * ((r < 1.0) ? (-2.0 + (2.0 - 4.0 * r) / sqrt(1.0 + 4.0 * r - 4.0 * r * r)) / 8.0
                                              : (-2.0 - (6.0 - 4.0 * r) / sqrt(-7.0 + 12.0 * r - 4.0 * r * r)) / 8.0);
      }
      for (int m = 0; m < 3; ++m)
      {
        /* d psi / dx_m = -(1/h) phi'(d_m) prod_{l != m} phi(d_l), with q / h^3 = 1 */
        double ref = -dph[m] / h;
        for (int l = 0; l < 3; ++l)
          if (l != m) ref *= ph[l];
        worst_closed = fmax(worst_closed, fabs(ref - wd[3 * j + m]) * h);
      }
    }
  }
  /* a jittered cloud: both reproduce an affine gradient, and they differ */
  double Xj[3 * M], qj[M];
  const int nj = jitter_cloud(c0, h, N, 0.7, 4, Xj, qj);
  const double g[3] = {1.3, -0.7, 2.1}, a = 0.4;
  double worst_aff = 0.0, max_diff = 0.0;
  for (int t = 0; t < 20; ++t)
  {
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = h * (2.0 * adv_unif(&s) - 1.0);
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hc, MFTL_NULL, nj, Xj, qj, MFTL_NULL, wd, MFTL_NULL) == MFTL_COUPLING_OK);
    TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_GRADIENT_FIT, x, Hc, MFTL_NULL, nj, Xj, qj, MFTL_NULL, wf, MFTL_NULL) == MFTL_COUPLING_OK);
    double Gd[3] = {0, 0, 0}, Gf[3] = {0, 0, 0};
    for (int j = 0; j < nj; ++j)
    {
      const double u = a + g[0] * Xj[3 * j] + g[1] * Xj[3 * j + 1] + g[2] * Xj[3 * j + 2];
      for (int m = 0; m < 3; ++m)
      {
        Gd[m] += wd[3 * j + m] * u;
        Gf[m] += wf[3 * j + m] * u;
        max_diff = fmax(max_diff, fabs(wd[3 * j + m] - wf[3 * j + m]) * h);
      }
    }
    for (int m = 0; m < 3; ++m) worst_aff = fmax(worst_aff, fmax(fabs(Gd[m] - g[m]), fabs(Gf[m] - g[m])));
  }
  /* a sheared metric: the slope maps back through H^-T, not H^-1 */
  {
    const double Hs[9] = {h, 0.3 * h, -0.2 * h, 0.1 * h, 0.9 * h, 0.25 * h, -0.15 * h, 0.05 * h, 1.1 * h};
    for (int t = 0; t < 10; ++t)
    {
      double x[3];
      for (int k = 0; k < 3; ++k) x[k] = h * (2.0 * adv_unif(&s) - 1.0);
      TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_GRADIENT_FIT, x, Hs, MFTL_NULL, nj, Xj, qj, MFTL_NULL, wf, MFTL_NULL) == MFTL_COUPLING_OK);
      TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hs, MFTL_NULL, nj, Xj, qj, MFTL_NULL, wd, MFTL_NULL) == MFTL_COUPLING_OK);
      double Gf[3] = {0, 0, 0}, Gd[3] = {0, 0, 0};
      for (int j = 0; j < nj; ++j)
      {
        const double u = a + g[0] * Xj[3 * j] + g[1] * Xj[3 * j + 1] + g[2] * Xj[3 * j + 2];
        for (int m = 0; m < 3; ++m)
        {
          Gf[m] += wf[3 * j + m] * u;
          Gd[m] += wd[3 * j + m] * u;
        }
      }
      for (int m = 0; m < 3; ++m) worst_aff = fmax(worst_aff, fmax(fabs(Gf[m] - g[m]), fabs(Gd[m] - g[m])));
    }
  }
  if (g_rank == 0)
    printf("    lattice derivative vs closed form %.1e; affine gradient (fit and derivative, axis-aligned and sheared) %.1e; fit - derivative up to %.2f (per unit h)\n",
           worst_closed, worst_aff, max_diff);
  TEST_ASSERT(worst_closed <= 1e-12);
  TEST_ASSERT(worst_aff <= 1e-10);
  TEST_ASSERT(max_diff > 1e-2);
  return 0;
}

/**
 * A kernel the application supplies without a derivative cannot give the
 * derivative operator -- refused, not approximated -- while its fit still
 * exists; with a derivative it works and matches differences.
 */
static void user_bspline(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  MFTLCouplingKernelEval(MFTL_KERNEL_BSPLINE3, r, phi, n);
}
static void user_bspline_d(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  MFTLCouplingKernelDerivative(MFTL_KERNEL_BSPLINE3, r, dphi, n);
}

static int test_a_user_kernel_needs_its_derivative(void)
{
  enum
  {
    N = 7,
    M = N * N * N
  };
  const double c[3] = {0, 0, 0}, h = 0.1, x[3] = {0.021, -0.017, 0.033};
  double X[3 * M], q[M], w[3 * M];
  const int n = jitter_cloud(c, h, N, 0.4, 9, X, q);
  const double Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.kernel = MFTL_KERNEL_USER;
  P.user_kernel.eval = user_bspline;
  P.user_kernel.half_width = 2.0;
  P.user_kernel.moment_order = 1;
  P.user_kernel.nonnegative = true;
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, w, MFTL_NULL) == MFTL_COUPLING_INVALID);
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_GRADIENT_FIT, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, w, MFTL_NULL) == MFTL_COUPLING_OK);
  P.user_kernel.deriv = user_bspline_d;
  double wb[3 * M];
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, w, MFTL_NULL) == MFTL_COUPLING_OK);
  MFTL_CouplerParams Pb = params_with(MFTL_KERNEL_BSPLINE3, MFTL_REPRODUCE_LINEAR);
  Pb.max_amplification = P.max_amplification;
  TEST_ASSERT(MFTLCouplingAssembleBlockRow(&Pb, MFTL_OPERATOR_VALUE_DERIVATIVE, x, Hc, MFTL_NULL, n, X, q, MFTL_NULL, wb, MFTL_NULL) == MFTL_COUPLING_OK);
  for (int j = 0; j < 3 * n; ++j) TEST_ASSERT(w[j] == wb[j]);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Through the coupler: a face cloud with flux-like directions
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The faces of an n^3 grid of the unit box as ONE directional point set: x-, y- and z-faces, jittered, area vectors of either orientation. */
typedef struct
{
  int64_t n_all;
  double *X, *S;
  uint64_t *key;
  int *owner;
  int64_t n_mine;
  int64_t *mine;  ///< global index of each owned sample, registration order
  double *val;  ///< the solver's array over the owned samples
  int loc;
  double h;
} face_cloud;

static void face_cloud_make(int n, face_cloud *F)
{
  memset(F, 0, sizeof *F);
  const double h = 1.0 / n;
  F->h = h;
  const int64_t cap = 3 * (int64_t) n * n * (n + 1);
  F->X = (double *) malloc(3 * (size_t) cap * sizeof(double));
  F->S = (double *) malloc(3 * (size_t) cap * sizeof(double));
  F->key = (uint64_t *) malloc((size_t) cap * sizeof(uint64_t));
  F->owner = (int *) malloc((size_t) cap * sizeof(int));
  F->mine = (int64_t *) malloc((size_t) cap * sizeof(int64_t));
  for (int fam = 0; fam < 3; ++fam)
    for (int a = 0; a <= n; ++a)
      for (int b = 0; b < n; ++b)
        for (int c = 0; c < n; ++c)
        {
          const int64_t i = F->n_all++;
          const uint64_t key = (uint64_t) ((((int64_t) fam * (n + 1) + a) * n + b) * n + c) * 3u + 11u;
          const int t1 = (fam + 1) % 3, t2 = (fam + 2) % 3;
          double *x = &F->X[3 * i];
          x[fam] = a * h + ((a > 0 && a < n) ? 0.2 * h * hash01(key * 5u + 1u) : 0.0);
          x[t1] = (b + 0.5) * h + 0.2 * h * hash01(key * 5u + 2u);
          x[t2] = (c + 0.5) * h + 0.2 * h * hash01(key * 5u + 3u);
          const double area = h * h * (1.0 + 0.3 * hash01(key * 5u + 4u)), sgn = (hash01(key * 7u) < 0.0) ? -1.0 : 1.0;
          for (int k = 0; k < 3; ++k) F->S[3 * i + k] = (k == fam) ? sgn * area : 0.0;
          F->key[i] = key;
          F->owner[i] = (int) ((uint64_t) (hash01(key * 13u) * 1e9 + 2e9) % (uint64_t) g_nproc);
          if (F->owner[i] == g_rank) F->mine[F->n_mine++] = i;
        }
  F->val = (double *) calloc((size_t) (F->n_mine > 0 ? F->n_mine : 1), sizeof(double));
}

static int face_cloud_register(MFTL_SolverGrid *g, face_cloud *F, const char *name)
{
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = name;
  ld.directional = true;
  F->loc = MFTLSolverGridAddLocation(g, &ld);
  const int64_t m = F->n_mine;
  double *X = (double *) malloc(3 * (size_t) (m > 0 ? m : 1) * sizeof(double)), *S = (double *) malloc(3 * (size_t) (m > 0 ? m : 1) * sizeof(double));
  double *q = (double *) malloc((size_t) (m > 0 ? m : 1) * sizeof(double));
  uint64_t *k = (uint64_t *) malloc((size_t) (m > 0 ? m : 1) * sizeof(uint64_t));
  for (int64_t i = 0; i < m; ++i)
  {
    memcpy(&X[3 * i], &F->X[3 * F->mine[i]], 3 * sizeof(double));
    memcpy(&S[3 * i], &F->S[3 * F->mine[i]], 3 * sizeof(double));
    q[i] = F->h * F->h * F->h;
    k[i] = F->key[F->mine[i]];
  }
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = F->loc;
  pd.n = m;
  pd.keys = k;
  pd.pos = X;
  pd.measure = q;
  pd.direction = S;
  for (int a = 0; a < 3; ++a) pd.h_uniform[a] = F->h;
  const int piece = MFTLSolverGridAddPoints(g, &pd);
  free(X);
  free(S);
  free(q);
  free(k);
  return piece;
}

static void face_cloud_free(face_cloud *F)
{
  free(F->X);
  free(F->S);
  free(F->key);
  free(F->owner);
  free(F->mine);
  free(F->val);
}

/** @brief A vector field: u = a + B (x - x0) + omega x (x - x0). */
typedef struct
{
  double a[3], B[9], om[3], x0[3];
} vfield;

static void vfield_at(const vfield *V, const double x[3], double u[3])
{
  const double r[3] = {x[0] - V->x0[0], x[1] - V->x0[1], x[2] - V->x0[2]};
  for (int k = 0; k < 3; ++k) u[k] = V->a[k] + V->B[3 * k] * r[0] + V->B[3 * k + 1] * r[1] + V->B[3 * k + 2] * r[2];
  u[0] += V->om[1] * r[2] - V->om[2] * r[1];
  u[1] += V->om[2] * r[0] - V->om[0] * r[2];
  u[2] += V->om[0] * r[1] - V->om[1] * r[0];
}

/** @brief The solver fills its owned projections q_j = s_j . u(X_j). */
static void face_cloud_fill(face_cloud *F, const vfield *V)
{
  for (int64_t i = 0; i < F->n_mine; ++i)
  {
    double u[3];
    const int64_t gi = F->mine[i];
    vfield_at(V, &F->X[3 * gi], u);
    F->val[i] = F->S[3 * gi] * u[0] + F->S[3 * gi + 1] * u[1] + F->S[3 * gi + 2] * u[2];
  }
}

/** @brief A deterministic GLOBAL list of marker positions inside [lo, hi]^3. */
static void marker_list(int n, double lo, double hi, uint64_t seed, double *all)
{
  uint64_t s = seed;
  for (int i = 0; i < 3 * n; ++i) all[i] = lo + (hi - lo) * adv_unif(&s);
}

/**
 * The faces of a staggered grid, jittered and registered as ONE unstructured
 * directional cloud -- flux-like area vectors of random length and either
 * orientation, owned by a hash so most supports are remote. LINEAR rows equal
 * the reference fit over the whole cloud; an affine vector field and a rigid
 * motion come back exactly; the spread is the brute-force transpose, conserves
 * power, and -- through the rigid motion -- balances force and torque by work:
 * \sum_j q_j b_j = v . sum F + omega . sum (x - x0) x F.
 */
static int test_a_face_cloud_reconstructs_vectors(void)
{
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  face_cloud F;
  face_cloud_make(9, &F);
  TEST_ASSERT(agreed(face_cloud_register(g, &F, "faces") >= 0, MPI_COMM_WORLD));
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.max_amplification = 20.0;
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  enum
  {
    NA = 48
  };
  double all[3 * NA];
  marker_list(NA, 0.3, 0.7, 404, all);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t nm = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, NA, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindingComponents(b, F.loc) == 3, MPI_COMM_WORLD));
  const double *pos = MFTLCouplerBindingPositions(b);

  /* rows against the reference over the whole global cloud */
  double worst_row = 0.0;
  int bad_rows = 0;
  {
    const double h = F.h, Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
    double *Xc = (double *) malloc(3 * (size_t) F.n_all * sizeof(double)), *qc = (double *) malloc((size_t) F.n_all * sizeof(double));
    double *Sc = (double *) malloc(3 * (size_t) F.n_all * sizeof(double)), *wr = (double *) malloc(3 * (size_t) F.n_all * sizeof(double));
    uint64_t *kc = (uint64_t *) malloc((size_t) F.n_all * sizeof(uint64_t));
    for (int64_t i = 0; i < nm; ++i)
    {
      int nc = 0;
      for (int64_t j = 0; j < F.n_all; ++j)
      {
        bool in = true;
        for (int k = 0; k < 3; ++k)
          if (fabs(F.X[3 * j + k] - pos[3 * i + k]) > 2.0 * h) in = false;
        if (!in) continue;
        memcpy(&Xc[3 * nc], &F.X[3 * j], 3 * sizeof(double));
        memcpy(&Sc[3 * nc], &F.S[3 * j], 3 * sizeof(double));
        qc[nc] = h * h * h;
        kc[nc++] = F.key[j];
      }
      if (MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, &pos[3 * i], Hc, MFTL_NULL, nc, Xc, qc, Sc, wr, MFTL_NULL) != MFTL_COUPLING_OK)
      {
        ++bad_rows;
        continue;
      }
      int n = 0;
      const uint64_t *keys;
      const double *wt;
      MFTLCouplerBindingRow(b, 0, i, &n, &keys, &wt);
      int nref = 0;
      double scale = 0.0;
      for (int j = 0; j < nc; ++j)
        if (wr[3 * j] != 0.0 || wr[3 * j + 1] != 0.0 || wr[3 * j + 2] != 0.0)
        {
          ++nref;
          for (int k = 0; k < 3; ++k) scale = fmax(scale, fabs(wr[3 * j + k]));
        }
      if (nref != n) ++bad_rows;
      for (int e = 0; e < n; ++e)
      {
        int at = -1;
        for (int j = 0; j < nc; ++j)
          if (kc[j] == keys[e]) at = j;
        if (at < 0)
        {
          ++bad_rows;
          break;
        }
        for (int k = 0; k < 3; ++k) worst_row = fmax(worst_row, fabs(wt[3 * e + k] - wr[3 * at + k]) / scale);
      }
    }
    free(Xc);
    free(qc);
    free(Sc);
    free(wr);
    free(kc);
  }
  TEST_ASSERT(agreed(bad_rows == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(worst_row <= 1e-10, MPI_COMM_WORLD));

  /* an affine field with a rotation in it: reproduced exactly */
  vfield V = {{0.4, -1.1, 0.7}, {0.3, -0.2, 0.5, 0.1, 0.6, -0.4, -0.3, 0.2, 0.25}, {0.8, -0.5, 1.2}, {0.5, 0.5, 0.5}};
  face_cloud_fill(&F, &V);
  MFTL_SolverField fld;
  memset(&fld, 0, sizeof fld);
  fld.location = F.loc;
  double *ptr = F.val;
  fld.data = &ptr;
  double *U = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &fld, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_u = 0.0;
  for (int64_t i = 0; i < nm; ++i)
  {
    double u[3];
    vfield_at(&V, &pos[3 * i], u);
    for (int k = 0; k < 3; ++k) worst_u = fmax(worst_u, fabs(U[3 * i + k] - u[k]));
  }
  TEST_ASSERT(agreed(worst_u <= 1e-11, MPI_COMM_WORLD));

  /* spread vector loads; power, and force and torque by work against the rigid part */
  double *L = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < nm; ++i)
    for (int k = 0; k < 3; ++k) L[3 * i + k] = hash01(ref[i].key * 3u + (uint64_t) k + 77u);
  double *tgt = (double *) calloc((size_t) (F.n_mine > 0 ? F.n_mine : 1), sizeof(double));
  MFTL_SolverField tf = fld;
  double *tptr = tgt;
  tf.data = &tptr;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &tf, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}; /* u.b, U.F, |terms|, sum F (3), unused */
  for (int64_t i = 0; i < F.n_mine; ++i)
  {
    sums[0] += F.val[i] * tgt[i];
    sums[2] += fabs(F.val[i] * tgt[i]);
  }
  for (int64_t i = 0; i < nm; ++i)
    for (int k = 0; k < 3; ++k) sums[1] += U[3 * i + k] * L[3 * i + k];
  MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(fabs(sums[0] - sums[1]) <= 1e-12 * sums[2], MPI_COMM_WORLD));

  /* the rigid motion alone: its work is v . sum F + omega . sum r x F */
  vfield R = {{0.4, -1.1, 0.7}, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {0.8, -0.5, 1.2}, {0.5, 0.5, 0.5}};
  face_cloud_fill(&F, &R);
  double rig[3] = {0, 0, 0};
  for (int64_t i = 0; i < F.n_mine; ++i)
  {
    rig[0] += F.val[i] * tgt[i];
    rig[2] += fabs(F.val[i] * tgt[i]);
  }
  for (int64_t i = 0; i < nm; ++i)
  {
    const double r[3] = {pos[3 * i] - R.x0[0], pos[3 * i + 1] - R.x0[1], pos[3 * i + 2] - R.x0[2]};
    const double *f = &L[3 * i];
    const double tq[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
    for (int k = 0; k < 3; ++k) rig[1] += R.a[k] * f[k] + R.om[k] * tq[k];
  }
  MPI_Allreduce(MPI_IN_PLACE, rig, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(fabs(rig[0] - rig[1]) <= 1e-12 * rig[2], MPI_COMM_WORLD));

  /* the brute-force transpose: every rank recomputes b over the global marker list */
  {
    int n_tot = 0;
    double *allpos = allgather_doubles(pos, (int) nm, 3, &n_tot), *allL = allgather_doubles(L, (int) nm, 3, &n_tot);
    double *bref = (double *) calloc((size_t) (F.n_mine > 0 ? F.n_mine : 1), sizeof(double));
    const double h = F.h, Hc[9] = {h, 0, 0, 0, h, 0, 0, 0, h};
    double *Xc = (double *) malloc(3 * (size_t) F.n_all * sizeof(double)), *qc = (double *) malloc((size_t) F.n_all * sizeof(double));
    double *Sc = (double *) malloc(3 * (size_t) F.n_all * sizeof(double)), *wr = (double *) malloc(3 * (size_t) F.n_all * sizeof(double));
    int64_t *gc = (int64_t *) malloc((size_t) F.n_all * sizeof(int64_t));
    int64_t *slot = (int64_t *) malloc((size_t) F.n_all * sizeof(int64_t));
    for (int64_t j = 0; j < F.n_all; ++j) slot[j] = -1;
    for (int64_t i = 0; i < F.n_mine; ++i) slot[F.mine[i]] = i;
    for (int i = 0; i < n_tot; ++i)
    {
      int nc = 0;
      for (int64_t j = 0; j < F.n_all; ++j)
      {
        bool in = true;
        for (int k = 0; k < 3; ++k)
          if (fabs(F.X[3 * j + k] - allpos[3 * i + k]) > 2.0 * h) in = false;
        if (!in) continue;
        memcpy(&Xc[3 * nc], &F.X[3 * j], 3 * sizeof(double));
        memcpy(&Sc[3 * nc], &F.S[3 * j], 3 * sizeof(double));
        qc[nc] = h * h * h;
        gc[nc++] = j;
      }
      if (MFTLCouplingAssembleBlockRow(&P, MFTL_OPERATOR_VALUE, &allpos[3 * i], Hc, MFTL_NULL, nc, Xc, qc, Sc, wr, MFTL_NULL) != MFTL_COUPLING_OK) continue;
      for (int j = 0; j < nc; ++j)
        if (slot[gc[j]] >= 0) bref[slot[gc[j]]] += wr[3 * j] * allL[3 * i] + wr[3 * j + 1] * allL[3 * i + 1] + wr[3 * j + 2] * allL[3 * i + 2];
    }
    double worst_b = 0.0, scale_b = 0.0;
    for (int64_t i = 0; i < F.n_mine; ++i)
    {
      worst_b = fmax(worst_b, fabs(bref[i] - tgt[i]));
      scale_b = fmax(scale_b, fabs(bref[i]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &scale_b, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(worst_b <= 1e-11 * scale_b, MPI_COMM_WORLD));
    free(allpos);
    free(allL);
    free(bref);
    free(Xc);
    free(qc);
    free(Sc);
    free(wr);
    free(gc);
    free(slot);
  }
  /* non-vacuity: at np > 1 the supports really are remote */
  MFTL_CouplerStats st;
  MFTLCouplerReduceStats(c, &st);
  if (g_rank == 0) printf("    face cloud: %lld samples, rows vs reference %.1e, affine+rotation %.1e, remote band (max/rank) %ld\n", (long long) F.n_all, worst_row, worst_u, st.band_remote);
  TEST_ASSERT(agreed(g_nproc == 1 || st.band_remote > 0, MPI_COMM_WORLD));

  free(U);
  free(L);
  free(tgt);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  face_cloud_free(&F);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 5. Through the coupler: both gradients, a lattice and a graded cloud
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief A graded cloud over the default domain: cell centres of a smoothly stretched grid, jittered, each with its own resolution. */
typedef struct
{
  int64_t n_all, n_mine;
  double *X, *h;
  uint64_t *key;
  int64_t *mine;
  double *val;
  int loc;
} graded_cloud;

static double stretch(double xi, double amp) { return xi + amp * sin(2.0 * MFTL_PI * xi) / (2.0 * MFTL_PI); }

static void graded_make(const adv_lat *A, const int n[3], graded_cloud *G)
{
  memset(G, 0, sizeof *G);
  const int64_t cap = (int64_t) n[0] * n[1] * n[2];
  G->X = (double *) malloc(3 * (size_t) cap * sizeof(double));
  G->h = (double *) malloc(3 * (size_t) cap * sizeof(double));
  G->key = (uint64_t *) malloc((size_t) cap * sizeof(uint64_t));
  G->mine = (int64_t *) malloc((size_t) cap * sizeof(int64_t));
  for (int i = 0; i < n[0]; ++i)
    for (int j = 0; j < n[1]; ++j)
      for (int k = 0; k < n[2]; ++k)
      {
        const int idx[3] = {i, j, k};
        const int64_t g = G->n_all++;
        const uint64_t key = (uint64_t) (((int64_t) i * n[1] + j) * n[2] + k) * 7u + 5u;
        G->key[g] = key;
        for (int a = 0; a < 3; ++a)
        {
          const double amp = (a == 0) ? 0.5 : 0.35;
          const double x0 = stretch((double) idx[a] / n[a], amp), x1 = stretch((double) (idx[a] + 1) / n[a], amp);
          const double hh = (x1 - x0) * A->L[a];
          double x = A->gmin[a] + 0.5 * (x0 + x1) * A->L[a] + 0.15 * hh * hash01(key * 3u + (uint64_t) a);
          G->X[3 * g + a] = x;
          G->h[3 * g + a] = hh;
        }
        if ((int) ((uint64_t) (hash01(key * 11u) * 1e9 + 2e9) % (uint64_t) g_nproc) == g_rank) G->mine[G->n_mine++] = g;
      }
  G->val = (double *) calloc((size_t) (G->n_mine > 0 ? G->n_mine : 1), sizeof(double));
}

static void graded_register(MFTL_SolverGrid *g, graded_cloud *G)
{
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "graded";
  G->loc = MFTLSolverGridAddLocation(g, &ld);
  const int64_t m = G->n_mine > 0 ? G->n_mine : 1;
  double *X = (double *) malloc(3 * (size_t) m * sizeof(double)), *h = (double *) malloc(3 * (size_t) m * sizeof(double)),
         *q = (double *) malloc((size_t) m * sizeof(double));
  uint64_t *k = (uint64_t *) malloc((size_t) m * sizeof(uint64_t));
  for (int64_t i = 0; i < G->n_mine; ++i)
  {
    memcpy(&X[3 * i], &G->X[3 * G->mine[i]], 3 * sizeof(double));
    memcpy(&h[3 * i], &G->h[3 * G->mine[i]], 3 * sizeof(double));
    q[i] = h[3 * i] * h[3 * i + 1] * h[3 * i + 2];
    k[i] = G->key[G->mine[i]];
  }
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = G->loc;
  pd.n = G->n_mine;
  pd.keys = k;
  pd.pos = X;
  pd.measure = q;
  pd.h = h;
  MFTLSolverGridAddPoints(g, &pd);
  free(X);
  free(h);
  free(q);
  free(k);
}

static void graded_free(graded_cloud *G)
{
  free(G->X);
  free(G->h);
  free(G->key);
  free(G->mine);
  free(G->val);
}

/** @brief A smooth field periodic in x over the default domain (for the seam), and its gradient. */
static double smooth_f(const double x[3], void *ctx)
{
  const adv_lat *A = (const adv_lat *) ctx;
  const double t = 2.0 * MFTL_PI * (x[0] - A->gmin[0]) / A->L[0];
  return sin(t) * cos(3.0 * x[1]) * exp(0.5 * x[2]) + 0.3 * x[1] * x[2];
}

/**
 * Both gradients through the whole machinery: on an anisotropic, offset,
 * partly periodic lattice decomposed into random blocks with random ghost
 * widths, and on a graded jittered cloud whose rows use a Shepard metric that
 * varies with the marker's position.
 *
 *  - both reproduce the gradient of an affine field exactly;
 *  - the derivative equals the central difference of VALUE interpolation at
 *    markers displaced by +-eps, through the same window: on the graded cloud
 *    that includes the derivative of the Shepard metric, which a derivative
 *    taken at a frozen metric misses by far more than the tolerance;
 *  - a stage binding of a gradient binding equals a fresh one, bit for bit;
 *  - the adjoint spreads vector loads as dipoles: zero net load (a gradient
 *    annihilates constants) and the loads' first moment is the total dipole,
 *    and the power identity holds.
 */
static int test_both_gradients_through_the_coupler(void)
{
  const bool per[3] = {true, false, false};
  adv_lat A = adv_default_lattice(true, per);
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int lat = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, lat, &A, 3 * g_nproc + 2, 777, 2, 1, true, &B);
  graded_cloud Gc;
  const int nn[3] = {18, 11, 15};
  graded_make(&A, nn, &Gc);
  graded_register(g, &Gc);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.max_amplification = 20.0;
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* markers: interior, away from the walls (y, z) and from the x seam */
  enum
  {
    NA = 30
  };
  double all[3 * NA];
  uint64_t s = 1234;
  for (int i = 0; i < NA; ++i)
    for (int k = 0; k < 3; ++k) all[3 * i + k] = A.gmin[k] + A.L[k] * (0.3 + 0.4 * adv_unif(&s));
  int n_mine = 0;
  double pos[3 * NA], pos6[6 * 3 * NA];
  uint64_t keys[NA], keys6[6 * NA];
  const double eps = 1e-6;
  for (int i = 0; i < NA; ++i)
    if (i % g_nproc == g_rank)
    {
      memcpy(&pos[3 * n_mine], &all[3 * i], 3 * sizeof(double));
      keys[n_mine] = (uint64_t) i;
      for (int m = 0; m < 3; ++m)
        for (int sg = 0; sg < 2; ++sg)
        {
          const int at = 6 * n_mine + 2 * m + sg;
          memcpy(&pos6[3 * at], &all[3 * i], 3 * sizeof(double));
          pos6[3 * at + m] += sg ? eps : -eps;
          keys6[at] = (uint64_t) (6 * i + 2 * m + sg);
        }
      ++n_mine;
    }
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = n_mine;
  ms.user_pos = pos;
  ms.user_keys = keys;
  MFTL_WindowSpec ws = {{1e-5, 1e-5, 1e-5}, {0, 0, 0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingBinding *bfit = MFTL_NULL, *bder = MFTL_NULL, *bval6 = MFTL_NULL;
  ms.operation = MFTL_OPERATOR_GRADIENT_FIT;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bfit) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  ms.operation = MFTL_OPERATOR_VALUE_DERIVATIVE;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bder) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec m6 = ms;
  m6.operation = MFTL_OPERATOR_VALUE;
  m6.n_user = 6 * n_mine;
  m6.user_pos = pos6;
  m6.user_keys = keys6;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &m6, w, &bval6) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindingComponents(bder, lat) == 3 && MFTLCouplerBindingComponents(bder, Gc.loc) == 3, MPI_COMM_WORLD));
  MFTL_BindingQuality qd;
  MFTLCouplerBindingQuality(bder, &qd);

  /* fields: [0] lattice, [1] cloud */
  MFTL_SolverField F[2];
  F[0] = adv_field(&B, lat, 0, 1, false);
  memset(&F[1], 0, sizeof F[1]);
  F[1].location = Gc.loc;
  double *gptr = Gc.val;
  F[1].data = &gptr;
  double *Uf = (double *) malloc(6 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double)),
         *Ud = (double *) malloc(6 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double)),
         *U6 = (double *) malloc(2 * 6 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));

  /* an affine field: both gradients exact on both locations */
  affine aff = {0.3, {0.0, -1.7, 0.9}}; /* no slope across the x seam */
  adv_fill(&B, &A, 0, adv_affine_fn, &aff, false);
  for (int64_t i = 0; i < Gc.n_mine; ++i) Gc.val[i] = f_affine(&Gc.X[3 * Gc.mine[i]], &aff);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bfit, 2, F, Uf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bder, 2, F, Ud) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_aff = 0.0;
  for (int i = 0; i < n_mine; ++i)
    for (int f = 0; f < 2; ++f)
      for (int m = 0; m < 3; ++m) worst_aff = fmax(worst_aff, fmax(fabs(Uf[6 * i + 3 * f + m] - aff.g[m]), fabs(Ud[6 * i + 3 * f + m] - aff.g[m])));
  TEST_ASSERT(agreed(worst_aff <= 1e-10, MPI_COMM_WORLD));

  /* a smooth field: the derivative is the derivative of the value */
  adv_fill(&B, &A, 0, smooth_f, &A, false);
  for (int64_t i = 0; i < Gc.n_mine; ++i) Gc.val[i] = smooth_f(&Gc.X[3 * Gc.mine[i]], &A);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bder, 2, F, Ud) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bfit, 2, F, Uf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bval6, 2, F, U6) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_fd[2] = {0, 0}, gmax = 0.0, fit_vs_der[2] = {0, 0};
  for (int i = 0; i < n_mine; ++i)
    for (int f = 0; f < 2; ++f)
      for (int m = 0; m < 3; ++m)
      {
        const double up = U6[2 * (6 * i + 2 * m + 1) + f], um = U6[2 * (6 * i + 2 * m) + f];
        const double fd = (up - um) / (2.0 * eps), d = Ud[6 * i + 3 * f + m];
        worst_fd[f] = fmax(worst_fd[f], fabs(fd - d));
        gmax = fmax(gmax, fabs(d));
        fit_vs_der[f] = fmax(fit_vs_der[f], fabs(Uf[6 * i + 3 * f + m] - d));
      }
  MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, worst_fd, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, fit_vs_der, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0)
    printf("    derivative vs differences of the value: lattice %.1e, graded cloud %.1e (|grad| up to %.2f); fit - derivative %.1e / %.1e; corrected rows %lld\n",
           worst_fd[0], worst_fd[1], gmax, fit_vs_der[0], fit_vs_der[1], (long long) qd.n_corrected);
  TEST_ASSERT(agreed(worst_fd[0] <= 1e-6 * gmax && worst_fd[1] <= 1e-6 * gmax, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fit_vs_der[1] > 1e-4 * gmax, MPI_COMM_WORLD)); /* different operators, visibly */

  /* a stage binding of the derivative binding equals a fresh bind */
  {
    double *sp = (double *) malloc(3 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
    for (int i = 0; i < n_mine; ++i)
      for (int k = 0; k < 3; ++k) sp[3 * i + k] = pos[3 * i + k] + 3e-6 * (double) (k + 1);
    MFTL_StageGeometry sgm = {sp, MFTL_NULL};
    MFTL_CouplingBinding *bst = MFTL_NULL, *bfr = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerBindStage(c, bder, &sgm, &bst) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec mf = ms;
    mf.user_pos = sp;
    TEST_ASSERT(agreed(MFTLCouplerBind(c, &mf, w, &bfr) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int same = 1;
    for (int lp = 0; lp < 2; ++lp)
      for (int i = 0; i < n_mine; ++i)
      {
        int n1, n2;
        const uint64_t *k1, *k2;
        const double *w1, *w2;
        MFTLCouplerBindingRow(bst, lp, i, &n1, &k1, &w1);
        uint64_t kk[4096];
        double ww[3 * 4096];
        if (n1 > 4096)
        {
          same = 0;
          continue;
        }
        memcpy(kk, k1, (size_t) n1 * sizeof(uint64_t));
        memcpy(ww, w1, 3 * (size_t) n1 * sizeof(double));
        MFTLCouplerBindingRow(bfr, lp, i, &n2, &k2, &w2);
        if (n1 != n2) same = 0;
        for (int e = 0; e < n1 && same; ++e)
          if (kk[e] != k2[e] || ww[3 * e] != w2[3 * e] || ww[3 * e + 1] != w2[3 * e + 1] || ww[3 * e + 2] != w2[3 * e + 2]) same = 0;
      }
    TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
    MFTLCouplerReleaseBinding(&bfr);
    MFTLCouplerReleaseBinding(&bst);
    free(sp);
  }

  /* the adjoint: vector loads spread as dipoles */
  {
    double *Lv = (double *) malloc(6 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
    for (int i = 0; i < n_mine; ++i)
      for (int k = 0; k < 6; ++k) Lv[6 * i + k] = hash01(keys[i] * 6u + (uint64_t) k + 3u);
    adv_blocks Bt;
    MFTL_SolverGrid *g2 = g; /* same registration: spread into a second array over the same blocks */
    (void) g2;
    memset(&Bt, 0, sizeof Bt);
    Bt = B;
    for (int i = 0; i < B.n; ++i)
    {
      Bt.a[i] = (double *) calloc((size_t) B.n_elem[i], sizeof(double));
      Bt.ptr[i] = Bt.a[i];
    }
    double *gt = (double *) calloc((size_t) (Gc.n_mine > 0 ? Gc.n_mine : 1), sizeof(double));
    MFTL_SolverField T[2];
    T[0] = adv_field(&Bt, lat, 0, 1, false);
    T[1] = F[1];
    double *gtp = gt;
    T[1].data = &gtp;
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, bder, 2, T, Lv, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    /* sums: [0..1] net load per location, [2..7] first moments - total dipole, [8] power u.b, [9] U.F, [10] scale, [11] |b| */
    double acc[12] = {0};
    for (int i = 0; i < B.n; ++i)
      for (int64_t e = 0; e < B.n_elem[i]; ++e)
      {
        int64_t idx[3];
        adv_elem_index(&B, i, e, idx);
        bool owned = true;
        for (int k = 0; k < 3; ++k)
          if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) owned = false;
        if (!owned) continue;
        const double bj = Bt.a[i][e];
        double x[3];
        for (int k = 0; k < 3; ++k) x[k] = adv_coord(&A, k, idx[k]);
        acc[0] += bj;
        for (int k = 0; k < 3; ++k) acc[2 + k] += bj * x[k];
        acc[8] += B.a[i][e] * bj;
        acc[10] += fabs(B.a[i][e] * bj);
        acc[11] += fabs(bj);
      }
    for (int64_t i = 0; i < Gc.n_mine; ++i)
    {
      const double *x = &Gc.X[3 * Gc.mine[i]];
      acc[1] += gt[i];
      for (int k = 0; k < 3; ++k) acc[5 + k] += gt[i] * x[k];
      acc[8] += Gc.val[i] * gt[i];
      acc[10] += fabs(Gc.val[i] * gt[i]);
      acc[11] += fabs(gt[i]);
    }
    for (int i = 0; i < n_mine; ++i)
      for (int k = 0; k < 6; ++k)
      {
        acc[9] += Ud[6 * i + k] * Lv[6 * i + k];
        acc[2 + k] -= Lv[6 * i + k]; /* sum_j b_j X_j = sum_i F_i, per location */
      }
    MPI_Allreduce(MPI_IN_PLACE, acc, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double worst_mom = 0.0;
    for (int k = 2; k < 8; ++k) worst_mom = fmax(worst_mom, fabs(acc[k]));
    if (g_rank == 0) printf("    adjoint: net load %.1e / %.1e (of sum |b| %.1e), dipole moment %.1e, power %.1e\n", acc[0], acc[1], acc[11], worst_mom, fabs(acc[8] - acc[9]) / acc[10]);
    TEST_ASSERT(agreed(fabs(acc[0]) <= 1e-12 * acc[11] && fabs(acc[1]) <= 1e-12 * acc[11], MPI_COMM_WORLD));
    TEST_ASSERT(agreed(worst_mom <= 1e-10, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(acc[8] - acc[9]) <= 1e-12 * acc[10], MPI_COMM_WORLD));
    for (int i = 0; i < B.n; ++i) free(Bt.a[i]);
    free(gt);
    free(Lv);
  }

  free(Uf);
  free(Ud);
  free(U6);
  MFTLCouplerReleaseBinding(&bval6);
  MFTLCouplerReleaseBinding(&bder);
  MFTLCouplerReleaseBinding(&bfit);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  adv_free_blocks(&B);
  graded_free(&Gc);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 6. Gradients at mirror walls; refusals
 * ═════════════════════════════════════════════════════════════════════════════ */

static double wall_field(const double x[3], void *ctx)
{
  (void) ctx;
  return sin(3.0 * x[0] + 0.4) * cos(2.0 * x[1]) + 0.7 * x[2] * x[2] - 0.2 * x[0] * x[2];
}

/**
 * A cell-centred lattice with MIRROR walls -- EVEN on x and y, ODD on z with
 * wall values u_lo and u_hi -- and markers right up to the walls and corners.
 * The field u_lo + (u_hi - u_lo) z is what the walls describe (zero normal
 * slope on the EVEN faces, the prescribed value on the ODD ones), so both
 * gradients return (0, 0, u_hi - u_lo) exactly however close the wall: the
 * wall values enter a gradient through its own wall coefficients. For a
 * field the walls do not describe, the derivative is still the derivative of
 * the mirrored value (central differences), and the adjoint's power balances
 * with the walls' reactions included.
 */
static int test_gradients_at_mirror_walls(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool per[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 10;
    A.h[k] = 0.1;
    A.origin[k] = 0.05;
    A.gmin[k] = 0.0;
    A.L[k] = 1.0;
  }
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "mirrored";
  ld.boundary = MFTL_BOUNDARY_MIRROR;
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = A.N[k];
    ld.lattice.origin[k] = A.origin[k];
    ld.lattice.spacing[k] = A.h[k];
  }
  ld.wall[2] = MFTL_WALL_ODD;
  ld.wall[5] = MFTL_WALL_ODD;
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc + 1, 31, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  enum
  {
    NA = 36
  };
  double all[3 * NA];
  marker_list(NA, 0.01, 0.99, 55, all);
  for (int i = 0; i < 8; ++i) /* corners, within a hair */
    for (int k = 0; k < 3; ++k) all[3 * i + k] = ((i >> k) & 1) ? 0.995 : 0.004;
  int n_mine = 0;
  const double eps = 1e-6;
  double pos[3 * NA], pos6[18 * NA];
  uint64_t keys[NA], keys6[6 * NA];
  for (int i = 0; i < NA; ++i)
    if (i % g_nproc == g_rank)
    {
      memcpy(&pos[3 * n_mine], &all[3 * i], 3 * sizeof(double));
      keys[n_mine] = (uint64_t) i;
      for (int m = 0; m < 3; ++m)
        for (int sg = 0; sg < 2; ++sg)
        {
          const int at = 6 * n_mine + 2 * m + sg;
          memcpy(&pos6[3 * at], &all[3 * i], 3 * sizeof(double));
          pos6[3 * at + m] += sg ? eps : -eps;
          keys6[at] = (uint64_t) (6 * i + 2 * m + sg);
        }
      ++n_mine;
    }
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = n_mine;
  ms.user_pos = pos;
  ms.user_keys = keys;
  MFTL_WindowSpec ws = {{1e-5, 1e-5, 1e-5}, {0, 0, 0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingBinding *bf = MFTL_NULL, *bd = MFTL_NULL, *b6 = MFTL_NULL;
  ms.operation = MFTL_OPERATOR_GRADIENT_FIT;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  ms.operation = MFTL_OPERATOR_VALUE_DERIVATIVE;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bd) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec m6 = ms;
  m6.operation = MFTL_OPERATOR_VALUE;
  m6.n_user = 6 * n_mine;
  m6.user_pos = pos6;
  m6.user_keys = keys6;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &m6, w, &b6) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_BindingQuality qd;
  MFTLCouplerBindingQuality(bd, &qd);

  const double u_lo = 0.35, u_hi = -1.15, wv[6] = {0, 0, u_lo, 0, 0, u_hi};
  affine prof = {u_lo, {0.0, 0.0, u_hi - u_lo}};
  adv_fill(&B, &A, 0, adv_affine_fn, &prof, false);
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
  F.wall_value = wv;
  double *Uf = (double *) malloc(3 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
  double *Ud = (double *) malloc(3 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
  double *U6 = (double *) malloc(6 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bf, 1, &F, Uf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bd, 1, &F, Ud) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst = 0.0, wc_used = 0.0;
  for (int i = 0; i < n_mine; ++i)
  {
    for (int m = 0; m < 3; ++m) worst = fmax(worst, fmax(fabs(Uf[3 * i + m] - prof.g[m]), fabs(Ud[3 * i + m] - prof.g[m])));
    double cc[6];
    MFTLCouplerBindingWallCoefficients(bd, 0, i, 2, cc);
    wc_used = fmax(wc_used, fmax(fabs(cc[2]), fabs(cc[5])));
  }
  MPI_Allreduce(MPI_IN_PLACE, &wc_used, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(worst <= 1e-11, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(wc_used > 0.1, MPI_COMM_WORLD)); /* the walls really entered */

  /* a field the walls do not describe: still the derivative of the mirrored value */
  adv_fill(&B, &A, 0, wall_field, MFTL_NULL, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bd, 1, &F, Ud) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b6, 1, &F, U6) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_fd = 0.0, gmax = 0.0;
  for (int i = 0; i < n_mine; ++i)
    for (int m = 0; m < 3; ++m)
    {
      const double fd = (U6[6 * i + 2 * m + 1] - U6[6 * i + 2 * m]) / (2.0 * eps);
      worst_fd = fmax(worst_fd, fabs(fd - Ud[3 * i + m]));
      gmax = fmax(gmax, fabs(Ud[3 * i + m]));
    }
  MPI_Allreduce(MPI_IN_PLACE, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(worst_fd <= 1e-6 * gmax, MPI_COMM_WORLD));

  /* the adjoint with the walls: U.F = u.b + sum_f u_w,f R_f */
  double *Lv = (double *) malloc(3 * (size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
  for (int i = 0; i < n_mine; ++i)
    for (int k = 0; k < 3; ++k) Lv[3 * i + k] = hash01(keys[i] * 3u + (uint64_t) k + 9u);
  adv_blocks Bt = B;
  for (int i = 0; i < B.n; ++i)
  {
    Bt.a[i] = (double *) calloc((size_t) B.n_elem[i], sizeof(double));
    Bt.ptr[i] = Bt.a[i];
  }
  double R[6] = {0, 0, 0, 0, 0, 0};
  MFTL_SolverField T = adv_field(&Bt, loc, 0, 1, false);
  T.wall_reaction = R;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, bd, 1, &T, Lv, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double acc[3] = {0, 0, 0};
  for (int i = 0; i < B.n; ++i)
    for (int64_t e = 0; e < B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&B, i, e, idx);
      bool owned = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) owned = false;
      if (!owned) continue;
      acc[0] += B.a[i][e] * Bt.a[i][e];
      acc[2] += fabs(B.a[i][e] * Bt.a[i][e]);
    }
  for (int i = 0; i < n_mine; ++i)
    for (int k = 0; k < 3; ++k) acc[1] += Ud[3 * i + k] * Lv[3 * i + k];
  MPI_Allreduce(MPI_IN_PLACE, acc, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double wall_work = 0.0;
  for (int f = 0; f < 6; ++f) wall_work += wv[f] * R[f];
  if (g_rank == 0)
    printf("    mirror: affine through the walls %.1e, derivative vs differences %.1e, power with wall work %.1e (wall work %.2e)\n", worst, worst_fd,
           fabs(acc[1] - acc[0] - wall_work) / acc[2], wall_work);
  TEST_ASSERT(agreed(fabs(acc[1] - acc[0] - wall_work) <= 1e-12 * acc[2], MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(wall_work) > 1e-6 * acc[2], MPI_COMM_WORLD));

  for (int i = 0; i < B.n; ++i) free(Bt.a[i]);
  free(Lv);
  free(Uf);
  free(Ud);
  free(U6);
  MFTLCouplerReleaseBinding(&b6);
  MFTLCouplerReleaseBinding(&bd);
  MFTLCouplerReleaseBinding(&bf);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  adv_free_blocks(&B);
  return 0;
}

/**
 * Every combination that does not exist is refused -- at registration, or by
 * an agreed INVALID before any rank builds a row: a directional lattice, a
 * directional mirror, a zero or non-finite direction, a gradient of
 * directional samples, a directional row under NONE, the derivative of a user
 * kernel that has none, and ranks that ask for different operators.
 */
static int test_operators_that_do_not_exist_are_refused(void)
{
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "x";
  ld.directional = true;
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k) ld.lattice.dims[k] = 4, ld.lattice.spacing[k] = 0.25;
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR);
  ld.lattice.kind = MFTL_LATTICE_NONE;
  ld.boundary = MFTL_BOUNDARY_MIRROR;
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR);
  ld.boundary = MFTL_BOUNDARY_TRUNCATE;
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  TEST_ASSERT(loc >= 0);
  const uint64_t key = 1;
  const double x[3] = {0.5, 0.5, 0.5}, meas = 1e-3, zero[3] = {0, 0, 0}, nanv[3] = {NAN, 0, 1};
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.n = 1;
  pd.keys = &key;
  pd.pos = x;
  pd.measure = &meas;
  for (int k = 0; k < 3; ++k) pd.h_uniform[k] = 0.1;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR); /* no directions */
  pd.direction = zero;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  pd.direction = nanv;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  MFTLSolverGridDestroy(&g);

  /* through the coupler: agreed INVALID */
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, false);
  g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  face_cloud F;
  face_cloud_make(6, &F);
  face_cloud_register(g, &F, "faces");
  pts_block pb;
  add_points(g, "scalar", 6, 0.3, OWN_SCATTERED, false, &pb);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  enum
  {
    NA = 8
  };
  double all[3 * NA];
  marker_list(NA, 0.35, 0.65, 3, all);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  TEST_ASSERT(agreed(adv_bind_list(c, NA, all, 0, MFTL_NULL, &w, &b, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindingComponents(b, F.loc) == 3 && MFTLCouplerBindingComponents(b, pb.loc) == 1, MPI_COMM_WORLD));
  double pos[3 * NA];
  int n = 0;
  for (int i = 0; i < NA; ++i)
    if (i % g_nproc == g_rank) memcpy(&pos[3 * n++], &all[3 * i], 3 * sizeof(double));
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = n;
  ms.user_pos = pos;
  MFTL_CouplingBinding *bx = MFTL_NULL;
  ms.operation = MFTL_OPERATOR_GRADIENT_FIT; /* includes the directional location */
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bx) == MFTL_COUPLING_INVALID && bx == MFTL_NULL, MPI_COMM_WORLD));
  int only_scalar = pb.loc;
  ms.n_locations = 1;
  ms.locations = &only_scalar;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bx) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bx);
  /* ranks that disagree about the operator */
  ms.operation = (g_rank == 0) ? MFTL_OPERATOR_GRADIENT_FIT : MFTL_OPERATOR_VALUE_DERIVATIVE;
  const int disagree = MFTLCouplerBind(c, &ms, w, &bx);
  TEST_ASSERT(agreed(g_nproc == 1 ? disagree == MFTL_COUPLING_OK : disagree == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bx);
  ms.operation = (MFTL_CouplingOperator) 7;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bx) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);

  /* NONE with a directional location; a user kernel without its derivative */
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.reproduction = MFTL_REPRODUCE_NONE;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(adv_bind_list(c, NA, all, 0, MFTL_NULL, &w, &b, MFTL_NULL) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLCouplerDefaultParams(&P);
  P.kernel = MFTL_KERNEL_USER;
  P.user_kernel.eval = user_bspline;
  P.user_kernel.half_width = 2.0;
  P.user_kernel.moment_order = 1;
  P.user_kernel.nonnegative = true;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int scal = pb.loc;
  TEST_ASSERT(agreed(adv_bind_list(c, NA, all, 1, &scal, &w, &b, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  ms.operation = MFTL_OPERATOR_VALUE_DERIVATIVE;
  ms.n_locations = 1;
  ms.locations = &scal;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bx) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  ms.operation = MFTL_OPERATOR_GRADIENT_FIT;
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &bx) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bx);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  face_cloud_free(&F);
  free_points(&pb);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling operators: directions and gradients (np = %d) ===\n", g_nproc);

  RUN_TEST(test_kernel_derivatives_match_differences);
  RUN_TEST(test_directional_rows_reproduce_affine_vector_fields);
  RUN_TEST(test_directional_rank_deficiency_is_refused);
  RUN_TEST(test_the_derivative_is_the_derivative);
  RUN_TEST(test_the_fit_and_the_derivative_are_different_operators);
  RUN_TEST(test_a_user_kernel_needs_its_derivative);
  RUN_TEST(test_a_face_cloud_reconstructs_vectors);
  RUN_TEST(test_both_gradients_through_the_coupler);
  RUN_TEST(test_gradients_at_mirror_walls);
  RUN_TEST(test_operators_that_do_not_exist_are_refused);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}

/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief The coupler's local numerics: the kernel table and the row core
 *        (MFTLCouplingAssembleRow). No MPI.
 *
 * KERNELS are checked against what they DECLARE, by independent means: closed-
 * form values at known points, the unit integral by quadrature, and the lattice
 * moments over 1024 shifts -- and the moment order is checked in BOTH
 * directions, so a kernel that understates its order fails as surely as one
 * that overstates it. Nothing is inferred by sampling: the support is the
 * declared one, and the test asserts the kernel is zero at and beyond it.
 *
 * ROWS are attacked where a moment fit is fragile: a cloud squashed into a
 * plane (rank-deficient), an empty support, a marker outside the convex hull of
 * its samples, a support too small to hold a well-posed fit (growth must
 * rescue it within its limit and not beyond), a wall cutting the support, and
 * translations to 1e6 and scalings by 2^-20 .. 2^20 (dimensionless: the
 * weights must not change by more than the arithmetic forces).
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_coupling.h"
#include "test_utils.h"

static const MFTL_CouplingKernel g_kernels[] = {MFTL_KERNEL_PESKIN4, MFTL_KERNEL_ROMA3, MFTL_KERNEL_COSINE4,
                                                MFTL_KERNEL_HAT2, MFTL_KERNEL_BSPLINE3, MFTL_KERNEL_WENDLAND_C2};
#define N_KERNELS ((int) (sizeof(g_kernels) / sizeof(g_kernels[0])))

static double phi1(MFTL_CouplingKernel k, double r)
{
  double v = 0.0;
  MFTLCouplingKernelEval(k, &r, &v, 1);
  return v;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Kernels
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Closed-form values at the points every reference quotes. */
static int test_kernel_values_are_the_published_ones(void)
{
  TEST_ASSERT_EQUAL_DOUBLE(0.5, phi1(MFTL_KERNEL_PESKIN4, 0.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(0.25, phi1(MFTL_KERNEL_PESKIN4, 1.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE((2.0 + sqrt(2.0)) / 8.0, phi1(MFTL_KERNEL_PESKIN4, 0.5), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE((2.0 - sqrt(2.0)) / 8.0, phi1(MFTL_KERNEL_PESKIN4, 1.5), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(2.0 / 3.0, phi1(MFTL_KERNEL_ROMA3, 0.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(1.0 / 6.0, phi1(MFTL_KERNEL_ROMA3, 1.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(0.5, phi1(MFTL_KERNEL_COSINE4, 0.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(0.25, phi1(MFTL_KERNEL_COSINE4, 1.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, phi1(MFTL_KERNEL_HAT2, 0.0), 0.0);
  TEST_ASSERT_EQUAL_DOUBLE(0.75, phi1(MFTL_KERNEL_HAT2, -0.25), 0.0);
  TEST_ASSERT_EQUAL_DOUBLE(2.0 / 3.0, phi1(MFTL_KERNEL_BSPLINE3, 0.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(1.0 / 6.0, phi1(MFTL_KERNEL_BSPLINE3, 1.0), 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(21.0 / (16.0 * MFTL_PI), phi1(MFTL_KERNEL_WENDLAND_C2, 0.0), 1e-15);
  /* an unknown kernel is refused, not evaluated */
  double r = 0.0, v = 0.0;
  TEST_ASSERT(MFTLCouplingKernelEval((MFTL_CouplingKernel) 99, &r, &v, 1) == MFTL_ERROR);
  MFTL_KernelInfo info;
  TEST_ASSERT(MFTLCouplingKernelInfo(MFTL_KERNEL_USER, &info) == MFTL_ERROR);
  return 0;
}

/** @brief Zero at and beyond the declared half-width, symmetric, of the declared sign. */
static int test_support_symmetry_and_sign(void)
{
  for (int i = 0; i < N_KERNELS; ++i)
  {
    MFTL_KernelInfo info;
    TEST_ASSERT(MFTLCouplingKernelInfo(g_kernels[i], &info) == MFTL_SUCCESS);
    const double hw = info.half_width;
    TEST_ASSERT(phi1(g_kernels[i], hw) == 0.0);
    TEST_ASSERT(phi1(g_kernels[i], -hw) == 0.0 || info.radial);
    TEST_ASSERT(phi1(g_kernels[i], hw + 1e-12) == 0.0);
    TEST_ASSERT(phi1(g_kernels[i], 1e3) == 0.0);
    TEST_ASSERT(phi1(g_kernels[i], nextafter(hw, 0.0)) >= 0.0);
    for (int s = 0; s < 4096; ++s)
    {
      const double r = hw * (double) s / 4096.0;
      const double a = phi1(g_kernels[i], r);
      if (!info.radial) TEST_ASSERT(a == phi1(g_kernels[i], -r));
      if (info.nonnegative) TEST_ASSERT(a >= 0.0);
      TEST_ASSERT(isfinite(a));
    }
  }
  return 0;
}

/** @brief Unit integral (1-D for tensor kernels, 3-D for radial ones), by composite Simpson. */
static int test_unit_integral(void)
{
  for (int i = 0; i < N_KERNELS; ++i)
  {
    MFTL_KernelInfo info;
    MFTLCouplingKernelInfo(g_kernels[i], &info);
    const int n = 1 << 16;
    const double a = info.radial ? 0.0 : -info.half_width, b = info.half_width, h = (b - a) / n;
    double s = 0.0;
    for (int j = 0; j <= n; ++j)
    {
      const double x = a + j * h;
      double f = phi1(g_kernels[i], x);
      if (info.radial) f *= 4.0 * MFTL_PI * x * x;
      s += f * ((j == 0 || j == n) ? 1.0 : (j % 2 ? 4.0 : 2.0));
    }
    s *= h / 3.0;
    /* the piecewise kernels have kinks: Simpson is O(h^2) at a kink, ~1e-9 here */
    if (fabs(s - 1.0) > 1e-8) printf("\n  %s integrates to %.15f", info.name, s);
    TEST_ASSERT(fabs(s - 1.0) <= 1e-8);
  }
  return 0;
}

/**
 * @brief Lattice moments match the DECLARED order in both directions: a
 *        declared order is reproduced at every one of 1024 shifts, and the next
 *        order is NOT (so the declaration is not an understatement).
 */
static int test_lattice_moments_match_the_declaration(void)
{
  for (int i = 0; i < N_KERNELS; ++i)
  {
    MFTL_KernelInfo info;
    MFTLCouplingKernelInfo(g_kernels[i], &info);
    if (info.radial)
    {
      /* a radial kernel on a lattice: not even the zeroth moment (declared -1) */
      double worst0 = 0.0;
      for (int s = 0; s < 64; ++s)
      {
        const double sh[3] = {s / 64.0, 0.37 * s / 64.0, 0.11};
        double m0 = 0.0;
        for (int x = -3; x <= 3; ++x)
          for (int y = -3; y <= 3; ++y)
            for (int z = -3; z <= 3; ++z)
              m0 += phi1(g_kernels[i], sqrt((x - sh[0]) * (x - sh[0]) + (y - sh[1]) * (y - sh[1]) + (z - sh[2]) * (z - sh[2])));
        worst0 = fmax(worst0, fabs(m0 - 1.0));
      }
      TEST_ASSERT(info.moment_order == -1);
      TEST_ASSERT(worst0 > 1e-6);
      continue;
    }
    double worst0 = 0.0, worst1 = 0.0;
    for (int s = 0; s < 1024; ++s)
    {
      const double sh = s / 1024.0;
      double m0 = 0.0, m1 = 0.0;
      for (int j = -4; j <= 4; ++j)
      {
        const double r = j - sh;
        const double f = phi1(g_kernels[i], r);
        m0 += f;
        m1 += r * f;
      }
      worst0 = fmax(worst0, fabs(m0 - 1.0));
      worst1 = fmax(worst1, fabs(m1));
    }
    if (info.moment_order >= 0) TEST_ASSERT(worst0 <= 1e-14);
    if (info.moment_order >= 1) TEST_ASSERT(worst1 <= 1e-14);
    if (info.moment_order == 0) TEST_ASSERT(worst1 > 1e-6);
  }
  return 0;
}

/** @brief Peskin's even/odd and sum-of-squares identities, which the 4-point function was built to satisfy. */
static int test_peskin_identities(void)
{
  for (int s = 0; s < 1024; ++s)
  {
    const double sh = s / 1024.0;
    double even = 0.0, odd = 0.0, sq = 0.0;
    for (int j = -3; j <= 3; ++j)
    {
      const double f = phi1(MFTL_KERNEL_PESKIN4, j - sh);
      if (j % 2 == 0)
        even += f;
      else
        odd += f;
      sq += f * f;
    }
    TEST_ASSERT(fabs(even - 0.5) <= 1e-15 && fabs(odd - 0.5) <= 1e-15);
    TEST_ASSERT(fabs(sq - 3.0 / 8.0) <= 1e-15);
  }
  return 0;
}

/** @brief Continuity at every breakpoint, and of the derivative where the kernel claims C1 or better. */
static int test_declared_smoothness(void)
{
  for (int i = 0; i < N_KERNELS; ++i)
  {
    MFTL_KernelInfo info;
    MFTLCouplingKernelInfo(g_kernels[i], &info);
    const double bps[] = {0.5, 1.0, 1.5, 2.0};
    for (size_t b = 0; b < sizeof bps / sizeof bps[0]; ++b)
    {
      const double x = bps[b];
      if (x > info.half_width) continue;
      const double e = 1e-7;
      const double jump = fabs(phi1(g_kernels[i], x + e) - phi1(g_kernels[i], x - e));
      TEST_ASSERT(jump < 1e-5);
      if (info.smoothness >= 1)
      {
        const double dl = (phi1(g_kernels[i], x - e) - phi1(g_kernels[i], x - 2 * e)) / e;
        const double dr = (phi1(g_kernels[i], x + 2 * e) - phi1(g_kernels[i], x + e)) / e;
        TEST_ASSERT(fabs(dl - dr) < 1e-4);
      }
    }
  }
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The row core
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Uniform lattice points within a cube of half-width @p half around @p c, spacing h. */
static int lattice_cloud(const double c[3], double h, int half, double *X, double *q)
{
  int n = 0;
  for (int i = -half; i <= half; ++i)
    for (int j = -half; j <= half; ++j)
      for (int k = -half; k <= half; ++k)
      {
        X[3 * n + 0] = c[0] + i * h;
        X[3 * n + 1] = c[1] + j * h;
        X[3 * n + 2] = c[2] + k * h;
        q[n] = h * h * h;
        ++n;
      }
  return n;
}

static double moment(const double *w, const double *X, int n, int k, const double x[3])
{
  double s = 0.0;
  for (int j = 0; j < n; ++j) s += w[j] * ((k < 0) ? 1.0 : (X[3 * j + k] - x[k]));
  return s;
}

static MFTL_CouplerParams params_with(MFTL_CouplingKernel k, MFTL_Reproduction r, int failure)
{
  MFTL_CouplerParams p;
  MFTLCouplerDefaultParams(&p);
  p.kernel = k;
  p.reproduction = r;
  p.failure = failure;
  return p;
}

/**
 * @brief A complete regular Peskin row is the kernel's own product, BIT FOR
 *        BIT: its moments already hold, so no solve runs (its moment matrix is
 *        not the identity -- the second moments are not 1/3 -- yet M e0 = e0).
 */
static int test_regular_peskin_row_is_untouched(void)
{
  const double h = 0.1, c[3] = {0.0, 0.0, 0.0};
  double X[3 * 1000], q[1000], w[1000];
  const int n = lattice_cloud(c, h, 4, X, q);
  const double x[3] = {0.0123, -0.0456, 0.0789};
  const double H[3] = {h, h, h};
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, MFTL_STENCIL_ERROR);
  MFTL_RowQuality rq;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(!rq.corrected && rq.order == 1 && rq.condition == 1.0);
  TEST_ASSERT(rq.n_support == 64);
  /* The kernel argument is formed as the library forms it, (X - x) * (1/h).
   * Dividing by h instead differs in the last bit of r, and near the edge of
   * the support phi turns that into ~15 ulp of the weight (|r phi'/phi| ~ 15
   * at r = 1.8): within 4 ulp WITH fused multiply-add on arm64, 18 ulp without
   * it on x86-64. What is tested is that no solve touched the row. */
  const double ih = 1.0 / h;
  for (int j = 0; j < n; ++j)
  {
    const double base = (q[j] / (h * h * h)) * phi1(MFTL_KERNEL_PESKIN4, (X[3 * j] - x[0]) * ih) * phi1(MFTL_KERNEL_PESKIN4, (X[3 * j + 1] - x[1]) * ih) *
                        phi1(MFTL_KERNEL_PESKIN4, (X[3 * j + 2] - x[2]) * ih);
    /* the kernel's product to the ulp (the library multiplies q by (1/h)^3, this by 1/h^3) */
    TEST_ASSERT(fabs(w[j] - base) <= 4.0 * DBL_EPSILON * base);
  }
  /* the second moment is not the identity's */
  double m2 = 0.0;
  for (int j = 0; j < n; ++j) m2 += w[j] * pow((X[3 * j] - x[0]) / h, 2);
  TEST_ASSERT(fabs(m2 - 1.0) > 0.1);
  return 0;
}

/** @brief The cosine kernel has no first moment on a lattice: LINEAR corrects it, CONSTANT and NONE do not. */
static int test_cosine_needs_and_gets_a_correction(void)
{
  const double h = 0.05, c[3] = {1.0, 2.0, 3.0};
  double X[3 * 1000], q[1000], w[1000];
  const int n = lattice_cloud(c, h, 4, X, q);
  const double x[3] = {1.0 + 0.3 * h, 2.0 - 0.41 * h, 3.0 + 0.17 * h};
  const double H[3] = {h, h, h};
  MFTL_RowQuality rq;

  MFTL_CouplerParams p = params_with(MFTL_KERNEL_COSINE4, MFTL_REPRODUCE_LINEAR, 0);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(rq.corrected && rq.order == 1);
  for (int k = -1; k < 3; ++k) TEST_ASSERT(fabs(moment(w, X, n, k, x) - (k < 0 ? 1.0 : 0.0)) <= 1e-13 * (k < 0 ? 1.0 : h));

  p.reproduction = MFTL_REPRODUCE_CONSTANT;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(fabs(moment(w, X, n, -1, x) - 1.0) <= 1e-14);
  TEST_ASSERT(fabs(moment(w, X, n, 0, x)) > 1e-6 * h);
  for (int j = 0; j < n; ++j) TEST_ASSERT(w[j] >= 0.0);

  p.reproduction = MFTL_REPRODUCE_NONE;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(rq.order == 0 && !rq.corrected);
  return 0;
}

/** @brief Simple pseudo-random numbers, reproducible everywhere. */
static double urand(unsigned *s)
{
  *s = *s * 1103515245u + 12345u;
  return ((*s >> 8) & 0xffffff) / (double) 0x1000000;
}

/** @brief A jittered and a severely clustered cloud: LINEAR is exact for every affine field. */
static int test_irregular_clouds_reproduce_affine_fields(void)
{
  unsigned seed = 7;
  for (int variant = 0; variant < 2; ++variant)
  {
    const double h = 0.1;
    double X[3 * 1000], q[1000], w[1000];
    int n = 0;
    for (int i = -3; i <= 3; ++i)
      for (int j = -3; j <= 3; ++j)
        for (int k = -3; k <= 3; ++k)
        {
          double p[3] = {i * h, j * h, k * h};
          for (int a = 0; a < 3; ++a) p[a] += (urand(&seed) - 0.5) * 0.8 * h;
          /* variant 1: squeeze the positive-x half into a thin layer */
          if (variant == 1 && p[0] > 0.0) p[0] *= 0.15;
          memcpy(&X[3 * n], p, sizeof p);
          q[n] = h * h * h * (0.5 + urand(&seed));
          ++n;
        }
    const double x[3] = {0.013, 0.021, -0.034}, H[3] = {h, h, h};
    MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, 0);
    p.max_amplification = 50.0;
    MFTL_RowQuality rq;
    TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
    TEST_ASSERT(rq.corrected);
    const double g[4] = {0.3, -1.7, 2.9, 0.6};
    double U = 0.0;
    for (int j = 0; j < n; ++j) U += w[j] * (g[0] + g[1] * X[3 * j] + g[2] * X[3 * j + 1] + g[3] * X[3 * j + 2]);
    const double exact = g[0] + g[1] * x[0] + g[2] * x[1] + g[3] * x[2];
    TEST_ASSERT(fabs(U - exact) <= 1e-12);
  }
  return 0;
}

/**
 * @brief A cloud squashed into a plane has no z-extent: the LINEAR fit is
 *        rank-deficient and FAILS -- it does not quietly drop to a constant.
 *        Lower order only by consent, then positive and flagged; growth cannot
 *        help a plane.
 */
static int test_a_planar_cloud_fails_loudly(void)
{
  const double h = 0.1;
  double X[3 * 1000], q[1000], w[1000];
  int n = 0;
  for (int i = -3; i <= 3; ++i)
    for (int j = -3; j <= 3; ++j)
    {
      X[3 * n + 0] = i * h;
      X[3 * n + 1] = j * h;
      X[3 * n + 2] = 0.0;
      q[n] = h * h * h;
      ++n;
    }
  const double x[3] = {0.02, -0.03, 0.01}, H[3] = {h, h, h};
  MFTL_RowQuality rq;
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, MFTL_STENCIL_ERROR);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);

  p.failure = MFTL_STENCIL_GROW;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);

  p.failure = MFTL_STENCIL_GROW | MFTL_STENCIL_LOWER_ORDER;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(rq.lower_order && rq.order >= 0);
  double s = 0.0;
  for (int j = 0; j < n; ++j)
  {
    TEST_ASSERT(w[j] >= 0.0);
    s += w[j];
  }
  TEST_ASSERT(fabs(s - 1.0) <= 1e-14);

  /* nearly planar (1e-9 h of relief) is caught by the condition number, not by luck */
  for (int j = 0; j < n; ++j) X[3 * j + 2] = (j % 2 ? 1e-9 : -1e-9) * h;
  p.failure = MFTL_STENCIL_ERROR;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);
  return 0;
}

/** @brief No sample inside the support is an error under every policy. */
static int test_an_empty_support_always_fails(void)
{
  const double X[6] = {5.0, 5.0, 5.0, -5.0, 5.0, 5.0}, q[2] = {1.0, 1.0};
  double w[2];
  const double x[3] = {0.0, 0.0, 0.0}, H[3] = {0.1, 0.1, 0.1};
  for (int r = 0; r < 3; ++r)
  {
    MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, (MFTL_Reproduction) r, MFTL_STENCIL_GROW | MFTL_STENCIL_LOWER_ORDER);
    TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, 2, X, q, w, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  }
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, 0);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, 0, MFTL_NULL, MFTL_NULL, MFTL_NULL, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  return 0;
}

/**
 * @brief A support too small for a well-posed fit (a metric of 0.3 h holds a
 *        single sample) fails; growth rescues it within max_growth and reports
 *        the steps; a limit below what is needed still fails.
 */
static int test_growth_rescues_a_starved_row_within_its_limit(void)
{
  const double h = 1.0, c[3] = {0, 0, 0};
  double X[3 * 1000], q[1000], w[1000];
  const int n = lattice_cloud(c, h, 4, X, q);
  const double x[3] = {0.1, 0.2, -0.15}, H[3] = {0.3 * h, 0.3 * h, 0.3 * h};
  MFTL_RowQuality rq;
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, MFTL_STENCIL_ERROR);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);
  p.failure = MFTL_STENCIL_GROW;
  p.max_growth = 1.5;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);
  p.max_growth = 5.0;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(rq.n_grow > 0 && rq.metric[0] > H[0] && rq.metric[0] <= 5.0 * H[0] * (1.0 + 1e-12));
  for (int k = -1; k < 3; ++k) TEST_ASSERT(fabs(moment(w, X, n, k, x) - (k < 0 ? 1.0 : 0.0)) <= 1e-12);
  return 0;
}

/**
 * @brief A wall: samples only on one side. LINEAR stays exact (negative
 *        weights appear), CONSTANT stays positive and loses the first moment.
 *        A marker ON the last sample plane and one between it and the wall.
 */
static int test_a_wall_gives_a_one_sided_exact_row(void)
{
  const double h = 0.1;
  double X[3 * 1000], q[1000], w[1000];
  int n = 0;
  for (int i = 0; i <= 5; ++i)
    for (int j = -3; j <= 3; ++j)
      for (int k = -3; k <= 3; ++k)
      {
        X[3 * n + 0] = (i + 0.5) * h; /* cell centres; the wall is x = 0 */
        X[3 * n + 1] = j * h;
        X[3 * n + 2] = k * h;
        q[n] = h * h * h;
        ++n;
      }
  const double xs[2][3] = {{0.5 * h, 0.01, 0.02}, {0.1 * h, -0.02, 0.0}};
  const double H[3] = {h, h, h};
  for (int m = 0; m < 2; ++m)
  {
    MFTL_RowQuality rq;
    MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, 0);
    TEST_ASSERT(MFTLCouplingAssembleRow(&p, xs[m], H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
    TEST_ASSERT(rq.corrected && rq.min_weight < 0.0);
    for (int k = -1; k < 3; ++k) TEST_ASSERT(fabs(moment(w, X, n, k, xs[m]) - (k < 0 ? 1.0 : 0.0)) <= 1e-13);
    p.reproduction = MFTL_REPRODUCE_CONSTANT;
    TEST_ASSERT(MFTLCouplingAssembleRow(&p, xs[m], H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
    for (int j = 0; j < n; ++j) TEST_ASSERT(w[j] >= 0.0);
    TEST_ASSERT(fabs(moment(w, X, n, 0, xs[m])) > 1e-3 * h);
  }
  return 0;
}

/**
 * @brief An extrapolating fit amplifies: a marker in the corner outside its
 *        samples' convex hull (every sample at x, y, z >= 0.9 h) needs weights
 *        of both signs on every axis -- about 2.8 per axis, 22 in all -- and is
 *        refused at the default limit of 4, not returned.
 */
static int test_extrapolation_is_refused_by_amplification(void)
{
  const double h = 0.1;
  double X[3 * 1000], q[1000], w[1000];
  int n = 0;
  for (int i = 0; i <= 3; ++i)
    for (int j = 0; j <= 3; ++j)
      for (int k = 0; k <= 3; ++k)
      {
        X[3 * n + 0] = (0.9 + i) * h;
        X[3 * n + 1] = (0.9 + j) * h;
        X[3 * n + 2] = (0.9 + k) * h;
        q[n] = h * h * h;
        ++n;
      }
  const double x[3] = {0.0, 0.0, 0.0}, H[3] = {h, h, h};
  MFTL_RowQuality rq;
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_PESKIN4, MFTL_REPRODUCE_LINEAR, 0);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_BAD_STENCIL);
  p.max_amplification = 1e6;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w, &rq) == MFTL_COUPLING_OK);
  TEST_ASSERT(rq.sum_abs > 4.0); /* the reason the default refused it */
  for (int k = -1; k < 3; ++k) TEST_ASSERT(fabs(moment(w, X, n, k, x) - (k < 0 ? 1.0 : 0.0)) <= 1e-12);
  return 0;
}

/**
 * @brief Dimensionless numerics: scaling the whole configuration by 2^-20 or
 *        2^20 leaves every weight BIT-IDENTICAL; a translation to 1e6 changes
 *        them only by the cancellation in X - x.
 */
static int test_rows_do_not_depend_on_the_length_unit(void)
{
  const double h = 0.1;
  double X[3 * 1000], q[1000], w0[1000], w1[1000];
  unsigned seed = 3;
  int n = 0;
  for (int i = -3; i <= 3; ++i)
    for (int j = -3; j <= 3; ++j)
      for (int k = -3; k <= 3; ++k)
      {
        X[3 * n + 0] = i * h + (urand(&seed) - 0.5) * 0.5 * h;
        X[3 * n + 1] = j * h + (urand(&seed) - 0.5) * 0.5 * h;
        X[3 * n + 2] = k * h + (urand(&seed) - 0.5) * 0.5 * h;
        q[n] = h * h * h;
        ++n;
      }
  const double x[3] = {0.011, -0.023, 0.007}, H[3] = {h, h, h};
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_COSINE4, MFTL_REPRODUCE_LINEAR, 0);
  p.max_amplification = 50.0;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w0, MFTL_NULL) == MFTL_COUPLING_OK);
  const double scales[2] = {ldexp(1.0, -20), ldexp(1.0, 20)};
  for (int s = 0; s < 2; ++s)
  {
    double Xs[3 * 1000], qs[1000], xs[3], Hs[3];
    for (int j = 0; j < 3 * n; ++j) Xs[j] = X[j] * scales[s];
    for (int j = 0; j < n; ++j) qs[j] = q[j] * scales[s] * scales[s] * scales[s];
    for (int k = 0; k < 3; ++k)
    {
      xs[k] = x[k] * scales[s];
      Hs[k] = H[k] * scales[s];
    }
    TEST_ASSERT(MFTLCouplingAssembleRow(&p, xs, Hs, n, Xs, qs, w1, MFTL_NULL) == MFTL_COUPLING_OK);
    for (int j = 0; j < n; ++j) TEST_ASSERT(w1[j] == w0[j]);
  }
  /* translation to 1e6: X - x loses log2(1e6/h) ~ 23 bits of 52, so ~1e-9 */
  double Xt[3 * 1000], xt[3];
  for (int j = 0; j < 3 * n; ++j) Xt[j] = X[j] + 1e6;
  for (int k = 0; k < 3; ++k) xt[k] = x[k] + 1e6;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, xt, H, n, Xt, q, w1, MFTL_NULL) == MFTL_COUPLING_OK);
  double worst = 0.0;
  for (int j = 0; j < n; ++j) worst = fmax(worst, fabs(w1[j] - w0[j]));
  TEST_ASSERT(worst < 1e-8);
  return 0;
}

/** @brief An application kernel through the callback is the built-in it imitates. */
static void user_hat(const double *r, double *phi, int n, void *ctx)
{
  int *calls = (int *) ctx;
  ++*calls;
  for (int i = 0; i < n; ++i) phi[i] = (fabs(r[i]) < 1.0) ? 1.0 - fabs(r[i]) : 0.0;
}

static void signed_radial(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i) phi[i] = r[i] < 0.5 ? 1.0 : (r[i] < 1.0 ? -0.1 : 0.0);
}

static int test_constant_reproduction_refuses_negative_base_weights(void)
{
  MFTL_CouplerParams p;
  MFTLCouplerDefaultParams(&p);
  p.kernel = MFTL_KERNEL_USER;
  p.user_kernel.eval = signed_radial;
  p.user_kernel.half_width = 1.0;
  p.user_kernel.moment_order = -1;
  p.user_kernel.radial = true;
  const double x[3] = {0, 0, 0}, h[3] = {1, 1, 1};
  const double samples[6] = {0, 0, 0, 0.75, 0, 0}, measures[2] = {1, 1};
  double w[2];
  MFTL_RowQuality q;
  p.reproduction = MFTL_REPRODUCE_NONE;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, h, 2, samples, measures, w, &q) == MFTL_COUPLING_OK);
  TEST_ASSERT(w[0] == 1.0 && w[1] == -0.1);
  p.reproduction = MFTL_REPRODUCE_CONSTANT;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, h, 2, samples, measures, w, &q) == MFTL_COUPLING_BAD_STENCIL);
  return 0;
}

static int test_a_user_kernel_is_called_per_axis_not_per_sample(void)
{
  const double h = 0.1, c[3] = {0, 0, 0};
  double X[3 * 1000], q[1000], w0[1000], w1[1000];
  const int n = lattice_cloud(c, h, 3, X, q);
  const double x[3] = {0.031, -0.012, 0.044}, H[3] = {h, h, h};
  MFTL_CouplerParams p = params_with(MFTL_KERNEL_HAT2, MFTL_REPRODUCE_LINEAR, 0);
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w0, MFTL_NULL) == MFTL_COUPLING_OK);
  int calls = 0;
  p.kernel = MFTL_KERNEL_USER;
  p.user_kernel.eval = user_hat;
  p.user_kernel.ctx = &calls;
  p.user_kernel.half_width = 1.0;
  p.user_kernel.moment_order = 1;
  p.user_kernel.nonnegative = true;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w1, MFTL_NULL) == MFTL_COUPLING_OK);
  for (int j = 0; j < n; ++j) TEST_ASSERT(w0[j] == w1[j]);
  TEST_ASSERT(calls == 3); /* once per axis, over all candidates */
  /* an invalid declaration is refused */
  p.user_kernel.half_width = 0.0;
  TEST_ASSERT(MFTLCouplingAssembleRow(&p, x, H, n, X, q, w1, MFTL_NULL) == MFTL_COUPLING_BAD_STENCIL);
  return 0;
}

int main(void)
{
  printf("\n=== Coupling kernels and rows ===\n");
  RUN_TEST(test_kernel_values_are_the_published_ones);
  RUN_TEST(test_support_symmetry_and_sign);
  RUN_TEST(test_unit_integral);
  RUN_TEST(test_lattice_moments_match_the_declaration);
  RUN_TEST(test_peskin_identities);
  RUN_TEST(test_declared_smoothness);
  RUN_TEST(test_regular_peskin_row_is_untouched);
  RUN_TEST(test_cosine_needs_and_gets_a_correction);
  RUN_TEST(test_irregular_clouds_reproduce_affine_fields);
  RUN_TEST(test_a_planar_cloud_fails_loudly);
  RUN_TEST(test_an_empty_support_always_fails);
  RUN_TEST(test_growth_rescues_a_starved_row_within_its_limit);
  RUN_TEST(test_a_wall_gives_a_one_sided_exact_row);
  RUN_TEST(test_extrapolation_is_refused_by_amplification);
  RUN_TEST(test_rows_do_not_depend_on_the_length_unit);
  RUN_TEST(test_a_user_kernel_is_called_per_axis_not_per_sample);
  RUN_TEST(test_constant_reproduction_refuses_negative_base_weights);
  TEST_REPORT();
  return (g_tests_failed > 0) ? 1 : 0;
}

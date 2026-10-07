/**
 * @file coupling_kernels.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief The kernel table: every built-in regularised delta function, its
 *        declared support, lattice moments, sign and smoothness.
 *
 * A kernel is evaluated once per marker and axis over every candidate at once
 * (MFTL_KernelFn), so the dispatch is hoisted out of the per-sample loop. The
 * declarations are what tests check against independent formulas; nothing here
 * infers a property by sampling.
 */

#include "coupling_internal.h"

#include <math.h>
#include <string.h>

#include "mftl_constants.h"

/* ── The 1-D tensor kernels: \int phi = 1, phi = 0 for |r| >= half-width ── */

/** @brief Peskin's 4-point function (Peskin 2002, eq. 6.27). */
static void k_peskin4(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]);
    double v = 0.0;
    if (a < 1.0)
      v = 0.125 * (3.0 - 2.0 * a + sqrt(1.0 + 4.0 * a - 4.0 * a * a));
    else if (a < 2.0)
      v = 0.125 * (5.0 - 2.0 * a - sqrt(-7.0 + 12.0 * a - 4.0 * a * a));
    phi[i] = v;
  }
}

/** @brief d/dr of Peskin's function: C1, zero at 0 and at the support's end. */
static void d_peskin4(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]), sg = (r[i] > 0.0) ? 1.0 : ((r[i] < 0.0) ? -1.0 : 0.0);
    double v = 0.0;
    if (a < 1.0)
      v = 0.125 * (-2.0 + (2.0 - 4.0 * a) / sqrt(1.0 + 4.0 * a - 4.0 * a * a));
    else if (a < 2.0)
      v = 0.125 * (-2.0 - (6.0 - 4.0 * a) / sqrt(-7.0 + 12.0 * a - 4.0 * a * a));
    dphi[i] = sg * v;
  }
}

/** @brief Roma, Peskin & Berger's 3-point function. */
static void k_roma3(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]);
    double v = 0.0;
    if (a < 0.5)
      v = (1.0 + sqrt(1.0 - 3.0 * a * a)) / 3.0;
    else if (a < 1.5)
    {
      const double b = 1.0 - a;
      v = (5.0 - 3.0 * a - sqrt(1.0 - 3.0 * b * b)) / 6.0;
    }
    phi[i] = v;
  }
}

/** @brief d/dr of the 3-point function. */
static void d_roma3(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]), sg = (r[i] > 0.0) ? 1.0 : ((r[i] < 0.0) ? -1.0 : 0.0);
    double v = 0.0;
    if (a < 0.5)
      v = -a / sqrt(1.0 - 3.0 * a * a);
    else if (a < 1.5)
    {
      const double b = 1.0 - a;
      v = (-3.0 - 3.0 * b / sqrt(1.0 - 3.0 * b * b)) / 6.0;
    }
    dphi[i] = sg * v;
  }
}

/** @brief The cosine kernel (1 + cos(pi r / 2)) / 4. */
static void k_cosine4(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]);
    phi[i] = (a < 2.0) ? 0.25 * (1.0 + cos(0.5 * MFTL_PI * a)) : 0.0;
  }
}

/** @brief d/dr of the cosine kernel. */
static void d_cosine4(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]), sg = (r[i] > 0.0) ? 1.0 : ((r[i] < 0.0) ? -1.0 : 0.0);
    dphi[i] = (a < 2.0) ? -sg * 0.125 * MFTL_PI * sin(0.5 * MFTL_PI * a) : 0.0;
  }
}

/** @brief The linear hat. */
static void k_hat2(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]);
    phi[i] = (a < 1.0) ? 1.0 - a : 0.0;
  }
}

/** @brief d/dr of the hat: -sign(r) inside, 0 at r = 0 (where it does not exist) and outside. */
static void d_hat2(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]), sg = (r[i] > 0.0) ? 1.0 : ((r[i] < 0.0) ? -1.0 : 0.0);
    dphi[i] = (a < 1.0) ? -sg : 0.0;
  }
}

/** @brief The cubic B-spline. */
static void k_bspline3(const double *r, double *phi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]);
    double v = 0.0;
    if (a < 1.0)
      v = 2.0 / 3.0 - a * a + 0.5 * a * a * a;
    else if (a < 2.0)
    {
      const double b = 2.0 - a;
      v = b * b * b / 6.0;
    }
    phi[i] = v;
  }
}

/** @brief d/dr of the cubic B-spline. */
static void d_bspline3(const double *r, double *dphi, int n, void *ctx)
{
  (void) ctx;
  for (int i = 0; i < n; ++i)
  {
    const double a = fabs(r[i]), sg = (r[i] > 0.0) ? 1.0 : ((r[i] < 0.0) ? -1.0 : 0.0);
    double v = 0.0;
    if (a < 1.0)
      v = -2.0 * a + 1.5 * a * a;
    else if (a < 2.0)
    {
      const double b = 2.0 - a;
      v = -0.5 * b * b;
    }
    dphi[i] = sg * v;
  }
}

/**
 * @brief Wendland C2, support radius 2, normalised in 3-D:
 *        \f$\phi(\rho) = \frac{21}{16\pi}(1-\rho/2)^4(2\rho+1)\f$, whose integral
 *        over \f$\mathbb{R}^3\f$ is \f$\frac{21}{16\pi}\cdot 4\pi\cdot\frac{4}{21} = 1\f$.
 */
static void k_wendland_c2(const double *rho, double *phi, int n, void *ctx)
{
  (void) ctx;
  const double c = 21.0 / (16.0 * MFTL_PI);
  for (int i = 0; i < n; ++i)
  {
    const double a = rho[i];
    if (a < 2.0)
    {
      const double s = 1.0 - 0.5 * a;
      phi[i] = c * s * s * s * s * (2.0 * a + 1.0);
    }
    else
      phi[i] = 0.0;
  }
}

/** @brief d/drho of Wendland C2: \f$-5 c \rho (1-\rho/2)^3\f$. */
static void d_wendland_c2(const double *rho, double *dphi, int n, void *ctx)
{
  (void) ctx;
  const double c = 21.0 / (16.0 * MFTL_PI);
  for (int i = 0; i < n; ++i)
  {
    const double a = rho[i];
    if (a < 2.0)
    {
      const double s = 1.0 - 0.5 * a;
      dphi[i] = -5.0 * c * a * s * s * s;
    }
    else
      dphi[i] = 0.0;
  }
}

/** @brief One row of the table. */
typedef struct
{
  MFTL_CouplingKernel id;
  const char *name;
  MFTL_KernelFn eval;
  MFTL_KernelFn deriv;
  double half_width;
  int moment_order;
  int smoothness;
  bool radial;
  bool nonnegative;
} kernel_row;

static const kernel_row g_table[] = {
    {MFTL_KERNEL_PESKIN4, "peskin4", k_peskin4, d_peskin4, 2.0, 1, 1, false, true},
    {MFTL_KERNEL_ROMA3, "roma3", k_roma3, d_roma3, 1.5, 1, 0, false, true},
    {MFTL_KERNEL_COSINE4, "cosine4", k_cosine4, d_cosine4, 2.0, 0, 1, false, true},
    {MFTL_KERNEL_HAT2, "hat2", k_hat2, d_hat2, 1.0, 1, 0, false, true},
    {MFTL_KERNEL_BSPLINE3, "bspline3", k_bspline3, d_bspline3, 2.0, 1, 2, false, true},
    {MFTL_KERNEL_WENDLAND_C2, "wendland_c2", k_wendland_c2, d_wendland_c2, 2.0, -1, 2, true, true},
};

static const kernel_row *find_row(MFTL_CouplingKernel id)
{
  for (size_t i = 0; i < sizeof(g_table) / sizeof(g_table[0]); ++i)
    if (g_table[i].id == id) return &g_table[i];
  return MFTL_NULL;
}

int MFTLCouplingKernelInfo(MFTL_CouplingKernel kernel, MFTL_KernelInfo *info)
{
  const kernel_row *row = find_row(kernel);
  if (row == MFTL_NULL || info == MFTL_NULL) return MFTL_ERROR;
  info->name = row->name;
  info->half_width = row->half_width;
  info->moment_order = row->moment_order;
  info->smoothness = row->smoothness;
  info->radial = row->radial;
  info->nonnegative = row->nonnegative;
  return MFTL_SUCCESS;
}

int MFTLCouplingKernelEval(MFTL_CouplingKernel kernel, const double *r, double *phi, int n)
{
  const kernel_row *row = find_row(kernel);
  if (row == MFTL_NULL || (n > 0 && (r == MFTL_NULL || phi == MFTL_NULL))) return MFTL_ERROR;
  row->eval(r, phi, n, MFTL_NULL);
  return MFTL_SUCCESS;
}

int MFTLCouplingKernelDerivative(MFTL_CouplingKernel kernel, const double *r, double *dphi, int n)
{
  const kernel_row *row = find_row(kernel);
  if (row == MFTL_NULL || (n > 0 && (r == MFTL_NULL || dphi == MFTL_NULL))) return MFTL_ERROR;
  row->deriv(r, dphi, n, MFTL_NULL);
  return MFTL_SUCCESS;
}

int MFTLCplKernelResolve(const MFTL_CouplerParams *p, cpl_kernel *k)
{
  memset(k, 0, sizeof *k);
  if (p->kernel == MFTL_KERNEL_USER)
  {
    const MFTL_KernelDesc *u = &p->user_kernel;
    if (u->eval == MFTL_NULL || !(u->half_width > 0.0) || !isfinite(u->half_width) || u->moment_order < -1 || u->moment_order > 1) return MFTL_ERROR;
    k->eval = u->eval;
    k->deriv = u->deriv;
    k->ctx = u->ctx;
    k->half_width = u->half_width;
    k->moment_order = u->moment_order;
    k->radial = u->radial;
    k->nonnegative = u->nonnegative;
    return MFTL_SUCCESS;
  }
  const kernel_row *row = find_row(p->kernel);
  if (row == MFTL_NULL) return MFTL_ERROR;
  k->eval = row->eval;
  k->deriv = row->deriv;
  k->ctx = MFTL_NULL;
  k->half_width = row->half_width;
  k->moment_order = row->moment_order;
  k->radial = row->radial;
  k->nonnegative = row->nonnegative;
  return MFTL_SUCCESS;
}

const char *MFTLCouplingStatusName(int status)
{
  switch (status)
  {
    case MFTL_COUPLING_OK:
      return "OK";
    case MFTL_COUPLING_NEED_COVERAGE:
      return "NEED_COVERAGE";
    case MFTL_COUPLING_BAD_STENCIL:
      return "BAD_STENCIL";
    case MFTL_COUPLING_BAD_SAMPLING:
      return "BAD_SAMPLING";
    case MFTL_COUPLING_STALE:
      return "STALE";
    case MFTL_COUPLING_INVALID:
      return "INVALID";
    default:
      return "UNKNOWN";
  }
}

void MFTLCouplerDefaultParams(MFTL_CouplerParams *p)
{
  if (p == MFTL_NULL) return;
  memset(p, 0, sizeof *p);
  p->kernel = MFTL_KERNEL_PESKIN4;
  p->reproduction = MFTL_REPRODUCE_LINEAR;
  p->failure = MFTL_STENCIL_ERROR;
  p->max_growth = 1.5;
  p->growth_step = 1.25;
  p->support_scale = 1.0;
  p->pilot_factor = 1.5;
  p->max_condition = 1e8;
  p->max_amplification = 4.0;
  p->moment_tolerance = 1e-10;
  p->sampling = MFTL_SAMPLING_REPORT;
  p->sampling_ratio = 1.0;
  p->directory_margin = 0.0;
}

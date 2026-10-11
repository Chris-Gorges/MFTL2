/**
 * @file coupling_rows.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief One interpolation row from its candidate samples: base weights, the
 *        reproduction policy, and the four separate quality results.
 *
 * Every row is judged on four things, and each is checked on its own because
 * none implies another: the support is non-empty (the caller has already
 * certified it COMPLETE -- a perfectly conditioned row over an incomplete
 * support is still wrong); the moment matrix has full numerical rank and an
 * admissible condition number; the requested moments are reached; and the
 * weights do not amplify (\f$\sum|W|\f$). A successful pivot is not a
 * certificate of any of them.
 *
 * The moment matrix is 4x4, symmetric and positive semi-definite. It is solved
 * through the eigen-decomposition of its EQUILIBRATED form (cyclic Jacobi):
 * rank-revealing, the condition number exact, and at this size cheaper than the
 * accumulation that built the matrix. A row whose base kernel already has the
 * requested moments -- Peskin over a complete uniform lattice -- is left exactly
 * as the kernel made it: its moment matrix reproduces constants and
 * coordinates, \f$M e_0 = e_0\f$, but is NOT the identity, so no solve is
 * needed and none is done.
 */

#include "coupling_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"

void MFTLCplRowScratchFree(cpl_row_scratch *s)
{
  MFTLCplFree(s->d);
  MFTLCplFree(s->phi);
  MFTLCplFree(s->psi);
  MFTLCplFree(s->r);
  memset(s, 0, sizeof *s);
}

static void scratch_reserve(cpl_row_scratch *s, int n)
{
  if (n <= s->cap) return;
  const int cap = MFTLGrowCapacity(s->cap, n);
  s->d = (double *) MFTLCplRealloc(s->d, 3 * (size_t) cap * sizeof(double), "coupling row scratch");
  s->phi = (double *) MFTLCplRealloc(s->phi, 3 * (size_t) cap * sizeof(double), "coupling row scratch");
  s->psi = (double *) MFTLCplRealloc(s->psi, (size_t) cap * sizeof(double), "coupling row scratch");
  s->r = (double *) MFTLCplRealloc(s->r, (size_t) cap * sizeof(double), "coupling row scratch");
  s->cap = cap;
}

/**
 * @brief Cyclic Jacobi eigen-decomposition of a symmetric 4x4 matrix.
 *
 * @param[in,out] a The matrix, destroyed; its diagonal holds the eigenvalues.
 * @param[out]    v Eigenvectors in the columns.
 */
static void jacobi4(double a[4][4], double v[4][4])
{
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) v[i][j] = (i == j) ? 1.0 : 0.0;

  for (int sweep = 0; sweep < 64; ++sweep)
  {
    double off = 0.0, diag = 0.0;
    for (int i = 0; i < 4; ++i)
    {
      diag += a[i][i] * a[i][i];
      for (int j = i + 1; j < 4; ++j) off += a[i][j] * a[i][j];
    }
    if (off <= 1e-34 * diag || off == 0.0) break;

    for (int p = 0; p < 3; ++p)
      for (int q = p + 1; q < 4; ++q)
      {
        if (a[p][q] == 0.0) continue;
        const double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        const double t = ((theta >= 0.0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < 4; ++k)
        {
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k)
        {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k)
        {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
  }
}

/** @brief The achieved zeroth and first moment residuals of weights @p w. */
static void moments_of(const double *w, const double *d, int n, double *r0, double *r1, double *sum_abs, double *dmax)
{
  double m0 = 0.0, m1[3] = {0.0, 0.0, 0.0}, sa = 0.0, dm = 1.0;
  for (int j = 0; j < n; ++j)
  {
    if (w[j] == 0.0) continue;
    m0 += w[j];
    sa += fabs(w[j]);
    for (int k = 0; k < 3; ++k)
    {
      m1[k] += w[j] * d[3 * j + k];
      if (fabs(d[3 * j + k]) > dm) dm = fabs(d[3 * j + k]);
    }
  }
  *r0 = fabs(m0 - 1.0);
  *r1 = fmax(fabs(m1[0]), fmax(fabs(m1[1]), fabs(m1[2])));
  *sum_abs = sa;
  *dmax = dm;
}

/**
 * @brief The row from dimensionless offsets `s->d` already filled for @p n
 *        candidates: base weights, the reproduction policy, the quality.
 *        Shared by the diagonal and the full-matrix metric.
 */
static int row_from_offsets(const MFTL_CouplerParams *p, const cpl_kernel *kern, double inv_det, int n, const double *q, double *w, MFTL_RowQuality *qo,
                            cpl_row_scratch *s);

int MFTLCplRowWeights(const MFTL_CouplerParams *p, const cpl_kernel *kern, const double x[3], const double Ht[3], int n, const double *X, const double *q,
                    double *w, MFTL_RowQuality *qo, cpl_row_scratch *s)
{
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  qo->order = -1;
  qo->condition = 1.0;
  for (int k = 0; k < 3; ++k) qo->metric[k] = Ht[k];
  if (n <= 0) return MFTL_COUPLING_BAD_STENCIL;
  if (!(Ht[0] > 0.0 && Ht[1] > 0.0 && Ht[2] > 0.0)) return MFTL_COUPLING_BAD_STENCIL;

  scratch_reserve(s, n);
  double *const d = s->d;
  const double inv[3] = {1.0 / Ht[0], 1.0 / Ht[1], 1.0 / Ht[2]};
  const double inv_det = inv[0] * inv[1] * inv[2];

  for (int j = 0; j < n; ++j)
    for (int k = 0; k < 3; ++k) d[3 * j + k] = (X[3 * j + k] - x[k]) * inv[k];
  return row_from_offsets(p, kern, inv_det, n, q, w, qo, s);
}

bool MFTLCplCellInverse(const double Hm[9], double inv[9], double *det)
{
  const double a = Hm[0], b = Hm[1], c = Hm[2], d = Hm[3], e = Hm[4], f = Hm[5], g = Hm[6], h = Hm[7], i = Hm[8];
  const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const double D = a * A + b * B + c * C;
  *det = D;
  /* scale-relative singularity: |det| against the product of the column norms */
  const double n0 = sqrt(a * a + d * d + g * g), n1 = sqrt(b * b + e * e + h * h), n2 = sqrt(c * c + f * f + i * i);
  if (!isfinite(D) || !(fabs(D) > 1e-12 * n0 * n1 * n2)) return false;
  const double id = 1.0 / D;
  inv[0] = A * id;
  inv[1] = -(b * i - c * h) * id;
  inv[2] = (b * f - c * e) * id;
  inv[3] = B * id;
  inv[4] = (a * i - c * g) * id;
  inv[5] = -(a * f - c * d) * id;
  inv[6] = C * id;
  inv[7] = -(a * h - b * g) * id;
  inv[8] = (a * e - b * d) * id;
  return true;
}

int MFTLCplRowWeightsCell(const MFTL_CouplerParams *p, const cpl_kernel *kern, const double x[3], const double Hm[9], int n, const double *X, const double *q,
                        double *w, MFTL_RowQuality *qo, cpl_row_scratch *s)
{
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  qo->order = -1;
  qo->condition = 1.0;
  for (int k = 0; k < 3; ++k) qo->metric[k] = fabs(Hm[3 * k]) + fabs(Hm[3 * k + 1]) + fabs(Hm[3 * k + 2]);
  double inv[9], det = 0.0;
  if (n <= 0 || !MFTLCplCellInverse(Hm, inv, &det)) return MFTL_COUPLING_BAD_STENCIL;
  scratch_reserve(s, n);
  double *const d = s->d;
  for (int j = 0; j < n; ++j)
  {
    const double r0 = X[3 * j] - x[0], r1 = X[3 * j + 1] - x[1], r2 = X[3 * j + 2] - x[2];
    for (int k = 0; k < 3; ++k) d[3 * j + k] = inv[3 * k] * r0 + inv[3 * k + 1] * r1 + inv[3 * k + 2] * r2;
  }
  return row_from_offsets(p, kern, 1.0 / fabs(det), n, q, w, qo, s);
}

static int row_from_offsets(const MFTL_CouplerParams *p, const cpl_kernel *kern, double inv_det, int n, const double *q, double *w, MFTL_RowQuality *qo,
                            cpl_row_scratch *s)
{
  double *const d = s->d, *const phi = s->phi, *const psi = s->psi, *const r = s->r;

  /* ── base weights psi_j = q_j / det(H) * phi(d_j): one kernel call per axis ── */
  if (kern->radial)
  {
    for (int j = 0; j < n; ++j) r[j] = sqrt(d[3 * j] * d[3 * j] + d[3 * j + 1] * d[3 * j + 1] + d[3 * j + 2] * d[3 * j + 2]);
    kern->eval(r, phi, n, kern->ctx);
    for (int j = 0; j < n; ++j) psi[j] = q[j] * inv_det * phi[j];
  }
  else
  {
    for (int j = 0; j < n; ++j) psi[j] = q[j] * inv_det;
    for (int k = 0; k < 3; ++k)
    {
      for (int j = 0; j < n; ++j) r[j] = d[3 * j + k];
      kern->eval(r, phi, n, kern->ctx);
      for (int j = 0; j < n; ++j) psi[j] *= phi[j];
    }
  }

  int n_support = 0;
  double m0 = 0.0, m1[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < n; ++j)
  {
    /* CONSTANT promises positivity, including for application kernels. A
     * signed kernel cannot be made positive merely by normalising its sum. */
    if (p->reproduction == MFTL_REPRODUCE_CONSTANT && psi[j] < 0.0) return MFTL_COUPLING_BAD_STENCIL;
    if (psi[j] == 0.0) continue;
    ++n_support;
    m0 += psi[j];
    for (int k = 0; k < 3; ++k) m1[k] += psi[j] * d[3 * j + k];
  }
  qo->n_support = n_support;
  if (n_support == 0) return MFTL_COUPLING_BAD_STENCIL;

  const double tol = p->moment_tolerance;
  const MFTL_Reproduction repro = p->reproduction;

  if (repro == MFTL_REPRODUCE_NONE)
  {
    for (int j = 0; j < n; ++j) w[j] = psi[j];
  }
  else if (repro == MFTL_REPRODUCE_CONSTANT)
  {
    if (!(m0 > 0.0)) return MFTL_COUPLING_BAD_STENCIL;
    const double inv_m0 = 1.0 / m0;
    for (int j = 0; j < n; ++j) w[j] = psi[j] * inv_m0;
    qo->corrected = fabs(m0 - 1.0) > tol;
  }
  else
  {
    const double m1max = fmax(fabs(m1[0]), fmax(fabs(m1[1]), fabs(m1[2])));
    if (fabs(m0 - 1.0) <= tol && m1max <= tol)
    {
      /* Already linear-exact: the kernel's own weights, untouched. */
      for (int j = 0; j < n; ++j) w[j] = psi[j];
    }
    else
    {
      double M[4][4];
      memset(M, 0, sizeof M);
      for (int j = 0; j < n; ++j)
      {
        if (psi[j] == 0.0) continue;
        const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
        for (int a = 0; a < 4; ++a)
          for (int b = a; b < 4; ++b) M[a][b] += psi[j] * pj[a] * pj[b];
      }
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < a; ++b) M[a][b] = M[b][a];

      /* Equilibrate: S M S with S = diag(M)^(-1/2). A zero diagonal entry is a
       * support with no extent along that axis: rank-deficient. */
      double S[4];
      for (int a = 0; a < 4; ++a)
      {
        if (!(M[a][a] > 0.0)) return MFTL_COUPLING_BAD_STENCIL;
        S[a] = 1.0 / sqrt(M[a][a]);
      }
      double A[4][4], V[4][4];
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) A[a][b] = S[a] * M[a][b] * S[b];
      jacobi4(A, V);

      double lmin = A[0][0], lmax = A[0][0];
      for (int a = 1; a < 4; ++a)
      {
        if (A[a][a] < lmin) lmin = A[a][a];
        if (A[a][a] > lmax) lmax = A[a][a];
      }
      if (!(lmin > 0.0) || !(lmax > 0.0)) return MFTL_COUPLING_BAD_STENCIL;
      const double cond = lmax / lmin;
      qo->condition = cond;
      if (!(cond <= p->max_condition)) return MFTL_COUPLING_BAD_STENCIL;

      /* c = S V L^-1 V^T S e0. */
      double y[4], c[4];
      for (int a = 0; a < 4; ++a) y[a] = V[0][a] * S[0] / A[a][a];
      for (int a = 0; a < 4; ++a)
      {
        double acc = 0.0;
        for (int b = 0; b < 4; ++b) acc += V[a][b] * y[b];
        c[a] = S[a] * acc;
      }
      for (int j = 0; j < n; ++j) w[j] = (psi[j] == 0.0) ? 0.0 : psi[j] * (c[0] + c[1] * d[3 * j] + c[2] * d[3 * j + 1] + c[3] * d[3 * j + 2]);
      qo->corrected = true;
    }
  }

  /* ── the achieved result, measured on the weights actually returned ── */
  double r0 = 0.0, r1 = 0.0, sum_abs = 0.0, dmax = 1.0;
  moments_of(w, d, n, &r0, &r1, &sum_abs, &dmax);
  qo->sum_abs = sum_abs;
  qo->min_weight = DBL_MAX;
  qo->max_weight = -DBL_MAX;
  for (int j = 0; j < n; ++j)
  {
    if (psi[j] == 0.0) continue;
    if (w[j] < qo->min_weight) qo->min_weight = w[j];
    if (w[j] > qo->max_weight) qo->max_weight = w[j];
  }

  /* A conditioning-aware bound: what a backward-stable solve of this matrix can
   * reach, or the tolerance, whichever is looser. */
  const double bound = fmax(tol, 1e3 * DBL_EPSILON * qo->condition) * fmax(sum_abs, 1.0) * dmax;
  const bool zeroth = r0 <= bound;
  const bool first = r1 <= bound;
  qo->order = (zeroth && first) ? 1 : (zeroth ? 0 : -1);
  qo->moment_residual = (repro == MFTL_REPRODUCE_LINEAR) ? fmax(r0, r1) : (repro == MFTL_REPRODUCE_CONSTANT ? r0 : 0.0);

  if (repro == MFTL_REPRODUCE_LINEAR && qo->order < 1) return MFTL_COUPLING_BAD_STENCIL;
  if (repro == MFTL_REPRODUCE_CONSTANT && qo->order < 0) return MFTL_COUPLING_BAD_STENCIL;
  if (repro != MFTL_REPRODUCE_NONE && !(sum_abs <= p->max_amplification)) return MFTL_COUPLING_BAD_STENCIL;
  if (!isfinite(sum_abs)) return MFTL_COUPLING_BAD_STENCIL;

  qo->status = MFTL_COUPLING_OK;
  return MFTL_COUPLING_OK;
}

int MFTLCouplingAssembleRow(const MFTL_CouplerParams *params, const double x[3], const double H[3], int n, const double *X, const double *q, double *w,
                            MFTL_RowQuality *quality)
{
  MFTL_RowQuality local;
  MFTL_RowQuality *const qo = quality ? quality : &local;
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  if (params == MFTL_NULL || x == MFTL_NULL || H == MFTL_NULL || (n > 0 && (X == MFTL_NULL || q == MFTL_NULL || w == MFTL_NULL)))
    return MFTL_COUPLING_BAD_STENCIL;

  cpl_kernel kern;
  if (MFTLCplKernelResolve(params, &kern) != MFTL_SUCCESS) return MFTL_COUPLING_BAD_STENCIL;

  cpl_row_scratch s;
  memset(&s, 0, sizeof s);

  MFTL_CouplerParams p = *params;
  const bool grow = (params->failure & MFTL_STENCIL_GROW) != 0 && params->max_growth > 1.0 && params->growth_step > 1.0;
  const bool lower = (params->failure & MFTL_STENCIL_LOWER_ORDER) != 0 && params->reproduction == MFTL_REPRODUCE_LINEAR;

  int rc = MFTL_COUPLING_BAD_STENCIL;
  for (int pass = 0; pass < (lower ? 2 : 1) && rc != MFTL_COUPLING_OK; ++pass)
  {
    if (pass == 1) p.reproduction = MFTL_REPRODUCE_CONSTANT;
    double g = 1.0;
    for (int step = 0;; ++step)
    {
      const double Ht[3] = {params->support_scale * g * H[0], params->support_scale * g * H[1], params->support_scale * g * H[2]};
      rc = MFTLCplRowWeights(&p, &kern, x, Ht, n, X, q, w, qo, &s);
      qo->n_grow = step;
      if (rc == MFTL_COUPLING_OK || !grow || g >= params->max_growth) break;
      g = fmin(g * params->growth_step, params->max_growth);
    }
    if (rc == MFTL_COUPLING_OK && pass == 1) qo->lower_order = true;
  }
  MFTLCplRowScratchFree(&s);
  return rc;
}

int MFTLCouplingAssembleRowCell(const MFTL_CouplerParams *params, const double x[3], const double Hc[9], int n, const double *X, const double *q, double *w,
                                MFTL_RowQuality *quality)
{
  MFTL_RowQuality local;
  MFTL_RowQuality *const qo = quality ? quality : &local;
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  if (params == MFTL_NULL || x == MFTL_NULL || Hc == MFTL_NULL || (n > 0 && (X == MFTL_NULL || q == MFTL_NULL || w == MFTL_NULL)))
    return MFTL_COUPLING_BAD_STENCIL;
  cpl_kernel kern;
  if (MFTLCplKernelResolve(params, &kern) != MFTL_SUCCESS) return MFTL_COUPLING_BAD_STENCIL;
  cpl_row_scratch s;
  memset(&s, 0, sizeof s);
  MFTL_CouplerParams p = *params;
  const bool grow = (params->failure & MFTL_STENCIL_GROW) != 0 && params->max_growth > 1.0 && params->growth_step > 1.0;
  const bool lower = (params->failure & MFTL_STENCIL_LOWER_ORDER) != 0 && params->reproduction == MFTL_REPRODUCE_LINEAR;
  int rc = MFTL_COUPLING_BAD_STENCIL;
  for (int pass = 0; pass < (lower ? 2 : 1) && rc != MFTL_COUPLING_OK; ++pass)
  {
    if (pass == 1) p.reproduction = MFTL_REPRODUCE_CONSTANT;
    double g = 1.0;
    for (int step = 0;; ++step)
    {
      double Ht[9];
      for (int a = 0; a < 9; ++a) Ht[a] = params->support_scale * g * Hc[a];
      rc = MFTLCplRowWeightsCell(&p, &kern, x, Ht, n, X, q, w, qo, &s);
      qo->n_grow = step;
      if (rc == MFTL_COUPLING_OK || !grow || g >= params->max_growth) break;
      g = fmin(g * params->growth_step, params->max_growth);
    }
    if (rc == MFTL_COUPLING_OK && pass == 1) qo->lower_order = true;
  }
  MFTLCplRowScratchFree(&s);
  return rc;
}

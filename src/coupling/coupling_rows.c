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
  free(s->d);
  free(s->phi);
  free(s->psi);
  free(s->r);
  free(s->aux);
  memset(s, 0, sizeof *s);
}

static void scratch_reserve(cpl_row_scratch *s, int n)
{
  if (n <= s->cap) return;
  const int cap = MFTLGrowCapacity(s->cap, n);
  s->d = (double *) MFTLReallocOrDie(s->d, 3 * (size_t) cap * sizeof(double), "coupling row scratch");
  s->phi = (double *) MFTLReallocOrDie(s->phi, 3 * (size_t) cap * sizeof(double), "coupling row scratch");
  s->psi = (double *) MFTLReallocOrDie(s->psi, (size_t) cap * sizeof(double), "coupling row scratch");
  s->r = (double *) MFTLReallocOrDie(s->r, (size_t) cap * sizeof(double), "coupling row scratch");
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Block rows: directional reconstruction and the two gradients
 *
 * Every row here is a weighted least-squares fit over the same base weights
 * \f$\psi_j = q_j |\det A|\, \Phi(d_j)\f$, \f$d_j = A (X_j - x)\f$, \f$A = \tilde H^{-1}\f$,
 * and every one is judged on the same four results as a value row: the
 * support, the rank and condition of its (equilibrated) normal matrix, the
 * moments it was built to reproduce -- measured on the weights returned --
 * and its amplification. Normal matrices are inverted through their
 * equilibrated eigen-decomposition (cyclic Jacobi, up to 12 x 12).
 * ═══════════════════════════════════════════════════════════════════════════ */

static double *aux_reserve(cpl_row_scratch *s, int n)
{
  const int need = CPL_ROW_AUX * n;
  if (need > s->aux_cap)
  {
    const int cap = MFTLGrowCapacity(s->aux_cap, need);
    s->aux = (double *) MFTLReallocOrDie(s->aux, (size_t) cap * sizeof(double), "coupling row scratch");
    s->aux_cap = cap;
  }
  return s->aux;
}

/** @brief Cyclic Jacobi of a symmetric n x n matrix (n <= 12, row-major): @p a's diagonal becomes the eigenvalues, @p v's columns the eigenvectors. */
static void jacobi_n(int n, double *a, double *v)
{
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) v[i * n + j] = (i == j) ? 1.0 : 0.0;
  for (int sweep = 0; sweep < 100; ++sweep)
  {
    double off = 0.0, diag = 0.0;
    for (int i = 0; i < n; ++i)
    {
      diag += a[i * n + i] * a[i * n + i];
      for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
    }
    if (off <= 1e-34 * diag || off == 0.0) break;
    for (int p = 0; p < n - 1; ++p)
      for (int q = p + 1; q < n; ++q)
      {
        const double apq = a[p * n + q];
        if (apq == 0.0) continue;
        const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
        const double t = ((theta >= 0.0) ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0), sn = t * c;
        for (int k = 0; k < n; ++k)
        {
          const double akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - sn * akq;
          a[k * n + q] = sn * akp + c * akq;
        }
        for (int k = 0; k < n; ++k)
        {
          const double apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - sn * aqk;
          a[q * n + k] = sn * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k)
        {
          const double vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - sn * vkq;
          v[k * n + q] = sn * vkp + c * vkq;
        }
      }
  }
}

/**
 * @brief Inverse of a symmetric matrix that must be positive definite, through
 *        the eigen-decomposition of its equilibrated form \f$S M S\f$,
 *        \f$S = \mathrm{diag}(M)^{-1/2}\f$. False when a diagonal entry or an
 *        eigenvalue is not positive (rank deficiency, an indefinite matrix from
 *        a signed kernel) or the condition number exceeds @p max_cond.
 */
static bool spd_inverse(int n, const double *M, double max_cond, double *inv, double *cond)
{
  double S[12], A[144], V[144];
  *cond = 1.0;
  for (int a = 0; a < n; ++a)
  {
    if (!(M[a * n + a] > 0.0)) return false;
    S[a] = 1.0 / sqrt(M[a * n + a]);
  }
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) A[a * n + b] = S[a] * M[a * n + b] * S[b];
  jacobi_n(n, A, V);
  double lmin = A[0], lmax = A[0];
  for (int a = 1; a < n; ++a)
  {
    lmin = fmin(lmin, A[a * n + a]);
    lmax = fmax(lmax, A[a * n + a]);
  }
  if (!(lmin > 0.0) || !(lmax > 0.0)) return false;
  *cond = lmax / lmin;
  if (!(*cond <= max_cond)) return false;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
    {
      double acc = 0.0;
      for (int k = 0; k < n; ++k) acc += V[a * n + k] * V[b * n + k] / A[k * n + k];
      inv[a * n + b] = S[a] * acc * S[b];
    }
  return true;
}

bool MFTLCplMetricSet(cpl_metric *m, bool cell, const double *H, const double *dH)
{
  memset(m, 0, sizeof *m);
  if (!cell)
  {
    if (!(H[0] > 0.0 && H[1] > 0.0 && H[2] > 0.0)) return false;
    const double inv[3] = {1.0 / H[0], 1.0 / H[1], 1.0 / H[2]};
    for (int k = 0; k < 3; ++k)
    {
      m->H[4 * k] = H[k];
      m->A[4 * k] = inv[k];
    }
    m->detA = inv[0] * inv[1] * inv[2];
    if (dH)
    {
      m->varying = true;
      for (int mm = 0; mm < 3; ++mm)
        for (int k = 0; k < 3; ++k)
        {
          m->dA[9 * mm + 4 * k] = -dH[3 * mm + k] * inv[k] * inv[k];
          m->dlogdet[mm] -= dH[3 * mm + k] * inv[k];
        }
    }
    return true;
  }
  double det = 0.0;
  memcpy(m->H, H, 9 * sizeof(double));
  if (!MFTLCplCellInverse(H, m->A, &det)) return false;
  m->detA = 1.0 / fabs(det);
  if (dH)
  {
    m->varying = true;
    for (int mm = 0; mm < 3; ++mm)
    {
      const double *D = &dH[9 * mm];
      double AD[9];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) AD[3 * r + c] = m->A[3 * r] * D[c] + m->A[3 * r + 1] * D[3 + c] + m->A[3 * r + 2] * D[6 + c];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m->dA[9 * mm + 3 * r + c] = -(AD[3 * r] * m->A[c] + AD[3 * r + 1] * m->A[3 + c] + AD[3 * r + 2] * m->A[6 + c]);
      m->dlogdet[mm] = -(AD[0] + AD[4] + AD[8]);
    }
  }
  return true;
}

/** @brief The quality bound every block row is judged by, as for a value row. */
static double block_bound(const MFTL_CouplerParams *p, double cond, double sum_abs, double dmax)
{
  return fmax(p->moment_tolerance, 1e3 * DBL_EPSILON * cond) * fmax(sum_abs, 1.0) * dmax;
}

/**
 * @brief Gradient-row quality in metric units: \f$\tilde D_j\f$ (`D`, 3 per
 *        candidate) must satisfy \f$\sum_j \tilde D_j = 0\f$ and
 *        \f$\sum_j \tilde D_j d_j^T = I\f$.
 */
static void gradient_moments(const double *D, const double *d, const double *psi, int n, double *sum_abs, double *r0, double *r1, double *dmax)
{
  double s0[3] = {0, 0, 0}, s1[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, sa[3] = {0, 0, 0}, dm = 1.0;
  for (int j = 0; j < n; ++j)
  {
    /* outside the support, unless a C0 kernel's derivative is nonzero exactly on its edge */
    if (psi[j] == 0.0 && D[3 * j] == 0.0 && D[3 * j + 1] == 0.0 && D[3 * j + 2] == 0.0) continue;
    for (int k = 0; k < 3; ++k)
    {
      s0[k] += D[3 * j + k];
      sa[k] += fabs(D[3 * j + k]);
      dm = fmax(dm, fabs(d[3 * j + k]));
      for (int l = 0; l < 3; ++l) s1[3 * k + l] += D[3 * j + k] * d[3 * j + l];
    }
  }
  *sum_abs = fmax(sa[0], fmax(sa[1], sa[2]));
  *r0 = fmax(fabs(s0[0]), fmax(fabs(s0[1]), fabs(s0[2])));
  double e = 0.0;
  for (int k = 0; k < 3; ++k)
    for (int l = 0; l < 3; ++l) e = fmax(e, fabs(s1[3 * k + l] - ((k == l) ? 1.0 : 0.0)));
  *r1 = e;
  *dmax = dm;
}

int MFTLCplBlockRow(const MFTL_CouplerParams *p, const cpl_kernel *kern, MFTL_CouplingOperator op, const double x[3], const cpl_metric *met, int n,
                    const double *X, const double *q, const double *dir, double *w, MFTL_RowQuality *qo, cpl_row_scratch *s)
{
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  qo->order = -1;
  qo->condition = 1.0;
  for (int k = 0; k < 3; ++k) qo->metric[k] = fabs(met->H[3 * k]) + fabs(met->H[3 * k + 1]) + fabs(met->H[3 * k + 2]);
  const MFTL_Reproduction repro = p->reproduction;
  if (dir != MFTL_NULL && (op != MFTL_OPERATOR_VALUE || repro == MFTL_REPRODUCE_NONE)) return MFTL_COUPLING_INVALID;
  if (op == MFTL_OPERATOR_VALUE_DERIVATIVE && kern->deriv == MFTL_NULL) return MFTL_COUPLING_INVALID;
  if (op != MFTL_OPERATOR_VALUE && op != MFTL_OPERATOR_GRADIENT_FIT && op != MFTL_OPERATOR_VALUE_DERIVATIVE) return MFTL_COUPLING_INVALID;
  if (n <= 0) return MFTL_COUPLING_BAD_STENCIL;
  const bool deriv = (op == MFTL_OPERATOR_VALUE_DERIVATIVE);

  scratch_reserve(s, n);
  double *const aux = aux_reserve(s, n);
  double *const d = s->d, *const phi = s->phi, *const psi = s->psi, *const rr = s->r;
  double *const dphi = aux, *const Phi = aux + 3 * n, *const tmp = aux + 4 * n, *const dpsi = aux + 5 * n, *const ddm = aux + 8 * n,
                *const W = aux + 17 * n, *const D = aux + 18 * n, *const rel = aux + 21 * n;

  /* ── offsets and base weights; the kernel's derivative where needed ── */
  const double *A = met->A;
  for (int j = 0; j < n; ++j)
  {
    const double r0 = X[3 * j] - x[0], r1 = X[3 * j + 1] - x[1], r2 = X[3 * j + 2] - x[2];
    rel[3 * j] = r0;
    rel[3 * j + 1] = r1;
    rel[3 * j + 2] = r2;
    for (int k = 0; k < 3; ++k) d[3 * j + k] = A[3 * k] * r0 + A[3 * k + 1] * r1 + A[3 * k + 2] * r2;
  }
  if (kern->radial)
  {
    for (int j = 0; j < n; ++j) rr[j] = sqrt(d[3 * j] * d[3 * j] + d[3 * j + 1] * d[3 * j + 1] + d[3 * j + 2] * d[3 * j + 2]);
    kern->eval(rr, Phi, n, kern->ctx);
    if (deriv) kern->deriv(rr, dphi, n, kern->ctx);
  }
  else
  {
    for (int j = 0; j < n; ++j) Phi[j] = 1.0;
    for (int k = 0; k < 3; ++k)
    {
      for (int j = 0; j < n; ++j) rr[j] = d[3 * j + k];
      kern->eval(rr, tmp, n, kern->ctx);
      for (int j = 0; j < n; ++j)
      {
        phi[3 * j + k] = tmp[j];
        Phi[j] *= tmp[j];
      }
      if (deriv)
      {
        kern->deriv(rr, tmp, n, kern->ctx);
        for (int j = 0; j < n; ++j) dphi[3 * j + k] = tmp[j];
      }
    }
  }
  int n_support = 0;
  for (int j = 0; j < n; ++j)
  {
    psi[j] = q[j] * met->detA * Phi[j];
    if (psi[j] == 0.0) continue;
    ++n_support;
    if (psi[j] < 0.0 && repro == MFTL_REPRODUCE_CONSTANT && op != MFTL_OPERATOR_GRADIENT_FIT) return MFTL_COUPLING_BAD_STENCIL;
  }
  qo->n_support = n_support;
  if (n_support == 0) return MFTL_COUPLING_BAD_STENCIL;
  const double tol = p->moment_tolerance;

  /* ════ directional value: U = sum_j A_j q_j ════ */
  if (dir != MFTL_NULL)
  {
    double cond = 1.0;
    double *const sig = tmp; /* |s_j| */
    for (int j = 0; j < n; ++j)
    {
      const double *sj = &dir[3 * j];
      sig[j] = sqrt(sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2]);
      if (!(sig[j] > 0.0) || !isfinite(sig[j])) return MFTL_COUPLING_BAD_STENCIL;
    }
    if (repro == MFTL_REPRODUCE_CONSTANT)
    {
      double T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, Ti[9];
      for (int j = 0; j < n; ++j)
      {
        if (psi[j] == 0.0) continue;
        const double nh[3] = {dir[3 * j] / sig[j], dir[3 * j + 1] / sig[j], dir[3 * j + 2] / sig[j]};
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) T[3 * a + b] += psi[j] * nh[a] * nh[b];
      }
      if (!spd_inverse(3, T, p->max_condition, Ti, &cond))
      {
        qo->condition = cond;
        return MFTL_COUPLING_BAD_STENCIL;
      }
      for (int j = 0; j < n; ++j)
      {
        const double nh[3] = {dir[3 * j] / sig[j], dir[3 * j + 1] / sig[j], dir[3 * j + 2] / sig[j]};
        for (int k = 0; k < 3; ++k)
          w[3 * j + k] = (psi[j] == 0.0) ? 0.0 : psi[j] * (Ti[3 * k] * nh[0] + Ti[3 * k + 1] * nh[1] + Ti[3 * k + 2] * nh[2]) / sig[j];
      }
    }
    else
    {
      double G[144], Gi[144];
      memset(G, 0, sizeof G);
      for (int j = 0; j < n; ++j)
      {
        if (psi[j] == 0.0) continue;
        double f[12];
        for (int a = 0; a < 3; ++a)
        {
          f[a] = dir[3 * j + a] / sig[j];
          for (int b = 0; b < 3; ++b) f[3 + 3 * a + b] = f[a] * d[3 * j + b];
        }
        for (int a = 0; a < 12; ++a)
          for (int b = a; b < 12; ++b) G[12 * a + b] += psi[j] * f[a] * f[b];
      }
      for (int a = 0; a < 12; ++a)
        for (int b = 0; b < a; ++b) G[12 * a + b] = G[12 * b + a];
      if (!spd_inverse(12, G, p->max_condition, Gi, &cond))
      {
        qo->condition = cond;
        return MFTL_COUPLING_BAD_STENCIL;
      }
      for (int j = 0; j < n; ++j)
      {
        if (psi[j] == 0.0)
        {
          w[3 * j] = w[3 * j + 1] = w[3 * j + 2] = 0.0;
          continue;
        }
        double f[12];
        for (int a = 0; a < 3; ++a)
        {
          f[a] = dir[3 * j + a] / sig[j];
          for (int b = 0; b < 3; ++b) f[3 + 3 * a + b] = f[a] * d[3 * j + b];
        }
        for (int k = 0; k < 3; ++k)
        {
          double acc = 0.0;
          for (int l = 0; l < 12; ++l) acc += Gi[12 * k + l] * f[l];
          w[3 * j + k] = psi[j] * acc / sig[j];
        }
      }
    }
    /* what the returned block weights reproduce: sum_j A_j s_j^T = I and sum_j A_j (s_j (x) d_j)^T = 0 */
    double R0[9] = {0}, R1[27] = {0}, sa[3] = {0, 0, 0}, dm = 1.0;
    qo->min_weight = DBL_MAX;
    qo->max_weight = -DBL_MAX;
    for (int j = 0; j < n; ++j)
    {
      if (psi[j] == 0.0) continue;
      for (int k = 0; k < 3; ++k)
      {
        const double a = w[3 * j + k];
        sa[k] += fabs(a) * sig[j];
        qo->min_weight = fmin(qo->min_weight, a);
        qo->max_weight = fmax(qo->max_weight, a);
        dm = fmax(dm, fabs(d[3 * j + k]));
        for (int l = 0; l < 3; ++l)
        {
          R0[3 * k + l] += a * dir[3 * j + l];
          for (int b = 0; b < 3; ++b) R1[9 * k + 3 * l + b] += a * dir[3 * j + l] * d[3 * j + b];
        }
      }
    }
    double r0 = 0.0, r1 = 0.0;
    for (int k = 0; k < 3; ++k)
      for (int l = 0; l < 3; ++l) r0 = fmax(r0, fabs(R0[3 * k + l] - ((k == l) ? 1.0 : 0.0)));
    for (int i = 0; i < 27; ++i) r1 = fmax(r1, fabs(R1[i]));
    const double sum_abs = fmax(sa[0], fmax(sa[1], sa[2]));
    qo->sum_abs = sum_abs;
    qo->condition = cond;
    qo->corrected = true;
    const double bound = block_bound(p, cond, sum_abs, dm);
    qo->order = (r0 <= bound && r1 <= bound) ? 1 : ((r0 <= bound) ? 0 : -1);
    qo->moment_residual = (repro == MFTL_REPRODUCE_LINEAR) ? fmax(r0, r1) : r0;
    if (repro == MFTL_REPRODUCE_LINEAR && qo->order < 1) return MFTL_COUPLING_BAD_STENCIL;
    if (qo->order < 0) return MFTL_COUPLING_BAD_STENCIL;
    if (!(sum_abs <= p->max_amplification) || !isfinite(sum_abs)) return MFTL_COUPLING_BAD_STENCIL;
    qo->status = MFTL_COUPLING_OK;
    return MFTL_COUPLING_OK;
  }

  /* ════ the gradient of the weighted affine fit ════ */
  if (op == MFTL_OPERATOR_GRADIENT_FIT)
  {
    double M[16], Mi[16], cond = 1.0;
    memset(M, 0, sizeof M);
    for (int j = 0; j < n; ++j)
    {
      if (psi[j] == 0.0) continue;
      const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
      for (int a = 0; a < 4; ++a)
        for (int b = a; b < 4; ++b) M[4 * a + b] += psi[j] * pj[a] * pj[b];
    }
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < a; ++b) M[4 * a + b] = M[4 * b + a];
    if (!spd_inverse(4, M, p->max_condition, Mi, &cond))
    {
      qo->condition = cond;
      return MFTL_COUPLING_BAD_STENCIL;
    }
    for (int j = 0; j < n; ++j)
    {
      const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
      for (int k = 0; k < 3; ++k)
        D[3 * j + k] = (psi[j] == 0.0) ? 0.0 : psi[j] * (Mi[4 * (k + 1)] * pj[0] + Mi[4 * (k + 1) + 1] * pj[1] + Mi[4 * (k + 1) + 2] * pj[2] + Mi[4 * (k + 1) + 3] * pj[3]);
      for (int m = 0; m < 3; ++m) w[3 * j + m] = A[m] * D[3 * j] + A[3 + m] * D[3 * j + 1] + A[6 + m] * D[3 * j + 2]; /* A^T beta */
    }
    double sum_abs, r0, r1, dm;
    gradient_moments(D, d, psi, n, &sum_abs, &r0, &r1, &dm);
    qo->min_weight = DBL_MAX;
    qo->max_weight = -DBL_MAX;
    for (int j = 0; j < n; ++j)
      if (psi[j] != 0.0)
        for (int k = 0; k < 3; ++k)
        {
          qo->min_weight = fmin(qo->min_weight, w[3 * j + k]);
          qo->max_weight = fmax(qo->max_weight, w[3 * j + k]);
        }
    qo->sum_abs = sum_abs;
    qo->condition = cond;
    qo->corrected = true;
    const double bound = block_bound(p, cond, sum_abs, dm);
    qo->order = (r0 <= bound && r1 <= bound) ? 1 : ((r0 <= bound) ? 0 : -1);
    qo->moment_residual = fmax(r0, r1);
    if (qo->order < 1 || !(sum_abs <= p->max_amplification) || !isfinite(sum_abs)) return MFTL_COUPLING_BAD_STENCIL;
    qo->status = MFTL_COUPLING_OK;
    return MFTL_COUPLING_OK;
  }

  /* ════ the derivative of the value row ════
   * First the value row exactly as the reproduction policy builds it (its
   * four checks included), then d/dx of every factor it depends on. */
  double M[16], Mi[16], c[4] = {1.0, 0.0, 0.0, 0.0}, cond = 1.0, S0 = 0.0;
  bool corrected = false;
  if (repro == MFTL_REPRODUCE_NONE)
    for (int j = 0; j < n; ++j) W[j] = psi[j];
  else if (repro == MFTL_REPRODUCE_CONSTANT)
  {
    for (int j = 0; j < n; ++j) S0 += psi[j];
    if (!(S0 > 0.0)) return MFTL_COUPLING_BAD_STENCIL;
    for (int j = 0; j < n; ++j) W[j] = psi[j] / S0;
    corrected = fabs(S0 - 1.0) > tol;
  }
  else
  {
    double m0 = 0.0, m1[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < n; ++j)
    {
      m0 += psi[j];
      for (int k = 0; k < 3; ++k) m1[k] += psi[j] * d[3 * j + k];
    }
    corrected = !(fabs(m0 - 1.0) <= tol && fmax(fabs(m1[0]), fmax(fabs(m1[1]), fabs(m1[2]))) <= tol);
    /* A correction that vanishes at x can still have a nonzero derivative.
     * Always form its moment inverse for VALUE_DERIVATIVE. */
    memset(M, 0, sizeof M);
    for (int j = 0; j < n; ++j)
    {
      if (psi[j] == 0.0) continue;
      const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
      for (int a = 0; a < 4; ++a)
        for (int b = a; b < 4; ++b) M[4 * a + b] += psi[j] * pj[a] * pj[b];
    }
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < a; ++b) M[4 * a + b] = M[4 * b + a];
    if (!spd_inverse(4, M, p->max_condition, Mi, &cond))
    {
      qo->condition = cond;
      return MFTL_COUPLING_BAD_STENCIL;
    }
    for (int a = 0; a < 4; ++a) c[a] = Mi[4 * a];
    for (int j = 0; j < n; ++j) W[j] = (psi[j] == 0.0) ? 0.0 : psi[j] * (c[0] + c[1] * d[3 * j] + c[2] * d[3 * j + 1] + c[3] * d[3 * j + 2]);
  }
  {
    double r0 = 0.0, r1 = 0.0, vsum = 0.0, dm = 1.0;
    moments_of(W, d, n, &r0, &r1, &vsum, &dm);
    const double bound = block_bound(p, cond, vsum, dm);
    const int vorder = (r0 <= bound && r1 <= bound) ? 1 : ((r0 <= bound) ? 0 : -1);
    if (repro == MFTL_REPRODUCE_LINEAR && vorder < 1) return MFTL_COUPLING_BAD_STENCIL;
    if (repro == MFTL_REPRODUCE_CONSTANT && vorder < 0) return MFTL_COUPLING_BAD_STENCIL;
    if (repro != MFTL_REPRODUCE_NONE && !(vsum <= p->max_amplification)) return MFTL_COUPLING_BAD_STENCIL;
    if (!isfinite(vsum)) return MFTL_COUPLING_BAD_STENCIL;
  }
  /* d(d_j)/dx_m = (dA/dx_m) r_j - A e_m;  d(psi_j)/dx_m = q_j |det A| (Phi_j dlog|det A|/dx_m + grad Phi . d(d_j)/dx_m) */
  for (int j = 0; j < n; ++j)
  {
    double g[3];
    if (kern->radial)
    {
      const double rho = rr[j];
      for (int k = 0; k < 3; ++k) g[k] = (rho > 0.0) ? dphi[j] * d[3 * j + k] / rho : 0.0;
    }
    else
    {
      g[0] = dphi[3 * j] * phi[3 * j + 1] * phi[3 * j + 2];
      g[1] = phi[3 * j] * dphi[3 * j + 1] * phi[3 * j + 2];
      g[2] = phi[3 * j] * phi[3 * j + 1] * dphi[3 * j + 2];
    }
    for (int m = 0; m < 3; ++m)
    {
      double *dd = &ddm[9 * j + 3 * m];
      for (int k = 0; k < 3; ++k)
      {
        dd[k] = -A[3 * k + m];
        if (met->varying) dd[k] += met->dA[9 * m + 3 * k] * rel[3 * j] + met->dA[9 * m + 3 * k + 1] * rel[3 * j + 1] + met->dA[9 * m + 3 * k + 2] * rel[3 * j + 2];
      }
      double v = g[0] * dd[0] + g[1] * dd[1] + g[2] * dd[2];
      if (met->varying) v += Phi[j] * met->dlogdet[m];
      dpsi[3 * j + m] = q[j] * met->detA * v;
    }
  }
  if (repro == MFTL_REPRODUCE_NONE)
    for (int i = 0; i < 3 * n; ++i) w[i] = dpsi[i];
  else if (repro == MFTL_REPRODUCE_CONSTANT)
  {
    double sd[3] = {0.0, 0.0, 0.0};
    for (int j = 0; j < n; ++j)
      for (int m = 0; m < 3; ++m) sd[m] += dpsi[3 * j + m];
    for (int j = 0; j < n; ++j)
      for (int m = 0; m < 3; ++m) w[3 * j + m] = (dpsi[3 * j + m] - W[j] * sd[m]) / S0;
  }
  else
  {
    /* W_j = psi_j u_j, u_j = c . p_j, M c = e0:  dc = -M^-1 (dM) c */
    double v[3][4];
    memset(v, 0, sizeof v);
    for (int j = 0; j < n; ++j)
    {
      if (psi[j] == 0.0 && dpsi[3 * j] == 0.0 && dpsi[3 * j + 1] == 0.0 && dpsi[3 * j + 2] == 0.0) continue;
      const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
      const double u = c[0] + c[1] * pj[1] + c[2] * pj[2] + c[3] * pj[3];
      for (int m = 0; m < 3; ++m)
      {
        const double *dd = &ddm[9 * j + 3 * m];
        const double dc = c[1] * dd[0] + c[2] * dd[1] + c[3] * dd[2];
        v[m][0] += dpsi[3 * j + m] * u + psi[j] * dc;
        for (int k = 0; k < 3; ++k) v[m][1 + k] += dpsi[3 * j + m] * pj[1 + k] * u + psi[j] * (dd[k] * u + pj[1 + k] * dc);
      }
    }
    double dcv[3][4];
    for (int m = 0; m < 3; ++m)
      for (int a = 0; a < 4; ++a) dcv[m][a] = -(Mi[4 * a] * v[m][0] + Mi[4 * a + 1] * v[m][1] + Mi[4 * a + 2] * v[m][2] + Mi[4 * a + 3] * v[m][3]);
    for (int j = 0; j < n; ++j)
    {
      const double pj[4] = {1.0, d[3 * j], d[3 * j + 1], d[3 * j + 2]};
      const double u = c[0] + c[1] * pj[1] + c[2] * pj[2] + c[3] * pj[3];
      for (int m = 0; m < 3; ++m)
      {
        const double *dd = &ddm[9 * j + 3 * m];
        const double dc = c[1] * dd[0] + c[2] * dd[1] + c[3] * dd[2];
        w[3 * j + m] = dpsi[3 * j + m] * u + psi[j] * (dc + dcv[m][0] + dcv[m][1] * pj[1] + dcv[m][2] * pj[2] + dcv[m][3] * pj[3]);
      }
    }
  }
  /* in metric units: D_j = H^T grad W_j; sum D = 0 always, sum D d^T = I for a LINEAR row */
  const double *H = met->H;
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < 3; ++k) D[3 * j + k] = H[k] * w[3 * j] + H[3 + k] * w[3 * j + 1] + H[6 + k] * w[3 * j + 2];
  double sum_abs, r0, r1, dm;
  gradient_moments(D, d, psi, n, &sum_abs, &r0, &r1, &dm);
  qo->min_weight = DBL_MAX;
  qo->max_weight = -DBL_MAX;
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < 3; ++k)
    {
      if (psi[j] == 0.0 && w[3 * j + k] == 0.0) continue;
      qo->min_weight = fmin(qo->min_weight, w[3 * j + k]);
      qo->max_weight = fmax(qo->max_weight, w[3 * j + k]);
    }
  qo->sum_abs = sum_abs;
  qo->condition = cond;
  qo->corrected = corrected;
  const double bound = block_bound(p, cond, sum_abs, dm);
  qo->order = (r0 <= bound && r1 <= bound) ? 1 : ((r0 <= bound) ? 0 : -1);
  qo->moment_residual = (repro == MFTL_REPRODUCE_LINEAR) ? fmax(r0, r1) : ((repro == MFTL_REPRODUCE_CONSTANT) ? r0 : 0.0);
  if (repro == MFTL_REPRODUCE_LINEAR && qo->order < 1) return MFTL_COUPLING_BAD_STENCIL;
  if (repro == MFTL_REPRODUCE_CONSTANT && qo->order < 0) return MFTL_COUPLING_BAD_STENCIL;
  if (repro != MFTL_REPRODUCE_NONE && !(sum_abs <= p->max_amplification)) return MFTL_COUPLING_BAD_STENCIL;
  if (!isfinite(sum_abs)) return MFTL_COUPLING_BAD_STENCIL;
  qo->status = MFTL_COUPLING_OK;
  return MFTL_COUPLING_OK;
}

int MFTLCouplingAssembleBlockRow(const MFTL_CouplerParams *params, MFTL_CouplingOperator op, const double x[3], const double Hc[9], const double *dHc, int n,
                                 const double *X, const double *q, const double *dir, double *w, MFTL_RowQuality *quality)
{
  MFTL_RowQuality local;
  MFTL_RowQuality *const qo = quality ? quality : &local;
  memset(qo, 0, sizeof *qo);
  qo->status = MFTL_COUPLING_BAD_STENCIL;
  if (params == MFTL_NULL || x == MFTL_NULL || Hc == MFTL_NULL || (n > 0 && (X == MFTL_NULL || q == MFTL_NULL || w == MFTL_NULL)))
    return MFTL_COUPLING_INVALID;
  cpl_kernel kern;
  if (MFTLCplKernelResolve(params, &kern) != MFTL_SUCCESS) return MFTL_COUPLING_INVALID;
  cpl_row_scratch s;
  memset(&s, 0, sizeof s);
  int rc;
  if (op == MFTL_OPERATOR_VALUE && dir == MFTL_NULL)
    rc = MFTLCplRowWeightsCell(params, &kern, x, Hc, n, X, q, w, qo, &s);
  else
  {
    cpl_metric met;
    if (!MFTLCplMetricSet(&met, true, Hc, (op == MFTL_OPERATOR_VALUE_DERIVATIVE) ? dHc : MFTL_NULL))
      rc = MFTL_COUPLING_BAD_STENCIL;
    else
      rc = MFTLCplBlockRow(params, &kern, op, x, &met, n, X, q, dir, w, qo, &s);
  }
  MFTLCplRowScratchFree(&s);
  return rc;
}

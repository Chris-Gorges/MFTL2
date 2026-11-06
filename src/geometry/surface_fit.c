/**
 * @file surface_fit.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_geometry
 *
 * @brief Small dense least squares (Householder QR with grouped column
 *        pivoting) and the weighted height-paraboloid fit built on it.
 *
 * Sized for LOCAL fits: at most #MFTL_LSQ_MAX_COLS unknowns and a few hundred
 * rows, everything on the stack. The library does not link LAPACK; at these
 * sizes a call into `dgelsy` spends more on its workspace and blocking logic
 * than on the arithmetic (studies/surface_fit/ measures both).
 *
 * Nothing here is dimensional: the rank test compares diagonal entries of R
 * with each other, and the caller scales its coordinates.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"

/** @brief Rows MFTLFitHeightParaboloid() can assemble: 64 samples, three rows each. */
#define MFTL_FIT_MAX_ROWS (3 * 64)

/**
 * @brief Downdated column norms are recomputed when they have lost this much
 *        of their value to cancellation (LAPACK's `dlaqp2` uses the same idea).
 */
#define MFTL_LSQ_NORM_RECOMPUTE 0.1

/**
 * @brief Squared 2-norm of column @p j of @p A from row @p r0 on.
 * @param A  Column-major matrix.
 * @param m  Rows (leading dimension).
 * @param j  Column.
 * @param r0 First row.
 * @return The squared norm.
 */
static inline double column_norm_sq(const double *A, int m, int j, int r0)
{
  const double *a = A + (size_t) j * m;
  double s = 0.0;
  for (int i = r0; i < m; ++i) s += a[i] * a[i];
  return s;
}

/**
 * @brief Swaps columns @p i and @p j of the column-major @p A.
 * @param A Matrix.
 * @param m Rows.
 * @param i First column.
 * @param j Second column.
 */
static inline void swap_columns(double *A, int m, int i, int j)
{
  double *a = A + (size_t) i * m;
  double *b = A + (size_t) j * m;
  for (int r = 0; r < m; ++r)
  {
    const double t = a[r];
    a[r] = b[r];
    b[r] = t;
  }
}

int MFTLLeastSquaresQRCP(int m, int n, int n_lead, double *A, double *b, double tau, double *x, unsigned char *kept)
{
  MFTL_ASSERT(n >= 0 && n <= MFTL_LSQ_MAX_COLS && n_lead >= 0 && n_lead <= n && m >= 0);

  int perm[MFTL_LSQ_MAX_COLS];
  double norm_sq[MFTL_LSQ_MAX_COLS];
  double norm_ref[MFTL_LSQ_MAX_COLS];  // norm at the last exact computation
  double rdiag[MFTL_LSQ_MAX_COLS];

  for (int j = 0; j < n; ++j)
  {
    perm[j] = j;
    norm_sq[j] = column_norm_sq(A, m, j, 0);
    norm_ref[j] = norm_sq[j];
    x[j] = 0.0;
    if (kept) kept[j] = 0;
  }

  /* Columns [k, active) are still candidates; dropped columns are swapped to
   * [active, n) and never factored. `group_end` is where the current group
   * stops: the lead group first, then everything that remains. */
  int active = n;
  int group_end = n_lead;
  int k = 0;
  double r00 = 0.0;

  while (k < active && k < m)
  {
    if (k >= group_end) group_end = active;  // the lead group is done (or dropped): the trailing one

    /* Pivot: the largest remaining norm in the current group. */
    int p = k;
    for (int j = k + 1; j < group_end; ++j)
      if (norm_sq[j] > norm_sq[p]) p = j;
    if (p != k)
    {
      swap_columns(A, m, k, p);
      double t = norm_sq[k];
      norm_sq[k] = norm_sq[p];
      norm_sq[p] = t;
      t = norm_ref[k];
      norm_ref[k] = norm_ref[p];
      norm_ref[p] = t;
      const int ti = perm[k];
      perm[k] = perm[p];
      perm[p] = ti;
    }

    /* The exact norm of the pivot column below the diagonal: the downdated one
     * picked it, the exact one decides. */
    double *a = A + (size_t) k * m;
    const double s = column_norm_sq(A, m, k, k);
    const double alpha = sqrt(s);
    if (k == 0) r00 = alpha;

    /* Rank test, relative to the first pivot. A failing pivot is the largest of
     * its group, so the whole rest of the group goes: swap it behind the active
     * range (order inside the dropped block is irrelevant). */
    if (!(alpha > tau * r00) || alpha == 0.0)
    {
      const int drop = group_end - k;
      for (int d = 0; d < drop; ++d)
      {
        const int from = group_end - 1 - d;
        const int to = active - 1 - d;
        if (from != to)
        {
          swap_columns(A, m, from, to);
          double t = norm_sq[from];
          norm_sq[from] = norm_sq[to];
          norm_sq[to] = t;
          t = norm_ref[from];
          norm_ref[from] = norm_ref[to];
          norm_ref[to] = t;
          const int ti = perm[from];
          perm[from] = perm[to];
          perm[to] = ti;
        }
      }
      active -= drop;
      group_end = k;  // the next loop turn moves on to the trailing group
      if (group_end < n_lead) n_lead = group_end;
      continue;
    }

    /* Householder reflector H = I - beta v v^T with v = a[k:m] - alpha' e_k,
     * alpha' of the sign opposite to a[k] (no cancellation in v_k). */
    const double alpha_s = (a[k] > 0.0) ? -alpha : alpha;
    const double v0 = a[k] - alpha_s;
    const double vnorm_sq = s - a[k] * a[k] + v0 * v0;
    a[k] = v0;
    rdiag[k] = alpha_s;
    const double beta = (vnorm_sq > 0.0) ? 2.0 / vnorm_sq : 0.0;

    for (int j = k + 1; j < active; ++j)
    {
      double *c = A + (size_t) j * m;
      double dot = 0.0;
      for (int i = k; i < m; ++i) dot += a[i] * c[i];
      const double f = beta * dot;
      for (int i = k; i < m; ++i) c[i] -= f * a[i];

      /* Downdate the norm of the part below row k; recompute it when the
       * downdate has eaten most of it. */
      norm_sq[j] -= c[k] * c[k];
      if (norm_sq[j] < MFTL_LSQ_NORM_RECOMPUTE * norm_ref[j])
      {
        norm_sq[j] = column_norm_sq(A, m, j, k + 1);
        norm_ref[j] = norm_sq[j];
      }
    }
    {
      double dot = 0.0;
      for (int i = k; i < m; ++i) dot += a[i] * b[i];
      const double f = beta * dot;
      for (int i = k; i < m; ++i) b[i] -= f * a[i];
    }
    ++k;
  }

  const int rank = k;

  /* Back substitution on the accepted block: R[0:rank, 0:rank] y = (Q^T b)[0:rank],
   * R's strict upper part in A, its diagonal in rdiag. */
  double y[MFTL_LSQ_MAX_COLS];
  for (int i = rank - 1; i >= 0; --i)
  {
    double t = b[i];
    for (int j = i + 1; j < rank; ++j) t -= A[(size_t) j * m + i] * y[j];
    y[i] = t / rdiag[i];
  }
  for (int i = 0; i < rank; ++i)
  {
    x[perm[i]] = y[i];
    if (kept) kept[perm[i]] = 1;
  }
  return rank;
}

int MFTLFitHeightParaboloid(int n, const double *pts, const double *slopes, const unsigned char *use_slope, const double *w, double tau, double c[6],
                            unsigned char kept[6])
{
  MFTL_ASSERT(n >= 0 && n <= MFTL_FIT_MAX_ROWS / 3);

  /* Rows first counted, then written column by column (column-major A). */
  int rows = 0;
  for (int j = 0; j < n; ++j)
  {
    if (!(w[j] > 0.0)) continue;
    rows += 1;
    if (slopes && (!use_slope || use_slope[j])) rows += 2;
  }

  double A[6 * MFTL_FIT_MAX_ROWS];
  double rhs[MFTL_FIT_MAX_ROWS];
  double *const c0 = A, *const c1 = A + rows, *const c2 = A + 2 * rows;
  double *const c3 = A + 3 * rows, *const c4 = A + 4 * rows, *const c5 = A + 5 * rows;

  int r = 0;
  for (int j = 0; j < n; ++j)
  {
    if (!(w[j] > 0.0)) continue;
    const double s = sqrt(w[j]);
    const double x = pts[3 * j + 0], y = pts[3 * j + 1], z = pts[3 * j + 2];

    c0[r] = s;
    c1[r] = s * x;
    c2[r] = s * y;
    c3[r] = s * x * x;
    c4[r] = s * x * y;
    c5[r] = s * y * y;
    rhs[r] = s * z;
    ++r;

    if (!slopes || (use_slope && !use_slope[j])) continue;

    c0[r] = 0.0;  // dz/dx = c1 + 2 c3 x + c4 y
    c1[r] = s;
    c2[r] = 0.0;
    c3[r] = 2.0 * s * x;
    c4[r] = s * y;
    c5[r] = 0.0;
    rhs[r] = s * slopes[2 * j + 0];
    ++r;

    c0[r] = 0.0;  // dz/dy = c2 + c4 x + 2 c5 y
    c1[r] = 0.0;
    c2[r] = s;
    c3[r] = 0.0;
    c4[r] = s * x;
    c5[r] = 2.0 * s * y;
    rhs[r] = s * slopes[2 * j + 1];
    ++r;
  }

  return MFTLLeastSquaresQRCP(rows, 6, 3, A, rhs, tau, c, kept);
}

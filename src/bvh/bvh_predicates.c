/**
 * @file bvh_predicates.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief The narrow phase: exact orientation, exact triangle-triangle
 *        intersection, and the triangle-triangle distance to rounding.
 *
 * Everything here is double precision. The broad phase may be loose (outward
 * rounded float boxes); the narrow phase may not be wrong.
 *
 * **Why only the orientation test is adaptive.** It is the only predicate whose
 * SIGN decides the answer: Guigue-Devillers branches on it up to eight times per
 * pair (six plane classifications, two in check_min_max; the permutation
 * cascade branches on signs already computed), and the coplanar and zero-area
 * cases decide by orientation signs as well -- the 2D ones are this predicate on
 * lifted points. A wrong sign sends the algorithm down a branch for a
 * configuration that does not exist and typically reports a clean miss for two
 * crossing triangles. Distances degrade continuously and only move decisions
 * sitting on the tolerance -- PROVIDED they are computed without the normal
 * equations, which is why they live in src/geometry/ (see section 5).
 *
 * The determinant is ill-conditioned exactly where this library works: thin
 * films (four points coplanar to within the gap) and small features at large
 * coordinates (the differences cancel the offset and its digits). Measured with
 * examples/orient3d_predicate_profile.c (20 000 samples per row, adaptive form
 * as the exact reference): the plain determinant misclassified 0 % at
 * gap/L >= 1e-12, up to 0.035 % at 1e-14, 0.125 % at 1e-15 and 4.7 % at 1e-16.
 * The adaptive form costs 3.66 ns against 1.83 ns in general position (40 ns
 * only for exactly coplanar input), i.e. at most 4.5 % of a 244 ns
 * MFTL_TriTriProximity() call. It is kept because the failure it removes is
 * silent; MFTL_Orient3DFast() is the plain form.
 *
 * Shewchuk's construction (Discrete Comput. Geom. 18, 1997) is a filter: the
 * plain determinant plus a comparison against a rigorous relative error bound,
 * with exact expansion arithmetic only when the estimate cannot certify the sign.
 */

/* The exact arithmetic requires a*b and x - a*b to be rounded SEPARATELY: a
 * fused multiply-add makes Two_Product's residue zero and silently voids every
 * guarantee. Hence the pragma, and -ffp-contract=off in CMakeLists.txt for
 * compilers that ignore it. GCC is one of them, up to at least GCC 15: it
 * warns that it ignores the pragma (-Wunknown-pragmas), so it is not given it
 * and relies on the flag alone. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#include "mftl_bvh.h"

#include <math.h>
#include <stddef.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Exact floating-point expansion arithmetic (Shewchuk)
 *
 * Every macro below is an error-free transformation: it produces a rounded
 * result AND the exact residue, so that the pair (x, y) represents the exact
 * value with x + y. An "expansion" is a list of such components, ordered by
 * magnitude and non-overlapping; its exact value is the sum of its parts.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Machine constants, computed once by exactinit(). */
static double mftl_splitter;  ///< 2^ceil(p/2) + 1, splits a double into two halves.
static double mftl_epsilon;  ///< Half of the last-place unit, 2^-53.
static double mftl_o3derrbound_a;  ///< Bound for the first (plain determinant) estimate.
static double mftl_o3derrbound_b;  ///< Bound after the first correction term.
static double mftl_o3derrbound_c;  ///< Bound after the second correction term.
static int mftl_predicates_ready = 0;  ///< Set by exactinit(); guards lazy initialisation.

/**
 * @brief Determines the machine's rounding behaviour and the error bounds.
 *
 * Measured rather than assumed: the loop finds the actual epsilon of whatever
 * the compiler and hardware do, so a target with extended-precision registers or
 * a different rounding mode gets bounds that are true for IT. The error bound
 * constants are Shewchuk's, derived for the 3x3 determinant.
 */
static void exactinit(void)
{
  double half = 0.5;
  double check = 1.0;
  double lastcheck;
  int every_other = 1;

  mftl_epsilon = 1.0;
  mftl_splitter = 1.0;

  do
  {
    lastcheck = check;
    mftl_epsilon *= half;
    if (every_other) mftl_splitter *= 2.0;
    every_other = !every_other;
    check = 1.0 + mftl_epsilon;
  } while ((check != 1.0) && (check != lastcheck));

  mftl_splitter += 1.0;

  mftl_o3derrbound_a = (7.0 + 56.0 * mftl_epsilon) * mftl_epsilon;
  mftl_o3derrbound_b = (3.0 + 28.0 * mftl_epsilon) * mftl_epsilon;
  mftl_o3derrbound_c = (26.0 + 288.0 * mftl_epsilon) * mftl_epsilon * mftl_epsilon;

  mftl_predicates_ready = 1;
}

/**
 * @brief Splits @p a into two non-overlapping halves.
 *
 * The product of two split halves is exactly representable, which is what makes
 * #MFTL_TWO_PRODUCT able to recover its own residue.
 *
 * @param a         The value to split.
 * @param[out] ahi  High half.
 * @param[out] alo  Low half. \f$a = ahi + alo\f$ exactly.
 */
#define MFTL_SPLIT(a, ahi, alo)      \
  do                                 \
  {                                  \
    double c = mftl_splitter * (a);  \
    double abig = c - (a);           \
    (ahi) = c - abig;                \
    (alo) = (a) - (ahi);             \
  } while (0)

/**
 * @brief \f$x + y = a + b\f$ exactly, requiring \f$\abs{a} \ge \abs{b}\f$.
 * @param a        Larger addend.
 * @param b        Smaller addend.
 * @param[out] x   Rounded sum.
 * @param[out] y   Exact residue.
 */
#define MFTL_FAST_TWO_SUM(a, b, x, y) \
  do                                  \
  {                                   \
    (x) = (a) + (b);                  \
    double bvirt = (x) - (a);         \
    (y) = (b) - bvirt;                \
  } while (0)

/**
 * @brief \f$x + y = a + b\f$ exactly, with no magnitude requirement.
 * @param a        First addend.
 * @param b        Second addend.
 * @param[out] x   Rounded sum.
 * @param[out] y   Exact residue.
 */
#define MFTL_TWO_SUM(a, b, x, y)      \
  do                                  \
  {                                   \
    (x) = (a) + (b);                  \
    double bvirt = (x) - (a);         \
    double avirt = (x) - bvirt;       \
    double bround = (b) - bvirt;      \
    double around = (a) - avirt;      \
    (y) = around + bround;            \
  } while (0)

/**
 * @brief \f$x + y = a - b\f$ exactly.
 * @param a        Minuend.
 * @param b        Subtrahend.
 * @param[out] x   Rounded difference.
 * @param[out] y   Exact residue.
 */
#define MFTL_TWO_DIFF(a, b, x, y)     \
  do                                  \
  {                                   \
    (x) = (a) - (b);                  \
    double bvirt = (a) - (x);         \
    double avirt = (x) + bvirt;       \
    double bround = bvirt - (b);      \
    double around = (a) - avirt;      \
    (y) = around + bround;            \
  } while (0)

/**
 * @brief \f$x + y = a \cdot b\f$ exactly.
 *
 * @param a        First factor.
 * @param b        Second factor.
 * @param[out] x   Rounded product.
 * @param[out] y   Exact residue.
 *
 * @warning **This is the macro a fused multiply-add would break.** The residue
 *          is recovered by evaluating \f$x - a_{hi}b_{hi}\f$ in a SECOND
 *          rounding step; an FMA contracts that into one rounding, @p y comes
 *          out zero, and every exactness guarantee in this file evaporates while
 *          the numbers still look plausible. Hence `-ffp-contract=off`.
 */
#define MFTL_TWO_PRODUCT(a, b, x, y)      \
  do                                      \
  {                                       \
    (x) = (a) * (b);                      \
    double ahi, alo, bhi, blo;            \
    MFTL_SPLIT((a), ahi, alo);            \
    MFTL_SPLIT((b), bhi, blo);            \
    double err1 = (x) - (ahi * bhi);      \
    double err2 = err1 - (alo * bhi);     \
    double err3 = err2 - (ahi * blo);     \
    (y) = (alo * blo) - err3;             \
  } while (0)

/**
 * @brief Subtracts a scalar from a 2-component expansion, giving 3 components.
 * @param a1 High component of the expansion.
 * @param a0 Low component of the expansion.
 * @param b  The scalar to subtract.
 * @param[out] x2 Highest result component.
 * @param[out] x1 Middle result component.
 * @param[out] x0 Lowest result component.
 */
#define MFTL_TWO_ONE_DIFF(a1, a0, b, x2, x1, x0) \
  do                                             \
  {                                              \
    double _i;                                   \
    MFTL_TWO_DIFF((a0), (b), _i, (x0));          \
    MFTL_TWO_SUM((a1), _i, (x2), (x1));          \
  } while (0)

/**
 * @brief Subtracts two 2-component expansions, giving 4 components.
 * @param a1 High component of the minuend.
 * @param a0 Low component of the minuend.
 * @param b1 High component of the subtrahend.
 * @param b0 Low component of the subtrahend.
 * @param[out] x3 Highest result component.
 * @param[out] x2 Result component.
 * @param[out] x1 Result component.
 * @param[out] x0 Lowest result component.
 */
#define MFTL_TWO_TWO_DIFF(a1, a0, b1, b0, x3, x2, x1, x0)   \
  do                                                        \
  {                                                         \
    double _j, _0;                                          \
    MFTL_TWO_ONE_DIFF((a1), (a0), (b0), _j, _0, (x0));      \
    MFTL_TWO_ONE_DIFF(_j, _0, (b1), (x3), (x2), (x1));      \
  } while (0)

/**
 * @brief Merges two expansions into their exact sum, dropping zero components.
 *
 * Shewchuk's fast_expansion_sum: both inputs are ordered by increasing
 * magnitude and non-overlapping, so the sum is a single merge pass with a
 * running carry, no comparisons beyond picking the next-smallest component.
 *
 * "zeroelim" is what keeps the expansions short: an exact operation on two
 * components usually produces a zero residue, and carrying it would lengthen
 * every subsequent pass for no information.
 *
 * ── One deliberate departure from the published code ──────────────────────
 *
 * The original advances with `enow = e[++eindex]` and therefore reads e[elen]
 * once, relying on the value never being used. That is a genuine out-of-bounds
 * read -- harmless in practice, fatal under the AddressSanitizer this project's
 * Debug build enables by default, and there is no reason to keep it. The merge
 * below picks its next component only after testing the bound, which produces
 * the identical sequence of arithmetic operations.
 *
 * @param elen Component count of @p e.
 * @param e    First expansion, ordered by increasing magnitude.
 * @param flen Component count of @p f.
 * @param f    Second expansion.
 * @param h    Output expansion. Must hold `elen + flen` components.
 * @return Number of components actually written to @p h.
 */
static int expansion_sum_zeroelim(int elen, const double *e, int flen, const double *f, double *h)
{
  double q, qnew, hh;
  int eindex = 0, findex = 0, hindex = 0;

  /* Seed the carry with the smaller-magnitude leading component. The comparison
   * is Shewchuk's branch-free magnitude test: it is true exactly when
   * |e[0]| <= |f[0]|. */
  if ((f[0] > e[0]) == (f[0] > -e[0]))
    q = e[eindex++];
  else
    q = f[findex++];

  int first = 1;
  while (eindex < elen || findex < flen)
  {
    double next;
    if (eindex < elen && findex < flen)
    {
      const double enow = e[eindex];
      const double fnow = f[findex];
      if ((fnow > enow) == (fnow > -enow))
        next = e[eindex++];
      else
        next = f[findex++];
    }
    else if (eindex < elen)
    {
      next = e[eindex++];
    }
    else
    {
      next = f[findex++];
    }

    /* The first merge step may use the cheap two-sum: `next` was chosen as the
     * next-smallest component, so |next| >= |q| is guaranteed. From the second
     * step on the carry has grown and that ordering no longer holds. */
    if (first)
    {
      MFTL_FAST_TWO_SUM(next, q, qnew, hh);
      first = 0;
    }
    else
    {
      MFTL_TWO_SUM(q, next, qnew, hh);
    }
    q = qnew;
    if (hh != 0.0) h[hindex++] = hh;
  }

  if ((q != 0.0) || (hindex == 0)) h[hindex++] = q;
  return hindex;
}

/**
 * @brief Multiplies an expansion by a scalar exactly, dropping zero components.
 *
 * @param elen Component count of @p e.
 * @param e    The expansion to scale.
 * @param b    The scalar factor.
 * @param h    Output expansion. Must hold `2 * elen` components.
 * @return Number of components actually written to @p h.
 */
static int scale_expansion_zeroelim(int elen, const double *e, double b, double *h)
{
  double q, sum, product1, product0, hh;
  int hindex = 0;

  MFTL_TWO_PRODUCT(e[0], b, q, hh);
  if (hh != 0.0) h[hindex++] = hh;

  for (int eindex = 1; eindex < elen; eindex++)
  {
    double enow = e[eindex];
    MFTL_TWO_PRODUCT(enow, b, product1, product0);
    MFTL_TWO_SUM(q, product0, sum, hh);
    if (hh != 0.0) h[hindex++] = hh;
    MFTL_FAST_TWO_SUM(product1, sum, q, hh);
    if (hh != 0.0) h[hindex++] = hh;
  }

  if ((q != 0.0) || (hindex == 0)) h[hindex++] = q;
  return hindex;
}

/**
 * @brief An approximation of an expansion's value, good to the last component.
 *
 * Summing smallest-first is deliberate: the components are ordered by increasing
 * magnitude and non-overlapping, so this accumulates the correction terms before
 * they are swamped by the leading one.
 *
 * @param elen Component count of @p e.
 * @param e    The expansion.
 * @return The plain floating-point sum of the components, i.e. the value the
 *         expansion approximates.
 */
static double estimate(int elen, const double *e)
{
  double q = e[0];
  for (int eindex = 1; eindex < elen; eindex++) q += e[eindex];
  return q;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The orientation predicate
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTL_Orient3DFast(const double pa[3], const double pb[3], const double pc[3], const double pd[3])
{
  const double adx = pa[0] - pd[0], bdx = pb[0] - pd[0], cdx = pc[0] - pd[0];
  const double ady = pa[1] - pd[1], bdy = pb[1] - pd[1], cdy = pc[1] - pd[1];
  const double adz = pa[2] - pd[2], bdz = pb[2] - pd[2], cdz = pc[2] - pd[2];

  return adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) + cdx * (ady * bdz - adz * bdy);
}

/**
 * @brief Accumulates (a * b) * c, and (a * b) * c_tail when that is non-zero,
 *        into the running expansion of orient3dadapt's third level.
 *
 * @param a         First factor.
 * @param b         Second factor.
 * @param c         Third factor.
 * @param c_tail    Residue of @p c from an earlier error-free transformation.
 * @param finlength Component count of the running result.
 * @param finnow    Address of the pointer to the current result buffer.
 * @param finother  Address of the pointer to the alternate buffer. The two are
 *                  swapped so the merge never writes into its own input.
 * @return The new component count.
 */
static int add_tail_product(double a, double b, double c, double c_tail, int finlength, double **finnow, double **finother)
{
  double uu[2];
  MFTL_TWO_PRODUCT(a, b, uu[1], uu[0]);

  double v[12];
  int vlength = scale_expansion_zeroelim(2, uu, c, v);
  finlength = expansion_sum_zeroelim(finlength, *finnow, vlength, v, *finother);
  {
    double *swap = *finnow;
    *finnow = *finother;
    *finother = swap;
  }

  if (c_tail != 0.0)
  {
    vlength = scale_expansion_zeroelim(2, uu, c_tail, v);
    finlength = expansion_sum_zeroelim(finlength, *finnow, vlength, v, *finother);
    double *swap = *finnow;
    *finnow = *finother;
    *finother = swap;
  }

  return finlength;
}

/**
 * @brief The exact stage, reached only when the filter cannot decide the sign.
 *
 * Computes the determinant as an exact expansion and returns its estimate, whose
 * SIGN is then guaranteed correct. Structured as three levels of increasing
 * effort -- the first two are themselves filtered, so most calls that reach here
 * still return before the full expansion is built.
 *
 * @param pa      First point of the reference plane.
 * @param pb      Second point.
 * @param pc      Third point.
 * @param pd      The point being classified.
 * @param permanent The magnitude sum from the first estimate, used to scale the
 *                  successive error bounds.
 * @return \\f$6V\\f$, refined only as far as the sign requires.
 */
static double orient3dadapt(const double pa[3], const double pb[3], const double pc[3], const double pd[3], double permanent)
{
  double bc[4], ca[4], ab[4];
  double adet[8], bdet[8], cdet[8], abdet[16], fin1[192];
  double at_b[4], at_c[4], bt_c[4], bt_a[4], ct_a[4], ct_b[4];
  double bct[8], cat[8], abt[8];

  /* NOT const: the MFTL_TWO_DIFF calls further down recompute each of these
   * alongside its exact residue, writing the identical value back. */
  double adx = pa[0] - pd[0], bdx = pb[0] - pd[0], cdx = pc[0] - pd[0];
  double ady = pa[1] - pd[1], bdy = pb[1] - pd[1], cdy = pc[1] - pd[1];
  double adz = pa[2] - pd[2], bdz = pb[2] - pd[2], cdz = pc[2] - pd[2];

  double bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0;
  MFTL_TWO_PRODUCT(bdx, cdy, bdxcdy1, bdxcdy0);
  MFTL_TWO_PRODUCT(cdx, bdy, cdxbdy1, cdxbdy0);
  MFTL_TWO_TWO_DIFF(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0, bc[3], bc[2], bc[1], bc[0]);
  int alen = scale_expansion_zeroelim(4, bc, adz, adet);

  double cdxady1, cdxady0, adxcdy1, adxcdy0;
  MFTL_TWO_PRODUCT(cdx, ady, cdxady1, cdxady0);
  MFTL_TWO_PRODUCT(adx, cdy, adxcdy1, adxcdy0);
  MFTL_TWO_TWO_DIFF(cdxady1, cdxady0, adxcdy1, adxcdy0, ca[3], ca[2], ca[1], ca[0]);
  int blen = scale_expansion_zeroelim(4, ca, bdz, bdet);

  double adxbdy1, adxbdy0, bdxady1, bdxady0;
  MFTL_TWO_PRODUCT(adx, bdy, adxbdy1, adxbdy0);
  MFTL_TWO_PRODUCT(bdx, ady, bdxady1, bdxady0);
  MFTL_TWO_TWO_DIFF(adxbdy1, adxbdy0, bdxady1, bdxady0, ab[3], ab[2], ab[1], ab[0]);
  int clen = scale_expansion_zeroelim(4, ab, cdz, cdet);

  int ablen = expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  int finlength = expansion_sum_zeroelim(ablen, abdet, clen, cdet, fin1);

  double det = estimate(finlength, fin1);
  double errbound = mftl_o3derrbound_b * permanent;
  if ((det >= errbound) || (-det >= errbound)) return det;

  /* Level two: recover the exact residues of the coordinate differences. If all
   * of them are zero the differences were exact, the expansion above IS the
   * answer, and no more work can change it. */
  double adxtail, bdxtail, cdxtail, adytail, bdytail, cdytail, adztail, bdztail, cdztail;
  MFTL_TWO_DIFF(pa[0], pd[0], adx, adxtail);
  MFTL_TWO_DIFF(pb[0], pd[0], bdx, bdxtail);
  MFTL_TWO_DIFF(pc[0], pd[0], cdx, cdxtail);
  MFTL_TWO_DIFF(pa[1], pd[1], ady, adytail);
  MFTL_TWO_DIFF(pb[1], pd[1], bdy, bdytail);
  MFTL_TWO_DIFF(pc[1], pd[1], cdy, cdytail);
  MFTL_TWO_DIFF(pa[2], pd[2], adz, adztail);
  MFTL_TWO_DIFF(pb[2], pd[2], bdz, bdztail);
  MFTL_TWO_DIFF(pc[2], pd[2], cdz, cdztail);

  if ((adxtail == 0.0) && (bdxtail == 0.0) && (cdxtail == 0.0) && (adytail == 0.0) && (bdytail == 0.0) && (cdytail == 0.0) && (adztail == 0.0) &&
      (bdztail == 0.0) && (cdztail == 0.0))
  {
    return det;
  }

  errbound = mftl_o3derrbound_c * permanent;
  det += (adz * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail)) + adztail * (bdx * cdy - bdy * cdx)) +
         (bdz * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail)) + bdztail * (cdx * ady - cdy * adx)) +
         (cdz * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail)) + cdztail * (adx * bdy - ady * bdx));
  if ((det >= errbound) || (-det >= errbound)) return det;

  /* Level three: the fully exact expansion. Reached only for input that is
   * degenerate to within the last bit, which for this library means genuinely
   * coplanar geometry rather than merely thin. */
  double *finnow = fin1;
  double fin2[192];
  double *finother = fin2;
  double v[12], w[16];
  int vlength, wlength;

  int at_blen = 0, at_clen = 0, bt_clen = 0, bt_alen = 0, ct_alen = 0, ct_blen = 0;

  /* Cross-terms of the x/y tails. Each of the six is built the same way; the
   * repetition is Shewchuk's and is left in rather than folded into a helper,
   * because every one of them has a different sign pattern and a loop over them
   * would need the pattern as data. */
  double ti1, ti0, tj1, tj0;

  if (adxtail == 0.0)
  {
    if (adytail == 0.0)
    {
      at_b[0] = 0.0;
      at_blen = 1;
      at_c[0] = 0.0;
      at_clen = 1;
    }
    else
    {
      double negate = -adytail;
      MFTL_TWO_PRODUCT(negate, bdx, at_b[1], at_b[0]);
      at_blen = 2;
      MFTL_TWO_PRODUCT(adytail, cdx, at_c[1], at_c[0]);
      at_clen = 2;
    }
  }
  else
  {
    if (adytail == 0.0)
    {
      MFTL_TWO_PRODUCT(adxtail, bdy, at_b[1], at_b[0]);
      at_blen = 2;
      double negate = -adxtail;
      MFTL_TWO_PRODUCT(negate, cdy, at_c[1], at_c[0]);
      at_clen = 2;
    }
    else
    {
      MFTL_TWO_PRODUCT(adxtail, bdy, ti1, ti0);
      MFTL_TWO_PRODUCT(adytail, bdx, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, at_b[3], at_b[2], at_b[1], at_b[0]);
      at_blen = 4;
      MFTL_TWO_PRODUCT(adytail, cdx, ti1, ti0);
      MFTL_TWO_PRODUCT(adxtail, cdy, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, at_c[3], at_c[2], at_c[1], at_c[0]);
      at_clen = 4;
    }
  }

  if (bdxtail == 0.0)
  {
    if (bdytail == 0.0)
    {
      bt_c[0] = 0.0;
      bt_clen = 1;
      bt_a[0] = 0.0;
      bt_alen = 1;
    }
    else
    {
      double negate = -bdytail;
      MFTL_TWO_PRODUCT(negate, cdx, bt_c[1], bt_c[0]);
      bt_clen = 2;
      MFTL_TWO_PRODUCT(bdytail, adx, bt_a[1], bt_a[0]);
      bt_alen = 2;
    }
  }
  else
  {
    if (bdytail == 0.0)
    {
      MFTL_TWO_PRODUCT(bdxtail, cdy, bt_c[1], bt_c[0]);
      bt_clen = 2;
      double negate = -bdxtail;
      MFTL_TWO_PRODUCT(negate, ady, bt_a[1], bt_a[0]);
      bt_alen = 2;
    }
    else
    {
      MFTL_TWO_PRODUCT(bdxtail, cdy, ti1, ti0);
      MFTL_TWO_PRODUCT(bdytail, cdx, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, bt_c[3], bt_c[2], bt_c[1], bt_c[0]);
      bt_clen = 4;
      MFTL_TWO_PRODUCT(bdytail, adx, ti1, ti0);
      MFTL_TWO_PRODUCT(bdxtail, ady, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, bt_a[3], bt_a[2], bt_a[1], bt_a[0]);
      bt_alen = 4;
    }
  }

  if (cdxtail == 0.0)
  {
    if (cdytail == 0.0)
    {
      ct_a[0] = 0.0;
      ct_alen = 1;
      ct_b[0] = 0.0;
      ct_blen = 1;
    }
    else
    {
      double negate = -cdytail;
      MFTL_TWO_PRODUCT(negate, adx, ct_a[1], ct_a[0]);
      ct_alen = 2;
      MFTL_TWO_PRODUCT(cdytail, bdx, ct_b[1], ct_b[0]);
      ct_blen = 2;
    }
  }
  else
  {
    if (cdytail == 0.0)
    {
      MFTL_TWO_PRODUCT(cdxtail, ady, ct_a[1], ct_a[0]);
      ct_alen = 2;
      double negate = -cdxtail;
      MFTL_TWO_PRODUCT(negate, bdy, ct_b[1], ct_b[0]);
      ct_blen = 2;
    }
    else
    {
      MFTL_TWO_PRODUCT(cdxtail, ady, ti1, ti0);
      MFTL_TWO_PRODUCT(cdytail, adx, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, ct_a[3], ct_a[2], ct_a[1], ct_a[0]);
      ct_alen = 4;
      MFTL_TWO_PRODUCT(cdytail, bdx, ti1, ti0);
      MFTL_TWO_PRODUCT(cdxtail, bdy, tj1, tj0);
      MFTL_TWO_TWO_DIFF(ti1, ti0, tj1, tj0, ct_b[3], ct_b[2], ct_b[1], ct_b[0]);
      ct_blen = 4;
    }
  }

  int bctlen = expansion_sum_zeroelim(bt_clen, bt_c, ct_blen, ct_b, bct);
  wlength = scale_expansion_zeroelim(bctlen, bct, adz, w);
  finlength = expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
  {
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }

  int catlen = expansion_sum_zeroelim(ct_alen, ct_a, at_clen, at_c, cat);
  wlength = scale_expansion_zeroelim(catlen, cat, bdz, w);
  finlength = expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
  {
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }

  int abtlen = expansion_sum_zeroelim(at_blen, at_b, bt_alen, bt_a, abt);
  wlength = scale_expansion_zeroelim(abtlen, abt, cdz, w);
  finlength = expansion_sum_zeroelim(finlength, finnow, wlength, w, finother);
  {
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }

  /* The z-tail contributions. */
  if (adztail != 0.0)
  {
    vlength = scale_expansion_zeroelim(4, bc, adztail, v);
    finlength = expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }
  if (bdztail != 0.0)
  {
    vlength = scale_expansion_zeroelim(4, ca, bdztail, v);
    finlength = expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }
  if (cdztail != 0.0)
  {
    vlength = scale_expansion_zeroelim(4, ab, cdztail, v);
    finlength = expansion_sum_zeroelim(finlength, finnow, vlength, v, finother);
    double *swap = finnow;
    finnow = finother;
    finother = swap;
  }

  /* Second-order tail terms: products of TWO tails, scaled by a z-coordinate and
   * by that coordinate's own tail. These are what make the result exact rather
   * than merely better, and every one of them is zero unless the coordinate
   * differences were themselves inexact -- which is why the whole block sits
   * behind the all-tails-zero early return above.
   *
   * The six cases differ only in which tails they multiply and which z they
   * scale by, so they go through one helper. Shewchuk writes them out; the sign
   * pattern is the only thing that varies and it is passed in as a negated
   * argument, so there is nothing left for the repetition to express. */
  if (adxtail != 0.0)
  {
    if (bdytail != 0.0) finlength = add_tail_product(adxtail, bdytail, cdz, cdztail, finlength, &finnow, &finother);
    if (cdytail != 0.0) finlength = add_tail_product(-adxtail, cdytail, bdz, bdztail, finlength, &finnow, &finother);
  }
  if (bdxtail != 0.0)
  {
    if (cdytail != 0.0) finlength = add_tail_product(bdxtail, cdytail, adz, adztail, finlength, &finnow, &finother);
    if (adytail != 0.0) finlength = add_tail_product(-bdxtail, adytail, cdz, cdztail, finlength, &finnow, &finother);
  }
  if (cdxtail != 0.0)
  {
    if (adytail != 0.0) finlength = add_tail_product(cdxtail, adytail, bdz, bdztail, finlength, &finnow, &finother);
    if (bdytail != 0.0) finlength = add_tail_product(-cdxtail, bdytail, adz, adztail, finlength, &finnow, &finother);
  }

  return finnow[finlength - 1];
}

double MFTL_Orient3D(const double pa[3], const double pb[3], const double pc[3], const double pd[3])
{
  if (!mftl_predicates_ready) exactinit();

  const double adx = pa[0] - pd[0], bdx = pb[0] - pd[0], cdx = pc[0] - pd[0];
  const double ady = pa[1] - pd[1], bdy = pb[1] - pd[1], cdy = pc[1] - pd[1];
  const double adz = pa[2] - pd[2], bdz = pb[2] - pd[2], cdz = pc[2] - pd[2];

  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;

  const double det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

  /* The "permanent" is the sum of the magnitudes of the terms the determinant is
   * built from. It is what turns a relative error bound into an absolute one:
   * the true value cannot differ from `det` by more than errbound, so a `det`
   * outside that band has a certain sign and the exact stage is not entered.
   * This is the branch essentially every well-separated pair takes. */
  const double permanent = (fabs(bdxcdy) + fabs(cdxbdy)) * fabs(adz) + (fabs(cdxady) + fabs(adxcdy)) * fabs(bdz) + (fabs(adxbdy) + fabs(bdxady)) * fabs(cdz);
  const double errbound = mftl_o3derrbound_a * permanent;

  if ((det > errbound) || (-det > errbound)) return det;

  return orient3dadapt(pa, pb, pc, pd, permanent);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Guigue-Devillers triangle-triangle intersection (J. Graphics Tools 8(1),
 * 2003), with every branch taken on an exact MFTL_Orient3D sign. The
 * permutations normalise both WHICH vertex is isolated and the ORIENTATION of
 * the second triangle. Guigue-Devillers assumes two proper triangles in
 * distinct planes; every other configuration -- coplanar, or a zero-area
 * triangle -- is recognised exactly and answered by section 4.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The terminal test of Guigue-Devillers, once both triangles are in
 *        canonical position.
 *
 * Each triangle meets the line of intersection of the two planes in an
 * interval; the intervals overlap iff both orientation tests below are
 * non-positive. No interval endpoint or intersection point is constructed.
 *
 * @param p1 First triangle's first corner.
 * @param q1 First triangle's second corner.
 * @param r1 First triangle's third corner.
 * @param p2 Second triangle's first corner.
 * @param q2 Second triangle's second corner.
 * @param r2 Second triangle's third corner.
 * @return 1 if the two triangles overlap, 0 otherwise.
 */
static int check_min_max(const double *p1, const double *q1, const double *r1, const double *p2, const double *q2, const double *r2)
{
  if (MFTL_Orient3D(q2, p2, p1, q1) > 0.0) return 0;
  if (MFTL_Orient3D(r2, p2, r1, p1) > 0.0) return 0;
  return 1;
}

/**
 * @brief Second half of the dispatch: permute B by the signs of its vertices
 *        against A's plane, then run the terminal test.
 *
 * Never reached with all three of @p dp2, @p dq2, @p dr2 zero: that pair is
 * flat (B in A's plane, or A of zero area) and MFTL_TriTriIntersect() has
 * already answered it.
 *
 * @param p1     First triangle's first corner.
 * @param q1     First triangle's second corner.
 * @param r1     First triangle's third corner.
 * @param p2     Second triangle's first corner.
 * @param q2     Second triangle's second corner.
 * @param r2     Second triangle's third corner.
 * @param dp2    Signed distance of @p p2 from the first triangle's plane.
 * @param dq2    Signed distance of @p q2 from that plane.
 * @param dr2    Signed distance of @p r2 from that plane.
 * @return 1 on intersection, 0 otherwise.
 */
static int tri_tri_3d(const double *p1, const double *q1, const double *r1, const double *p2, const double *q2, const double *r2, double dp2, double dq2,
                      double dr2)
{
  if (dp2 > 0.0)
  {
    if (dq2 > 0.0) return check_min_max(p1, r1, q1, r2, p2, q2);
    if (dr2 > 0.0) return check_min_max(p1, r1, q1, q2, r2, p2);
    return check_min_max(p1, q1, r1, p2, q2, r2);
  }
  if (dp2 < 0.0)
  {
    if (dq2 < 0.0) return check_min_max(p1, q1, r1, r2, p2, q2);
    if (dr2 < 0.0) return check_min_max(p1, q1, r1, q2, r2, p2);
    return check_min_max(p1, r1, q1, p2, q2, r2);
  }

  if (dq2 < 0.0)
  {
    if (dr2 >= 0.0) return check_min_max(p1, r1, q1, q2, r2, p2);
    return check_min_max(p1, q1, r1, p2, q2, r2);
  }
  if (dq2 > 0.0)
  {
    if (dr2 > 0.0) return check_min_max(p1, r1, q1, p2, q2, r2);
    return check_min_max(p1, q1, r1, q2, r2, p2);
  }

  if (dr2 > 0.0) return check_min_max(p1, q1, r1, r2, p2, q2);
  MFTL_ASSERT(dr2 < 0.0 && "tri_tri_3d: a flat pair reached Guigue-Devillers");
  return check_min_max(p1, r1, q1, r2, p2, q2);
}

/**
 * @brief Are the three values all strictly positive or all strictly negative?
 *
 * Compares signs, not products: the product of two orientations of a small
 * feature underflows to zero (and would merely skip the early rejection), but a
 * sign never does.
 *
 * @param a First value.
 * @param b Second value.
 * @param c Third value.
 * @return 1 if the three share one strict sign.
 */
static inline int same_strict_sign(double a, double b, double c) { return (a > 0.0 && b > 0.0 && c > 0.0) || (a < 0.0 && b < 0.0 && c < 0.0); }

/* Section 4: a pair Guigue-Devillers cannot take. */
static int tri_tri_flat(const double A[3][3], const double B[3][3], int a_in_plane_of_b, int b_in_plane_of_a);

int MFTL_TriTriIntersect(const double A[3][3], const double B[3][3])
{
  const double *p1 = A[0], *q1 = A[1], *r1 = A[2];
  const double *p2 = B[0], *q2 = B[1], *r2 = B[2];

  /* A's corners against B's plane; all strictly on one side -> disjoint (the
   * common case, 3 predicates). The QUERY POINT comes first: plane-first order
   * is an odd permutation of the four arguments and negates every sign, which
   * the early-outs survive but the dispatch does not
   * (test_tri_tri_intersect_basic_cases). */
  const double dp1 = MFTL_Orient3D(p1, p2, q2, r2);
  const double dq1 = MFTL_Orient3D(q1, p2, q2, r2);
  const double dr1 = MFTL_Orient3D(r1, p2, q2, r2);
  if (same_strict_sign(dp1, dq1, dr1)) return 0;

  const double dp2 = MFTL_Orient3D(p2, p1, q1, r1);
  const double dq2 = MFTL_Orient3D(q2, p1, q1, r1);
  const double dr2 = MFTL_Orient3D(r2, p1, q1, r1);
  if (same_strict_sign(dp2, dq2, dr2)) return 0;

  /* Three exact zeros: A lies in B's plane, or B has zero area (every point is
   * "in" the plane of a degenerate triangle). Either way Guigue-Devillers, which
   * needs two proper triangles in distinct planes, does not apply. */
  const int a_in_b = (dp1 == 0.0 && dq1 == 0.0 && dr1 == 0.0);
  const int b_in_a = (dp2 == 0.0 && dq2 == 0.0 && dr2 == 0.0);
  if (a_in_b || b_in_a) return tri_tri_flat(A, B, a_in_b, b_in_a);

  /* Permute A so that its isolated vertex (the one alone on its side of B's
   * plane) comes first, carrying B's permutation along to keep the orientation
   * consistent. */
  if (dp1 > 0.0)
  {
    if (dq1 > 0.0) return tri_tri_3d(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2);
    if (dr1 > 0.0) return tri_tri_3d(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2);
    return tri_tri_3d(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2);
  }
  if (dp1 < 0.0)
  {
    if (dq1 < 0.0) return tri_tri_3d(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2);
    if (dr1 < 0.0) return tri_tri_3d(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2);
    return tri_tri_3d(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2);
  }

  if (dq1 < 0.0)
  {
    if (dr1 >= 0.0) return tri_tri_3d(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2);
    return tri_tri_3d(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2);
  }
  if (dq1 > 0.0)
  {
    if (dr1 > 0.0) return tri_tri_3d(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2);
    return tri_tri_3d(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2);
  }

  if (dr1 > 0.0) return tri_tri_3d(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2);
  MFTL_ASSERT(dr1 < 0.0 && "MFTL_TriTriIntersect: a flat pair reached the dispatch");
  return tri_tri_3d(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Flat pairs: coplanar triangles, and triangles of zero area.
 *
 * Every decision is an exact sign. In 2D that is orient2d on the projected
 * points, evaluated as MFTL_Orient3D of the points lifted to z = 0 against a
 * point at z = 1 -- the lifted determinant IS -orient2d, so its sign is exact
 * and no second adaptive predicate is needed. A plain 2D determinant reported
 * two disjoint coplanar triangles as touching when a small orientation rounded
 * to zero (test_coplanar_triangles_are_decided_exactly).
 *
 * A triangle of zero area is the segment (or point) its corners span, and is
 * tested as one: against a proper triangle as a segment crossing it, against
 * another degenerate one as two segments, which meet only if coplanar. Treating
 * it as a triangle with a plane reported a point far away as intersecting, and
 * two collinear triangles a unit apart as touching
 * (test_zero_area_triangles_are_segments).
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Twice the signed area of (a, b, c) in the plane, with an EXACT sign.
 *
 * @param a First point.
 * @param b Second point.
 * @param c Third point.
 * @return \f$(\vect{b}-\vect{a})\times(\vect{c}-\vect{a})\f$, positive when
 *         a, b, c are counter-clockwise; the sign is exact.
 */
static double orient2d(const double a[2], const double b[2], const double c[2])
{
  const double pa[3] = {a[0], a[1], 0.0}, pb[3] = {b[0], b[1], 0.0}, pc[3] = {c[0], c[1], 0.0};
  const double pd[3] = {c[0], c[1], 1.0};
  return -MFTL_Orient3D(pa, pb, pc, pd);
}

/**
 * @brief Is @p p inside the axis-aligned box of @p a and @p b (closed)?
 *
 * For a point already known to lie on the line through @p a and @p b this is
 * exactly "on the closed segment", and it is a pure comparison.
 *
 * @param p The point.
 * @param a First endpoint.
 * @param b Second endpoint.
 * @return 1 if inside.
 */
static inline int in_box_2d(const double p[2], const double a[2], const double b[2])
{
  return p[0] >= fmin(a[0], b[0]) && p[0] <= fmax(a[0], b[0]) && p[1] >= fmin(a[1], b[1]) && p[1] <= fmax(a[1], b[1]);
}

/** @brief Strictly opposite signs. @param a First value. @param b Second value. @return 1 if one is > 0 and the other < 0. */
static inline int strictly_opposite(double a, double b) { return (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0); }

/**
 * @brief Do the convex hulls of two corner triples in 2D intersect? Either hull
 *        may be a segment or a point.
 *
 * Two closed convex sets meet iff their boundaries do or one contains the
 * other. The nine segment pairs cover the boundaries (and the whole of a
 * degenerate hull, which is its own boundary); containment is possible only in
 * a hull of positive area, so a corner of the other is tested there. The 18
 * edge orientations are shared by the segment tests and the containment tests.
 *
 * @param      P      Corners of the first, projected.
 * @param      Q      Corners of the second, projected.
 * @param[out] proper 1 if either projected triple has non-zero area, i.e. the
 *                    projection is one-to-one on a common plane.
 * @return 1 if the hulls intersect.
 */
static int tri_tri_2d(const double P[3][2], const double Q[3][2], int *proper)
{
  double oP[3][3], oQ[3][3];  // oP[i][j]: Q_j against edge i of P; oQ[j][i]: P_i against edge j of Q
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      oP[i][j] = orient2d(P[i], P[(i + 1) % 3], Q[j]);
      oQ[i][j] = orient2d(Q[i], Q[(i + 1) % 3], P[j]);
    }
  const int area_p = (orient2d(P[0], P[1], P[2]) != 0.0);
  const int area_q = (orient2d(Q[0], Q[1], Q[2]) != 0.0);
  *proper = area_p || area_q;

  for (int i = 0; i < 3; ++i)
  {
    const int i1 = (i + 1) % 3;
    for (int j = 0; j < 3; ++j)
    {
      const int j1 = (j + 1) % 3;
      const double d1 = oQ[j][i], d2 = oQ[j][i1];  // P's edge ends against Q's edge line
      const double d3 = oP[i][j], d4 = oP[i][j1];  // Q's edge ends against P's edge line
      if (strictly_opposite(d1, d2) && strictly_opposite(d3, d4)) return 1;
      /* An endpoint ON the other segment: a zero orientation and the box. This
       * also decides collinear overlap and zero-length segments. */
      if (d1 == 0.0 && in_box_2d(P[i], Q[j], Q[j1])) return 1;
      if (d2 == 0.0 && in_box_2d(P[i1], Q[j], Q[j1])) return 1;
      if (d3 == 0.0 && in_box_2d(Q[j], P[i], P[i1])) return 1;
      if (d4 == 0.0 && in_box_2d(Q[j1], P[i], P[i1])) return 1;
    }
  }

  /* No boundaries meet: disjoint, or one strictly inside the other. */
  if (area_p && !(fmin(oP[0][0], fmin(oP[1][0], oP[2][0])) < 0.0 && fmax(oP[0][0], fmax(oP[1][0], oP[2][0])) > 0.0)) return 1;
  if (area_q && !(fmin(oQ[0][0], fmin(oQ[1][0], oQ[2][0])) < 0.0 && fmax(oQ[0][0], fmax(oQ[1][0], oQ[2][0])) > 0.0)) return 1;
  return 0;
}

/**
 * @brief Drops axis @p k: copies the other two coordinates of each corner.
 *
 * @param      T   Corners in 3D.
 * @param      k   The axis dropped.
 * @param[out] out The projected corners.
 */
static void project_dropping(const double T[3][3], int k, double out[3][2])
{
  const int i0 = (k == 0) ? 1 : 0;
  const int i1 = (k == 2) ? 1 : 2;
  for (int c = 0; c < 3; ++c)
  {
    out[c][0] = T[c][i0];
    out[c][1] = T[c][i1];
  }
}

/**
 * @brief Intersection of two triangles whose six corners are COPLANAR (either
 *        may have zero area).
 *
 * A projection along a coordinate axis preserves intersection; it also
 * preserves disjointness when it is one-to-one on the common plane, which is
 * certified exactly by a projected triangle of non-zero area. Some axis is
 * one-to-one on the plane (or on the line, when everything is collinear), so:
 * a disjoint projection proves disjoint, a certified one proves intersecting,
 * and three intersecting uncertified projections prove intersecting. The axis
 * the larger normal is most aligned with is tried first, so a proper pair is
 * one projection.
 *
 * @param A Corners of the first triangle.
 * @param B Corners of the second; coplanar with @p A.
 * @return 1 if they intersect.
 */
static int coplanar_tri_tri(const double A[3][3], const double B[3][3])
{
  double nA[3], nB[3];
  for (int s = 0; s < 2; ++s)
  {
    const double(*T)[3] = (s == 0) ? A : B;
    double *const n = (s == 0) ? nA : nB;
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    n[0] = fabs(e1[1] * e2[2] - e1[2] * e2[1]);
    n[1] = fabs(e1[2] * e2[0] - e1[0] * e2[2]);
    n[2] = fabs(e1[0] * e2[1] - e1[1] * e2[0]);
  }
  const double *const n = (nA[0] + nA[1] + nA[2] >= nB[0] + nB[1] + nB[2]) ? nA : nB;
  int first = 0;
  if (n[1] > n[first]) first = 1;
  if (n[2] > n[first]) first = 2;

  for (int step = 0; step < 3; ++step)
  {
    const int k = (first + step) % 3;
    double PA[3][2], PB[3][2];
    project_dropping(A, k, PA);
    project_dropping(B, k, PB);
    int proper = 0;
    if (!tri_tri_2d(PA, PB, &proper)) return 0;
    if (proper) return 1;
  }
  return 1;
}

/**
 * @brief Are the three corners exactly collinear (a zero-area triangle)?
 *
 * Three points are collinear iff all three coordinate projections have zero
 * area: a plane through them has a normal with some non-zero component, and
 * the projection along that axis keeps a non-zero area.
 *
 * @param T The corners.
 * @return 1 if collinear (including coincident corners).
 */
static int is_collinear(const double T[3][3])
{
  for (int k = 0; k < 3; ++k)
  {
    double P[3][2];
    project_dropping(T, k, P);
    if (orient2d(P[0], P[1], P[2]) != 0.0) return 0;
  }
  return 1;
}

/**
 * @brief The two corners spanning a collinear triple.
 *
 * Along an axis on which the corners are not all equal, the coordinate is an
 * injective affine function of the position on their line, so the extremes
 * there are the hull's endpoints (exact comparisons). All three coincide when
 * no such axis exists.
 *
 * @param      T  Collinear corners.
 * @param[out] lo Index of one end.
 * @param[out] hi Index of the other (equal to @p lo for a point).
 */
static void hull_ends(const double T[3][3], int *lo, int *hi)
{
  *lo = *hi = 0;
  for (int k = 0; k < 3; ++k)
  {
    if (T[0][k] == T[1][k] && T[1][k] == T[2][k]) continue;
    for (int c = 1; c < 3; ++c)
    {
      if (T[c][k] < T[*lo][k]) *lo = c;
      if (T[c][k] > T[*hi][k]) *hi = c;
    }
    return;
  }
}

/**
 * @brief Does the segment [s0, s1] meet the proper triangle @p T, given that it
 *        is not in T's plane and its ends are not strictly on one side of it?
 *
 * The segment then meets the plane in one point X. For any point Y on the
 * segment's line, orient3d(t_i, t_i+1, s1, s0) = det[s1 - s0, t_i - Y, t_i+1 - Y]
 * does not depend on Y; at Y = X it is (s1 - s0).n times the in-plane
 * orientation of X against edge i. X is in the closed triangle iff those three
 * signs are not mixed.
 *
 * @param s0 One end of the segment.
 * @param s1 The other end (distinct from @p s0).
 * @param T  The triangle, of non-zero area.
 * @return 1 if they intersect.
 */
static int segment_meets_triangle(const double s0[3], const double s1[3], const double T[3][3])
{
  int pos = 0, neg = 0;
  for (int i = 0; i < 3; ++i)
  {
    const double o = MFTL_Orient3D(T[i], T[(i + 1) % 3], s1, s0);
    pos |= (o > 0.0);
    neg |= (o < 0.0);
  }
  return !(pos && neg);
}

/**
 * @brief Intersection for a pair Guigue-Devillers cannot take: one triangle's
 *        corners all in the other's plane, or a triangle of zero area.
 *
 * @param A               Corners of the first triangle.
 * @param B               Corners of the second.
 * @param a_in_plane_of_b All of A's corners have a zero orientation against B.
 * @param b_in_plane_of_a All of B's corners have a zero orientation against A.
 * @return 1 if the closed triangles (or the segments and points they reduce to)
 *         intersect.
 */
static int tri_tri_flat(const double A[3][3], const double B[3][3], int a_in_plane_of_b, int b_in_plane_of_a)
{
  const int a_flat = is_collinear(A);
  const int b_flat = is_collinear(B);

  /* Two proper triangles here are coplanar: one's corners lie in the other's
   * plane, which is then the plane of both. */
  if (!a_flat && !b_flat) return coplanar_tri_tri(A, B);

  if (a_flat && b_flat)
  {
    /* Two segments (or points) meet only if coplanar; a point and a segment
     * always are, and the orientation is then exactly zero. */
    int a0, a1, b0, b1;
    hull_ends(A, &a0, &a1);
    hull_ends(B, &b0, &b1);
    if (MFTL_Orient3D(A[a0], A[a1], B[b0], B[b1]) != 0.0) return 0;
    return coplanar_tri_tri(A, B);
  }

  /* One proper triangle T and one segment S. S in T's plane is the coplanar
   * case; otherwise S crosses or touches the plane (the early rejection ruled
   * out both ends strictly on one side, and the middle corner lies between). */
  const double(*const T)[3] = a_flat ? B : A;
  const double(*const S)[3] = a_flat ? A : B;
  const int s_in_plane = a_flat ? a_in_plane_of_b : b_in_plane_of_a;
  if (s_in_plane) return coplanar_tri_tri(A, B);
  int s0, s1;
  hull_ends(S, &s0, &s1);
  MFTL_ASSERT(s0 != s1 && "tri_tri_flat: a point off the plane escaped the early rejection");
  return segment_meets_triangle(S[s0], S[s1], T);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Distance between two DISJOINT closed triangles: the minimum is attained
 * between two edges or between a vertex and the interior of the other face, so
 * nine segment pairs and six face projections are a complete cover. One
 * implementation, MFTLClosestPointsBetweenTriangles() (src/geometry/), which
 * topology detection also uses for its witnesses: it is accurate to rounding at
 * every angle and aspect ratio (src/geometry/README.md, "Closest points").
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTL_TriTriDistanceSq(const double A[3][3], const double B[3][3])
{
  /* An edge piercing the other face has positive distance to every edge and
   * vertex, so only the intersection predicate sees it: it runs first. */
  if (MFTL_TriTriIntersect(A, B)) return 0.0;
  double pa[3], pb[3];
  return MFTLClosestPointsBetweenTriangles(A, B, pa, pb);
}

int MFTL_TriTriProximity(const double A[3][3], const double B[3][3], double tolerance)
{
  if (MFTL_TriTriIntersect(A, B)) return 1;
  if (tolerance <= 0.0) return 0;
  double pa[3], pb[3];
  return (MFTLClosestPointsBetweenTriangles(A, B, pa, pb) <= tolerance * tolerance) ? 1 : 0;
}

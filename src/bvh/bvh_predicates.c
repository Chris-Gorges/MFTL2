/**
 * @file bvh_predicates.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief The narrow phase: exact orientation, triangle-triangle intersection,
 *        and exact triangle-triangle distance.
 *
 * Everything in this file is DOUBLE and stays double. The broad phase may round
 * outward and be loose; this may not be wrong at all, because it is the last
 * thing that runs before a contact force is applied.
 *
 * ## WHY THE ORIENTATION TEST IS ADAPTIVE AND NOTHING ELSE HERE IS
 *
 * MFTL_Orient3D is the only predicate in the module whose SIGN decides the
 * answer. Guigue-Devillers evaluates it up to twelve times per triangle pair and
 * branches on each result; every other quantity here -- distances, areas,
 * interval endpoints -- degrades continuously, so a last-ulp error moves a
 * contact decision only for pairs sitting exactly on the tolerance, which is a
 * tie the caller accepted when it chose a tolerance.
 *
 * A wrong SIGN is different in kind. It sends the algorithm down a branch that
 * describes a configuration which does not exist, and the usual outcome is a
 * clean "no intersection" reported for two triangles that plainly cross.
 *
 * ### The failure is not hypothetical, and it is aimed at this library
 *
 * The determinant is a difference of products of coordinate DIFFERENCES. Its
 * condition number is the ratio of the terms' magnitude to the result's, and for
 * a nearly coplanar tetrahedron that ratio is unbounded. Two configurations this
 * code exists to handle sit exactly there:
 *
 *   THIN FILM / LUBRICATION. Two surfaces separated by a gap orders of magnitude
 *   below their own size -- a draining film between coalescing droplets. The four
 *   points of the predicate are then coplanar to within the film thickness.
 *
 *   LARGE COORDINATES, SMALL FEATURES. A millimetre droplet at x = 0.5 m spends
 *   most of its mantissa on the OFFSET. The differences taken inside the
 *   determinant cancel that offset, and the cancellation is where the digits go:
 *   the same droplet at the origin is well conditioned, and translating it is not
 *   supposed to change whether two triangles touch.
 *
 * ### What was actually measured, and what it says
 *
 * examples/orient3d_predicate_profile.c, Apple silicon, Release, best of five.
 * The adaptive form is the reference, its sign being exact by construction, so
 * any disagreement IS a misclassification by the plain determinant:
 *
 *   MISCLASSIFICATION vs GAP  (triangle size L, general-orientation plane)
 *
 *       gap/L >= 1e-12     0.000 %   of 20 000 samples, at EVERY offset tried
 *       gap/L  = 1e-14     0.000 - 0.035 %
 *       gap/L  = 1e-15     0.000 - 0.125 %
 *       gap/L  = 1e-16     0.135 - 4.745 %
 *
 *   COST                      general position    exactly coplanar
 *       plain determinant        1.83 ns              1.83 ns
 *       adaptive                 3.66 ns             40.13 ns
 *                               (2.00x)             (21.9x)
 *
 * The honest reading is that THE PLAIN DETERMINANT IS FINE FOR THE PHYSICS AS
 * STATED. A lubrication or thin-film model works at gaps of 1e-6 to 1e-9 of the
 * body size, and at 1e-12 -- three orders below that -- the plain form did not
 * misclassify once in 20 000 samples. The failures only start within about two
 * orders of magnitude of the double epsilon itself.
 *
 * The adaptive form is kept anyway, on the cost side of the argument rather than
 * the accuracy side:
 *
 *   - the 2x is on the PREDICATE, not on the query. A full
 *     MFTL_TriTriProximity call costs 176 ns, of which six orientation tests at
 *     1.83 ns of extra each is AT MOST 6.2 %. The narrow phase is dominated by
 *     the fifteen distance sub-tests, not by the determinant;
 *   - the 21.9x worst case needs EXACT coplanarity, which moving geometry
 *     essentially never produces. It is the price of a constructed test mesh,
 *     not of a timestep;
 *   - the failure it removes is silent and unbounded: a clean "no intersection"
 *     for two triangles that cross, from which a contact force is then never
 *     applied. There is no error term to bound it with and nothing downstream
 *     that can notice.
 *
 * 6.2 % of the narrow phase to make a whole class of silent wrong answers
 * unreachable is the same trade CLAUDE.md records for the unit-normal guard
 * (+1.1 % to remove a silently short normal) and it is taken the same way.
 * REVISIT IT if a profile ever shows the narrow phase dominating a timestep --
 * the plain form is one call away in MFTL_Orient3DFast, and the measurement to
 * repeat is in the example.
 *
 * ### Why the cost is as low as it is
 *
 * Shewchuk's construction is a filter, not a slow path: the plain determinant
 * plus a comparison against a rigorous relative error bound, and exact expansion
 * arithmetic ONLY when the estimate falls inside that bound. Well-conditioned
 * input -- which is every pair the broad phase rejects and most of what it does
 * not -- pays one extra compare and nothing else.
 */

/* The exact expansion arithmetic below depends on a*b being computed with a
 * single rounding, so that Two_Product's error term is the true residue. A fused
 * multiply-add contracts `x - a*b` into one operation with one rounding, which
 * makes that residue come out zero and silently destroys every guarantee in this
 * file. The pragma is the portable statement of that requirement; CMake also sets
 * -ffp-contract=off on this source, because not every compiler honours it. */
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 12)
#pragma STDC FP_CONTRACT OFF
#endif

#include "mftl_bvh.h"

#include <math.h>
#include <stddef.h>

#include "mftl_constants.h"

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
 * 3. Guigue-Devillers triangle-triangle intersection
 *
 * The published decision tree, expressed over MFTL_Orient3D so that every branch
 * it takes is one the exact predicate agreed to. The previous version's
 * hand-rolled interval permutation is replaced by it for two reasons: the
 * permutation there did not normalise the ORIENTATION of the second triangle
 * (only which vertex was isolated), and its coplanar case returned "intersect"
 * unconditionally -- so two coplanar triangles a metre apart reported contact.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The terminal test of the decision tree.
 *
 * Both triangles have been permuted so that the intersection of each with the
 * other's plane is an interval on a common line, with a known orientation. Two
 * orientation tests then decide whether those intervals overlap. Expressed as
 * determinants of the original points rather than as explicit interval
 * endpoints, which is the whole point of the method: no division, no
 * intersection point is ever constructed, and therefore nothing to be
 * ill-conditioned.
 *
 * @brief The Guigue-Devillers decision tree, once the vertex permutation is fixed.
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

/* Forward declaration: the coplanar case is a 2D problem and lives below. */
static int coplanar_tri_tri(const double *p1, const double *q1, const double *r1, const double *p2, const double *q2, const double *r2, const double normal[3]);

/**
 * @brief Second half of the dispatch: permute B by the signs of its vertices
 *        against A's plane, then run the terminal test.
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
 * @param normal Normal of the first triangle, for the coplanar fallback.
 * @return 1 on intersection, 0 otherwise.
 */
static int tri_tri_3d(const double *p1, const double *q1, const double *r1, const double *p2, const double *q2, const double *r2, double dp2, double dq2,
                      double dr2, const double normal[3])
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
  if (dr2 < 0.0) return check_min_max(p1, r1, q1, r2, p2, q2);

  return coplanar_tri_tri(p1, q1, r1, p2, q2, r2, normal);
}

int MFTL_TriTriIntersect(const double A[3][3], const double B[3][3])
{
  const double *p1 = A[0], *q1 = A[1], *r1 = A[2];
  const double *p2 = B[0], *q2 = B[1], *r2 = B[2];

  /* Signed distances of A's vertices to B's plane. If all three are strictly on
   * the same side there is nothing to test -- and this is the branch the great
   * majority of leaf pairs take, so it comes first.
   *
   * ── The argument order is load-bearing ──────────────────────────────────
   *
   * The QUERY POINT COMES FIRST and the plane's three vertices follow, because
   * that is the convention the decision tree below and check_min_max were
   * derived under. Writing it the other way round -- plane first, point last,
   * which reads more naturally -- is a CYCLIC permutation of four arguments,
   * and a 4-cycle is an ODD permutation, so it negates every one of these six
   * values.
   *
   * The early-out tests survive that (they compare products, which are
   * sign-invariant), and so does any picture: the sign flip is invisible until
   * the dispatch below branches on an individual sign and takes the mirror
   * branch of the tree. The observable symptom is a clean "no intersection" for
   * a triangle pair that visibly crosses. Guarded by
   * test_tri_tri_intersect_basic_cases. */
  const double dp1 = MFTL_Orient3D(p1, p2, q2, r2);
  const double dq1 = MFTL_Orient3D(q1, p2, q2, r2);
  const double dr1 = MFTL_Orient3D(r1, p2, q2, r2);
  if (((dp1 * dq1) > 0.0) && ((dp1 * dr1) > 0.0)) return 0;

  const double dp2 = MFTL_Orient3D(p2, p1, q1, r1);
  const double dq2 = MFTL_Orient3D(q2, p1, q1, r1);
  const double dr2 = MFTL_Orient3D(r2, p1, q1, r1);
  if (((dp2 * dq2) > 0.0) && ((dp2 * dr2) > 0.0)) return 0;

  /* A's normal, needed only if the pair turns out coplanar; computed here
   * because the coplanar branch is reached from six places. */
  double n1[3];
  {
    const double e1[3] = {q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2]};
    const double e2[3] = {r1[0] - p1[0], r1[1] - p1[1], r1[2] - p1[2]};
    n1[0] = e1[1] * e2[2] - e1[2] * e2[1];
    n1[1] = e1[2] * e2[0] - e1[0] * e2[2];
    n1[2] = e1[0] * e2[1] - e1[1] * e2[0];
  }

  /* Permute A so that its isolated vertex (the one alone on its side of B's
   * plane) comes first, carrying B's permutation along to keep the orientation
   * consistent. */
  if (dp1 > 0.0)
  {
    if (dq1 > 0.0) return tri_tri_3d(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2, n1);
    if (dr1 > 0.0) return tri_tri_3d(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2, n1);
    return tri_tri_3d(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2, n1);
  }
  if (dp1 < 0.0)
  {
    if (dq1 < 0.0) return tri_tri_3d(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2, n1);
    if (dr1 < 0.0) return tri_tri_3d(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2, n1);
    return tri_tri_3d(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2, n1);
  }

  if (dq1 < 0.0)
  {
    if (dr1 >= 0.0) return tri_tri_3d(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2, n1);
    return tri_tri_3d(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2, n1);
  }
  if (dq1 > 0.0)
  {
    if (dr1 > 0.0) return tri_tri_3d(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2, n1);
    return tri_tri_3d(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2, n1);
  }

  if (dr1 > 0.0) return tri_tri_3d(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2, n1);
  if (dr1 < 0.0) return tri_tri_3d(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2, n1);

  return coplanar_tri_tri(p1, q1, r1, p2, q2, r2, n1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The coplanar case
 *
 * Reached only when all six orientation tests returned exactly zero, i.e. when
 * the two triangles are coplanar to the last bit of the EXACT predicate. That is
 * a measure-zero configuration in a moving simulation and a common one in a
 * constructed test, which is the split this implementation is sized for: correct
 * and simple rather than fast.
 *
 * Projected onto the coordinate plane the normal is most aligned with -- the
 * projection with the largest Jacobian, so the flattest of the three and the
 * best conditioned. Then a complete 2D overlap test: nine segment crossings
 * catch every boundary intersection, and two containment tests catch the case of
 * one triangle strictly inside the other, which has no crossing at all.
 *
 * The 2D orientation here is a plain determinant, not the adaptive predicate.
 * That is a deliberate stopping point: this branch already required an exact
 * coplanarity verdict to be reached, and the production predicate the traversal
 * uses is MFTL_TriTriProximity, whose distance test resolves a coplanar pair
 * within the tolerance regardless of what the overlap test decides.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Twice the signed area of (a, b, c) in the projected plane.
 *
 * @param a First point.
 * @param b Second point.
 * @param c Third point.
 * @return Twice the signed area of the triangle (a, b, c).
 */
static inline double orient2d(const double a[2], const double b[2], const double c[2])
{
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/**
 * @brief True if the closed segments p1p2 and q1q2 share at least one point.
 *
 * @brief Do the two closed 2D segments cross?
 * @param p1 First endpoint of segment 1.
 * @param p2 Second endpoint of segment 1.
 * @param q1 First endpoint of segment 2.
 * @param q2 Second endpoint of segment 2.
 * @return 1 if they intersect, 0 otherwise.
 */
static int seg_seg_2d(const double p1[2], const double p2[2], const double q1[2], const double q2[2])
{
  const double d1 = orient2d(q1, q2, p1);
  const double d2 = orient2d(q1, q2, p2);
  const double d3 = orient2d(p1, p2, q1);
  const double d4 = orient2d(p1, p2, q2);

  if (((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))) return 1;

  /* Collinear touching: a zero orientation plus containment in the bounding
   * interval. Checked per endpoint rather than by a general overlap formula,
   * because a degenerate (zero-length) segment must still be handled. */
  const double *pts[4] = {p1, p2, q1, q2};
  const double dd[4] = {d1, d2, d3, d4};
  const double *seg_a[4] = {q1, q1, p1, p1};
  const double *seg_b[4] = {q2, q2, p2, p2};

  for (int i = 0; i < 4; ++i)
  {
    if (dd[i] != 0.0) continue;
    const double lo0 = (seg_a[i][0] < seg_b[i][0]) ? seg_a[i][0] : seg_b[i][0];
    const double hi0 = (seg_a[i][0] < seg_b[i][0]) ? seg_b[i][0] : seg_a[i][0];
    const double lo1 = (seg_a[i][1] < seg_b[i][1]) ? seg_a[i][1] : seg_b[i][1];
    const double hi1 = (seg_a[i][1] < seg_b[i][1]) ? seg_b[i][1] : seg_a[i][1];
    if (pts[i][0] >= lo0 && pts[i][0] <= hi0 && pts[i][1] >= lo1 && pts[i][1] <= hi1) return 1;
  }
  return 0;
}

/**
 * @brief True if @p p lies in the closed triangle (a, b, c), either winding.
 *
 * @brief Is @p p inside the closed 2D triangle (a, b, c)?
 * @param p The query point.
 * @param a First corner.
 * @param b Second corner.
 * @param c Third corner.
 * @return 1 if inside or on the boundary, 0 otherwise.
 */
static int point_in_tri_2d(const double p[2], const double a[2], const double b[2], const double c[2])
{
  const double d1 = orient2d(a, b, p);
  const double d2 = orient2d(b, c, p);
  const double d3 = orient2d(c, a, p);

  const int has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
  const int has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
  return !(has_neg && has_pos);
}

/**
 * @brief 2D overlap test for two triangles already known to be COPLANAR.
 *
 * Projected onto the coordinate plane @p normal is most aligned with -- the
 * projection with the largest Jacobian, so the flattest of the three and the
 * best conditioned. Then nine segment-crossing tests catch every boundary
 * intersection, and two containment tests catch one triangle lying strictly
 * inside the other, which has no crossing at all.
 *
 * @param p1     First triangle's first corner.
 * @param q1     First triangle's second corner.
 * @param r1     First triangle's third corner.
 * @param p2     Second triangle's first corner.
 * @param q2     Second triangle's second corner.
 * @param r2     Second triangle's third corner.
 * @param normal Normal of the common plane; only its dominant axis is used.
 * @return 1 if the two coplanar triangles overlap, 0 otherwise.
 */
static int coplanar_tri_tri(const double *p1, const double *q1, const double *r1, const double *p2, const double *q2, const double *r2, const double normal[3])
{
  /* Drop the axis the normal is most aligned with; the remaining two carry the
   * best-conditioned projection. */
  const double ax = fabs(normal[0]), ay = fabs(normal[1]), az = fabs(normal[2]);
  int i0, i1;
  if (ax > ay)
  {
    if (ax > az)
    {
      i0 = 1;
      i1 = 2;
    }
    else
    {
      i0 = 0;
      i1 = 1;
    }
  }
  else
  {
    if (ay > az)
    {
      i0 = 0;
      i1 = 2;
    }
    else
    {
      i0 = 0;
      i1 = 1;
    }
  }

  const double P1[2] = {p1[i0], p1[i1]}, Q1[2] = {q1[i0], q1[i1]}, R1[2] = {r1[i0], r1[i1]};
  const double P2[2] = {p2[i0], p2[i1]}, Q2[2] = {q2[i0], q2[i1]}, R2[2] = {r2[i0], r2[i1]};

  const double *TA[3] = {P1, Q1, R1};
  const double *TB[3] = {P2, Q2, R2};

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (seg_seg_2d(TA[i], TA[(i + 1) % 3], TB[j], TB[(j + 1) % 3])) return 1;

  if (point_in_tri_2d(P1, P2, Q2, R2)) return 1;
  if (point_in_tri_2d(P2, P1, Q1, R1)) return 1;

  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Exact distance between two DISJOINT triangles
 *
 * The minimum distance between two disjoint closed triangles is attained either
 * between two edges or between a vertex and the interior of the other face.
 * Nine edge pairs and six point-face projections is a complete cover of both, so
 * the minimum over the fifteen is exact -- no iteration, no tolerance.
 *
 * ── The scale bug this replaces ────────────────────────────────────────────
 *
 * The previous version guarded both routines with ABSOLUTE thresholds on
 * quantities that do not have units of length. `det < 1e-12` in the point-face
 * routine tests a Gram determinant, which has units of length^4: for a droplet
 * meshed at 20 um the determinant of a healthy triangle is around 1e-19, so the
 * guard fired on EVERY triangle and the point-face branch returned "no
 * contribution" every time. The distance was then whatever the edge-edge tests
 * found, which misses exactly the vertex-into-face configuration that a resolved
 * contact consists of -- silently, and only at small scales.
 *
 * The tests here are dimensionless: `det <= tiny * a * c` compares the Gram
 * determinant against the product of the two edge norms it is built from, which
 * is the only formulation that means the same thing at 1e-5 m and at 1 m.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Dot product of two 3-vectors.
 * @param u First vector.
 * @param v Second vector.
 */
#define MFTL_BVH_DOT(u, v) ((u)[0] * (v)[0] + (u)[1] * (v)[1] + (u)[2] * (v)[2])

/**
 * @brief Componentwise difference, @p dest = @p v1 - @p v2.
 * @param[out] dest Result. May alias neither input.
 * @param      v1   Minuend.
 * @param      v2   Subtrahend.
 */
#define MFTL_BVH_SUB(dest, v1, v2)   \
  do                                 \
  {                                  \
    (dest)[0] = (v1)[0] - (v2)[0];   \
    (dest)[1] = (v1)[1] - (v2)[1];   \
    (dest)[2] = (v1)[2] - (v2)[2];   \
  } while (0)

/**
 * @brief Squared distance between the closed segments P0P1 and Q0Q1.
 *
 * The standard clamped-parameter solution. `D <= tiny * a * c` is the
 * dimensionless test for parallel segments: D is the Gram determinant
 * a*c - b^2 and a*c bounds it, so their ratio is the sine squared of the angle
 * between the two directions and is scale-free.
 *
 * @param P0 First endpoint of segment P.
 * @param P1 Second endpoint of segment P.
 * @param Q0 First endpoint of segment Q.
 * @param Q1 Second endpoint of segment Q.
 * @return Squared minimum distance between the two closed segments.
 */
static double edge_edge_dist_sq(const double P0[3], const double P1[3], const double Q0[3], const double Q1[3])
{
  double u[3], v[3], w[3];
  MFTL_BVH_SUB(u, P1, P0);
  MFTL_BVH_SUB(v, Q1, Q0);
  MFTL_BVH_SUB(w, P0, Q0);

  const double a = MFTL_BVH_DOT(u, u);
  const double b = MFTL_BVH_DOT(u, v);
  const double c = MFTL_BVH_DOT(v, v);
  const double d = MFTL_BVH_DOT(u, w);
  const double e = MFTL_BVH_DOT(v, w);
  const double D = a * c - b * b;

  double sN, sD = D;
  double tN, tD = D;

  if (D <= MFTL_VERYVERYSMALL * a * c)
  {
    /* Parallel, or one of them degenerate. Pin s to the start of the first
     * segment and let the t clamp below find the foot of the perpendicular. */
    sN = 0.0;
    sD = 1.0;
    tN = e;
    tD = c;
  }
  else
  {
    sN = (b * e - c * d);
    tN = (a * e - b * d);
    if (sN < 0.0)
    {
      sN = 0.0;
      tN = e;
      tD = c;
    }
    else if (sN > sD)
    {
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0.0)
  {
    tN = 0.0;
    if (-d < 0.0)
      sN = 0.0;
    else if (-d > a)
      sN = sD;
    else
    {
      sN = -d;
      sD = a;
    }
  }
  else if (tN > tD)
  {
    tN = tD;
    if ((-d + b) < 0.0)
      sN = 0.0;
    else if ((-d + b) > a)
      sN = sD;
    else
    {
      sN = (-d + b);
      sD = a;
    }
  }

  /* A zero denominator here means a zero-length segment, so the parameter is
   * irrelevant and zero is as good as anything. */
  const double sc = (sD > 0.0) ? sN / sD : 0.0;
  const double tc = (tD > 0.0) ? tN / tD : 0.0;

  const double dP[3] = {w[0] + sc * u[0] - tc * v[0], w[1] + sc * u[1] - tc * v[1], w[2] + sc * u[2] - tc * v[2]};
  return MFTL_BVH_DOT(dP, dP);
}

/**
 * @brief Squared distance from @p P to the INTERIOR of triangle (A, B, C).
 *
 * Returns MFTL_DBL_MAX when the orthogonal projection falls outside the
 * triangle: the closest point is then on the boundary, which the edge-edge pass
 * covers exactly. Returning a sentinel rather than clamping keeps the two passes
 * from computing the same boundary answer twice.
 *
 * @param P Query point.
 * @param A First corner of the triangle.
 * @param B Second corner.
 * @param C Third corner.
 * @return Squared distance from @p P to the closed triangle.
 */
static double point_face_dist_sq(const double P[3], const double A[3], const double B[3], const double C[3])
{
  double edge0[3], edge1[3], v0[3];
  MFTL_BVH_SUB(edge0, B, A);
  MFTL_BVH_SUB(edge1, C, A);
  MFTL_BVH_SUB(v0, A, P);

  const double a = MFTL_BVH_DOT(edge0, edge0);
  const double b = MFTL_BVH_DOT(edge0, edge1);
  const double c = MFTL_BVH_DOT(edge1, edge1);
  const double d = MFTL_BVH_DOT(edge0, v0);
  const double e = MFTL_BVH_DOT(edge1, v0);

  const double det = a * c - b * b;

  /* Dimensionless degeneracy test: det/(a*c) is sin^2 of the corner angle. */
  if (det <= MFTL_VERYVERYSMALL * a * c) return MFTL_DBL_MAX;

  double s = b * e - c * d;
  double t = b * d - a * e;

  if (s >= 0.0 && t >= 0.0 && (s + t) <= det)
  {
    const double inv_det = 1.0 / det;
    s *= inv_det;
    t *= inv_det;
    const double diff[3] = {A[0] + s * edge0[0] + t * edge1[0] - P[0], A[1] + s * edge0[1] + t * edge1[1] - P[1], A[2] + s * edge0[2] + t * edge1[2] - P[2]};
    return MFTL_BVH_DOT(diff, diff);
  }

  return MFTL_DBL_MAX;
}

/**
 * @brief Minimum squared distance, assuming the two triangles do NOT intersect.
 *
 * @param A Corners of the first triangle.
 * @param B Corners of the second.
 * @return Squared minimum distance, valid only when the two do NOT intersect --
 *         the fifteen sub-tests cover every witness pair of two disjoint
 *         triangles: nine edge-edge and six point-face.
 */
static double tri_tri_dist_sq_disjoint(const double A[3][3], const double B[3][3])
{
  double min_sq = MFTL_DBL_MAX;

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      const double d = edge_edge_dist_sq(A[i], A[(i + 1) % 3], B[j], B[(j + 1) % 3]);
      if (d < min_sq) min_sq = d;
    }
  }

  for (int i = 0; i < 3; ++i)
  {
    const double da = point_face_dist_sq(A[i], B[0], B[1], B[2]);
    if (da < min_sq) min_sq = da;
    const double db = point_face_dist_sq(B[i], A[0], A[1], A[2]);
    if (db < min_sq) min_sq = db;
  }

  return min_sq;
}

double MFTL_TriTriDistanceSq(const double A[3][3], const double B[3][3])
{
  /* Intersecting triangles are at distance zero, and the fifteen sub-tests would
   * NOT find that: an edge piercing the interior of the other face has a
   * positive distance to every edge and to every vertex. The predicate is the
   * only thing that sees it, so it has to run first. */
  if (MFTL_TriTriIntersect(A, B)) return 0.0;
  return tri_tri_dist_sq_disjoint(A, B);
}

int MFTL_TriTriProximity(const double A[3][3], const double B[3][3], double tolerance)
{
  if (MFTL_TriTriIntersect(A, B)) return 1;
  if (tolerance <= 0.0) return 0;
  return (tri_tri_dist_sq_disjoint(A, B) <= tolerance * tolerance) ? 1 : 0;
}

/*
 * @brief Unit tests for mftl_vectoroperations.h (pure math, no MPI needed).
 *
 * ============================================================================
 * COVERAGE MAP -- what this file checks, function by function:
 * ============================================================================
 *
 * MFTLNormalizeVector
 *   - a simple axis vector normalizes to the corresponding unit axis
 *   - a general 3-4-0 vector normalizes to unit length with the expected
 *     components (3-4-5 right triangle)
 *   - the degeneracy guard keeps a zero vector finite (no NaN, no +/-inf)
 *     rather than dividing by zero, and yields exactly zero as a sentinel
 *   - THE RESULT IS UNIT LENGTH ACROSS 300 ORDERS OF MAGNITUDE of input scale,
 *     which is the property the whole function exists to provide and the one
 *     that was silently false below 1e-6 until 2026-08-15
 *
 * MFTLVectorInProd (dot product)
 *   - a hand-computed non-trivial case
 *   - two orthogonal unit axes dot to zero
 *   - a vector dotted with itself equals its squared norm (definitional
 *     identity, independent of the hand-computed case above)
 *
 * MFTLVectorCrossVector (cross product)
 *   - all three right-handed basis identities: x*y=z, y*z=x, z*x=y (not just
 *     one axis pair -- a sign error confined to a single component pair would
 *     otherwise slip through)
 *   - anti-commutativity: y*x = -(x*y)
 *   - the result is orthogonal to both inputs (dot product with each is zero)
 *   - two parallel vectors cross to the zero vector
 *
 * NOT covered here (by design): MFTLGetMinimumImageVector, MFTLApplyPeriodicity
 * and triangle-property math live in mftl_geometry.h and are exercised by
 * test_geometry.c instead, since they need a domain (periodicity) to be
 * meaningful.
 */

#include "test_utils.h"

#include "mftl_constants.h"
#include "mftl_vectoroperations.h"

static int test_normalize_unit_axis(void)
{
  double v[3] = {5.0, 0.0, 0.0};
  MFTLNormalizeVector(v);

  TEST_ASSERT_EQUAL_DOUBLE(1.0, v[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, v[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, v[2], MFTL_EPSILON);
  return 0;
}

static int test_normalize_general(void)
{
  // 3-4-0 has length 5
  double v[3] = {3.0, 4.0, 0.0};
  MFTLNormalizeVector(v);

  TEST_ASSERT_EQUAL_DOUBLE(0.6, v[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.8, v[1], MFTL_EPSILON);

  const double len = MFTL_SQRT(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, len, MFTL_EPSILON);
  return 0;
}

/* The MFTL_MAX(MFTL_VERYSMALL, ...) guard must keep a zero vector finite rather
 * than producing inf/NaN -- callers rely on being able to normalise blindly. */
static int test_normalize_zero_vector_is_finite(void)
{
  double v[3] = {0.0, 0.0, 0.0};
  MFTLNormalizeVector(v);

  for (int i = 0; i < 3; ++i)
  {
    TEST_ASSERT(v[i] == v[i]);  // not NaN
    TEST_ASSERT(v[i] < MFTL_DBL_MAX && v[i] > -MFTL_DBL_MAX);  // not +/-inf

    /* Exactly zero, which is the sentinel a caller can test against: a unit
     * vector never has zero length, so "not normalisable" stays distinguishable
     * from "normalised". */
    TEST_ASSERT_EQUAL_DOUBLE(0.0, v[i], 0.0);
  }
  return 0;
}

/**
 * @brief The output is UNIT LENGTH regardless of the input's scale.
 *
 * This is the function's entire contract, and it is worth a sweep rather than a
 * spot check because the way it fails is invisible. Until 2026-08-15 the guard
 * was applied to the SQUARED length before the square root, which floors the
 * magnitude at sqrt(MFTL_VERYSMALL) = 1e-6: any shorter vector was divided by the
 * clamp instead of by its own length and came out proportionally SHORT. A vector
 * of length 1e-9 normalised to length 1e-3.
 *
 * In computational physics that is the worst available failure mode. The
 * direction is right, so every sign test, every visual check and every
 * orthogonality assertion still passes -- and the magnitude silently scales
 * whatever the vector multiplies. A surface tension force built on it is wrong by
 * a factor of a thousand and looks completely reasonable.
 *
 * The sweep stops at 1e+-150 because the sum of squares is computed directly:
 * 14*(1e155)^2 overflows and 14*(1e-170)^2 underflows to zero, so outside roughly
 * (1e-162, 7e153) the naive norm loses the magnitude before the guard ever sees
 * it. That is a property of the formula, not of the guard -- avoiding it needs
 * hypot()-style scaling at several times the cost, which is not worth it for a
 * library whose coordinates are simulation-domain sized. Outside that range the
 * result is the zero sentinel rather than a wrong magnitude, which is the safe
 * direction to fail in.
 */
static int test_normalize_is_unit_across_magnitudes(void)
{
  for (int e = -150; e <= 150; e += 10)
  {
    const double s = pow(10.0, (double) e);

    /* An asymmetric direction, so a component-wise mistake cannot cancel. */
    double v[3] = {1.0 * s, -2.0 * s, 3.0 * s};
    MFTLNormalizeVector(v);

    const double len = MFTL_SQRT(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (fabs(len - 1.0) > 1e-12)
    {
      printf("\n  scale 1e%d -> length %.17g ", e, len);
      TEST_FAIL();
    }

    /* ...and still pointing the same way: direction and magnitude are separate
     * claims, and only one of them used to hold. */
    const double inv_sqrt14 = 1.0 / sqrt(14.0);
    TEST_ASSERT_EQUAL_DOUBLE(1.0 * inv_sqrt14, v[0], 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE(-2.0 * inv_sqrt14, v[1], 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE(3.0 * inv_sqrt14, v[2], 1e-12);
  }
  return 0;
}

static int test_dot_product(void)
{
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {4.0, -5.0, 6.0};

  // 4 - 10 + 18 = 12
  TEST_ASSERT_EQUAL_DOUBLE(12.0, MFTLVectorInProd(a, b), MFTL_EPSILON);
  return 0;
}

static int test_dot_product_orthogonal(void)
{
  double x[3] = {1.0, 0.0, 0.0};
  double y[3] = {0.0, 1.0, 0.0};

  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTLVectorInProd(x, y), MFTL_EPSILON);
  return 0;
}

/* v . v must equal the squared norm, independent of the hand-computed dot
 * product case above -- catches a regression that happens to cancel out on
 * mixed-sign inputs but not on a vector dotted with itself. */
static int test_dot_product_self_equals_norm_squared(void)
{
  double v[3] = {2.0, -3.0, 6.0};  // norm 7 (2-3-6-7 Pythagorean-like quadruple: 4+9+36=49)

  TEST_ASSERT_EQUAL_DOUBLE(49.0, MFTLVectorInProd(v, v), MFTL_EPSILON);
  return 0;
}

/* All three right-handed basis identities, not just x*y=z -- a sign error
 * confined to one component pair (e.g. only the z-component formula) would
 * otherwise slip through on a single-axis-pair test. */
static int test_cross_product_right_handed(void)
{
  double x[3] = {1.0, 0.0, 0.0};
  double y[3] = {0.0, 1.0, 0.0};
  double z[3] = {0.0, 0.0, 1.0};
  double r[3];

  MFTLVectorCrossVector(r, x, y);  // x * y == z
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, r[2], MFTL_EPSILON);

  MFTLVectorCrossVector(r, y, z);  // y * z == x
  TEST_ASSERT_EQUAL_DOUBLE(1.0, r[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[2], MFTL_EPSILON);

  MFTLVectorCrossVector(r, z, x);  // z * x == y
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, r[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[2], MFTL_EPSILON);

  // Anti-commutativity, all three components, not just one
  MFTLVectorCrossVector(r, y, x);  // y * x == -z
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(-1.0, r[2], MFTL_EPSILON);
  return 0;
}

static int test_cross_product_is_orthogonal(void)
{
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {-2.0, 0.5, 4.0};
  double r[3] = {0.0, 0.0, 0.0};

  MFTLVectorCrossVector(r, a, b);

  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTLVectorInProd(r, a), MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTLVectorInProd(r, b), MFTL_EPSILON);
  return 0;
}

static int test_cross_product_parallel_is_zero(void)
{
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {2.0, 4.0, 6.0};  // 2*a
  double r[3] = {9.0, 9.0, 9.0};

  MFTLVectorCrossVector(r, a, b);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[0], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[1], MFTL_EPSILON);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, r[2], MFTL_EPSILON);
  return 0;
}

int main(void)
{
  printf("=== MFTL Vector Operations Tests ===\n");

  RUN_TEST(test_normalize_unit_axis);
  RUN_TEST(test_normalize_general);
  RUN_TEST(test_normalize_zero_vector_is_finite);
  RUN_TEST(test_normalize_is_unit_across_magnitudes);
  RUN_TEST(test_dot_product);
  RUN_TEST(test_dot_product_orthogonal);
  RUN_TEST(test_dot_product_self_equals_norm_squared);
  RUN_TEST(test_cross_product_right_handed);
  RUN_TEST(test_cross_product_is_orthogonal);
  RUN_TEST(test_cross_product_parallel_is_zero);

  TEST_REPORT();
  return (g_tests_failed > 0) ? 1 : 0;
}

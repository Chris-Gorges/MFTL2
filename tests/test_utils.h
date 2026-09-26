/*
 * @Author: Christian Gorges
 * @Date: 2026-01-25
 * @Brief: Minimal assertion macros for MFTL unit testing.
 */

#ifndef MFTL_TEST_UTILS_H
#define MFTL_TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_REPORT()                                                                                                                   \
  do                                                                                                                                    \
  {                                                                                                                                     \
    printf("\n-------------------------------------------------\n");                                                                    \
    printf("TEST REPORT: Run: %d | Passed: %s%d%s | Failed: %s%d%s\n", g_tests_run, ANSI_COLOR_GREEN, g_tests_passed, ANSI_COLOR_RESET, \
           (g_tests_failed > 0) ? ANSI_COLOR_RED : ANSI_COLOR_GREEN, g_tests_failed, ANSI_COLOR_RESET);                                 \
    printf("-------------------------------------------------\n");                                                                      \
  } while (0)

#define RUN_TEST(func)                                            \
  do                                                              \
  {                                                               \
    g_tests_run++;                                                \
    printf("[RUN     ] %s...", #func);                            \
    if (func() == 0)                                              \
    {                                                             \
      printf("%sPASSED%s\n", ANSI_COLOR_GREEN, ANSI_COLOR_RESET); \
      g_tests_passed++;                                           \
    }                                                             \
    else                                                          \
    {                                                             \
      printf("%sFAILED%s\n", ANSI_COLOR_RED, ANSI_COLOR_RESET);   \
      g_tests_failed++;                                           \
    }                                                             \
  } while (0)

// Assertions return 0 on success, 1 on failure
#define TEST_ASSERT(cond)                                                                                                       \
  do                                                                                                                            \
  {                                                                                                                             \
    if (!(cond))                                                                                                                \
    {                                                                                                                           \
      printf("\n  %sAssertion failed: (%s)%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, #cond, ANSI_COLOR_RESET, __FILE__, __LINE__); \
      return 1;                                                                                                                 \
    }                                                                                                                           \
  } while (0)

#define TEST_ASSERT_EQUAL_INT(expected, actual)                                                                                                                \
  do                                                                                                                                                           \
  {                                                                                                                                                            \
    if ((expected) != (actual))                                                                                                                                \
    {                                                                                                                                                          \
      printf("\n  %sAssertion failed: Expected %d, Got %d%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, (int) (expected), (int) (actual), ANSI_COLOR_RESET, __FILE__, \
             __LINE__);                                                                                                                                        \
      return 1;                                                                                                                                                \
    }                                                                                                                                                          \
  } while (0)

#define TEST_ASSERT_EQUAL_DOUBLE(expected, actual, tol)                                                                                            \
  do                                                                                                                                               \
  {                                                                                                                                                \
    if (fabs((expected) - (actual)) > (tol))                                                                                                       \
    {                                                                                                                                              \
      printf("\n  %sAssertion failed: Expected %f, Got %f (Tol: %f)%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, (double) (expected), (double) (actual), \
             (double) (tol), ANSI_COLOR_RESET, __FILE__, __LINE__);                                                                                \
      return 1;                                                                                                                                    \
    }                                                                                                                                              \
  } while (0)

#define TEST_ASSERT_NOT_EQUAL_INT(expected, actual)                                                                                                      \
  do                                                                                                                                                     \
  {                                                                                                                                                      \
    if ((expected) == (actual))                                                                                                                          \
    {                                                                                                                                                    \
      printf("\n  %sAssertion failed: Expected NOT %d, Got %d%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, (int) (expected), (int) (actual), ANSI_COLOR_RESET, \
             __FILE__, __LINE__);                                                                                                                        \
      return 1;                                                                                                                                          \
    }                                                                                                                                                    \
  } while (0)

#define TEST_ASSERT_TRUE(cond) TEST_ASSERT(cond)

/**
 * @brief The process grid the multi-rank tests run on at @p nproc ranks.
 *
 * 2x2x1 at 4, 3x2x1 at 6, 2x2x2 at 8, 3x3x3 at 27, and a slab (nproc x 1 x 1)
 * at any other count.
 *
 * 3x3x3 is worth a run of its own. At cuts[k] == 2 a periodic axis has both of a
 * cell's neighbours be the SAME rank, so "one cell towards -x" and "one cell
 * towards +x" are indistinguishable and a direction error cannot show up; three
 * is the smallest count at which they are different ranks, and the smallest grid
 * with an interior cell whose 26 neighbours are 26 distinct ranks.
 *
 * @param      nproc Ranks in the job.
 * @param[out] cuts  The grid; its product is @p nproc.
 */
static inline void test_grid_for(int nproc, int cuts[3])
{
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
  if (nproc == 4)
  {
    cuts[0] = 2;
    cuts[1] = 2;
  }
  else if (nproc == 6)
  {
    cuts[0] = 3;
    cuts[1] = 2;
  }
  else if (nproc == 8)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 2;
  }
  else if (nproc == 27)
  {
    cuts[0] = 3;
    cuts[1] = 3;
    cuts[2] = 3;
  }
}

#define TEST_FAIL()                                                                                            \
  do                                                                                                           \
  {                                                                                                            \
    printf("\n  %sForced Failure%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, ANSI_COLOR_RESET, __FILE__, __LINE__); \
    return 1;                                                                                                  \
  } while (0)

#endif  // MFTL_TEST_UTILS_H

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

#define TEST_FAIL()                                                                                            \
  do                                                                                                           \
  {                                                                                                            \
    printf("\n  %sForced Failure%s\n  File: %s:%d\n  ", ANSI_COLOR_RED, ANSI_COLOR_RESET, __FILE__, __LINE__); \
    return 1;                                                                                                  \
  } while (0)

#endif  // MFTL_TEST_UTILS_H

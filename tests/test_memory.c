/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for mftl_memory.h.
 *
 * The block-layout helpers are the mechanism the whole pool design rests on, so
 * they are tested as a mechanism: alignment, non-overlap, and the exact size of
 * the block they describe. A bug here is not a wrong number anywhere -- it is
 * two SoA arrays sharing memory.
 */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_memory.h"
#include "test_utils.h"

/* ── MFTLAlignUp ────────────────────────────────────────────────────────── */

static int test_align_up_rounds_to_the_next_multiple(void)
{
  TEST_ASSERT_EQUAL_INT(0, (int) MFTLAlignUp(0, 64));
  TEST_ASSERT_EQUAL_INT(64, (int) MFTLAlignUp(1, 64));
  TEST_ASSERT_EQUAL_INT(64, (int) MFTLAlignUp(63, 64));
  TEST_ASSERT_EQUAL_INT(64, (int) MFTLAlignUp(64, 64));  // already aligned: unchanged
  TEST_ASSERT_EQUAL_INT(128, (int) MFTLAlignUp(65, 64));
  TEST_ASSERT_EQUAL_INT(8, (int) MFTLAlignUp(5, 8));
  return 0;
}

/* ── MFTLGrowCapacity ───────────────────────────────────────────────────── */

static int test_grow_capacity_doubles_from_the_initial_size(void)
{
  /* From empty, the first growth jumps straight to MFTL_INIT_CAPACITY rather
   * than to the requested size, so a mesh built one entity at a time does not
   * realloc on every append. */
  TEST_ASSERT_EQUAL_INT(MFTL_INIT_CAPACITY, MFTLGrowCapacity(0, 1));
  TEST_ASSERT_EQUAL_INT(MFTL_INIT_CAPACITY, MFTLGrowCapacity(0, MFTL_INIT_CAPACITY));
  TEST_ASSERT_EQUAL_INT(2 * MFTL_INIT_CAPACITY, MFTLGrowCapacity(0, MFTL_INIT_CAPACITY + 1));

  // Never shrinks.
  TEST_ASSERT_EQUAL_INT(1000, MFTLGrowCapacity(1000, 10));
  TEST_ASSERT_EQUAL_INT(1000, MFTLGrowCapacity(1000, 1000));

  // Doubles until the request fits.
  TEST_ASSERT_EQUAL_INT(2000, MFTLGrowCapacity(1000, 1001));
  TEST_ASSERT_EQUAL_INT(8000, MFTLGrowCapacity(1000, 5000));
  return 0;
}

static int test_grow_capacity_is_amortised_logarithmic(void)
{
  /* The property that matters: growing to n one element at a time costs
   * O(log n) reallocations, not O(n). */
  int cap = 0;
  int reallocs = 0;
  for (int n = 1; n <= 100000; ++n)
  {
    if (n > cap)
    {
      cap = MFTLGrowCapacity(cap, n);
      reallocs++;
    }
  }
  TEST_ASSERT(cap >= 100000);
  TEST_ASSERT(reallocs <= 20);
  return 0;
}

/** @brief The pre-MFTLGrowSize() int ladder, kept here as the reference. */
static int reference_grow_capacity(int current, int needed)
{
  if (needed <= current) return current;
  int cap = (current > 0) ? current : MFTL_INIT_CAPACITY;
  while (cap < needed)
  {
    if (cap > (2147483647 / 2)) return needed;
    cap *= 2;
  }
  return cap;
}

static int test_grow_capacity_is_the_old_ladder_to_the_last_int(void)
{
  /* The int forms are now MFTLGrowSize() underneath. Every capacity a pool or a
   * growth macro ever had must be unchanged, near INT_MAX included (where the
   * ladder saturates to the request instead of overflowing). */
  const int currents[] = {-5, 0, 1, 3, 100, 128, 1000, 1 << 20, (1 << 30) - 1, 1 << 30, (1 << 30) + 7, INT_MAX - 1, INT_MAX};
  const int needs[] = {-1, 0, 1, 2, 127, 128, 129, 1001, 5000, 1 << 29, (1 << 30) + 1, INT_MAX - 1, INT_MAX};
  for (size_t i = 0; i < sizeof currents / sizeof currents[0]; ++i)
    for (size_t j = 0; j < sizeof needs / sizeof needs[0]; ++j)
      TEST_ASSERT_EQUAL_INT(reference_grow_capacity(currents[i], needs[j]), MFTLGrowCapacity(currents[i], needs[j]));
  return 0;
}

static int test_grow_size_doubles_from_its_own_initial_size(void)
{
  TEST_ASSERT(MFTLGrowSize(0, 1, 256) == 256);
  TEST_ASSERT(MFTLGrowSize(0, 257, 256) == 512);
  TEST_ASSERT(MFTLGrowSize(0, 5, 0) == 8);  // initial 0 is 1
  TEST_ASSERT(MFTLGrowSize(300, 200, 16) == 300);  // never shrinks
  TEST_ASSERT(MFTLGrowSize(300, 601, 16) == 1200);
  /* Past half of size_t it hands out exactly the request instead of wrapping. */
  const size_t big = SIZE_MAX / 2 + 2;
  TEST_ASSERT(MFTLGrowSize(SIZE_MAX / 2 + 1, big, 16) == big);
  TEST_ASSERT(MFTLGrowSize(SIZE_MAX / 4, SIZE_MAX - 1, 16) == SIZE_MAX - 1);

  TEST_ASSERT_EQUAL_INT(64, MFTLGrowCapacityFrom(0, 1, 64));
  TEST_ASSERT_EQUAL_INT(128, MFTLGrowCapacityFrom(64, 65, 8));
  TEST_ASSERT_EQUAL_INT(1, MFTLGrowCapacityFrom(0, 1, 0));
  TEST_ASSERT_EQUAL_INT(INT_MAX - 3, MFTLGrowCapacityFrom(1 << 30, INT_MAX - 3, 8));  // saturates, never negative
  return 0;
}

static int test_checked_count_adds_inside_the_int_range(void)
{
  TEST_ASSERT_EQUAL_INT(7, MFTLCheckedCount(3, 4, "test"));
  TEST_ASSERT_EQUAL_INT(INT_MAX, MFTLCheckedCount(INT_MAX - 1, 1, "test"));
  TEST_ASSERT_EQUAL_INT(0, MFTLCheckedCount(0, 0, "test"));
  return 0;
}

/* ── MFTLLayoutAdd ──────────────────────────────────────────────────────── */

static int test_layout_offsets_are_aligned_and_disjoint(void)
{
  const size_t counts[5] = {7, 21, 1, 300, 12};
  const size_t sizes[5] = {sizeof(int), sizeof(double), sizeof(char), sizeof(double), sizeof(int)};

  size_t cursor = 0;
  size_t off[5];
  for (int i = 0; i < 5; ++i) off[i] = MFTLLayoutAdd(&cursor, counts[i], sizes[i]);

  // Every sub-array starts on a cache line.
  for (int i = 0; i < 5; ++i) TEST_ASSERT(off[i] % MFTL_CACHE_LINE == 0);

  // Every sub-array fits strictly before the next one starts.
  for (int i = 0; i + 1 < 5; ++i) TEST_ASSERT(off[i] + counts[i] * sizes[i] <= off[i + 1]);

  // The last one fits inside the block the cursor reports.
  TEST_ASSERT(off[4] + counts[4] * sizes[4] <= cursor);

  // Offsets are strictly increasing, so nothing was handed out twice.
  for (int i = 0; i + 1 < 5; ++i) TEST_ASSERT(off[i] < off[i + 1]);
  return 0;
}

static int test_layout_of_zero_sized_pool_is_empty(void)
{
  size_t cursor = 0;
  const size_t a = MFTLLayoutAdd(&cursor, 0, sizeof(double));
  const size_t b = MFTLLayoutAdd(&cursor, 0, sizeof(int));
  TEST_ASSERT_EQUAL_INT(0, (int) a);
  TEST_ASSERT_EQUAL_INT(0, (int) b);
  TEST_ASSERT_EQUAL_INT(0, (int) cursor);
  return 0;
}

/* ── MFTLAlignedAllocOrDie ──────────────────────────────────────────────── */

static int test_aligned_alloc_is_cache_line_aligned_and_writable(void)
{
  for (size_t bytes = 1; bytes <= 4096; bytes *= 3)
  {
    unsigned char *p = (unsigned char *) MFTLAlignedAllocOrDie(bytes, "test");
    TEST_ASSERT(p != MFTL_NULL);
    TEST_ASSERT(((uintptr_t) p % MFTL_CACHE_LINE) == 0);

    // The whole request must be usable; ASan catches an under-allocation here.
    memset(p, 0xAB, bytes);
    TEST_ASSERT_EQUAL_INT(0xAB, p[bytes - 1]);
    free(p);
  }
  return 0;
}

/* ── The growth macros ──────────────────────────────────────────────────── */

static int test_ensure_capacity_grows_geometrically_and_preserves(void)
{
  int *arr = MFTL_NULL;
  int count = 0;
  int capacity = 0;

  for (int i = 0; i < 500; ++i)
  {
    MFTL_ENSURE_CAPACITY(arr, &count, &capacity, 1, int);
    arr[count++] = i;
  }

  TEST_ASSERT_EQUAL_INT(500, count);
  TEST_ASSERT(capacity >= 500);
  TEST_ASSERT(capacity <= 1024);  // geometric, not linear

  for (int i = 0; i < 500; ++i) TEST_ASSERT_EQUAL_INT(i, arr[i]);

  free(arr);
  return 0;
}

static int test_reserve_additional_grows_exactly(void)
{
  double *arr = MFTL_NULL;
  int count = 0;
  int capacity = 0;

  MFTL_RESERVE_ADDITIONAL(arr, &count, &capacity, 300, double);
  TEST_ASSERT_EQUAL_INT(300, capacity);  // exact, no doubling slack

  for (int i = 0; i < 300; ++i) arr[i] = (double) i;
  count = 300;

  MFTL_RESERVE_ADDITIONAL(arr, &count, &capacity, 7, double);
  TEST_ASSERT_EQUAL_INT(307, capacity);
  for (int i = 0; i < 300; ++i) TEST_ASSERT_EQUAL_DOUBLE((double) i, arr[i], 0.0);

  // A request that already fits must not reallocate.
  const double *before = arr;
  MFTL_RESERVE_ADDITIONAL(arr, &count, &capacity, 5, double);
  TEST_ASSERT(arr == before);
  TEST_ASSERT_EQUAL_INT(307, capacity);

  free(arr);
  return 0;
}

int main(void)
{
  printf("--- Running MFTL Memory Tests ---\n");

  RUN_TEST(test_align_up_rounds_to_the_next_multiple);
  RUN_TEST(test_grow_capacity_doubles_from_the_initial_size);
  RUN_TEST(test_grow_capacity_is_amortised_logarithmic);
  RUN_TEST(test_grow_capacity_is_the_old_ladder_to_the_last_int);
  RUN_TEST(test_grow_size_doubles_from_its_own_initial_size);
  RUN_TEST(test_checked_count_adds_inside_the_int_range);
  RUN_TEST(test_layout_offsets_are_aligned_and_disjoint);
  RUN_TEST(test_layout_of_zero_sized_pool_is_empty);
  RUN_TEST(test_aligned_alloc_is_cache_line_aligned_and_writable);
  RUN_TEST(test_ensure_capacity_grows_geometrically_and_preserves);
  RUN_TEST(test_reserve_additional_grows_exactly);

  TEST_REPORT();
  return (g_tests_failed > 0) ? 1 : 0;
}

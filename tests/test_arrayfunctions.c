/*
 * @brief Unit tests for mftl_arrayfunctions.h (pure array code, no MPI needed).
 *
 * ============================================================================
 * COVERAGE MAP -- what this file checks, and why each case exists:
 * ============================================================================
 *
 * The radix sort has three internal code paths that a naive "sort a random
 * array" test would never separate, so every case below states which one it
 * drives:
 *
 *   (A) insertion sort         n <= 32
 *   (B) radix, all digits used every byte of the key varies
 *   (C) radix, digits skipped  a key byte is constant across all keys, so that
 *                              pass is dropped -- and if an ODD number of
 *                              passes survives, the result has to be copied
 *                              back out of the scratch buffer
 *   (D) early-out              the input is already sorted, no pass runs at all
 *
 * MFTLSortIds / MFTLRadixSort32  (signed 32-bit keys)
 *   - degenerate lengths 0/1 and a NULL array with count 0 (no crash)          (A)
 *   - lengths 31/32/33 straddling the insertion cutoff: both paths must
 *     produce the same answer as qsort                                      (A,B)
 *   - key ranges 8/16/24/32 bits wide, which make 1/2/3/4 passes survive --
 *     the 1- and 3-pass cases are the ones that need the copy-back           (B,C)
 *   - NEGATIVE keys: they must sort BEFORE the non-negative ones. Reading the
 *     keys as unsigned (the obvious implementation) puts them after, so this
 *     is checked against qsort over the whole signed range                    (B)
 *   - all-equal keys and an already-sorted array                              (D)
 *   - MFTLRadixSort32 must leave the result in keys_in for every pass count,
 *     not just for the even ones                                            (B,C)
 *
 * MFTLRadixSort64  (unsigned 64-bit keys)
 *   - full-width random keys                                                  (B)
 *   - keys above 2^63: these must sort LAST (unsigned order), which is the
 *     opposite of what the signed 32-bit path does with its top bit           (B)
 *   - narrow keys (< 2^8), i.e. seven of eight passes skipped                 (C)
 *   - the result-in-keys_in contract, again across pass counts              (B,C)
 *
 * MFTLSortPairs32 / MFTLSortPairs64
 *   - values follow their keys: with values seeded as 0..n-1, the identity
 *     keys[i] == original_keys[values[i]] must hold after the sort, and the
 *     values must still be a permutation (nothing duplicated or lost)
 *   - STABILITY on heavily duplicated keys: within a run of equal keys the
 *     values must stay in ascending original order. MFTLUniquePairs keeps the
 *     first value of each run, so without stability the surviving entity would
 *     be arbitrary -- and different on different ranks
 *   - both widths at n = 32 and n = 33, so the insertion path is held to the
 *     same stability requirement as the radix path
 *
 * MFTLUniqueIntArray / MFTLUniquePairs / MFTLGetUniqueIntegers
 *   - empty input, no duplicates, all duplicates, runs at both array ends
 *   - MFTLUniquePairs keeps the FIRST value of each key run
 *   - MFTLGetUniqueIntegers preserves first-seen order and silently truncates
 *     at output_capacity (documented behaviour -- pinned here so a future
 *     change to it is a deliberate one)
 *
 * The oracle throughout is qsort() over a copy of the same data, so the tests
 * check the actual ordering rather than a hand-written expected array.
 */

#include "test_utils.h"

#include <stdint.h>

#include "mftl_arrayfunctions.h"

/* ─── Deterministic PRNG (splitmix64) ─────────────────────────────────────────
 * Fixed seed per test: a failure is reproducible, and CI never flakes. */
static uint64_t g_rng = 0;

static void rng_seed(uint64_t s) { g_rng = s; }

static uint64_t rng_next(void)
{
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/* ─── Oracles and helpers ──────────────────────────────────────────────────── */

static int cmp_int(const void *a, const void *b)
{
  const int x = *(const int *) a;
  const int y = *(const int *) b;
  return (x > y) - (x < y);
}

static int cmp_u64(const void *a, const void *b)
{
  const uint64_t x = *(const uint64_t *) a;
  const uint64_t y = *(const uint64_t *) b;
  return (x > y) - (x < y);
}

/** @brief Reinterprets a bit pattern as int without relying on implementation-defined conversion. */
static int bits_to_int(uint32_t u)
{
  int out;
  memcpy(&out, &u, sizeof out);
  return out;
}

/**
 * @brief Fills with random keys whose absolute value fits in `mask`.
 * @param signed_keys When non-zero, roughly half the keys come out negative.
 */
static void fill_random_ints(int *a, int n, uint32_t mask, int signed_keys)
{
  for (int i = 0; i < n; ++i)
  {
    const uint32_t u = (uint32_t) rng_next() & mask;
    a[i] = signed_keys ? bits_to_int(u) : (int) (u & 0x7fffffffu);
  }
}

/** @brief Sorts `src` with MFTLSortIds and compares element-wise against qsort. */
static int expect_sorted_like_qsort(const int *src, int n, const char *what)
{
  int *a = (int *) malloc(sizeof(int) * (size_t) (n > 0 ? n : 1));
  int *ref = (int *) malloc(sizeof(int) * (size_t) (n > 0 ? n : 1));
  int failed = 0;

  memcpy(a, src, sizeof(int) * (size_t) n);
  memcpy(ref, src, sizeof(int) * (size_t) n);
  qsort(ref, (size_t) n, sizeof(int), cmp_int);

  MFTLSortIds(a, n);

  for (int i = 0; i < n && !failed; ++i)
  {
    if (a[i] != ref[i])
    {
      printf("\n  %s[%s]: element %d is %d, expected %d%s", ANSI_COLOR_RED, what, i, a[i], ref[i], ANSI_COLOR_RESET);
      failed = 1;
    }
  }

  free(a);
  free(ref);
  return failed;
}

/** @brief Same for 64-bit keys, through MFTLRadixSort64 with a caller-owned temp buffer. */
static int expect_sorted_like_qsort64(const uint64_t *src, int n, const char *what)
{
  const size_t bytes = sizeof(uint64_t) * (size_t) (n > 0 ? n : 1);
  uint64_t *a = (uint64_t *) malloc(bytes);
  uint64_t *tmp = (uint64_t *) malloc(bytes);
  uint64_t *ref = (uint64_t *) malloc(bytes);
  int failed = 0;

  memcpy(a, src, sizeof(uint64_t) * (size_t) n);
  memcpy(ref, src, sizeof(uint64_t) * (size_t) n);
  qsort(ref, (size_t) n, sizeof(uint64_t), cmp_u64);

  MFTLRadixSort64(a, tmp, n);

  for (int i = 0; i < n && !failed; ++i)
  {
    if (a[i] != ref[i])
    {
      printf("\n  %s[%s]: element %d is %llu, expected %llu%s", ANSI_COLOR_RED, what, i, (unsigned long long) a[i], (unsigned long long) ref[i],
             ANSI_COLOR_RESET);
      failed = 1;
    }
  }

  free(a);
  free(tmp);
  free(ref);
  return failed;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Key-only sort, 32-bit
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_sort_ids_degenerate(void)
{
  int one[1] = {42};
  int two[2] = {7, 3};

  MFTLSortIds(NULL, 0); /* must not dereference */
  MFTLSortIds(one, 0);
  MFTLSortIds(one, 1);
  TEST_ASSERT_EQUAL_INT(42, one[0]);

  MFTLSortIds(two, 2);
  TEST_ASSERT_EQUAL_INT(3, two[0]);
  TEST_ASSERT_EQUAL_INT(7, two[1]);
  return 0;
}

static int test_sort_ids_cutoff_boundaries(void)
{
  /* 32 is the insertion/radix cutoff: the two paths must be indistinguishable. */
  const int lengths[] = {2, 3, 17, 31, 32, 33, 34, 63, 64, 65, 127, 128};
  int buf[256];

  rng_seed(0x51D5C0DEu);
  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    const int n = lengths[c];
    fill_random_ints(buf, n, 0xffffu, 0);
    TEST_ASSERT(expect_sorted_like_qsort(buf, n, "cutoff") == 0);
  }
  return 0;
}

static int test_sort_ids_pass_skipping(void)
{
  /* Key width controls how many of the four passes survive:
   *   0xff       -> 1 pass  (odd  -> result must be copied back)
   *   0xffff     -> 2 passes
   *   0xffffff   -> 3 passes (odd -> copied back)
   *   0x7fffffff -> 4 passes                                                */
  const uint32_t masks[] = {0xffu, 0xffffu, 0xffffffu, 0x7fffffffu};
  const int n = 5000;
  int *buf = (int *) malloc(sizeof(int) * (size_t) n);

  rng_seed(20260810u);
  for (size_t c = 0; c < sizeof masks / sizeof masks[0]; ++c)
  {
    fill_random_ints(buf, n, masks[c], 0);
    if (expect_sorted_like_qsort(buf, n, "pass-skip") != 0)
    {
      free(buf);
      TEST_FAIL();
    }
  }
  free(buf);
  return 0;
}

static int test_sort_ids_negative_keys_sort_first(void)
{
  /* The whole point: read as unsigned, -1 would land at the very end. */
  int mixed[200];
  rng_seed(0xBADCAFEu);
  for (int i = 0; i < 200; ++i) mixed[i] = (int) (rng_next() % 2001u) - 1000; /* [-1000, 1000] */

  TEST_ASSERT(expect_sorted_like_qsort(mixed, 200, "small-mixed") == 0);

  /* Full signed range, radix path, all four passes. */
  const int n = 4000;
  int *wide = (int *) malloc(sizeof(int) * (size_t) n);
  fill_random_ints(wide, n, 0xffffffffu, 1);
  const int failed = expect_sorted_like_qsort(wide, n, "full-range-signed");
  free(wide);
  TEST_ASSERT(failed == 0);

  /* Explicit ordering check around zero, radix path (n > cutoff). */
  const int m = 300;
  int *around = (int *) malloc(sizeof(int) * (size_t) m);
  for (int i = 0; i < m; ++i) around[i] = (i % 2) ? -(i + 1) : (i + 1);
  MFTLSortIds(around, m);
  TEST_ASSERT(around[0] < 0);
  TEST_ASSERT(around[m - 1] > 0);
  for (int i = 1; i < m; ++i)
  {
    if (around[i - 1] > around[i])
    {
      free(around);
      TEST_FAIL();
    }
  }
  free(around);
  return 0;
}

static int test_sort_ids_already_sorted_and_constant(void)
{
  const int n = 1000;
  int *a = (int *) malloc(sizeof(int) * (size_t) n);

  /* (D) already sorted -- the early-out must leave the array untouched. */
  for (int i = 0; i < n; ++i) a[i] = i * 3 - 500;
  MFTLSortIds(a, n);
  for (int i = 0; i < n; ++i)
  {
    if (a[i] != i * 3 - 500)
    {
      free(a);
      TEST_FAIL();
    }
  }

  /* All keys equal: every digit is constant, so no pass runs. */
  for (int i = 0; i < n; ++i) a[i] = -7;
  MFTLSortIds(a, n);
  for (int i = 0; i < n; ++i)
  {
    if (a[i] != -7)
    {
      free(a);
      TEST_FAIL();
    }
  }

  /* Strictly descending: the worst case for the early-out probe. */
  for (int i = 0; i < n; ++i) a[i] = n - i;
  MFTLSortIds(a, n);
  for (int i = 0; i < n; ++i)
  {
    if (a[i] != i + 1)
    {
      free(a);
      TEST_FAIL();
    }
  }

  free(a);
  return 0;
}

static int test_radixsort32_result_lands_in_input(void)
{
  /* The documented contract is "result is always in keys_in". With constant
   * digits skipped the pass count can be odd, so this holds only because of the
   * copy-back -- check it for every surviving pass count. */
  const uint32_t masks[] = {0xffu, 0xffffu, 0xffffffu, 0x7fffffffu};
  const int n = 1000;
  int *a = (int *) malloc(sizeof(int) * (size_t) n);
  int *tmp = (int *) malloc(sizeof(int) * (size_t) n);
  int *ref = (int *) malloc(sizeof(int) * (size_t) n);
  int failed = 0;

  rng_seed(99u);
  for (size_t c = 0; c < sizeof masks / sizeof masks[0] && !failed; ++c)
  {
    fill_random_ints(a, n, masks[c], 0);
    memcpy(ref, a, sizeof(int) * (size_t) n);
    qsort(ref, (size_t) n, sizeof(int), cmp_int);

    MFTLRadixSort32(a, tmp, n);
    for (int i = 0; i < n; ++i)
    {
      if (a[i] != ref[i]) failed = 1;
    }
  }

  free(a);
  free(tmp);
  free(ref);
  TEST_ASSERT(failed == 0);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Key-only sort, 64-bit
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_radixsort64_full_width(void)
{
  const int n = 5000;
  uint64_t *a = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);

  rng_seed(0x64B175u);
  for (int i = 0; i < n; ++i) a[i] = rng_next();

  const int failed = expect_sorted_like_qsort64(a, n, "u64-full");
  free(a);
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_radixsort64_is_unsigned(void)
{
  /* Keys with the top bit set must sort LAST -- the 64-bit path is unsigned,
   * unlike the signed 32-bit one. */
  const int n = 200;
  uint64_t *a = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);
  uint64_t *tmp = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);

  for (int i = 0; i < n; ++i) a[i] = (i % 2) ? (UINT64_MAX - (uint64_t) i) : (uint64_t) i;

  MFTLRadixSort64(a, tmp, n);

  int failed = (a[0] != 0u) || (a[n - 1] != UINT64_MAX - 1u);
  for (int i = 1; i < n; ++i)
  {
    if (a[i - 1] > a[i]) failed = 1;
  }

  free(a);
  free(tmp);
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_radixsort64_narrow_keys(void)
{
  /* Only the low byte varies: seven of eight passes are skipped, one survives
   * (odd) -- the copy-back path for 64-bit keys. */
  const int n = 3000;
  uint64_t *a = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);

  rng_seed(4242u);
  for (int i = 0; i < n; ++i) a[i] = rng_next() & 0xffu;
  int failed = expect_sorted_like_qsort64(a, n, "u64-narrow");

  /* Two surviving passes (even, no copy-back), and a wider odd case. */
  for (int i = 0; i < n; ++i) a[i] = rng_next() & 0xffffu;
  failed |= expect_sorted_like_qsort64(a, n, "u64-16bit");
  for (int i = 0; i < n; ++i) a[i] = rng_next() & 0xffffffu;
  failed |= expect_sorted_like_qsort64(a, n, "u64-24bit");

  /* Degenerate: all equal, and already sorted. */
  for (int i = 0; i < n; ++i) a[i] = 12345u;
  failed |= expect_sorted_like_qsort64(a, n, "u64-constant");
  for (int i = 0; i < n; ++i) a[i] = (uint64_t) i * 7u;
  failed |= expect_sorted_like_qsort64(a, n, "u64-presorted");

  free(a);
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_radixsort64_cutoff_boundaries(void)
{
  const int lengths[] = {2, 31, 32, 33, 34, 64, 65};
  uint64_t buf[128];
  rng_seed(0xC0FFEEu);

  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    for (int i = 0; i < lengths[c]; ++i) buf[i] = rng_next();
    TEST_ASSERT(expect_sorted_like_qsort64(buf, lengths[c], "u64-cutoff") == 0);
  }
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Key/value pair sorts
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Checks that a pair sort moved the values with their keys.
 *
 * With values seeded 0..n-1, three things must hold afterwards:
 *   1. the keys are non-decreasing,
 *   2. the values are still a permutation of 0..n-1 (nothing lost/duplicated),
 *   3. keys[i] == orig_keys[values[i]] -- each value still names the slot its
 *      key came from, which is the property every caller relies on.
 */
static int check_pairs32(const int *orig_keys, int *keys, int *values, int n)
{
  char *seen = (char *) calloc((size_t) n, 1);
  int failed = 0;

  for (int i = 1; i < n; ++i)
  {
    if (keys[i - 1] > keys[i]) failed = 1;
  }
  for (int i = 0; i < n; ++i)
  {
    const int v = values[i];
    if (v < 0 || v >= n || seen[v]) failed = 1;
    else
    {
      seen[v] = 1;
      if (keys[i] != orig_keys[v]) failed = 1;
    }
  }

  free(seen);
  return failed;
}

static int check_pairs64(const uint64_t *orig_keys, uint64_t *keys, int *values, int n)
{
  char *seen = (char *) calloc((size_t) n, 1);
  int failed = 0;

  for (int i = 1; i < n; ++i)
  {
    if (keys[i - 1] > keys[i]) failed = 1;
  }
  for (int i = 0; i < n; ++i)
  {
    const int v = values[i];
    if (v < 0 || v >= n || seen[v]) failed = 1;
    else
    {
      seen[v] = 1;
      if (keys[i] != orig_keys[v]) failed = 1;
    }
  }

  free(seen);
  return failed;
}

static int test_sort_pairs32_moves_values(void)
{
  const int lengths[] = {2, 32, 33, 5000};
  const uint32_t masks[] = {0xffu, 0xffffffu, 0x7fffffffu};
  int failed = 0;

  rng_seed(0xA17F5u);
  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    for (size_t m = 0; m < sizeof masks / sizeof masks[0]; ++m)
    {
      const int n = lengths[c];
      int *keys = (int *) malloc(sizeof(int) * (size_t) n);
      int *orig = (int *) malloc(sizeof(int) * (size_t) n);
      int *vals = (int *) malloc(sizeof(int) * (size_t) n);

      fill_random_ints(keys, n, masks[m], 1);
      memcpy(orig, keys, sizeof(int) * (size_t) n);
      for (int i = 0; i < n; ++i) vals[i] = i;

      MFTLSortPairs32(keys, vals, n);
      failed |= check_pairs32(orig, keys, vals, n);

      free(keys);
      free(orig);
      free(vals);
    }
  }
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_sort_pairs32_is_stable(void)
{
  /* Only 8 distinct keys over 3000 entries: long runs of equal keys, whose
   * values must come out in ascending original order.
   * n = 32 additionally holds the insertion path to the same standard. */
  const int lengths[] = {32, 33, 3000};

  rng_seed(0x57AB1Eu);
  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    const int n = lengths[c];
    int *keys = (int *) malloc(sizeof(int) * (size_t) n);
    int *vals = (int *) malloc(sizeof(int) * (size_t) n);
    int failed = 0;

    for (int i = 0; i < n; ++i)
    {
      keys[i] = (int) (rng_next() % 8u) - 4; /* includes negative keys */
      vals[i] = i;
    }

    MFTLSortPairs32(keys, vals, n);

    for (int i = 1; i < n; ++i)
    {
      if (keys[i - 1] > keys[i]) failed = 1;
      if (keys[i - 1] == keys[i] && vals[i - 1] >= vals[i]) failed = 1; /* stability */
    }

    free(keys);
    free(vals);
    TEST_ASSERT(failed == 0);
  }
  return 0;
}

static int test_sort_pairs64_moves_values(void)
{
  const int lengths[] = {2, 32, 33, 5000};
  int failed = 0;

  rng_seed(0x6427D1u);
  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    const int n = lengths[c];
    uint64_t *keys = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);
    uint64_t *orig = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);
    int *vals = (int *) malloc(sizeof(int) * (size_t) n);

    for (int i = 0; i < n; ++i)
    {
      keys[i] = rng_next() >> 1; /* 63-bit, like a 21-bit-per-axis Morton code */
      vals[i] = i;
    }
    memcpy(orig, keys, sizeof(uint64_t) * (size_t) n);

    MFTLSortPairs64(keys, vals, n);
    failed |= check_pairs64(orig, keys, vals, n);

    free(keys);
    free(orig);
    free(vals);
  }
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_sort_pairs64_is_stable(void)
{
  /* Duplicate Morton codes are the normal case for co-located centroids. */
  const int lengths[] = {32, 33, 3000};

  rng_seed(0xDBB1Eu);
  for (size_t c = 0; c < sizeof lengths / sizeof lengths[0]; ++c)
  {
    const int n = lengths[c];
    uint64_t *keys = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) n);
    int *vals = (int *) malloc(sizeof(int) * (size_t) n);
    int failed = 0;

    for (int i = 0; i < n; ++i)
    {
      keys[i] = rng_next() % 8u;
      vals[i] = i;
    }

    MFTLSortPairs64(keys, vals, n);

    for (int i = 1; i < n; ++i)
    {
      if (keys[i - 1] > keys[i]) failed = 1;
      if (keys[i - 1] == keys[i] && vals[i - 1] >= vals[i]) failed = 1; /* stability */
    }

    free(keys);
    free(vals);
    TEST_ASSERT(failed == 0);
  }
  return 0;
}

static int test_generic_dispatch(void)
{
  /* The _Generic macros must pick the right implementation for each key type. */
  int k32[80];
  int t32[80];
  int v32[80];
  uint64_t k64[80];
  int v64[80];

  rng_seed(1u);
  for (int i = 0; i < 80; ++i)
  {
    k32[i] = (int) (rng_next() % 1000u) - 500;
    v32[i] = i;
    k64[i] = rng_next();
    v64[i] = i;
  }

  /* Pointers, not arrays: the C11 text (before DR 481) does not apply
   * array-to-pointer conversion to a _Generic controlling expression, and older
   * GCCs follow it literally. Every real call site passes a pointer anyway. */
  int *pk32 = k32;
  int *pt32 = t32;
  int *pv32 = v32;
  uint64_t *pk64 = k64;
  int *pv64 = v64;

  MFTLRadixSort(pk32, pt32, 80);
  for (int i = 1; i < 80; ++i) TEST_ASSERT(k32[i - 1] <= k32[i]);

  MFTLSortPairs(pk64, pv64, 80);
  for (int i = 1; i < 80; ++i) TEST_ASSERT(k64[i - 1] <= k64[i]);

  MFTLSortPairs(pk32, pv32, 80);
  for (int i = 1; i < 80; ++i) TEST_ASSERT(k32[i - 1] <= k32[i]);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Duplicate removal
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_unique_int_array(void)
{
  int empty[1] = {0};
  int distinct[4] = {1, 2, 3, 4};
  int leading[6] = {1, 1, 1, 2, 3, 4};
  int trailing[6] = {1, 2, 3, 4, 4, 4};
  int all_same[5] = {9, 9, 9, 9, 9};
  int negatives[7] = {-5, -5, -1, 0, 0, 3, 3};

  TEST_ASSERT_EQUAL_INT(0, MFTLUniqueIntArray(empty, 0));
  TEST_ASSERT_EQUAL_INT(0, MFTLUniqueIntArray(NULL, 0));

  TEST_ASSERT_EQUAL_INT(4, MFTLUniqueIntArray(distinct, 4));
  TEST_ASSERT_EQUAL_INT(3, distinct[2]);

  TEST_ASSERT_EQUAL_INT(4, MFTLUniqueIntArray(leading, 6));
  TEST_ASSERT_EQUAL_INT(1, leading[0]);
  TEST_ASSERT_EQUAL_INT(4, leading[3]);

  TEST_ASSERT_EQUAL_INT(4, MFTLUniqueIntArray(trailing, 6));
  TEST_ASSERT_EQUAL_INT(4, trailing[3]);

  TEST_ASSERT_EQUAL_INT(1, MFTLUniqueIntArray(all_same, 5));
  TEST_ASSERT_EQUAL_INT(9, all_same[0]);

  TEST_ASSERT_EQUAL_INT(4, MFTLUniqueIntArray(negatives, 7));
  TEST_ASSERT_EQUAL_INT(-5, negatives[0]);
  TEST_ASSERT_EQUAL_INT(3, negatives[3]);
  return 0;
}

static int test_unique_pairs_keeps_first_value(void)
{
  int keys[8] = {1, 1, 1, 4, 7, 7, 9, 9};
  int vals[8] = {10, 11, 12, 40, 70, 71, 90, 91};

  const int n = MFTLUniquePairs(keys, vals, 8);
  TEST_ASSERT_EQUAL_INT(4, n);

  TEST_ASSERT_EQUAL_INT(1, keys[0]);
  TEST_ASSERT_EQUAL_INT(10, vals[0]); /* first of the run, not the last */
  TEST_ASSERT_EQUAL_INT(4, keys[1]);
  TEST_ASSERT_EQUAL_INT(40, vals[1]);
  TEST_ASSERT_EQUAL_INT(7, keys[2]);
  TEST_ASSERT_EQUAL_INT(70, vals[2]);
  TEST_ASSERT_EQUAL_INT(9, keys[3]);
  TEST_ASSERT_EQUAL_INT(90, vals[3]);

  TEST_ASSERT_EQUAL_INT(0, MFTLUniquePairs(keys, vals, 0));
  return 0;
}

static int test_sort_then_unique_roundtrip(void)
{
  /* The idiom redistribution uses: collect ids, sort, collapse duplicates. */
  const int n = 2000;
  int *ids = (int *) malloc(sizeof(int) * (size_t) n);

  rng_seed(7u);
  for (int i = 0; i < n; ++i) ids[i] = (int) (rng_next() % 500u);

  MFTLSortIds(ids, n);
  const int unique = MFTLUniqueIntArray(ids, n);

  int failed = (unique <= 0 || unique > 500);
  for (int i = 1; i < unique; ++i)
  {
    if (ids[i - 1] >= ids[i]) failed = 1; /* strictly increasing after unique */
  }

  free(ids);
  TEST_ASSERT(failed == 0);
  return 0;
}

static int test_get_unique_integers(void)
{
  const int input[8] = {5, 3, 5, 7, 3, 3, 9, 5};
  int out[8];

  const int n = MFTLGetUniqueIntegers(input, 8, out, 8);
  TEST_ASSERT_EQUAL_INT(4, n);
  TEST_ASSERT_EQUAL_INT(5, out[0]); /* first-seen order, not sorted order */
  TEST_ASSERT_EQUAL_INT(3, out[1]);
  TEST_ASSERT_EQUAL_INT(7, out[2]);
  TEST_ASSERT_EQUAL_INT(9, out[3]);

  /* Documented behaviour: silently truncates at capacity. */
  const int capped = MFTLGetUniqueIntegers(input, 8, out, 2);
  TEST_ASSERT_EQUAL_INT(2, capped);
  TEST_ASSERT_EQUAL_INT(5, out[0]);
  TEST_ASSERT_EQUAL_INT(3, out[1]);

  TEST_ASSERT_EQUAL_INT(0, MFTLGetUniqueIntegers(input, 0, out, 8));
  return 0;
}

int main(void)
{
  printf("=== MFTL Array Functions Tests ===\n");

  RUN_TEST(test_sort_ids_degenerate);
  RUN_TEST(test_sort_ids_cutoff_boundaries);
  RUN_TEST(test_sort_ids_pass_skipping);
  RUN_TEST(test_sort_ids_negative_keys_sort_first);
  RUN_TEST(test_sort_ids_already_sorted_and_constant);
  RUN_TEST(test_radixsort32_result_lands_in_input);

  RUN_TEST(test_radixsort64_full_width);
  RUN_TEST(test_radixsort64_is_unsigned);
  RUN_TEST(test_radixsort64_narrow_keys);
  RUN_TEST(test_radixsort64_cutoff_boundaries);

  RUN_TEST(test_sort_pairs32_moves_values);
  RUN_TEST(test_sort_pairs32_is_stable);
  RUN_TEST(test_sort_pairs64_moves_values);
  RUN_TEST(test_sort_pairs64_is_stable);
  RUN_TEST(test_generic_dispatch);

  RUN_TEST(test_unique_int_array);
  RUN_TEST(test_unique_pairs_keeps_first_value);
  RUN_TEST(test_sort_then_unique_roundtrip);
  RUN_TEST(test_get_unique_integers);

  TEST_REPORT();
  return (g_tests_failed > 0) ? 1 : 0;
}

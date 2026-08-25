/**
 * @file radixsort.c
 * @author Christian Gorges
 * @date 2025-08-12
 * @ingroup mftl_arrayfunctions
 *
 * @brief LSD radix sort over 8-bit digits, for keys and for key/value pairs.
 *
 * Derived from the public-domain-style radix sort by Cameron Hart (see the
 * licence notice below), specialised for MFTL: unified 32/64-bit key width,
 * SoA key/value variants, signed 32-bit ordering, constant-digit pass skipping
 * and a small-array cutoff.
 */

/*
 * Copyright (c) 2014 Cameron Hart
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "mftl_arrayfunctions.h"

#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"

/* The 32-bit paths reinterpret int as uint32_t for digit extraction, and the
 * pair sorts move int values through a uint32_t shuffle. Both assumptions are
 * checked here rather than discovered at run time on an exotic target.
 * (Accessing an int object through an unsigned int lvalue is explicitly allowed
 * by C11 6.5p7 -- this is not a strict-aliasing violation.) */
_Static_assert(sizeof(int) == sizeof(uint32_t), "MFTL radix sort assumes a 32-bit int");
_Static_assert((-1 & 3) == 3, "MFTL radix sort assumes two's complement int");

/* ─── Internal config ────────────────────────────────────────────────────────
 *
 *  RADIX_BITS    8       — process one byte per pass
 *  HIST_SIZE   256       — 2^8 buckets
 *  PASSES_32     4       — 32 / 8 = 4 passes
 *  PASSES_64     8       — 64 / 8 = 8 passes
 *
 *  Stack cost of the histogram arrays:
 *    32-bit:  4 * 256 * 4 bytes =  4 KB
 *    64-bit:  8 * 256 * 4 bytes =  8 KB
 *  Both fit comfortably in L1D, which is what makes the "count every digit in
 *  one pass over the data" strategy below worthwhile: the keys are streamed
 *  exactly once and every histogram update hits L1.
 * ─────────────────────────────────────────────────────────────────────────── */
/** @brief Bits consumed per pass. One byte, so a histogram is 256 counters. */
#define RADIX_BITS 8
/** @brief Buckets per pass, \f$2^{\texttt{RADIX\_BITS}} = 256\f$. */
#define HIST_SIZE (1u << RADIX_BITS)
/** @brief Mask that extracts one digit. */
#define HIST_MASK (HIST_SIZE - 1u)
/** @brief Passes for a 32-bit key: \f$32/8\f$. */
#define PASSES_32 4u
/** @brief Passes for a 64-bit key: \f$64/8\f$. */
#define PASSES_64 8u

/**
 * @brief Order-preserving map from `int32` to `uint32`.
 *
 * \f$x \mapsto x \oplus \texttt{0x80000000}\f$ turns signed comparison into
 * unsigned comparison. Used ONLY for the already-sorted probe; the sort itself
 * gets signed ordering for free -- see prefix_sum_histograms().
 */
#define SIGN_FLIP_32 0x80000000u

/* Below this length a radix sort loses to insertion sort: the histogram memset,
 * the 256-entry-per-pass prefix sum and (in the wrapper functions) a malloc/free
 * pair are all independent of n, and together they outweigh the O(n^2/4) moves
 * of an array that never leaves L1. The redistribution code sorts one small
 * buffer per neighbour rank, so this is not a corner case -- it is the common
 * case there.
 *
 * 32 is where insertion sort still wins on BOTH input patterns, measured
 * (ns/call, Apple M-series, -O3, MFTLSortIds / MFTLSortPairs64):
 *
 *                        random keys            descending keys (worst case)
 *   n     radix    insertion            radix    insertion
 *   16     577          78               277          75
 *   32     749         496               323         287
 *   48     859         866               369         639   <- insertion loses
 *   64     973        1310               415        1299
 *
 * A larger cutoff looks better on random data (64-bit keys, 8 passes, still win
 * 1.6x at n = 64) but turns into a 1.4-3x loss on descending input, where
 * insertion sort hits its O(n^2) worst case and the radix path gets faster
 * because a narrow key range skips passes. One conservative cutoff for both key
 * widths keeps that trap closed. */
/** @brief Below this element count, a stable insertion sort replaces the radix passes. */
#define INSERTION_CUTOFF 32

/* ═══════════════════════════════════════════════════════════════════════════
 * Small-array path: stable insertion sort
 *
 * Strict '>' in the shift condition is what keeps it stable, matching the
 * stability of the radix passes.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Stable insertion sort of 32-bit signed keys.
 * @param k Keys, sorted ascending in place.
 * @param n Number of keys.
 */
static void insertion_sort_keys32(int *restrict k, size_t n)
{
  for (size_t i = 1; i < n; ++i)
  {
    const int key = k[i];
    size_t j = i;
    while (j > 0 && k[j - 1] > key)
    {
      k[j] = k[j - 1];
      --j;
    }
    k[j] = key;
  }
}

/**
 * @brief Stable insertion sort of 64-bit unsigned keys.
 * @param k Keys, sorted ascending in place.
 * @param n Number of keys.
 */
static void insertion_sort_keys64(uint64_t *restrict k, size_t n)
{
  for (size_t i = 1; i < n; ++i)
  {
    const uint64_t key = k[i];
    size_t j = i;
    while (j > 0 && k[j - 1] > key)
    {
      k[j] = k[j - 1];
      --j;
    }
    k[j] = key;
  }
}

/**
 * @brief Stable insertion sort of 32-bit key / 32-bit value pairs.
 * @param k Keys, sorted ascending in place.
 * @param v Values, permuted alongside the keys.
 * @param n Number of pairs.
 */
static void insertion_sort_pairs32(int *restrict k, int *restrict v, size_t n)
{
  for (size_t i = 1; i < n; ++i)
  {
    const int key = k[i];
    const int val = v[i];
    size_t j = i;
    while (j > 0 && k[j - 1] > key)
    {
      k[j] = k[j - 1];
      v[j] = v[j - 1];
      --j;
    }
    k[j] = key;
    v[j] = val;
  }
}

/**
 * @brief Stable insertion sort of 64-bit key / 32-bit value pairs.
 * @param k Keys, sorted ascending in place.
 * @param v Values, permuted alongside the keys.
 * @param n Number of pairs.
 */
static void insertion_sort_pairs64(uint64_t *restrict k, int *restrict v, size_t n)
{
  for (size_t i = 1; i < n; ++i)
  {
    const uint64_t key = k[i];
    const int val = v[i];
    size_t j = i;
    while (j > 0 && k[j - 1] > key)
    {
      k[j] = k[j - 1];
      v[j] = v[j - 1];
      --j;
    }
    k[j] = key;
    v[j] = val;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Histogram counting
 *
 * All digit histograms are built in a single pass over the keys: each key is
 * loaded once and updates PASSES_* counters that all live in the same few KB of
 * L1. Doing one pass per digit instead would multiply the memory traffic by the
 * number of passes for no gain.
 *
 * The same pass also answers "is this already sorted?" for one extra compare
 * per key (branchless, so it costs nothing measurable next to the counter
 * updates). Redistribution feeds these sorts arrays of entity IDs collected in
 * array order, which are frequently ascending already -- detecting that here
 * removes every scatter pass instead of executing four of them.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @return true if the keys are already in ascending SIGNED order.
 *
 * @param[out] hist `PASSES_32 * HIST_SIZE` counters, zeroed and filled here.
 * @param      keys The keys to profile.
 * @param      n    Number of keys.
 */
static bool init_histograms32(uint32_t *restrict hist, const uint32_t *restrict keys, size_t n)
{
  memset(hist, 0, sizeof(uint32_t) * PASSES_32 * HIST_SIZE);

  uint32_t sorted = 1u;
  uint32_t prev = keys[0] ^ SIGN_FLIP_32; /* n >= 1 is a precondition */

  for (size_t i = 0; i < n; ++i)
  {
    const uint32_t k = keys[i];
    hist[0 * HIST_SIZE + (k & HIST_MASK)]++;
    hist[1 * HIST_SIZE + ((k >> 8) & HIST_MASK)]++;
    hist[2 * HIST_SIZE + ((k >> 16) & HIST_MASK)]++;
    hist[3 * HIST_SIZE + ((k >> 24) & HIST_MASK)]++;

    const uint32_t ordered = k ^ SIGN_FLIP_32;
    sorted &= (uint32_t) (prev <= ordered);
    prev = ordered;
  }
  return sorted != 0u;
}

/**
 * @return true if the keys are already in ascending UNSIGNED order.
 *
 * @param[out] hist `PASSES_64 * HIST_SIZE` counters, zeroed and filled here.
 * @param      keys The keys to profile.
 * @param      n    Number of keys.
 */
static bool init_histograms64(uint32_t *restrict hist, const uint64_t *restrict keys, size_t n)
{
  memset(hist, 0, sizeof(uint32_t) * PASSES_64 * HIST_SIZE);

  uint32_t sorted = 1u;
  uint64_t prev = keys[0]; /* n >= 1 is a precondition */

  for (size_t i = 0; i < n; ++i)
  {
    const uint64_t k = keys[i];
    hist[0 * HIST_SIZE + (uint32_t) (k & HIST_MASK)]++;
    hist[1 * HIST_SIZE + (uint32_t) ((k >> 8) & HIST_MASK)]++;
    hist[2 * HIST_SIZE + (uint32_t) ((k >> 16) & HIST_MASK)]++;
    hist[3 * HIST_SIZE + (uint32_t) ((k >> 24) & HIST_MASK)]++;
    hist[4 * HIST_SIZE + (uint32_t) ((k >> 32) & HIST_MASK)]++;
    hist[5 * HIST_SIZE + (uint32_t) ((k >> 40) & HIST_MASK)]++;
    hist[6 * HIST_SIZE + (uint32_t) ((k >> 48) & HIST_MASK)]++;
    hist[7 * HIST_SIZE + (uint32_t) ((k >> 56) & HIST_MASK)]++;

    sorted &= (uint32_t) (prev <= k);
    prev = k;
  }
  return sorted != 0u;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Prefix sum: counts -> exclusive output offsets, plus the pass plan
 *
 * Two things happen here that the plain textbook version does not do, both for
 * free (this loop touches 256 * PASSES counters, never the data):
 *
 * 1. Constant-digit passes are dropped. If every key shares the same value in
 *    byte b, that pass is a pure copy of the array onto itself. Its histogram
 *    then has a single bucket holding all n keys, and the byte of the first key
 *    says which bucket to test -- an O(1) check that saves a full read+scatter
 *    over the data. Entity IDs never fill 32 bits, so this typically removes
 *    one or two of the four passes; on Morton codes it usually removes none.
 *
 * 2. Signed ordering. For the most significant pass of a signed key, the
 *    offsets are accumulated in the rotated bucket order 0x80..0xFF, 0x00..0x7F,
 *    so negative keys are placed before non-negative ones. Only the order in
 *    which 256 offsets are summed changes -- the scatter loop, and therefore the
 *    whole data path, is untouched.
 *
 * @return Bitmask of the passes that actually have to run.
 * ═══════════════════════════════════════════════════════════════════════════ */
/**
 * @brief Turns each pass's counts into starting OFFSETS, in place.
 *
 * Also decides which passes can be skipped: a digit position whose histogram
 * has a single non-zero bucket takes the same value in all @p n keys, so its
 * pass is the identity. Mesh global ids are dense in \f$[0, n)\f$, so the top
 * bytes of a 32-bit id are usually constant and half the passes vanish.
 *
 * @param hist       `passes * HIST_SIZE` counters, converted from counts to
 *                   exclusive prefix sums.
 * @param passes     Number of digit passes.
 * @param n          Number of keys, i.e. the total each histogram sums to.
 * @param first_key  Any one key; its digits identify the constant buckets.
 * @param signed_msb `true` for signed 32-bit keys: the top pass's prefix sum starts
 *                   at bucket 128 and wraps to 0, which places negative keys first
 *                   without ever flipping a bit.
 * @return Bit mask of the passes that must actually run.
 */
static uint32_t prefix_sum_histograms(uint32_t *restrict hist, uint32_t passes, size_t n, uint64_t first_key, bool signed_msb)
{
  uint32_t active = 0u;

  for (uint32_t b = 0; b < passes; ++b)
  {
    uint32_t *restrict off = hist + b * HIST_SIZE;

    const uint32_t bucket0 = (uint32_t) ((first_key >> (b * RADIX_BITS)) & HIST_MASK);
    if (off[bucket0] == (uint32_t) n) continue; /* digit is constant -> no-op pass */

    /* bias == 128 walks the buckets as 0x80..0xFF, 0x00..0x7F (signed order). */
    const uint32_t bias = (signed_msb && b == passes - 1u) ? (HIST_SIZE / 2u) : 0u;

    uint32_t running = 0u;
    for (uint32_t j = 0; j < HIST_SIZE; ++j)
    {
      const uint32_t i = (j + bias) & HIST_MASK;
      const uint32_t count = off[i];
      off[i] = running;
      running += count;
    }
    active |= 1u << b;
  }
  return active;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Scatter passes
 *
 * The one hot loop of the whole file. Reads are sequential (hardware prefetch
 * friendly), writes go to up to 256 independent output positions -- that
 * scattered write stream is the cost of a radix sort and the reason the pass
 * skipping above pays off directly.
 *
 * Not vectorisable in portable C: every store depends on a counter that the
 * previous iteration may have incremented (a read-after-write on hist[pos] that
 * only AVX-512 conflict detection could break), and the stores themselves are a
 * scatter. The loops are left in the plain form the compiler handles best.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One stable counting-sort pass over 32-bit keys.
 * @param hist  Offsets for this pass, advanced as elements are placed.
 * @param shift Bit position of the digit.
 * @param in    Source keys.
 * @param out   Destination keys.
 * @param n     Number of keys.
 */
static inline void radixpass32(uint32_t *restrict hist, uint32_t shift, const uint32_t *restrict in, uint32_t *restrict out, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    const uint32_t k = in[i];
    const uint32_t pos = (k >> shift) & HIST_MASK;
    out[hist[pos]++] = k;
  }
}

/**
 * @brief One stable counting-sort pass over 64-bit keys.
 * @param hist  Offsets for this pass, advanced as elements are placed.
 * @param shift Bit position of the digit.
 * @param in    Source keys.
 * @param out   Destination keys.
 * @param n     Number of keys.
 */
static inline void radixpass64(uint32_t *restrict hist, uint32_t shift, const uint64_t *restrict in, uint64_t *restrict out, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    const uint64_t k = in[i];
    const uint32_t pos = (uint32_t) ((k >> shift) & HIST_MASK);
    out[hist[pos]++] = k;
  }
}

/* Key + value, SoA. Values are always uint32_t in the internal shuffle
 * regardless of key width -- they are int indices, checked by the static
 * assertion at the top of this file. */
/**
 * @brief One stable pass over 32-bit key/value pairs held as two arrays.
 * @param hist  Offsets for this pass, advanced as elements are placed.
 * @param shift Bit position of the digit.
 * @param ki    Source keys.
 * @param ko    Destination keys.
 * @param vi    Source values.
 * @param vo    Destination values.
 * @param n     Number of pairs.
 */
static inline void radixpass_soa32(uint32_t *restrict hist, uint32_t shift, const uint32_t *restrict ki, uint32_t *restrict ko, const uint32_t *restrict vi,
                                   uint32_t *restrict vo, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    const uint32_t k = ki[i];
    const uint32_t pos = (k >> shift) & HIST_MASK;
    const uint32_t idx = hist[pos]++;
    ko[idx] = k;
    vo[idx] = vi[i];
  }
}

/**
 * @brief One stable pass over 64-bit key / 32-bit value pairs.
 * @param hist  Offsets for this pass, advanced as elements are placed.
 * @param shift Bit position of the digit.
 * @param ki    Source keys.
 * @param ko    Destination keys.
 * @param vi    Source values.
 * @param vo    Destination values.
 * @param n     Number of pairs.
 */
static inline void radixpass_soa64(uint32_t *restrict hist, uint32_t shift, const uint64_t *restrict ki, uint64_t *restrict ko, const uint32_t *restrict vi,
                                   uint32_t *restrict vo, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    const uint64_t k = ki[i];
    const uint32_t pos = (uint32_t) ((k >> shift) & HIST_MASK);
    const uint32_t idx = hist[pos]++;
    ko[idx] = k; /* 64-bit key scatter */
    vo[idx] = vi[i]; /* 32-bit value scatter -- independent of key width */
  }
}

/* Allocation is via MFTLMallocOrDie (mftl_memory.h): returning quietly on a
 * failed malloc leaves the caller with an array it believes is sorted, after
 * which MFTLUniqueIntArray drops the wrong elements. In HPC code OOM is not
 * considered recoverable. */

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: key-only sorts
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRadixSort32(int *keys_in, int *keys_temp, int size)
{
  if (size < 2) return;
  if (size <= INSERTION_CUTOFF)
  {
    insertion_sort_keys32(keys_in, (size_t) size);
    return;
  }

  MFTL_ASSERT(keys_in != MFTL_NULL);
  MFTL_ASSERT(keys_temp != MFTL_NULL);

  const size_t n = (size_t) size;
  uint32_t *ui = (uint32_t *) keys_in;
  uint32_t *ut = (uint32_t *) keys_temp;

  _Alignas(64) uint32_t hist[PASSES_32 * HIST_SIZE]; /* 4 KB stack */

  if (init_histograms32(hist, ui, n)) return; /* already sorted */
  const uint32_t active = prefix_sum_histograms(hist, PASSES_32, n, (uint64_t) ui[0], true);

  /* Ping-pong between input and scratch. With constant digits skipped the
   * number of executed passes can be odd, so the result is copied back when it
   * lands in the scratch buffer -- one sequential memcpy in exchange for the
   * scatter pass that was skipped. */
  uint32_t *buf[2] = {ui, ut};
  uint32_t cur = 0u;
  for (uint32_t b = 0; b < PASSES_32; ++b)
  {
    if (!(active & (1u << b))) continue;
    radixpass32(hist + b * HIST_SIZE, b * RADIX_BITS, buf[cur], buf[cur ^ 1u], n);
    cur ^= 1u;
  }
  if (cur != 0u) memcpy(ui, ut, n * sizeof *ui);
}

void MFTLRadixSort64(uint64_t *keys_in, uint64_t *keys_temp, int size)
{
  if (size < 2) return;
  if (size <= INSERTION_CUTOFF)
  {
    insertion_sort_keys64(keys_in, (size_t) size);
    return;
  }

  MFTL_ASSERT(keys_in != MFTL_NULL);
  MFTL_ASSERT(keys_temp != MFTL_NULL);

  const size_t n = (size_t) size;

  _Alignas(64) uint32_t hist[PASSES_64 * HIST_SIZE]; /* 8 KB stack */

  if (init_histograms64(hist, keys_in, n)) return; /* already sorted */
  const uint32_t active = prefix_sum_histograms(hist, PASSES_64, n, keys_in[0], false);

  uint64_t *buf[2] = {keys_in, keys_temp};
  uint32_t cur = 0u;
  for (uint32_t b = 0; b < PASSES_64; ++b)
  {
    if (!(active & (1u << b))) continue;
    radixpass64(hist + b * HIST_SIZE, b * RADIX_BITS, buf[cur], buf[cur ^ 1u], n);
    cur ^= 1u;
  }
  if (cur != 0u) memcpy(keys_in, keys_temp, n * sizeof *keys_in);
}

void MFTLSortIds(int *data, int count)
{
  if (count < 2) return;
  if (count <= INSERTION_CUTOFF)
  {
    insertion_sort_keys32(data, (size_t) count);
    return;
  }

  int *temp = (int *) MFTLMallocOrDie(sizeof(int) * (size_t) count, "MFTLSortIds");
  MFTLRadixSort32(data, temp, count);
  free(temp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: key/value pair sorts (SoA)
 *
 * The histogram is built before the scratch buffers are allocated, so an
 * already-sorted or single-valued input returns without touching the allocator
 * at all.
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLSortPairs32(int *keys, int *values, int count)
{
  if (count < 2) return;
  if (count <= INSERTION_CUTOFF)
  {
    insertion_sort_pairs32(keys, values, (size_t) count);
    return;
  }

  MFTL_ASSERT(keys != MFTL_NULL);
  MFTL_ASSERT(values != MFTL_NULL);

  const size_t n = (size_t) count;
  uint32_t *ki = (uint32_t *) keys;
  uint32_t *vi = (uint32_t *) values;

  _Alignas(64) uint32_t hist[PASSES_32 * HIST_SIZE]; /* 4 KB stack */

  if (init_histograms32(hist, ki, n)) return; /* already sorted (stable: values stay put) */
  const uint32_t active = prefix_sum_histograms(hist, PASSES_32, n, (uint64_t) ki[0], true);
  if (active == 0u) return; /* no pass to run; guards against allocating for a no-op */

  uint32_t *kt = (uint32_t *) MFTLMallocOrDie(sizeof(int) * n, "MFTLSortPairs32 keys");
  uint32_t *vt = (uint32_t *) MFTLMallocOrDie(sizeof(int) * n, "MFTLSortPairs32 values");

  uint32_t *kb[2] = {ki, kt};
  uint32_t *vb[2] = {vi, vt};
  uint32_t cur = 0u;
  for (uint32_t b = 0; b < PASSES_32; ++b)
  {
    if (!(active & (1u << b))) continue;
    radixpass_soa32(hist + b * HIST_SIZE, b * RADIX_BITS, kb[cur], kb[cur ^ 1u], vb[cur], vb[cur ^ 1u], n);
    cur ^= 1u;
  }
  if (cur != 0u)
  {
    memcpy(ki, kt, n * sizeof *ki);
    memcpy(vi, vt, n * sizeof *vi);
  }

  free(kt);
  free(vt);
}

/* MFTLSortPairs64 -- the Morton sort case:
 *   keys   : uint64_t Morton codes     (8 bytes each, up to 8 passes)
 *   values : int original entity index (4 bytes each, shuffled with the keys) */
void MFTLSortPairs64(uint64_t *keys, int *values, int count)
{
  if (count < 2) return;
  if (count <= INSERTION_CUTOFF)
  {
    insertion_sort_pairs64(keys, values, (size_t) count);
    return;
  }

  MFTL_ASSERT(keys != MFTL_NULL);
  MFTL_ASSERT(values != MFTL_NULL);

  const size_t n = (size_t) count;
  uint32_t *vi = (uint32_t *) values;

  _Alignas(64) uint32_t hist[PASSES_64 * HIST_SIZE]; /* 8 KB stack */

  if (init_histograms64(hist, keys, n)) return; /* already sorted (stable: values stay put) */
  const uint32_t active = prefix_sum_histograms(hist, PASSES_64, n, keys[0], false);
  if (active == 0u) return; /* no pass to run; guards against allocating for a no-op */

  uint64_t *kt = (uint64_t *) MFTLMallocOrDie(sizeof(uint64_t) * n, "MFTLSortPairs64 keys");
  uint32_t *vt = (uint32_t *) MFTLMallocOrDie(sizeof(int) * n, "MFTLSortPairs64 values");

  uint64_t *kb[2] = {keys, kt};
  uint32_t *vb[2] = {vi, vt};
  uint32_t cur = 0u;
  for (uint32_t b = 0; b < PASSES_64; ++b)
  {
    if (!(active & (1u << b))) continue;
    radixpass_soa64(hist + b * HIST_SIZE, b * RADIX_BITS, kb[cur], kb[cur ^ 1u], vb[cur], vb[cur ^ 1u], n);
    cur ^= 1u;
  }
  if (cur != 0u)
  {
    memcpy(keys, kt, n * sizeof *keys);
    memcpy(vi, vt, n * sizeof *vi);
  }

  free(kt);
  free(vt);
}

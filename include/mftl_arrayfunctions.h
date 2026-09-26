/**
 * @file mftl_arrayfunctions.h
 * @author Christian Gorges
 * @date 2025-08-12
 * @ingroup mftl_arrayfunctions
 *
 * @brief Flat-array utilities: LSD radix sort and duplicate removal.
 *
 * These are plain array helpers -- they know nothing about MFTL types, which is
 * what puts them at layer 0 next to mftl_memory.h. Everything here is linear in
 * the array length: the sorts are least-significant-digit radix sorts over
 * 8-bit digits, the unique helpers are single scans over an already sorted
 * array.
 *
 * ### Ordering contract
 *
 *   int      keys  ->  SIGNED order    (..., -2, -1, 0, 1, ...)
 *   uint64_t keys  ->  UNSIGNED order  (0, 1, ..., UINT64_MAX)
 *
 * The key type in the signature *is* the contract; there is no separate flag.
 * Sorting an unsigned 32-bit quantity by casting it to int* is therefore wrong
 * for values >= 2^31 -- add an explicit u32 entry point if that case ever comes
 * up rather than reinterpreting one of these.
 *
 * ### Stability
 *
 * Every sort here is stable: equal keys keep their original relative order.
 * That is what makes the "sort pairs, then unique" idiom used by redistribution
 * deterministic -- the surviving duplicate is always the one that came first.
 */

#ifndef MFTL_ARRAYFUNCTIONS_H
#define MFTL_ARRAYFUNCTIONS_H

#include <stdint.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Typed function declarations
 *
 * These are the concrete implementations. Use the generic macros below
 * (MFTLRadixSort, MFTLSortPairs) for automatic dispatch; call the typed
 * functions directly when you need to be explicit about the key width.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Low-level key-only sort: 32-bit signed keys, caller-managed temp buffer.
 *
 * The result is always left in @p keys_in, whatever the internal pass count
 * turns out to be. @p keys_temp is scratch space and holds an unspecified
 * permutation of the input afterwards.
 *
 * @param keys_in   Keys to sort in place. Must hold @p size elements.
 * @param keys_temp Scratch buffer of at least @p size elements. Only read when
 *                  @p size exceeds the internal small-array cutoff, but must
 *                  always be valid.
 * @param size      Number of keys. Values < 2 are a no-op.
 */
void MFTLRadixSort32(int *keys_in, int *keys_temp, int size);

/**
 * @brief Low-level key-only sort: 64-bit unsigned keys, caller-managed temp buffer.
 *
 * Same contract as MFTLRadixSort32 -- the result always ends up in @p keys_in.
 *
 * @param keys_in   Array of 64-bit unsigned keys, sorted in place.
 * @param keys_temp Scratch array of the same length. Contents unspecified afterwards.
 * @param size      Number of elements.
 */
void MFTLRadixSort64(uint64_t *keys_in, uint64_t *keys_temp, int size);

/**
 * @brief High-level wrapper: sorts an int array in place, temp buffer internal.
 *
 * 32-bit only. For Morton codes use MFTLSortPairs / MFTLSortPairs64.
 * Allocation failure is fatal (the process aborts) -- a silently unsorted array
 * would be deduplicated wrongly by MFTLUniqueIntArray further downstream.
 *
 * @param data  Array of ints, sorted ascending in place.
 * @param count Number of elements.
 */
void MFTLSortIds(int *data, int count);

/**
 * @brief Sorts 32-bit key / 32-bit value pairs together (SoA layout).
 *
 * Keys and values live in two separate arrays and move as one, which avoids the
 * padding and the partial-cacheline writes an array of {key, value} structs
 * would cost. Temp buffers are allocated internally; allocation failure is fatal.
 *
 * @param keys   Signed keys. Sorted in place.
 * @param values Associated values, permuted to match the sorted keys.
 * @param count  Number of pairs.
 */
void MFTLSortPairs32(int *keys, int *values, int count);

/**
 * @brief Sorts 64-bit key / 32-bit int value pairs together (SoA layout).
 *
 * The Morton-code case: 64-bit spatial keys carrying the original entity index
 * along as the value. Temp buffers are allocated internally; allocation failure
 * is fatal.
 *
 * @param keys   64-bit unsigned keys. Sorted in place.
 * @param values Associated int indices, permuted to match the sorted keys.
 * @param count  Number of pairs.
 */
void MFTLSortPairs64(uint64_t *keys, int *values, int count);

/* ═══════════════════════════════════════════════════════════════════════════
 * Generic dispatch macros  (C11 _Generic -- compile-time, zero overhead)
 *
 *   MFTLRadixSort(keys_in, keys_temp, size)
 *       keys_in is int*      -> MFTLRadixSort32
 *       keys_in is uint64_t* -> MFTLRadixSort64
 *
 *   MFTLSortPairs(keys, values, count)
 *       keys is int*         -> MFTLSortPairs32
 *       keys is uint64_t*    -> MFTLSortPairs64
 *
 * C99 fallback (no _Generic): the macros call the 32-bit variants, so passing
 * 64-bit keys is a compile error rather than a silent mis-sort. Code that needs
 * 64-bit sorting under C99 must call MFTLSortPairs64 / MFTLRadixSort64 directly.
 * ═══════════════════════════════════════════════════════════════════════════ */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)

/**
 * @brief Key-only radix sort, dispatched on the key type at compile time.
 *
 * `int *` selects MFTLRadixSort32(), `uint64_t *` selects MFTLRadixSort64().
 * The selection is a C11 `_Generic` expression, so it costs nothing at run time
 * and a key type that is neither is a compile error.
 *
 * @param keys_in   Array to sort, in place.
 * @param keys_temp Scratch array of the same length and type. Caller-owned;
 *                  after the sort its contents are unspecified.
 * @param size      Number of elements.
 *
 * @note Under C99 (no `_Generic`) this expands to the 32-bit variant
 *       unconditionally, so passing 64-bit keys is a compile error rather than a
 *       silent mis-sort. Call MFTLRadixSort64() directly there.
 */
#define MFTLRadixSort(keys_in, keys_temp, size) _Generic((keys_in), int *: MFTLRadixSort32, uint64_t *: MFTLRadixSort64)((keys_in), (keys_temp), (size))

/**
 * @brief Key/value pair sort, dispatched on the key type at compile time.
 *
 * `int *` selects MFTLSortPairs32(), `uint64_t *` selects MFTLSortPairs64().
 *
 * @param keys   Key array, sorted in place.
 * @param values Value array, permuted alongside the keys (SoA layout).
 * @param count  Number of pairs.
 *
 * @note Under C99 this expands to the 32-bit variant unconditionally. See
 *       #MFTLRadixSort.
 */
#define MFTLSortPairs(keys, values, count) _Generic((keys), int *: MFTLSortPairs32, uint64_t *: MFTLSortPairs64)((keys), (values), (count))

#else /* C99 fallback */
/** @brief C99 fallback: always the 32-bit key sort. See the C11 definition. */
#define MFTLRadixSort(k, t, s) MFTLRadixSort32((k), (t), (s))
/** @brief C99 fallback: always the 32-bit pair sort. See the C11 definition. */
#define MFTLSortPairs(k, v, c) MFTLSortPairs32((k), (v), (c))
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Inline utility functions
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Collects the distinct values of an array, preserving first-seen order.
 *
 * O(input_count * num_unique) -- intended for the tiny fixed-capacity case
 * (a handful of destination ranks), not as a general deduplicator. For anything
 * larger, sort first and use MFTLUniqueIntArray, which is O(n) after the sort.
 *
 * @note Silently stops appending once @p output_capacity is reached; values
 *       beyond that are dropped, not reported. Size the output for the worst
 *       case the caller can actually produce.
 *
 * @param      input_array     Values to scan. Not modified, need not be sorted.
 * @param      input_count     Number of values in @p input_array.
 * @param[out] output_array    Receives the distinct values, in first-seen order.
 * @param      output_capacity Elements @p output_array can hold.
 * @return The number of distinct values written to @p output_array.
 */
static inline int MFTLGetUniqueIntegers(const int *input_array, int input_count, int *output_array, int output_capacity)
{
  int num_unique = 0;
  for (int j = 0; j < input_count; ++j)
  {
    int d = input_array[j];
    int found = 0;
    for (int k = 0; k < num_unique; ++k)
    {
      if (output_array[k] == d)
      {
        found = 1;
        break;
      }
    }
    if (!found && num_unique < output_capacity) output_array[num_unique++] = d;
  }
  return num_unique;
}

/**
 * @brief Binary search for @p key in an ascending int array.
 *
 * The library keeps its sparse, id-keyed directories as SORTED arrays so that
 * `id -> slot` is a binary search rather than a hash map; this is that search,
 * spelled once.
 *
 * @param sorted Ascending array.
 * @param n      Number of elements.
 * @param key    The value.
 * @return An index `i` with `sorted[i] == key`, or -1 (#MFTL_ERROR) if absent.
 *         With duplicate keys it is one of their indices.
 */
static inline int MFTLSortedIndexOf(const int *sorted, int n, int key)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    if (sorted[mid] == key) return mid;
    if (sorted[mid] < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

/**
 * @brief Removes adjacent duplicates from a SORTED int array, in place.
 *
 * Only compares neighbours, so the array must be sorted first (typically by
 * MFTLSortIds) -- on unsorted input it merely collapses runs.
 *
 * @param arr Sorted array, compacted in place.
 * @param n   Number of elements on entry.
 * @return The new element count; elements beyond it are left as garbage.
 */
static inline int MFTLUniqueIntArray(int *arr, int n)
{
  if (n <= 0) return 0;
  int dest = 0;
  for (int src = 1; src < n; ++src)
  {
    if (arr[src] != arr[dest]) arr[++dest] = arr[src];
  }
  return dest + 1;
}

/**
 * @brief Removes adjacent duplicate KEYS from a SORTED key/value pair, in place.
 *
 * The value kept for a run of equal keys is the first one. Combined with the
 * stability of the sorts above, that is the value whose entry came first in the
 * caller's original array -- a deterministic, rank-independent choice.
 *
 * @param keys   Sorted key array, compacted in place.
 * @param values Value array, compacted alongside @p keys.
 * @param n      Number of pairs on entry.
 * @return The new pair count.
 */
static inline int MFTLUniquePairs(int *keys, int *values, int n)
{
  if (n <= 0) return 0;
  int dest = 0;
  for (int src = 1; src < n; ++src)
  {
    if (keys[src] != keys[dest])
    {
      ++dest;
      keys[dest] = keys[src];
      values[dest] = values[src];
    }
  }
  return dest + 1;
}

#endif /* MFTL_ARRAYFUNCTIONS_H */

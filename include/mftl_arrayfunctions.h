/**
 * @file mftl_arrayfunctions.h
 * @author Christian Gorges
 * @date 2025-08-12
 * @ingroup mftl_arrayfunctions
 *
 * @brief Flat-array utilities: LSD radix sort, duplicate removal and a byte hash.
 *
 * Layer 0, no MFTL types. The sorts are least-significant-digit radix sorts
 * over 8-bit digits, \f$O(n)\f$ for a fixed key width (a stable insertion sort
 * below 33 elements); the unique helpers are single scans over sorted input.
 * Theory and measurements: src/arrayfunctions/README.md.
 *
 * **Ordering contract.** The key type is the contract: `int` keys sort in
 * SIGNED order, `uint64_t` keys in UNSIGNED order. Do not sort unsigned 32-bit
 * values through an `int *` cast (wrong for values \f$\ge 2^{31}\f$).
 *
 * **Stability.** Every sort is stable, so "sort pairs, then MFTLUniquePairs()"
 * keeps the value that came first -- a deterministic, rank-independent choice.
 */

#ifndef MFTL_ARRAYFUNCTIONS_H
#define MFTL_ARRAYFUNCTIONS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Typed entry points. The generic macros MFTLRadixSort / MFTLSortPairs below
 * dispatch to them on the key type.
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
 * @brief Low-level pair sort: 32-bit signed keys with int values, caller-managed
 *        scratch.
 *
 * The same sort as MFTLSortPairs32(); the scratch is the caller's, so a caller
 * that sorts repeatedly reuses one buffer and a caller with its own allocator
 * (the coupler's budget) accounts for every byte. The result is always left in
 * @p keys / @p values.
 *
 * @param keys        Keys, sorted in place.
 * @param values      Values, permuted alongside the keys.
 * @param keys_temp   Scratch of at least @p count elements, not aliasing
 *                    @p keys or @p values. Contents unspecified afterwards.
 * @param values_temp Scratch of at least @p count elements, likewise.
 * @param count       Number of pairs. Values < 2 are a no-op.
 */
void MFTLRadixSortPairs32(int *keys, int *values, int *keys_temp, int *values_temp, int count);

/**
 * @brief Low-level pair sort: 64-bit unsigned keys with int values, caller-managed
 *        scratch.
 *
 * The same sort as MFTLSortPairs64(), scratch as in MFTLRadixSortPairs32().
 *
 * @param keys        Keys, sorted in place.
 * @param values      Values, permuted alongside the keys.
 * @param keys_temp   Scratch of at least @p count elements, not aliasing
 *                    @p keys. Contents unspecified afterwards.
 * @param values_temp Scratch of at least @p count elements, not aliasing
 *                    @p values.
 * @param count       Number of pairs. Values < 2 are a no-op.
 */
void MFTLRadixSortPairs64(uint64_t *keys, int *values, uint64_t *keys_temp, int *values_temp, int count);

/**
 * @brief High-level wrapper: sorts an int array in place, temp buffer internal.
 *
 * 32-bit signed keys. Allocation failure aborts the process.
 *
 * @param data  Array of ints, sorted ascending in place.
 * @param count Number of elements.
 */
void MFTLSortIds(int *data, int count);

/**
 * @brief Sorts 32-bit key / 32-bit value pairs together (SoA layout).
 *
 * Keys and values live in two arrays and are permuted together. Scratch is
 * allocated internally (only when a pass actually runs); allocation failure
 * aborts. MFTLRadixSortPairs32() is the same sort with the caller's scratch.
 *
 * @param keys   Signed keys. Sorted in place.
 * @param values Associated values, permuted to match the sorted keys.
 * @param count  Number of pairs.
 */
void MFTLSortPairs32(int *keys, int *values, int count);

/**
 * @brief Sorts 64-bit key / 32-bit int value pairs together (SoA layout).
 *
 * The Morton-code case: 64-bit spatial keys carrying an entity index as the
 * value. Scratch is allocated internally; allocation failure aborts.
 * MFTLRadixSortPairs64() is the same sort with the caller's scratch.
 *
 * @param keys   64-bit unsigned keys. Sorted in place.
 * @param values Associated int indices, permuted to match the sorted keys.
 * @param count  Number of pairs.
 */
void MFTLSortPairs64(uint64_t *keys, int *values, int count);

/* ═══════════════════════════════════════════════════════════════════════════
 * Generic dispatch (C11 _Generic, resolved at compile time). Under C99 the
 * macros fall back to the 32-bit variants, so 64-bit keys fail to compile
 * rather than mis-sort.
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
 * \f$O(n \cdot u)\f$ for \f$u\f$ distinct values: for tiny sets (a handful of
 * ranks). For larger inputs sort, then MFTLUniqueIntArray().
 *
 * @note Values beyond @p output_capacity are dropped silently; size the output
 *       for the worst case.
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
 * Sparse id-keyed directories in the library are sorted arrays, so an
 * `id -> slot` lookup is this binary search rather than a hash map. It is a
 * lower bound: with duplicate keys it returns the FIRST of them, which after a
 * stable sort is the entry that came first.
 *
 * @param sorted Ascending array.
 * @param n      Number of elements.
 * @param key    The value.
 * @return The smallest `i` with `sorted[i] == key`, or -1 (#MFTL_ERROR) if absent.
 */
static inline int MFTLSortedIndexOf(const int *sorted, int n, int key)
{
  int lo = 0, hi = n;
  while (lo < hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    if (sorted[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (lo < n && sorted[lo] == key) ? lo : -1;
}

/**
 * @brief Binary search for @p key in an ascending `uint64_t` array.
 *
 * MFTLSortedIndexOf() for 64-bit keys (lattice keys, Morton cells, solver
 * sample keys), in UNSIGNED order, the order MFTLRadixSort64() and
 * MFTLSortPairs64() produce. The count is 64-bit because the coupler's sample
 * arrays are; an `int` count converts implicitly.
 *
 * @param sorted Ascending array.
 * @param n      Number of elements.
 * @param key    The value.
 * @return The smallest `i` with `sorted[i] == key`, or -1 if absent.
 */
static inline int64_t MFTLSortedIndexOf64(const uint64_t *sorted, int64_t n, uint64_t key)
{
  int64_t lo = 0, hi = n;
  while (lo < hi)
  {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (sorted[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (lo < n && sorted[lo] == key) ? lo : -1;
}

/**
 * @brief Removes adjacent duplicates from a SORTED int array, in place.
 *
 * Compares neighbours only, so on unsorted input it merely collapses runs.
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
 * @brief Removes adjacent duplicates from a SORTED `uint64_t` array, in place.
 *
 * MFTLUniqueIntArray() for 64-bit keys; the count is 64-bit as in
 * MFTLSortedIndexOf64().
 *
 * @param arr Sorted array, compacted in place.
 * @param n   Number of elements on entry.
 * @return The new element count; elements beyond it are left as garbage.
 */
static inline int64_t MFTLUniqueU64Array(uint64_t *arr, int64_t n)
{
  if (n <= 0) return 0;
  int64_t dest = 0;
  for (int64_t src = 1; src < n; ++src)
  {
    if (arr[src] != arr[dest]) arr[++dest] = arr[src];
  }
  return dest + 1;
}

/**
 * @brief Removes adjacent duplicate KEYS from a SORTED key/value pair, in place.
 *
 * Keeps the first value of each run of equal keys; after a stable sort that is
 * the entry that came first in the caller's original array.
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

/** @brief The FNV-1a 64-bit offset basis: the seed of MFTLHashBytes(). */
#define MFTL_HASH_SEED 0xcbf29ce484222325ull

/**
 * @brief FNV-1a, 64 bit: folds @p n bytes into the running hash @p h.
 *
 * The library's one byte hash, for agreement checks and signatures, where only
 * equality is asked of it (not a hash-map hash, not cryptographic). Chain calls
 * to hash several blocks; start from #MFTL_HASH_SEED. A struct's bytes include
 * its padding, so a struct to be hashed is zeroed before it is filled.
 *
 * @param h    The running hash (#MFTL_HASH_SEED for the first block).
 * @param data The bytes. May be NULL when @p n is 0.
 * @param n    Their number.
 * @return The hash with the bytes folded in.
 */
static inline uint64_t MFTLHashBytes(uint64_t h, const void *data, size_t n)
{
  const unsigned char *const p = (const unsigned char *) data;
  for (size_t i = 0; i < n; ++i)
  {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

#ifdef __cplusplus
}
#endif

#endif /* MFTL_ARRAYFUNCTIONS_H */

/*
 * MFTL2 core — integer sorting (LSD radix sort, src/core/radixsort.c) and de-duplication
 * helpers for id arrays.
 */
#ifndef MFTL_ARRAYFUNCTIONS_H
#define MFTL_ARRAYFUNCTIONS_H

#include <stdint.h>
#include <stddef.h>
#include "mftl_core.h"

/**
 * Sorts count ints in place (allocates a temporary buffer internally).
 */
void MFTLSortIds(int* data, int count);

/**
 * Sorts keys in place and applies the same permutation to values (structure of arrays).
 * @param keys   Array of IDs to sort.
 * @param values Array of associated data (e.g. original indices) to move with keys.
 * @param count  Number of elements.
 */
void MFTLSortPairs(int* keys, int* values, int count);

/**
 * Core sort with a caller-provided temporary buffer (re-usable across many sorts).
 * @param keys_in   Input array to sort.
 * @param keys_temp Temporary buffer of same size as keys_in.
 * @param size      Number of elements.
 */
void MFTLRadixSort(int* keys_in, int* keys_temp, int size);

/**
 * @brief Copies the distinct integers of input_array to output_array in first-occurrence order.
 *
 * O(N^2): intended for short lists (e.g. neighbour ranks). For long arrays sort first and use
 * MFTLUniqueIntArray. output_capacity is not checked; output_array must hold input_count items.
 *
 * @param input_array     A pointer to the const input array of integers.
 * @param input_count     The number of elements in the input array.
 * @param output_array    A pointer to the output array to store the unique integers.
 * @param output_capacity The maximum number of elements the output array can hold.
 * @return The number of unique integers found and placed in the output array.
 */
static inline int MFTLGetUniqueIntegers(const int* input_array, int input_count, int* output_array, int output_capacity)
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

    if (!found)
    {
      output_array[num_unique++] = d;
    }
  }
  return num_unique;
}

/**
 * @brief Removes consecutive duplicates from a SORTED array in place, O(N).
 *
 * @param arr  A pointer to the integer array to be deduplicated. 
 * The array is modified in-place.
 * @param n    The number of elements in the array.
 * @return     The new number of unique elements in the array.
 */
static inline int MFTLUniqueIntArray(int* arr, int n)
{
  if (n == 0) return 0;

  int dest = 0;
  for (int src = 1; src < n; ++src)
  {
    if (arr[src] != arr[dest])
    {
      dest++;
      arr[dest] = arr[src];
    }
  }
  return dest + 1;
}

/**
 * @brief Removes duplicate keys from a SORTED key array in place and compacts the parallel
 * value array in step; for equal keys the first entry is kept.
 *
 * @param keys   Pointer to the sorted array of keys (IDs).
 * @param values Pointer to the array of associated values.
 * @param n      The number of elements in the arrays.
 * @return       The new number of unique elements.
 */
static inline int MFTLUniquePairs(int* keys, int* values, int n)
{
  if (n <= 0) return 0;

  int dest = 0;
  for (int src = 1; src < n; ++src)
  {
    if (keys[src] != keys[dest])
    {
      dest++;
      keys[dest] = keys[src];
      values[dest] = values[src];
    }
  }
  return dest + 1;
}

#endif  // MFTL_ARRAYFUNCTIONS_H
#ifndef MFTL_ARRAYFUNCTIONS_H
#define MFTL_ARRAYFUNCTIONS_H

#include <stdint.h>
#include <stddef.h>  // For size_t
#include "mftl_core.h"

/**
 * High-level wrapper. 
 * Allocates temporary memory internally and sorts the data in-place.
 */
void MFTLSortIds(mftl_int_t* data, mftl_int_t count);

/**
 * Sorts keys and moves associated values in parallel.
 * Fastest possible approach (Structure of Arrays).
 * @param keys   Array of IDs to sort.
 * @param values Array of associated data (e.g. original indices) to move with keys.
 * @param count  Number of elements.
 */
void MFTLSortPairs(mftl_int_t* keys, mftl_int_t* values, mftl_int_t count);

/**
 * Low-level core sort.
 * Use this if you want to manage the temporary memory buffer yourself
 * (e.g., re-using one buffer for many sorts to reduce malloc overhead).
 * @param keys_in   Input array to sort.
 * @param keys_temp Temporary buffer of same size as keys_in.
 * @param size      Number of elements.
 */
void MFTLRadixSort(mftl_int_t* keys_in, mftl_int_t* keys_temp, mftl_int_t size);

/**
 * @brief Finds the unique integers in a given input array.
 *
 * This function iterates through an array of integers and populates an output
 * array with only the unique elements. For small input sizes, this simple
 * O(N^2) approach is highly efficient as it avoids function call overhead.
 * For N > 64, consider sorting the array first and then using unique_int_array.
 *
 * @param input_array     A pointer to the const input array of integers.
 * @param input_count     The number of elements in the input array.
 * @param output_array    A pointer to the output array to store the unique integers.
 * @param output_capacity The maximum number of elements the output array can hold.
 * @return The number of unique integers found and placed in the output array.
 */
static inline mftl_int_t MFTLGetUniqueIntegers(const mftl_int_t* input_array, mftl_int_t input_count, mftl_int_t* output_array, mftl_int_t output_capacity)
{
  mftl_int_t num_unique = 0;
  for (mftl_int_t j = 0; j < input_count; ++j)
  {
    mftl_int_t d = input_array[j];
    mftl_int_t found = 0;

    // Check if we've already added this integer to our output list.
    for (mftl_int_t k = 0; k < num_unique; ++k)
    {
      if (output_array[k] == d)
      {
        found = 1;
        break;
      }
    }

    // If it wasn't found, add it to the list.
    if (!found)
    {
      output_array[num_unique++] = d;
    }
  }
  return num_unique;
}

/**
 * @brief Removes duplicate integers from a sorted array in-place.
 *
 * This function iterates through a sorted array and compacts it by removing 
 * consecutive duplicates. It operates in O(N) linear time and is extremely 
 * cache-efficient as it reads and writes sequentially.
 * * @warning The input array MUST be sorted (e.g. via qsort or radix sort) for this function 
 * to work correctly.
 *
 * @param arr  A pointer to the integer array to be deduplicated. 
 * The array is modified in-place.
 * @param n    The number of elements in the array.
 * @return     The new number of unique elements in the array.
 */
static inline mftl_int_t MFTLUniqueIntArray(mftl_int_t* arr, mftl_int_t n)
{
  if (n == 0) return 0;

  mftl_int_t dest = 0;
  for (mftl_int_t src = 1; src < n; ++src)
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
 * @brief Removes duplicate entries from sorted parallel arrays (SoA) in-place.
 *
 * This function iterates through the sorted `keys` array. When it finds a
 * unique key, it keeps both that key AND the corresponding value from the
 * `values` array. It effectively compacts both arrays in sync.
 *
 * @warning The `keys` array MUST be sorted for this function to work.
 * @note If multiple entries have the same key, the first one encountered 
 * is kept, and subsequent duplicates are discarded.
 *
 * @param keys   Pointer to the sorted array of keys (IDs).
 * @param values Pointer to the array of associated values.
 * @param n      The number of elements in the arrays.
 * @return       The new number of unique elements.
 */
static inline mftl_int_t MFTLUniquePairs(mftl_int_t* keys, mftl_int_t* values, mftl_int_t n)
{
  if (n <= 0) return 0;

  mftl_int_t dest = 0;
  for (mftl_int_t src = 1; src < n; ++src)
  {
    // Check for unique key
    if (keys[src] != keys[dest])
    {
      dest++;
      keys[dest] = keys[src];  // Compact Key
      values[dest] = values[src];  // Compact Value (carry payload)
    }
  }
  return dest + 1;
}

#endif  // MFTL_ARRAYFUNCTIONS_H
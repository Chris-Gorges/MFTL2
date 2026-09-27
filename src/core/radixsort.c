/*
 * MFTL2 core — stable LSD radix sort of int keys (8-bit digits, 4 passes), optionally moving a
 * parallel value array; the implementation behind mftl_arrayfunctions.h. Keys are ordered as
 * UNSIGNED 32-bit values (the sorted ids and Morton codes are non-negative). Adapted from the
 * zlib-licensed code below.
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
 *
 * Modified for MFTL Library:
 * - Unified 32/64-bit support via MFTL_USE_64BIT
 * - Removed float/value support
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "mftl_core.h"
#include "mftl_arrayfunctions.h"

// -----------------------------------------------------------------------------
// Internal Config & Macros (Not visible outside this file)
// -----------------------------------------------------------------------------

#ifdef MFTL_USE_64BIT
// Internal unsigned type for bitwise operations
typedef uint64_t mftl_uint_t;
// 64 bits / 8 bits = 8 passes (Even number)
#define MFTL_RADIX_PASSES 8
#else
// Internal unsigned type for bitwise operations
typedef uint32_t mftl_uint_t;
// 32 bits / 8 bits = 4 passes (Even number)
#define MFTL_RADIX_PASSES 4
#endif

#define RADIX_BITS 8
#define HIST_SIZE (1 << RADIX_BITS)

// -----------------------------------------------------------------------------
// Internal Static Helpers
// -----------------------------------------------------------------------------

static void init_histograms(uint32_t* __restrict hist, const mftl_uint_t* __restrict keys_in, const size_t size)
{
  memset(hist, 0, sizeof(uint32_t) * MFTL_RADIX_PASSES * HIST_SIZE);
  const uint32_t kHistMask = HIST_SIZE - 1;

  for (size_t i = 0; i < size; ++i)
  {
    const mftl_uint_t key = keys_in[i];
    for (uint32_t bucket = 0; bucket < MFTL_RADIX_PASSES; ++bucket)
    {
      const uint32_t shift = bucket * RADIX_BITS;
      const uint32_t pos = (uint32_t) ((key >> shift) & kHistMask);
      hist[(bucket * HIST_SIZE) + pos]++;
    }
  }
}

static void sum_histograms(uint32_t* __restrict sum, uint32_t* __restrict hist)
{
  for (uint32_t bucket = 0; bucket < MFTL_RADIX_PASSES; ++bucket)
  {
    uint32_t* __restrict offset = hist + (bucket * HIST_SIZE);
    sum[bucket] = offset[0];
    offset[0] = 0;
  }

  uint32_t tsum;
  for (uint32_t i = 1; i < HIST_SIZE; ++i)
  {
    for (uint32_t bucket = 0; bucket < MFTL_RADIX_PASSES; ++bucket)
    {
      uint32_t* __restrict offset = hist + (bucket * HIST_SIZE);
      tsum = offset[i] + sum[bucket];
      offset[i] = sum[bucket];
      sum[bucket] = tsum;
    }
  }
}

static inline void radixpass(uint32_t* __restrict hist, const uint32_t shift, const mftl_uint_t* __restrict keys_in, mftl_uint_t* __restrict keys_out,
                             const size_t size)
{
  const uint32_t kHistMask = HIST_SIZE - 1;
  for (size_t i = 0; i < size; ++i)
  {
    const mftl_uint_t key = keys_in[i];
    const uint32_t pos = (uint32_t) ((key >> shift) & kHistMask);
    const uint32_t index = hist[pos]++;
    keys_out[index] = key;
  }
}

// -----------------------------------------------------------------------------
// Public Implementation
// -----------------------------------------------------------------------------

void MFTLRadixSort(int* keys_in, int* keys_temp, int size)
{
  if (size < 0) return;

  mftl_uint_t* __restrict keys_u_in = (mftl_uint_t*) keys_in;
  mftl_uint_t* __restrict keys_u_temp = (mftl_uint_t*) keys_temp;

  uint32_t hist[MFTL_RADIX_PASSES * HIST_SIZE];
  uint32_t sum[MFTL_RADIX_PASSES];

  size_t u_size = (size_t) size;

  init_histograms(hist, keys_u_in, u_size);
  sum_histograms(sum, hist);

  mftl_uint_t* __restrict buffers[2] = {keys_u_in, keys_u_temp};

  for (uint32_t bucket = 0; bucket < MFTL_RADIX_PASSES; ++bucket)
  {
    const uint32_t in_idx = bucket & 1;
    const uint32_t out_idx = !in_idx;

    uint32_t* __restrict offset = hist + (bucket * HIST_SIZE);

    radixpass(offset, bucket * RADIX_BITS, buffers[in_idx], buffers[out_idx], u_size);
  }
}

void MFTLSortIds(int* data, int count)
{
  if (count < 2) return;

  int* temp = (int*) malloc(sizeof(int) * (size_t) count);

  if (temp)
  {
    MFTLRadixSort(data, temp, count);
    free(temp);
  }
}

// =============================================================================
// Key/value sort (separate key and value arrays)
// =============================================================================

static inline void radixpass_soa(uint32_t* __restrict hist, const uint32_t shift, const mftl_uint_t* __restrict keys_in, mftl_uint_t* __restrict keys_out,
                                 const mftl_uint_t* __restrict vals_in, mftl_uint_t* __restrict vals_out, const size_t size)
{
  const uint32_t kHistMask = HIST_SIZE - 1;
  for (size_t i = 0; i < size; ++i)
  {
    const mftl_uint_t key = keys_in[i];

    const uint32_t pos = (uint32_t) ((key >> shift) & kHistMask);
    const uint32_t index = hist[pos]++;

    keys_out[index] = key;
    vals_out[index] = vals_in[i];
  }
}

void MFTLSortPairs(int* keys, int* values, int count)
{
  if (count < 2) return;

  size_t u_count = (size_t) count;

  int* temp_keys = (int*) malloc(sizeof(int) * u_count);
  int* temp_vals = (int*) malloc(sizeof(int) * u_count);

  if (!temp_keys || !temp_vals)
  {
    free(temp_keys);
    free(temp_vals);
    return;
  }

  mftl_uint_t* __restrict k_in = (mftl_uint_t*) keys;
  mftl_uint_t* __restrict k_tmp = (mftl_uint_t*) temp_keys;

  mftl_uint_t* __restrict v_in = (mftl_uint_t*) values;
  mftl_uint_t* __restrict v_tmp = (mftl_uint_t*) temp_vals;

  uint32_t hist[MFTL_RADIX_PASSES * HIST_SIZE];
  uint32_t sum[MFTL_RADIX_PASSES];

  init_histograms(hist, k_in, u_count);
  sum_histograms(sum, hist);

  mftl_uint_t* __restrict k_buf[2] = {k_in, k_tmp};
  mftl_uint_t* __restrict v_buf[2] = {v_in, v_tmp};

  for (uint32_t bucket = 0; bucket < MFTL_RADIX_PASSES; ++bucket)
  {
    const uint32_t in_idx = bucket & 1;
    const uint32_t out_idx = !in_idx;

    radixpass_soa(hist + (bucket * HIST_SIZE), bucket * RADIX_BITS, k_buf[in_idx], k_buf[out_idx], v_buf[in_idx], v_buf[out_idx], u_count);
  }

  free(temp_keys);
  free(temp_vals);
}
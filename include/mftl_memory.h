/*
 * @Author: Christian Gorges
 * @Date: 2025-10-27
 */

#ifndef MFTL_MEMORY_H
#define MFTL_MEMORY_H

#include <stdlib.h>
#include <stdio.h>
#include "mftl_constants.h"

/**
 * @brief Ensures an array has enough capacity to hold 'needed' additional items.
 *
 * This macro implements a geometric growth strategy (x2) to amortize allocation costs.
 * It strictly handles allocation failures by printing an error and exiting,
 * which is standard practice for HPC applications where recovery from OOM is impractical.
 *
 * @param ptr      Pointer to the dynamic array (will be updated).
 * @param capacity Pointer to the current capacity (will be updated).
 * @param target_cap The target capacity (will be updated).
 * @param type     The type of the elements (for sizeof).
 */
#define _MFTL_REALLOC_CORE(ptr, capacity, target_cap, type)                                                            \
  do                                                                                                                   \
  {                                                                                                                    \
    int _req_cap = (target_cap);                                                                                       \
    /* Safety: Only realloc if we actually need more space */                                                          \
    if (_req_cap > (*(capacity)))                                                                                      \
    {                                                                                                                  \
      void *_new_ptr = realloc((ptr), _req_cap * sizeof(type));                                                        \
      if (!_new_ptr && _req_cap > 0) /* Allow 0-size reallocs if valid */                                              \
      {                                                                                                                \
        fprintf(stderr, "FATAL ERROR: MFTL Alloc failed for %lu bytes.\n", (unsigned long) (_req_cap * sizeof(type))); \
        exit(EXIT_FAILURE);                                                                                            \
      }                                                                                                                \
      (ptr) = (type *) _new_ptr;                                                                                       \
      (*(capacity)) = _req_cap;                                                                                        \
    }                                                                                                                  \
  } while (0)

/**
 * @brief Default Geometric Growth (x2).
 * Use this for incremental additions (e.g., inside a loop).
 */
#define MFTL_ENSURE_CAPACITY(ptr, count, capacity, needed, type)              \
  do                                                                          \
  {                                                                           \
    if ((*(count)) + (needed) > (*(capacity)))                                \
    {                                                                         \
      int _new_cap = (*(capacity) == 0) ? MFTL_INIT_CAPACITY : (*(capacity)); \
      /* Geometric doubling until we fit the need */                          \
      while (_new_cap < (*(count)) + (needed))                                \
      {                                                                       \
        _new_cap *= 2;                                                        \
      }                                                                       \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);                      \
    }                                                                         \
  } while (0)

/**
 * @brief Exact Growth.
 * Use this when you know exactly how many items you are about to add (Bulk Loading).
 * This saves memory by avoiding the "x2" overhead.
 */
#define MFTL_RESERVE_ADDITIONAL(ptr, count, capacity, needed, type) \
  do                                                                \
  {                                                                 \
    if ((*(count)) + (needed) > (*(capacity)))                      \
    {                                                               \
      int _new_cap = (*(count)) + (needed);                         \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);            \
    }                                                               \
  } while (0)
#endif  // MFTL_MEMORY_H

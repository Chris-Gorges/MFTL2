/*
 * MFTL2 core — growth of the flat entity arrays (MFTL_ENSURE_CAPACITY / MFTL_RESERVE_ADDITIONAL).
 *
 * Grown regions are zero-filled, and an allocation failure terminates the program (recovery
 * from out-of-memory is impractical in an MPI job).
 */

#ifndef MFTL_MEMORY_H
#define MFTL_MEMORY_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mftl_constants.h"

/**
 * @brief Grows *capacity to target_cap (no-op when it is already large enough).
 *
 * The newly grown region is always zeroed, so slots handed out from fresh
 * capacity never inherit garbage from realloc'd memory (in particular stray
 * flag bits such as MFTL_FLAG_VISITED, which silently corrupt flag-based
 * traversals like MFTLGetNeighborhood and get transported to ghost copies on
 * other ranks).
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
                                                                                                                       \
    if (_req_cap > (*(capacity)))                                                                                      \
    {                                                                                                                  \
      int _old_cap = (*(capacity));                                                                                    \
      void *_new_ptr = realloc((ptr), _req_cap * sizeof(type));                                                        \
      if (!_new_ptr && _req_cap > 0) /* Allow 0-size reallocs if valid */                                              \
      {                                                                                                                \
        fprintf(stderr, "FATAL ERROR: MFTL Alloc failed for %lu bytes.\n", (unsigned long) (_req_cap * sizeof(type))); \
        exit(EXIT_FAILURE);                                                                                            \
      }                                                                                                                \
      (ptr) = (type *) _new_ptr;                                                                                       \
      (*(capacity)) = _req_cap;                                                                                        \
      /* Zero the grown region so entities handed out from fresh capacity (verts,  */                                 \
      /* tris, half-edges, ...) never inherit garbage from realloc'd memory — in    */                                 \
      /* particular stray flag bits (e.g. MFTL_FLAG_VISITED) which corrupt          */                                 \
      /* flag-based traversals and are transported to ghost copies on other ranks.  */                                 \
      memset((ptr) + _old_cap, 0, ((size_t) _req_cap - (size_t) _old_cap) * sizeof(type));                             \
    }                                                                                                                  \
  } while (0)

/**
 * @brief Geometric growth (x2, starting at MFTL_INIT_CAPACITY) so that `needed` more items fit;
 * for incremental additions.
 */
#define MFTL_ENSURE_CAPACITY(ptr, count, capacity, needed, type)              \
  do                                                                          \
  {                                                                           \
    if ((*(count)) + (needed) > (*(capacity)))                                \
    {                                                                         \
      int _new_cap = (*(capacity) == 0) ? MFTL_INIT_CAPACITY : (*(capacity)); \
                                                                              \
      while (_new_cap < (*(count)) + (needed))                                \
      {                                                                       \
        _new_cap *= 2;                                                        \
      }                                                                       \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);                      \
    }                                                                         \
  } while (0)

/**
 * @brief Exact growth to count + needed; for bulk loading when the number of items is known.
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

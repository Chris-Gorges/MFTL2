/**
 * @file mftl_memory.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_memory
 *
 * @brief Allocation policy, cache-line-aligned blocks, and array growth.
 *
 * Layer 0. Two distinct jobs live here and it is worth keeping them apart:
 *
 *   SMALL, INDEPENDENT ARRAYS  -- the ensemble's object list, scratch buffers
 *   inside an algorithm. These use MFTLMallocOrDie / MFTLCallocOrDie and the
 *   growth macros.
 *
 *   POOL BLOCKS -- the vertex and triangle pools of an MFTL_Object. Those are
 *   NOT independent arrays: the AoS topology records and every SoA geometry
 *   array describe the SAME entities and must have the SAME capacity, always.
 *   Holding that together with three capacity fields and a rule ("every site that
 *   grows a pool must call Reserve") fails at the first missed site, and it fails
 *   as a heap overflow rather than as a wrong number.
 *
 *   Here the rule is replaced by a structure: one pool is ONE allocation, cut
 *   into cache-line-aligned sub-arrays by MFTLLayoutAdd below. There is one
 *   pointer to free, one capacity to update, and no way to grow half of it.
 *   src/object/object.c is the only file that calls MFTLLayoutAdd.
 *
 * Allocation failure is fatal everywhere. In HPC code OOM is not recoverable,
 * and a caller that quietly carries on with a NULL buffer corrupts data far away
 * from the actual failure.
 */

#ifndef MFTL_MEMORY_H
#define MFTL_MEMORY_H

#include <stdio.h>
#include <stdlib.h>

#include "mftl_constants.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Fail-fast allocation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief `malloc` that aborts instead of returning NULL.
 *
 * @param bytes Number of bytes to allocate.
 * @param what  Human-readable name of the caller, printed in the fatal message.
 * @return A pointer to @p bytes uninitialised bytes. Never NULL -- the process
 *         exits instead.
 */
static inline void *MFTLMallocOrDie(size_t bytes, const char *what)
{
  void *p = malloc(bytes);
  if (!p)
  {
    fprintf(stderr, "FATAL ERROR: %s allocation failed for %lu bytes.\n", what, (unsigned long) bytes);
    exit(EXIT_FAILURE);
  }
  return p;
}

/**
 * @brief `calloc` that aborts instead of returning NULL.
 *
 * @param count Number of elements.
 * @param size  Size of one element in bytes.
 * @param what  Human-readable name of the caller, printed in the fatal message.
 * @return A pointer to @p count zeroed elements. Never NULL.
 */
static inline void *MFTLCallocOrDie(size_t count, size_t size, const char *what)
{
  void *p = calloc(count, size);
  if (!p)
  {
    fprintf(stderr, "FATAL ERROR: %s allocation failed for %lu elements of %lu bytes.\n", what, (unsigned long) count, (unsigned long) size);
    exit(EXIT_FAILURE);
  }
  return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Cache-line-aligned block allocation and sub-array layout
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Rounds @p x up to the next multiple of @p a.
 *
 * @param x Value to round.
 * @param a Alignment. **Must be a power of two** -- the implementation is a mask,
 *          not a modulo, so a non-power-of-two silently returns nonsense.
 * @return The smallest multiple of @p a that is not less than @p x.
 */
static inline size_t MFTLAlignUp(size_t x, size_t a) { return (x + a - 1u) & ~(a - 1u); }

/**
 * @brief Allocates @p bytes, 64-byte aligned, aborting on failure.
 *
 * C11 aligned_alloc requires the size to be a multiple of the alignment, so the
 * request is rounded up here rather than at every call site. The result is freed
 * with plain free().
 *
 * The alignment is what lets a pool block be cut into sub-arrays that each start
 * on a cache line: no false sharing between two components a kernel writes, and
 * no vectoriser peel loop at the head of a streaming loop.
 *
 * @param bytes Requested size. Rounded up internally; 0 is treated as 1.
 * @param what  Human-readable name of the caller, printed in the fatal message.
 * @return A pointer aligned to #MFTL_CACHE_LINE bytes. Never NULL. Release it
 *         with plain `free()`.
 */
static inline void *MFTLAlignedAllocOrDie(size_t bytes, const char *what)
{
  const size_t padded = MFTLAlignUp(bytes ? bytes : 1u, MFTL_CACHE_LINE);
  void *p = aligned_alloc(MFTL_CACHE_LINE, padded);
  if (!p)
  {
    fprintf(stderr, "FATAL ERROR: %s aligned allocation failed for %lu bytes.\n", what, (unsigned long) padded);
    exit(EXIT_FAILURE);
  }
  return p;
}

/**
 * @brief Reserves one sub-array inside a pool block and returns its byte offset.
 *
 * Call it once per field, in order, starting from a zeroed @p cursor; afterwards
 * @p cursor holds the total block size. Every returned offset is a multiple of
 * MFTL_CACHE_LINE.
 *
 *     size_t cursor = 0;
 *     const size_t o_rec = MFTLLayoutAdd(&cursor, cap,     sizeof(MFTL_Vertex));
 *     const size_t o_pos = MFTLLayoutAdd(&cursor, 3 * cap, sizeof(double));
 *     ...
 *     unsigned char *block = MFTLAlignedAllocOrDie(cursor, "vertex pool");
 *
 * The point of doing it this way rather than with one malloc per field: the
 * capacity is a property of the BLOCK, so "AoS grew but the SoA arrays did not"
 * is not an error that can be made. It is not expressible.
 *
 * @param cursor    Running block size in bytes, updated in place. Start at 0.
 * @param count     Number of elements in this sub-array.
 * @param elem_size Size of one element in bytes.
 * @return Byte offset of the sub-array from the start of the block. Always a
 *         multiple of #MFTL_CACHE_LINE.
 */
static inline size_t MFTLLayoutAdd(size_t *cursor, size_t count, size_t elem_size)
{
  const size_t offset = *cursor;
  *cursor = MFTLAlignUp(offset + count * elem_size, MFTL_CACHE_LINE);
  return offset;
}

/**
 * @brief Geometric growth policy: at least @p needed, at least MFTL_INIT_CAPACITY,
 *        otherwise double until it fits.
 *
 * Shared by both pools so that a mesh which grows entity by entity performs
 * \f$O(\log n)\f$ reallocations rather than \f$O(n)\f$, and so that both pools
 * follow the same schedule.
 *
 * @param current Present capacity. May be 0.
 * @param needed  Capacity that must fit afterwards.
 * @return @p current if it already suffices, otherwise the first value in
 *         \f$\{\texttt{MFTL\_INIT\_CAPACITY} \cdot 2^k\}\f$ that is at least
 *         @p needed. Saturates to exactly @p needed near `INT_MAX` rather than
 *         overflowing into a negative -- i.e. a silently undersized allocation.
 */
static inline int MFTLGrowCapacity(int current, int needed)
{
  if (needed <= current) return current;
  int cap = (current > 0) ? current : MFTL_INIT_CAPACITY;
  while (cap < needed)
  {
    /* int overflow would turn a huge request into a negative capacity, i.e. a
     * silently undersized allocation. Saturate instead and let the allocator
     * report the real problem. */
    if (cap > (2147483647 / 2)) return needed;
    cap *= 2;
  }
  return cap;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Growth macros for plain, independent arrays
 *
 * These are for arrays that genuinely stand alone (the ensemble's object list,
 * per-neighbour scratch buffers). Do NOT use them on anything that lives in a
 * pool block -- there the capacity belongs to the block, and the entry points in
 * mftl_object.h are the only correct way to change it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Implementation shared by #MFTL_ENSURE_CAPACITY and #MFTL_RESERVE_ADDITIONAL.
 *
 * Reallocates only when @p target_cap exceeds the present capacity, aborts on
 * failure, and writes the new capacity back through @p capacity.
 *
 * @param ptr        Lvalue holding the array pointer. Updated in place.
 * @param capacity   Pointer to the capacity counter. Updated in place.
 * @param target_cap Capacity wanted, in elements.
 * @param type       Element type, used for `sizeof` and for the cast.
 *
 * @internal Not part of the public API; use one of the two wrappers.
 */
#define _MFTL_REALLOC_CORE(ptr, capacity, target_cap, type)                                                             \
  do                                                                                                                    \
  {                                                                                                                     \
    int _req_cap = (target_cap);                                                                                        \
    if (_req_cap > (*(capacity)))                                                                                       \
    {                                                                                                                   \
      void *_new_ptr = realloc((ptr), (size_t) _req_cap * sizeof(type));                                                \
      if (!_new_ptr && _req_cap > 0)                                                                                    \
      {                                                                                                                 \
        fprintf(stderr, "FATAL ERROR: MFTL Alloc failed for %lu bytes.\n", (unsigned long) ((size_t) _req_cap * sizeof(type))); \
        exit(EXIT_FAILURE);                                                                                             \
      }                                                                                                                 \
      (ptr) = (type *) _new_ptr;                                                                                        \
      (*(capacity)) = _req_cap;                                                                                         \
    }                                                                                                                   \
  } while (0)

/**
 * @brief Geometric growth (&times;2). For incremental appends inside a loop.
 *
 * Grows only when `*count + needed` exceeds `*capacity`, following
 * MFTLGrowCapacity()'s doubling schedule, so \f$n\f$ appends cost
 * \f$O(\log n)\f$ reallocations.
 *
 * @param ptr      Lvalue holding the array pointer. Updated in place.
 * @param count    Pointer to the element counter. Read, never written.
 * @param capacity Pointer to the capacity counter. Updated in place.
 * @param needed   Additional elements that must fit.
 * @param type     Element type.
 *
 * @warning For genuinely independent arrays only. Never use it on anything
 *          inside a pool block -- there the capacity belongs to the block, and
 *          the entry points in mftl_object.h are the only correct way to change
 *          it.
 */
#define MFTL_ENSURE_CAPACITY(ptr, count, capacity, needed, type)                  \
  do                                                                              \
  {                                                                               \
    if ((*(count)) + (needed) > (*(capacity)))                                    \
    {                                                                             \
      int _new_cap = MFTLGrowCapacity(*(capacity), (*(count)) + (needed));        \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);                          \
    }                                                                             \
  } while (0)

/**
 * @brief Exact growth. For bulk loads where the final count is known up front.
 *
 * Allocates precisely `*count + needed` elements rather than rounding up to a
 * power of two, because a bulk load knows its own size and the slack a geometric
 * schedule leaves behind is never reclaimed.
 *
 * @param ptr      Lvalue holding the array pointer. Updated in place.
 * @param count    Pointer to the element counter. Read, never written.
 * @param capacity Pointer to the capacity counter. Updated in place.
 * @param needed   Additional elements that must fit.
 * @param type     Element type.
 *
 * @warning For genuinely independent arrays only. See #MFTL_ENSURE_CAPACITY.
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

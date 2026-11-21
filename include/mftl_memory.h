/**
 * @file mftl_memory.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_memory
 *
 * @brief Allocation policy, cache-line-aligned pool blocks, and array growth.
 *
 * Layer 0. Two kinds of storage, handled differently:
 *
 * - **Pool blocks** (the vertex and triangle pools of an #MFTL_Object). The AoS
 *   records and every SoA array of a pool describe the same entities, so they
 *   must always have the same capacity. A pool is therefore ONE allocation, cut
 *   into cache-line-aligned sub-arrays by MFTLLayoutAdd(): one pointer to free,
 *   one capacity, and no way to grow half of it. Only `src/object/object.c`
 *   calls MFTLLayoutAdd().
 * - **Independent arrays** (object lists, per-neighbour buffers, algorithm
 *   scratch) use MFTLMallocOrDie() / MFTLCallocOrDie() / MFTLReallocOrDie() and
 *   the growth macros #MFTL_ENSURE_CAPACITY and #MFTL_RESERVE_ADDITIONAL.
 *
 * Every growable buffer is sized by ONE doubling rule, MFTLGrowSize() (`int`
 * forms MFTLGrowCapacity() and MFTLGrowCapacityFrom()), whatever its counter
 * type and whichever allocator it then calls (the coupler's budget included).
 * A bulk append that may jump far past its capacity can size itself exactly to
 * the request instead (`max(need, 2 * cap)`), so a large buffer is never
 * over-allocated by a jump.
 *
 * Allocation failure is fatal everywhere: in an MPI job an out-of-memory rank
 * cannot recover, and a NULL buffer carried on corrupts data far from the cause.
 */

#ifndef MFTL_MEMORY_H
#define MFTL_MEMORY_H

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_constants.h"

#ifdef __cplusplus
extern "C"
{
#endif

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

/**
 * @brief `realloc` that aborts instead of returning NULL.
 *
 * For growable scratch arrays whose sizes move together, where the growth
 * macros would need one capacity counter per array.
 *
 * @param ptr   Existing block, or NULL.
 * @param bytes New size in bytes.
 * @param what  Human-readable name of the caller, printed in the fatal message.
 * @return The (possibly moved) block. Never NULL for a non-zero size.
 */
static inline void *MFTLReallocOrDie(void *ptr, size_t bytes, const char *what)
{
  void *p = realloc(ptr, bytes);
  if (!p && bytes > 0)
  {
    fprintf(stderr, "FATAL ERROR: %s reallocation failed for %lu bytes.\n", what, (unsigned long) bytes);
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
 * C11 `aligned_alloc` requires the size to be a multiple of the alignment, so the
 * request is rounded up here.
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
 * @p cursor holds the total block size:
 * \f$o_0 = 0,\; o_{i+1} = L\lceil (o_i + n_i s_i)/L \rceil\f$ with
 * \f$L = \f$ #MFTL_CACHE_LINE.
 *
 *     size_t cursor = 0;
 *     const size_t o_rec = MFTLLayoutAdd(&cursor, cap,     sizeof(MFTL_Vertex));
 *     const size_t o_pos = MFTLLayoutAdd(&cursor, 3 * cap, sizeof(double));
 *     ...
 *     unsigned char *block = MFTLAlignedAllocOrDie(cursor, "vertex pool");
 *
 * Because the capacity is a property of the block, "the AoS grew but an SoA
 * array did not" cannot be expressed.
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
 * @brief The library's ONE growth rule: at least @p needed, otherwise @p current
 *        (or @p initial when empty) doubled until it fits.
 *
 * Every growable buffer follows it, whatever its counter type: byte buffers and
 * the coupler's 64-bit counts call it directly, `int` capacities through
 * MFTLGrowCapacity(). Appending \f$n\f$ elements one at a time therefore
 * costs \f$O(\log n)\f$ reallocations.
 *
 * @param current Present capacity. May be 0.
 * @param needed  Capacity that must fit afterwards.
 * @param initial First capacity of an empty buffer; 0 is treated as 1.
 * @return @p current if it already suffices, otherwise the smallest
 *         \f$c_0 2^k \ge\f$ @p needed, with \f$c_0\f$ = @p current, or
 *         @p initial when @p current is 0. Saturates to exactly @p needed where
 *         doubling would leave `size_t` instead of wrapping to a small size.
 */
static inline size_t MFTLGrowSize(size_t current, size_t needed, size_t initial)
{
  if (needed <= current) return current;
  size_t cap = (current > 0) ? current : ((initial > 0) ? initial : 1);
  while (cap < needed)
  {
    if (cap > SIZE_MAX / 2) return needed;
    cap *= 2;
  }
  return cap;
}

/**
 * @brief MFTLGrowSize() for an `int` capacity.
 *
 * For the short lists whose first capacity is their own (ranks, blocks, events);
 * everything else uses MFTLGrowCapacity().
 *
 * @param current Present capacity. May be 0.
 * @param needed  Capacity that must fit afterwards.
 * @param initial First capacity of an empty list; 0 is treated as 1.
 * @return @p current if it already suffices, otherwise the smallest
 *         \f$c_0 2^k \ge\f$ @p needed, with \f$c_0\f$ = @p current, or
 *         @p initial when @p current is 0. Saturates to exactly @p needed near
 *         `INT_MAX` instead of overflowing to a negative size.
 */
static inline int MFTLGrowCapacityFrom(int current, int needed, int initial)
{
  if (needed <= current) return current;
  const size_t init = (initial > 0) ? (size_t) initial : 1u;
  if (needed <= 0) return (int) init;
  const size_t cap = MFTLGrowSize((current > 0) ? (size_t) current : 0u, (size_t) needed, init);
  /* int overflow would turn a huge request into a negative capacity, i.e. a
   * silently undersized allocation. Saturate instead and let the allocator
   * report the real problem. */
  return (cap > (size_t) INT_MAX) ? needed : (int) cap;
}

/**
 * @brief MFTLGrowCapacityFrom() starting at #MFTL_INIT_CAPACITY: the growth
 *        ladder of the pools and of every growth macro.
 *
 * A mesh that grows entity by entity therefore performs \f$O(\log n)\f$
 * reallocations.
 *
 * @param current Present capacity. May be 0.
 * @param needed  Capacity that must fit afterwards.
 * @return @p current if it already suffices, otherwise the smallest
 *         \f$c_0 2^k \ge\f$ @p needed, with \f$c_0\f$ = @p current, or
 *         #MFTL_INIT_CAPACITY when @p current is 0. Saturates to exactly
 *         @p needed near `INT_MAX` instead of overflowing to a negative size.
 */
static inline int MFTLGrowCapacity(int current, int needed) { return MFTLGrowCapacityFrom(current, needed, MFTL_INIT_CAPACITY); }

/**
 * @brief `count + more` for an `int` element count, or a fatal error if the sum
 *        leaves the `int` range.
 *
 * The growth macros call it before sizing anything, so a request past
 * `INT_MAX` elements stops with a message instead of wrapping to a negative
 * count (undefined behaviour) and a silently undersized buffer.
 *
 * @param count Elements already present.
 * @param more  Elements to add.
 * @param what  Human-readable name of the caller, printed in the fatal message.
 * @return `count + more`.
 */
static inline int MFTLCheckedCount(int count, int more, const char *what)
{
  const long long sum = (long long) count + (long long) more;
  if (sum > INT_MAX || sum < 0)
  {
    fprintf(stderr, "FATAL ERROR: %s: %d + %d elements leave the int index range.\n", what, count, more);
    exit(EXIT_FAILURE);
  }
  return (int) sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Growth macros for independent arrays. Never for pool storage: a pool's
 * capacity belongs to its block and changes only through mftl_object.h.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Implementation shared by #MFTL_ENSURE_CAPACITY and #MFTL_RESERVE_ADDITIONAL.
 *
 * Reallocates (MFTLReallocOrDie()) only when @p target_cap exceeds the present
 * capacity, and writes the new capacity back through @p capacity.
 *
 * @param ptr        Lvalue holding the array pointer. Updated in place.
 * @param capacity   Pointer to the capacity counter. Updated in place.
 * @param target_cap Capacity wanted, in elements.
 * @param type       Element type, used for `sizeof` and for the cast.
 *
 * @internal Not part of the public API; use one of the two wrappers.
 */
#define _MFTL_REALLOC_CORE(ptr, capacity, target_cap, type)                                            \
  do                                                                                                   \
  {                                                                                                    \
    const int _req_cap = (target_cap);                                                                 \
    if (_req_cap > (*(capacity)))                                                                      \
    {                                                                                                  \
      (ptr) = (type *) MFTLReallocOrDie((ptr), (size_t) _req_cap * sizeof(type), "MFTL array growth"); \
      (*(capacity)) = _req_cap;                                                                        \
    }                                                                                                  \
  } while (0)

/**
 * @brief Geometric growth (&times;2). For incremental appends inside a loop.
 *
 * Grows only when `*count + needed` exceeds `*capacity`, following
 * MFTLGrowCapacity()'s doubling schedule, so \f$n\f$ appends cost
 * \f$O(\log n)\f$ reallocations. The test is `needed > *capacity - *count`,
 * which cannot overflow (`*count <= *capacity`); the sum itself is formed only
 * on the growth path, by MFTLCheckedCount().
 *
 * @param ptr      Lvalue holding the array pointer. Updated in place.
 * @param count    Pointer to the element counter. Read, never written.
 * @param capacity Pointer to the capacity counter. Updated in place.
 * @param needed   Additional elements that must fit.
 * @param type     Element type.
 *
 * @warning Independent arrays only; never pool storage (see mftl_object.h).
 */
#define MFTL_ENSURE_CAPACITY(ptr, count, capacity, needed, type)                                                        \
  do                                                                                                                    \
  {                                                                                                                     \
    if ((needed) > (*(capacity)) - (*(count)))                                                                          \
    {                                                                                                                   \
      const int _new_cap = MFTLGrowCapacity(*(capacity), MFTLCheckedCount(*(count), (needed), "MFTL_ENSURE_CAPACITY")); \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);                                                                \
    }                                                                                                                   \
  } while (0)

/**
 * @brief Exact growth. For bulk loads where the final count is known up front.
 *
 * Allocates precisely `*count + needed` elements: a bulk load knows its size, and
 * geometric slack would never be reclaimed. Overflow-safe as #MFTL_ENSURE_CAPACITY.
 *
 * @param ptr      Lvalue holding the array pointer. Updated in place.
 * @param count    Pointer to the element counter. Read, never written.
 * @param capacity Pointer to the capacity counter. Updated in place.
 * @param needed   Additional elements that must fit.
 * @param type     Element type.
 *
 * @warning Independent arrays only. See #MFTL_ENSURE_CAPACITY.
 */
#define MFTL_RESERVE_ADDITIONAL(ptr, count, capacity, needed, type)                         \
  do                                                                                        \
  {                                                                                         \
    if ((needed) > (*(capacity)) - (*(count)))                                              \
    {                                                                                       \
      const int _new_cap = MFTLCheckedCount(*(count), (needed), "MFTL_RESERVE_ADDITIONAL"); \
      _MFTL_REALLOC_CORE(ptr, capacity, _new_cap, type);                                    \
    }                                                                                       \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif  // MFTL_MEMORY_H

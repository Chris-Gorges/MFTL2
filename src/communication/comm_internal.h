/**
 * @file comm_internal.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Private helpers shared by the communication module's translation units.
 *
 * Not a public header: nothing outside src/communication/ may include it, and
 * nothing here is exported. It holds the two data structures that both the
 * redistribution and the ghost synchronisation need.
 */

#ifndef MFTL_COMM_INTERNAL_H
#define MFTL_COMM_INTERNAL_H

#include <stdint.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Growable int list
 *
 * One per destination rank, so there are nproc of them. Allocation is LAZY: a
 * redistribution touches its 26 neighbours and nobody else, and eagerly giving
 * every rank a 4 KB buffer costs 32 MB of untouched pages at nproc = 4096. The
 * first append jumps straight to the density-derived hint, so the lazy path does
 * not pay for a growth staircase either.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A growable `int` list that allocates NOTHING until first used.
 *
 * One of these exists per destination rank, so at `nproc = 4096` a 4 KB eager
 * buffer would cost 32 MB of untouched pages on every rank. The first append
 * jumps straight to the density-derived @c hint, so the lazy path does not pay
 * for a growth staircase either.
 */
typedef struct
{
  int *data;  ///< [capacity] the elements, or MFTL_NULL while untouched.
  int size;  ///< Elements currently held.
  int capacity;  ///< Elements allocated.
  int hint;  ///< Capacity to jump to on the first append.
} mftl_int_buffer;

/**
 * @brief Initialises an empty buffer. Allocates nothing.
 * @param b    The buffer.
 * @param hint Capacity the first append should jump to.
 */
static inline void ib_init(mftl_int_buffer *b, int hint)
{
  b->data = MFTL_NULL;
  b->size = 0;
  b->capacity = 0;
  b->hint = hint;
}

/**
 * @brief Makes room for @p needed more elements.
 * @param b      The buffer.
 * @param needed Additional elements that must fit.
 */
static inline void ib_grow(mftl_int_buffer *b, int needed)
{
  if (b->capacity == 0)
  {
    const int cap = (b->hint > needed) ? b->hint : needed;
    b->data = (int *) MFTLMallocOrDie((size_t) cap * sizeof(int), "MFTL communication id list");
    b->capacity = cap;
    return;
  }
  MFTL_ENSURE_CAPACITY(b->data, &b->size, &b->capacity, needed, int);
}

/**
 * @brief Appends one value.
 * @param b The buffer.
 * @param v The value.
 */
static inline void ib_add(mftl_int_buffer *b, int v)
{
  if (b->size + 1 > b->capacity) ib_grow(b, 1);
  b->data[b->size++] = v;
}

/**
 * @brief Appends two values with a single capacity check.
 * @param b The buffer.
 * @param a First value.
 * @param c Second value.
 */
static inline void ib_add2(mftl_int_buffer *b, int a, int c)
{
  if (b->size + 2 > b->capacity) ib_grow(b, 2);
  b->data[b->size + 0] = a;
  b->data[b->size + 1] = c;
  b->size += 2;
}

/**
 * @brief Appends three values with a single capacity check.
 * @param b The buffer.
 * @param a First value.
 * @param c Second value.
 * @param d Third value.
 */
static inline void ib_add3(mftl_int_buffer *b, int a, int c, int d)
{
  if (b->size + 3 > b->capacity) ib_grow(b, 3);
  b->data[b->size + 0] = a;
  b->data[b->size + 1] = c;
  b->data[b->size + 2] = d;
  b->size += 3;
}

/**
 * @brief Empties the buffer, keeping the allocation.
 * @param b The buffer.
 */
static inline void ib_reset(mftl_int_buffer *b) { b->size = 0; }

/**
 * @brief Releases the allocation and returns the buffer to its initial state.
 * @param b The buffer.
 */
static inline void ib_free(mftl_int_buffer *b)
{
  free(b->data);
  b->data = MFTL_NULL;
  b->size = 0;
  b->capacity = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Epoch-stamped global-id table -- what replaced khash
 *
 * The question every rebuild has to answer is "do I already hold global id g, and
 * where?". A hash map per entity class answers it, and pays a hash insert per
 * received entity plus a rehash per redistribution.
 *
 * Global ids are DENSE per entity class (inv_global_ids_are_dense), so the answer
 * is an array index. Two things make a plain array practical here:
 *
 *   - the id range is the OBJECT's, not the ensemble's, so a thousand droplets
 *     get a thousand small ranges rather than one huge one;
 *   - the table is never cleared. `stamp[g] == epoch` is what makes an entry
 *     valid, and bumping the epoch retires every entry at once. Clearing a table
 *     sized by the GLOBAL id range once per object per timestep would cost more
 *     than the lookups it serves.
 *
 * The allocation is grown monotonically and shared across objects and calls, and
 * the freshly grown tail is zeroed so that no stale stamp can ever match. Epochs
 * start at 1, so a zeroed slot is always "absent".
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Epoch-stamped global-id lookup table -- what replaced the hash map.
 *
 * Answers "do I already hold global id \f$g\f$, and where?" in \f$O(1)\f$ with
 * no hashing, because global ids are DENSE per entity class and therefore ARE
 * array indices.
 *
 * The table is **never cleared**: `stamp[g] == epoch` is what makes an entry
 * valid, and bumping the epoch retires every entry at once. Clearing a table
 * sized by the id range once per object per timestep would cost more than the
 * lookups it serves.
 */
typedef struct
{
  unsigned *stamp;  ///< [capacity] epoch the entry belongs to; 0 = never written.
  int *value;  ///< [capacity] payload, meaningful only while stamp == epoch.
  int capacity;  ///< Ids the table can address.
  unsigned epoch;  ///< Unsigned: wraparound is defined, and gt_begin() handles it.
} mftl_gid_table;

/**
 * @brief Initialises an empty table. Allocates nothing.
 * @param t The table.
 */
static inline void gt_init(mftl_gid_table *t)
{
  t->stamp = MFTL_NULL;
  t->value = MFTL_NULL;
  t->capacity = 0;
  t->epoch = 0;
}

/**
 * @brief Releases the table's allocations.
 * @param t The table.
 */
static inline void gt_free(mftl_gid_table *t)
{
  free(t->stamp);
  free(t->value);
  gt_init(t);
}

/**
 * @brief Ensures the table can address ids in [0, range) and retires all entries.
 *
 * Growth is geometric so a sequence of objects with rising id ranges does not
 * reallocate on every one of them.
 *
 * @param t     The table.
 * @param range Highest addressable id plus one.
 */
static inline void gt_begin(mftl_gid_table *t, int range)
{
  if (range > t->capacity)
  {
    const int cap = MFTLGrowCapacity(t->capacity, range);
    unsigned *s = (unsigned *) MFTLMallocOrDie((size_t) cap * sizeof(unsigned), "MFTL gid table stamps");
    int *v = (int *) MFTLMallocOrDie((size_t) cap * sizeof(int), "MFTL gid table values");

    /* Only the stamps need defined contents; a value is never read unless its
     * stamp matches. Copying the old stamps preserves entries from an earlier
     * epoch, which costs nothing and keeps the "never cleared" property true. */
    if (t->capacity > 0) memcpy(s, t->stamp, (size_t) t->capacity * sizeof(unsigned));
    memset(s + t->capacity, 0, (size_t) (cap - t->capacity) * sizeof(unsigned));
    if (t->capacity > 0) memcpy(v, t->value, (size_t) t->capacity * sizeof(int));

    free(t->stamp);
    free(t->value);
    t->stamp = s;
    t->value = v;
    t->capacity = cap;
  }

  /* The one case where "never cleared" cannot hold: after 2^32 epochs the counter
   * comes back to a value some ancient stamp still carries, and that entry would
   * read as present. Wrapping to 0 is defined for unsigned, so it is detectable,
   * and the table is wiped exactly once every four billion objects. Epochs start
   * at 1 so that a zeroed slot always means "absent". */
  t->epoch++;
  if (t->epoch == 0u)
  {
    if (t->capacity > 0) memset(t->stamp, 0, (size_t) t->capacity * sizeof(unsigned));
    t->epoch = 1u;
  }
}

/**
 * @brief True if @p gid carries a value in the current epoch.
 * @param t   The table.
 * @param gid Global id, asserted to be inside the table range.
 * @return `true` if gt_put() stored a value for @p gid since the last gt_begin().
 */
static inline bool gt_has(const mftl_gid_table *t, int gid)
{
  MFTL_ASSERT(gid >= 0 && gid < t->capacity && "gt_has: global id outside the table range");
  return t->stamp[gid] == t->epoch;
}

/**
 * @brief The value stored for @p gid.
 * @param t   The table.
 * @param gid Global id.
 * @return The stored value. **Undefined unless gt_has() is true** -- asserted in
 *         Debug, unchecked in Release.
 */
static inline int gt_get(const mftl_gid_table *t, int gid)
{
  MFTL_ASSERT(gt_has(t, gid) && "gt_get: global id not present in this epoch");
  return t->value[gid];
}

/**
 * @brief Stores @p value for @p gid in the current epoch.
 * @param t     The table.
 * @param gid   Global id, asserted to be inside the table range.
 * @param value The payload, usually a pool index.
 */
static inline void gt_put(mftl_gid_table *t, int gid, int value)
{
  MFTL_ASSERT(gid >= 0 && gid < t->capacity && "gt_put: global id outside the table range");
  t->stamp[gid] = t->epoch;
  t->value[gid] = value;
}

#endif  // MFTL_COMM_INTERNAL_H

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

#include <mpi.h>
#include <stdint.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_memory.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Instrumentation
 *
 * One file-scope accumulator, defined in comm_stats.c, written from every
 * translation unit of this module. A singleton rather than a `stats` parameter
 * because the alternative is a signature change on four public functions and
 * some sixty call sites in tests and examples, for a diagnostic.
 *
 * The macros exist to make the cost of measuring visible in the same place it is
 * paid: every one of them bumps `n_clock_reads`, so the perturbation can be
 * subtracted afterwards rather than assumed negligible.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The accumulator. Rank-local, zeroed by MFTLRedistributeStatsReset(). */
extern MFTL_RedistributeStats MFTLCommStats;

/* ── What is always on, and what has to be asked for ─────────────────────────
 *
 * The COUNTERS are unconditional. They are a handful of integer adds per call or
 * per body, they answer most of the questions anyone asks of this module -- how
 * much moved, how many bytes, how many bodies were empty, how many allocations --
 * and they cost nothing measurable.
 *
 * The TIMERS are not, and MFTL_COMM_PROFILE is off by default. Two of them are
 * genuinely expensive:
 *
 *   - the per-body timeline reads the clock at ten phase boundaries per BODY, so
 *     a thousand-droplet ensemble reads it some sixteen thousand times per
 *     timestep. MPI_Wtime measures 25-37 ns on the development machine, which is
 *     1.5-2 % of a ten-millisecond phase;
 *   - the ownership-change count adds a compare to compute_owners' two hot loops,
 *     and those run once per entity per timestep. CLAUDE.md records a previous
 *     modification to exactly those loops measuring 1.70x.
 *
 * Neither is acceptable in a production timestep, and both are indispensable when
 * the question is where the time goes -- so they are a build option rather than a
 * judgement call baked into the source. Configure with
 * `-DMFTL_COMM_PROFILE=ON`; with it off the timer fields of
 * #MFTL_RedistributeStats stay at zero and the counters are still filled in. */

#ifdef MFTL_COMM_PROFILE

/** @brief Reads the clock and counts the read. */
static inline double cs_now(void)
{
  MFTLCommStats.n_clock_reads++;
  return MPI_Wtime();
}

/** @brief Opens a timed region: declares a `double` named @p v. */
#define CS_BEGIN(v) const double v = cs_now()

/** @brief Closes a timed region opened with #CS_BEGIN, into @p field. */
#define CS_END(v, field) (MFTLCommStats.field += cs_now() - (v))

/** @brief Opens a timeline: declares a `double` named @p tm, for #CS_LAP. */
#define CS_TIMELINE(tm) double tm = cs_now()

/**
 * @brief Charges the time since @p tm to @p field and restarts the clock at @p tm.
 *
 * The TIMELINE form, for a run of adjacent regions. One clock read serves as the
 * end of one bucket and the start of the next, which halves the reads against a
 * CS_BEGIN/CS_END pair per region.
 */
#define CS_LAP(tm, field)                                                                                                                                      \
  do                                                                                                                                                          \
  {                                                                                                                                                            \
    const double cs_lap_now_ = cs_now();                                                                                                                       \
    MFTLCommStats.field += cs_lap_now_ - (tm);                                                                                                               \
    (tm) = cs_lap_now_;                                                                                                                                        \
  } while (0)

/** @brief True when the ownership-change counters are wanted. */
#define CS_COUNT_OWNER_CHANGES 1

#else

#define CS_BEGIN(v) ((void) 0)
#define CS_END(v, field) ((void) 0)
#define CS_TIMELINE(tm) ((void) 0)
#define CS_LAP(tm, field) ((void) 0)
#define CS_COUNT_OWNER_CHANGES 0

#endif  // MFTL_COMM_PROFILE

/**
 * @brief Records an allocation of @p bytes into the redistribution counters.
 *
 * The byte count is what was REQUESTED, summed over the run -- not a peak and
 * not what the allocator actually mapped. It is a churn measure: the question it
 * answers is how much a persistent workspace would stop re-requesting.
 */
static inline void cs_alloc(size_t bytes)
{
  MFTLCommStats.n_allocs++;
  MFTLCommStats.n_bytes_allocated += (long) bytes;
}

/** @brief Records a free on the redistribution path. */
static inline void cs_free(void) { MFTLCommStats.n_frees++; }

/** @brief Records an allocation of @p bytes into the ghost-sync counters. */
static inline void cs_galloc(size_t bytes)
{
  MFTLCommStats.g_n_allocs++;
  MFTLCommStats.g_n_bytes_allocated += (long) bytes;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Growable int list
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
    cs_alloc((size_t) cap * sizeof(int));
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
 * 3. Global id -> local slot table -- what replaced khash, and why it is a hash
 *
 * "Have I already got global id g, and in which slot" is asked once per arriving
 * entity. CLAUDE.md's standing rule is that there are no hash maps in this
 * library, because global ids are DENSE per entity class and a dense id is
 * already an array index.
 *
 * That rule holds mesh-wide and fails on a rank. Ids are dense across the whole
 * BODY; a rank holds a partition of it, so the ids it must look up are a sparse
 * subset scattered over the body's whole range. A direct-address array is
 * therefore sized by the GLOBAL entity count while holding only the LOCAL one --
 * and it is rebuilt every call, because the scratch does not survive between
 * redistributions.
 *
 * Measured, before this was changed: one 327 680-triangle body at np = 2/4/8 asked
 * for 15.2 / 31.0 / 62.0 MB of id table PER RANK PER TIMESTEP, allocated and
 * zeroed each time. Linear in the process count at a FIXED problem size, which in
 * a weak-scaling sweep is quadratic. The table held 163 842 entries at np = 1 and
 * roughly 20 000 at np = 8, so at np = 8 it was 97 % empty.
 *
 * Hence: an open-addressed table sized from the LOCAL entry count, and the
 * documented exception to the no-hash-map rule. Everything that made the direct
 * array attractive is kept --
 *
 *   - EPOCH STAMPS instead of clearing. A slot belongs to the current epoch only
 *     if its stamp matches, so retiring every entry is `epoch++` rather than a
 *     memset. That is what makes a per-object table free to reuse across the
 *     bodies of one call.
 *   - NO DELETION, ever. Without tombstones a probe stops at the first slot whose
 *     stamp is stale, which is what keeps the probe loop two compares.
 *   - LOAD FACTOR AT MOST 1/2, capacity a power of two, so the probe is a mask
 *     rather than a modulo and clustering stays bounded.
 *
 * Growth DISCARDS rather than rehashes, which is free here: gt_begin() retires
 * every entry anyway, so there is nothing worth carrying across a resize.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Global-id to local-slot map. Two indexing modes, chosen per use.
 *
 * Sized from the number of ENTRIES it will hold and the ID RANGE they are drawn
 * from, and it picks its indexing from the ratio. See the section comment.
 */
typedef struct
{
  /**
   * `[capacity]` epoch in the high 32 bits, global id in the low 32.
   *
   * ONE array rather than a separate stamp and key, and the reason is the probe:
   * it must answer "is this slot live, and is it mine" on every step, and two
   * arrays make that two loads that miss the cache independently. Packed into one
   * 64-bit word it is a single load and a single compare against a value the
   * caller already has in a register.
   */
  uint64_t *tag;
  int *value;  ///< [capacity] the local slot. Touched only on a hit.
  int capacity;  ///< Slots allocated. In sparse mode a power of two, so `capacity - 1` is the probe mask.
  unsigned epoch;  ///< Bumped by gt_begin(). Unsigned: wraparound is defined and handled.
  bool direct;  ///< Indexing mode: `tag[gid]` if set, hashed with linear probing if not.
} mftl_gid_table;

/** @brief The tag a live entry for @p gid carries in epoch @p epoch. */
static inline uint64_t gt_tag(unsigned epoch, int gid) { return ((uint64_t) epoch << 32) | (uint32_t) gid; }

/** @brief Zeroes a table without allocating. */
static inline void gt_init(mftl_gid_table *t)
{
  t->tag = MFTL_NULL;
  t->value = MFTL_NULL;
  t->capacity = 0;
  t->epoch = 0u;
  t->direct = MFTL_FALSE;
}

/** @brief Releases a table. */
static inline void gt_free(mftl_gid_table *t)
{
  free(t->tag);
  free(t->value);
  gt_init(t);
}

/**
 * @brief Scatters a global id over a sparse table's index space.
 *
 * An avalanche multiply. The identity index is the obvious alternative -- global
 * ids are dense, so consecutive ids would land in consecutive slots and the table
 * would be read almost sequentially -- and it measured 127 ms against 0.76 ms on
 * a 327 680-triangle body. Dense runs that ALIAS under the fold are the textbook
 * primary-clustering input: the second run probes past the whole of the first, one
 * slot further each time, and insertion becomes quadratic. A partition's ids are
 * many runs scattered over the body's range by a subdivision order unrelated to
 * space, so aliasing is the normal case rather than the unlucky one.
 *
 * @param gid The global id.
 * @return A well-mixed 32-bit value.
 */
static inline unsigned gt_hash(int gid)
{
  unsigned x = (unsigned) gid;
  x *= 0x9e3779b1u;
  return x ^ (x >> 16);
}

/**
 * @brief How dense the ids must be for direct addressing to be worth its memory.
 *
 * At `range <= 4 * n_entries` the direct table costs `range` slots against the
 * sparse table's `2 * n_entries` rounded up to a power of two -- so at most about
 * twice the memory, for a lookup that is one load instead of a multiply, a mask
 * and a probe. Beyond that ratio the memory gap grows without bound while the
 * speed gap does not, which is the whole content of the measurement.
 */
#define MFTL_GT_DIRECT_RATIO 4

/**
 * @brief Sizes the table for @p n_entries insertions drawn from `[0, range)`,
 *        picks an indexing mode, and retires everything in it.
 *
 * @param t         The table.
 * @param n_entries How many gt_put() calls will follow.
 * @param range     One past the largest global id that will be presented. Only
 *                  used to decide the mode and, in direct mode, the size.
 */
static inline void gt_begin(mftl_gid_table *t, int n_entries, int range)
{
  const bool direct = (range <= MFTL_GT_DIRECT_RATIO * n_entries + 64);

  /* A POWER OF TWO IN BOTH MODES, and the reason is thrash rather than the mask.
   * Sparse mode needs it for `capacity - 1` to be a probe mask; direct mode does
   * not, and sizing direct mode to exactly `range` looks like the tighter choice.
   * It is not, because ONE table serves every body of a call: a body this rank
   * holds most of chooses direct, the next body it holds a sliver of chooses
   * sparse, and a mode change that also changes the size frees and reallocates on
   * every one of them. Measured on a thousand-droplet ensemble: 19x the id-table
   * bytes per step and three times the allocation count, for tables of a few
   * kilobytes.
   *
   * With one size rule the allocation survives a mode change untouched. Nothing
   * else has to: the epoch bump below retires every entry, so slots written under
   * the other interpretation read as absent, which is exactly what a retired
   * entry should do. Direct mode pays up to 2x the memory for it. */
  const int need = direct ? range : 2 * n_entries + 2;
  int want = 16;
  while (want < need) want <<= 1;

  if (want > t->capacity)
  {
    /* Discarded rather than rehashed: every entry is retired by this call
     * regardless, so carrying them across would be work for nothing. */
    free(t->tag);
    free(t->value);
    t->tag = (uint64_t *) MFTLCallocOrDie((size_t) want, sizeof(uint64_t), "MFTL gid table tags");
    t->value = (int *) MFTLMallocOrDie((size_t) want * sizeof(int), "MFTL gid table values");
    t->capacity = want;
    t->epoch = 0u;
    cs_alloc((size_t) want * (sizeof(uint64_t) + sizeof(int)));
    MFTLCommStats.n_gid_table_bytes += (long) want * (long) (sizeof(uint64_t) + sizeof(int));
    if (direct) MFTLCommStats.n_gid_table_direct++;
    else MFTLCommStats.n_gid_table_sparse++;
  }
  else if (direct)
  {
    MFTLCommStats.n_gid_table_direct++;
  }
  else
  {
    MFTLCommStats.n_gid_table_sparse++;
  }

  t->direct = direct;

  /* The one case where "never cleared" cannot hold: after 2^32 epochs the counter
   * comes back to a value some ancient tag still carries, and that entry would
   * read as present. Wrapping to 0 is defined for unsigned, so it is detectable,
   * and the table is wiped exactly once every four billion uses. Epochs start at
   * 1 so that a zeroed slot always means "absent". */
  t->epoch++;
  if (t->epoch == 0u)
  {
    memset(t->tag, 0, (size_t) t->capacity * sizeof(uint64_t));
    t->epoch = 1u;
  }
}

/**
 * @brief Finds the slot @p gid occupies, or the free slot it would occupy.
 *
 * In sparse mode the probe stops at the first slot that is not of the current
 * epoch, which is sound only because entries are never deleted -- a tombstone
 * would need a third state and a second compare in this loop.
 *
 * The mode test is a branch on a value that is constant for the whole of a
 * body's rebuild, so it predicts perfectly and costs nothing measurable.
 *
 * @param      t     The table.
 * @param      gid   The global id.
 * @param[out] found Set to `true` if the returned slot holds @p gid.
 * @return The slot index.
 */
static inline int gt_probe(const mftl_gid_table *t, int gid, bool *found)
{
  const uint64_t want = gt_tag(t->epoch, gid);

  if (t->direct)
  {
    /* An id above the declared range is ABSENT, not an error. Callers legitimately
     * probe ids they have no reason to expect -- the spatial halo asks whether an
     * arriving triangle's corner is among the vertices that arrived with it, and a
     * corner whose vertex was not sent has an id from anywhere in the body. The
     * test costs one compare and it is what makes the two indexing modes actually
     * interchangeable: a sparse table answers such a probe without it, so leaving
     * it out would make the modes disagree on exactly the inputs nothing tests. */
    if (gid >= t->capacity)
    {
      *found = false;
      return 0;
    }
    *found = (t->tag[gid] == want);
    return gid;
  }

  const int mask = t->capacity - 1;
  const uint64_t live = (uint64_t) t->epoch << 32;
  int i = (int) (gt_hash(gid) & (unsigned) mask);

  for (;;)
  {
    const uint64_t tag = t->tag[i];
    if (tag == want)
    {
      *found = true;
      return i;
    }
    if ((tag & 0xffffffff00000000ULL) != live)
    {
      *found = false;
      return i;  // stale or never written: this is where the id would go
    }
    i = (i + 1) & mask;
  }
}

/**
 * @brief Has @p gid been stored since the last gt_begin()?
 * @param t   The table.
 * @param gid The global id. Must be non-negative; an id above the declared range
 *            is simply absent.
 * @return `true` if gt_put() stored a value for @p gid in this epoch.
 */
static inline bool gt_has(const mftl_gid_table *t, int gid)
{
  MFTL_ASSERT(gid >= 0 && "gt_has: negative global id");
  bool found = false;
  (void) gt_probe(t, gid, &found);
  return found;
}

/**
 * @brief Reads the value stored for @p gid.
 * @param t   The table.
 * @param gid The global id.
 * @return The stored value. **Undefined unless gt_has() is true** -- asserted in
 *         Debug, unchecked in Release, because every caller has just tested it.
 */
static inline int gt_get(const mftl_gid_table *t, int gid)
{
  bool found = false;
  const int i = gt_probe(t, gid, &found);
  MFTL_ASSERT(found && "gt_get: global id not present in this epoch");
  (void) found;
  return t->value[i];
}

/**
 * @brief Stores @p value under @p gid for the current epoch.
 * @param t     The table.
 * @param gid   The global id. Must be non-negative and below the declared range.
 * @param value The local slot.
 */
static inline void gt_put(mftl_gid_table *t, int gid, int value)
{
  MFTL_ASSERT(gid >= 0 && "gt_put: negative global id");
  bool found = false;
  const int i = gt_probe(t, gid, &found);
  t->tag[i] = gt_tag(t->epoch, gid);
  t->value[i] = value;
}

/**
 * @brief The ghost sync's persistent owner-side id table, on the ensemble.
 *
 * INTERNAL despite the MFTL prefix -- it takes a type that only exists inside
 * this module, so it can never appear in a public header. The prefix is there
 * because it has external linkage and the library exports nothing else.
 *
 * Defined in redistribute.c, which owns the workspace type. Both paths need a
 * table sized by a body's id range and neither wants to build one per timestep,
 * so they share the one handle rather than each keeping their own.
 *
 * @param ens The ensemble. Its workspace is created on first use.
 * @return The table. Owned by the ensemble; never freed by the caller.
 */
mftl_gid_table *MFTLCommWorkspaceGhostTable(MFTL_Ensemble *ens);

#endif  // MFTL_COMM_INTERNAL_H

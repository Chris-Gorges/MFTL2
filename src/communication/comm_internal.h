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
 * 1. Instrumentation: one accumulator (comm_stats.c). Counters are always on
 * (a few integer adds per call or body). Phase TIMERS are compiled in only with
 * -DMFTL_COMM_PROFILE=ON: the per-body timeline reads MPI_Wtime at ten
 * boundaries per body (25-37 ns each on the development machine, 1.5-2 % of a
 * 10 ms phase at a thousand bodies). Every clock read is counted in
 * `n_clock_reads`, so the perturbation can be estimated.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The accumulator. Rank-local, zeroed by MFTLRedistributeStatsReset(). */
extern MFTL_RedistributeStats MFTLCommStats;

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

/** @brief 1 in a profiling build. Currently unused: owner changes are always counted. */
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
 * 2. Growable int list, one per destination rank. Allocation is LAZY (at
 * nproc = 4096 an eager 4 KB buffer each would be 32 MB of untouched pages), and
 * the first append jumps straight to a density-derived hint.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A growable `int` list that allocates nothing until first used.
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
 * 3. Global id -> local slot table
 *
 * "Do I already hold global id g, and in which slot?" is asked once per arriving
 * entity. Ids are dense per BODY, but a rank holds a scattered subset of a
 * body's range, so a direct-address array is sized by the global count while
 * holding the local one (measured: 15-62 MB per rank per step at np = 2-8 for
 * one 327 680-triangle body, 97 % empty at np = 8). The table therefore has two
 * modes chosen per use from the ratio of id range to entries: DIRECT
 * (`tag[gid]`) when the ids are dense enough, SPARSE (open addressing, linear
 * probing, load factor <= 1/2) otherwise. This is the library's one hash table.
 *
 * - EPOCH stamps instead of clearing: a slot is live only if its tag carries the
 *   current epoch, so retiring every entry is `epoch++`.
 * - No deletion ever, so a probe stops at the first stale slot.
 * - Capacity is a power of two in both modes, so a mode change never
 *   reallocates; growth discards instead of rehashing (every entry is retired
 *   anyway).
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Global-id to local-slot map with a direct and a sparse (hashed) mode.
 */
typedef struct
{
  /**
   * `[capacity]` epoch in the high 32 bits, global id in the low 32: one load
   * and one compare decide "live, and mine".
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
 * @brief Scatters a global id over a sparse table's index space (multiplicative
 *        avalanche hash).
 *
 * The identity index clusters badly: a partition's ids are many dense runs that
 * alias under the mask, and linear probing then degrades to quadratic insertion
 * (measured 127 ms against 0.76 ms on a 327 680-triangle body).
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
 * @brief Direct addressing is used when `range <= 4 * n_entries + 64`: then it
 *        costs at most about twice the memory of the sparse table for a
 *        one-load lookup.
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

  /* A power of two in both modes: one table serves every body of a call, and a
   * size rule shared by both modes lets it survive mode changes without
   * reallocating (sizing direct mode exactly measured 19x the table bytes on a
   * thousand-droplet ensemble). The epoch bump retires entries of either mode. */
  const int need = direct ? range : 2 * n_entries + 2;
  int want = 16;
  while (want < need) want <<= 1;

  if (want > t->capacity)
  {
    /* Discarded, not rehashed: every entry is retired anyway. */
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

  /* Epochs start at 1 (a zero tag is always absent). On unsigned wrap-around,
   * after 2^32 uses, the table is wiped once. */
  t->epoch++;
  if (t->epoch == 0u)
  {
    memset(t->tag, 0, (size_t) t->capacity * sizeof(uint64_t));
    t->epoch = 1u;
  }
}

/**
 * @brief gt_begin() in DIRECT mode whatever the entry count: for a caller that
 *        has decided `range` slots are affordable (it declares range / 4
 *        entries, which is what the mode rule asks for).
 * @param t     The table.
 * @param range One past the largest global id that will be presented.
 */
static inline void gt_begin_direct(mftl_gid_table *t, int range) { gt_begin(t, (range + 3) / 4, range); }

/**
 * @brief Finds the slot @p gid occupies, or the free slot it would occupy.
 *
 * Sparse mode stops at the first slot not of the current epoch (sound because
 * nothing is ever deleted). The mode branch is constant per use and predicts
 * perfectly.
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
    /* An id beyond the table is ABSENT, as in sparse mode (the spatial halo
     * legitimately probes ids of vertices that were not sent). */
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
 * @brief The value stored for @p gid, or #MFTL_ERROR: gt_has() and gt_get() in
 *        ONE probe, for a caller that answers absent ids instead of skipping them.
 * @param t   The table.
 * @param gid The global id, non-negative.
 * @return The stored value, or #MFTL_ERROR if @p gid was not stored this epoch.
 */
static inline int gt_find(const mftl_gid_table *t, int gid)
{
  bool found = false;
  const int i = gt_probe(t, gid, &found);
  return found ? t->value[i] : MFTL_ERROR;
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
 * @brief The ghost sync's persistent owner-side id table, held by the ensemble's
 *        communication workspace (created on first use). Internal, although it
 *        carries the MFTL prefix required of every external symbol.
 *
 * @param ens The ensemble.
 * @return The table; owned by the ensemble, never freed by the caller.
 */
mftl_gid_table *MFTLCommWorkspaceGhostTable(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * The routed exchange's transport (exchange.c): the route vote, the count
 * exchange and the data posts that MFTLExchangeStart(), MFTLExchangeRecords(),
 * MFTLExchangeRecordsTo() and MFTLGhostPull() are front-ends of. A caller
 * fills in its destinations and per-destination counts, and gets back PEERS --
 * the graph neighbours, or on the global route every rank it sends to or
 * receives from -- with what it sends and receives from each; everything after
 * that is indexed by peer, the same way on both routes.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Tag of the global route's count discovery on `comm_world`. Nothing
 *         else in the library sends point to point there (the coupler has its own
 *         communicator). */
#define MFTL_TAG_ROUTE_COUNTS 20800

/** @brief First tag of the global route's data; a caller adds its payload index (< 16). */
#define MFTL_TAG_ROUTE_DATA 20816

/** @brief Peers and counts of one routed exchange. */
typedef struct
{
  bool global;  ///< The global route; otherwise `comm_graph`.
  int k;  ///< Count ints per peer.
  int n;  ///< Peers.
  const int *peer;  ///< [n] ranks: `domain->neighbors` (graph), ascending (global).
  const int *scnt;  ///< [n * k] counts sent to each peer, peer-major.
  int *rcnt;  ///< [n * k] counts received from each peer, peer-major.

  /* private */
  int *own;  ///< The allocation behind `rcnt` (and on the global route `peer`, `scnt`).
  MPI_Request req;  ///< The graph route's count exchange in flight.
  MPI_Request *sreq;  ///< The global route's discovery sends.
  int n_sreq;  ///< Entries of `sreq`.
  int n_dst;  ///< Global: destinations handed to begin.
  const int *dst;  ///< Global: their ranks, ascending.
  const int *dcnt;  ///< Global: their counts, `[n_dst * k]`.
} mftl_route;

/**
 * @brief THE route vote of the library: `false` (graph) under #MFTL_ROUTE_GRAPH
 *        or on one rank; under #MFTL_ROUTE_AUTO, whether ANY rank passes
 *        @p far, by one `MPI_Allreduce(LOR)`.
 *
 * Every global-route discovery is preceded by its own vote, which is what keeps two
 * consecutive NBX rounds on one tag apart (MFTLNbxComplete()).
 *
 * @param domain The domain.
 * @param route  Identical on every rank.
 * @param far    This rank addresses a rank that is neither itself nor a graph neighbour.
 * @return The agreed route: `true` for the global one.
 */
bool MFTLExchangeAgreeRoute(const MFTL_Domain *domain, MFTL_Route route, bool far);

/**
 * @brief Starts the count exchange of a routed exchange.
 *
 * Graph route: @p dst must be `domain->neighbors` (`n_dst == num_neighbors`),
 * @p dcnt by slot; one `MPI_(I)neighbor_alltoall` of @p k ints. Global route:
 * @p dst ascending and distinct; a synchronous send of @p k ints to each one
 * with a non-zero count. @p dst and @p dcnt must stay valid until
 * MFTLRouteCountsEnd().
 *
 * @param domain  The domain.
 * @param rt      Out: the route; `global` must be set by the caller.
 * @param k       Count ints per destination.
 * @param n_dst   Destinations.
 * @param dst     `[n_dst]` their ranks.
 * @param dcnt    `[n_dst * k]` counts, destination-major.
 * @param overlap Graph route: post the count exchange non-blocking (for work
 *                between begin and end); otherwise it completes here.
 */
void MFTLRouteCountsBegin(const MFTL_Domain *domain, mftl_route *rt, int k, int n_dst, const int *dst, const int *dcnt, bool overlap);

/**
 * @brief Completes the count exchange: on the global route the NBX discovery,
 *        then the peers (the ascending union of destinations and sources).
 * @param domain The domain.
 * @param rt     The route of MFTLRouteCountsBegin(); `n`, `peer`, `scnt`, `rcnt` set on return.
 */
void MFTLRouteCountsEnd(const MFTL_Domain *domain, mftl_route *rt);

/**
 * @brief Posts one payload of a routed exchange: `sc[i]` elements from
 *        `sbuf + sd[i]` to peer i, `rc[i]` into `rbuf + rd[i]` from peer i
 *        (counts and displacements in elements of @p type).
 *
 * Graph route: `MPI_Neighbor_alltoallv`, or `MPI_Ineighbor_alltoallv` when
 * @p reqs is given. Global route: one `MPI_Irecv` / `MPI_Isend` per non-empty
 * direction of each peer, on `comm_world` with @p tag; completed here unless
 * @p reqs is given.
 *
 * @param domain The domain.
 * @param global The agreed route.
 * @param n      Peers.
 * @param peer   `[n]` their ranks (graph: `domain->neighbors`).
 * @param type   The element type.
 * @param sbuf   Send buffer.
 * @param sc     `[n]` elements to each peer.
 * @param sd     `[n]` their displacements.
 * @param rbuf   Receive buffer.
 * @param rc     `[n]` elements from each peer.
 * @param rd     `[n]` their displacements.
 * @param tag    Global route's tag, #MFTL_TAG_ROUTE_DATA plus the payload index.
 * @param[out] reqs   #MFTL_NULL to complete here; else a `malloc`'d request array the caller waits on and frees.
 * @param[out] n_reqs Entries of `*reqs`.
 */
void MFTLRoutePost(const MFTL_Domain *domain, bool global, int n, const int *peer, MPI_Datatype type, const void *sbuf, const int *sc, const int *sd,
                   void *rbuf, const int *rc, const int *rd, int tag, MPI_Request **reqs, int *n_reqs);

/**
 * @brief Releases a route. Safe to call twice.
 * @param rt The route.
 */
void MFTLRouteFree(mftl_route *rt);

#endif  // MFTL_COMM_INTERNAL_H

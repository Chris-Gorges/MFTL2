/**
 * @file ghost_pull.c
 * @author Christian Gorges
 * @date 2026-11-20
 * @ingroup mftl_communication
 *
 * @brief The library's one ghost request/reply engine, MFTLGhostPull().
 *
 * Only the HOLDER of a ghost knows it holds it, so refreshing ghosts is a pull:
 * every ghost (the pool suffix `[n_local, n)`) names itself to its owner, the
 * owner looks the name up among what it OWNS and answers, and the answers come
 * back in request order, so they are applied by position. Positions, normals,
 * registered fields, the remesher's ids, sizing and feature classes, and the ids
 * a topology change hands out are payloads on this one schedule (five copies of
 * it existed before; `docs/analysis.md` §2.7).
 *
 * ## What is a parameter, and what is not
 *
 * The pools (a reply width per pool, zero leaves it out), the key (the global
 * id, or a caller's 64-bit key for entities whose ids are not agreed yet), the
 * route (graph only, or graph with an agreed global fallback) and the payload
 * (two callbacks). The payload callbacks run once per (body, pool) BLOCK, never
 * per ghost: an earlier shared version that asked "which payload?" once per
 * ghost slowed the per-timestep position sync (remesh_comm.h), and a block call
 * keeps each payload's inner loop its own.
 *
 * ## Cost per call
 *
 * Requests 8 B per ghost (12 B keyed), replies `row_bytes`; three exchanges on
 * the route, the count exchange overlapped with packing the requests, and the
 * reply needing no count exchange because there is one reply per request. One
 * `MPI_Allreduce` more under the AUTO route, none under GRAPH. Nothing per call
 * is sized by the process count on the graph route (the rank-to-neighbour
 * table lives in the communication workspace) or by a large body's id range:
 * the owner keys its lookup by the REQUESTS and streams its owned ids past it
 * (resolve_ids()), so its table is a bounded direct one for a small body and
 * O(requests) for a large one, and the owned ids are read once, sequentially.
 */

#include "mftl_communication.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_memory.h"

/* Both records carry (global_id, owner_proc) as two adjacent ints, so one
 * strided view reads either pool without a branch per entity. */
_Static_assert(offsetof(MFTL_Vertex, owner_proc) == offsetof(MFTL_Vertex, global_id) + sizeof(int), "MFTL_Vertex: owner_proc must follow global_id");
_Static_assert(offsetof(MFTL_Triangle, owner_proc) == offsetof(MFTL_Triangle, global_id) + sizeof(int), "MFTL_Triangle: owner_proc must follow global_id");
_Static_assert(sizeof(MFTL_Vertex) % sizeof(int) == 0 && sizeof(MFTL_Triangle) % sizeof(int) == 0, "records must be whole ints");

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Pool views and the plan
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One pool of one body, read through its identity ints. */
typedef struct
{
  const int *id;  ///< `&rec[0].global_id`: entity i's id is `id[i * stride]`, its owner the next int.
  int stride;  ///< Ints per record.
  int n_local;  ///< Owned entities, the prefix.
  int n;  ///< All entities; `[n_local, n)` are ghosts.
  int range;  ///< One past the largest id the body hands out (`next_*_gid`).
} pool_view;

/**
 * @brief The view of pool @p pool of @p obj.
 * @param obj  The body.
 * @param pool #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE.
 * @return The view; empty for an unallocated pool.
 */
static pool_view view_of(const MFTL_Object *obj, int pool)
{
  pool_view v;
  if (pool == MFTL_FIELD_VERTEX)
  {
    v.id = (obj->verts.rec != MFTL_NULL) ? &obj->verts.rec[0].global_id : MFTL_NULL;
    v.stride = (int) (sizeof(MFTL_Vertex) / sizeof(int));
    v.n_local = obj->verts.n_local;
    v.n = obj->verts.n;
    v.range = obj->next_vertex_gid;
  }
  else
  {
    v.id = (obj->tris.rec != MFTL_NULL) ? &obj->tris.rec[0].global_id : MFTL_NULL;
    v.stride = (int) (sizeof(MFTL_Triangle) / sizeof(int));
    v.n_local = obj->tris.n_local;
    v.n = obj->tris.n;
    v.range = obj->next_triangle_gid;
  }
  if (v.id == MFTL_NULL) v.n_local = v.n = 0;
  return v;
}

/** @brief What one call decided: route, slots, request layout. */
typedef struct
{
  MFTL_Ensemble *ens;
  const MFTL_GhostPull *pull;
  const int *nslot;  ///< `[nproc]` rank -> graph slot (#MFTL_ERROR: not a neighbour).
  int nproc;
  int n_slots;  ///< Graph degree, or nproc on the global route.
  int rq;  ///< Ints per request: 2 (body, id) or 3 (body, 64-bit key).
  bool global;  ///< The global route was agreed.
  bool keyed;  ///< Ghosts are named by the caller's keys, not by global id.
} pull_plan;

/**
 * @brief Whether pool @p p of local body @p o takes part.
 * @param pl The plan.
 * @param o  Local slot.
 * @param p  Pool.
 * @return True if its ghosts ask and it answers.
 */
static inline bool takes_part(const pull_plan *pl, int o, int p)
{
  if (pl->pull->row_bytes[p] <= 0) return MFTL_FALSE;
  return !pl->keyed || (pl->pull->keys[p] != MFTL_NULL && pl->pull->keys[p][o] != MFTL_NULL);
}

/**
 * @brief The exchange slot that reaches rank @p owner, or #MFTL_ERROR.
 * @param pl    The plan.
 * @param owner A ghost's `owner_proc`.
 * @return Graph slot or rank; #MFTL_ERROR if unreachable on this route or corrupt.
 */
static inline int slot_of_owner(const pull_plan *pl, int owner)
{
  if (owner < 0 || owner >= pl->nproc) return MFTL_ERROR;
  return pl->global ? owner : pl->nslot[owner];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Requester side: count, then pack
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Counts the requests per (slot, pool): `cnt[2 * slot + pool]`.
 * @param      pl   The plan.
 * @param[out] cnt  `[2 * n_slots]`, zeroed by the caller.
 * @param[out] far  Ghosts whose owner is a valid rank but not reachable on this route.
 * @param[out] bad  Ghosts whose `owner_proc` is outside the communicator.
 */
static void count_requests(const pull_plan *pl, int *cnt, int *far, int *bad)
{
  for (int o = 0; o < pl->ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = pl->ens->local_objects[o];
    for (int p = 0; p < 2; ++p)
    {
      if (!takes_part(pl, o, p)) continue;
      const pool_view v = view_of(obj, p);
      for (int i = v.n_local; i < v.n; ++i)
      {
        const int owner = v.id[(size_t) i * v.stride + 1];
        const int d = slot_of_owner(pl, owner);
        if (d != MFTL_ERROR)
          cnt[2 * d + p]++;
        else if (owner >= 0 && owner < pl->nproc)
          (*far)++;
        else
          (*bad)++;
      }
    }
  }
}

/**
 * @brief Packs every request at its (slot, pool) cursor, and beside it the
 *        TARGET (local slot, ghost index) that never leaves this rank.
 *
 * The same walk as count_requests(), so each (slot, pool) segment lists its
 * bodies in local-slot order: the owner meets one run per body, and the apply
 * phase meets the same runs.
 *
 * @param         pl    The plan.
 * @param[in,out] cur   `[2 * n_slots]` next request index per (slot, pool).
 * @param[out]    sreq  Requests, `rq` ints each.
 * @param[out]    tslot Local slot of each request's body.
 * @param[out]    tidx  Ghost index of each request.
 */
static void pack_requests(const pull_plan *pl, int *cur, int *sreq, int *tslot, int *tidx)
{
  const int rq = pl->rq;
  for (int o = 0; o < pl->ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = pl->ens->local_objects[o];
    for (int p = 0; p < 2; ++p)
    {
      if (!takes_part(pl, o, p)) continue;
      const pool_view v = view_of(obj, p);
      const long long *const key = pl->keyed ? pl->pull->keys[p][o] : MFTL_NULL;
      for (int i = v.n_local; i < v.n; ++i)
      {
        const int d = slot_of_owner(pl, v.id[(size_t) i * v.stride + 1]);
        if (d == MFTL_ERROR) continue;  // counted, reported by the caller
        const int pos = cur[2 * d + p]++;
        int *const r = sreq + (size_t) pos * rq;
        r[0] = obj->global_id;  // the body by GLOBAL id: the owner has its own slot for it
        if (key != MFTL_NULL)
          memcpy(&r[1], &key[i], sizeof(long long));
        else
          r[1] = v.id[(size_t) i * v.stride];
        tslot[pos] = o;
        tidx[pos] = i;
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Owner side: the lookups
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Scratch of the keyed lookup, grown across the bodies of one call. */
typedef struct
{
  uint64_t *key;  ///< [cap] owned keys, sorted.
  int *idx;  ///< [cap] their pool indices.
  int cap;  ///< Entries allocated.
} key_index;

/**
 * @brief The largest id range resolved through a DIRECT table: 2^16 slots, 768 KB
 *        of tags and values, L2-resident on the cores the library targets.
 *        Above it a direct table's misses cost more than the hashed regime's
 *        filter (one 655 362-vertex body at np = 4: an owned-keyed 12 MB direct
 *        table 425 us per call, the filter 117 us).
 */
#define GHOST_DIRECT_RANGE 65536

/** @brief The hashed regime's bit filter over the requested ids, grown across the bodies of one call. */
typedef struct
{
  uint64_t *bits;  ///< [cap] words.
  int cap;  ///< Words allocated.
} id_filter;

/**
 * @brief Resolves @p n global ids against the OWNED entities of one pool.
 *
 * The table is keyed by the REQUESTED ids and the owned ids stream past it
 * once. Keying it by the owned ids -- the obvious way round, and what every
 * former copy did -- writes one entry per OWNED entity to answer a few
 * requests: on a body spanning the ranks (82 000 owned ids for 1 500 requests
 * at np = 8) that is random writes into megabytes. Two regimes; memory is
 * bounded by a constant in the first and O(n) in the second, never by what the
 * rank owns or by a large body's id range:
 *
 * - DIRECT, when the body's whole id range fits #GHOST_DIRECT_RANGE slots (a
 *   droplet): the request table is indexed by id, and an owned id is checked
 *   with one load and a predictable branch. What decides this is whether the
 *   table stays in cache, not how it compares to the request count: a droplet
 *   grazing a cut plane gets a few requests and still wants the direct table.
 * - HASHED otherwise (a body spanning the ranks): nearly every owned id is
 *   unasked, and a MISS in an open-addressed table mispredicts its loop branch
 *   whenever the first slot is taken (5.6 ns per owned id, measured). A bit
 *   filter of at least 32 bits per request (L1-sized) rejects ~97 % of them
 *   with one load first, and only its hits probe the table.
 *
 * @param      tbl  The workspace's epoch-stamped table.
 * @param      f    The filter scratch.
 * @param      v    The pool.
 * @param      rreq Received requests.
 * @param      rq   Ints per request.
 * @param      rk   `[n]` request index of each entry.
 * @param      n    Entries.
 * @param[out] idx  `[n]` owned index, or #MFTL_ERROR.
 */
static void resolve_ids(mftl_gid_table *tbl, id_filter *f, const pool_view *v, const int *rreq, int rq, const int *rk, int n, int *idx)
{
  /* Ids at or beyond the body's range cannot be owned; keeping them out of the
   * table also keeps direct mode inside its bounds. */
  const int range = MFTL_MAX(v->range, 1);
  const int *const restrict id = v->id;
  const size_t stride = (size_t) v->stride;

  if (range <= GHOST_DIRECT_RANGE)
  {
    gt_begin_direct(tbl, range);
    for (int j = 0; j < n; ++j)
    {
      const int gid = rreq[(size_t) rk[j] * rq + 1];
      if (gid >= 0 && gid < range) gt_put(tbl, gid, MFTL_ERROR);
    }
    for (int i = 0; i < v->n_local; ++i)
    {
      const int gid = id[i * stride];
      MFTL_ASSERT(gid >= 0 && gid < range && "ghost pull: owned id outside the body's id range");
      bool asked = false;
      const int s = gt_probe(tbl, gid, &asked);
      if (asked) tbl->value[s] = i;
    }
  }
  else
  {
    int fbits = 6;  // 2^fbits >= 32 n bits
    while (fbits < 30 && (1LL << fbits) < 32LL * n) ++fbits;
    const int words = 1 << (fbits - 6);
    if (words > f->cap)
    {
      free(f->bits);
      f->cap = words;
      f->bits = (uint64_t *) MFTLMallocOrDie((size_t) words * sizeof(uint64_t), "ghost pull id filter");
    }
    memset(f->bits, 0, (size_t) words * sizeof(uint64_t));
    uint64_t *const restrict bits = f->bits;
    const int shift = 32 - fbits;  // the filter takes the HIGH bits of the hash, the table starts its probe at the low ones

    gt_begin(tbl, n, range);
    for (int j = 0; j < n; ++j)
    {
      const int gid = rreq[(size_t) rk[j] * rq + 1];
      if (gid < 0 || gid >= range) continue;
      gt_put(tbl, gid, MFTL_ERROR);
      const unsigned b = gt_hash(gid) >> shift;
      bits[b >> 6] |= 1ULL << (b & 63u);
    }
    for (int i = 0; i < v->n_local; ++i)
    {
      const int gid = id[i * stride];
      MFTL_ASSERT(gid >= 0 && gid < range && "ghost pull: owned id outside the body's id range");
      const unsigned b = gt_hash(gid) >> shift;
      if (!((bits[b >> 6] >> (b & 63u)) & 1ULL)) continue;  // not asked: the common case
      bool asked = false;
      const int s = gt_probe(tbl, gid, &asked);
      if (asked) tbl->value[s] = i;
    }
  }

  for (int j = 0; j < n; ++j)
  {
    const int gid = rreq[(size_t) rk[j] * rq + 1];
    idx[j] = (gid >= 0 && gid < range) ? gt_find(tbl, gid) : MFTL_ERROR;
  }
}

/**
 * @brief Resolves @p n 64-bit keys against the OWNED entities of one pool:
 *        sort once (radix), then bisect. Keys need not be dense or bounded.
 * @param      ki   Scratch.
 * @param      key  `[pool size]` the body's keys, or #MFTL_NULL (nothing served).
 * @param      v    The pool.
 * @param      rreq Received requests.
 * @param      rq   Ints per request.
 * @param      rk   `[n]` request index of each entry.
 * @param      n    Entries.
 * @param[out] idx  `[n]` owned index, or #MFTL_ERROR.
 */
static void resolve_keys(key_index *ki, const long long *key, const pool_view *v, const int *rreq, int rq, const int *rk, int n, int *idx)
{
  const int m = (key != MFTL_NULL) ? v->n_local : 0;
  if (m > ki->cap)
  {
    free(ki->key);
    free(ki->idx);
    ki->cap = m;
    ki->key = (uint64_t *) MFTLMallocOrDie((size_t) m * sizeof(uint64_t), "ghost pull keys");
    ki->idx = (int *) MFTLMallocOrDie((size_t) m * sizeof(int), "ghost pull key indices");
  }
  for (int i = 0; i < m; ++i)
  {
    ki->key[i] = (uint64_t) key[i];  // any total order will do for a bisection
    ki->idx[i] = i;
  }
  if (m > 1) MFTLSortPairs64(ki->key, ki->idx, m);

  for (int j = 0; j < n; ++j)
  {
    long long sk;
    memcpy(&sk, &rreq[(size_t) rk[j] * rq + 1], sizeof sk);
    const uint64_t want = (uint64_t) sk;
    int lo = 0, hi = m - 1, hit = MFTL_ERROR;
    while (lo <= hi)
    {
      const int mid = lo + (hi - lo) / 2;
      if (ki->key[mid] == want)
      {
        hit = ki->idx[mid];
        break;
      }
      if (ki->key[mid] < want)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    idx[j] = hit;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The engine
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Aborts the job when an exchange would exceed MPI's int counts.
 * @param dom   The domain.
 * @param value The count, computed in 64 bits.
 * @return @p value as an int.
 */
static int checked_count(const MFTL_Domain *dom, long long value)
{
  if (value < 0 || value > INT_MAX)
  {
    fprintf(stderr, "[MFTL rank %d] ghost pull: %lld elements exceed MPI's int count range\n", dom->rank, value);
    MPI_Abort(dom->comm_world, MFTL_ERROR);
    abort();
  }
  return (int) value;
}

int MFTLGhostPull(MFTL_Ensemble *ens, const MFTL_GhostPull *pull)
{
  /* Arguments are identical on every rank, so these refusals are unanimous. */
  if (ens == MFTL_NULL || pull == MFTL_NULL || pull->serve == MFTL_NULL || pull->apply == MFTL_NULL) return MFTL_ERROR;
  if (pull->row_bytes[0] < 0 || pull->row_bytes[1] < 0 || pull->row_bytes[0] + pull->row_bytes[1] == 0) return MFTL_ERROR;
  if (pull->route != MFTL_GHOST_ROUTE_GRAPH && pull->route != MFTL_GHOST_ROUTE_AUTO) return MFTL_ERROR;

  const MFTL_Domain *const dom = &ens->domain;
  const int nproc = dom->nproc;
  const int nn = dom->num_neighbors;
  const int w[2] = {pull->row_bytes[0], pull->row_bytes[1]};

  /* Guards on GLOBAL state only: a rank holding nothing must still enter every
   * collective below. A rank without graph neighbours expects no message on the
   * graph route. */
  if (ens->n_bodies_global == 0 || nproc == 1) return MFTL_SUCCESS;
  if (pull->route == MFTL_GHOST_ROUTE_GRAPH && nn == 0) return MFTL_SUCCESS;

  CS_BEGIN(ts_total);
  MFTLCommStats.g_n_calls++;
  CS_BEGIN(ts_collect);

  pull_plan pl;
  pl.ens = ens;
  pl.pull = pull;
  pl.nslot = MFTLCommWorkspaceNeighborSlots(ens);
  pl.nproc = nproc;
  pl.n_slots = nn;
  pl.global = MFTL_FALSE;
  pl.keyed = (pull->keys[0] != MFTL_NULL || pull->keys[1] != MFTL_NULL);
  pl.rq = pl.keyed ? 1 + (int) (sizeof(long long) / sizeof(int)) : 2;
  const int rq = pl.rq;

  /* ── 1. Count per (slot, pool); agree the route ──────────────────────────── */
  int *cnt = (int *) MFTLCallocOrDie((size_t) 2 * MFTL_MAX(nn, 1), sizeof(int), "ghost pull counts");
  int far = 0, bad = 0;
  count_requests(&pl, cnt, &far, &bad);

  if (pull->route == MFTL_GHOST_ROUTE_AUTO)
  {
    /* An owner outside the 26-neighbourhood (a coarsened placement, a body that
     * made redistribution go global) cannot be reached on the graph, and a
     * dropped request is silent: a stale id later tears the surface in the
     * closing redistribution. One int decides, and only a global verdict makes
     * any rank pay the nproc-sized arrays (and a second counting pass). */
    int go_global = (far > 0);
    MPI_Allreduce(MPI_IN_PLACE, &go_global, 1, MPI_INT, MPI_LOR, dom->comm_world);
    if (go_global)
    {
      MFTLCommStats.g_n_route_global++;
      pl.global = MFTL_TRUE;
      pl.n_slots = nproc;
      free(cnt);
      cnt = (int *) MFTLCallocOrDie((size_t) 2 * nproc, sizeof(int), "ghost pull counts");
      far = bad = 0;
      count_requests(&pl, cnt, &far, &bad);
    }
    else if (nn == 0)
    {
      free(cnt);
      CS_END(ts_collect, g_t_collect);
      CS_END(ts_total, g_t_total);
      return (bad == 0) ? MFTL_SUCCESS : MFTL_ERROR;
    }
  }

  const int S = pl.n_slots;
  const int orphaned = far + bad;
  MFTLCommStats.g_n_orphaned += orphaned;
  if (orphaned > 0)
  {
    /* Skipped, not aborted (aborting would deadlock the other ranks), but loud:
     * the stale value makes two ranks see the same entity differently. On the
     * graph route there is no fallback by design -- a partition whose halo
     * leaves the neighbourhood must not outlive the call that made it. */
    fprintf(stderr, "[MFTL rank %d] ghost pull: %d ghosts whose owner is not reachable (%d outside the communicator) were not refreshed\n", dom->rank, orphaned,
            bad);
    MFTL_ASSERT(0 && "ghost pull: ghost whose owner is unreachable -- its value is now stale");
  }
  CS_END(ts_collect, g_t_collect);

  /* ── 2. Counts out, overlapped with packing the requests ─────────────────── */
  CS_BEGIN(ts_counts);
  int *const rc = (int *) MFTLMallocOrDie((size_t) 2 * S * sizeof(int), "ghost pull receive counts");
  MPI_Request count_req;
  if (pl.global)
    MPI_Ialltoall(cnt, 2, MPI_INT, rc, 2, MPI_INT, dom->comm_world, &count_req);
  else
    MPI_Ineighbor_alltoall(cnt, 2, MPI_INT, rc, 2, MPI_INT, dom->comm_graph, &count_req);

  /* [send counts | send displs | recv counts | recv displs] in INTS for the
   * request exchange, then the same four in BYTES for the reply. */
  int *const ix = (int *) MFTLMallocOrDie((size_t) 8 * S * sizeof(int), "ghost pull displacements");
  int *const s_ints = ix, *const s_disp = ix + S, *const r_ints = ix + 2 * S, *const r_disp = ix + 3 * S;
  int *const y_bytes = ix + 4 * S, *const y_disp = ix + 5 * S, *const x_bytes = ix + 6 * S, *const x_disp = ix + 7 * S;

  int *const seg = (int *) MFTLMallocOrDie((size_t) 4 * S * sizeof(int), "ghost pull segments");  // [begin | cursor] per (slot, pool)
  int *const cur = seg + 2 * S;
  long long n_out_ll = 0, x_total = 0;
  for (int d = 0; d < S; ++d)
  {
    s_disp[d] = checked_count(dom, n_out_ll * rq);
    seg[2 * d] = (int) n_out_ll;
    n_out_ll += cnt[2 * d];
    seg[2 * d + 1] = (int) n_out_ll;
    n_out_ll += cnt[2 * d + 1];
    s_ints[d] = checked_count(dom, ((long long) cnt[2 * d] + cnt[2 * d + 1]) * rq);
    x_disp[d] = checked_count(dom, x_total);
    x_bytes[d] = checked_count(dom, (long long) cnt[2 * d] * w[0] + (long long) cnt[2 * d + 1] * w[1]);
    x_total += x_bytes[d];
  }
  const int n_out = checked_count(dom, n_out_ll);
  (void) checked_count(dom, n_out_ll * rq);
  const int n_x = checked_count(dom, x_total);
  memcpy(cur, seg, (size_t) 2 * S * sizeof(int));

  int *const sreq = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_out * rq, 1) * sizeof(int), "ghost pull requests");
  int *const tslot = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_out, 1) * sizeof(int), "ghost pull target bodies");
  int *const tidx = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_out, 1) * sizeof(int), "ghost pull targets");
  pack_requests(&pl, cur, sreq, tslot, tidx);

  MPI_Wait(&count_req, MPI_STATUS_IGNORE);
  CS_END(ts_counts, g_t_counts);

  /* ── 3. Requests out ─────────────────────────────────────────────────────── */
  CS_BEGIN(ts_req);
  long long n_in_ll = 0, y_total = 0;
  for (int s = 0; s < S; ++s)
  {
    r_disp[s] = checked_count(dom, n_in_ll * rq);
    r_ints[s] = checked_count(dom, ((long long) rc[2 * s] + rc[2 * s + 1]) * rq);
    n_in_ll += (long long) rc[2 * s] + rc[2 * s + 1];
    y_disp[s] = checked_count(dom, y_total);
    y_bytes[s] = checked_count(dom, (long long) rc[2 * s] * w[0] + (long long) rc[2 * s + 1] * w[1]);
    y_total += y_bytes[s];
  }
  const int n_in = checked_count(dom, n_in_ll);
  (void) checked_count(dom, n_in_ll * rq);
  const int n_y = checked_count(dom, y_total);

  int *const rreq = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in * rq, 1) * sizeof(int), "ghost pull incoming requests");
  if (pl.global)
    MPI_Alltoallv(sreq, s_ints, s_disp, MPI_INT, rreq, r_ints, r_disp, MPI_INT, dom->comm_world);
  else
    MPI_Neighbor_alltoallv(sreq, s_ints, s_disp, MPI_INT, rreq, r_ints, r_disp, MPI_INT, dom->comm_graph);
  free(sreq);
  CS_END(ts_req, g_t_requests);

  /* ── 4. Answer, grouped per (body, pool) ──────────────────────────────────
   * The stream arrives by sender, and within a sender by pool and body, so
   * answering in arrival order would build a body's table once per SENDER. A
   * counting sort into groups 2*slot + pool -- plus one group per pool for
   * requests naming a body this rank does not hold -- builds it once. Each
   * reply goes to its request's ORIGINAL position, which is how the requester
   * matches it. */
  CS_TIMELINE(tm);
  const int nb = ens->n_local_objects;
  const int ng = 2 * nb + 2;
  unsigned char *const reply = (unsigned char *) MFTLMallocOrDie((size_t) MFTL_MAX(n_y, 1), "ghost pull replies");
  int *const goff = (int *) MFTLCallocOrDie((size_t) ng + 1, sizeof(int), "ghost pull groups");
  int *const scratch = (int *) MFTLMallocOrDie((size_t) 4 * MFTL_MAX(n_in, 1) * sizeof(int), "ghost pull grouping");
  int *const gk = scratch;  // group of request k
  int *const rk = scratch + MFTL_MAX(n_in, 1);  // request index, in group order
  int *const roff = scratch + 2 * (size_t) MFTL_MAX(n_in, 1);  // its reply offset, in group order
  int *const ridx = scratch + 3 * (size_t) MFTL_MAX(n_in, 1);  // the owned index it resolves to

  {
    /* Body ids arrive in runs: resolve each run once, not each request. */
    int k = 0, last_body = INT_MIN, last_slot = MFTL_ERROR;
    for (int s = 0; s < S; ++s)
      for (int p = 0; p < 2; ++p)
        for (int j = 0; j < rc[2 * s + p]; ++j, ++k)
        {
          const int body = rreq[(size_t) k * rq];
          if (body != last_body)
          {
            last_body = body;
            last_slot = MFTLEnsembleFindLocalSlot(ens, body);
          }
          const int o = (last_slot != MFTL_ERROR && takes_part(&pl, last_slot, p)) ? last_slot : MFTL_ERROR;
          gk[k] = (o != MFTL_ERROR) ? 2 * o + p : 2 * nb + p;
          goff[gk[k] + 1]++;
        }
  }
  for (int g = 0; g < ng; ++g) goff[g + 1] += goff[g];
  {
    int *const cursor = (int *) MFTLMallocOrDie((size_t) ng * sizeof(int), "ghost pull group cursors");
    memcpy(cursor, goff, (size_t) ng * sizeof(int));
    int k = 0;
    for (int s = 0; s < S; ++s)
    {
      int off = y_disp[s];
      for (int p = 0; p < 2; ++p)
        for (int j = 0; j < rc[2 * s + p]; ++j, ++k, off += w[p])
        {
          const int pos = cursor[gk[k]]++;
          rk[pos] = k;
          roff[pos] = off;
        }
    }
    free(cursor);
  }

  int unserved = 0;
  {
    mftl_gid_table *const tbl = pl.keyed ? MFTL_NULL : MFTLCommWorkspaceGhostTable(ens);
    key_index ki = {MFTL_NULL, MFTL_NULL, 0};
    id_filter filter = {MFTL_NULL, 0};

    for (int g = 0; g < ng; ++g)
    {
      const int first = goff[g];
      const int n = goff[g + 1] - first;
      if (n == 0) continue;
      const int p = g % 2;

      if (g >= 2 * nb)
      {
        /* The requester named this rank as the owner of a body it does not hold:
         * a disagreement about ownership. Answered with the payload's sentinel. */
        MFTL_ASSERT(0 && "ghost pull: request names a body this rank does not hold");
        for (int j = 0; j < n; ++j) ridx[first + j] = MFTL_ERROR;
        unserved += n;
        pull->serve(MFTL_NULL, MFTL_ERROR, (MFTL_FieldEntity) p, ridx + first, roff + first, n, reply, pull->ctx);
        continue;
      }

      const int o = g / 2;
      const MFTL_Object *const obj = ens->local_objects[o];
      const pool_view v = view_of(obj, p);

      CS_LAP(tm, g_t_answer);
      if (pl.keyed)
        resolve_keys(&ki, pull->keys[p][o], &v, rreq, rq, rk + first, n, ridx + first);
      else
        resolve_ids(tbl, &filter, &v, rreq, rq, rk + first, n, ridx + first);
      CS_LAP(tm, g_t_table);

      for (int j = 0; j < n; ++j)
      {
        if (ridx[first + j] != MFTL_ERROR) continue;
        MFTL_ASSERT(0 && "ghost pull: asked for an entity this rank does not own");
        unserved++;
      }
      pull->serve(obj, o, (MFTL_FieldEntity) p, ridx + first, roff + first, n, reply, pull->ctx);
    }
    free(ki.key);
    free(ki.idx);
    free(filter.bits);
  }
  free(scratch);
  free(goff);
  free(rreq);
  CS_LAP(tm, g_t_answer);

  /* ── 5. Replies back: one per request, so both sides know the counts ─────── */
  CS_BEGIN(ts_rep);
  unsigned char *const incoming = (unsigned char *) MFTLMallocOrDie((size_t) MFTL_MAX(n_x, 1), "ghost pull incoming replies");
  if (pl.global)
    MPI_Alltoallv(reply, y_bytes, y_disp, MPI_BYTE, incoming, x_bytes, x_disp, MPI_BYTE, dom->comm_world);
  else
    MPI_Neighbor_alltoallv(reply, y_bytes, y_disp, MPI_BYTE, incoming, x_bytes, x_disp, MPI_BYTE, dom->comm_graph);
  free(reply);
  CS_END(ts_rep, g_t_replies);

  /* ── 6. Apply, one call per run of one (body, pool) ──────────────────────── */
  CS_BEGIN(ts_apply);
  int skipped = 0;
  for (int d = 0; d < S; ++d)
  {
    const unsigned char *rows = incoming + x_disp[d];
    for (int p = 0; p < 2; ++p)
    {
      const int begin = seg[2 * d + p];
      const int end = begin + cnt[2 * d + p];
      for (int k = begin; k < end;)
      {
        const int o = tslot[k];
        int k1 = k + 1;
        while (k1 < end && tslot[k1] == o) ++k1;
        skipped += pull->apply(ens->local_objects[o], o, (MFTL_FieldEntity) p, tidx + k, k1 - k, rows + (size_t) (k - begin) * w[p], pull->ctx);
        k = k1;
      }
      rows += (size_t) cnt[2 * d + p] * w[p];
    }
  }
  CS_END(ts_apply, g_t_apply);

  CS_BEGIN(ts_free);
  MFTLCommStats.g_n_ghosts_requested += n_out;
  MFTLCommStats.g_n_bytes_sent += (long) n_out * rq * (long) sizeof(int) + (long) n_y;
  MFTLCommStats.g_n_bytes_recv += (long) n_in * rq * (long) sizeof(int) + (long) n_x;
  free(incoming);
  free(tslot);
  free(tidx);
  free(seg);
  free(ix);
  free(rc);
  free(cnt);
  CS_END(ts_free, g_t_teardown);
  CS_END(ts_total, g_t_total);

  return (orphaned == 0 && unserved == 0 && skipped == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}

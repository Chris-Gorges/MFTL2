/**
 * @file exchange.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief The MPI datatypes, the routed personalised exchange (with the
 *        library's one NBX), and the neighbourhood allgather.
 *
 * All three are shared by several modules and none knows anything about meshes
 * -- which is why they live in their own translation unit.
 */

#include "mftl_communication.h"

#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_memory.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. MPI datatypes
 *
 * Every wire struct groups its members into runs of ONE base type
 * (mftl_communication.h), so each datatype below is one or two CONTIGUOUS blocks
 * rather than one block per field. The static assertions pin exactly that: they
 * fail the BUILD if a future field breaks a run, instead of letting MPI silently
 * transmit the wrong bytes.
 * ═══════════════════════════════════════════════════════════════════════════ */

MPI_Datatype MFTL_MPI_PackedVertex = MPI_DATATYPE_NULL;
MPI_Datatype MFTL_MPI_PackedTriangleComplex = MPI_DATATYPE_NULL;
MPI_Datatype MFTL_MPI_GhostUpdatePayload = MPI_DATATYPE_NULL;

_Static_assert(offsetof(MFTL_PackedVertex, global_id) == 3 * sizeof(double), "MFTL_PackedVertex: the doubles are not one run");
_Static_assert(sizeof(MFTL_PackedVertex) == 3 * sizeof(double) + 4 * sizeof(int), "MFTL_PackedVertex: the ints are not one run");
_Static_assert(offsetof(MFTL_PackedTriangleComplex, vertex_gid) == 4 * sizeof(int), "MFTL_PackedTriangleComplex: not one int run");
_Static_assert(sizeof(MFTL_PackedTriangleComplex) == 10 * sizeof(int), "MFTL_PackedTriangleComplex: not one int run");
_Static_assert(offsetof(MFTL_GhostUpdatePayload, global_id) == 3 * sizeof(double), "MFTL_GhostUpdatePayload: the doubles are not one run");

/**
 * @brief Builds a struct datatype whose extent is pinned to the C `sizeof`
 *        (`MPI_Type_create_resized`), so an array of it has the C stride
 *        regardless of how trailing padding falls.
 *
 * @param n_blocks     Number of member blocks.
 * @param blocklengths Element count of each block.
 * @param offsets      Byte offset of each block from the struct base.
 * @param types        MPI datatype of each block's elements.
 * @param c_size       The C `sizeof` of the struct, pinned as the type's extent.
 * @return The committed datatype.
 */
static MPI_Datatype commit_struct(int n_blocks, const int *blocklengths, const MPI_Aint *offsets, const MPI_Datatype *types, size_t c_size)
{
  MPI_Datatype raw, out;
  MPI_Type_create_struct(n_blocks, (int *) blocklengths, (MPI_Aint *) offsets, (MPI_Datatype *) types, &raw);
  MPI_Type_create_resized(raw, 0, (MPI_Aint) c_size, &out);
  MPI_Type_free(&raw);
  MPI_Type_commit(&out);
  return out;
}

void MFTLRegisterMpiTypes(void)
{
  if (MFTL_MPI_PackedVertex != MPI_DATATYPE_NULL) return;  // already registered

  {
    const int blocklengths[] = {3, 4};
    const MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT};
    const MPI_Aint offsets[] = {offsetof(MFTL_PackedVertex, pos), offsetof(MFTL_PackedVertex, global_id)};
    MFTL_MPI_PackedVertex = commit_struct(2, blocklengths, offsets, types, sizeof(MFTL_PackedVertex));
  }

  {
    /* Ten ints: a plain contiguous run. */
    MPI_Type_contiguous(10, MPI_INT, &MFTL_MPI_PackedTriangleComplex);
    MPI_Type_commit(&MFTL_MPI_PackedTriangleComplex);
  }

  {
    const int blocklengths[] = {3, 1};
    const MPI_Datatype types[] = {MPI_DOUBLE, MPI_INT};
    const MPI_Aint offsets[] = {offsetof(MFTL_GhostUpdatePayload, value), offsetof(MFTL_GhostUpdatePayload, global_id)};
    MFTL_MPI_GhostUpdatePayload = commit_struct(2, blocklengths, offsets, types, sizeof(MFTL_GhostUpdatePayload));
  }
}

void MFTLFreeMpiTypes(void)
{
  if (MFTL_MPI_PackedVertex == MPI_DATATYPE_NULL) return;

  MPI_Type_free(&MFTL_MPI_PackedVertex);
  MPI_Type_free(&MFTL_MPI_PackedTriangleComplex);
  MPI_Type_free(&MFTL_MPI_GhostUpdatePayload);

  MFTL_MPI_PackedVertex = MPI_DATATYPE_NULL;
  MFTL_MPI_PackedTriangleComplex = MPI_DATATYPE_NULL;
  MFTL_MPI_GhostUpdatePayload = MPI_DATATYPE_NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The routed personalised exchange
 *
 * Every "send these elements to those ranks" of the library is a count
 * exchange and then the data, on the graph route (neighbourhood collectives)
 * or the global route (NBX discovery, then point to point). The front-ends --
 * typed payloads (2d), fixed-width records and their replies (2e), the ghost
 * pull (ghost_pull.c) -- differ only in how they lay out their buffers.
 *
 * The global route replaced a dense MPI_Alltoall of counts plus an
 * MPI_Alltoallv over comm_world, both O(P) per rank per call however few
 * partners a rank had; three modules carried their own copy of that pair and of
 * the vote in front of it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── 2a. NBX ──────────────────────────────────────────────────────────────── */

void MFTLNbxComplete(MPI_Comm comm, int tag, int n_sends, MPI_Request *sends, MFTL_NbxReceiveFn receive, void *ctx)
{
  MPI_Request barrier = MPI_REQUEST_NULL;
  bool in_barrier = MFTL_FALSE;
  for (;;)
  {
    int flag = 0;
    MPI_Status st;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &st);
    if (flag)
    {
      receive(comm, &st, ctx);
      continue;
    }
    int done = 0;
    if (!in_barrier)
    {
      /* Synchronous sends complete only once MATCHED: entering the barrier says
       * "every message of mine has been received". */
      if (n_sends > 0)
        MPI_Testall(n_sends, sends, &done, MPI_STATUSES_IGNORE);
      else
        done = 1;
      if (done)
      {
        MPI_Ibarrier(comm, &barrier);
        in_barrier = MFTL_TRUE;
      }
    }
    else
    {
      MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
      if (done) return;
    }
  }
}

/* ── 2b. The route ───────────────────────────────────────────────────────── */

bool MFTLExchangeAgreeRoute(const MFTL_Domain *domain, MFTL_Route route, bool far)
{
  if (route != MFTL_ROUTE_AUTO || domain->nproc < 2) return MFTL_FALSE;
  int g = far ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_INT, MPI_LOR, domain->comm_world);
  return g != 0;
}

/* ── 2c. Counts and data ─────────────────────────────────────────────────── */

/** @brief Discovery messages as they arrive: `[n * (1 + k)]` (source, k counts). */
typedef struct
{
  int *rec;  ///< The records.
  int n;  ///< Records held.
  int cap;  ///< Records allocated.
  int k;  ///< Counts per record.
} discovered;

/** @brief MFTL_NbxReceiveFn of the count discovery: one record per source. */
static void receive_counts(MPI_Comm comm, const MPI_Status *probed, void *ctx)
{
  discovered *const d = (discovered *) ctx;
  const size_t w = (size_t) (1 + d->k);
  if (d->n == d->cap)
  {
    d->cap = MFTL_MAX(8, 2 * d->cap);
    d->rec = (int *) MFTLReallocOrDie(d->rec, (size_t) d->cap * w * sizeof(int), "routed exchange discovery");
  }
  int *const r = d->rec + (size_t) d->n++ * w;
  r[0] = probed->MPI_SOURCE;
  MPI_Recv(r + 1, d->k, MPI_INT, probed->MPI_SOURCE, probed->MPI_TAG, comm, MPI_STATUS_IGNORE);
}

/** @brief qsort comparator on a record's first int (its rank). */
static int cmp_leading_rank(const void *a, const void *b)
{
  const int x = *(const int *) a, y = *(const int *) b;
  return (x > y) - (x < y);
}

void MFTLRouteCountsBegin(const MFTL_Domain *domain, mftl_route *rt, int k, int n_dst, const int *dst, const int *dcnt, bool overlap)
{
  rt->k = k;
  rt->own = MFTL_NULL;
  rt->req = MPI_REQUEST_NULL;
  rt->sreq = MFTL_NULL;
  rt->n_sreq = 0;
  rt->n_dst = n_dst;
  rt->dst = dst;
  rt->dcnt = dcnt;

  if (!rt->global)
  {
    const int nn = domain->num_neighbors;
    MFTL_ASSERT(n_dst == nn && "routed exchange: the graph route's destinations are the graph neighbours, by slot");
    rt->n = nn;
    rt->peer = domain->neighbors;
    rt->scnt = dcnt;
    rt->rcnt = rt->own = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nn * k, 1) * sizeof(int), "routed exchange receive counts");
    if (overlap)
      MPI_Ineighbor_alltoall(dcnt, k, MPI_INT, rt->rcnt, k, MPI_INT, domain->comm_graph, &rt->req);
    else
      MPI_Neighbor_alltoall(dcnt, k, MPI_INT, rt->rcnt, k, MPI_INT, domain->comm_graph);
    return;
  }

  /* One synchronous send per destination that gets anything; MFTLRouteCountsEnd()
   * completes the NBX round. */
  rt->sreq = (MPI_Request *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dst, 1) * sizeof(MPI_Request), "routed exchange discovery sends");
  for (int i = 0; i < n_dst; ++i)
  {
    MFTL_ASSERT((i == 0 || dst[i - 1] < dst[i]) && "routed exchange: global-route destinations must ascend");
    bool any = MFTL_FALSE;
    for (int j = 0; j < k; ++j) any |= (dcnt[(size_t) i * k + j] != 0);
    if (any) MPI_Issend(dcnt + (size_t) i * k, k, MPI_INT, dst[i], MFTL_TAG_ROUTE_COUNTS, domain->comm_world, &rt->sreq[rt->n_sreq++]);
  }
}

void MFTLRouteCountsEnd(const MFTL_Domain *domain, mftl_route *rt)
{
  if (!rt->global)
  {
    if (rt->req != MPI_REQUEST_NULL) MPI_Wait(&rt->req, MPI_STATUS_IGNORE);
    return;
  }

  const int k = rt->k;
  discovered d = {MFTL_NULL, 0, 0, k};
  MFTLNbxComplete(domain->comm_world, MFTL_TAG_ROUTE_COUNTS, rt->n_sreq, rt->sreq, receive_counts, &d);
  free(rt->sreq);
  rt->sreq = MFTL_NULL;
  rt->n_sreq = 0;

  /* Peers: the ascending union of the destinations that were sent something and
   * the sources. Arrival order is not deterministic; this is. */
  const size_t w = (size_t) (1 + k);
  if (d.n > 1) qsort(d.rec, (size_t) d.n, w * sizeof(int), cmp_leading_rank);

  const int cap = MFTL_MAX(rt->n_dst + d.n, 1);
  int *const own = (int *) MFTLMallocOrDie((size_t) cap * (1 + 2 * (size_t) k) * sizeof(int), "routed exchange peers");
  int *const peer = own;
  int *const scnt = own + cap;
  int *const rcnt = scnt + (size_t) cap * k;

  int n = 0, i = 0, j = 0;
  for (;;)
  {
    while (i < rt->n_dst)  // a destination that was sent nothing is not a peer
    {
      bool any = MFTL_FALSE;
      for (int q = 0; q < k; ++q) any |= (rt->dcnt[(size_t) i * k + q] != 0);
      if (any) break;
      ++i;
    }
    if (i == rt->n_dst && j == d.n) break;
    const int ri = (i < rt->n_dst) ? rt->dst[i] : INT_MAX;
    const int rj = (j < d.n) ? d.rec[(size_t) j * w] : INT_MAX;
    const int r = MFTL_MIN(ri, rj);
    peer[n] = r;
    for (int q = 0; q < k; ++q)
    {
      scnt[(size_t) n * k + q] = (ri == r) ? rt->dcnt[(size_t) i * k + q] : 0;
      rcnt[(size_t) n * k + q] = (rj == r) ? d.rec[(size_t) j * w + 1 + q] : 0;
    }
    if (ri == r) ++i;
    if (rj == r) ++j;
    ++n;
  }
  free(d.rec);

  rt->n = n;
  rt->peer = peer;
  rt->scnt = scnt;
  rt->rcnt = rcnt;
  rt->own = own;
}

void MFTLRoutePost(const MFTL_Domain *domain, bool global, int n, const int *peer, MPI_Datatype type, const void *sbuf, const int *sc, const int *sd,
                   void *rbuf, const int *rc, const int *rd, int tag, MPI_Request **reqs, int *n_reqs)
{
  if (!global)
  {
    if (reqs == MFTL_NULL)
    {
      MPI_Neighbor_alltoallv(sbuf, sc, sd, type, rbuf, rc, rd, type, domain->comm_graph);
      return;
    }
    *reqs = (MPI_Request *) MFTLMallocOrDie(sizeof(MPI_Request), "routed exchange request");
    *n_reqs = 1;
    MPI_Ineighbor_alltoallv(sbuf, sc, sd, type, rbuf, rc, rd, type, domain->comm_graph, *reqs);
    return;
  }

  MPI_Aint lb = 0, extent = 0;
  MPI_Type_get_extent(type, &lb, &extent);
  MPI_Request *const r = (MPI_Request *) MFTLMallocOrDie((size_t) MFTL_MAX(2 * n, 1) * sizeof(MPI_Request), "routed exchange requests");
  int m = 0;
  /* Receives first, so no message arrives unexpected. */
  for (int i = 0; i < n; ++i)
    if (rc[i] > 0) MPI_Irecv((char *) rbuf + (MPI_Aint) rd[i] * extent, rc[i], type, peer[i], tag, domain->comm_world, &r[m++]);
  for (int i = 0; i < n; ++i)
    if (sc[i] > 0) MPI_Isend((const char *) sbuf + (MPI_Aint) sd[i] * extent, sc[i], type, peer[i], tag, domain->comm_world, &r[m++]);

  if (reqs == MFTL_NULL)
  {
    if (m > 0) MPI_Waitall(m, r, MPI_STATUSES_IGNORE);
    free(r);
    return;
  }
  *reqs = r;
  *n_reqs = m;
}

void MFTLRouteFree(mftl_route *rt)
{
  free(rt->own);
  free(rt->sreq);
  rt->own = MFTL_NULL;
  rt->sreq = MFTL_NULL;
  rt->n_sreq = 0;
}

/** @brief Aborts the job on a count outside MPI's int range; returns @p value as an int. */
static int checked_int(const MFTL_Domain *domain, long long value, const char *what)
{
  if (value < 0 || value > INT_MAX)
  {
    fprintf(stderr, "[MFTL rank %d] routed exchange: %s (%lld) is outside MPI's int count range\n", domain->rank, what, value);
    MPI_Abort(domain->comm_world, MFTL_ERROR);
    abort();
  }
  return (int) value;
}

/* ── 2d. Typed payloads ──────────────────────────────────────────────────── */

void MFTLExchangeStart(const MFTL_Domain *domain, MFTL_Exchange *x, int n_payloads, MFTL_Route route, bool far, MFTL_ExchangeCtx *ctx)
{
  MFTL_ASSERT(n_payloads >= 1 && n_payloads <= MFTL_MAX_EXCHANGE_PAYLOADS && "MFTLExchangeStart: payload count out of range");
  MFTL_ASSERT((route == MFTL_ROUTE_GRAPH || route == MFTL_ROUTE_AUTO) && "MFTLExchangeStart: unknown route");
  MFTL_ASSERT(domain->comm_graph != MPI_COMM_NULL && "MFTLExchangeStart: no graph communicator");

  const int np = n_payloads;
  ctx->n_payloads = np;
  for (int p = 0; p < MFTL_MAX_EXCHANGE_PAYLOADS; ++p)
  {
    ctx->requests[p] = MFTL_NULL;
    ctx->n_requests[p] = 0;
    ctx->flat_send[p] = MFTL_NULL;
  }

  CS_BEGIN(ts_route);
  const bool global = MFTLExchangeAgreeRoute(domain, route, far);
  CS_END(ts_route, t_route);
  ctx->global = global;

  /* ── 1. ONE count exchange for all payloads (latency, not volume, is its
   * cost), staged [destination][payload]. Graph: every neighbour by slot.
   * Global: the ranks this one has anything for, ascending. */
  CS_BEGIN(ts_counts);
  int n_dst = domain->num_neighbors;
  int *const dst_own = global ? (int *) MFTLMallocOrDie((size_t) domain->nproc * sizeof(int), "MFTLExchangeStart destinations") : MFTL_NULL;
  if (global)
  {
    n_dst = 0;
    for (int r = 0; r < domain->nproc; ++r)
    {
      bool any = MFTL_FALSE;
      for (int p = 0; p < np; ++p) any |= (x[p].send_bufs[r].count != 0);
      if (any) dst_own[n_dst++] = r;
    }
  }
  const int *const dst = global ? dst_own : domain->neighbors;
  int *const stage = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dst * np, 1) * sizeof(int), "MFTLExchangeStart staging");
  for (int i = 0; i < n_dst; ++i)
    for (int p = 0; p < np; ++p) stage[(size_t) i * np + p] = x[p].send_bufs[dst[i]].count;

  mftl_route rt;
  rt.global = global;
  MFTLRouteCountsBegin(domain, &rt, np, n_dst, dst, stage, MFTL_FALSE);
  MFTLRouteCountsEnd(domain, &rt);

  /* One allocation holding [send counts | recv counts | send displs | recv displs],
   * each np * peers ints laid out payload-major. */
  const int nt = rt.n;
  ctx->n_targets = nt;
  const size_t stride = (size_t) (nt > 0 ? nt : 1);
  const size_t block = (size_t) np * stride;
  ctx->counts = (int *) MFTLCallocOrDie(4 * block, sizeof(int), "MFTLExchangeStart counts");
  int *const s_cnt = ctx->counts;
  int *const r_cnt = ctx->counts + block;
  int *const s_dsp = ctx->counts + 2 * block;
  int *const r_dsp = ctx->counts + 3 * block;
  for (int p = 0; p < np; ++p)
    for (int i = 0; i < nt; ++i)
    {
      s_cnt[(size_t) p * stride + i] = rt.scnt[(size_t) i * np + p];
      r_cnt[(size_t) p * stride + i] = rt.rcnt[(size_t) i * np + p];
    }
  cs_alloc(6 * block * sizeof(int));  // the counts, and the staging both ways
  cs_free();
  CS_END(ts_counts, t_exchange_counts);

  /* ── 2. Displacements, buffers, and the send image in peer order ────────── */
  CS_BEGIN(ts_data);
  for (int p = 0; p < np; ++p)
  {
    int *const sc = s_cnt + (size_t) p * stride;
    int *const rc = r_cnt + (size_t) p * stride;
    int *const sd = s_dsp + (size_t) p * stride;
    int *const rd = r_dsp + (size_t) p * stride;

    long long s_tot = 0, r_tot = 0;
    for (int i = 0; i < nt; ++i)
    {
      sd[i] = checked_int(domain, s_tot, "a send displacement");
      rd[i] = checked_int(domain, r_tot, "a receive displacement");
      s_tot += checked_int(domain, sc[i], "a send count");
      r_tot += checked_int(domain, rc[i], "a receive count");
    }
    x[p].recv_count = checked_int(domain, r_tot, "the elements received");
    (void) checked_int(domain, s_tot, "the elements sent");

    /* Never hand MPI a null buffer: the standard permits it for zero counts, but
     * implementations differ on whether they validate the pointer first. */
    x[p].recv_buf = MFTLMallocOrDie((size_t) (r_tot > 0 ? r_tot : 1) * x[p].type_size, "MFTLExchangeStart recv");
    ctx->flat_send[p] = (char *) MFTLMallocOrDie((size_t) (s_tot > 0 ? s_tot : 1) * x[p].type_size, "MFTLExchangeStart send");
    cs_alloc(((size_t) (r_tot > 0 ? r_tot : 1) + (size_t) (s_tot > 0 ? s_tot : 1)) * x[p].type_size);

    for (int i = 0; i < nt; ++i)
    {
      if (sc[i] <= 0) continue;
      memcpy(ctx->flat_send[p] + (size_t) sd[i] * x[p].type_size, x[p].send_bufs[rt.peer[i]].buffer, (size_t) sc[i] * x[p].type_size);
    }
  }

  /* ── 3. Post every payload back to back; the caller may consume one while the
   * others are in flight. The posting order is fixed by the code. */
  for (int p = 0; p < np; ++p)
  {
    const size_t o = (size_t) p * stride;
    MFTLRoutePost(domain, global, nt, rt.peer, x[p].type, ctx->flat_send[p], s_cnt + o, s_dsp + o, x[p].recv_buf, r_cnt + o, r_dsp + o, MFTL_TAG_ROUTE_DATA + p,
                  &ctx->requests[p], &ctx->n_requests[p]);
  }

  MFTLRouteFree(&rt);
  free(stage);
  free(dst_own);
  CS_END(ts_data, t_exchange_data);
}

void MFTLExchangeWait(MFTL_ExchangeCtx *ctx, int payload)
{
  if (payload < 0 || payload >= ctx->n_payloads) return;

  if (ctx->n_requests[payload] > 0) MPI_Waitall(ctx->n_requests[payload], ctx->requests[payload], MPI_STATUSES_IGNORE);
  free(ctx->requests[payload]);
  ctx->requests[payload] = MFTL_NULL;
  ctx->n_requests[payload] = 0;

  /* The send image is dead the instant the transfer completed. */
  free(ctx->flat_send[payload]);
  ctx->flat_send[payload] = MFTL_NULL;
}

void MFTLExchangeFree(MFTL_ExchangeCtx *ctx)
{
  for (int p = 0; p < ctx->n_payloads; ++p) MFTLExchangeWait(ctx, p);
  free(ctx->counts);
  ctx->counts = MFTL_NULL;
  ctx->n_payloads = 0;
  ctx->n_targets = 0;
}

/* ── 2e. Fixed-width records, and their replies ──────────────────────────── */

/** @brief One destination of a record exchange, for sorting the global route's list. */
typedef struct
{
  int rank;  ///< The destination.
  int count;  ///< Its records.
  int displ;  ///< Where they start in the send buffer.
} record_block;

/** @brief A contiguous datatype of @p width bytes, committed; the caller frees it. */
static MPI_Datatype record_type(const MFTL_Domain *domain, size_t width)
{
  MPI_Datatype t;
  MPI_Type_contiguous(checked_int(domain, (long long) width, "the record width"), MPI_BYTE, &t);
  MPI_Type_commit(&t);
  return t;
}

/** @brief Aborts the job on a destination outside the communicator. */
static void check_rank(const MFTL_Domain *domain, int r)
{
  if (r >= 0 && r < domain->nproc) return;
  fprintf(stderr, "[MFTL rank %d] routed exchange: destination %d is outside the communicator\n", domain->rank, r);
  MPI_Abort(domain->comm_world, MFTL_ERROR);
  abort();
}

/**
 * @brief The body of both record exchanges: blocks for other ranks at their
 *        displacements in @p send, and this rank's own records, which never
 *        touch the wire and are copied once, straight into the result.
 *
 * @param domain     The domain.
 * @param width      Bytes per record.
 * @param send       The send buffer the blocks point into.
 * @param blk        `[n_blk]` blocks: other ranks, non-empty, distinct; reordered here.
 * @param n_blk      Blocks.
 * @param global     The agreed route (MFTLExchangeAgreeRoute(), by the caller).
 * @param own        This rank's own records: contiguous, or the base @p own_idx indexes.
 * @param own_idx    #MFTL_NULL, or `[n_self]` record indices into @p own.
 * @param n_self     Own records.
 * @param self_displ Where the own records were in the caller's send order (for the reply).
 * @param n_send     Records in the caller's send order, own included.
 * @param plan       Out, if not #MFTL_NULL.
 * @param n_recv     Out: records received, own included.
 * @return The received records; own first, then one block per source in receive order.
 */
static void *exchange_blocks(const MFTL_Domain *domain, size_t width, const void *send, record_block *blk, int n_blk, bool global, const unsigned char *own,
                             const int *own_idx, int n_self, int self_displ, int n_send, MFTL_ExchangePlan *plan, int *n_recv)
{
  /* One rank: nothing on the wire. */
  const bool alone = (domain->nproc == 1);

  /* The destinations as the route wants them: by slot on the graph, ascending
   * ranks on the global route. Displacements stay where the caller put them. */
  const int nn = alone ? 0 : domain->num_neighbors;
  const int cap = MFTL_MAX(MFTL_MAX(n_blk, nn), 1);
  int n_dst = 0;
  const int *dst = MFTL_NULL;
  int *const dlist = (int *) MFTLCallocOrDie((size_t) 3 * cap, sizeof(int), "routed exchange destinations");
  int *const dcnt = dlist + cap;
  int *const ddsp = dcnt + cap;
  if (!global)
  {
    n_dst = nn;
    dst = domain->neighbors;
    for (int b = 0; b < n_blk; ++b)
    {
      const int s = MFTLDomainNeighborIndex(domain, blk[b].rank);
      MFTL_ASSERT(s != MFTL_ERROR && dcnt[s] == 0 && "routed exchange: destinations must be distinct");
      dcnt[s] = blk[b].count;
      ddsp[s] = blk[b].displ;
    }
  }
  else
  {
    qsort(blk, (size_t) n_blk, sizeof(record_block), cmp_leading_rank);
    n_dst = n_blk;
    dst = dlist;
    for (int b = 0; b < n_blk; ++b)
    {
      MFTL_ASSERT((b == 0 || blk[b - 1].rank < blk[b].rank) && "routed exchange: destinations must be distinct");
      dlist[b] = blk[b].rank;
      dcnt[b] = blk[b].count;
      ddsp[b] = blk[b].displ;
    }
  }

  mftl_route rt;
  memset(&rt, 0, sizeof rt);
  rt.global = global;
  if (!alone)
  {
    MFTLRouteCountsBegin(domain, &rt, 1, n_dst, dst, dcnt, MFTL_FALSE);
    MFTLRouteCountsEnd(domain, &rt);
  }

  /* Per peer: sent | send displacement | received | receive displacement. Own
   * records go first in the receive buffer. */
  const int n = rt.n;
  int *const cnt = (int *) MFTLCallocOrDie((size_t) 4 * MFTL_MAX(n, 1), sizeof(int), "routed exchange counts");
  int *const sc = cnt, *const sd = cnt + n, *const rc = cnt + 2 * n, *const rd = cnt + 3 * n;
  long long total = n_self;
  for (int i = 0, b = 0; i < n; ++i)
  {
    if (!global)
    {
      sc[i] = dcnt[i];
      sd[i] = ddsp[i];
    }
    else
    {
      while (b < n_dst && dlist[b] < rt.peer[i]) ++b;
      if (b < n_dst && dlist[b] == rt.peer[i])
      {
        sc[i] = dcnt[b];
        sd[i] = ddsp[b];
      }
    }
    rc[i] = rt.rcnt[i];
    rd[i] = checked_int(domain, total, "the records received");
    total += rc[i];
  }
  const int got = checked_int(domain, total, "the records received");

  unsigned char *const out = (unsigned char *) MFTLMallocOrDie(MFTL_MAX((size_t) got * width, 1), "routed exchange received");
  if (own_idx == MFTL_NULL)
  {
    if (n_self > 0) memcpy(out, own, (size_t) n_self * width);
  }
  else
    for (int j = 0; j < n_self; ++j) memcpy(out + (size_t) j * width, own + (size_t) own_idx[j] * width, width);
  if (!alone)
  {
    MPI_Datatype t = record_type(domain, width);
    MFTLRoutePost(domain, global, n, rt.peer, t, send, sc, sd, out, rc, rd, MFTL_TAG_ROUTE_DATA, MFTL_NULL, MFTL_NULL);
    MPI_Type_free(&t);
  }

  if (plan != MFTL_NULL)
  {
    plan->global = global;
    plan->n_peers = n;
    plan->peer = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(int), "routed exchange plan");
    if (n > 0) memcpy(plan->peer, rt.peer, (size_t) n * sizeof(int));
    plan->count = cnt;
    plan->n_self = n_self;
    plan->self_displ = self_displ;
    plan->n_send = n_send;
    plan->n_recv = got;
  }
  else
    free(cnt);

  MFTLRouteFree(&rt);
  free(dlist);
  *n_recv = got;
  return out;
}

void *MFTLExchangeRecords(const MFTL_Domain *domain, size_t width, const void *send, int n_dest, const int *dest, const int *dest_count,
                          MFTL_ExchangePlan *plan, int *n_recv)
{
  MFTL_ASSERT(width >= 1 && "MFTLExchangeRecords: a record has at least one byte");
  const int me = domain->rank;

  /* Where each destination's block starts, this rank's own block, and whether
   * anything leaves the neighbourhood. */
  record_block *const blk = (record_block *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(record_block), "MFTLExchangeRecords blocks");
  long long n_send = 0;
  int n_self = 0, self_displ = 0, n_blk = 0;
  bool far = MFTL_FALSE;
  for (int i = 0; i < n_dest; ++i)
  {
    const int r = dest[i];
    const int c = checked_int(domain, dest_count[i], "a record count");
    check_rank(domain, r);
    const int displ = checked_int(domain, n_send, "the records sent");
    n_send += c;
    if (r == me)
    {
      n_self = c;
      self_displ = displ;
      continue;
    }
    if (c == 0) continue;
    if (MFTLDomainNeighborIndex(domain, r) == MFTL_ERROR) far = MFTL_TRUE;
    blk[n_blk].rank = r;
    blk[n_blk].count = c;
    blk[n_blk].displ = displ;
    ++n_blk;
  }
  const int n_all = checked_int(domain, n_send, "the records sent");

  const bool global = MFTLExchangeAgreeRoute(domain, MFTL_ROUTE_AUTO, far);
  void *const out = exchange_blocks(domain, width, send, blk, n_blk, global, (const unsigned char *) send + (size_t) self_displ * width, MFTL_NULL, n_self,
                                    self_displ, n_all, plan, n_recv);
  free(blk);
  return out;
}

void *MFTLExchangeRecordsTo(const MFTL_Domain *domain, size_t width, const void *rec, const int *dest, int n_rec, int *n_recv)
{
  MFTL_ASSERT(width >= 1 && "MFTLExchangeRecordsTo: a record has at least one byte");
  const int me = domain->rank, nproc = domain->nproc;

  /* One scan: a block per distinct destination (a rank-indexed table, local
   * memory only), the counts, and whether anything leaves the neighbourhood. */
  int *const block_of = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "MFTLExchangeRecordsTo table");
  for (int r = 0; r < nproc; ++r) block_of[r] = MFTL_ERROR;
  int n_blk = 0, cap = 8, n_self = 0;
  record_block *blk = (record_block *) MFTLMallocOrDie((size_t) cap * sizeof(record_block), "MFTLExchangeRecordsTo blocks");
  bool far = MFTL_FALSE;
  for (int i = 0; i < n_rec; ++i)
  {
    const int r = dest[i];
    check_rank(domain, r);
    if (r == me)
    {
      ++n_self;
      continue;
    }
    if (block_of[r] == MFTL_ERROR)
    {
      if (n_blk == cap)
      {
        cap *= 2;
        blk = (record_block *) MFTLReallocOrDie(blk, (size_t) cap * sizeof(record_block), "MFTLExchangeRecordsTo blocks");
      }
      block_of[r] = n_blk;
      blk[n_blk].rank = r;
      blk[n_blk].count = 0;
      ++n_blk;
      if (MFTLDomainNeighborIndex(domain, r) == MFTL_ERROR) far = MFTL_TRUE;
    }
    blk[block_of[r]].count++;
  }

  /* The route is agreed before packing, so every rank enters the vote after
   * the same cheap scan, not after its share of the copying. */
  const bool global = MFTLExchangeAgreeRoute(domain, MFTL_ROUTE_AUTO, far);

  /* Pack: a stable counting sort, every other rank's record copied once; this
   * rank's own stay where they are and are copied once, into the result. */
  int packed = 0;
  for (int b = 0; b < n_blk; ++b)
  {
    blk[b].displ = packed;
    packed += blk[b].count;
  }
  int *const cursor = (int *) MFTLMallocOrDie((size_t) (n_blk + 1) * sizeof(int), "MFTLExchangeRecordsTo cursors");
  for (int b = 0; b < n_blk; ++b) cursor[b] = blk[b].displ;
  int *const own_idx = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_self, 1) * sizeof(int), "MFTLExchangeRecordsTo own");
  unsigned char *const image = (unsigned char *) MFTLMallocOrDie(MFTL_MAX((size_t) packed * width, 1), "MFTLExchangeRecordsTo send");
  const unsigned char *const src = (const unsigned char *) rec;
  for (int i = 0, k = 0; i < n_rec; ++i)
  {
    if (dest[i] == me)
    {
      own_idx[k++] = i;
      continue;
    }
    memcpy(image + (size_t) cursor[block_of[dest[i]]]++ * width, src + (size_t) i * width, width);
  }
  free(cursor);
  free(block_of);

  void *const out = exchange_blocks(domain, width, image, blk, n_blk, global, src, own_idx, n_self, 0, n_rec, MFTL_NULL, n_recv);
  free(image);
  free(own_idx);
  free(blk);
  return out;
}

void *MFTLExchangeReply(const MFTL_Domain *domain, const MFTL_ExchangePlan *plan, size_t width, const void *reply)
{
  MFTL_ASSERT(width >= 1 && "MFTLExchangeReply: a record has at least one byte");
  unsigned char *const out = (unsigned char *) MFTLMallocOrDie(MFTL_MAX((size_t) plan->n_send * width, 1), "MFTLExchangeReply received");
  if (plan->n_self > 0) memcpy(out + (size_t) plan->self_displ * width, reply, (size_t) plan->n_self * width);
  if (domain->nproc == 1) return out;

  /* The forward exchange backwards: send what was received, where it was
   * received; receive what was sent, where it was sent from. */
  const int n = plan->n_peers;
  const int *const c = plan->count;
  MPI_Datatype t = record_type(domain, width);
  MFTLRoutePost(domain, plan->global, n, plan->peer, t, reply, c + 2 * n, c + 3 * n, out, c, c + n, MFTL_TAG_ROUTE_DATA + 1, MFTL_NULL, MFTL_NULL);
  MPI_Type_free(&t);
  return out;
}

void MFTLExchangePlanFree(MFTL_ExchangePlan *plan)
{
  if (plan == MFTL_NULL) return;
  free(plan->peer);
  free(plan->count);
  plan->peer = MFTL_NULL;
  plan->count = MFTL_NULL;
  plan->n_peers = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Neighbourhood allgather
 * ═══════════════════════════════════════════════════════════════════════════ */

double *MFTLNeighborAllgatherDoubles(const MFTL_Domain *domain, const double *send, int n_send, int *rcount, int *rdispl, int *n_recv)
{
  const int d = domain->num_neighbors;

  *n_recv = 0;
  if (domain->nproc < 2 || d < 1) return (double *) MFTLMallocOrDie(sizeof(double), "MFTLNeighborAllgatherDoubles (empty)");

  MPI_Neighbor_allgather(&n_send, 1, MPI_INT, rcount, 1, MPI_INT, domain->comm_graph);

  size_t total = 0;
  for (int i = 0; i < d; ++i)
  {
    rdispl[i] = (int) total;
    total += (size_t) rcount[i];
  }

  /* MPI counts are ints (reached only with millions of bodies on one rank). */
  MFTL_ASSERT(total <= (size_t) 0x7fffffff && "MFTLNeighborAllgatherDoubles: neighbour payload exceeds an MPI int count");

  double *const recv = (double *) MFTLMallocOrDie(MFTL_MAX(total, (size_t) 1) * sizeof(double), "MFTLNeighborAllgatherDoubles");
  MPI_Neighbor_allgatherv(send, n_send, MPI_DOUBLE, recv, rcount, rdispl, MPI_DOUBLE, domain->comm_graph);

  *n_recv = (int) total;
  return recv;
}

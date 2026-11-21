/**
 * @file test_exchange.c
 * @brief The library's routed personalised exchange: MFTLExchangeRecords() and
 *        its reply, MFTLExchangeStart() with typed payloads, and the NBX loop
 *        MFTLNbxComplete() under both.
 *
 * Every rank can compute what every OTHER rank sends it -- counts and contents
 * are functions of (pattern, source, destination, index) defined here -- so the
 * reference shares no code with the engine and every received byte is checked,
 * in the order the contract promises: this rank's own records first, then one
 * block per source, graph neighbours in `domain->neighbors` order or global
 * sources ascending. Each case counts the cross-rank records it verified and
 * the exchanges that took the global route, and asserts both where the grid
 * makes them possible, so a decomposition that kept everything local, or never
 * left the neighbourhood, fails instead of passing vacuously.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_utils.h"

static int rank, nproc;

/** @brief All ranks' verdict, so no rank returns from a test while another waits in a collective. */
static int agreed(int ok)
{
  int all = 0;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all;
}

static long sum_long(long v)
{
  long all = 0;
  MPI_Allreduce(&v, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return all;
}

/** @brief The default grid (periodic) or non-periodic slabs {P,1,1}, whose ranks two cells apart are not neighbours. */
static MFTL_Ensemble *make_ensemble(bool slabs)
{
  int cuts[3];
  test_grid_for(nproc, cuts);
  if (slabs)
  {
    cuts[0] = nproc;
    cuts[1] = cuts[2] = 1;
  }
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool periodic[3] = {!slabs, !slabs, !slabs};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Patterns: how many records source s sends destination d. Evaluated on rank d
 * for every s, so "is d a neighbour of s" is asked as "is s a neighbour of d"
 * (the graph is symmetric).
 * ═══════════════════════════════════════════════════════════════════════════ */

enum
{
  PAT_NEIGHBOURS,  ///< every neighbour and itself
  PAT_MIXED,  ///< about two thirds of all ranks, near and far
  PAT_GATHER,  ///< everybody to rank 0
  PAT_SCATTER,  ///< rank 0 to everybody (the initial distribution)
  PAT_EMPTY,  ///< nobody sends
  PAT_ONE_FAR,  ///< the neighbours, and rank 0 also to the farthest rank
  N_PATTERNS
};

static int mix(int pat, int s, int d) { return (s * 31 + d * 17 + pat * 7) % 4; }

/** @brief Records s sends d under @p pat, given whether s and d are graph neighbours. */
static int pattern_count(int pat, int s, int d, bool neighbours)
{
  switch (pat)
  {
    case PAT_NEIGHBOURS: return (s == d || neighbours) ? 1 + mix(pat, s, d) : 0;
    case PAT_MIXED: return ((s + d) % 3 != 0) ? mix(pat, s, d) : 0;  // zero counts included on purpose
    case PAT_GATHER: return (d == 0) ? 1 + mix(pat, s, d) : 0;
    case PAT_SCATTER: return (s == 0) ? 1 + mix(pat, s, d) : 0;
    case PAT_ONE_FAR:
      if (s == d || neighbours) return 1 + mix(pat, s, d);
      return (s == 0 && d == nproc / 2) ? 3 : 0;
    default: return 0;
  }
}

/** @brief Whether ranks a and b are graph neighbours, asked on rank b. */
static bool nb_of_me(const MFTL_Domain *dom, int a) { return a != dom->rank && MFTLDomainNeighborIndex(dom, a) != MFTL_ERROR; }

/** @brief Record component k of the j-th record s sends d: exact in binary64. */
static double record_value(int s, int d, int j, int k) { return s * 1048576.0 + d * 1024.0 + j * 8.0 + k; }

/** @brief The order sources arrive in: graph neighbours in MPI's order, or ascending. Excludes this rank. */
static int source_order(const MFTL_Domain *dom, bool global, int *src)
{
  int n = 0;
  if (!global)
  {
    for (int i = 0; i < dom->num_neighbors; ++i)
      if (dom->neighbors[i] != dom->rank) src[n++] = dom->neighbors[i];
  }
  else
  {
    for (int r = 0; r < dom->nproc; ++r)
      if (r != dom->rank) src[n++] = r;
  }
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Records
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One record exchange of @p pat with @p width doubles per record, and its
 *        reply; checks every byte and the order. Counts what it verified.
 */
static int run_records(const MFTL_Domain *dom, int pat, int width, long *cross, long *global_calls)
{
  const int me = dom->rank;
  /* Destinations in DESCENDING order: the API takes any order. */
  int *dest = (int *) malloc((size_t) nproc * sizeof(int));
  int *count = (int *) malloc((size_t) nproc * sizeof(int));
  int n_dest = 0, n_send = 0;
  for (int d = nproc - 1; d >= 0; --d)
  {
    const int c = pattern_count(pat, me, d, nb_of_me(dom, d));
    if (c == 0 && d % 2 == 0) continue;  // some zero-count entries stay in the list
    dest[n_dest] = d;
    count[n_dest++] = c;
    n_send += c;
  }
  double *send = (double *) malloc((size_t) (n_send > 0 ? n_send : 1) * width * sizeof(double));
  double *seq_of = (double *) malloc((size_t) (n_send > 0 ? n_send : 1) * sizeof(double));
  for (int i = 0, w = 0; i < n_dest; ++i)
    for (int j = 0; j < count[i]; ++j, ++w)
    {
      for (int k = 0; k < width; ++k) send[(size_t) w * width + k] = record_value(me, dest[i], j, k);
      seq_of[w] = dest[i] * 64.0 + j;
    }

  MFTL_ExchangePlan plan;
  int n_recv = -1;
  double *recv = (double *) MFTLExchangeRecords(dom, (size_t) width * sizeof(double), send, n_dest, dest, count, &plan, &n_recv);

  /* Expected: own records, then sources in the route's order. */
  int ok = (recv != MFTL_NULL) && (plan.n_send == n_send) && (plan.n_recv == n_recv);
  int *src = (int *) malloc((size_t) nproc * sizeof(int));
  const int n_src = source_order(dom, plan.global, src);
  int pos = 0;
  for (int t = -1; t < n_src && ok; ++t)
  {
    const int s = (t < 0) ? me : src[t];
    const int c = pattern_count(pat, s, me, nb_of_me(dom, s));
    for (int j = 0; j < c && ok; ++j, ++pos)
    {
      if (pos >= n_recv)
      {
        ok = 0;
        break;
      }
      for (int k = 0; k < width; ++k) ok &= (recv[(size_t) pos * width + k] == record_value(s, me, j, k));
      if (s != me) (*cross)++;
    }
  }
  ok &= (pos == n_recv);

  /* The reply: one double per received record, back to its sender's slot. */
  double *answer = (double *) malloc((size_t) (n_recv > 0 ? n_recv : 1) * sizeof(double));
  for (int i = 0; i < n_recv; ++i)
  {
    const long v = (long) recv[(size_t) i * width];  // record_value(s, d, j, 0): bit fields
    const int d = (int) ((v >> 10) & 1023), j = (int) ((v & 1023) >> 3);
    answer[i] = d * 64.0 + j;
  }
  double *back = (double *) MFTLExchangeReply(dom, &plan, sizeof(double), answer);
  for (int i = 0; i < n_send; ++i) ok &= (back[i] == seq_of[i]);
  if (plan.global) (*global_calls)++;

  /* The same records, each naming its destination, dealt round-robin over the
   * destinations so no two neighbours share one: MFTLExchangeRecordsTo() must
   * deliver exactly the stream above, on the same route. */
  {
    double *mixed = (double *) malloc((size_t) (n_send > 0 ? n_send : 1) * width * sizeof(double));
    int *to = (int *) malloc((size_t) (n_send > 0 ? n_send : 1) * sizeof(int));
    int *next = (int *) calloc((size_t) (n_dest > 0 ? n_dest : 1), sizeof(int));
    for (int w = 0; w < n_send;)
      for (int i = 0; i < n_dest; ++i)
        if (next[i] < count[i])
        {
          for (int k = 0; k < width; ++k) mixed[(size_t) w * width + k] = record_value(me, dest[i], next[i], k);
          to[w++] = dest[i];
          next[i]++;
        }
    int n_got = -1;
    double *got = (double *) MFTLExchangeRecordsTo(dom, (size_t) width * sizeof(double), mixed, to, n_send, &n_got);
    ok &= (got != MFTL_NULL) && (n_got == n_recv);
    for (int i = 0; ok && i < n_recv * width; ++i) ok &= (got[i] == recv[i]);
    free(got);
    free(next);
    free(to);
    free(mixed);
  }

  MFTLExchangePlanFree(&plan);
  MFTLExchangePlanFree(&plan);  // safe twice
  free(back);
  free(answer);
  free(src);
  free(recv);
  free(seq_of);
  free(send);
  free(count);
  free(dest);
  return ok;
}

static int test_records_arrive_in_the_promised_order_and_the_reply_finds_its_sender(void)
{
  int ok = 1;
  long cross = 0, global_calls = 0;
  for (int g = 0; g < 2; ++g)
  {
    MFTL_Ensemble *e = make_ensemble(g == 1);
    for (int pat = 0; pat < N_PATTERNS; ++pat)
      for (int width = 1; width <= 5; width += 4) ok &= run_records(&e->domain, pat, width, &cross, &global_calls);
    MFTLEnsembleDestroy(&e);
  }
  TEST_ASSERT(agreed(ok));

  /* Non-vacuity: something crossed ranks, and slabs of three or more have
   * ranks that are not neighbours, which the far patterns address. */
  const long all_cross = sum_long(cross);
  long global_any = 0;
  MPI_Allreduce(&global_calls, &global_any, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) printf("\n  cross-rank records verified %ld, global-route exchanges %ld  ", all_cross, global_any);
  TEST_ASSERT(nproc == 1 || all_cross > 0);
  TEST_ASSERT(nproc < 3 || global_any >= 3);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Typed payloads
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Elements of payload @p p that s sends d: the neighbour pattern, varied per payload. */
static int typed_count(int p, int s, int d, bool neighbours) { return neighbours ? (s * 5 + d * 3 + p) % 4 : 0; }

static int run_typed(const MFTL_Domain *dom, MFTL_Route route, bool far, long *cross, int *was_global)
{
  const int me = dom->rank;
  MFTL_SendBuffer *sb[3];
  for (int p = 0; p < 3; ++p) sb[p] = (MFTL_SendBuffer *) calloc((size_t) nproc, sizeof(MFTL_SendBuffer));
  for (int d = 0; d < nproc; ++d)
  {
    const bool nb = nb_of_me(dom, d);
    const int c0 = typed_count(0, me, d, nb), c1 = typed_count(1, me, d, nb), c2 = typed_count(2, me, d, nb);
    if (c0)
    {
      int *b = (int *) malloc((size_t) c0 * sizeof(int));
      for (int j = 0; j < c0; ++j) b[j] = me * 1000 + d * 10 + j;
      sb[0][d].buffer = b;
      sb[0][d].count = sb[0][d].capacity = c0;
    }
    if (c1)
    {
      double *b = (double *) malloc((size_t) c1 * sizeof(double));
      for (int j = 0; j < c1; ++j) b[j] = record_value(me, d, j, 7);
      sb[1][d].buffer = b;
      sb[1][d].count = sb[1][d].capacity = c1;
    }
    if (c2)
    {
      MFTL_PackedVertex *b = (MFTL_PackedVertex *) calloc((size_t) c2, sizeof(MFTL_PackedVertex));
      for (int j = 0; j < c2; ++j)
      {
        for (int k = 0; k < 3; ++k) b[j].pos[k] = record_value(me, d, j, k);
        b[j].global_id = me;
        b[j].owner_proc = d;
        b[j].flags = j;
        b[j].body_gid = -7;
      }
      sb[2][d].buffer = b;
      sb[2][d].count = sb[2][d].capacity = c2;
    }
  }

  MFTL_Exchange x[3];
  x[0].send_bufs = sb[0];
  x[0].type = MPI_INT;
  x[0].type_size = sizeof(int);
  x[1].send_bufs = sb[1];
  x[1].type = MPI_DOUBLE;
  x[1].type_size = sizeof(double);
  x[2].send_bufs = sb[2];
  x[2].type = MFTL_MPI_PackedVertex;  // a resized struct type: its extent, not a byte count, must step the buffer
  x[2].type_size = sizeof(MFTL_PackedVertex);

  MFTL_ExchangeCtx ctx;
  MFTLExchangeStart(dom, x, 3, route, far, &ctx);
  *was_global = ctx.global;

  int ok = 1;
  int *src = (int *) malloc((size_t) nproc * sizeof(int));
  const int n_src = source_order(dom, ctx.global, src);
  /* Payload 2 first, as redistribution waits on its headers first. */
  for (int q = 0; q < 3; ++q)
  {
    const int p = (q == 0) ? 2 : q - 1;
    MFTLExchangeWait(&ctx, p);
    int pos = 0;
    for (int t = 0; t < n_src; ++t)
    {
      const int s = src[t];
      const int c = typed_count(p, s, me, nb_of_me(dom, s));
      for (int j = 0; j < c; ++j, ++pos)
      {
        if (pos >= x[p].recv_count)
        {
          ok = 0;
          continue;
        }
        if (p == 0) ok &= (((int *) x[0].recv_buf)[pos] == s * 1000 + me * 10 + j);
        if (p == 1) ok &= (((double *) x[1].recv_buf)[pos] == record_value(s, me, j, 7));
        if (p == 2)
        {
          const MFTL_PackedVertex *v = &((MFTL_PackedVertex *) x[2].recv_buf)[pos];
          for (int k = 0; k < 3; ++k) ok &= (v->pos[k] == record_value(s, me, j, k));
          ok &= (v->global_id == s && v->owner_proc == me && v->flags == j && v->body_gid == -7);
        }
        (*cross)++;
      }
    }
    ok &= (pos == x[p].recv_count);
  }
  MFTLExchangeFree(&ctx);
  MFTLExchangeFree(&ctx);  // safe twice

  for (int p = 0; p < 3; ++p)
  {
    free(x[p].recv_buf);
    for (int d = 0; d < nproc; ++d) free(sb[p][d].buffer);
    free(sb[p]);
  }
  free(src);
  return ok;
}

static int test_typed_payloads_are_the_same_bits_on_both_routes(void)
{
  int ok = 1;
  long cross = 0;
  int global_seen = 0;
  for (int g = 0; g < 2; ++g)
  {
    MFTL_Ensemble *e = make_ensemble(g == 1);
    int s = -1;
    ok &= run_typed(&e->domain, MFTL_ROUTE_GRAPH, MFTL_FALSE, &cross, &s);
    ok &= (s == 0);
    /* One rank claiming a far destination sends EVERY rank onto the global route. */
    ok &= run_typed(&e->domain, MFTL_ROUTE_AUTO, rank == 0, &cross, &s);
    ok &= (s == (nproc > 1));
    global_seen |= s;
    ok &= run_typed(&e->domain, MFTL_ROUTE_AUTO, MFTL_FALSE, &cross, &s);
    ok &= (s == 0);
    MFTLEnsembleDestroy(&e);
  }
  TEST_ASSERT(agreed(ok));
  TEST_ASSERT(nproc == 1 || (sum_long(cross) > 0 && global_seen));
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Back to back: consecutive global-route rounds on one tag must not mix
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_back_to_back_global_exchanges_keep_their_messages_apart(void)
{
  MFTL_Ensemble *e = make_ensemble(MFTL_TRUE);
  int ok = 1;
  long cross = 0, global_calls = 0;
  /* No collective of the test's own between rounds: the engine's vote is the
   * only thing separating one round's discovery from the next. */
  for (int it = 0; it < 40; ++it)
  {
    const int pat = (it % 3 == 0) ? PAT_MIXED : (it % 3 == 1) ? PAT_SCATTER : PAT_GATHER;
    if (it % 4 == 3)
    {
      int s = 0;
      ok &= run_typed(&e->domain, MFTL_ROUTE_AUTO, rank == it % nproc, &cross, &s);
      global_calls += s;
    }
    else
      ok &= run_records(&e->domain, pat, 1 + it % 3, &cross, &global_calls);
  }
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(agreed(ok));
  long global_max = 0;
  MPI_Allreduce(&global_calls, &global_max, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(nproc < 3 || global_max >= 20);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NBX alone
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int *from;
  int n;
} inbox;

static void collect(MPI_Comm c, const MPI_Status *st, void *ctx)
{
  inbox *b = (inbox *) ctx;
  int v[2];
  MPI_Recv(v, 2, MPI_INT, st->MPI_SOURCE, st->MPI_TAG, c, MPI_STATUS_IGNORE);
  b->from[b->n++] = (v[0] == st->MPI_SOURCE && v[1] == rank) ? v[0] : -1;
}

/** @brief Whether rank s sends rank d a message in round @p it. */
static bool nbx_sends(int it, int s, int d) { return ((s * 7 + d * 3 + it) % 5) < 2; }

static int test_nbx_delivers_every_message_exactly_once(void)
{
  MPI_Comm c;
  MPI_Comm_dup(MPI_COMM_WORLD, &c);
  int ok = 1;
  long got_total = 0;
  for (int it = 0; it < 12; ++it)
  {
    int *payload = (int *) malloc(2 * (size_t) nproc * sizeof(int));
    MPI_Request *rq = (MPI_Request *) malloc((size_t) nproc * sizeof(MPI_Request));
    int n = 0;
    for (int d = 0; d < nproc; ++d)
    {
      if (!nbx_sends(it, rank, d)) continue;
      payload[2 * n] = rank;
      payload[2 * n + 1] = d;
      MPI_Issend(&payload[2 * n], 2, MPI_INT, d, 4242, c, &rq[n]);  // itself included
      ++n;
    }
    inbox b = {(int *) malloc((size_t) nproc * sizeof(int)), 0};
    MFTLNbxComplete(c, 4242, n, rq, collect, &b);
    MPI_Barrier(c);  // the documented separation between two rounds on one tag

    int expected = 0;
    char *seen = (char *) calloc((size_t) nproc, 1);
    for (int s = 0; s < nproc; ++s) expected += nbx_sends(it, s, rank);
    ok &= (b.n == expected);
    for (int i = 0; i < b.n; ++i)
    {
      const int s = b.from[i];
      ok &= (s >= 0 && s < nproc && nbx_sends(it, s, rank) && !seen[s]);
      if (s >= 0 && s < nproc) seen[s] = 1;
    }
    got_total += b.n;
    free(seen);
    free(b.from);
    free(rq);
    free(payload);
  }
  MPI_Comm_free(&c);
  TEST_ASSERT(agreed(ok));
  TEST_ASSERT(sum_long(got_total) > 0);
  return 0;
}

/** @brief Word @p i of rank @p r in agreement case @p c (see the test). */
static uint64_t agree_word(int c, int r, int i)
{
  const uint64_t base = 0x9E3779B97F4A7C15ull * (uint64_t) (i + 1);
  const int last = nproc - 1;
  switch (c)
  {
    case 0: return base;  // everyone agrees
    case 1: return base + (uint64_t) (r == last && i == 0);  // the first word, on one rank
    case 2: return base + (uint64_t) (r == last && i == 2);  // a later word, on one rank
    case 3: return (i == 1) ? ((r == last) ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull) : base;  // neighbours unsigned, extremes signed
    case 4: return (i == 1) ? ((r == last) ? ~0ull : 0ull) : base;  // the two "identities"
    case 5: return (uint64_t) r;  // every rank its own
    default: return base + (uint64_t) (r == last && i == 69);  // the heap path, last word
  }
}

/**
 * @brief MFTLAgreeWords(), the library's one agreement check, names the FIRST
 *        word on which any two ranks differ, identically on every rank, and
 *        MAX-reduces the optional status in the same call.
 *
 * Reference: every rank's words gathered to every rank (no complement trick),
 * scanned for the first column that is not constant. Cases 3 and 4 put the
 * difference where a signed comparison of `MPI_UINT64_T` (MPICH 4.0) would
 * misorder it and where 0 / `UINT64_MAX` would pass for identities; case 6 has
 * more words than the stack buffer. At np = 1 every case agrees, which is also
 * checked; a difference is only produced, and asserted, at np >= 2.
 */
static int test_agreement_names_the_first_differing_word_on_every_rank(void)
{
  enum
  {
    N_CASES = 7
  };
  static const int len[N_CASES] = {3, 3, 3, 3, 3, 1, 70};
  int ok = 1, differences = 0;
  for (int c = 0; c < N_CASES; ++c)
  {
    const int n = len[c];
    uint64_t *mine = (uint64_t *) malloc((size_t) n * sizeof(uint64_t));
    uint64_t *all = (uint64_t *) malloc((size_t) n * (size_t) nproc * sizeof(uint64_t));
    for (int i = 0; i < n; ++i) mine[i] = agree_word(c, rank, i);
    MPI_Allgather(mine, n, MPI_UINT64_T, all, n, MPI_UINT64_T, MPI_COMM_WORLD);
    int expect = -1;
    for (int i = 0; i < n && expect < 0; ++i)
      for (int r = 1; r < nproc; ++r)
        if (all[(size_t) r * n + i] != all[i]) expect = i;
    differences += (expect >= 0);

    /* With and without a status; the status is (rank * 7) % 5 on each rank. */
    int status = (rank * 7) % 5, expect_status = 0;
    for (int r = 0; r < nproc; ++r) expect_status = MFTL_MAX(expect_status, (r * 7) % 5);
    ok &= (MFTLAgreeWords(MPI_COMM_WORLD, mine, n, MFTL_NULL) == expect);
    ok &= (MFTLAgreeWords(MPI_COMM_WORLD, mine, n, &status) == expect);
    ok &= (status == expect_status);
    free(all);
    free(mine);
  }
  /* No words: only the status travels. */
  int status = (rank == nproc - 1) ? 3 : 1;  // the last rank's is the largest at every np
  ok &= (MFTLAgreeWords(MPI_COMM_WORLD, MFTL_NULL, 0, &status) == -1);
  ok &= (status == 3);
  ok &= (differences == ((nproc > 1) ? 6 : 0));  // cases 1-6 differ, case 0 never
  TEST_ASSERT(agreed(ok));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  RUN_TEST(test_records_arrive_in_the_promised_order_and_the_reply_finds_its_sender);
  RUN_TEST(test_typed_payloads_are_the_same_bits_on_both_routes);
  RUN_TEST(test_back_to_back_global_exchanges_keep_their_messages_apart);
  RUN_TEST(test_nbx_delivers_every_message_exactly_once);
  RUN_TEST(test_agreement_names_the_first_differing_word_on_every_rank);
  if (rank == 0) TEST_REPORT();
  const int fail = g_tests_failed;
  MPI_Finalize();
  return fail ? 1 : 0;
}

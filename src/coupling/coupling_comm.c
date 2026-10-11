/**
 * @file coupling_comm.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief EVERY MPI call of the coupling module.
 *
 * Four shapes, all on the coupler's PRIVATE duplicate of the ensemble's
 * communicator (never the solver's -- this file never probes a message it did
 * not cause):
 *
 *   agreements    one small Allreduce per collective decision (status, hashes,
 *                 registration bounds); O(1) payload, independent of P.
 *   NBX           sparse exchange to receivers that do not know they will
 *                 receive (directory publication, support requests): Hoefler,
 *                 Siebert & Lumsdaine's protocol -- synchronous sends, probe and
 *                 receive, a non-blocking barrier entered once this rank's sends
 *                 have been MATCHED, and receiving continues until it completes.
 *                 Messages of one phase therefore never leak into the next.
 *   known peers   replies: both sides know the partner set, the receiver does
 *                 not know the size, so it probes the named source.
 *   schedules     fetch and spread: sizes known on both sides, split into
 *                 Begin (post) and End (wait).
 *
 * Every message larger than the chunk limit (1 GiB, or the test hook's value)
 * travels as several messages on the same (source, tag): MPI's non-overtaking
 * rule keeps them in order, and no count ever exceeds an int.
 * Fatal allocation/budget failure is the exception to communicator isolation:
 * it aborts MPI_COMM_WORLD to avoid stranding peers, including during registration.
 */

#include "coupling_internal.h"



#include <mpi.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"

_Noreturn void MFTLCplMemoryAbort(const char *what, size_t bytes, size_t current, size_t limit)
{
  fprintf(stderr, "MFTL coupling: allocation/budget failure in %s: requested %zu bytes, live %zu, limit %zu (0=unlimited)\n",
          what, bytes, current, limit);
  fflush(stderr);
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  exit(EXIT_FAILURE);
}

static size_t g_chunk = (size_t) 1 << 30;

void MFTLCouplerSetChunkLimit(size_t bytes) { g_chunk = (bytes == 0) ? ((size_t) 1 << 30) : (bytes < 16 ? 16 : bytes); }

size_t MFTLCplChunkLimit(void) { return g_chunk; }

int MFTLCplCommDup(MPI_Comm base, MPI_Comm *out) { return (MPI_Comm_dup(base, out) == MPI_SUCCESS) ? MFTL_SUCCESS : MFTL_ERROR; }

void MFTLCplCommFree(MPI_Comm *c)
{
  if (*c != MPI_COMM_NULL) MPI_Comm_free(c);
}

void MFTLCplCommRankSize(MPI_Comm c, int *rank, int *size)
{
  MPI_Comm_rank(c, rank);
  MPI_Comm_size(c, size);
}

bool MFTLCplCommSameGroup(MPI_Comm a, MPI_Comm b)
{
  if (a == MPI_COMM_NULL || b == MPI_COMM_NULL) return false;
  int res = MPI_UNEQUAL;
  MPI_Comm_compare(a, b, &res);
  return res != MPI_UNEQUAL;
}

bool MFTLCplTranslateRanks(MPI_Comm from, MPI_Comm to, int64_t n, const int *ranks, int *translated)
{
  int size;
  MPI_Comm_size(from, &size);
  for (int64_t i = 0; i < n; ++i)
    if (ranks[i] < 0 || ranks[i] >= size) return false;
  MPI_Group src, dst;
  MPI_Comm_group(from, &src);
  MPI_Comm_group(to, &dst);
  for (int64_t off = 0; off < n;)
  {
    const int count = (n - off > INT_MAX) ? INT_MAX : (int) (n - off);
    MPI_Group_translate_ranks(src, count, ranks + off, dst, translated + off);
    off += count;
  }
  MPI_Group_free(&src);
  MPI_Group_free(&dst);
  for (int64_t i = 0; i < n; ++i)
    if (translated[i] == MPI_UNDEFINED) return false;
  return true;
}

int MFTLCplAgreeMax(MPI_Comm c, int v)
{
  int g = v;
  MPI_Allreduce(&v, &g, 1, MPI_INT, MPI_MAX, c);
  return g;
}

int MFTLCplAgreeTransfer(MPI_Comm c, int status, uint64_t token)
{
  uint64_t v[3] = {(uint64_t) status, token, ~token}, g[3];
  MPI_Allreduce(v, g, 3, MPI_UINT64_T, MPI_MAX, c);
  return g[1] == ~g[2] ? (int) g[0] : MFTL_COUPLING_INVALID;
}

int MFTLCplAgreeValidated(MPI_Comm c, int status, uint64_t signature)
{
  uint64_t v[3] = {(uint64_t) status, signature, ~signature}, g[3];
  MPI_Allreduce(v, g, 3, MPI_UINT64_T, MPI_MAX, c);
  if (g[0] != MFTL_COUPLING_OK) return (int) g[0];
  return g[1] == ~g[2] ? MFTL_COUPLING_OK : MFTL_COUPLING_INVALID;
}

void MFTLCplAgreeMaxN(MPI_Comm c, int *v, int n)
{
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_INT, MPI_MAX, c);
}

bool MFTLCplAgreeHash(MPI_Comm c, uint64_t h)
{
  uint64_t v[2] = {h, ~h}, g[2];
  MPI_Allreduce(v, g, 2, MPI_UINT64_T, MPI_MAX, c);
  /* max(h) == min(h)  <=>  max(h) == ~max(~h) */
  return g[0] == ~g[1];
}

void MFTLCplAllreduceMax(MPI_Comm c, double *v, int n)
{
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, c);
}

void MFTLCplAllreduceSum(MPI_Comm c, double *v, int n)
{
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, c);
}

void MFTLCplAllreduceSumLong(MPI_Comm c, long *v, int n)
{
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_LONG, MPI_SUM, c);
}

void MFTLCplAllreduceMaxLong(MPI_Comm c, long *v, int n)
{
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_LONG, MPI_MAX, c);
}

void MFTLCplMaxloc(MPI_Comm c, double v, double *vmax, int *rank)
{
  struct
  {
    double v;
    int r;
  } in, out;
  in.v = v;
  MPI_Comm_rank(c, &in.r);
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, c);
  *vmax = out.v;
  *rank = out.r;
}

void MFTLCplBcastInt(MPI_Comm c, int *v, int n, int root) { MPI_Bcast(v, n, MPI_INT, root, c); }

double MFTLCplWtime(void) { return MPI_Wtime(); }

void MFTLCplCartCoords(const MFTL_Domain *d, int rank, int coords[3]) { MPI_Cart_coords(d->comm_cart, rank, 3, coords); }

/* ═══════════════════════════════════════════════════════════════════════════
 * Framed, chunked messages
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Growable request array. */
typedef struct
{
  MPI_Request *r;
  int n, cap;
} reqs;

static MPI_Request *req_push(reqs *q)
{
  if (q->n == q->cap)
  {
    q->cap = MFTLGrowCapacity(q->cap, q->n + 1);
    q->r = (MPI_Request *) MFTLCplRealloc(q->r, (size_t) q->cap * sizeof(MPI_Request), "coupling requests");
  }
  return &q->r[q->n++];
}

/** @brief A payload with its 8-byte length header, the unit NBX and replies send. */
static unsigned char *frame(const unsigned char *data, size_t bytes, size_t *framed)
{
  unsigned char *f = (unsigned char *) MFTLCplAlloc(bytes + 8, "coupling frame");
  const uint64_t n = (uint64_t) bytes;
  memcpy(f, &n, 8);
  if (bytes > 0) memcpy(f + 8, data, bytes);
  *framed = bytes + 8;
  return f;
}

/** @brief Posts a framed buffer as chunks; synchronous sends for NBX. */
static void post_chunks(MPI_Comm c, reqs *q, int tag, int peer, const unsigned char *buf, size_t bytes, bool synchronous, long *messages)
{
  const size_t lim = g_chunk;
  size_t off = 0;
  do
  {
    const size_t len = (bytes - off < lim) ? bytes - off : lim;
    MPI_Request *r = req_push(q);
    if (synchronous)
      MPI_Issend(buf + off, (int) len, MPI_BYTE, peer, tag, c, r);
    else
      MPI_Isend(buf + off, (int) len, MPI_BYTE, peer, tag, c, r);
    if (messages) ++*messages;
    off += len;
  } while (off < bytes);
}

/**
 * @brief Receives one framed message whose first chunk is already matched by
 *        @p st, then the rest of its chunks from the same source.
 */
static void recv_framed(MPI_Comm c, int tag, MPI_Status *st, cpl_msg *m)
{
  int count = 0;
  MPI_Get_count(st, MPI_BYTE, &count);
  const int src = st->MPI_SOURCE;
  unsigned char *first = (unsigned char *) MFTLCplAlloc((size_t) MFTL_MAX(count, 8), "coupling receive");
  MPI_Recv(first, count, MPI_BYTE, src, tag, c, MPI_STATUS_IGNORE);
  uint64_t total = 0;
  memcpy(&total, first, 8);
  m->peer = src;
  m->bytes = (size_t) total;
  m->data = (unsigned char *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1), "coupling message");
  size_t have = (size_t) count - 8;
  if (have > 0) memcpy(m->data, first + 8, have);
  MFTLCplFree(first);
  while (have < (size_t) total)
  {
    MPI_Status s2;
    MPI_Probe(src, tag, c, &s2);
    int n2 = 0;
    MPI_Get_count(&s2, MPI_BYTE, &n2);
    MPI_Recv(m->data + have, n2, MPI_BYTE, src, tag, c, MPI_STATUS_IGNORE);
    have += (size_t) n2;
  }
}

static int cmp_msg(const void *a, const void *b)
{
  const int x = ((const cpl_msg *) a)->peer, y = ((const cpl_msg *) b)->peer;
  return (x > y) - (x < y);
}

int MFTLCplNbx(MPI_Comm c, int tag, int n_out, const int *dest, unsigned char *const *sbuf, const size_t *sbytes, cpl_msg **in, int *n_in, long *bytes,
            long *messages)
{
  reqs q = {MFTL_NULL, 0, 0};
  unsigned char **framed = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_out, 1) * sizeof(unsigned char *), "nbx frames");
  for (int i = 0; i < n_out; ++i)
  {
    size_t fb = 0;
    framed[i] = frame(sbuf[i], sbytes[i], &fb);
    post_chunks(c, &q, tag, dest[i], framed[i], fb, true, messages);
    if (bytes) *bytes += (long) fb;
  }

  int cap = 8, n = 0;
  cpl_msg *got = (cpl_msg *) MFTLCplAlloc((size_t) cap * sizeof(cpl_msg), "nbx receive list");
  MPI_Request barrier = MPI_REQUEST_NULL;
  bool barrier_active = false;
  for (;;)
  {
    int flag = 0;
    MPI_Status st;
    MPI_Iprobe(MPI_ANY_SOURCE, tag, c, &flag, &st);
    if (flag)
    {
      if (n == cap)
      {
        cap *= 2;
        got = (cpl_msg *) MFTLCplRealloc(got, (size_t) cap * sizeof(cpl_msg), "nbx receive list");
      }
      recv_framed(c, tag, &st, &got[n]);
      if (bytes) *bytes += (long) got[n].bytes + 8;
      ++n;
      continue;
    }
    if (!barrier_active)
    {
      int done = 0;
      MPI_Testall(q.n, q.r, &done, MPI_STATUSES_IGNORE);
      if (done)
      {
        MPI_Ibarrier(c, &barrier);
        barrier_active = true;
      }
    }
    else
    {
      int done = 0;
      MPI_Test(&barrier, &done, MPI_STATUS_IGNORE);
      if (done) break;
    }
  }
  for (int i = 0; i < n_out; ++i) MFTLCplFree(framed[i]);
  MFTLCplFree(framed);
  MFTLCplFree(q.r);
  qsort(got, (size_t) n, sizeof(cpl_msg), cmp_msg);
  *in = got;
  *n_in = n;
  return MFTL_SUCCESS;
}

int MFTLCplExchangeKnown(MPI_Comm c, int tag, int n_out, const int *dest, unsigned char *const *sbuf, const size_t *sbytes, int n_src, const int *src,
                       cpl_msg *in, long *bytes, long *messages)
{
  reqs q = {MFTL_NULL, 0, 0};
  unsigned char **framed = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_out, 1) * sizeof(unsigned char *), "reply frames");
  for (int i = 0; i < n_out; ++i)
  {
    size_t fb = 0;
    framed[i] = frame(sbuf[i], sbytes[i], &fb);
    post_chunks(c, &q, tag, dest[i], framed[i], fb, false, messages);
    if (bytes) *bytes += (long) fb;
  }
  for (int i = 0; i < n_src; ++i)
  {
    MPI_Status st;
    MPI_Probe(src[i], tag, c, &st);
    recv_framed(c, tag, &st, &in[i]);
    if (bytes) *bytes += (long) in[i].bytes + 8;
  }
  MPI_Waitall(q.n, q.r, MPI_STATUSES_IGNORE);
  for (int i = 0; i < n_out; ++i) MFTLCplFree(framed[i]);
  MFTLCplFree(framed);
  MFTLCplFree(q.r);
  return MFTL_SUCCESS;
}

void MFTLCplMsgsFree(cpl_msg *m, int n)
{
  if (m == MFTL_NULL) return;
  for (int i = 0; i < n; ++i) MFTLCplFree(m[i].data);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Fixed schedules
 * ═══════════════════════════════════════════════════════════════════════════ */

static MPI_Request *sched_push(cpl_sched *s)
{
  if (s->n_req == s->cap_req)
  {
    s->cap_req = MFTLGrowCapacity(s->cap_req, s->n_req + 1);
    s->req = MFTLCplRealloc(s->req, (size_t) s->cap_req * sizeof(MPI_Request), "coupling schedule");
  }
  return &((MPI_Request *) s->req)[s->n_req++];
}

void MFTLCplSchedRecv(MPI_Comm c, cpl_sched *s, int tag, int peer, void *buf, size_t bytes)
{
  const size_t lim = g_chunk;
  for (size_t off = 0; off < bytes; off += lim)
  {
    const size_t len = (bytes - off < lim) ? bytes - off : lim;
    MPI_Irecv((unsigned char *) buf + off, (int) len, MPI_BYTE, peer, tag, c, sched_push(s));
  }
}

void MFTLCplSchedSend(MPI_Comm c, cpl_sched *s, int tag, int peer, const void *buf, size_t bytes, long *messages)
{
  const size_t lim = g_chunk;
  for (size_t off = 0; off < bytes; off += lim)
  {
    const size_t len = (bytes - off < lim) ? bytes - off : lim;
    MPI_Isend((const unsigned char *) buf + off, (int) len, MPI_BYTE, peer, tag, c, sched_push(s));
    if (messages) ++*messages;
  }
}

void MFTLCplSchedWait(cpl_sched *s)
{
  if (s->n_req > 0) MPI_Waitall(s->n_req, (MPI_Request *) s->req, MPI_STATUSES_IGNORE);
  s->n_req = 0;
}

void MFTLCplSchedFree(cpl_sched *s)
{
  MFTLCplFree(s->req);
  memset(s, 0, sizeof *s);
}

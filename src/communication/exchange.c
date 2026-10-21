/**
 * @file exchange.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief The MPI datatypes, the generic variable-length exchange, and the
 *        neighbourhood allgather.
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
 * 2. Generic variable-length exchange
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLExchangeStart(const MFTL_Domain *domain, MFTL_Exchange *x, int n_payloads, bool use_global_path, MFTL_ExchangeCtx *ctx)
{
  MFTL_ASSERT(n_payloads >= 1 && n_payloads <= MFTL_MAX_EXCHANGE_PAYLOADS && "MFTLExchangeStart: payload count out of range");
  MFTL_ASSERT((use_global_path || domain->comm_graph != MPI_COMM_NULL) && "MFTLExchangeStart: neighbour path without a graph communicator");

  const int np = n_payloads;
  const int nt = use_global_path ? domain->nproc : domain->num_neighbors;
  const MPI_Comm comm = use_global_path ? domain->comm_cart : domain->comm_graph;

  ctx->n_payloads = np;
  ctx->n_targets = nt;
  for (int p = 0; p < np; ++p)
  {
    ctx->requests[p] = MPI_REQUEST_NULL;
    ctx->flat_send[p] = MFTL_NULL;
  }

  /* One allocation holding [send counts | recv counts | send displs | recv displs],
   * each np*nt ints laid out payload-major. */
  const size_t stride = (size_t) (nt > 0 ? nt : 1);
  const size_t block = (size_t) np * stride;
  ctx->counts = (int *) MFTLCallocOrDie(4 * block, sizeof(int), "MFTLExchangeStart counts");
  int *const s_cnt = ctx->counts;
  int *const r_cnt = ctx->counts + block;
  int *const s_dsp = ctx->counts + 2 * block;
  int *const r_dsp = ctx->counts + 3 * block;

  /* ── 1. ONE count exchange for all payloads (latency, not volume, is its
   * cost), staged [target][payload] and de-interleaved. */
  CS_BEGIN(ts_counts);
  int *stage = (int *) MFTLMallocOrDie(2 * block * sizeof(int), "MFTLExchangeStart staging");
  int *stage_recv = stage + block;

  for (int i = 0; i < nt; ++i)
  {
    const int rank = use_global_path ? i : domain->neighbors[i];
    for (int p = 0; p < np; ++p) stage[(size_t) i * np + p] = x[p].send_bufs[rank].count;
  }

  if (use_global_path)
    MPI_Alltoall(stage, np, MPI_INT, stage_recv, np, MPI_INT, comm);
  else
    MPI_Neighbor_alltoall(stage, np, MPI_INT, stage_recv, np, MPI_INT, comm);

  for (int p = 0; p < np; ++p)
  {
    for (int i = 0; i < nt; ++i)
    {
      s_cnt[(size_t) p * stride + i] = stage[(size_t) i * np + p];
      r_cnt[(size_t) p * stride + i] = stage_recv[(size_t) i * np + p];
    }
  }
  free(stage);
  cs_alloc(4 * block * sizeof(int) + 2 * block * sizeof(int));
  cs_free();
  CS_END(ts_counts, t_exchange_counts);

  /* ── 2. Displacements, buffers, and the rank-ordered send image ─────────── */
  CS_BEGIN(ts_data);
  for (int p = 0; p < np; ++p)
  {
    int *const sc = s_cnt + (size_t) p * stride;
    int *const rc = r_cnt + (size_t) p * stride;
    int *const sd = s_dsp + (size_t) p * stride;
    int *const rd = r_dsp + (size_t) p * stride;

    int s_tot = 0, r_tot = 0;
    for (int i = 0; i < nt; ++i)
    {
      if (sc[i] < 0 || rc[i] < 0 || sc[i] > INT_MAX - s_tot || rc[i] > INT_MAX - r_tot)
      {
        fprintf(stderr, "MFTL exchange count exceeds MPI int displacement range\n");
        MPI_Abort(domain->comm_world, MFTL_ERROR);
        abort();
      }
      sd[i] = s_tot;
      s_tot += sc[i];
      rd[i] = r_tot;
      r_tot += rc[i];
    }

    x[p].recv_count = r_tot;

    /* Never hand MPI a null buffer: the standard permits it for zero counts, but
     * implementations differ on whether they validate the pointer first. */
    x[p].recv_buf = MFTLMallocOrDie((size_t) (r_tot > 0 ? r_tot : 1) * x[p].type_size, "MFTLExchangeStart recv");
    ctx->flat_send[p] = (char *) MFTLMallocOrDie((size_t) (s_tot > 0 ? s_tot : 1) * x[p].type_size, "MFTLExchangeStart send");
    cs_alloc(((size_t) (r_tot > 0 ? r_tot : 1) + (size_t) (s_tot > 0 ? s_tot : 1)) * x[p].type_size);

    for (int i = 0; i < nt; ++i)
    {
      if (sc[i] <= 0) continue;
      const int rank = use_global_path ? i : domain->neighbors[i];
      memcpy(ctx->flat_send[p] + (size_t) sd[i] * x[p].type_size, x[p].send_bufs[rank].buffer, (size_t) sc[i] * x[p].type_size);
    }
  }

  /* ── 3. Post every payload back to back; the caller may consume one while the
   * others are in flight. The posting order is fixed by the code. */
  for (int p = 0; p < np; ++p)
  {
    int *const sc = s_cnt + (size_t) p * stride;
    int *const rc = r_cnt + (size_t) p * stride;
    int *const sd = s_dsp + (size_t) p * stride;
    int *const rd = r_dsp + (size_t) p * stride;

    if (use_global_path)
      MPI_Ialltoallv(ctx->flat_send[p], sc, sd, x[p].type, x[p].recv_buf, rc, rd, x[p].type, comm, &ctx->requests[p]);
    else
      MPI_Ineighbor_alltoallv(ctx->flat_send[p], sc, sd, x[p].type, x[p].recv_buf, rc, rd, x[p].type, comm, &ctx->requests[p]);
  }

  CS_END(ts_data, t_exchange_data);
}

void MFTLExchangeWait(MFTL_ExchangeCtx *ctx, int payload)
{
  if (payload < 0 || payload >= ctx->n_payloads) return;

  if (ctx->requests[payload] != MPI_REQUEST_NULL) MPI_Wait(&ctx->requests[payload], MPI_STATUS_IGNORE);

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

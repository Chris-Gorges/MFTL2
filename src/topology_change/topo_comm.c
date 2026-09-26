/**
 * @file topo_comm.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief EVERY MPI call of the topology-change module.
 *
 * Seven shapes, and how each scales:
 *
 *   topo_comm_allreduce_*   a handful of values PER CLUSTER or per listed body.
 *                           Clusters exist only where events were detected, so
 *                           this is bounded by what happened, not by the
 *                           ensemble.
 *   MFTLTopoCommAllgather     the event list, and one record per (component,
 *                           holder). Both are paid only on a step with events,
 *                           and both are proportional to what changed. It is a
 *                           comm_world collective ON PURPOSE: every rank must
 *                           agree on the event list, including a rank two boxes
 *                           away that will reconstruct part of a body it does
 *                           not yet hold.
 *   MFTLTopoCommExchange /    26-neighbour personalised exchanges: the lattice
 *   MFTLTopoCommReply         overlap, the component labels, the ghost ids. The
 *                           global communicator only when some destination is
 *                           provably outside the neighbourhood, which the caller
 *                           detects and this file agrees on.
 */

#include "topo_internal.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_memory.h"

void MFTLTopoCommCoords(const MFTL_Domain *dom, int coords[3])
{
  /* comm_cart is created with reorder = 0, so its ranks are comm_world's. */
  MPI_Cart_coords(dom->comm_cart, dom->rank, 3, coords);
}

double MFTLTopoCommTime(void) { return MPI_Wtime(); }

void MFTLTopoCommAllreduceDouble(const MFTL_Domain *dom, double *v, int n, int op)
{
  if (n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, op == 0 ? MPI_MIN : (op == 1 ? MPI_MAX : MPI_SUM), dom->comm_world);
}

void MFTLTopoCommAllreduceLongLong(const MFTL_Domain *dom, long long *v, int n, int op)
{
  if (n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_LONG_LONG, op == 0 ? MPI_MIN : (op == 1 ? MPI_MAX : MPI_SUM), dom->comm_world);
}

bool MFTLTopoCommAny(const MFTL_Domain *dom, bool local)
{
  int v = local ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_LOR, dom->comm_world);
  return v != 0;
}

double *MFTLTopoCommAllgather(const MFTL_Domain *dom, const double *mine, int n_mine, int stride, int *n_out)
{
  const int np = dom->nproc;
  int *const counts = (int *) MFTLMallocOrDie(2 * (size_t) np * sizeof(int), "topology allgather counts");
  int *const displs = counts + np;

  const int my_count = n_mine * stride;
  MPI_Allgather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, dom->comm_world);

  int total = 0;
  for (int r = 0; r < np; ++r)
  {
    displs[r] = total;
    total += counts[r];
  }

  double *const all = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(total, 1) * sizeof(double), "topology allgather records");
  MPI_Allgatherv(mine, my_count, MPI_DOUBLE, all, counts, displs, MPI_DOUBLE, dom->comm_world);

  free(counts);
  *n_out = total / stride;
  return all;
}

/**
 * @brief The shared body of the exchange and the reply: an alltoallv of
 *        doubles, on the graph or the global communicator.
 *
 * On the graph path the per-rank counts are gathered into neighbour order and
 * the result is scattered back into rank order, so callers see ranks either
 * way. A rank reached through two stencil offsets (a thin periodic grid) is ONE
 * entry of `neighbors` per offset; only the first occurrence carries data, the
 * rest carry zero, so nothing is delivered twice. The graph's self-loop, if it
 * has one, carries zero as well: callers merge their own contributions locally.
 */
static double *alltoallv_doubles(const MFTL_Domain *dom, const double *send, const int *send_counts_by_rank, int stride, bool global_path,
                                 int *recv_counts_by_rank, int *n_recv)
{
  const int np = dom->nproc;

  /* Agree on the route: one rank's far destination forces everybody global. */
  int g = global_path ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_INT, MPI_LOR, dom->comm_world);

  /* Send displacements in RANK order, which is how callers packed. */
  int *const sdispl_rank = (int *) MFTLMallocOrDie((size_t) np * sizeof(int), "topology send displacements");
  int total_send = 0;
  for (int r = 0; r < np; ++r)
  {
    sdispl_rank[r] = total_send;
    total_send += send_counts_by_rank[r] * stride;
  }

  double *recv = MFTL_NULL;
  int total_recv = 0;

  if (g)
  {
    int *const sc = (int *) MFTLMallocOrDie(4 * (size_t) np * sizeof(int), "topology global counts");
    int *const rc = sc + np, *const sd = sc + 2 * np, *const rd = sc + 3 * np;
    for (int r = 0; r < np; ++r)
    {
      sc[r] = (r == dom->rank) ? 0 : send_counts_by_rank[r] * stride;
      sd[r] = sdispl_rank[r];
    }
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, dom->comm_world);
    for (int r = 0; r < np; ++r)
    {
      rd[r] = total_recv;
      total_recv += rc[r];
    }
    recv = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(total_recv, 1) * sizeof(double), "topology global receive");
    MPI_Alltoallv(send, sc, sd, MPI_DOUBLE, recv, rc, rd, MPI_DOUBLE, dom->comm_world);
    if (recv_counts_by_rank)
      for (int r = 0; r < np; ++r) recv_counts_by_rank[r] = rc[r] / stride;
    free(sc);
  }
  else
  {
    const int nn = dom->num_neighbors;
    int *const sc = (int *) MFTLCallocOrDie(4 * (size_t) MFTL_MAX(nn, 1), sizeof(int), "topology graph counts");
    int *const rc = sc + nn, *const sd = sc + 2 * nn, *const rd = sc + 3 * nn;

    /* First occurrence of each rank in the neighbour list carries its data. */
    for (int i = 0; i < nn; ++i)
    {
      const int r = dom->neighbors[i];
      bool first = MFTL_TRUE;
      for (int j = 0; j < i; ++j)
        if (dom->neighbors[j] == r) first = MFTL_FALSE;
      sc[i] = (first && r != dom->rank) ? send_counts_by_rank[r] * stride : 0;
      sd[i] = sdispl_rank[r];
    }
    MPI_Neighbor_alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, dom->comm_graph);
    for (int i = 0; i < nn; ++i)
    {
      rd[i] = total_recv;
      total_recv += rc[i];
    }

    /* Receive in neighbour order, then regroup into rank order. */
    double *const tmp = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(total_recv, 1) * sizeof(double), "topology graph receive");
    MPI_Neighbor_alltoallv(send, sc, sd, MPI_DOUBLE, tmp, rc, rd, MPI_DOUBLE, dom->comm_graph);

    int *const by_rank = (int *) MFTLCallocOrDie((size_t) np, sizeof(int), "topology graph per-rank counts");
    for (int i = 0; i < nn; ++i) by_rank[dom->neighbors[i]] += rc[i];
    int *const cursor = (int *) MFTLMallocOrDie((size_t) np * sizeof(int), "topology graph cursors");
    int acc = 0;
    for (int r = 0; r < np; ++r)
    {
      cursor[r] = acc;
      acc += by_rank[r];
    }
    recv = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(total_recv, 1) * sizeof(double), "topology graph regrouped");
    for (int i = 0; i < nn; ++i)
    {
      if (rc[i] == 0) continue;
      memcpy(recv + cursor[dom->neighbors[i]], tmp + rd[i], (size_t) rc[i] * sizeof(double));
      cursor[dom->neighbors[i]] += rc[i];
    }
    if (recv_counts_by_rank)
      for (int r = 0; r < np; ++r) recv_counts_by_rank[r] = by_rank[r] / stride;

    free(cursor);
    free(by_rank);
    free(tmp);
    free(sc);
  }

  free(sdispl_rank);
  *n_recv = total_recv / stride;
  return recv;
}

double *MFTLTopoCommExchange(const MFTL_Domain *dom, const double *records, const int *dest_counts, int stride, bool global_path, int *n_recv, int *src_counts)
{
  return alltoallv_doubles(dom, records, dest_counts, stride, global_path, src_counts, n_recv);
}

double *MFTLTopoCommReply(const MFTL_Domain *dom, const double *reply, const int *src_counts, const int *dest_counts, int stride, bool global_path)
{
  int n = 0;
  double *const back = alltoallv_doubles(dom, reply, src_counts, stride, global_path, MFTL_NULL, &n);
  (void) dest_counts;
  return back;
}

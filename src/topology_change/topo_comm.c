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
 *   MFTLTopoCommExchange /    personalised exchanges of the lattice overlap and
 *   MFTLTopoCommReply         the component labels, on the library's routed
 *                           exchange (MFTLExchangeRecords()): the neighbourhood
 *                           graph, or the global route when some window reaches
 *                           past it. The reply reuses the request's plan, so it
 *                           costs one data exchange and nothing else.
 *
 * The ghost ids of the new bodies are not here: they are a keyed pull on the
 * library's one ghost engine, MFTLGhostPull() (MFTLTopoSyncGhostIds()). The
 * label rounds stay here, because they are not a pull: the request carries a
 * label the owner folds into a minimum over every request and its own
 * components before any answer exists, and the "ghosts" are triangles of
 * per-cluster scratch meshes, not the pool suffix of an ensemble body.
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

double *MFTLTopoCommExchange(const MFTL_Domain *dom, const double *records, const int *dest_counts, int stride, int *n_recv, MFTL_ExchangePlan *plan)
{
  /* The packers group by rank in rank order; the engine takes the non-empty
   * groups as its destination list. */
  int *const list = (int *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(dom->nproc, 1) * sizeof(int), "topology destinations");
  int *const count = list + dom->nproc;
  int n_dest = 0;
  for (int r = 0; r < dom->nproc; ++r)
  {
    if (dest_counts[r] == 0) continue;
    list[n_dest] = r;
    count[n_dest++] = dest_counts[r];
  }
  double *const recv = (double *) MFTLExchangeRecords(dom, (size_t) stride * sizeof(double), records, n_dest, list, count, plan, n_recv);
  free(list);
  return recv;
}

double *MFTLTopoCommReply(const MFTL_Domain *dom, const MFTL_ExchangePlan *plan, const double *reply, int stride)
{
  return (double *) MFTLExchangeReply(dom, plan, (size_t) stride * sizeof(double), reply);
}

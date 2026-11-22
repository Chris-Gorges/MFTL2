/**
 * @file topo_comm.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief EVERY MPI call of the topology-change module.
 *
 * Two shapes, and how each scales:
 *
 *   MFTLTopoCommAllgather     the event list, and one record per (component,
 *                           holder). Both are paid only on a step with events,
 *                           and both are proportional to what changed. It is a
 *                           comm_world collective ON PURPOSE: every rank must
 *                           agree on the event list, including a rank two boxes
 *                           away that will reconstruct part of a body it does
 *                           not yet hold.
 *   MFTLTopoCommExchangeTo /  personalised exchanges of the lattice overlap (each
 *   MFTLTopoCommExchange /    record names its destination) and of the component
 *   MFTLTopoCommReply         labels (grouped in place by the library's grouping,
 *                           MFTLExchangeGroupRecords()), on the library's routed
 *                           exchange: the neighbourhood graph, or the global
 *                           route when some window reaches past it. Nothing per
 *                           call is sized by the process count. The reply
 *                           reuses the request's plan, so it costs one data
 *                           exchange and nothing else.
 *
 * The small reductions are not here: the handful of values reduced PER
 * CLUSTER or per listed body, and the label rounds' "did anything change"
 * vote, are the library's MFTLAllreduceDoubles() and its siblings, called from
 * topo_driver.c. Clusters exist only where events were detected, so those are
 * bounded by what happened, not by the ensemble.
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

double *MFTLTopoCommExchangeTo(const MFTL_Domain *dom, const double *records, const int *dest, int n_records, int stride, int *n_recv)
{
  return (double *) MFTLExchangeRecordsTo(dom, (size_t) stride * sizeof(double), records, dest, n_records, n_recv);
}

double *MFTLTopoCommExchange(const MFTL_Domain *dom, const double *records, int n_dest, const int *ranks, const int *counts, int stride, int *n_recv,
                             MFTL_ExchangePlan *plan)
{
  return (double *) MFTLExchangeRecords(dom, (size_t) stride * sizeof(double), records, n_dest, ranks, counts, plan, n_recv);
}

double *MFTLTopoCommReply(const MFTL_Domain *dom, const MFTL_ExchangePlan *plan, const double *reply, int stride)
{
  return (double *) MFTLExchangeReply(dom, plan, (size_t) stride * sizeof(double), reply);
}

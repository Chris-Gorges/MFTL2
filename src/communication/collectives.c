/**
 * @file collectives.c
 * @author Christian Gorges
 * @date 2026-11-21
 * @ingroup mftl_communication
 *
 * @brief The small reductions and the wall clock that every module shares.
 *
 * Before this file, the topology change, the corrections, the remesher and the
 * coupler each kept private copies of these wrappers. They were the same
 * `MPI_Allreduce` written four times, and they existed only so that each
 * module's confinement test would pass. These are the only copies now, and a
 * module's confinement scan refuses a new one.
 */

#include "mftl_communication.h"

#include <mpi.h>

#include "mftl_constants.h"

static MPI_Op mpi_op(MFTL_ReduceOp op)
{
  MFTL_ASSERT((op == MFTL_REDUCE_MIN || op == MFTL_REDUCE_MAX || op == MFTL_REDUCE_SUM) && "MFTLAllreduce: unknown operator");
  if (op == MFTL_REDUCE_MIN) return MPI_MIN;
  if (op == MFTL_REDUCE_MAX) return MPI_MAX;
  return MPI_SUM;
}

/* Every rank passes the same n, so this early return is taken on every rank or
 * on none. On a one-rank communicator the reduction is the identity. */
static void allreduce_in_place(MPI_Comm comm, void *v, int n, MPI_Datatype type, MFTL_ReduceOp op)
{
  if (n <= 0) return;
  int nproc = 1;
  MPI_Comm_size(comm, &nproc);
  if (nproc == 1) return;
  MPI_Allreduce(MPI_IN_PLACE, v, n, type, mpi_op(op), comm);
}

void MFTLAllreduceDoubles(MPI_Comm comm, double *v, int n, MFTL_ReduceOp op) { allreduce_in_place(comm, v, n, MPI_DOUBLE, op); }

void MFTLAllreduceInts(MPI_Comm comm, int *v, int n, MFTL_ReduceOp op) { allreduce_in_place(comm, v, n, MPI_INT, op); }

void MFTLAllreduceLongLongs(MPI_Comm comm, long long *v, int n, MFTL_ReduceOp op) { allreduce_in_place(comm, v, n, MPI_LONG_LONG, op); }

double MFTLWallTime(void) { return MPI_Wtime(); }

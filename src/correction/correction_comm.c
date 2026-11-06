/**
 * @file correction_comm.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief EVERY MPI call of the front-correction module.
 *
 * Two shapes, both small and fixed in size: a handful of statistics per call,
 * and one flag per colour of the undulation schedule (at most 27). Everything
 * per body goes through MFTLEnsembleReduceBodies() and everything per vertex
 * through MFTLUpdateGhostPositions(), which choose their own routes.
 */

#include "correction_internal.h"

#include <mpi.h>

double MFTLCorrectionTime(void) { return MPI_Wtime(); }

void MFTLCorrectionAllreduceDouble(const MFTL_Domain *dom, double *v, int n, int op)
{
  if (n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, op == 0 ? MPI_MIN : (op == 1 ? MPI_MAX : MPI_SUM), dom->comm_world);
}

void MFTLCorrectionAllreduceLongLong(const MFTL_Domain *dom, long long *v, int n, int op)
{
  if (n <= 0) return;
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_LONG_LONG, op == 0 ? MPI_MIN : (op == 1 ? MPI_MAX : MPI_SUM), dom->comm_world);
}

void MFTLCorrectionCoords(const MFTL_Domain *dom, int coords[3])
{
  /* comm_cart is created with reorder = 0, so its ranks are comm_world's. */
  MPI_Cart_coords(dom->comm_cart, dom->rank, 3, coords);
}

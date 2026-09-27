/*
 * Manual example ("Remeshing"): a serial remesh cycle inside the serial bracket (the whole
 * ensemble gathered on rank 0) and a parallel remesh cycle on the distributed ensemble. The
 * body is a sphere that is first stretched into an ellipsoid, so that the edge lengths leave
 * the target window. Both drivers end with a volume recovery against the pre-remesh volume.
 *
 * Run: mpirun -np <N> ./manual_remeshing             (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_remeshing.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* x-only cuts {nproc,1,1}: the decomposition the load-balanced parallel remesher uses */
  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0}, hi[3] = {2.0, 2.0, 2.0};
  bool periodic[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);
  ens->type = MFTL_ENSEMBLE_SPANNING; /* the parallel driver runs the distributed shift scheme */

  MFTL_Object* body = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, body);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(body, c, 0.6, 3); /* 1280 triangles, all on rank 0 */
  const double stretch[3] = {1.6, 1.0, 1.0}; /* an ellipsoid: long edges along x */
  MFTLScaleObject(body, stretch);             /* recomputes the triangle properties */
  MFTLRedistribute(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  const double V0 = MFTLObjectVolume(body);
  const int n0 = body->global_n_triangles;

  /* Remesh parameters: isotropic window [0.8 h, 4/3 h] around the target length h. Every field
     is initialised; defaults follow MFTL1 (Botsch-Kobbelt criterion, paraboloid fit with LAPACK
     or barycentric fit without). */
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.08, 5, MFTL_SMOOTH_TRIANGLE_BASED, 45.0);
  p.quiet = 1;

  /* Serial remeshing: the operators are rank-local, so the ensemble is gathered on rank 0 for
     the serial driver and redistributed afterwards. */
  MFTL_SerialBracket bracket;
  MFTLEnsembleBeginSerial(ens, &bracket);
  MFTLRemeshEnsemble(ens, &p);
  MFTLEnsembleEndSerial(ens, &bracket, 1);
  MFTLEnsembleUpdateGlobalCounts(ens);
  const double V1 = MFTLObjectVolume(body);
  const int n1 = body->global_n_triangles;

  /* Parallel remeshing on the distributed ensemble (a new, finer target). */
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.06, 5, MFTL_SMOOTH_TRIANGLE_BASED, 45.0);
  p.quiet = 1;
  if (MFTLRemeshEnsembleParallel(ens, &p) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  MFTLEnsembleUpdateGlobalCounts(ens);
  const double V2 = MFTLObjectVolume(body);
  const int n2 = body->global_n_triangles;

  long ops[3] = {p.stat_splits, p.stat_collapses, p.stat_flips}, gops[3];
  MPI_Allreduce(ops, gops, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("start:    %5d triangles  V = %.10f\n", n0, V0);
    printf("serial:   %5d triangles  V = %.10f  (rel. change %.2e)\n", n1, V1, (V1 - V0) / V0);
    printf("parallel: %5d triangles  V = %.10f  (rel. change %.2e)\n", n2, V2, (V2 - V0) / V0);
    printf("parallel pass operations: %ld splits, %ld collapses, %ld flips\n", gops[0], gops[1], gops[2]);
  }
  const int ok = fabs(V1 - V0) < 1e-6 * V0 && fabs(V2 - V0) < 1e-6 * V0 && n2 > n0;
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}

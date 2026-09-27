/*
 * Manual example (doc/MFTL2.tex, "A first program"): create an ensemble on an x-slab
 * decomposition, generate a sphere, write and re-read it as GTS, redistribute it, evaluate the
 * single-body integrals, apply a transform and refresh the ghosts, and clean up.
 *
 * Run: mpirun -np <N> ./manual_basic_workflow        (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_io.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* 1. The ensemble: a box [-2,2]^3 cut into nproc slabs along x, not periodic. */
  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0}, hi[3] = {2.0, 2.0, 2.0};
  bool periodic[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 4);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  /* 2. A body: an icosphere of radius 1 (4 refinements: 5120 triangles). The generator is
        collective; the mesh is built on rank 0. */
  MFTL_Object* sphere = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, sphere);
  double centre[3] = {0.1, 0.0, 0.0};
  MFTLGenerateSphere(sphere, centre, 1.0, 4);
  MFTLComputeTriangleProperties(sphere);

  /* ... written to a GTS file (before redistribution: the file is serial) and read back into
         a second body. */
  char gts[64];
  snprintf(gts, sizeof(gts), "manual_sphere_np%d.gts", nproc);
  MFTLWriteGTS(sphere, gts);
  MFTL_Object* copy = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, copy);
  if (MFTLReadGTS(copy, gts) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 2);
  if (rank == 0) remove(gts);
  MFTLComputeTriangleProperties(copy);

  /* 3. Redistribution: every entity moves to the rank that owns its position; one ghost layer
        is added and completed. Afterwards the mesh is quiescent. */
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 3);
  MFTLEnsembleUpdateGlobalCounts(ens);

  /* 4. Integrals: collective sums over the owned triangles. */
  double V = MFTLObjectVolume(sphere);
  double cm[3];
  double A = MFTLSurfaceCenter(sphere, cm);
  double I[3][3];
  MFTLMomentOfInertia(sphere, I, cm, 1.0);
  double Vc = MFTLObjectVolume(copy); /* collective: every rank calls it */
  if (rank == 0)
  {
    printf("triangles %d  V = %.10f (4/3 pi = %.10f)  A = %.10f\n", sphere->global_n_triangles, V, 4.0 * MFTL_PI / 3.0, A);
    printf("surface centre (%.3e, %.3e, %.3e)  Ixx = %.6f\n", cm[0], cm[1], cm[2], I[0][0]);
    printf("GTS copy: V = %.10f\n", Vc);
  }

  /* 5. A transform of the owned vertices, then the ghosts are refreshed from their owners and
        the body is redistributed (the triangles moved relative to the slabs). */
  const double shift[3] = {0.5, 0.0, 0.0};
  MFTLTranslateObject(sphere, shift);
  MFTLUpdateGhostPositions(ens, MFTL_NULL);
  MFTLComputeTriangleProperties(sphere);
  MFTLRedistribute(ens);
  double V2 = MFTLObjectVolume(sphere);
  MFTLSurfaceCenter(sphere, cm);
  if (rank == 0) printf("after translation: V = %.10f  centre x = %.6f\n", V2, cm[0]);

  MFTLEnsembleDestroy(&ens); /* destroys the objects too */
  MPI_Finalize();
  return (fabs(V - V2) < 1e-9 * V && fabs(V - Vc) < 1e-12 * V) ? 0 : 1;
}

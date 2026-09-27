/*
 * Manual example ("Curvature"): vertex normals, the cotangent Laplace-Beltrami curvatures and
 * the roughness of a distributed sphere. For a sphere of radius R the stored mean curvature is
 * 2/R (= 2H, the magnitude of the Laplace-Beltrami vector) and the Gaussian curvature 2/R^2
 * (MFTL1 convention).
 *
 * Run: mpirun -np <N> ./manual_curvature             (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0}, hi[3] = {2.0, 2.0, 2.0};
  bool periodic[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);

  const double R = 0.75;
  MFTL_Object* body = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, body);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(body, c, R, 4);
  /* Cached triangle area/normal/centre: the centre decides the owner, and the cached values
     travel with the (ghost) triangles. */
  MFTLComputeTriangleProperties(body);
  MFTLRedistribute(ens); /* the fan of every owned vertex is now stored (ghost completion G2) */

  /* The one-ring kernels read the positions and the cached triangle properties of the fan. */
  MFTLComputeVertexNormals(body);
  MFTLComputeVertexCurvatures(body);
  const double rough = MFTLComputeRoughness(body); /* collective */
  MFTLUpdateGhostPositions(ens, MFTL_NULL);          /* ghosts receive normal/curvature/area */

  /* statistics over the owned vertices */
  double hmin = 1e300, hmax = -1e300, kmin = 1e300, kmax = -1e300, ndev = 0.0;
  for (int i = 0; i < body->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &body->vertices[i];
    if (v->is_ghost) continue;
    hmin = fmin(hmin, v->meancurvature), hmax = fmax(hmax, v->meancurvature);
    kmin = fmin(kmin, v->gaussiancurvature), kmax = fmax(kmax, v->gaussiancurvature);
    /* the vertex normal of a sphere is radial and outward (outward winding) */
    const double r = sqrt(v->pos[0] * v->pos[0] + v->pos[1] * v->pos[1] + v->pos[2] * v->pos[2]);
    double d = 0.0;
    for (int k = 0; k < 3; ++k) d += (v->normal[k] - v->pos[k] / r) * (v->normal[k] - v->pos[k] / r);
    ndev = fmax(ndev, sqrt(d));
  }
  double loc[5] = {-hmin, hmax, -kmin, kmax, ndev}, glob[5];
  MPI_Allreduce(loc, glob, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0)
  {
    printf("mean curvature in [%.5f, %.5f]   2/R   = %.5f\n", -glob[0], glob[1], 2.0 / R);
    printf("Gaussian       in [%.5f, %.5f]   2/R^2 = %.5f\n", -glob[2], glob[3], 2.0 / (R * R));
    printf("max |n - radial| = %.2e   roughness = %.3e\n", glob[4], rough);
  }
  const int ok = fabs(-glob[0] - 2.0 / R) < 0.05 * 2.0 / R && fabs(glob[1] - 2.0 / R) < 0.05 * 2.0 / R && glob[4] < 1e-2;
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}

/*
 * Manual example ("Periodicity and image counters"): a sphere placed across the x = 1 seam of a
 * triply periodic unit box. Positions are stored wrapped into [0,1)^3; the image counters img
 * record which periodic image each vertex belongs to, so the continuous coordinate
 * X = pos + img * L is the unbroken sphere. The body is translated through the seam, wrapped
 * (counting the wraps into img) and redistributed; its volume and its COMPACT class are
 * unchanged. A closed film that spans x and y shows the WRAPPING class.
 *
 * Run: mpirun -np <N> ./manual_periodic              (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_images.h"
#include "mftl_topology.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
  bool periodic[3] = {true, true, true};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 2);

  /* The generator takes CONTINUOUS coordinates: the centre (0.95, 0.5, 0.5) puts half of the
     sphere beyond x = 1; those vertices are stored at pos_x - 1 with img_x = 1. */
  MFTL_Object* drop = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, drop);
  double c[3] = {0.95, 0.5, 0.5};
  MFTLGenerateSphere(drop, c, 0.2, 3);
  MFTLComputeTriangleProperties(drop); /* the cached centre decides the triangle owner */
  MFTLRedistribute(ens);

  int imin = 1 << 30, imax = -(1 << 30);
  for (int i = 0; i < drop->n_vertices; ++i)
  {
    if (drop->vertices[i].is_ghost) continue;
    const int m = drop->vertices[i].img[0];
    imin = m < imin ? m : imin, imax = m > imax ? m : imax;
  }
  int gmin, gmax;
  MPI_Allreduce(&imin, &gmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&imax, &gmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  /* The single-body integrals use continuous coordinates, so the seam does not matter. */
  const double V0 = MFTLObjectVolume(drop);
  double cm[3];
  MFTLSurfaceCenter(drop, cm);
  int wrap[3];
  MFTLObjectWrapFlags(drop, wrap);

  /* Move by 0.3 along x (owned vertices), wrap the stored positions back into the box -- every
     wrap is counted into img, so X is unchanged by the wrap -- refresh the ghosts and
     redistribute. */
  const double d[3] = {0.3, 0.0, 0.0};
  MFTLTranslateObject(drop, d);
  MFTLWrapObjectIntoDomain(drop);
  MFTLUpdateGhostPositions(ens, MFTL_NULL);
  MFTLComputeTriangleProperties(drop);
  MFTLRedistribute(ens);
  const double V1 = MFTLObjectVolume(drop);
  double cm1[3];
  MFTLSurfaceCenter(drop, cm1);

  /* Edge vectors are minimum-image differences; the continuous coordinate of a vertex: */
  double X[3] = {0, 0, 0};
  if (drop->n_vertices > 0) MFTLVertexContinuous(drop, &drop->vertices[0], X);

  /* A closed film z = 0.5 spanning the periodic x and y directions: built from a soup of
     continuous coordinates (triangles given on rank 0 only), it WRAPS x and y. */
  const int N = 8;
  MFTL_Object* film = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, film);
  int nv = rank == 0 ? N * N : 0, nt = rank == 0 ? 2 * N * N : 0;
  int* vg = malloc((size_t) (N * N) * sizeof(int));
  double(*P)[3] = malloc((size_t) (N * N) * sizeof(*P));
  int* tg = malloc((size_t) (2 * N * N) * sizeof(int));
  int(*tv)[3] = malloc((size_t) (2 * N * N) * sizeof(*tv));
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      const int v = j * N + i;
      vg[v] = v, P[v][0] = (double) i / N, P[v][1] = (double) j / N, P[v][2] = 0.5;
      const int a = v, b = j * N + (i + 1) % N, cc = ((j + 1) % N) * N + (i + 1) % N, q = ((j + 1) % N) * N + i;
      tg[2 * v] = 2 * v, tv[2 * v][0] = a, tv[2 * v][1] = b, tv[2 * v][2] = cc;
      tg[2 * v + 1] = 2 * v + 1, tv[2 * v + 1][0] = a, tv[2 * v + 1][1] = cc, tv[2 * v + 1][2] = q;
    }
  MFTLBuildFromSoup(film, nv, vg, (const double(*)[3]) P, MFTL_NULL, nt, tg, (const int(*)[3]) tv, MFTL_NULL);
  MFTLComputeTriangleProperties(film);
  MFTLRedistribute(ens);
  int fwrap[3];
  MFTLObjectWrapFlags(film, fwrap);
  const int compact = MFTLObjectIsCompact(film);

  if (rank == 0)
  {
    printf("img_x range %d..%d  wrap flags drop (%d,%d,%d)  film (%d,%d,%d) compact=%d\n", gmin, gmax, wrap[0], wrap[1], wrap[2], fwrap[0], fwrap[1], fwrap[2], compact);
    printf("V = %.12f before, %.12f after the seam crossing (4/3 pi r^3 ~ %.6f)\n", V0, V1, 4.0 / 3.0 * MFTL_PI * 0.008);
    /* the centre is a continuous coordinate in the image of an anchor vertex: its periodic
       image (not its value) is gauge-independent, so print it wrapped into the box */
    printf("surface centre x (wrapped): %.6f -> %.6f\n", cm[0] - floor(cm[0]), cm1[0] - floor(cm1[0]));
    printf("vertex 0 on rank 0: pos (%.4f, %.4f, %.4f) img (%d, %d, %d) -> X = (%.4f, %.4f, %.4f)\n", drop->vertices[0].pos[0], drop->vertices[0].pos[1],
           drop->vertices[0].pos[2], drop->vertices[0].img[0], drop->vertices[0].img[1], drop->vertices[0].img[2], X[0], X[1], X[2]);
  }
  const int ok = gmin == 0 && gmax == 1 && !wrap[0] && fwrap[0] && fwrap[1] && !fwrap[2] && !compact && fabs(V1 - V0) < 1e-10 * V0 && fabs((cm1[0] - floor(cm1[0])) - 0.25) < 1e-9;
  free(vg), free(P), free(tg), free(tv);
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}

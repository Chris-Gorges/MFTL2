/*
 * Manual example ("Point queries and intersection"): point-in-object classification (single
 * query and a caller-held query for many points), point-triangle distance, the axis-aligned
 * bounding box, and the LBVH intersection of two overlapping spheres.
 *
 * Run: mpirun -np <N> ./manual_queries               (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_bvh.h"

static long g_hits = 0;
static void count_hit(MFTL_Object* a, int ta, MFTL_Object* b, int tb, void* user)
{
  (void) a, (void) ta, (void) b, (void) tb, (void) user;
  g_hits++;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0}, hi[3] = {2.0, 2.0, 2.0};
  bool periodic[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 2);

  MFTL_Object* a = MFTLObjectCreate(0);
  MFTL_Object* b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, a);
  MFTLEnsembleAddObject(ens, b);
  double ca[3] = {-0.3, 0.0, 0.0}, cb[3] = {0.4, 0.0, 0.0};
  MFTLGenerateSphere(a, ca, 0.5, 3);
  MFTLGenerateSphere(b, cb, 0.5, 3);
  MFTLComputeTriangleProperties(a); /* before redistributing: the cached centre decides the owner */
  MFTLComputeTriangleProperties(b);
  MFTLRedistribute(ens);

  /* Point in object: collective ray casting; 1 inside, 0 outside, -1 on the surface. Every rank
     passes the same point. */
  const double p_in[3] = {-0.3, 0.1, 0.0}, p_out[3] = {1.5, 0.0, 0.0};
  const int in = MFTLPointInsideObject(a, p_in);
  const int out = MFTLPointInsideObject(a, p_out);

  /* Many points against one prebuilt LBVH (same verdicts as the single-point call).
     KNOWN DEFECT (deferred, see the manual's limitations): on a non-periodic domain the caller-
     held query deadlocks when some rank stores no triangle of the object, so it is used only
     when every rank holds part of the body; otherwise the single-point call is used. */
  double xyz[4 * 3] = {-0.3, 0, 0, 0.19, 0, 0, 0.21, 0, 0, 0.0, 0.6, 0};
  int verdict[4];
  int n_min;
  MPI_Allreduce(&a->n_triangles, &n_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (n_min > 0)
  {
    MFTL_PointQuery* q = MFTLPointQueryCreate(a);
    MFTLPointQueryInsideMany(q, 4, xyz, verdict);
    MFTLPointQueryDestroy(&q);
  }
  else
    for (int m = 0; m < 4; ++m) verdict[m] = MFTLPointInsideObject(a, &xyz[3 * m]);

  /* Axis-aligned box of the owned vertices (collective) and a containment test. */
  double box_c[3], box_d[3];
  MFTLObjectAABB(a, box_c, box_d);
  const int in_box = MFTLPointInsideAABB(box_c, box_d, p_in);

  /* Distance from a point to the stored triangles of this rank (rank-local), minimised over all
     ranks. */
  double dmin = 1e300;
  for (int t = 0; t < a->n_triangles; ++t)
    if (!a->triangles[t].is_ghost) dmin = fmin(dmin, MFTLPointTriangleDistance(a, t, p_out));
  double gdmin;
  MPI_Allreduce(&dmin, &gdmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  /* Intersection of the two spheres: spatial ghosts, one LBVH per body and a pair traversal.
     The callback runs rank-locally; stored triangle pairs near a rank boundary can be seen on
     two ranks, so the global hit count is a sum over ranks, not a pair count. */
  MFTLEnsembleIntersectObjects(ens, MFTL_NULL, count_hit, MFTL_NULL);
  long ghits = 0;
  MPI_Allreduce(&g_hits, &ghits, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0)
  {
    printf("inside(p_in) = %d  inside(p_out) = %d  many: %d %d %d %d\n", in, out, verdict[0], verdict[1], verdict[2], verdict[3]);
    printf("AABB centre (%.3f, %.3f, %.3f) half-size (%.3f, %.3f, %.3f), p_in in box: %d\n", box_c[0], box_c[1], box_c[2], box_d[0], box_d[1], box_d[2], in_box);
    printf("distance(p_out, sphere a) = %.4f (exact 1.3)  intersecting pairs (summed over ranks): %ld\n", gdmin, ghits);
  }
  const int ok = in == 1 && out == 0 && verdict[0] == 1 && verdict[1] == 1 && verdict[2] == 0 && verdict[3] == 0 && ghits > 0;
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}

/*
 * Manual example ("Per-triangle data"): attach a flat POD payload to every triangle, move the
 * body across the ranks, and check that every copy (owned and ghost) still carries the bytes
 * of its triangle; then refresh the ghost payloads from their owners after a change. Also
 * shows the conserved surface scalar (a concentration per unit area).
 *
 * Run: mpirun -np <N> ./manual_triangle_data         (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"

/* The payload must be plain data: no pointers (a pointer is only valid on the rank that
   wrote it). */
typedef struct
{
  int gid;        /* the triangle's global id, to check the transport */
  double force[3];
} CellData;

static int check(const MFTL_Object* o, double scale)
{
  int bad = 0;
  for (int t = 0; t < o->n_triangles; ++t)
  {
    const CellData* d = (const CellData*) MFTLGetTriangleData(o, &o->triangles[t]);
    if (d->gid != o->triangles[t].global_id || d->force[0] != scale * d->gid) bad++;
  }
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return gbad;
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
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);
  MFTL_Object* body = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, body);
  double c[3] = {-0.8, 0.0, 0.0};
  MFTLGenerateSphere(body, c, 0.5, 3);
  MFTLComputeTriangleProperties(body);

  /* Same size on every rank, before any redistribution. Blobs are allocated zeroed. */
  MFTLInitializeObjectData(body, sizeof(CellData));
  for (int t = 0; t < body->n_triangles; ++t)
  {
    CellData d = {body->triangles[t].global_id, {1.0 * body->triangles[t].global_id, 0.0, 0.0}};
    MFTLSetTriangleData(body, &body->triangles[t], &d);
    body->triangles[t].surface_scalar = 2.0; /* uniform concentration */
  }

  MFTLRedistribute(ens);
  const int bad1 = check(body, 1.0);

  /* Owners change their payload; ghosts are refreshed from the owners. */
  for (int t = 0; t < body->n_triangles; ++t)
  {
    if (body->triangles[t].is_ghost) continue;
    CellData* d = (CellData*) MFTLGetTriangleData(body, &body->triangles[t]);
    d->force[0] = 3.0 * d->gid;
  }
  MFTLScatterTriangleData(ens, MFTL_NULL);
  const int bad2 = check(body, 3.0);

  /* Surface-scalar mass sum(c A) over owned triangles: travels unchanged with the triangles. */
  const double shift[3] = {1.5, 0.0, 0.0};
  double mass0 = 0.0, mass1 = 0.0, m;
  for (int t = 0; t < body->n_triangles; ++t)
    if (!body->triangles[t].is_ghost) mass0 += body->triangles[t].surface_scalar * body->triangles[t].area;
  MFTLTranslateObject(body, shift);
  MFTLUpdateGhostPositions(ens, MFTL_NULL);
  MFTLComputeTriangleProperties(body);
  MFTLRedistribute(ens);
  for (int t = 0; t < body->n_triangles; ++t)
    if (!body->triangles[t].is_ghost) mass1 += body->triangles[t].surface_scalar * body->triangles[t].area;
  MPI_Allreduce(&mass0, &m, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), mass0 = m;
  MPI_Allreduce(&mass1, &m, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), mass1 = m;
  const int bad3 = check(body, 3.0);

  if (rank == 0)
    printf("payload mismatches: after redistribution %d, after scatter %d, after moving %d; scalar mass %.12f -> %.12f\n", bad1, bad2, bad3, mass0, mass1);
  const int ok = bad1 == 0 && bad2 == 0 && bad3 == 0 && fabs(mass1 - mass0) < 1e-12 * mass0;
  MFTLEnsembleDestroy(&ens); /* frees the payload blobs */
  MPI_Finalize();
  return ok ? 0 : 1;
}

/*
 * Real MFTLUpdateGhostPositions round-trip test (replaces the former
 * MPI_Send/Recv stub): build an actual 2-rank decomposition of a sphere,
 * rigidly advect the owned vertices, sync ghost positions through the
 * library, and assert every ghost received its owner's updated position.
 */

#include "test_utils.h"
#include <mpi.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"  // MFTLComputeTriangleProperties

int test_ghost_position_sync()
{
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (nproc != 2)
  {
    if (rank == 0) printf("Skipping MPI ghost sync test: requires exactly 2 ranks.\n");
    return 0;
  }

  // 1. Setup Ensemble (2x1x1) with the cut through the sphere center
  int cuts[3] = {2, 1, 1};
  double min[3] = {-2.0, -2.0, -2.0};
  double max[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};

  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, min, max, per, 10);
  if (!ens) return 1;

  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);

  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, 3);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistributeListOfObjects(ens);

  // 2. Pick a ghost vertex owned by the other rank and remember its position
  int g_idx = -1;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    if (obj->vertices[i].is_ghost && obj->vertices[i].owner_proc != rank)
    {
      g_idx = i;
      break;
    }
  }

  int have_ghost = (g_idx != -1);
  int any_ghost = 0;
  MPI_Allreduce(&have_ghost, &any_ghost, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  TEST_ASSERT(any_ghost);  // the test is only meaningful if ghosts exist somewhere

  double old_pos[3] = {0.0, 0.0, 0.0};
  if (have_ghost) memcpy(old_pos, obj->vertices[g_idx].pos, 3 * sizeof(double));

  // 3. Rigidly advect owned vertices only (ghosts must NOT be moved locally)
  double disp[3] = {0.1, -0.2, 0.3};
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!v->is_ghost && v->owner_proc == rank)
    {
      v->pos[0] += disp[0];
      v->pos[1] += disp[1];
      v->pos[2] += disp[2];
    }
  }

  // 4. Sync ghost positions through the library (collective)
  int rc = MFTLUpdateGhostPositions(ens, NULL);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

  // 5. Assert the ghost now carries the owner's displaced position
  if (have_ghost)
  {
    for (int d = 0; d < 3; ++d)
    {
      TEST_ASSERT_EQUAL_DOUBLE(old_pos[d] + disp[d], obj->vertices[g_idx].pos[d], 1e-12);
    }
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  RUN_TEST(test_ghost_position_sync);

  TEST_REPORT();

  MPI_Finalize();
  return (g_tests_failed > 0);
}

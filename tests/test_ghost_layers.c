#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"  // For periodicity if needed
#include <mpi.h>
#include <math.h>

int test_ghost_extension_accuracy()
{
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (nproc != 4)
  {
    if (rank == 0) printf("Skipping Ghost Test: Requires exactly 4 ranks.\n");
    return 0;  // Skip if environment is wrong
  }

  // 1. Setup Ensemble (2x2x1)
  int cuts[3] = {2, 2, 1};
  double min[3] = {-2.0, -2.0, -2.0};
  double max[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};  // Non-periodic for this test to keep it simple at boundaries?
  // User said "Calculate distance from origin... if bad vertex... likely (0,0,0)".
  // Let's use non-periodic.

  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, min, max, per, 10);
  if (!ens) return 1;

  // 2. Generate Sphere at Origin
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);

  double center[3] = {0.0, 0.0, 0.0};
  double radius = 1.0;
  int ref = 3;  // Enough to have good density

  // All ranks participate in valid generation logic (collective inside)
  MFTLGenerateSphere(obj, center, radius, ref);
  MFTLComputeTriangleProperties(obj);

  // 3. Redistribute
  MFTLRedistributeListOfObjects(ens);

  // Assert partitioning
  // With 2x2x1 cuts at 0,0,0, the sphere is cut exactly in 4.
  // Each rank should own roughly 1/4th.
  if (obj->n_vertices == 0)
  {
    printf("Rank %d has 0 vertices! Partitioning likely failed.\n", rank);
    g_tests_failed++;
  }

  // 4. Ghost Extension (2 layers)
  if (rank == 0) printf("Extending Ghost Layer by 2 steps...\n");
  MFTLExtendGhostLayer(ens, 2);

  // 5. Verification
  int ghost_count = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    if (obj->vertices[i].is_ghost)
    {
      ghost_count++;

      double* p = obj->vertices[i].pos;
      double dist = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      double err = fabs(dist - 1.0);

      if (err > 1e-4)
      {
        printf("Rank %d: Bad Ghost Vertex %d (GID %lld). Pos: (%f, %f, %f). Dist: %f. Err: %f\n", rank, i, (long long) obj->vertices[i].global_id, p[0], p[1],
               p[2], dist, err);
        TEST_ASSERT(err < 1e-4);
      }
    }
  }

  if (rank == 0) printf("Rank %d: Verified geometric accuracy of ghosts.\n", rank);

  // Topology Check on a Ghost
  int valid_ghost_structure = 0;
  if (ghost_count > 0)
  {
    // Find a ghost vertex
    int g_idx = -1;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if (obj->vertices[i].is_ghost)
      {
        g_idx = i;
        break;
      }
    }

    if (g_idx != -1)
    {
      // Check Neighborhood
      MFTL_Neighborhood nb;
      int seed = g_idx;
      MFTLGetNeighborhood(obj, &seed, 1, true, 1, &nb);  // V->T (1 step)

      if (nb.n_tris > 0)
      {
        // Must have at least one triangle that is either ghost or boundary
        // Actually, if we extend layer, we pull triangles.
        // The ghost vertex MUST be connected to something local or another ghost.
        valid_ghost_structure = 1;
      }
      else
      {
        printf("Rank %d: Ghost Vertex %d (GID %lld) has 0 incident triangles!\n", rank, g_idx, (long long) obj->vertices[g_idx].global_id);
      }
      MFTLFreeNeighborhood(&nb);
    }
  }
  else
  {
    // It is possible (unlikely with this setup) to have no ghosts if partitioning is perfect and no adjacency?
    // No, Sphere is continuous.
  }

  // MFTL has a "ghosts are only vertices" mode or "ghosts include triangles"?
  // communication.c usually sends triangles (PackedTriangleComplex) implicitly when extending.
  // So we expect triangles.
  if (ghost_count > 0) TEST_ASSERT(valid_ghost_structure);

  // 6. Stress (Extend 5 more)
  int old_ghosts = ghost_count;
  MFTLExtendGhostLayer(ens, 5);

  int new_ghost_count = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
    if (obj->vertices[i].is_ghost) new_ghost_count++;

  if (rank == 0) printf("Rank %d: Ghost Count: %d -> %d\n", rank, (int) old_ghosts, new_ghost_count);

  // Ideally, ghost count increases or stays same if we covered everything.
  // For a sphere distributed on 4 ranks, 2 layers != full mesh. 5 more should add more.
  TEST_ASSERT(new_ghost_count >= old_ghosts);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  RUN_TEST(test_ghost_extension_accuracy);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}

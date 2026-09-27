/*
 * MFTLGenerateSphere on one rank: global triangle count and mesh integrity.
 */
#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_topology.h"
#include <mpi.h>

int test_sphere_generation()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // 1. Setup Ensemble
  int cuts[3] = {1, 1, 1};
  double min[3] = {-2, -2, -2};
  double max[3] = {2, 2, 2};
  bool per[3] = {false, false, false};

  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, min, max, per, 5);
  if (!ens) return 1;

  // 2. Add Object
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);

  // 3. Generate Sphere (Refinement 3)
  // Icosahedron Base:
  // Ref 0: V=12, T=20
  // Ref 1: T=80 (20*4)
  // Ref 2: T=320 (80*4)
  // Ref 3: T=1280 (320*4)
  // Vertices roughly V ~ T/2 + 2 for closed genus 0
  // Expected T = 1280
  // Expected V = 10 * 4^3 + 2 = 642

  double center[3] = {0, 0, 0};
  int ref = 3;
  MFTLGenerateSphere(obj, center, 1.0, ref);

  // 4. Verification
  // Count local entities (serial run)
  int local_t = obj->n_triangles;
  int local_v = obj->n_vertices;

  if (rank == 0)
  {
    TEST_ASSERT_EQUAL_INT(1280, local_t);
    TEST_ASSERT_EQUAL_INT(642, local_v);

    // Also check global stats which should be populated
    TEST_ASSERT_EQUAL_INT(1280, obj->global_n_triangles);
    TEST_ASSERT_EQUAL_INT(642, obj->global_n_vertices);
  }

  // 5. Integrity
  int integrity = MFTLCheckMeshIntegrity(obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, integrity);

  // Cleanup
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  RUN_TEST(test_sphere_generation);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}

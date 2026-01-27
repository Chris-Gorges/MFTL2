#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"  // For periodicity
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

// --- Helpers ---
typedef struct
{
  int x, y, z;
} CutConfig;

// Helper to find valid factorizations of nproc
// Simplistic: Brute force restricted range
int GetValidCutConfigurations(int nproc, CutConfig* configs, int max_configs)
{
  int count = 0;
  // We want permutations roughly.
  // Shuffle logic is complex in C, let's just generate then shuffle roughly or iterate strided.

  // Find all factors
  for (int i = 1; i <= nproc; i++)
  {
    if (nproc % i == 0)
    {
      int rem1 = nproc / i;
      for (int j = 1; j <= rem1; j++)
      {
        if (rem1 % j == 0)
        {
          int k = rem1 / j;
          if (count < max_configs)
          {
            configs[count].x = i;
            configs[count].y = j;
            configs[count].z = k;
            count++;
          }
        }
      }
    }
  }

  // Very basic shuffle
  for (int i = 0; i < count; i++)
  {
    int r = rand() % count;
    CutConfig temp = configs[i];
    configs[i] = configs[r];
    configs[r] = temp;
  }

  return count;
}

int test_redistribution_stress()
{
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  srand(42);  // Deterministic-ish per rank

  // 1. Get Configurations
  CutConfig configs[20];
  int num_configs = GetValidCutConfigurations(nproc, configs, 10);  // Limit to 10

  if (rank == 0) printf("Running Stress Test with %d Processes. Testing %d Configurations.\n", nproc, num_configs);

  // 2. Loop Configs
  for (int c = 0; c < num_configs; c++)
  {
    int cuts[3] = {configs[c].x, configs[c].y, configs[c].z};
    if (rank == 0) printf("  Config %d: [%d, %d, %d]\n", c, cuts[0], cuts[1], cuts[2]);

    double min[3] = {-5, -5, -5};
    double max[3] = {5, 5, 5};
    bool per[3] = {true, true, true};

    // Create Ensemble
    MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, min, max, per, 20);
    if (!ens)
    {
      if (rank == 0) printf("    Failed to create ensemble!\n");
      return 1;
    }

    // 3. Initialize Spheres (Refinement 3 is lighter than 4, aiming for speed vs stress)
    // Let's use Ref 3 (~1280 tris per sphere). 10 Spheres = 12,800 total tris.
    // Ref 4 (~5000 tris). 10 Spheres = 50,000 tris. Good stress.
    int num_spheres = 50;
    int ref = 3;

    for (int s = 0; s < num_spheres; s++)
    {
      MFTL_Object* obj = MFTLObjectCreate(s);
      MFTLEnsembleAddObject(ens, obj);

      // Random Position
      // --- Generate random center coordinates ---
      double rand_x = (double) rand() / RAND_MAX;
      double rand_y = (double) rand() / RAND_MAX;
      double rand_z = (double) rand() / RAND_MAX;

      double center[3] = {-4.0 + rand_x * (4.0 - (-4.0)), -4.0 + rand_y * (4.0 - (-4.0)), -4.0 + rand_z * (4.0 - (-4.0))};

      MFTLGenerateSphere(obj, center, 0.5, ref);
      MFTLComputeTriangleProperties(obj);
    }

    // Initial Distribution
    MFTLRedistributeListOfObjects(ens);
    MFTLEnsembleUpdateGlobalCounts(ens);

    mftl_int_t initial_tris = ens->total_global_triangles;

    if (rank == 0) printf("    Initial Tris: %lld\n", (long long) initial_tris);

    // 4. Simulation Loop (Advection + Redist)
    double dt = 0.2;
    double velocity[3] = {1.0, 0.5, -0.5};  // Move diagonally
    int steps = 25;

    for (int t = 0; t < steps; t++)
    {
      // Advect Vertices
      for (int o = 0; o < ens->num_objects; o++)
      {
        MFTL_Object* obj = ens->objects[o];
        for (int i = 0; i < obj->n_vertices; i++)
        {
          if (obj->vertices[i].owner_proc == rank)
          {
            obj->vertices[i].pos[0] += velocity[0] * dt;
            obj->vertices[i].pos[1] += velocity[1] * dt;
            obj->vertices[i].pos[2] += velocity[2] * dt;
            MFTLApplyPeriodicity(&ens->domain, obj->vertices[i].pos);
          }
        }
      }

      // Sync Ghosts
      MFTLUpdateGhostPositions(ens, NULL);

      // Recompute Geometry
      for (int o = 0; o < ens->num_objects; o++) MFTLComputeTriangleProperties(ens->objects[o]);

      // Redistribute
      MFTLRedistributeListOfObjects(ens);

      // VERIFY CONSERVATION
      MFTLEnsembleUpdateGlobalCounts(ens);

      if (ens->total_global_triangles != initial_tris)
      {
        if (rank == 0)
        {
          printf("    [FAIL] Step %d: Lost Triangles! Expected %lld, Got %lld\n", t, (long long) initial_tris, (long long) ens->total_global_triangles);
        }
        MFTLEnsembleDestroy(&ens);
        return 1;  // Immediate Fail
      }
    }

    if (rank == 0) printf("    [PASS] Config %d conserved mass over %d steps.\n", c, steps);

    MFTLEnsembleDestroy(&ens);
  }

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // MFTLRegisterMpiTypes usually called inside EnsembleCreate?
  // No, doc says "should be called once by every process after MPI_Init" in mftl_communication.h
  // But EnsembleCreate might handle it lazily or user must do it.
  // Looking at examples/sphere_creation_serial.c -> It doesn't call Register explicitly.
  // It says "Ensemble handles type reg if first active".
  // So we rely on EnsembleCreate.

  RUN_TEST(test_redistribution_stress);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}

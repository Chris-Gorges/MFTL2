/*
 * Redistribution stress test (8 ranks, fully periodic box): several spheres are advected with
 * random velocities and redistributed repeatedly on every valid Cartesian decomposition of the
 * rank count; the owned volume (seam-safe) and the triangle count are conserved.
 */
#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"  // For periodicity
#include "mftl_topology.h"  // MFTLGetTriangleVertices
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

// Signed tetrahedron volume of one triangle about the origin:
//   V = (1/6) * v0 . (v1 x v2)
// (Stage-02 conservation formula, computed inline until the integrals API
// lands. Sums over owned, non-ghost triangles only.)
//
// Sets *straddled when an owned triangle crosses a periodic boundary (raw
// corner-to-corner span larger than half the domain): the plain triple
// product is only meaningful for a consistently embedded surface, so callers
// must skip the comparison on such steps. Minimum-image integral handling is
// Stage 02/13 work.
static double compute_owned_volume(MFTL_Object* obj, int rank, const MFTL_Domain* dom, int* straddled)
{
  double vol = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != rank) continue;

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);

    for (int k = 0; k < 3; ++k)
    {
      if (!dom->is_periodic[k]) continue;
      double half = 0.5 * dom->domain_size[k];
      if (fabs(vv[1]->pos[k] - vv[0]->pos[k]) > half || fabs(vv[2]->pos[k] - vv[0]->pos[k]) > half) *straddled = 1;
    }

    const double* a = vv[0]->pos;
    const double* b = vv[1]->pos;
    const double* c = vv[2]->pos;

    double bc[3] = {b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
    vol += (a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]) / 6.0;
  }
  return vol;
}

static double compute_global_volume(MFTL_Ensemble* ens, int rank, int* straddled)
{
  double local = 0.0;
  int local_straddled = 0;
  for (int o = 0; o < ens->num_objects; ++o) local += compute_owned_volume(ens->objects[o], rank, &ens->domain, &local_straddled);

  double global = 0.0;
  int global_straddled = 0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_straddled, &global_straddled, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  *straddled = global_straddled;
  return global;
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

    int initial_tris = ens->total_global_triangles;
    int ignored = 0;
    double initial_volume = compute_global_volume(ens, rank, &ignored);

    if (rank == 0) printf("    Initial Tris: %lld\n", (long long) initial_tris);

    // 4. Simulation Loop (Advection + Redist)
    double dt = 0.2;
    double velocity[3] = {1.0, 0.5, -0.5};  // Move diagonally
    int steps = 25;

    // Volume-conservation bookkeeping: a configuration whose steps ALL get
    // skipped (periodic-wall crossings) must NOT report conservation as
    // passed — require at least one real comparison per configuration.
    int vol_checks = 0;
    int vol_skips = 0;

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

      // Volume must be conserved as well: the mesh is rigidly advected, so
      // the signed tetrahedron sum may only drift by round-off. Steps where a
      // body crosses a periodic wall are skipped: the raw triple product is
      // not defined for a boundary-straddling embedding (minimum-image
      // integrals arrive with Stage 02/13).
      int straddled = 0;
      double volume = compute_global_volume(ens, rank, &straddled);
      if (!straddled)
      {
        vol_checks++;
        double vol_tol = 1e-9 * fmax(1.0, fabs(initial_volume));
        if (fabs(volume - initial_volume) > vol_tol)
        {
          if (rank == 0)
          {
            printf("    [FAIL] Step %d: Volume not conserved! Expected %.15e, Got %.15e (diff %.3e)\n", t, initial_volume, volume,
                   fabs(volume - initial_volume));
          }
          MFTLEnsembleDestroy(&ens);
          return 1;  // Immediate Fail
        }
      }
      else
      {
        vol_skips++;
        if (rank == 0) printf("    [SKIP] Step %d: body crossing periodic wall, volume comparison not defined.\n", t);
      }
    }

    if (vol_checks == 0)
    {
      if (rank == 0)
      {
        printf("    [FAIL] Config %d: volume conservation was never checked (all %d steps skipped due to periodic-wall crossings)!\n", c, steps);
      }
      MFTLEnsembleDestroy(&ens);
      return 1;  // A fully skipped configuration must not count as conserved.
    }

    if (rank == 0)
      printf("    [PASS] Config %d conserved mass over %d steps (%d volume checks, %d boundary-skips).\n", c, steps, vol_checks, vol_skips);

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

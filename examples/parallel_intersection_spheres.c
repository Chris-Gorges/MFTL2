/*
 * Example: parallel intersection detection of two spheres (LBVH pair traversal with the
 * Guigue-Devillers triangle test and pairwise spatial ghosting). The spheres start at opposite
 * ends of the domain and move towards each other across rank boundaries.
 */

#include "mftl_core.h"
#include "mftl_topology.h"
#include "mftl_bvh.h"
#include "mftl_geometry.h"
#include "mftl_communication.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int local_intersections = 0;

void sphere_intersection_cb(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, void* user_data) { local_intersections++; }

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Provide a valid geometric domain mapping dynamically based on parallel threads
  // For 4 procs, cuts will be {4, 1, 1} over a domain [-2.0, 2.0].
  // Processor boundaries will be at x = -1.0, 0.0, 1.0.
  int cuts[3] = {nprocs, 1, 1};
  double global_min[3] = {-2.0, -2.0, -2.0};
  double global_max[3] = {2.0, 2.0, 2.0};
  bool periodicity[3] = {false, false, false};

  MFTL_Ensemble* ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, global_min, global_max, periodicity, 10);
  if (!ensemble)
  {
    fprintf(stderr, "Failed to create MFTL Ensemble on rank %d\n", rank);
    MPI_Finalize();
    return -1;
  }

  // --- Generate Objects ---
  double radius = 0.5;

  // Sphere A: Starts at x = -1.5, moves right (+1.0 velocity)
  MFTL_Object* objA = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ensemble, objA);
  double centerA[3] = {-1.5, 0.0, 0.0};
  MFTLGenerateSphere(objA, centerA, radius, 3);
  MFTLComputeTriangleProperties(objA);
  MFTLRedistributeObject(objA);

  // Sphere B: Starts at x = 1.5, moves left (-1.0 velocity)
  MFTL_Object* objB = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(ensemble, objB);
  double centerB[3] = {1.5, 0.0, 0.0};
  MFTLGenerateSphere(objB, centerB, radius, 3);
  MFTLComputeTriangleProperties(objB);
  MFTLRedistributeObject(objB);

  if (rank == 0)
  {
    printf("========================================\n");
    printf("MFTL Dynamic Sphere Intersection Test\n");
    printf("========================================\n");
    printf("Ranks: %d\n", nprocs);
    printf("Domain: [%.1f, %.1f] in X\n", global_min[0], global_max[0]);
    printf("Sphere A starting at x=-1.5, vel=+1.0\n");
    printf("Sphere B starting at x=+1.5, vel=-1.0\n");
    printf("Expected intersection around t=1.0 at x=0.0 (domain center/boundary!)\n");
    printf("========================================\n");
  }

  // --- Time Loop ---
  double dt = 0.1;
  double velA = 1.0;
  double velB = -1.0;

  for (double time = 0.0; time <= 2.0 + 1e-9; time += dt)
  {
    if (rank == 0) printf("\n--- Timestep t = %.1f ---\n", time);

    // 1. Advect vertices
    if (time > 0.0)  // skip movement on frame 0 to check initial state
    {
      for (int i = 0; i < objA->n_vertices; i++)
      {
        if (objA->vertices[i].owner_proc == rank) objA->vertices[i].pos[0] += velA * dt;
      }
      for (int i = 0; i < objB->n_vertices; i++)
      {
        if (objB->vertices[i].owner_proc == rank) objB->vertices[i].pos[0] += velB * dt;
      }

      // Recompute properties and redistribute to handle migrating triangles
      MFTLUpdateGhostPositions(ensemble, MFTL_NULL);
      MFTLComputeTriangleProperties(objA);
      MFTLComputeTriangleProperties(objB);
      MFTLRedistributeListOfObjects(ensemble);
    }

    // 2. Flatten Topology for fast array access in intersection tests
    // MFTLFlattenConnectivity(objA);
    // MFTLFlattenConnectivity(objB);

    // 3. SOTA Spatial Ghosting exchange to handle cross-boundary collisions accurately
    // MFTL_CreateSpatialGhosts(objA, objB);

    // 4. Bounding Volume Hierarchies
    MFTL_BVH* bvhA = MFTL_GenerateLBVH(objA);
    MFTL_BVH* bvhB = MFTL_GenerateLBVH(objB);

    // 5. Intersection Check
    local_intersections = 0;
    if (bvhA && bvhB)
    {
      MFTL_IntersectObjects(objA, bvhA, objB, bvhB, sphere_intersection_cb, NULL);
    }

    // Aggregate intersections across all domains
    long long local_cnt = (long long) local_intersections;
    long long global_cnt = 0;
    MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
      if (global_cnt > 0)
      {
        printf("[!] Intersection DETECTED! %lld triangle pairs overlapping.\n", global_cnt);
      }
      else
      {
        printf("    No intersection.\n");
      }
    }

    // Cleanup for next step
    if (bvhA) MFTL_DestroyBVH(bvhA);
    if (bvhB) MFTL_DestroyBVH(bvhB);
  }

  if (rank == 0) printf("========================================\n");

  MFTLEnsembleDestroy(&ensemble);
  MPI_Finalize();
  return 0;
}

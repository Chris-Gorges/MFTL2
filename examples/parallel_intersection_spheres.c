/*
 * @Author: MFTL
 * @Date: 2026-02-28
 *
 * This program checks the intersection of two spheres in parallel.
 * It strictly tests the new LBVH approach and robust SAT triangles intersections.
 */

#include "mftl_core.h"
#include "mftl_topology.h"
#include "mftl_bvh.h"
#include "mftl_geometry.h"
#include "mftl_communication.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

mftl_int_t local_intersections = 0;

void sphere_intersection_cb(MFTL_Object* objA, mftl_int_t triA_id, MFTL_Object* objB, mftl_int_t triB_id, void* user_data) { local_intersections++; }

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Provide a valid geometric domain mapping dynamically based on parallel threads
  int cuts[3] = {nprocs, 1, 1};
  double global_min[3] = {-5.0, -5.0, -5.0};
  double global_max[3] = {5.0, 5.0, 5.0};
  bool periodicity[3] = {false, false, false};

  MFTL_Ensemble* ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, global_min, global_max, periodicity, 10);
  if (!ensemble)
  {
    fprintf(stderr, "Failed to create MFTL Ensemble on rank %d\n", rank);
    MPI_Finalize();
    return -1;
  }

  // --- Generate Objects ---

  // Sphere A
  MFTL_Object* objA = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ensemble, objA);
  double centerA[3] = {-0.5, 0.0, 0.0};
  MFTLGenerateSphere(objA, centerA, 1.0, 4);  // Refinement level 4 generates hundreds of triangles
  MFTLComputeTriangleProperties(objA);
  MFTLRedistributeObject(objA);

  // Sphere B
  MFTL_Object* objB = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(ensemble, objB);
  // Placed symmetrically across the X=0 processor boundary to test spatial ghosting
  double centerB[3] = {0.5, 0.0, 0.0};
  MFTLGenerateSphere(objB, centerB, 1.0, 4);
  MFTLComputeTriangleProperties(objB);
  MFTLRedistributeObject(objB);

  // VERY IMPORTANT: Our optimized SAT math operates on purely contiguous vertex memory,
  // so we map the topological half-edge loops out into `flat_tri_verts`
  MFTLFlattenConnectivity(objA);
  MFTLFlattenConnectivity(objB);

  // Invoke the SOTA Spatial Ghosting exchange to handle cross-boundary collisions
  MFTL_CreateSpatialGhosts(objA, objB);

  // --- Bounding Volume Hierarchies ---

  // Time the BVH builds
  double t_bvh_start = MPI_Wtime();
  MFTL_BVH* bvhA = MFTL_GenerateLBVH(objA);
  MFTL_BVH* bvhB = MFTL_GenerateLBVH(objB);
  double t_bvh_end = MPI_Wtime();

  if (rank == 0)
  {
    printf("========================================\n");
    printf("MFTL Parallel Sphere Intersection Test\n");
    printf("========================================\n");
    printf("Ranks: %d\n", nprocs);
    printf("Sphere A: %lld triangles\n", (long long) objA->n_triangles);
    printf("Sphere B: %lld triangles\n", (long long) objB->n_triangles);
    printf("BVH build time: %.5f ms\n", (t_bvh_end - t_bvh_start) * 1000.0);
    printf("Starting Traversal...\n");
  }

  // --- Intersection Check ---

  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  local_intersections = 0;
  if (bvhA && bvhB)
  {
    MFTL_IntersectObjects(objA, bvhA, objB, bvhB, sphere_intersection_cb, NULL);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();

  // Aggregate intersections across all domains
  long long local_cnt = (long long) local_intersections;
  long long global_cnt = 0;
  MPI_Reduce(&local_cnt, &global_cnt, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    double dt = t_end - t_start;
    printf("Traversal and SAT intersection time: %.5f ms\n", dt * 1000.0);
    printf("Total Sub-Triangle Intersections Found = %lld\n", global_cnt);
    printf("========================================\n");
  }

  // --- Cleanup ---
  if (bvhA) MFTL_DestroyBVH(bvhA);
  if (bvhB) MFTL_DestroyBVH(bvhB);
  MFTLEnsembleDestroy(&ensemble);

  MPI_Finalize();
  return 0;
}

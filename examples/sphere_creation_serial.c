/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:44:56
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-08-14 12:54:56
 *
 * This program initializes a mesh, generates a sphere (icosahedron),
 * writes it to a VTK file for visualization, and then cleans up all resources.
 */

#include "mftl_types.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_io.h"
#include <mpi.h>
#include <stdio.h>

mftl_int_t main(mftl_int_t argc, char *argv[])
{
  // --- MPI Initialization ---
  MPI_Init(&argc, &argv);

  mftl_int_t rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // --- 1. Create the Mesh Object ---
  // We define our simulation domain as non-periodic for this simple case.
  const bool periodicity[3] = {false, false, false};
  MFTL_Object *object = MFTLObjectCreate(MPI_COMM_WORLD);

  if (!object)
  {
    fprintf(stderr, "Rank %d: Failed to create mesh object.\n", rank);
    MPI_Finalize();
    return -1;
  }
  printf("Rank %d: Mesh object created.\n", rank);

  // --- 2. Initialize the Domain ---
  // We'll define a simple 1x1x1 processor grid (i.e., serial)
  // and a domain from [-2, -2, -2] to [2, 2, 2].
  const mftl_int_t cuts[3] = {1, 1, 1};
  const double global_min[3] = {-2.0, -2.0, -2.0};
  const double global_max[3] = {2.0, 2.0, 2.0};
  MFTLObjectInitDomain(object, cuts, global_min, global_max, periodicity);
  printf("Rank %d: Mesh domain initialized.\n", rank);

  // --- 3. Generate the Sphere ---
  // This function currently only runs on rank 0.
  if (rank == 0)
  {
    const double center[3] = {0.0, 0.0, 0.0};
    const double radius = 1.0;
    const mftl_int_t refinements = 5;  // 0 refinements = base icosahedron

    printf("Rank %d: Generating sphere (refinements=%d)...\n", rank, refinements);
    mftl_int_t result = MFTLGenerateSphere(object, center, radius, refinements);
    if (result != 0)
    {
      fprintf(stderr, "Rank %d: Failed to generate sphere.\n", rank);
    }
    else
    {
      printf("Rank %d: Sphere generated with %d vertices and %d triangles.\n", rank, object->n_vertices, object->n_triangles);
    }
  }

  // --- 4. Write to VTK File ---
  // Only the rank that generated the mesh should write it.
  if (rank == 0)
  {
    printf("Rank %d: Writing mesh to VTK file...\n", rank);
    MFTLWriteVtk(object, "icosahedron_output");

    MFTLCheckMeshIntegrity(object);

    // print_mesh_details(object);
  }

  // --- 5. Destroy the Mesh Object ---
  // All processes must call destroy to free the mesh and communicator.
  MFTLObjectDestroy(&object);
  printf("Rank %d: Mesh object destroyed.\n", rank);

  // --- MPI Finalization ---
  MPI_Finalize();
  return 0;
}
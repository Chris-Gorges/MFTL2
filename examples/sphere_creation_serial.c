/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:44:56
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-08-14 12:54:56
 *
 * This program initializes a mesh, generates a sphere (icosahedron),
 * writes it to a VTK file for visualization, and then cleans up all resources.
 */

#include "mftl_core.h"
#include "mftl_topology.h"
#include "mftl_io.h"
#include <mpi.h>
#include <stdio.h>

mftl_int_t main(int argc, char *argv[])
{
  // --- MPI Initialization ---
  MPI_Init(&argc, &argv);

  int rank;  // standard MPI uses int
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // --- 1. Define Simulation Parameters ---
  // Note: MPI cuts are standard 'int', not 'mftl_int_t'
  int cuts[3] = {1, 1, 1};
  double global_min[3] = {-2.0, -2.0, -2.0};
  double global_max[3] = {2.0, 2.0, 2.0};
  bool periodicity[3] = {false, false, false};

  // --- 2. Create the Ensemble (The Context) ---
  // This sets up the Domain, MPI Topologies, and registers MPI Types.
  // We reserve space for 10 objects initially.
  MFTL_Ensemble *ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, global_min, global_max, periodicity, 10);

  if (!ensemble)
  {
    fprintf(stderr, "Rank %d: Failed to create MFTL Ensemble.\n", rank);
    MPI_Finalize();
    return -1;
  }
  if (rank == 0) printf("Rank %d: Ensemble created and Domain initialized.\n", rank);

  // --- 3. Create & Add the Object ---
  // Create an empty mesh container with Global ID 1
  MFTL_Object *object = MFTLObjectCreate(1);

  // CRITICAL: Link the object to the ensemble.
  // This sets object->domain = &ensemble->domain
  MFTLEnsembleAddObject(ensemble, object);

  if (rank == 0) printf("Rank %d: Object created and added to Ensemble.\n", rank);

  // --- 4. Generate the Sphere ---
  // This is a COLLECTIVE call. All ranks must participate because of MPI_Bcast inside.
  // The function internally checks 'if (object->domain->rank == 0)' for the generation part.

  double center[3] = {0.0, 0.0, 0.0};
  double radius = 1.0;
  mftl_int_t refinements = 5;

  if (rank == 0) printf("Rank %d: Generating sphere (refinements=%lld)...\n", rank, (long long) refinements);

  // Notice: We pass 'object', and it knows its own domain now.
  mftl_int_t result = MFTLGenerateSphere(object, center, radius, refinements);

  if (result != MFTL_SUCCESS)
  {
    fprintf(stderr, "Rank %d: Failed to generate sphere.\n", rank);
  }
  else
  {
    // Access global stats updated by the generate function
    if (rank == 0)
    {
      printf("Rank %d: Sphere generated. Global Stats: %lld Vertices, %lld Triangles.\n", rank, (long long) object->global_n_vertices,
             (long long) object->global_n_triangles);
    }
  }

  // --- 5. Write to VTK File ---
  // Assuming WriteVtk runs on Rank 0 or is parallel-aware.
  // If it's a serial writer gathering data, handle logic accordingly.
  if (rank == 0)
  {
    printf("Rank %d: Writing mesh to VTK file...\n", rank);
    MFTLWriteVtk(object, "icosahedron_output");

    MFTLCheckMeshIntegrity(object);
  }

  // --- 6. Destroy the Ensemble ---
  // This frees the Ensemble, the Domain, AND all added Objects (including 'object').
  // You do NOT need to call MFTLObjectDestroy(&object) manually.
  MFTLEnsembleDestroy(&ensemble);

  if (rank == 0) printf("Rank %d: Ensemble and objects destroyed.\n", rank);

  // --- MPI Finalization ---
  // Note: Your Reference Counting in EnsembleDestroy will handle MFTL_Free_mpi_types() automatically here.
  MPI_Finalize();
  return 0;
}
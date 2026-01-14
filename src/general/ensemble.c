/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:54:07
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 19:41:02
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"

// --- Static Reference Counter ---
// This variable is hidden inside this translation unit.
// It tracks how many active Ensembles currently exist.
static int g_mftl_ensemble_ref_count = 0;

// Helper: Internal function to setup the complex MPI Graph (moved from Object)
static mftl_int_t MFTLDomainInitComm(MFTL_Domain* domain)
{
  if (!domain) return MFTL_ERROR;

  // --- 1. Create Cartesian Communicator ---
  int periods[3] = {(int) domain->is_periodic[0], (int) domain->is_periodic[1], (int) domain->is_periodic[2]};
  // For now, we disable reordering to ensure ranks are in the same order as the input communicator.
  int reorder = 0;

  if (MPI_Cart_create(domain->comm_world, 3, domain->cuts, periods, reorder, &domain->comm_cart) != MPI_SUCCESS)
  {
    return MFTL_ERROR;
  }

  // --- Pre-calculate Rank Map ---
  domain->rank_map = (int*) malloc(domain->nproc * sizeof(int));

  // Pre-calculate math factors to replace division with multiplication later
  for (int d = 0; d < 3; d++)
  {
    domain->grid_factors[d] = domain->cuts[d] / domain->domain_size[d];
  }

  // Populate the map
  int coords[3];
  for (int k = 0; k < domain->cuts[2]; ++k)
  {
    for (int j = 0; j < domain->cuts[1]; ++j)
    {
      for (int i = 0; i < domain->cuts[0]; ++i)
      {
        coords[0] = i;
        coords[1] = j;
        coords[2] = k;
        int rank_out;

        // We call the slow MPI function here, only once per cell!
        MPI_Cart_rank(domain->comm_cart, coords, &rank_out);

        // Calculate linear index: i + j*cuts[0] + k*cuts[0]*cuts[1]
        int linear_idx = i + (j * domain->cuts[0]) + (k * domain->cuts[0] * domain->cuts[1]);
        domain->rank_map[linear_idx] = rank_out;
      }
    }
  }

  // --- 2. Manually Find All 26 Neighbors ---
  int my_coords[3];
  // Get my own coordinates within the new Cartesian communicator
  MPI_Cart_coords(domain->comm_cart, domain->rank, 3, my_coords);

  int neighbor_ranks[26];
  int num_neighbors = 0;

  // Iterate through the 3x3x3 cube centered on the current process
  for (int dz = -1; dz <= 1; ++dz)
  {
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        // Skip the center point (myself)
        if (dx == 0 && dy == 0 && dz == 0)
        {
          continue;
        }

        int neighbor_coords[3] = {my_coords[0] + dx, my_coords[1] + dy, my_coords[2] + dz};
        bool is_valid_neighbor = MFTL_TRUE;

        // Handle periodicity and boundary conditions for each dimension
        for (int i = 0; i < 3; ++i)
        {
          if (periods[i])
          {
            // Periodic wrap-around. This formula handles negative results correctly.
            neighbor_coords[i] = (neighbor_coords[i] + domain->cuts[i]) % domain->cuts[i];
          }
          else
          {
            // Non-periodic: check if the neighbor is out of bounds
            if (neighbor_coords[i] < 0 || neighbor_coords[i] >= domain->cuts[i])
            {
              is_valid_neighbor = MFTL_FALSE;
              break;  // This is not a real neighbor
            }
          }
        }

        if (is_valid_neighbor)
        {
          int neighbor_rank;
          // Find the rank corresponding to the valid neighbor coordinates
          MPI_Cart_rank(domain->comm_cart, neighbor_coords, &neighbor_rank);
          neighbor_ranks[num_neighbors++] = neighbor_rank;
        }
      }
    }
  }

  int unique_neighbor_ranks[26];
  int num_unique_neighbors = 0;

  for (int j = 0; j < num_neighbors; ++j)
  {
    int d = neighbor_ranks[j];
    int found = 0;

    // Check if we've already added this integer to our output list.
    for (int k = 0; k < num_unique_neighbors; ++k)
    {
      if (unique_neighbor_ranks[k] == d)
      {
        found = 1;
        break;
      }
    }

    // If it wasn't found, add it to the list.
    if (!found)
    {
      unique_neighbor_ranks[num_unique_neighbors++] = d;
    }
  }

  // --- 3. Create the Distributed Graph Communicator ---
  // For an adjacent graph, the sources of incoming messages are the same
  // as the destinations of outgoing messages.
  if (MPI_Dist_graph_create_adjacent(domain->comm_world,
                                     num_unique_neighbors,  // indegree
                                     unique_neighbor_ranks,  // sources
                                     MPI_UNWEIGHTED,  // source weights
                                     num_unique_neighbors,  // outdegree
                                     unique_neighbor_ranks,  // destinations
                                     MPI_UNWEIGHTED,  // destination weights
                                     MPI_INFO_NULL,  // info
                                     reorder,  // allow reordering
                                     &domain->comm_graph) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create distributed graph communicator.\n", domain->rank);
    // Clean up the Cartesian communicator before failing
    MPI_Comm_free(&domain->comm_cart);
    return MFTL_ERROR;
  }

  // Retrieve neighbor list
  int weighted;
  MPI_Dist_graph_neighbors_count(domain->comm_graph, &domain->num_neighbors, &domain->num_neighbors, &weighted);
  domain->neighbors = (int*) malloc(domain->num_neighbors * sizeof(int));

  // Create temp buffer for weights/etc if needed by API, then free
  int* temp_neighbors = (int*) malloc(domain->num_neighbors * sizeof(int));
  MPI_Dist_graph_neighbors(domain->comm_graph, domain->num_neighbors, domain->neighbors, MPI_UNWEIGHTED, domain->num_neighbors, temp_neighbors, MPI_UNWEIGHTED);
  free(temp_neighbors);

  return MFTL_SUCCESS;
}

MFTL_Ensemble* MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3],
                                  mftl_int_t initial_object_capacity)
{
  MFTL_Ensemble* ens = (MFTL_Ensemble*) calloc(1, sizeof(MFTL_Ensemble));
  if (!ens) return MFTL_NULL;

  // --- 1. Setup Domain Data ---
  MFTL_Domain* dom = &ens->domain;

  // Duplicate world comm
  MPI_Comm_dup(comm, &dom->comm_world);
  MPI_Comm_size(dom->comm_world, &dom->nproc);
  MPI_Comm_rank(dom->comm_world, &dom->rank);

  // Check Sanity
  if (cuts[0] * cuts[1] * cuts[2] != dom->nproc)
  {
    if (dom->rank == 0) fprintf(stderr, "MFTL Error: Cuts do not match MPI Size.\n");
    MFTLEnsembleDestroy(&ens);  // Safe cleanup
    return MFTL_NULL;
  }

  // Copy Bounds & Flags
  for (int i = 0; i < 3; ++i)
  {
    dom->cuts[i] = cuts[i];
    dom->global_min[i] = global_min[i];
    dom->global_max[i] = global_max[i];
    dom->domain_size[i] = global_max[i] - global_min[i];
    dom->domain_half_size[i] = dom->domain_size[i] * 0.5;
    if (is_periodic) dom->is_periodic[i] = is_periodic[i];
  }

  // --- 2. Setup Topology (The heavy lifting) ---
  if (MFTLDomainInitComm(dom) != MFTL_SUCCESS)
  {
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  // --- 3. Setup Object List ---
  ens->capacity_objects = (initial_object_capacity > 0) ? initial_object_capacity : 1;
  ens->objects = (MFTL_Object**) calloc(ens->capacity_objects, sizeof(MFTL_Object*));
  ens->num_objects = 0;

  // --- 1. Global Initialization (First time only) ---
  if (g_mftl_ensemble_ref_count == 0)
  {
    MFTLRegisterMpiTypes();
  }
  g_mftl_ensemble_ref_count++;

  return ens;
}

void MFTLEnsembleDestroy(MFTL_Ensemble** p_ens)
{
  if (!p_ens || !(*p_ens)) return;
  MFTL_Ensemble* ens = *p_ens;

  // 1. Destroy all contained Objects
  for (mftl_int_t i = 0; i < ens->num_objects; ++i)
  {
    MFTLObjectDestroy(&ens->objects[i]);
  }
  free(ens->objects);

  // 2. Destroy Domain Communicators
  MFTL_Domain* dom = &ens->domain;
  if (dom->comm_graph != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_graph);
  if (dom->comm_cart != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_cart);
  if (dom->comm_world != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_world);
  if (dom->neighbors) free(dom->neighbors);

  // 3. Free Ensemble
  free(ens);
  *p_ens = MFTL_NULL;

  // --- 2. Global Finalization ---
  g_mftl_ensemble_ref_count--;

  // Sanity check to prevent negative counts if user calls destroy twice
  if (g_mftl_ensemble_ref_count <= 0)
  {
    g_mftl_ensemble_ref_count = 0;  // Reset to clean state
    MFTLFreeMpiTypes();
  }
}

mftl_int_t MFTLEnsembleAddObject(MFTL_Ensemble* ensemble, MFTL_Object* object)
{
  // 1. Link the Object to the Domain
  object->domain = &ensemble->domain;

  // 2. Simple dynamic array resize logic here if num_objects == capacity
  if (ensemble->num_objects == ensemble->capacity_objects)
  {
    mftl_int_t new_capacity = ensemble->capacity_objects * 2;

    // 1. Use a temporary pointer
    MFTL_Object** temp = (MFTL_Object**) realloc(ensemble->objects, new_capacity * sizeof(MFTL_Object*));

    // 2. Check for failure
    assert(temp);

    // 3. Assign only on success
    ensemble->objects = temp;
    ensemble->capacity_objects = new_capacity;
  }
  ensemble->objects[ensemble->num_objects++] = object;
  return MFTL_SUCCESS;
}
/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:54:07
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 19:41:02
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "khash.h"
#include "mftl_types.h"
#include "mftl_communication.h"
#include <stdint.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"

/**
 * @brief Creates and performs basic initialization of a new MFTL_Object struct.
 *
 * This function allocates the main mesh struct and sets up the MPI
 * communicator. It does NOT allocate the main data arrays (pools) or define the
 * domain decomposition. That is handled by other functions.
 *
 * @param comm The MPI communicator the mesh will be part of. If MPI_COMM_NULL
 * is passed, the function will use MPI_COMM_WORLD.
 * @return A pointer to the newly allocated mesh, or NULL on failure.
 */
MFTL_Object* MFTLObjectCreate(MPI_Comm comm)
{
  // Use calloc to allocate and zero-initialize the entire struct.
  // This sets all pointers to NULL and all numbers to 0, which is safe.
  MFTL_Object* object = (MFTL_Object*) calloc(1, sizeof(MFTL_Object));
  if (!object)
  {
    fprintf(stderr, "MFTL Error: Failed to allocate memory for MFTL_Object struct.\n");
    return MFTL_NULL;
  }

  MPI_Comm comm_to_use = (comm == MPI_COMM_NULL) ? MPI_COMM_WORLD : comm;

  // We duplicate the communicator to ensure that operations on this mesh's
  // communicator do not interfere with other parts of the user's application.
  MPI_Comm_dup(comm_to_use, &object->comm);
  MPI_Comm_rank(object->comm, &object->rank);
  MPI_Comm_size(object->comm, &object->nproc);

  object->vertex_lookup = kh_init(vertex_map);
  object->triangle_lookup = kh_init(triangle_map);
  object->halfedge_lookup = kh_init(halfedge_map);

  object->next_vertex_gid = 0;
  object->next_triangle_gid = 0;
  object->next_halfedge_gid = 0;

  // The data pools and domain info are left uninitialized, to be set later.
  // calloc has already set them to NULL/0.

  return object;
}

/**
 * @brief Creates a topology-aware MPI graph communicator for 26-neighbor communication.
 * @details This function first establishes a virtual Cartesian topology using
 * MPI_Cart_create. It then manually calculates the ranks of all 26 potential
 * neighbors (faces, edges, and corners) for the current process, respecting
 * the domain's periodicity. Finally, it uses this explicit neighbor list to
 * create a highly optimized distributed graph communicator via
 * MPI_Dist_graph_create_adjacent.
 *
 * MPI works with int, therefore we need to pay attention with mftl_int_t.
 *
 * @param object The MFTL object where the new communicator will be stored.
 * @param cuts An array of 3 integers specifying the number of divisions in x,y,z.
 * @param is_periodic An array of 3 booleans indicating periodicity in x,y,z.
 * @return MFTL_SUCCESS on success, MFTL_ERROR on MPI failure.
 */
static mftl_int_t MFTLObjectInitComm(MFTL_Object* object, int cuts[3], bool is_periodic[3])
{
  if (!object) return MFTL_ERROR;

  // --- 1. Create the Cartesian Communicator ---
  // MPI requires an int array for periods, so we convert from bool.
  int periods[3];  // Use 'int', not 'mftl_int_t'
  for (int i = 0; i < 3; ++i)
  {
    periods[i] = (int) is_periodic[i];
  }

  // Allow MPI to reorder ranks for better performance
  // mftl_int_t reorder = 1;
  // For now, we disable reordering to ensure ranks are in the same order as the input communicator.
  int reorder = 0;
  if (MPI_Cart_create(object->comm, 3, cuts, periods, reorder, &object->cart_comm) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create Cartesian communicator.\n", object->rank);
    return MFTL_ERROR;
  }

  // --- 2. Manually Find All 26 Neighbors ---
  int my_coords[3];
  // Get my own coordinates within the new Cartesian communicator
  MPI_Cart_coords(object->cart_comm, (int) object->rank, 3, my_coords);

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
            neighbor_coords[i] = (neighbor_coords[i] + cuts[i]) % cuts[i];
          }
          else
          {
            // Non-periodic: check if the neighbor is out of bounds
            if (neighbor_coords[i] < 0 || neighbor_coords[i] >= cuts[i])
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
          MPI_Cart_rank(object->cart_comm, neighbor_coords, &neighbor_rank);
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
  if (MPI_Dist_graph_create_adjacent(object->comm,
                                     num_unique_neighbors,  // indegree
                                     unique_neighbor_ranks,  // sources
                                     MPI_UNWEIGHTED,  // source weights
                                     num_unique_neighbors,  // outdegree
                                     unique_neighbor_ranks,  // destinations
                                     MPI_UNWEIGHTED,  // destination weights
                                     MPI_INFO_NULL,  // info
                                     reorder,  // allow reordering
                                     &object->graph_comm) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create distributed graph communicator.\n", object->rank);
    // Clean up the Cartesian communicator before failing
    MPI_Comm_free(&object->cart_comm);
    return MFTL_ERROR;
  }

  int weighted;
  MPI_Dist_graph_neighbors_count(object->graph_comm, &object->num_neighbors, &object->num_neighbors, &weighted);

  object->neighbors = (int*) malloc(object->num_neighbors * sizeof(int));
  int* temp_neighbors = (int*) malloc(object->num_neighbors * sizeof(int));

  MPI_Dist_graph_neighbors(object->graph_comm, object->num_neighbors, object->neighbors, MPI_UNWEIGHTED, object->num_neighbors, temp_neighbors, MPI_UNWEIGHTED);

  free(temp_neighbors);

#ifdef DEBUG
  // --- Debug: Print Graph Neighbors ---
  {
    // 1. Gather info safely
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Use world rank for clear identification

    // 2. Build a string buffer (prevents interleaved output lines from different procs)
    // 1024 chars is usually enough for ~26 neighbors
    char debug_buf[1024];
    int offset = 0;

    offset +=
        snprintf(debug_buf + offset, sizeof(debug_buf) - offset, "[DEBUG Rank %d] Graph Comm Created. %d Neighbors: {", rank, (int) object->num_neighbors);

    for (int i = 0; i < object->num_neighbors; ++i)
    {
      offset += snprintf(debug_buf + offset, sizeof(debug_buf) - offset, "%d%s", object->neighbors[i], (i < object->num_neighbors - 1) ? ", " : "");
    }

    // Close the brace
    snprintf(debug_buf + offset, sizeof(debug_buf) - offset, "}\n");

    // 3. Print to stderr (unbuffered, usually appears immediately)
    fprintf(stderr, "%s", debug_buf);

    // Optional: Barrier to keep logs ordered (slows down init but makes reading easier)
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  return MFTL_SUCCESS;
}

/**
 * @brief Initializes the Cartesian domain decomposition for the mesh.
 *
 * This function calculates the spatial region (bounding box) owned by each
 * MPI rank based on a specified number of "cuts" in each dimension.
 *
 * @param mesh The mesh object to initialize.
 * @param cuts An array of 3 integers specifying the number of divisions in x,
 * y, z.
 * @param global_min The minimum coordinates of the global simulation domain.
 * @param global_max The maximum coordinates of the global simulation domain.
 * @param is_periodic An array of 3 booleans indicating periodicity in x,y,z.
 * @return MFTL_SUCCESS on success, MFTL_ERROR on failure.
 */
mftl_int_t MFTLObjectInitDomain(MFTL_Object* object, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3])
{
  if (!object) return MFTL_ERROR;

  // --- Sanity Check ---
  if (cuts[0] * cuts[1] * cuts[2] != object->nproc)
  {
    if (object->rank == 0)
    {
      fprintf(stderr,
              "MFTL Error: Processor grid dimensions (%d x %d x %d = %d) do "
              "not match the number of MPI ranks (%d).\n",
              cuts[0], cuts[1], cuts[2], cuts[0] * cuts[1] * cuts[2], object->nproc);
    }
    return MFTL_ERROR;
  }

  // Store periodicity flags passed by the user.
  if (is_periodic)
  {
    for (mftl_int_t i = 0; i < 3; ++i)
    {
      object->is_periodic[i] = is_periodic[i];
    }
  }  // Otherwise, they remain false from calloc.

  // This establishes the rank-to-coordinate mapping we will use.
  if (MFTLObjectInitComm(object, cuts, is_periodic) != MFTL_SUCCESS)
  {
    return MFTL_ERROR;
  }

  // --- Store Domain Info & Calculate Periodicity Variables ---
  for (mftl_int_t i = 0; i < 3; ++i)
  {
    object->cuts[i] = cuts[i];
    object->global_bbox_min[i] = global_min[i];
    object->global_bbox_max[i] = global_max[i];

    // Calculate domain size and half-size.
    // This is done once here for efficiency.
    object->domain_size[i] = global_max[i] - global_min[i];
    object->domain_half_size[i] = object->domain_size[i] * 0.5;
  }

  //   double dx[3];
  //   for (mftl_int_t i = 0; i < 3; i++)
  //   {
  //     dx[i] = (object->global_bbox_max[i] - object->global_bbox_min[i]) / ((double) cuts[i]);
  //   }

  //   // Instead of using the MPI_COMM_WORLD rank, query the Cartesian communicator
  //   // to get our true coordinates in the topology.
  //   mftl_int_t my_cart_rank;
  //   mftl_int_t my_coords[3];
  //   MPI_Comm_rank(object->cart_comm, &my_cart_rank);  // Get my rank IN THE NEW COMM
  //   MPI_Cart_coords(object->cart_comm, my_cart_rank, 3, my_coords);  // Get my coords

  //   const mftl_int_t ix = my_coords[0];
  //   const mftl_int_t iy = my_coords[1];
  //   const mftl_int_t iz = my_coords[2];

  //   // Calculate the local domain using the correct coordinates
  //   object->local_bbox_min[0] = object->global_bbox_min[0] + dx[0] * ((double) ix);
  //   object->local_bbox_max[0] = object->global_bbox_min[0] + dx[0] * ((double) ix + 1);

  //   object->local_bbox_min[1] = object->global_bbox_min[1] + dx[1] * ((double) iy);
  //   object->local_bbox_max[1] = object->global_bbox_min[1] + dx[1] * ((double) iy + 1);

  //   object->local_bbox_min[2] = object->global_bbox_min[2] + dx[2] * ((double) iz);
  //   object->local_bbox_max[2] = object->global_bbox_min[2] + dx[2] * ((double) iz + 1);

  return MFTL_SUCCESS;
}

/**
 * @brief Frees all memory associated with an MFTL_Object mesh.
 *
 * This function cleans up all allocated data pools, the domain decomposition
 * information, and the MPI communicator before freeing the mesh struct itself.
 *
 * @param p_mesh A pointer to the object pointer (MFTL_Object**). This allows
 * the function to set the original pointer to NULL after freeing.
 */
void MFTLObjectDestroy(MFTL_Object** p_object)
{
  if (!p_object || !(*p_object))
  {
    return;
  }

  MFTL_Object* object = *p_object;

  // 1. Free the main data pools
  free(object->vertices);
  free(object->triangles);
  free(object->half_edges);

  kh_destroy(vertex_map, object->vertex_lookup);
  kh_destroy(triangle_map, object->triangle_lookup);
  kh_destroy(halfedge_map, object->halfedge_lookup);

  // 3. Free the duplicated MPI communicator
  if (object->comm != MPI_COMM_NULL)
  {
    MPI_Comm_free(&object->comm);
  }
  if (object->cart_comm != MPI_COMM_NULL)
  {
    MPI_Comm_free(&object->cart_comm);
  }
  if (object->graph_comm != MPI_COMM_NULL)
  {
    MPI_Comm_free(&object->graph_comm);
    free(object->neighbors);
  }

  // 4. Free the mesh struct itself
  free(object);

  //   MFTL_Free_mpi_types();

  // 5. Set the user's pointer to NULL to prevent dangling pointer access
  *p_object = NULL;
}

void MFTLUpdateObjectGlobalCounts(MFTL_Object** objects, mftl_int_t num_objects)
{
  if (num_objects == 0) return;

  // Use the communicator from the first object (assuming shared comm)
  MPI_Comm comm = objects[0]->cart_comm;
  mftl_int_t my_rank = objects[0]->rank;

  // 1. Allocate buffers for batch reduction
  // Layout: [Obj0_Tris, Obj0_Verts, Obj1_Tris, Obj1_Verts, ... ]
  mftl_int_t* local_counts = (mftl_int_t*) calloc(2 * num_objects, sizeof(mftl_int_t));
  mftl_int_t* global_counts = (mftl_int_t*) malloc(2 * num_objects * sizeof(mftl_int_t));

  // 2. Count Local Non-Ghost Entities
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];

    // Count Owned Triangles
    mftl_int_t t_count = 0;
    for (mftl_int_t i = 0; i < obj->n_triangles; ++i)
    {
      if (!obj->triangles[i].is_ghost)
      {
        t_count++;
      }
    }

    // Count Owned Vertices
    mftl_int_t v_count = 0;
    for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
    {
      if (!obj->vertices[i].is_ghost)
      {
        v_count++;
      }
    }

    // Store in buffer
    local_counts[2 * o + 0] = t_count;
    local_counts[2 * o + 1] = v_count;
  }

  // 3. Single Global Reduction (Sum)
  MPI_Allreduce(local_counts, global_counts, 2 * num_objects, MPI_MFTL_INT, MPI_SUM, comm);

  // 4. Unpack results back into the Objects
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    objects[o]->global_n_triangles = global_counts[2 * o + 0];
    objects[o]->global_n_vertices = global_counts[2 * o + 1];
  }

  // Cleanup
  free(local_counts);
  free(global_counts);
}

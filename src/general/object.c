/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:54:07
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 19:41:02
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mftl_core.h"
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
MFTL_Object* MFTLObjectCreate(int global_id)
{
  // Allocate zero-initialized struct
  MFTL_Object* object = (MFTL_Object*) calloc(1, sizeof(MFTL_Object));
  if (!object)
  {
    fprintf(stderr, "MFTL Error: Failed to allocate memory for MFTL_Object struct.\n");
    return MFTL_NULL;
  }

  object->global_id = global_id;

  // Initialize Domain to NULL
  object->domain = MFTL_NULL;

  // Initialize Arrays to NULL
  object->vertices = MFTL_NULL;
  object->triangles = MFTL_NULL;
  object->half_edges = MFTL_NULL;

  // Initialize Hash Maps
  object->vertex_lookup = kh_init(vertex_map);
  object->triangle_lookup = kh_init(triangle_map);
  object->halfedge_lookup = kh_init(halfedge_map);

  // Initialize Acceleration Pointers to NULL
  object->flat_tri_verts = MFTL_NULL;
  object->flat_tri_neighbors = MFTL_NULL;
  object->vert_offsets = MFTL_NULL;
  object->vert_tris = MFTL_NULL;

  // Initialize ID counters
  object->next_vertex_gid = 0;
  object->next_triangle_gid = 0;
  object->next_halfedge_gid = 0;

  // Note: Arrays (vertices, triangles, halfedges) are NULL until populated/reserved
  return object;
}

/**
 * @brief Initializes the opaque per-triangle user payload for an object.
 *
 * Sets object->data_size and allocates a zeroed blob of data_size bytes for
 * every existing triangle (local and ghost) that does not already carry one.
 * Must be called collectively with the SAME data_size on all ranks before
 * any redistribution/ghosting (see the contract on MFTL_Object.data_size).
 */
int MFTLInitializeObjectData(MFTL_Object* object, size_t data_size)
{
  if (!object) return MFTL_ERROR;

  if (data_size == object->data_size) return MFTL_SUCCESS;  // idempotent no-op

  if (object->data_size != 0)
  {
    fprintf(stderr,
            "MFTL Error: MFTLInitializeObjectData: object %d already has data_size=%zu; "
            "re-initializing with a different size (%zu) is not supported.\n",
            object->global_id, object->data_size, data_size);
    return MFTL_ERROR;
  }

  object->data_size = data_size;
  if (data_size == 0) return MFTL_SUCCESS;  // fast path: blobs stay NULL

  for (int i = 0; i < object->n_triangles; ++i)
  {
    if (object->triangles[i].data == MFTL_NULL)  // don't leak an existing blob
    {
      object->triangles[i].data = calloc(1, data_size);
      if (!object->triangles[i].data)
      {
        fprintf(stderr, "MFTL Error: Failed to allocate %zu bytes of triangle data.\n", data_size);
        return MFTL_ERROR;
      }
    }
  }
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
  if (!p_object || !(*p_object)) return;
  MFTL_Object* object = *p_object;

  // Free per-triangle user payloads (Stage 12)
  if (object->data_size > 0 && object->triangles)
  {
    for (int i = 0; i < object->n_triangles; ++i) free(object->triangles[i].data);
  }

  // Free Data Pools
  free(object->vertices);
  free(object->wind_gen);  // Stage 16B
  free(object->bcurve);  // Stage 16C
  free(object->triangles);
  free(object->half_edges);

  // Free Hash Maps
  kh_destroy(vertex_map, object->vertex_lookup);
  kh_destroy(triangle_map, object->triangle_lookup);
  kh_destroy(halfedge_map, object->halfedge_lookup);

  // Free Acceleration Structures
  free(object->flat_tri_verts);
  free(object->flat_tri_neighbors);
  free(object->vert_tris);
  free(object->vert_offsets);

  free(object);
  *p_object = MFTL_NULL;
}

/**
 * @brief Updates the global vertex and triangle counts for all objects in the ensemble.
 * * It sums up the locally owned (non-ghost) entities across all MPI ranks 
 * and stores the result in each object's global_n_* fields.
 * * @param ensemble The MFTL ensemble containing the domain and objects.
 */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble* ensemble)
{
  if (!ensemble) return;

  int num_objects = ensemble->num_objects;
  if (num_objects == 0) return;

  MPI_Comm comm = ensemble->domain.comm_world;

  // 1. Allocate buffers for batch reduction
  // Layout: [Obj0_Tris, Obj0_Verts, Obj1_Tris, Obj1_Verts, ... ]
  int* local_counts = (int*) calloc(2 * num_objects, sizeof(int));
  assert(local_counts);

  int* global_counts = (int*) malloc(2 * num_objects * sizeof(int));
  assert(global_counts);

  // 2. Count Local Non-Ghost Entities
  for (int o = 0; o < num_objects; ++o)
  {
    // Access object via the ensemble array
    MFTL_Object* obj = ensemble->objects[o];
    assert(obj);

    // Count Owned Triangles
    int t_count = 0;
    for (int i = 0; i < obj->n_triangles; ++i)
    {
      if (!obj->triangles[i].is_ghost)
      {
        t_count++;
      }
    }

    // Count Owned Vertices
    int v_count = 0;
    for (int i = 0; i < obj->n_vertices; ++i)
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
  // Assumes every rank has the same list of objects in the same order.
  MPI_Allreduce(local_counts, global_counts, 2 * num_objects, MPI_INT, MPI_SUM, comm);

  // 4. Unpack results back into the Objects
  int total_ensemble_tris = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = ensemble->objects[o];
    assert(obj);
    obj->global_n_triangles = global_counts[2 * o + 0];
    obj->global_n_vertices = global_counts[2 * o + 1];

    total_ensemble_tris += obj->global_n_triangles;
  }

  // Store the grand total in the ensemble struct
  ensemble->total_global_triangles = total_ensemble_tris;

  // Cleanup
  free(local_counts);
  free(global_counts);
}

/**
 * @brief Updates the global vertex and triangle counts for a SINGLE object.
 * * This is useful when you have modified only one object (e.g. during generation)
 * and don't want to pay the cost of reducing data for the entire ensemble.
 * * @param object The MFTL object to update. Must have a valid domain link.
 */
void MFTLObjectUpdateGlobalStats(MFTL_Object* object)
{
  // Safety checks
  assert(object);

  MFTL_Domain* domain = object->domain;
  assert(domain);

  // 1. Calculate Local Counts (Owned entities only)
  int t_count = 0;
  int v_count = 0;

  // Count Owned Triangles
  for (int i = 0; i < object->n_triangles; ++i)
  {
    if (!object->triangles[i].is_ghost)
    {
      t_count++;
    }
  }

  // Count Owned Vertices
  for (int i = 0; i < object->n_vertices; ++i)
  {
    if (!object->vertices[i].is_ghost)
    {
      v_count++;
    }
  }

  // Pack into buffer: [Triangles, Vertices]
  int local_counts[2];
  local_counts[0] = t_count;
  local_counts[1] = v_count;

  // 2. Global Reduction
  int global_counts[2];
  // We reduce 2 integers: one for tris, one for verts
  MPI_Allreduce(local_counts, global_counts, 2, MPI_INT, MPI_SUM, domain->comm_world);

  // 3. Store Results
  object->global_n_triangles = global_counts[0];
  object->global_n_vertices = global_counts[1];
}

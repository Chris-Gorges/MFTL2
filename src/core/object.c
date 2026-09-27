/*
 * MFTL2 core — object lifecycle (create/destroy), the opaque per-triangle payload and the
 * global entity counts.
 *
 * Global counts sum OWNED (non-ghost) entities only, so they are independent of the ghost
 * layers; they are collective over the domain's communicator.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mftl_core.h"
#include "mftl_constants.h"

/* Allocates an empty object (no entities, no domain link) with the given object id. */
MFTL_Object* MFTLObjectCreate(int global_id)
{
  MFTL_Object* object = (MFTL_Object*) calloc(1, sizeof(MFTL_Object));
  if (!object)
  {
    fprintf(stderr, "MFTL Error: Failed to allocate memory for MFTL_Object struct.\n");
    return MFTL_NULL;
  }

  object->global_id = global_id;

  object->domain = MFTL_NULL;

  object->vertices = MFTL_NULL;
  object->triangles = MFTL_NULL;
  object->half_edges = MFTL_NULL;

  object->vertex_lookup = kh_init(vertex_map);
  object->triangle_lookup = kh_init(triangle_map);
  object->halfedge_lookup = kh_init(halfedge_map);

  object->flat_tri_verts = MFTL_NULL;
  object->flat_tri_neighbors = MFTL_NULL;
  object->vert_offsets = MFTL_NULL;
  object->vert_tris = MFTL_NULL;

  object->next_vertex_gid = 0;
  object->next_triangle_gid = 0;
  object->next_halfedge_gid = 0;

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

/* Frees the entity arrays, payload blobs, hash maps and flattened connectivity of an object
   and sets *p_object to NULL. The domain is not owned by the object and is not touched. */
void MFTLObjectDestroy(MFTL_Object** p_object)
{
  if (!p_object || !(*p_object)) return;
  MFTL_Object* object = *p_object;

  if (object->data_size > 0 && object->triangles)
  {
    for (int i = 0; i < object->n_triangles; ++i) free(object->triangles[i].data);
  }

  free(object->vertices);
  free(object->wind_gen);
  free(object->bcurve);
  free(object->triangles);
  free(object->half_edges);

  kh_destroy(vertex_map, object->vertex_lookup);
  kh_destroy(triangle_map, object->triangle_lookup);
  kh_destroy(halfedge_map, object->halfedge_lookup);

  free(object->flat_tri_verts);
  free(object->flat_tri_neighbors);
  free(object->vert_tris);
  free(object->vert_offsets);

  free(object);
  *p_object = MFTL_NULL;
}

/* Sets global_n_triangles / global_n_vertices of every object (owned entities summed over all
   ranks, one batched Allreduce) and the ensemble's total_global_triangles. Collective; every
   rank must hold the same object list in the same order. */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble* ensemble)
{
  if (!ensemble) return;

  int num_objects = ensemble->num_objects;
  if (num_objects == 0) return;

  MPI_Comm comm = ensemble->domain.comm_world;

  // layout: [obj0 triangles, obj0 vertices, obj1 triangles, ...]
  int* local_counts = (int*) calloc(2 * num_objects, sizeof(int));
  assert(local_counts);

  int* global_counts = (int*) malloc(2 * num_objects * sizeof(int));
  assert(global_counts);

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = ensemble->objects[o];
    assert(obj);

    int t_count = 0;
    for (int i = 0; i < obj->n_triangles; ++i)
    {
      if (!obj->triangles[i].is_ghost)
      {
        t_count++;
      }
    }

    int v_count = 0;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if (!obj->vertices[i].is_ghost)
      {
        v_count++;
      }
    }

    local_counts[2 * o + 0] = t_count;
    local_counts[2 * o + 1] = v_count;
  }

  MPI_Allreduce(local_counts, global_counts, 2 * num_objects, MPI_INT, MPI_SUM, comm);

  int total_ensemble_tris = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = ensemble->objects[o];
    assert(obj);
    obj->global_n_triangles = global_counts[2 * o + 0];
    obj->global_n_vertices = global_counts[2 * o + 1];

    total_ensemble_tris += obj->global_n_triangles;
  }

  ensemble->total_global_triangles = total_ensemble_tris;

  free(local_counts);
  free(global_counts);
}

/* Single-object variant of MFTLEnsembleUpdateGlobalCounts (collective over the object's
   domain; the object must be linked to a domain). */
void MFTLObjectUpdateGlobalStats(MFTL_Object* object)
{
  assert(object);

  MFTL_Domain* domain = object->domain;
  assert(domain);

  int t_count = 0;
  int v_count = 0;

  for (int i = 0; i < object->n_triangles; ++i)
  {
    if (!object->triangles[i].is_ghost)
    {
      t_count++;
    }
  }

  for (int i = 0; i < object->n_vertices; ++i)
  {
    if (!object->vertices[i].is_ghost)
    {
      v_count++;
    }
  }

  int local_counts[2];
  local_counts[0] = t_count;
  local_counts[1] = v_count;

  int global_counts[2];
  MPI_Allreduce(local_counts, global_counts, 2, MPI_INT, MPI_SUM, domain->comm_world);

  object->global_n_triangles = global_counts[0];
  object->global_n_vertices = global_counts[1];
}

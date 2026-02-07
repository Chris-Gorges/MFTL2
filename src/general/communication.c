/*
 * @Author: Christian Gorges
 * @Date: 2025-08-19 10:50:05
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-21 16:48:34
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <mpi.h>
#include <khash.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_arrayfunctions.h"
#include "mftl_topology.h"
#include "mftl_memory.h"

// --- Global Variable Definitions ---
MPI_Datatype MPI_MFTL_INT;
MPI_Datatype MPI_PackedVertex;
MPI_Datatype MPI_PackedTriangle;
MPI_Datatype MPI_PackedHalfEdge;
MPI_Datatype MPI_PackedTriangleComplex;
MPI_Datatype MPI_GhostUpdatePayload;

void MFTLRegisterMpiTypes()
{
#ifdef MFTL_USE_64BIT
  MPI_MFTL_INT = MPI_INT64_T;
#else
  MPI_MFTL_INT = MPI_INT32_T;
#endif

  // --- 1. Create MPI Datatype for PackedVertex ---
  {
#define NITEMS_VERTEX 11
    int blocklengths[] = {1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_MFTL_INT,  // global_id (was MPI_UINT64_T)
        MPI_MFTL_INT,  // owner_proc (was MPI_INT - likely wrong if owner_proc > 2B or if mftl_int_t is 64bit)
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
        MPI_C_BOOL,  // is_ghost
        MPI_MFTL_INT,  // flags (was MPI_INT)
        MPI_MFTL_INT,  // id_he_out (was MPI_UINT64_T)
        MPI_MFTL_INT  // body_id (was MPI_UINT64_T)
    };

    MPI_Aint offsets[NITEMS_VERTEX];
    offsets[0] = offsetof(PackedVertex, global_id);
    offsets[1] = offsetof(PackedVertex, owner_proc);
    offsets[2] = offsetof(PackedVertex, pos);
    offsets[3] = offsetof(PackedVertex, normal);
    offsets[4] = offsetof(PackedVertex, meancurvature);
    offsets[5] = offsetof(PackedVertex, gaussiancurvature);
    offsets[6] = offsetof(PackedVertex, area);
    offsets[7] = offsetof(PackedVertex, is_ghost);
    offsets[8] = offsetof(PackedVertex, flags);
    offsets[9] = offsetof(PackedVertex, id_he_out);
    offsets[10] = offsetof(PackedVertex, body_id);

    MPI_Type_create_struct(NITEMS_VERTEX, blocklengths, offsets, types, &MPI_PackedVertex);
    MPI_Type_commit(&MPI_PackedVertex);
  }

  // --- 2. Create MPI Datatype for PackedTriangle ---
  {
#define NITEMS_TRIANGLE 9
    int blocklengths[] = {1, 1, 1, 3, 1, 3, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_MFTL_INT,  // global_id
        MPI_MFTL_INT,  // owner_proc
        MPI_MFTL_INT,  // he_id
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // center[3]
        MPI_C_BOOL,  // is_ghost
        MPI_MFTL_INT,  // flags
        MPI_MFTL_INT  // body_id
    };

    MPI_Aint offsets[NITEMS_TRIANGLE];
    offsets[0] = offsetof(PackedTriangle, global_id);
    offsets[1] = offsetof(PackedTriangle, owner_proc);
    offsets[2] = offsetof(PackedTriangle, he_id);
    offsets[3] = offsetof(PackedTriangle, normal);
    offsets[4] = offsetof(PackedTriangle, area);
    offsets[5] = offsetof(PackedTriangle, center);
    offsets[6] = offsetof(PackedTriangle, is_ghost);
    offsets[7] = offsetof(PackedTriangle, flags);
    offsets[8] = offsetof(PackedTriangle, body_id);

    MPI_Type_create_struct(NITEMS_TRIANGLE, blocklengths, offsets, types, &MPI_PackedTriangle);
    MPI_Type_commit(&MPI_PackedTriangle);
  }

  // --- 3. Create MPI Datatype for PackedHalfEdge ---
  {
#define NITEMS_HALFEDGE 7
    int blocklengths[] = {1, 1, 1, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_MFTL_INT,  // global_id
        MPI_MFTL_INT,  // vertex_gid
        MPI_MFTL_INT,  // twin_gid
        MPI_MFTL_INT,  // next_gid
        MPI_MFTL_INT,  // triangle_gid
        MPI_C_BOOL,  // boundary
        MPI_MFTL_INT  // flags
    };

    MPI_Aint offsets[NITEMS_HALFEDGE];
    offsets[0] = offsetof(PackedHalfEdge, global_id);
    offsets[1] = offsetof(PackedHalfEdge, vertex_gid);
    offsets[2] = offsetof(PackedHalfEdge, twin_gid);
    offsets[3] = offsetof(PackedHalfEdge, next_gid);
    offsets[4] = offsetof(PackedHalfEdge, triangle_gid);
    offsets[5] = offsetof(PackedHalfEdge, boundary);
    offsets[6] = offsetof(PackedHalfEdge, flags);

    MPI_Type_create_struct(NITEMS_HALFEDGE, blocklengths, offsets, types, &MPI_PackedHalfEdge);
    MPI_Type_commit(&MPI_PackedHalfEdge);
  }

  // --- 4. PackedTriangleComplex (Nested) ---
  // Your implementation using MPI_Get_address is EXCELLENT and correct.
  // It handles padding between structs perfectly. Keep it as is.
  {
#define NITEMS_COMPLEX 2
    int blocklengths[] = {1, 3};
    MPI_Datatype types[] = {MPI_PackedTriangle, MPI_PackedHalfEdge};

    MPI_Aint offsets[NITEMS_COMPLEX];
    PackedTriangleComplex dummy;
    MPI_Aint base_addr, tri_addr, he_addr;

    MPI_Get_address(&dummy, &base_addr);
    MPI_Get_address(&dummy.triangle, &tri_addr);
    MPI_Get_address(&dummy.half_edges, &he_addr);

    offsets[0] = MPI_Aint_diff(tri_addr, base_addr);
    offsets[1] = MPI_Aint_diff(he_addr, base_addr);

    MPI_Type_create_struct(NITEMS_COMPLEX, blocklengths, offsets, types, &MPI_PackedTriangleComplex);
    MPI_Type_commit(&MPI_PackedTriangleComplex);
  }

  // --- 5. Create MPI Datatype for MFTL_GhostUpdatePayload ---
  {
#define NITEMS_GHOST 6
    int blocklengths[] = {1, 3, 3, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_MFTL_INT,  // global_id
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
    };

    MPI_Aint offsets[NITEMS_GHOST];
    offsets[0] = offsetof(GhostUpdatePayload, global_id);
    offsets[1] = offsetof(GhostUpdatePayload, pos);
    offsets[2] = offsetof(GhostUpdatePayload, normal);
    offsets[3] = offsetof(GhostUpdatePayload, meancurvature);
    offsets[4] = offsetof(GhostUpdatePayload, gaussiancurvature);
    offsets[5] = offsetof(GhostUpdatePayload, area);

    MPI_Type_create_struct(NITEMS_GHOST, blocklengths, offsets, types, &MPI_GhostUpdatePayload);
    MPI_Type_commit(&MPI_GhostUpdatePayload);
  }
}

void MFTLFreeMpiTypes()
{
  MPI_Type_free(&MPI_PackedVertex);
  MPI_Type_free(&MPI_PackedTriangle);
  MPI_Type_free(&MPI_PackedHalfEdge);
  MPI_Type_free(&MPI_PackedTriangleComplex);
  MPI_Type_free(&MPI_GhostUpdatePayload);
}

typedef struct
{
  mftl_int_t* data;
  mftl_int_t size;
  mftl_int_t capacity;
} IntBuffer;

// Initialize once
void buffer_init(IntBuffer* buf, mftl_int_t initial_cap)
{
  buf->data = (mftl_int_t*) malloc(initial_cap * sizeof(mftl_int_t));
  buf->size = 0;
  buf->capacity = initial_cap;
}

// Fast append with geometric growth
static inline void buffer_add(IntBuffer* buf, mftl_int_t val)
{
  MFTL_ENSURE_CAPACITY(buf->data, &buf->size, &buf->capacity, 1, mftl_int_t);
  buf->data[buf->size++] = val;
}

// Reset without freeing memory
static inline void buffer_reset(IntBuffer* buf) { buf->size = 0; }

/**
 * @brief Calculates robust initial buffer capacities based on mesh density.
 */
static inline void MFTLCalcBufferCapacities(mftl_int_t n_tris, mftl_int_t n_verts, mftl_int_t nproc, mftl_int_t* out_tri_cap, mftl_int_t* out_vert_cap)
{
  double factor = 0.2;  // Safety factor: assume up to 20% of data might move to one rank
  double avg_tris = (double) n_tris / (double) nproc;
  double avg_verts = (double) n_verts / (double) nproc;

  *out_tri_cap = (mftl_int_t) (avg_tris * factor);
  *out_vert_cap = (mftl_int_t) (avg_verts * factor);

  // Clamp to reasonable range [1024, 1MB] to prevent tiny allocs or massive over-allocs
  *out_tri_cap = MFTL_MIN(MFTL_MAX(1024, *out_tri_cap), 1000000);
  *out_vert_cap = MFTL_MIN(MFTL_MAX(1024, *out_vert_cap), 1000000);
}

/**
 * @brief Packs a single triangle and its half-edges into the transfer struct.
 */
static inline void MFTLPackSingleTriangle(MFTL_Object* obj, mftl_int_t tri_idx, PackedTriangleComplex* dst, bool is_ghost)
{
  MFTL_Triangle* src = &obj->triangles[tri_idx];

  // 1. Pack Triangle Data
  dst->triangle.global_id = src->global_id;
  dst->triangle.owner_proc = src->owner_proc;
  dst->triangle.he_id = MFTLTHe(obj, src)->global_id;
  dst->triangle.area = src->area;
  dst->triangle.is_ghost = is_ghost ? MFTL_TRUE : MFTL_FALSE;
  dst->triangle.flags = src->flags;
  dst->triangle.body_id = obj->global_id;
  memcpy(dst->triangle.normal, src->normal, 3 * sizeof(double));
  memcpy(dst->triangle.center, src->center, 3 * sizeof(double));

  // 2. Pack the 3 Half-Edges
  MFTL_HalfEdge* t_he[3];
  MFTLGetTriangleHalfedges(obj, src, t_he);

  for (int j = 0; j < 3; ++j)
  {
    dst->half_edges[j].global_id = t_he[j]->global_id;
    dst->half_edges[j].vertex_gid = MFTLHeTarget(obj, t_he[j])->global_id;
    // If twin exists (not boundary), save twin's GID. Else -1.
    dst->half_edges[j].twin_gid = (t_he[j]->boundary == MFTL_FALSE) ? MFTLHeTwin(obj, t_he[j])->global_id : -1;
    dst->half_edges[j].next_gid = MFTLHeNext(obj, t_he[j])->global_id;
    dst->half_edges[j].triangle_gid = MFTLHeTriangle(obj, t_he[j])->global_id;
    dst->half_edges[j].boundary = MFTL_TRUE;  // Will be re-evaluated at destination
    dst->half_edges[j].flags = t_he[j]->flags;
  }
}

/**
 * @brief Packs a single vertex into the transfer struct.
 */
static inline void MFTLPackSingleVertex(MFTL_Object* obj, mftl_int_t vert_idx, PackedVertex* dst, bool is_ghost)
{
  MFTL_Vertex* src = &obj->vertices[vert_idx];

  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->meancurvature = src->meancurvature;
  dst->gaussiancurvature = src->gaussiancurvature;
  dst->area = src->area;
  dst->is_ghost = is_ghost ? MFTL_TRUE : MFTL_FALSE;
  dst->flags = src->flags;
  dst->id_he_out = MFTLVOutHe(obj, src)->global_id;
  dst->body_id = obj->global_id;

  memcpy(dst->pos, src->pos, 3 * sizeof(double));
  memcpy(dst->normal, src->normal, 3 * sizeof(double));
}

/**
 * @brief Wipes an MFTL_Object clean to prepare for reconstruction.
 */
static inline void MFTLResetObjectData(MFTL_Object* obj)
{
  free(obj->triangles);
  free(obj->vertices);
  free(obj->half_edges);

  obj->triangles = MFTL_NULL;
  obj->vertices = MFTL_NULL;
  obj->half_edges = MFTL_NULL;

  obj->n_triangles = 0;
  obj->n_vertices = 0;
  obj->n_half_edges = 0;
  obj->capacity_triangles = 0;
  obj->capacity_vertices = 0;
  obj->capacity_half_edges = 0;

  // Clear Maps
  kh_clear(triangle_map, obj->triangle_lookup);
  kh_clear(halfedge_map, obj->halfedge_lookup);
  kh_clear(vertex_map, obj->vertex_lookup);
}

/**
 * @brief Generic driver for exchanging variable-length data via MPI.
 * Handles both "Global All-to-All" and "Neighbor Sparse" paths.
 * MPI strictly works with 32 bit integers, so we stick with int in this function.
 * @param ref_obj Reference object for MPI communicator and process count.
 * @param send_bufs Array of send buffers.
 * @param type MPI datatype of the data to be sent.
 * @param type_size Size of the MPI datatype.
 * @param use_global_path Whether to use the global path.
 * @param out_recv_buf Pointer to the received buffer.
 * @param out_recv_count Pointer to the received count.
 */
static void MFTLMpiExchangeBuffers(MFTL_Domain* domain, MFTL_SendBuffer* send_bufs, MPI_Datatype type, size_t type_size, bool use_global_path,
                                   void** out_recv_buf, int* out_recv_count)
{
  int nproc = (int) domain->nproc;
  *out_recv_count = 0;

  if (use_global_path)
  {
    // --- GLOBAL PATH ---
    // printf("Global Path\n");

    // 1. Setup Arrays
    int* send_counts = calloc(nproc, sizeof(int));
    int* recv_counts = malloc(nproc * sizeof(int));
    int* send_displs = calloc(nproc + 1, sizeof(int));
    int* recv_displs = calloc(nproc + 1, sizeof(int));

    // 2. Exchange Counts
    for (int r = 0; r < nproc; ++r) send_counts[r] = send_bufs[r].count;
    MPI_Alltoall(send_counts, 1, MPI_MFTL_INT, recv_counts, 1, MPI_MFTL_INT, domain->comm_cart);

    // 3. Calculate Displacements
    int total_send = 0;
    for (int i = 0; i < nproc; ++i)
    {
      total_send += send_counts[i];
      *out_recv_count += recv_counts[i];
      send_displs[i + 1] = send_displs[i] + send_counts[i];
      recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    }

    // 4. Pack Linear Buffer (Rank Ordered)
    char* flat_send_buf = malloc(total_send * type_size);
    for (int r = 0; r < nproc; ++r)
    {
      if (send_counts[r] > 0)
      {
        memcpy(flat_send_buf + (send_displs[r] * type_size), send_bufs[r].buffer, send_counts[r] * type_size);
      }
    }

    // 5. Execute Exchange
    *out_recv_buf = malloc((*out_recv_count) * type_size);
    MPI_Alltoallv(flat_send_buf, send_counts, send_displs, type, *out_recv_buf, recv_counts, recv_displs, type, domain->comm_cart);

    // Cleanup
    free(flat_send_buf);
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
  }
  else
  {
    // --- NEIGHBOR PATH ---
    // printf("Neighbor Path\n");

    int num_neighbors = domain->num_neighbors;

    // 1. Setup Arrays (Sized to Neighbors, not NPROC)
    int* n_send = calloc(num_neighbors, sizeof(int));
    int* n_recv = malloc(num_neighbors * sizeof(int));
    int* s_displs = calloc(num_neighbors + 1, sizeof(int));
    int* r_displs = calloc(num_neighbors + 1, sizeof(int));

    // 2. Exchange Counts
    for (int i = 0; i < num_neighbors; ++i)
    {
      n_send[i] = send_bufs[domain->neighbors[i]].count;
    }
    MPI_Neighbor_alltoall(n_send, 1, MPI_MFTL_INT, n_recv, 1, MPI_MFTL_INT, domain->comm_graph);

    // 3. Calculate Displacements
    int total_send = 0;
    for (int i = 0; i < num_neighbors; ++i)
    {
      total_send += n_send[i];
      *out_recv_count += n_recv[i];
      s_displs[i + 1] = s_displs[i] + n_send[i];
      r_displs[i + 1] = r_displs[i] + n_recv[i];
    }

    // 4. Pack Linear Buffer (Neighbor Ordered)
    char* flat_send_buf = malloc(total_send * type_size);
    for (int i = 0; i < num_neighbors; ++i)
    {
      if (n_send[i] > 0)
      {
        // Map Neighbor Index -> Global Rank -> Buffer
        int target_rank = domain->neighbors[i];
        memcpy(flat_send_buf + (s_displs[i] * type_size), send_bufs[target_rank].buffer, n_send[i] * type_size);
      }
    }

    // 5. Execute Exchange
    *out_recv_buf = malloc((*out_recv_count) * type_size);
    MPI_Neighbor_alltoallv(flat_send_buf, n_send, s_displs, type, *out_recv_buf, n_recv, r_displs, type, domain->comm_graph);

    // Cleanup
    free(flat_send_buf);
    free(n_send);
    free(n_recv);
    free(s_displs);
    free(r_displs);
  }
}

/**
 * @brief Reorders the triangle array in-place based on the sorted indices.
 */
static inline void permute_triangles(PackedTriangleComplex* array, const mftl_int_t* indices, mftl_int_t count)
{
  assert(count > 0);

  // 1. Allocate temporary buffer
  PackedTriangleComplex* tmp = (PackedTriangleComplex*) malloc(count * sizeof(PackedTriangleComplex));
  assert(tmp != MFTL_NULL);  // Safety check

  // 2. Gather data into temp buffer in sorted order
  // Random Read (from array) -> Sequential Write (to tmp)
  for (mftl_int_t i = 0; i < count; ++i)
  {
    tmp[i] = array[indices[i]];
  }

  // 3. Block copy back to original array
  // Sequential Read (from tmp) -> Sequential Write (to array)
  memcpy(array, tmp, count * sizeof(PackedTriangleComplex));

  // 4. Cleanup
  free(tmp);
}

/**
 * @brief Reorders the vertex array in-place based on the sorted indices.
 */
static inline void permute_vertices(PackedVertex* array, const mftl_int_t* indices, mftl_int_t count)
{
  assert(count > 0);

  PackedVertex* tmp = (PackedVertex*) malloc(count * sizeof(PackedVertex));
  assert(tmp != MFTL_NULL);  // Safety check

  for (mftl_int_t i = 0; i < count; ++i)
  {
    tmp[i] = array[indices[i]];
  }

  memcpy(array, tmp, count * sizeof(PackedVertex));

  free(tmp);
}

/**
 * @brief Intelligently redistributes the objects using the fastest available communication pattern.
 *
 * @details This function implements a robust, multi-phase redistribution algorithm.
 *          It currently assumes that all objects in the list of objects share the same MPI communication pattern.
 *
 * 1.  SETUP & ANALYSIS: It begins by querying the MPI graph communicator to get a
 * list of direct neighbors. It then iterates through all local triangles,
 * determines their correct owner, and checks if the owner is a direct neighbor.
 *
 * 2.  DECISION: Based on the analysis, a global consensus is reached using
 * MPI_Allreduce. If all data movement is between direct neighbors, the function
 * selects a highly-optimized "fast path". Otherwise, it falls back to a
 * robust "global path".
 *
 * 3.  COMMUNICATION:
 * - Fast Path: Uses MPI_Neighbor_alltoallv for efficient, localized data exchange.
 * - Global Path: Uses MPI_Alltoallv to handle arbitrary data exchange between any
 * two processes in the simulation.
 *
 * 4.  REBUILDING: Each rank unpacks the received data, combines it with the
 * triangles it kept, de-duplicates vertices by their unique global IDs,
 * and reconstructs its new local portion of the mesh.
 *
 * @param object The MFTL object containing the mesh and MPI info.
 * @param base The geometric basis for determining ownership (VERTEX_BASED or CENTER_BASED).
 * @return 0 on success, -1 on failure.
 * 
 * @todo incorporate void data pointer to triangle communication & ghost layer
 */
mftl_int_t MFTLRedistributeListOfObjects(MFTL_Ensemble* ensemble)
{
  if (ensemble->num_objects == 0 || ensemble->objects == MFTL_NULL) return MFTL_ERROR;

  MFTL_Domain* domain = &ensemble->domain;
  mftl_int_t num_objects = ensemble->num_objects;
  MFTL_Object** objects = ensemble->objects;

  if (domain->nproc < 2) return MFTL_SUCCESS;  // No redistribution needed if there is only one process.

  // --- Phase 0: Setup and MPI Queries ---
  int nproc = domain->nproc;
  int my_rank = domain->rank;
  mftl_int_t requires_global_comm_local = 0;

  // Create a fast O(1) lookup map locally
  char* is_neighbor_map = (char*) calloc(nproc, sizeof(char));
  for (mftl_int_t i = 0; i < domain->num_neighbors; ++i)
  {
    is_neighbor_map[domain->neighbors[i]] = 1;
  }

  // 1. Transient Buffers: Used for sorting/uniquing current object's data
  // We need separate lists for Triangles and Vertices
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* master_buf = calloc(nproc, sizeof(IntBuffer));

  mftl_int_t* total_tris_all_objects = calloc(nproc, sizeof(mftl_int_t));
  mftl_int_t* total_verts_all_objects = calloc(nproc, sizeof(mftl_int_t));

  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));

  // Pre-allocate to avoid early reallocs (Heuristic: 1024 ints)
  mftl_int_t total_global_tris = 0;
  mftl_int_t total_global_verts = 0;
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    total_global_tris += object->global_n_triangles;
    total_global_verts += object->global_n_vertices;
  }

  mftl_int_t init_tri_cap, init_vert_cap;
  MFTLCalcBufferCapacities(total_global_tris, total_global_verts, nproc, &init_tri_cap, &init_vert_cap);

  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    buffer_init(&trans_tri_bufs[r], init_tri_cap);  // Transient Triangles
    buffer_init(&trans_vert_bufs[r], init_vert_cap);  // Transient Vertices
    buffer_init(&master_buf[r], init_tri_cap + init_vert_cap);  // Master Buffer
  }

  // --- Phase 1: Packing and Analysis ---

  // Store owners in simple flat arrays, not inside structs scattered in memory
  // Do this outside the loop to avoid heap thrashing and realloc if size is not enough.
  mftl_int_t last_size_tris = objects[0]->n_triangles;
  mftl_int_t last_size_verts = objects[0]->n_vertices;
  mftl_int_t* vert_owners = malloc(last_size_verts * sizeof(mftl_int_t));
  mftl_int_t* tri_owners = malloc(last_size_tris * sizeof(mftl_int_t));

  // Dummy variable: We always start writing from index 0 for each object.
  // We use this to tell the macro "The buffer is effectively empty right now".
  mftl_int_t zero_count = 0;

  // 1a. Calculate destinations
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    MFTL_Triangle* tri_base = object->triangles;
    MFTL_Vertex* vert_base = object->vertices;
    mftl_int_t n_triangles = object->n_triangles;
    mftl_int_t n_vertices = object->n_vertices;

    // A. RESET Transient Buffers
    for (mftl_int_t r = 0; r < nproc; ++r)
    {
      buffer_reset(&trans_tri_bufs[r]);
      buffer_reset(&trans_vert_bufs[r]);
    }

    MFTL_ENSURE_CAPACITY(tri_owners, &zero_count, &last_size_tris, n_triangles, mftl_int_t);
    MFTL_ENSURE_CAPACITY(vert_owners, &zero_count, &last_size_verts, n_vertices, mftl_int_t);

    last_size_tris = n_triangles;
    last_size_verts = n_vertices;

    // Ensure topology is built. If NULL, build it now (Lazy Init).
    if (!object->flat_tri_verts || !object->flat_tri_neighbors)
    {
      MFTLFlattenConnectivity(object);
    }

    // Direct pointers to object memory (Zero Copy)
    mftl_int_t* flat_tri_verts = object->flat_tri_verts;
    mftl_int_t* flat_tri_neighbors = object->flat_tri_neighbors;

    // --- Pre-calculate Vertex Owners ---
    for (mftl_int_t i = 0; i < n_vertices; ++i)
    {
      MFTL_Vertex* v = &vert_base[i];
      v->owner_proc = MFTLGetOwnerProc(domain, v->pos);
      vert_owners[i] = v->owner_proc;
    }

    // --- Pre-calculate Triangle Owners ---
    for (mftl_int_t i = 0; i < n_triangles; ++i)
    {
      MFTL_Triangle* tri = &tri_base[i];
      tri->owner_proc = MFTLGetOwnerProc(domain, tri->center);
      tri_owners[i] = tri->owner_proc;
    }

    for (mftl_int_t i = 0; i < n_triangles; ++i)
    {
      // This prevents "Ghost Forwarding" crashes and automatically cleans up stale ghosts.
      // Don't use owner_proc because it already contains the new owner.
      if (tri_base[i].is_ghost)
      {
        continue;
      }

      mftl_int_t distinct_ranks[7];
      mftl_int_t count = 0;

      // --- 1. Who owns me? ---
      distinct_ranks[count++] = tri_owners[i];

      // --- 2. Who owns my vertices? ---
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 1]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 2]];

      // --- 3. Who owns my neighbors? ---
      for (mftl_int_t j = 0; j < 3; ++j)
      {
        mftl_int_t adj_idx = flat_tri_neighbors[3 * i + j];

        if (adj_idx != -1)
        {
          distinct_ranks[count++] = tri_owners[adj_idx];
        }
      }

      mftl_int_t unique_dests[7];
      mftl_int_t num_unique_dests = MFTLGetUniqueIntegers(distinct_ranks, count, unique_dests, 7);

      // Loop through the unique destinations
      for (mftl_int_t k = 0; k < num_unique_dests; ++k)
      {
        mftl_int_t dest_rank = unique_dests[k];
        if (dest_rank != my_rank && !is_neighbor_map[dest_rank])
        {
          requires_global_comm_local = 1;
        }

        // Blind Fill
        // Add Triangle ID
        buffer_add(&trans_tri_bufs[dest_rank], i);

        // Add Vertex IDs
        buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 0]);
        buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 1]);
        buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 2]);
      }
    }

    // Sort -> Unique
    for (mftl_int_t r = 0; r < nproc; ++r)
    {
      // Skip if nothing to send to this rank
      if (trans_tri_bufs[r].size == 0)
      {
        // We must still signal "0 items" for this object to keep alignment
        // if the receiver expects a stream of objects.
        // Format: [ObjID][NumTris][NumVerts]...
        buffer_add(&master_buf[r], object->global_id);
        buffer_add(&master_buf[r], 0);  // 0 Triangles
        buffer_add(&master_buf[r], 0);  // 0 Vertices
        continue;
      }
      // 1. Sort & Unique Triangles
      MFTLSortIds(trans_tri_bufs[r].data, trans_tri_bufs[r].size);
      mftl_int_t n_unique_tris = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      // 2. Sort & Unique Vertices
      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      mftl_int_t n_unique_verts = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      // 3. Append to master buf ({object_id, n_unique_tris, n_unique_verts, triangle_ids, vertex_ids})
      total_tris_all_objects[r] += n_unique_tris;
      total_verts_all_objects[r] += n_unique_verts;
      buffer_add(&master_buf[r], object->global_id);
      buffer_add(&master_buf[r], n_unique_tris);
      buffer_add(&master_buf[r], n_unique_verts);
      for (mftl_int_t i = 0; i < n_unique_tris; ++i)
      {
        buffer_add(&master_buf[r], trans_tri_bufs[r].data[i]);
      }
      for (mftl_int_t i = 0; i < n_unique_verts; ++i)
      {
        buffer_add(&master_buf[r], trans_vert_bufs[r].data[i]);
      }
    }
  }

  free(vert_owners);
  free(tri_owners);

  // Allocate send buffers
  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    send_tri_bufs[r].buffer = malloc(total_tris_all_objects[r] * sizeof(PackedTriangleComplex));
    send_tri_bufs[r].capacity = total_tris_all_objects[r];
    send_tri_bufs[r].count = 0;  // Initialize packing index to 0

    send_vert_bufs[r].buffer = malloc(total_verts_all_objects[r] * sizeof(PackedVertex));
    send_vert_bufs[r].capacity = total_verts_all_objects[r];
    send_vert_bufs[r].count = 0;  // Initialize packing index to 0
  }

  // Pack MFTL Send Buffers with help of master buffer
  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    mftl_int_t master_idx = 0;

    // Safety casts for correct pointer arithmetic
    PackedTriangleComplex* tri_buf_typed = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
    PackedVertex* vert_buf_typed = (PackedVertex*) send_vert_bufs[r].buffer;

    for (mftl_int_t o = 0; o < num_objects; ++o)
    {
      MFTL_Object* object = objects[o];
      MFTL_Triangle* tri_base = object->triangles;
      MFTL_Vertex* vert_base = object->vertices;

      mftl_int_t object_id = master_buf[r].data[master_idx++];
      assert(object->global_id == object_id);

      mftl_int_t n_unique_tris = master_buf[r].data[master_idx++];
      mftl_int_t n_unique_verts = master_buf[r].data[master_idx++];

      // 1. Pack Triangles
      for (mftl_int_t i = master_idx; i < master_idx + n_unique_tris; ++i)
      {
        mftl_int_t tri_idx = master_buf[r].data[i];

        // FIX: Use typed pointer for indexing
        PackedTriangleComplex* packed_tri = &tri_buf_typed[send_tri_bufs[r].count++];

        packed_tri->triangle.global_id = tri_base[tri_idx].global_id;
        packed_tri->triangle.owner_proc = tri_base[tri_idx].owner_proc;
        packed_tri->triangle.he_id = MFTLTHe(object, &tri_base[tri_idx])->global_id;
        memcpy(packed_tri->triangle.normal, tri_base[tri_idx].normal, 3 * sizeof(double));
        packed_tri->triangle.area = tri_base[tri_idx].area;
        memcpy(packed_tri->triangle.center, tri_base[tri_idx].center, 3 * sizeof(double));
        packed_tri->triangle.is_ghost = MFTL_FALSE;
        packed_tri->triangle.flags = tri_base[tri_idx].flags;
        packed_tri->triangle.body_id = object->global_id;

        MFTL_HalfEdge* t_he[3];
        MFTLGetTriangleHalfedges(object, &tri_base[tri_idx], t_he);
        for (mftl_int_t j = 0; j < 3; ++j)
        {
          PackedHalfEdge* he = &packed_tri->half_edges[j];
          he->global_id = t_he[j]->global_id;
          he->vertex_gid = MFTLHeTarget(object, t_he[j])->global_id;
          he->twin_gid = (t_he[j]->boundary == MFTL_FALSE) ? MFTLHeTwin(object, t_he[j])->global_id : -1;
          he->next_gid = MFTLHeNext(object, t_he[j])->global_id;
          he->triangle_gid = MFTLHeTriangle(object, t_he[j])->global_id;
          he->boundary = MFTL_TRUE;
          he->flags = t_he[j]->flags;
        }
      }

      // 2. Pack Vertices
      for (mftl_int_t i = master_idx + n_unique_tris; i < master_idx + n_unique_tris + n_unique_verts; ++i)
      {
        mftl_int_t vert_idx = master_buf[r].data[i];

        // FIX: Use typed pointer for indexing
        PackedVertex* packed_vert = &vert_buf_typed[send_vert_bufs[r].count++];

        packed_vert->global_id = vert_base[vert_idx].global_id;
        packed_vert->owner_proc = vert_base[vert_idx].owner_proc;
        memcpy(packed_vert->pos, vert_base[vert_idx].pos, 3 * sizeof(double));
        memcpy(packed_vert->normal, vert_base[vert_idx].normal, 3 * sizeof(double));
        packed_vert->meancurvature = vert_base[vert_idx].meancurvature;
        packed_vert->gaussiancurvature = vert_base[vert_idx].gaussiancurvature;
        packed_vert->area = vert_base[vert_idx].area;
        packed_vert->is_ghost = MFTL_FALSE;
        packed_vert->flags = vert_base[vert_idx].flags;
        packed_vert->id_he_out = MFTLVOutHe(object, &vert_base[vert_idx])->global_id;
        packed_vert->body_id = object->global_id;
      }

      master_idx += n_unique_tris + n_unique_verts;
    }
  }

  // Cleanup Transient Buffers
  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    // Assuming you have a buffer_free or similar helper:
    free(trans_tri_bufs[r].data);
    free(trans_vert_bufs[r].data);
    free(master_buf[r].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);
  free(master_buf);

  // --- Before Phase 2: Separate Kept data from Sent data ---

  // These will hold the data that stays on the current rank.
  PackedTriangleComplex* kept_tris = MFTL_NULL;
  PackedVertex* kept_verts = MFTL_NULL;
  mftl_int_t kept_count_tris = 0, kept_count_verts = 0;

  // Look for our own rank in the triangle send buffer
  // Do the same for vertices
  int own_rank = domain->rank;
  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    if (r == own_rank)
    {
      // We can "steal" the buffer pointer directly instead of copying.
      kept_tris = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      kept_verts = (PackedVertex*) send_vert_bufs[r].buffer;
      kept_count_tris = send_tri_bufs[r].count;
      kept_count_verts = send_vert_bufs[r].count;

      // Clear Triangles
      send_tri_bufs[r].buffer = MFTL_NULL;
      send_tri_bufs[r].count = 0;
      send_tri_bufs[r].capacity = 0;

      // Clear Vertices
      send_vert_bufs[r].buffer = MFTL_NULL;
      send_vert_bufs[r].count = 0;
      send_vert_bufs[r].capacity = 0;
    }
  }

  // --- Phase 2: Decide on Communication Strategy ---
  mftl_int_t global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  // --- Phase 3: Execute Communication ---
  PackedTriangleComplex* recv_buf_tris = MFTL_NULL;
  PackedVertex* recv_buf_verts = MFTL_NULL;
  mftl_int_t total_recv_tris = 0, total_recv_verts = 0;

  // Use the Generic MPI Exchange Helper
  MFTLMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_buf_tris,
                         &total_recv_tris);

  MFTLMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_buf_verts, &total_recv_verts);

  // --- Final Cleanup of Send Buffers ---
  // At this point, all data has been sent. We can free the send maps and the buffers they hold.
  for (mftl_int_t i = 0; i < nproc; ++i)
  {
    free(send_tri_bufs[i].buffer);
    free(send_vert_bufs[i].buffer);
  }

  free(total_tris_all_objects);
  free(total_verts_all_objects);
  free(send_tri_bufs);
  free(send_vert_bufs);
  free(is_neighbor_map);

  // --- Phase 4: Unpack and Rebuild Local Mesh ---

  // Pre-wipe all objects. We are about to rebuild them.
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    MFTLResetObjectData(obj);
  }

  // =========================================================================
  // PART A: TRIANGLE RECONSTRUCTION
  // =========================================================================

  // 4.1. Combine kept and received tris
  mftl_int_t final_tris_count = kept_count_tris + total_recv_tris;
  PackedTriangleComplex* all_tris = MFTL_NULL;

  if (final_tris_count > 0)
  {
    all_tris = (PackedTriangleComplex*) malloc(final_tris_count * sizeof(PackedTriangleComplex));

    // Copy Kept
    if (kept_count_tris > 0) memcpy(all_tris, kept_tris, kept_count_tris * sizeof(PackedTriangleComplex));

    // Copy Recv
    if (total_recv_tris > 0) memcpy(all_tris + kept_count_tris, recv_buf_tris, total_recv_tris * sizeof(PackedTriangleComplex));
  }

  free(kept_tris);
  free(recv_buf_tris);

  // 4.2. HIERARCHICAL RADIX SORT & REBUILD
  if (final_tris_count > 0)
  {
    // =========================================================
    // PASS 1: GROUP BY BODY_ID
    // =========================================================

    // 1. Allocate Key/Index arrays for the whole batch
    mftl_int_t* keys = malloc(final_tris_count * sizeof(mftl_int_t));
    mftl_int_t* indices = malloc(final_tris_count * sizeof(mftl_int_t));

    // 2. Extract Body IDs as keys
    for (mftl_int_t i = 0; i < final_tris_count; ++i)
    {
      keys[i] = all_tris[i].triangle.body_id;
      indices[i] = i;
    }

    // 3. Radix Sort by Body ID
    // After this, 'indices' tells us how to reorder 'all_tris' so bodies are grouped
    MFTLSortPairs(keys, indices, final_tris_count);

    // 4. Permute the main data array to match the sorted order
    // Now all_tris is physically grouped by object: [Obj1, Obj1, Obj2, Obj2...]
    permute_triangles(all_tris, indices, final_tris_count);

    // =========================================================
    // PASS 2: PROCESS EACH OBJECT (Range -> Sort -> Unique)
    // =========================================================

    mftl_int_t current_idx = 0;
    while (current_idx < final_tris_count)
    {
      // A. Identify Range for current Body ID
      mftl_int_t current_body_id = all_tris[current_idx].triangle.body_id;
      mftl_int_t start_idx = current_idx;

      // Fast scan to find the end of this object
      while (current_idx < final_tris_count && all_tris[current_idx].triangle.body_id == current_body_id)
      {
        current_idx++;
      }
      mftl_int_t end_idx = current_idx;
      mftl_int_t range_len = end_idx - start_idx;

      // Find the actual MFTL_Object pointer
      // TODO: This is a linear search. We should use a hash map.
      MFTL_Object* target_obj = MFTL_NULL;
      for (mftl_int_t o = 0; o < num_objects; ++o)
      {
        if (objects[o]->global_id == current_body_id)
        {
          target_obj = objects[o];
          break;
        }
      }

      if (target_obj && range_len > 0)
      {
        // B. Prepare Local Sort (Global ID)
        // We reuse the 'keys' and 'indices' buffers we allocated earlier!
        // We only need the first 'range_len' slots.

        // Extract Global IDs for just this range
        for (mftl_int_t k = 0; k < range_len; ++k)
        {
          keys[k] = all_tris[start_idx + k].triangle.global_id;
          indices[k] = k;  // Relative index (0 to range_len)
        }

        // C. Radix Sort by Global ID
        MFTLSortPairs(keys, indices, range_len);

        // D. Deduplicate (Unique)
        // Since we have the sorted keys array, we can count uniques purely on keys first
        mftl_int_t n_unique = 0;
        if (range_len > 0)
        {
          n_unique = 1;
          for (mftl_int_t k = 1; k < range_len; ++k)
          {
            // keys[k] is the global_id of the element at sorted position k
            mftl_int_t prev_gid = keys[k - 1];  // Note: keys array is modified by sort_pairs to be sorted
            mftl_int_t curr_gid = keys[k];
            if (curr_gid != prev_gid) n_unique++;
          }
        }

        // E. Allocate Target Object Memory
        target_obj->n_triangles = n_unique;
        target_obj->capacity_triangles = n_unique;
        target_obj->triangles = malloc(n_unique * sizeof(MFTL_Triangle));
        target_obj->n_half_edges = 3 * n_unique;
        target_obj->capacity_half_edges = 3 * n_unique;
        target_obj->half_edges = malloc(3 * n_unique * sizeof(MFTL_HalfEdge));

        // F. Fill Target Object (using the sorted indices)
        mftl_int_t write_count = 0;
        for (mftl_int_t k = 0; k < range_len; ++k)
        {
          // We use indices[k] to grab the element from the range in sorted order
          mftl_int_t sorted_relative_idx = indices[k];
          PackedTriangleComplex* src = &all_tris[start_idx + sorted_relative_idx];

          // Duplicate check
          if (write_count > 0)
          {
            if (target_obj->triangles[write_count - 1].global_id == src->triangle.global_id)
            {
              continue;  // Skip Duplicate
            }
          }

          // Copy Data
          MFTL_Triangle* dst = &target_obj->triangles[write_count];
          dst->global_id = src->triangle.global_id;
          dst->owner_proc = src->triangle.owner_proc;
          dst->flags = src->triangle.flags;
          dst->is_ghost = (dst->owner_proc == my_rank) ? MFTL_FALSE : MFTL_TRUE;
          dst->area = src->triangle.area;
          memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
          memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
          dst->he_idx = write_count * 3;

          int ret;
          khiter_t k = kh_put(triangle_map, target_obj->triangle_lookup, dst->global_id, &ret);
          kh_value(target_obj->triangle_lookup, k) = write_count;  // local index

          // Copy Half Edges
          for (mftl_int_t i = 0; i < 3; ++i)
          {
            MFTL_HalfEdge* he = &target_obj->half_edges[write_count * 3 + i];
            he->global_id = src->half_edges[i].global_id;
            he->flags = src->half_edges[i].flags;
            he->triangle_idx = write_count;
            he->next_idx = write_count * 3 + (i + 1) % 3;
            he->twin_idx = src->half_edges[i].twin_gid;  // needs to be converted to local index
            he->vertex_idx = src->half_edges[i].vertex_gid;  // needs to be converted to local index
            he->boundary = MFTL_TRUE;  // set later
            int ret;
            khiter_t k = kh_put(halfedge_map, target_obj->halfedge_lookup, he->global_id, &ret);
            kh_value(target_obj->halfedge_lookup, k) = write_count * 3 + i;  // local index
          }

          write_count++;
        }
      }
    }

    // Cleanup
    free(keys);
    free(indices);
    free(all_tris);
  }

  // =========================================================================
  // PART B: VERTEX RECONSTRUCTION
  // =========================================================================

  // 4.3. Combine kept and received vertices
  mftl_int_t final_verts_count = kept_count_verts + total_recv_verts;
  PackedVertex* all_verts = MFTL_NULL;

  if (final_verts_count > 0)
  {
    all_verts = (PackedVertex*) malloc(final_verts_count * sizeof(PackedVertex));

    // Copy Kept
    if (kept_count_verts > 0) memcpy(all_verts, kept_verts, kept_count_verts * sizeof(PackedVertex));

    // Copy Recv
    if (total_recv_verts > 0) memcpy(all_verts + kept_count_verts, recv_buf_verts, total_recv_verts * sizeof(PackedVertex));
  }

  free(kept_verts);
  free(recv_buf_verts);

  // 4.4. HIERARCHICAL RADIX SORT & REBUILD (VERTICES)
  if (final_verts_count > 0)
  {
    // =========================================================
    // PASS 1: GROUP BY BODY_ID
    // =========================================================

    // 1. Allocate Key/Index arrays (Reuse logic from Triangles if you want,
    //    but allocating fresh here is safer to avoid size mismatches)
    mftl_int_t* keys = malloc(final_verts_count * sizeof(mftl_int_t));
    mftl_int_t* indices = malloc(final_verts_count * sizeof(mftl_int_t));

    // 2. Extract Body IDs
    for (mftl_int_t i = 0; i < final_verts_count; ++i)
    {
      keys[i] = all_verts[i].body_id;
      indices[i] = i;
    }

    // 3. Global Radix Sort (Body ID)
    MFTLSortPairs(keys, indices, final_verts_count);

    // 4. Permute to align memory
    permute_vertices(all_verts, indices, final_verts_count);

    // =========================================================
    // PASS 2: PROCESS EACH OBJECT
    // =========================================================

    mftl_int_t current_idx = 0;
    while (current_idx < final_verts_count)
    {
      // A. Identify Range
      mftl_int_t current_body_id = all_verts[current_idx].body_id;
      mftl_int_t start_idx = current_idx;

      while (current_idx < final_verts_count && all_verts[current_idx].body_id == current_body_id)
      {
        current_idx++;
      }
      mftl_int_t range_len = current_idx - start_idx;

      // Find MFTL_Object
      // TODO: This is a linear search. We should use a hash map.
      MFTL_Object* target_obj = MFTL_NULL;
      for (mftl_int_t o = 0; o < num_objects; ++o)
      {
        if (objects[o]->global_id == current_body_id)
        {
          target_obj = objects[o];
          break;
        }
      }

      if (target_obj && range_len > 0)
      {
        // B. Prepare Local Sort (Global ID)

        // Extract Global IDs
        for (mftl_int_t k = 0; k < range_len; ++k)
        {
          keys[k] = all_verts[start_idx + k].global_id;
          indices[k] = k;
        }

        // C. Local Radix Sort (Global ID)
        MFTLSortPairs(keys, indices, range_len);

        // D. Count Uniques
        mftl_int_t n_unique = 0;
        if (range_len > 0)
        {
          n_unique = 1;
          for (mftl_int_t k = 1; k < range_len; ++k)
          {
            mftl_int_t prev_gid = keys[k - 1];
            mftl_int_t curr_gid = keys[k];
            if (curr_gid != prev_gid) n_unique++;
          }
        }

        // E. Reallocate Object Memory
        target_obj->n_vertices = n_unique;
        target_obj->capacity_vertices = n_unique;
        target_obj->vertices = malloc(n_unique * sizeof(MFTL_Vertex));

        // F. Fill Target Object
        mftl_int_t write_count = 0;
        for (mftl_int_t k = 0; k < range_len; ++k)
        {
          mftl_int_t sorted_relative_idx = indices[k];
          PackedVertex* src = &all_verts[start_idx + sorted_relative_idx];

          // Duplicate check
          if (write_count > 0)
          {
            if (target_obj->vertices[write_count - 1].global_id == src->global_id)
            {
              continue;  // Skip Duplicate
            }
          }

          // Copy Data
          MFTL_Vertex* dst = &target_obj->vertices[write_count];
          dst->global_id = src->global_id;
          dst->owner_proc = src->owner_proc;
          dst->flags = src->flags;
          dst->is_ghost = (dst->owner_proc == my_rank) ? MFTL_FALSE : MFTL_TRUE;
          dst->area = src->area;
          dst->meancurvature = src->meancurvature;
          dst->gaussiancurvature = src->gaussiancurvature;

          memcpy(dst->pos, src->pos, 3 * sizeof(double));
          memcpy(dst->normal, src->normal, 3 * sizeof(double));

          dst->out_he_idx = src->id_he_out;  // needs to be converted to local index
          int ret;
          khiter_t k = kh_put(vertex_map, target_obj->vertex_lookup, dst->global_id, &ret);
          kh_value(target_obj->vertex_lookup, k) = write_count;  // local index

          write_count++;
        }
      }
    }

    free(keys);
    free(indices);
    free(all_verts);
  }

  // Rebuild twin pointers across the new local mesh
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    // STEP 1: Convert GIDs to Local Indices
    // This fixes vertex_idx and he_out_idx.
    MFTLConvertGlobalToLocalIndices(object);

    // STEP 2: Rebuild Twins (Topology)
    // This relies on the valid local vertex indices set in Step 1
    MFTLRebuildTwins(object);

    // STEP 3: Rebuild flattened connectivity arrays
    MFTLFlattenConnectivity(object);
  }

  return MFTL_SUCCESS;
}

/**
 * @brief Redistributes a SINGLE object. For cases with 1 or very few objects.
 * @details Removes the overhead of sorting/grouping multiple bodies.
 * Assumes all ranks call this for the SAME object at the same time.
 */
mftl_int_t MFTLRedistributeObject(MFTL_Object* object)
{
  assert(object);

  MFTL_Domain* domain = object->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  int my_rank = domain->rank;
  mftl_int_t requires_global_comm_local = 0;

  // --- 1. Setup ---
  // Fast neighbor lookup
  char* is_neighbor_map = calloc(nproc, sizeof(char));
  for (mftl_int_t i = 0; i < domain->num_neighbors; ++i) is_neighbor_map[domain->neighbors[i]] = 1;

  // Buffers
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));

  // Heuristic Allocation
  mftl_int_t tri_cap, vert_cap;
  MFTLCalcBufferCapacities(object->global_n_triangles, object->global_n_vertices, nproc, &tri_cap, &vert_cap);

  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    buffer_init(&trans_tri_bufs[r], tri_cap);
    buffer_init(&trans_vert_bufs[r], vert_cap);
  }

  // --- 2. Analysis (Fill Transient Buffers) ---
  MFTL_Triangle* tri_base = object->triangles;
  MFTL_Vertex* vert_base = object->vertices;
  mftl_int_t n_triangles = object->n_triangles;
  mftl_int_t n_vertices = object->n_vertices;

  mftl_int_t* vert_owners = malloc(n_vertices * sizeof(mftl_int_t));
  mftl_int_t* tri_owners = malloc(n_triangles * sizeof(mftl_int_t));

  // Ensure topology is built. If NULL, build it now (Lazy Init).
  if (!object->flat_tri_verts || !object->flat_tri_neighbors)
  {
    MFTLFlattenConnectivity(object);
  }

  // Direct pointers to object memory (Zero Copy)
  mftl_int_t* flat_tri_verts = object->flat_tri_verts;
  mftl_int_t* flat_tri_neighbors = object->flat_tri_neighbors;

  // Pre-calculate Owners
  for (mftl_int_t i = 0; i < n_vertices; ++i)
  {
    MFTL_Vertex* v = &vert_base[i];
    v->owner_proc = MFTLGetOwnerProc(domain, v->pos);
    vert_owners[i] = v->owner_proc;
  }
  for (mftl_int_t i = 0; i < n_triangles; ++i)
  {
    MFTL_Triangle* tri = &tri_base[i];
    tri->owner_proc = MFTLGetOwnerProc(domain, tri->center);
    tri_owners[i] = tri->owner_proc;
  }

  // Loop over all triangles to determine destinations
  for (mftl_int_t i = 0; i < n_triangles; ++i)
  {
    // This prevents "Ghost Forwarding" crashes and automatically cleans up stale ghosts.
    // Don't use owner_proc because it already contains the new owner.
    if (tri_base[i].is_ghost)
    {
      continue;
    }

    mftl_int_t distinct_ranks[7];
    mftl_int_t count = 0;

    distinct_ranks[count++] = tri_owners[i];  // Triangle Owner
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];  // Vertex Owners
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 1]];
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 2]];

    for (mftl_int_t j = 0; j < 3; ++j)
    {
      mftl_int_t adj_idx = flat_tri_neighbors[3 * i + j];
      if (adj_idx != -1) distinct_ranks[count++] = tri_owners[adj_idx];
    }

    mftl_int_t unique_dests[7];
    mftl_int_t num_unique_dests = MFTLGetUniqueIntegers(distinct_ranks, count, unique_dests, 7);

    for (mftl_int_t k = 0; k < num_unique_dests; ++k)
    {
      mftl_int_t dest_rank = unique_dests[k];
      if (dest_rank != my_rank && !is_neighbor_map[dest_rank])
      {
        requires_global_comm_local = 1;
      }

      buffer_add(&trans_tri_bufs[dest_rank], i);
      buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 0]);
      buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 1]);
      buffer_add(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 2]);
    }
  }

  // Cleanup helper arrays
  free(vert_owners);
  free(tri_owners);

  // --- 3. Process Transient Buffers (Sort -> Unique -> Pack Direct) ---
  for (mftl_int_t r = 0; r < nproc; ++r)
  {
    // A. Triangles
    if (trans_tri_bufs[r].size > 0)
    {
      MFTLSortIds(trans_tri_bufs[r].data, trans_tri_bufs[r].size);
      mftl_int_t n_unique = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      // Alloc Exact
      send_tri_bufs[r].buffer = malloc(n_unique * sizeof(PackedTriangleComplex));
      send_tri_bufs[r].capacity = n_unique;
      send_tri_bufs[r].count = n_unique;

      // Pack Direct
      PackedTriangleComplex* buf = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      for (mftl_int_t i = 0; i < n_unique; ++i)
      {
        mftl_int_t tri_idx = trans_tri_bufs[r].data[i];
        MFTLPackSingleTriangle(object, tri_idx, &buf[i], MFTL_FALSE);
      }
    }

    // B. Vertices
    if (trans_vert_bufs[r].size > 0)
    {
      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      mftl_int_t n_unique = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      // Alloc Exact
      send_vert_bufs[r].buffer = malloc(n_unique * sizeof(PackedVertex));
      send_vert_bufs[r].capacity = n_unique;
      send_vert_bufs[r].count = n_unique;

      // Pack Direct
      PackedVertex* buf = (PackedVertex*) send_vert_bufs[r].buffer;
      for (mftl_int_t i = 0; i < n_unique; ++i)
      {
        mftl_int_t vert_idx = trans_vert_bufs[r].data[i];
        MFTLPackSingleVertex(object, vert_idx, &buf[i], MFTL_FALSE);
      }
    }

    // Cleanup Transient immediately
    free(trans_tri_bufs[r].data);
    free(trans_vert_bufs[r].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);

  // --- 4. Separate Kept Data (Zero-Copy Steal) ---
  PackedTriangleComplex* kept_tris = MFTL_NULL;
  PackedVertex* kept_verts = MFTL_NULL;
  mftl_int_t kept_count_tris = 0, kept_count_verts = 0;

  if (send_tri_bufs[my_rank].count > 0)
  {
    kept_tris = (PackedTriangleComplex*) send_tri_bufs[my_rank].buffer;
    kept_count_tris = send_tri_bufs[my_rank].count;
    send_tri_bufs[my_rank].buffer = MFTL_NULL;
    send_tri_bufs[my_rank].count = 0;
  }
  if (send_vert_bufs[my_rank].count > 0)
  {
    kept_verts = (PackedVertex*) send_vert_bufs[my_rank].buffer;
    kept_count_verts = send_vert_bufs[my_rank].count;
    send_vert_bufs[my_rank].buffer = MFTL_NULL;
    send_vert_bufs[my_rank].count = 0;
  }

  // --- 5. Communication (Phase 2 & 3) ---
  mftl_int_t global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_MFTL_INT, MPI_LOR, domain->comm_cart);

  PackedTriangleComplex* recv_tris;
  PackedVertex* recv_verts;
  mftl_int_t n_recv_tris, n_recv_verts;

  // Use the Generic MPI Exchange Helper
  MFTLMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_tris, &n_recv_tris);

  MFTLMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_verts, &n_recv_verts);

  // Cleanup Send Buffers
  for (int i = 0; i < nproc; ++i)
  {
    if (send_tri_bufs[i].buffer) free(send_tri_bufs[i].buffer);
    if (send_vert_bufs[i].buffer) free(send_vert_bufs[i].buffer);
  }
  free(send_tri_bufs);
  free(send_vert_bufs);
  free(is_neighbor_map);

  // --- 6. Reconstruction ---
  MFTLResetObjectData(object);  // Wipe old data

  // Combine Kept + Recv (Triangles)
  mftl_int_t total_tris = kept_count_tris + n_recv_tris;
  PackedTriangleComplex* all_tris = MFTL_NULL;
  if (total_tris > 0)
  {
    all_tris = malloc(total_tris * sizeof(PackedTriangleComplex));
    if (kept_count_tris) memcpy(all_tris, kept_tris, kept_count_tris * sizeof(PackedTriangleComplex));
    if (n_recv_tris) memcpy(all_tris + kept_count_tris, recv_tris, n_recv_tris * sizeof(PackedTriangleComplex));
  }
  free(kept_tris);
  free(recv_tris);

  // Combine Kept + Recv (Vertices)
  mftl_int_t total_verts = kept_count_verts + n_recv_verts;
  PackedVertex* all_verts = MFTL_NULL;
  if (total_verts > 0)
  {
    all_verts = malloc(total_verts * sizeof(PackedVertex));
    if (kept_count_verts) memcpy(all_verts, kept_verts, kept_count_verts * sizeof(PackedVertex));
    if (n_recv_verts) memcpy(all_verts + kept_count_verts, recv_verts, n_recv_verts * sizeof(PackedVertex));
  }
  free(kept_verts);
  free(recv_verts);

  // Rebuild Triangles (Direct GID Sort)
  if (total_tris > 0)
  {
    mftl_int_t* keys = malloc(total_tris * sizeof(mftl_int_t));
    mftl_int_t* indices = malloc(total_tris * sizeof(mftl_int_t));
    for (mftl_int_t i = 0; i < total_tris; ++i)
    {
      keys[i] = all_tris[i].triangle.global_id;
      indices[i] = i;
    }
    MFTLSortPairs(keys, indices, total_tris);

    // Count Uniques
    mftl_int_t n_unique = 1;
    for (mftl_int_t i = 1; i < total_tris; ++i)
      if (keys[i] != keys[i - 1]) n_unique++;

    object->n_triangles = n_unique;
    object->capacity_triangles = n_unique;
    object->triangles = malloc(n_unique * sizeof(MFTL_Triangle));
    object->n_half_edges = 3 * n_unique;
    object->capacity_half_edges = 3 * n_unique;
    object->half_edges = malloc(3 * n_unique * sizeof(MFTL_HalfEdge));

    mftl_int_t write_idx = 0;
    for (mftl_int_t k = 0; k < total_tris; ++k)
    {
      mftl_int_t sorted_idx = indices[k];
      PackedTriangleComplex* src = &all_tris[sorted_idx];

      // Deduplicate
      if (write_idx > 0 && object->triangles[write_idx - 1].global_id == src->triangle.global_id) continue;

      // Copy Triangle
      MFTL_Triangle* dst = &object->triangles[write_idx];
      dst->global_id = src->triangle.global_id;
      dst->owner_proc = src->triangle.owner_proc;
      dst->flags = src->triangle.flags;
      dst->is_ghost = (dst->owner_proc == my_rank) ? MFTL_FALSE : MFTL_TRUE;
      dst->area = src->triangle.area;
      memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
      memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
      dst->he_idx = write_idx * 3;

      int ret;
      khiter_t iter = kh_put(triangle_map, object->triangle_lookup, dst->global_id, &ret);
      kh_value(object->triangle_lookup, iter) = write_idx;

      // Copy Halfedges
      for (int j = 0; j < 3; ++j)
      {
        MFTL_HalfEdge* he = &object->half_edges[write_idx * 3 + j];
        he->global_id = src->half_edges[j].global_id;
        he->flags = src->half_edges[j].flags;
        he->triangle_idx = write_idx;
        he->next_idx = write_idx * 3 + (j + 1) % 3;
        he->twin_idx = src->half_edges[j].twin_gid;  // Fix later
        he->vertex_idx = src->half_edges[j].vertex_gid;  // Fix later
        he->boundary = MFTL_TRUE;

        iter = kh_put(halfedge_map, object->halfedge_lookup, he->global_id, &ret);
        kh_value(object->halfedge_lookup, iter) = write_idx * 3 + j;
      }
      write_idx++;
    }
    free(keys);
    free(indices);
    free(all_tris);
  }

  // Rebuild Vertices (Direct GID Sort)
  if (total_verts > 0)
  {
    mftl_int_t* keys = malloc(total_verts * sizeof(mftl_int_t));
    mftl_int_t* indices = malloc(total_verts * sizeof(mftl_int_t));
    for (mftl_int_t i = 0; i < total_verts; ++i)
    {
      keys[i] = all_verts[i].global_id;
      indices[i] = i;
    }
    MFTLSortPairs(keys, indices, total_verts);

    mftl_int_t n_unique = 1;
    for (mftl_int_t i = 1; i < total_verts; ++i)
      if (keys[i] != keys[i - 1]) n_unique++;

    object->n_vertices = n_unique;
    object->capacity_vertices = n_unique;
    object->vertices = malloc(n_unique * sizeof(MFTL_Vertex));

    mftl_int_t write_idx = 0;
    for (mftl_int_t k = 0; k < total_verts; ++k)
    {
      mftl_int_t sorted_idx = indices[k];
      PackedVertex* src = &all_verts[sorted_idx];

      if (write_idx > 0 && object->vertices[write_idx - 1].global_id == src->global_id) continue;

      MFTL_Vertex* dst = &object->vertices[write_idx];
      dst->global_id = src->global_id;
      dst->owner_proc = src->owner_proc;
      dst->flags = src->flags;
      dst->is_ghost = (dst->owner_proc == my_rank) ? MFTL_FALSE : MFTL_TRUE;
      dst->area = src->area;
      dst->meancurvature = src->meancurvature;
      dst->gaussiancurvature = src->gaussiancurvature;
      memcpy(dst->pos, src->pos, sizeof(double) * 3);
      memcpy(dst->normal, src->normal, sizeof(double) * 3);
      dst->out_he_idx = src->id_he_out;  // Fix later

      int ret;
      khiter_t iter = kh_put(vertex_map, object->vertex_lookup, dst->global_id, &ret);
      kh_value(object->vertex_lookup, iter) = write_idx;

      write_idx++;
    }
    free(keys);
    free(indices);
    free(all_verts);
  }

  // 7. Final Topology
  MFTLConvertGlobalToLocalIndices(object);
  MFTLRebuildTwins(object);
  MFTLFlattenConnectivity(object);

  return MFTL_SUCCESS;
}

/**
 * @brief Synchronizes ghost vertex positions for a list of objects.
 * @param ens Ensemble pointer.
 * @param object_lookup Optional hash map (BodyID -> Index). Pass MFTL_NULL to use linear search fallback.
 * @return MFTL_SUCCESS on success.
 */
mftl_int_t MFTLUpdateGhostPositions(MFTL_Ensemble* ens, khash_t(object_map) * object_lookup)
{
  if (ens->num_objects == 0 || ens->objects == MFTL_NULL) return MFTL_SUCCESS;

  MFTL_Object** objects = ens->objects;
  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  int my_rank = domain->rank;

  mftl_int_t num_neighbors = domain->num_neighbors;
  mftl_int_t num_objects = ens->num_objects;

  if (num_neighbors == 0 || nproc == 1) return MFTL_SUCCESS;

  // --- 1. Setup Buffers ---
  IntBuffer* req_send_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) buffer_init(&req_send_bufs[i], 1024);  // TODO: Optimize Buffer Size

  // Fast Map: Global Rank -> Neighbor Index
  int* rank_to_neigh_idx = malloc(nproc * sizeof(int));
  memset(rank_to_neigh_idx, -1, nproc * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh_idx[domain->neighbors[i]] = i;

  // --- 2. Build Requests (Iterate ALL objects) ---
  for (mftl_int_t o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    MFTL_Vertex* verts = obj->vertices;
    for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &verts[i];

      if (v->is_ghost)
      {
        mftl_int_t owner = v->owner_proc;
        assert(owner >= 0 && owner < nproc && "Ghost owner rank out of bounds!");
        int n_idx = rank_to_neigh_idx[owner];

        // Safety: Ghost owner MUST be a neighbor in the graph communicator
        assert(n_idx != -1);

        // Append (BodyID, GID) to the request buffer
        buffer_add(&req_send_bufs[n_idx], obj->global_id);
        buffer_add(&req_send_bufs[n_idx], v->global_id);
      }
    }
  }

  // --- 3. Exchange Request Counts ---
  mftl_int_t* n_send_reqs = calloc(num_neighbors, sizeof(mftl_int_t));
  mftl_int_t* n_recv_reqs = malloc(num_neighbors * sizeof(mftl_int_t));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_reqs[i] = req_send_bufs[i].size;  // Actual count is size (2 * num_requests)
  }

  MPI_Neighbor_alltoall(n_send_reqs, 1, MPI_MFTL_INT, n_recv_reqs, 1, MPI_MFTL_INT, domain->comm_graph);

  // --- 4. Exchange Requests ---
  mftl_int_t* s_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t* r_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t total_send_ints = 0;
  mftl_int_t total_recv_ints = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_displs[i + 1] = s_displs[i] + n_send_reqs[i];
    r_displs[i + 1] = r_displs[i] + n_recv_reqs[i];
    total_send_ints += n_send_reqs[i];
    total_recv_ints += n_recv_reqs[i];
  }

  mftl_int_t* flat_send_reqs = malloc(total_send_ints * sizeof(mftl_int_t));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (n_send_reqs[i] > 0)
    {
      memcpy(flat_send_reqs + s_displs[i], req_send_bufs[i].data, n_send_reqs[i] * sizeof(mftl_int_t));
    }
  }

  mftl_int_t* flat_recv_reqs = malloc(total_recv_ints * sizeof(mftl_int_t));

  MPI_Neighbor_alltoallv(flat_send_reqs, n_send_reqs, s_displs, MPI_MFTL_INT, flat_recv_reqs, n_recv_reqs, r_displs, MPI_MFTL_INT, domain->comm_graph);

  free(flat_send_reqs);

  // --- 5. Process Requests & Build Replies ---
  mftl_int_t total_reply_payloads = total_recv_ints / 2;
  GhostUpdatePayload* reply_payloads = malloc(total_reply_payloads * sizeof(GhostUpdatePayload));

  mftl_int_t reply_idx = 0;

  // Cache the last accessed object to speed up sequential access
  mftl_int_t last_body_id = -1;
  MFTL_Object* current_obj = MFTL_NULL;

  for (mftl_int_t k = 0; k < total_recv_ints; k += 2)
  {
    mftl_int_t req_body_id = flat_recv_reqs[k];
    mftl_int_t req_gid = flat_recv_reqs[k + 1];

    // --- A. HYBRID OBJECT LOOKUP ---
    if (req_body_id != last_body_id)
    {
      if (object_lookup != MFTL_NULL)
      {
        // PATH 1: O(1) Hash Map Lookup
        khiter_t k_obj = kh_get(object_map, object_lookup, req_body_id);
        if (k_obj != kh_end(object_lookup))
        {
          current_obj = objects[kh_value(object_lookup, k_obj)];
        }
        else
        {
          current_obj = MFTL_NULL;
        }
      }
      else
      {
        // PATH 2: O(N) Linear Search Fallback
        // (Only runs when body_id changes, so still fast for grouped requests)
        current_obj = MFTL_NULL;
        for (mftl_int_t o = 0; o < num_objects; ++o)
        {
          if (objects[o]->global_id == req_body_id)
          {
            current_obj = objects[o];
            break;
          }
        }
      }
      last_body_id = req_body_id;
    }

    // --- B. VERTEX LOOKUP (Using existing Hash Map) ---
    MFTL_Vertex* found_v = MFTL_NULL;
    assert(current_obj);

    khiter_t k_vert = kh_get(vertex_map, current_obj->vertex_lookup, req_gid);
    if (k_vert != kh_end(current_obj->vertex_lookup))
    {
      mftl_int_t local_idx = kh_value(current_obj->vertex_lookup, k_vert);
      found_v = &current_obj->vertices[local_idx];
    }

    // --- C. Pack Payload ---
    GhostUpdatePayload* p = &reply_payloads[reply_idx++];
    p->global_id = req_gid;

    assert(found_v);

    memcpy(p->pos, found_v->pos, 3 * sizeof(double));
    memcpy(p->normal, found_v->normal, 3 * sizeof(double));
    p->meancurvature = found_v->meancurvature;
    p->gaussiancurvature = found_v->gaussiancurvature;
    p->area = found_v->area;
  }

  free(flat_recv_reqs);

  // --- 6. Exchange Replies ---
  mftl_int_t* n_send_replies = calloc(num_neighbors, sizeof(mftl_int_t));
  mftl_int_t* n_recv_replies = malloc(num_neighbors * sizeof(mftl_int_t));
  mftl_int_t* s_rep_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t* r_rep_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_replies[i] = n_recv_reqs[i] / 2;
    n_recv_replies[i] = n_send_reqs[i] / 2;
    s_rep_displs[i + 1] = s_rep_displs[i] + n_send_replies[i];
    r_rep_displs[i + 1] = r_rep_displs[i] + n_recv_replies[i];
  }

  mftl_int_t total_recv_replies = total_send_ints / 2;
  GhostUpdatePayload* recv_payloads = malloc(total_recv_replies * sizeof(GhostUpdatePayload));

  MPI_Neighbor_alltoallv(reply_payloads, n_send_replies, s_rep_displs, MPI_GhostUpdatePayload, recv_payloads, n_recv_replies, r_rep_displs,
                         MPI_GhostUpdatePayload, domain->comm_graph);

  free(reply_payloads);

  // --- 7. Apply Updates ---
  mftl_int_t current_payload_idx = 0;

  // Reset cache for application phase
  last_body_id = -1;
  current_obj = MFTL_NULL;

  for (int n = 0; n < num_neighbors; ++n)
  {
    IntBuffer* buf = &req_send_bufs[n];
    mftl_int_t count = buf->size;

    for (mftl_int_t k = 0; k < count; k += 2)
    {
      mftl_int_t req_body_id = buf->data[k];
      mftl_int_t req_gid = buf->data[k + 1];
      GhostUpdatePayload* p = &recv_payloads[current_payload_idx++];

      // --- A. HYBRID OBJECT LOOKUP ---
      if (req_body_id != last_body_id)
      {
        if (object_lookup != MFTL_NULL)
        {
          khiter_t k_obj = kh_get(object_map, object_lookup, req_body_id);
          current_obj = (k_obj != kh_end(object_lookup)) ? objects[kh_value(object_lookup, k_obj)] : MFTL_NULL;
        }
        else
        {
          current_obj = MFTL_NULL;
          for (mftl_int_t o = 0; o < num_objects; ++o)
          {
            if (objects[o]->global_id == req_body_id)
            {
              current_obj = objects[o];
              break;
            }
          }
        }
        last_body_id = req_body_id;
      }

      assert(current_obj);

      // --- B. VERTEX LOOKUP (O(1)) ---
      khiter_t k_vert = kh_get(vertex_map, current_obj->vertex_lookup, req_gid);
      if (k_vert != kh_end(current_obj->vertex_lookup))
      {
        mftl_int_t local_idx = kh_value(current_obj->vertex_lookup, k_vert);
        MFTL_Vertex* v = &current_obj->vertices[local_idx];

        // Apply Update
        memcpy(v->pos, p->pos, 3 * sizeof(double));
        memcpy(v->normal, p->normal, 3 * sizeof(double));
        v->meancurvature = p->meancurvature;
        v->gaussiancurvature = p->gaussiancurvature;
        v->area = p->area;
      }
    }
  }

  for (int i = 0; i < num_neighbors; ++i)
  {
    if (req_send_bufs[i].data) free(req_send_bufs[i].data);
  }

  // --- Cleanup ---
  free(req_send_bufs);
  free(recv_payloads);
  free(rank_to_neigh_idx);

  free(n_send_reqs);
  free(n_recv_reqs);
  free(n_send_replies);
  free(n_recv_replies);
  free(s_displs);
  free(r_displs);
  free(s_rep_displs);
  free(r_rep_displs);

  return MFTL_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* GHOST LAYER EXTENSION LOGIC                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Helper to append a new vertex to the object during ghost integration.
 */
static void MFTLAppendGhostVertex(MFTL_Object* obj, PackedVertex* pv)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);

  mftl_int_t idx = obj->n_vertices++;
  MFTL_Vertex* v = &obj->vertices[idx];

  v->global_id = pv->global_id;
  v->owner_proc = pv->owner_proc;
  v->is_ghost = MFTL_TRUE;
  v->flags = pv->flags;
  v->area = pv->area;
  v->meancurvature = pv->meancurvature;
  v->gaussiancurvature = pv->gaussiancurvature;
  v->out_he_idx = pv->id_he_out;  // GID, resolved later

  memcpy(v->pos, pv->pos, 3 * sizeof(double));
  memcpy(v->normal, pv->normal, 3 * sizeof(double));

  int ret;
  khiter_t k = kh_put(vertex_map, obj->vertex_lookup, v->global_id, &ret);
  kh_value(obj->vertex_lookup, k) = idx;
}

/**
 * @brief Helper to append a new triangle (and its halfedges) during ghost integration.
 */
static void MFTLAppendGhostTriangle(MFTL_Object* obj, PackedTriangleComplex* ptc)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 3, MFTL_HalfEdge);

  mftl_int_t t_idx = obj->n_triangles++;
  MFTL_Triangle* t = &obj->triangles[t_idx];

  t->global_id = ptc->triangle.global_id;
  t->owner_proc = ptc->triangle.owner_proc;
  t->is_ghost = MFTL_TRUE;
  t->flags = ptc->triangle.flags;
  t->area = ptc->triangle.area;
  t->he_idx = obj->n_half_edges;

  memcpy(t->normal, ptc->triangle.normal, 3 * sizeof(double));
  memcpy(t->center, ptc->triangle.center, 3 * sizeof(double));

  int ret;
  khiter_t k = kh_put(triangle_map, obj->triangle_lookup, t->global_id, &ret);
  kh_value(obj->triangle_lookup, k) = t_idx;

  for (int i = 0; i < 3; ++i)
  {
    mftl_int_t he_idx = obj->n_half_edges++;
    MFTL_HalfEdge* he = &obj->half_edges[he_idx];
    PackedHalfEdge* phe = &ptc->half_edges[i];

    he->global_id = phe->global_id;
    he->triangle_idx = t_idx;
    he->vertex_idx = phe->vertex_gid;  // GID, resolved later
    he->twin_idx = phe->twin_gid;  // GID, resolved later
    he->next_idx = t->he_idx + ((i + 1) % 3);
    he->boundary = MFTL_TRUE;
    he->flags = phe->flags;

    k = kh_put(halfedge_map, obj->halfedge_lookup, he->global_id, &ret);
    kh_value(obj->halfedge_lookup, k) = he_idx;
  }
}

/**
 * @brief Extends the ghost layer by 'steps' topological layers.
 * Uses a Request-Reply pattern with direct Neighbor communication.
 */
mftl_int_t MFTLExtendGhostLayer(MFTL_Ensemble* ens, int steps)
{
  if (steps <= 0) return MFTL_SUCCESS;

  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  int num_neighbors = domain->num_neighbors;

  // =========================================================================
  // PHASE 1: IDENTIFY & REQUEST
  // "I have ghost X, please expand it for me."
  // =========================================================================

  IntBuffer* req_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) buffer_init(&req_bufs[i], MFTL_INIT_CAPACITY);

  // Optimization: Pre-compute Rank -> Neighbor Index map once
  // (Assuming nproc isn't massive; if nproc > 100k, use a hash map or linear search on small neighbor lists)
  int* rank_to_neigh = malloc(nproc * sizeof(int));
  // Initialize to -1
  for (int i = 0; i < nproc; ++i) rank_to_neigh[i] = -1;
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh[domain->neighbors[i]] = i;

  for (mftl_int_t o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];

    // Iterate Vertices to find Ghosts (Seeds)
    for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost)
      {
        int n_idx = rank_to_neigh[v->owner_proc];
        // Safety: Ghost owner MUST be a neighbor in the graph communicator
        if (n_idx != -1)
        {
          buffer_add(&req_bufs[n_idx], obj->global_id);
          buffer_add(&req_bufs[n_idx], v->global_id);
        }
      }
    }
  }
  free(rank_to_neigh);

  // Exchange Counts
  mftl_int_t* n_send_ints = calloc(num_neighbors, sizeof(mftl_int_t));
  mftl_int_t* n_recv_ints = malloc(num_neighbors * sizeof(mftl_int_t));

  for (int i = 0; i < num_neighbors; ++i) n_send_ints[i] = req_bufs[i].size;

  MPI_Neighbor_alltoall(n_send_ints, 1, MPI_MFTL_INT, n_recv_ints, 1, MPI_MFTL_INT, domain->comm_graph);

  // Calc Displacements
  mftl_int_t* s_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t* r_displs = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t total_send = 0, total_recv = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_displs[i + 1] = s_displs[i] + n_send_ints[i];
    r_displs[i + 1] = r_displs[i] + n_recv_ints[i];
    total_send += n_send_ints[i];
    total_recv += n_recv_ints[i];
  }

  // Flatten Requests
  mftl_int_t* flat_send_req = malloc(total_send * sizeof(mftl_int_t));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (req_bufs[i].size > 0)
    {
      memcpy(flat_send_req + s_displs[i], req_bufs[i].data, req_bufs[i].size * sizeof(mftl_int_t));
    }
    // Cleanup early
    free(req_bufs[i].data);
  }
  free(req_bufs);

  // Exchange Requests
  mftl_int_t* flat_recv_req = malloc(total_recv * sizeof(mftl_int_t));
  MPI_Neighbor_alltoallv(flat_send_req, n_send_ints, s_displs, MPI_MFTL_INT, flat_recv_req, n_recv_ints, r_displs, MPI_MFTL_INT, domain->comm_graph);

  free(flat_send_req);
  free(s_displs);
  free(n_send_ints);

  // =========================================================================
  // PHASE 2: BATCH PROCESSING & PACKING
  // =========================================================================

  // Use raw dynamically resizing arrays instead of MFTL_SendBuffer struct
  // to avoid mapping confusion.
  PackedTriangleComplex** raw_send_tris = calloc(num_neighbors, sizeof(PackedTriangleComplex*));
  PackedVertex** raw_send_verts = calloc(num_neighbors, sizeof(PackedVertex*));

  mftl_int_t* cnt_send_tris = calloc(num_neighbors, sizeof(mftl_int_t));
  mftl_int_t* cap_send_tris = calloc(num_neighbors, sizeof(mftl_int_t));

  mftl_int_t* cnt_send_verts = calloc(num_neighbors, sizeof(mftl_int_t));
  mftl_int_t* cap_send_verts = calloc(num_neighbors, sizeof(mftl_int_t));

  // Collector reuse
  IntBuffer* seed_collector = calloc(ens->num_objects, sizeof(IntBuffer));
  for (int i = 0; i < ens->num_objects; ++i) buffer_init(&seed_collector[i], 256);

  for (int n = 0; n < num_neighbors; ++n)
  {
    mftl_int_t start_idx = r_displs[n];
    mftl_int_t end_idx = r_displs[n + 1];

    // 1. Reset Collector
    for (int i = 0; i < ens->num_objects; ++i) buffer_reset(&seed_collector[i]);

    // 2. Binning
    for (mftl_int_t k = start_idx; k < end_idx; k += 2)
    {
      mftl_int_t body_id = flat_recv_req[k];
      mftl_int_t gid = flat_recv_req[k + 1];

      // Find Object (Optimize: use hash map if many objects)
      for (mftl_int_t o = 0; o < ens->num_objects; ++o)
      {
        if (ens->objects[o]->global_id == body_id)
        {
          MFTL_Object* obj = ens->objects[o];
          khiter_t it = kh_get(vertex_map, obj->vertex_lookup, gid);
          if (it != kh_end(obj->vertex_lookup))
          {
            buffer_add(&seed_collector[o], kh_value(obj->vertex_lookup, it));
          }
          break;
        }
      }
    }

    // 3. Expansion & Packing
    MFTL_Neighborhood nb;
    for (mftl_int_t o = 0; o < ens->num_objects; ++o)
    {
      if (seed_collector[o].size == 0) continue;

      MFTL_Object* obj = ens->objects[o];

      // Batch Topology Search
      MFTLGetNeighborhood(obj, seed_collector[o].data, seed_collector[o].size, true, steps * 2, &nb);

      // --- Pack Triangles ---
      if (nb.n_tris > 0)
      {
        // Geometric Growth for Amortized O(1)
        if (cnt_send_tris[n] + nb.n_tris > cap_send_tris[n])
        {
          mftl_int_t new_cap = (cap_send_tris[n] == 0) ? (nb.n_tris + 128) : (cap_send_tris[n] * 2 + nb.n_tris);
          raw_send_tris[n] = realloc(raw_send_tris[n], new_cap * sizeof(PackedTriangleComplex));
          cap_send_tris[n] = new_cap;
        }

        for (mftl_int_t t = 0; t < nb.n_tris; ++t)
        {
          MFTLPackSingleTriangle(obj, nb.tri_indices[t], &raw_send_tris[n][cnt_send_tris[n]++], false);
        }
      }

      // --- Pack Vertices ---
      if (nb.n_verts > 0)
      {
        if (cnt_send_verts[n] + nb.n_verts > cap_send_verts[n])
        {
          mftl_int_t new_cap = (cap_send_verts[n] == 0) ? (nb.n_verts + 128) : (cap_send_verts[n] * 2 + nb.n_verts);
          raw_send_verts[n] = realloc(raw_send_verts[n], new_cap * sizeof(PackedVertex));
          cap_send_verts[n] = new_cap;
        }

        for (mftl_int_t v = 0; v < nb.n_verts; ++v)
        {
          MFTLPackSingleVertex(obj, nb.vert_indices[v], &raw_send_verts[n][cnt_send_verts[n]++], false);
        }
      }
      MFTLFreeNeighborhood(&nb);
    }
  }

  free(flat_recv_req);
  free(r_displs);
  free(n_recv_ints);  // from phase 1
  for (int i = 0; i < ens->num_objects; ++i) free(seed_collector[i].data);
  free(seed_collector);

  // =========================================================================
  // PHASE 3: DIRECT DATA EXCHANGE
  // =========================================================================

  // --- Exchange Counts ---
  mftl_int_t* n_recv_tris = malloc(num_neighbors * sizeof(mftl_int_t));
  mftl_int_t* n_recv_verts = malloc(num_neighbors * sizeof(mftl_int_t));

  MPI_Neighbor_alltoall(cnt_send_tris, 1, MPI_MFTL_INT, n_recv_tris, 1, MPI_MFTL_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_verts, 1, MPI_MFTL_INT, n_recv_verts, 1, MPI_MFTL_INT, domain->comm_graph);

  // --- Exchange Triangles ---
  mftl_int_t* s_disp_t = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t* r_disp_t = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t tot_send_t = 0, tot_recv_t = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_disp_t[i + 1] = s_disp_t[i] + cnt_send_tris[i];
    r_disp_t[i + 1] = r_disp_t[i] + n_recv_tris[i];
    tot_send_t += cnt_send_tris[i];
    tot_recv_t += n_recv_tris[i];
  }

  // Linearize Send Buffer (Triangles)
  PackedTriangleComplex* flat_send_tris = malloc(tot_send_t * sizeof(PackedTriangleComplex));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_tris[i] > 0) memcpy(flat_send_tris + s_disp_t[i], raw_send_tris[i], cnt_send_tris[i] * sizeof(PackedTriangleComplex));
    free(raw_send_tris[i]);  // Free individual buffers immediately
  }
  free(raw_send_tris);

  PackedTriangleComplex* recv_tris = malloc(tot_recv_t * sizeof(PackedTriangleComplex));
  MPI_Neighbor_alltoallv(flat_send_tris, cnt_send_tris, s_disp_t, MPI_PackedTriangleComplex, recv_tris, n_recv_tris, r_disp_t, MPI_PackedTriangleComplex,
                         domain->comm_graph);

  free(flat_send_tris);
  free(s_disp_t);
  // free(r_disp_t); // Keep for Phase 4 loops if needed, or just iterate recv buffer linearly

  // --- Exchange Vertices ---
  mftl_int_t* s_disp_v = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t* r_disp_v = calloc(num_neighbors + 1, sizeof(mftl_int_t));
  mftl_int_t tot_send_v = 0, tot_recv_v = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_disp_v[i + 1] = s_disp_v[i] + cnt_send_verts[i];
    r_disp_v[i + 1] = r_disp_v[i] + n_recv_verts[i];
    tot_send_v += cnt_send_verts[i];
    tot_recv_v += n_recv_verts[i];
  }

  PackedVertex* flat_send_verts = malloc(tot_send_v * sizeof(PackedVertex));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_verts[i] > 0) memcpy(flat_send_verts + s_disp_v[i], raw_send_verts[i], cnt_send_verts[i] * sizeof(PackedVertex));
    free(raw_send_verts[i]);
  }
  free(raw_send_verts);

  PackedVertex* recv_verts = malloc(tot_recv_v * sizeof(PackedVertex));
  MPI_Neighbor_alltoallv(flat_send_verts, cnt_send_verts, s_disp_v, MPI_PackedVertex, recv_verts, n_recv_verts, r_disp_v, MPI_PackedVertex, domain->comm_graph);

  free(flat_send_verts);
  free(s_disp_v);

  // Cleanup Counts
  free(cnt_send_tris);
  free(cap_send_tris);
  free(cnt_send_verts);
  free(cap_send_verts);
  free(n_recv_tris);
  free(n_recv_verts);

  // =========================================================================
  // PHASE 4: INTEGRATION
  // =========================================================================

  // Integrate Vertices
  for (int i = 0; i < tot_recv_v; ++i)
  {
    PackedVertex* pv = &recv_verts[i];

    // Simple Linear Search for Object (Assumption: low N_objects)
    MFTL_Object* obj = MFTL_NULL;
    for (mftl_int_t o = 0; o < ens->num_objects; ++o)
    {
      if (ens->objects[o]->global_id == pv->body_id)
      {
        obj = ens->objects[o];
        break;
      }
    }

    if (obj)
    {
      khiter_t it = kh_get(vertex_map, obj->vertex_lookup, pv->global_id);
      if (it == kh_end(obj->vertex_lookup))
      {
        MFTLAppendGhostVertex(obj, pv);
      }
    }
  }

  // Integrate Triangles
  for (int i = 0; i < tot_recv_t; ++i)
  {
    PackedTriangleComplex* ptc = &recv_tris[i];

    MFTL_Object* obj = MFTL_NULL;
    for (mftl_int_t o = 0; o < ens->num_objects; ++o)
    {
      if (ens->objects[o]->global_id == ptc->triangle.body_id)
      {
        obj = ens->objects[o];
        break;
      }
    }

    if (obj)
    {
      khiter_t it = kh_get(triangle_map, obj->triangle_lookup, ptc->triangle.global_id);
      if (it == kh_end(obj->triangle_lookup))
      {
        MFTLAppendGhostTriangle(obj, ptc);
      }
    }
  }

  free(recv_tris);
  free(recv_verts);
  free(r_disp_t);
  free(r_disp_v);

  // =========================================================================
  // PHASE 5: REPAIR TOPOLOGY
  // =========================================================================
  for (mftl_int_t o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];
    MFTLConvertGlobalToLocalIndices(obj);
    MFTLRebuildTwins(obj);
    MFTLFlattenConnectivity(obj);
  }

  return MFTL_SUCCESS;
}

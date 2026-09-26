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
#include "mftl_twin_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_topology.h"
#include "mftl_memory.h"

// --- Global Variable Definitions ---
MPI_Datatype MPI_PackedVertex;
MPI_Datatype MPI_PackedTriangle;
MPI_Datatype MPI_PackedHalfEdge;
MPI_Datatype MPI_PackedTriangleComplex;
MPI_Datatype MPI_GhostUpdatePayload;

// Stage 15: fix every packed datatype's extent to the C struct size. The MPI extent of a
// struct type is implementation-defined in the presence of trailing padding; arrays of
// packed records are exchanged with counts > 1, so the stride MUST equal sizeof(record).
static void mftlResizeToSizeof(MPI_Datatype* type, size_t size)
{
  MPI_Datatype resized;
  MPI_Type_create_resized(*type, 0, (MPI_Aint) size, &resized);
  MPI_Type_free(type);
  MPI_Type_commit(&resized);
  *type = resized;
}

void MFTLRegisterMpiTypes()
{
  // --- 1. Create MPI Datatype for PackedVertex ---
  {
#define NITEMS_VERTEX 14
    int blocklengths[] = {1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id (was MPI_UINT64_T)
        MPI_INT,  // owner_proc (was MPI_INT - likely wrong if owner_proc > 2B or if int is 64bit)
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // roughness_mean (Stage 10)
        MPI_DOUBLE,  // roughness_variance (Stage 10)
        MPI_C_BOOL,  // is_ghost
        MPI_INT,  // flags (was MPI_INT)
        MPI_INT,  // id_he_out (was MPI_UINT64_T)
        MPI_INT,  // body_id (was MPI_UINT64_T)
        MPI_INT  // img[3] (Stage 16B)
    };

    MPI_Aint offsets[NITEMS_VERTEX];
    offsets[0] = offsetof(PackedVertex, global_id);
    offsets[1] = offsetof(PackedVertex, owner_proc);
    offsets[2] = offsetof(PackedVertex, pos);
    offsets[3] = offsetof(PackedVertex, normal);
    offsets[4] = offsetof(PackedVertex, meancurvature);
    offsets[5] = offsetof(PackedVertex, gaussiancurvature);
    offsets[6] = offsetof(PackedVertex, area);
    offsets[7] = offsetof(PackedVertex, roughness_mean);
    offsets[8] = offsetof(PackedVertex, roughness_variance);
    offsets[9] = offsetof(PackedVertex, is_ghost);
    offsets[10] = offsetof(PackedVertex, flags);
    offsets[11] = offsetof(PackedVertex, id_he_out);
    offsets[12] = offsetof(PackedVertex, body_id);
    offsets[13] = offsetof(PackedVertex, img);

    MPI_Type_create_struct(NITEMS_VERTEX, blocklengths, offsets, types, &MPI_PackedVertex);
    mftlResizeToSizeof(&MPI_PackedVertex, sizeof(PackedVertex));
  }

  // --- 2. Create MPI Datatype for PackedTriangle ---
  {
#define NITEMS_TRIANGLE 10
    int blocklengths[] = {1, 1, 1, 3, 1, 3, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // owner_proc
        MPI_INT,  // he_id
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // center[3]
        MPI_C_BOOL,  // is_ghost
        MPI_INT,  // flags
        MPI_INT,  // body_id
        MPI_DOUBLE  // surface_scalar (Stage 15)
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
    offsets[9] = offsetof(PackedTriangle, surface_scalar);

    MPI_Type_create_struct(NITEMS_TRIANGLE, blocklengths, offsets, types, &MPI_PackedTriangle);
    mftlResizeToSizeof(&MPI_PackedTriangle, sizeof(PackedTriangle));
  }

  // --- 3. Create MPI Datatype for PackedHalfEdge ---
  {
#define NITEMS_HALFEDGE 7
    int blocklengths[] = {1, 1, 1, 1, 1, 1, 1};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // vertex_gid
        MPI_INT,  // twin_gid
        MPI_INT,  // next_gid
        MPI_INT,  // triangle_gid
        MPI_C_BOOL,  // boundary
        MPI_INT  // flags
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
    mftlResizeToSizeof(&MPI_PackedHalfEdge, sizeof(PackedHalfEdge));
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
    mftlResizeToSizeof(&MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex));
  }

  // --- 5. Create MPI Datatype for MFTL_GhostUpdatePayload ---
  {
#define NITEMS_GHOST 10
    int blocklengths[] = {1, 1, 3, 3, 1, 1, 1, 1, 1, 3};

    MPI_Datatype types[] = {
        MPI_INT,  // global_id
        MPI_INT,  // found (Stage 10)
        MPI_DOUBLE,  // pos[3]
        MPI_DOUBLE,  // normal[3]
        MPI_DOUBLE,  // meancurvature
        MPI_DOUBLE,  // gaussiancurvature
        MPI_DOUBLE,  // area
        MPI_DOUBLE,  // roughness_mean (Stage 10)
        MPI_DOUBLE,  // roughness_variance (Stage 10)
        MPI_INT,  // img[3] (Stage 16B)
    };

    MPI_Aint offsets[NITEMS_GHOST];
    offsets[0] = offsetof(GhostUpdatePayload, global_id);
    offsets[1] = offsetof(GhostUpdatePayload, found);
    offsets[2] = offsetof(GhostUpdatePayload, pos);
    offsets[3] = offsetof(GhostUpdatePayload, normal);
    offsets[4] = offsetof(GhostUpdatePayload, meancurvature);
    offsets[5] = offsetof(GhostUpdatePayload, gaussiancurvature);
    offsets[6] = offsetof(GhostUpdatePayload, area);
    offsets[7] = offsetof(GhostUpdatePayload, roughness_mean);
    offsets[8] = offsetof(GhostUpdatePayload, roughness_variance);
    offsets[9] = offsetof(GhostUpdatePayload, img);

    MPI_Type_create_struct(NITEMS_GHOST, blocklengths, offsets, types, &MPI_GhostUpdatePayload);
    mftlResizeToSizeof(&MPI_GhostUpdatePayload, sizeof(GhostUpdatePayload));
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
  int* data;
  int size;
  int capacity;
} IntBuffer;

// Initialize once
void buffer_init(IntBuffer* buf, int initial_cap)
{
  buf->data = (int*) malloc(initial_cap * sizeof(int));
  buf->size = 0;
  buf->capacity = initial_cap;
}

// Fast append with geometric growth
static inline void buffer_add(IntBuffer* buf, int val)
{
  MFTL_ENSURE_CAPACITY(buf->data, &buf->size, &buf->capacity, 1, int);
  buf->data[buf->size++] = val;
}

// Reset without freeing memory
static inline void buffer_reset(IntBuffer* buf) { buf->size = 0; }

/**
 * @brief Calculates robust initial buffer capacities based on mesh density.
 */
static inline void MFTLCalcBufferCapacities(int n_tris, int n_verts, int nproc, int* out_tri_cap, int* out_vert_cap)
{
  double factor = 0.2;  // Safety factor: assume up to 20% of data might move to one rank
  double avg_tris = (double) n_tris / (double) nproc;
  double avg_verts = (double) n_verts / (double) nproc;

  *out_tri_cap = (int) (avg_tris * factor);
  *out_vert_cap = (int) (avg_verts * factor);

  // Clamp to reasonable range [1024, 1MB] to prevent tiny allocs or massive over-allocs
  *out_tri_cap = MFTL_MIN(MFTL_MAX(1024, *out_tri_cap), 1000000);
  *out_vert_cap = MFTL_MIN(MFTL_MAX(1024, *out_vert_cap), 1000000);
}

/**
 * @brief Packs a single triangle and its half-edges into the transfer struct.
 */
void MFTLPackSingleTriangle(MFTL_Object* obj, int tri_idx, PackedTriangleComplex* dst, bool is_ghost)
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
  dst->triangle.surface_scalar = src->surface_scalar;
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
    dst->half_edges[j].twin_gid = MFTLHeTwinCode(obj, t_he[j]);  // state code: gid / boundary code (Stage 16A)
    dst->half_edges[j].next_gid = MFTLHeNext(obj, t_he[j])->global_id;
    dst->half_edges[j].triangle_gid = MFTLHeTriangle(obj, t_he[j])->global_id;
    dst->half_edges[j].boundary = MFTLHeIsBoundary(t_he[j]);  // re-derived at destination
    dst->half_edges[j].flags = t_he[j]->flags;
  }
}

/**
 * @brief Packs a single vertex into the transfer struct.
 */
void MFTLPackSingleVertex(MFTL_Object* obj, int vert_idx, PackedVertex* dst, bool is_ghost)
{
  MFTL_Vertex* src = &obj->vertices[vert_idx];

  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->meancurvature = src->meancurvature;
  dst->gaussiancurvature = src->gaussiancurvature;
  dst->area = src->area;
  dst->roughness_mean = src->roughness_mean;
  dst->roughness_variance = src->roughness_variance;
  dst->is_ghost = is_ghost ? MFTL_TRUE : MFTL_FALSE;
  dst->flags = src->flags;
  dst->id_he_out = MFTLVOutHe(obj, src)->global_id;
  dst->body_id = obj->global_id;

  memcpy(dst->pos, src->pos, 3 * sizeof(double));
  memcpy(dst->img, src->img, 3 * sizeof(int));  // Stage 16B
  memcpy(dst->normal, src->normal, 3 * sizeof(double));
}

/**
 * @brief Wipes an MFTL_Object clean to prepare for reconstruction.
 */
static inline void MFTLResetObjectData(MFTL_Object* obj)
{
  // Free per-triangle user payloads before the triangle array itself (Stage 12)
  if (obj->data_size > 0 && obj->triangles)
  {
    for (int i = 0; i < obj->n_triangles; ++i) free(obj->triangles[i].data);
  }

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
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, domain->comm_cart);

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
    MPI_Neighbor_alltoall(n_send, 1, MPI_INT, n_recv, 1, MPI_INT, domain->comm_graph);

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
 * @brief Stage 12: collective contract check — obj->data_size must be identical
 * on every rank before any triangle-payload transport. The payload streams
 * derive their byte counts and offsets from the LOCAL data_size, so a
 * cross-rank mismatch would silently corrupt them; refuse loudly instead.
 *
 * Cheap: two tiny Allreduces over obj->domain->comm_cart.
 *
 * @return MFTL_SUCCESS when all ranks agree, MFTL_ERROR on mismatch (nothing
 *         has been communicated or modified in that case).
 */
int MFTLCheckTriangleDataSizeConsistency(MFTL_Object* obj)
{
  unsigned long long local = (unsigned long long) obj->data_size;
  unsigned long long lo = 0, hi = 0;
  MPI_Allreduce(&local, &lo, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, obj->domain->comm_cart);
  MPI_Allreduce(&local, &hi, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, obj->domain->comm_cart);
  if (lo != hi)
  {
    fprintf(stderr,
            "FATAL ERROR: MFTLCheckTriangleDataSizeConsistency: object (global_id %d) has data_size "
            "min=%llu / max=%llu across ranks (this rank: %llu) — MFTLInitializeObjectData must be called "
            "with the SAME size on all ranks before any transport. Refusing to communicate.\n",
            obj->global_id, lo, hi, local);
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/**
 * @brief Stage 12 helper: payload size (data_size) of the object with the given
 * body id. Same linear-search idiom as the target_obj lookups in the
 * redistribution rebuild passes.
 */
static inline size_t mftl_data_size_of_body(MFTL_Object** objects, int num_objects, int body_id)
{
  for (int o = 0; o < num_objects; ++o)
  {
    if (objects[o]->global_id == body_id) return objects[o]->data_size;
  }
  return 0;
}

/**
 * @brief Reorders the triangle array in-place based on the sorted indices.
 */
static inline void permute_triangles(PackedTriangleComplex* array, const int* indices, int count)
{
  assert(count > 0);

  // 1. Allocate temporary buffer
  PackedTriangleComplex* tmp = (PackedTriangleComplex*) malloc(count * sizeof(PackedTriangleComplex));
  assert(tmp != MFTL_NULL);  // Safety check

  // 2. Gather data into temp buffer in sorted order
  // Random Read (from array) -> Sequential Write (to tmp)
  for (int i = 0; i < count; ++i)
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
static inline void permute_vertices(PackedVertex* array, const int* indices, int count)
{
  assert(count > 0);

  PackedVertex* tmp = (PackedVertex*) malloc(count * sizeof(PackedVertex));
  assert(tmp != MFTL_NULL);  // Safety check

  for (int i = 0; i < count; ++i)
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
 * Per-triangle user payloads (MFTL_Object.data_size > 0) ride in a parallel
 * MPI_BYTE stream in the same element order as the packed triangles.
 */
int MFTLRedistributeListOfObjects(MFTL_Ensemble* ensemble)
{
  if (ensemble->num_objects == 0 || ensemble->objects == MFTL_NULL) return MFTL_ERROR;

  MFTL_Domain* domain = &ensemble->domain;
  int num_objects = ensemble->num_objects;
  MFTL_Object** objects = ensemble->objects;

  if (domain->nproc < 2) return MFTL_SUCCESS;  // No redistribution needed if there is only one process.

  // Stage 12: payload byte counts are derived from the LOCAL data_size of each
  // object — a cross-rank mismatch would silently corrupt the streams. Refuse
  // loudly before touching anything. (Per-object check: data_size may differ
  // BETWEEN objects, but must be identical per object across ranks.)
  for (int o = 0; o < num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  // --- Phase 0: Setup and MPI Queries ---
  int nproc = domain->nproc;
  int my_rank = domain->rank;
  int requires_global_comm_local = 0;

  // Create a fast O(1) lookup map locally
  char* is_neighbor_map = (char*) calloc(nproc, sizeof(char));
  for (int i = 0; i < domain->num_neighbors; ++i)
  {
    is_neighbor_map[domain->neighbors[i]] = 1;
  }

  // 1. Transient Buffers: Used for sorting/uniquing current object's data
  // We need separate lists for Triangles and Vertices
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* master_buf = calloc(nproc, sizeof(IntBuffer));

  int* total_tris_all_objects = calloc(nproc, sizeof(int));
  int* total_verts_all_objects = calloc(nproc, sizeof(int));
  int* total_data_bytes = calloc(nproc, sizeof(int));  // Stage 12: per-rank payload byte counts

  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_data_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));  // Stage 12: parallel MPI_BYTE stream

  // Pre-allocate to avoid early reallocs (Heuristic: 1024 ints)
  int total_global_tris = 0;
  int total_global_verts = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    total_global_tris += object->global_n_triangles;
    total_global_verts += object->global_n_vertices;
  }

  int init_tri_cap, init_vert_cap;
  MFTLCalcBufferCapacities(total_global_tris, total_global_verts, nproc, &init_tri_cap, &init_vert_cap);

  for (int r = 0; r < nproc; ++r)
  {
    buffer_init(&trans_tri_bufs[r], init_tri_cap);  // Transient Triangles
    buffer_init(&trans_vert_bufs[r], init_vert_cap);  // Transient Vertices
    buffer_init(&master_buf[r], init_tri_cap + init_vert_cap);  // Master Buffer
  }

  // --- Phase 1: Packing and Analysis ---

  // Store owners in simple flat arrays, not inside structs scattered in memory
  // Do this outside the loop to avoid heap thrashing and realloc if size is not enough.
  int last_size_tris = objects[0]->n_triangles;
  int last_size_verts = objects[0]->n_vertices;
  int* vert_owners = malloc(last_size_verts * sizeof(int));
  int* tri_owners = malloc(last_size_tris * sizeof(int));

  // Dummy variable: We always start writing from index 0 for each object.
  // We use this to tell the macro "The buffer is effectively empty right now".
  int zero_count = 0;

  // 1a. Calculate destinations
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    MFTL_Triangle* tri_base = object->triangles;
    MFTL_Vertex* vert_base = object->vertices;
    int n_triangles = object->n_triangles;
    int n_vertices = object->n_vertices;

    // A. RESET Transient Buffers
    for (int r = 0; r < nproc; ++r)
    {
      buffer_reset(&trans_tri_bufs[r]);
      buffer_reset(&trans_vert_bufs[r]);
    }

    MFTL_ENSURE_CAPACITY(tri_owners, &zero_count, &last_size_tris, n_triangles, int);
    MFTL_ENSURE_CAPACITY(vert_owners, &zero_count, &last_size_verts, n_vertices, int);

    last_size_tris = n_triangles;
    last_size_verts = n_vertices;

    // Rebuild the flattened connectivity: remeshing may have changed the
    // topology since the arrays were last built (the previous lazy init
    // only rebuilt when NULL, which silently shipped stale connectivity
    // when redistributing a remeshed mesh — Stage 10).
    MFTLFlattenConnectivity(object);

    // Direct pointers to object memory (Zero Copy)
    int* flat_tri_verts = object->flat_tri_verts;
    int* flat_tri_neighbors = object->flat_tri_neighbors;

    // --- Pre-calculate Vertex Owners ---
    for (int i = 0; i < n_vertices; ++i)
    {
      MFTL_Vertex* v = &vert_base[i];
      v->owner_proc = MFTLGetOwnerProc(domain, v->pos);
      vert_owners[i] = v->owner_proc;
    }

    // --- Pre-calculate Triangle Owners ---
    for (int i = 0; i < n_triangles; ++i)
    {
      MFTL_Triangle* tri = &tri_base[i];
      tri->owner_proc = MFTLGetOwnerProc(domain, tri->center);
      tri_owners[i] = tri->owner_proc;
    }

    for (int i = 0; i < n_triangles; ++i)
    {
      // This prevents "Ghost Forwarding" crashes and automatically cleans up stale ghosts.
      // Don't use owner_proc because it already contains the new owner.
      if (tri_base[i].is_ghost)
      {
        continue;
      }

      int distinct_ranks[7];
      int count = 0;

      // --- 1. Who owns me? ---
      distinct_ranks[count++] = tri_owners[i];

      // --- 2. Who owns my vertices? ---
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 1]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 2]];

      // --- 3. Who owns my neighbors? ---
      for (int j = 0; j < 3; ++j)
      {
        int adj_idx = flat_tri_neighbors[3 * i + j];

        if (adj_idx != -1)
        {
          distinct_ranks[count++] = tri_owners[adj_idx];
        }
      }

      int unique_dests[7];
      int num_unique_dests = MFTLGetUniqueIntegers(distinct_ranks, count, unique_dests, 7);

      // Loop through the unique destinations
      for (int k = 0; k < num_unique_dests; ++k)
      {
        int dest_rank = unique_dests[k];
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

    // Authority completeness: a non-ghost vertex with NO non-ghost incident
    // triangle (happens transiently near partition boundaries, and for
    // seam-stretched rings on periodic domains) never enters any triangle's
    // vertex list, so its ownership-authoritative copy would not ship and the
    // receiver's local/ghost classification (which only trusts non-ghost
    // sender copies) would leave it ownerless. Send such vertices explicitly
    // to their recomputed owner.
    if (n_vertices > 0)
    {
      char* covered = (char*) calloc((size_t) n_vertices, 1);
      for (int i = 0; i < n_triangles; ++i)
      {
        if (tri_base[i].is_ghost) continue;
        covered[flat_tri_verts[3 * i + 0]] = 1;
        covered[flat_tri_verts[3 * i + 1]] = 1;
        covered[flat_tri_verts[3 * i + 2]] = 1;
      }
      for (int i = 0; i < n_vertices; ++i)
      {
        if (vert_base[i].is_ghost || covered[i]) continue;
        int dest_rank = vert_owners[i];
        if (dest_rank != my_rank && !is_neighbor_map[dest_rank]) requires_global_comm_local = 1;
        buffer_add(&trans_vert_bufs[dest_rank], i);
      }
      free(covered);
    }

    // Sort -> Unique
    for (int r = 0; r < nproc; ++r)
    {
      // Skip if nothing to send to this rank
      if (trans_tri_bufs[r].size == 0 && trans_vert_bufs[r].size == 0)
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
      int n_unique_tris = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      // 2. Sort & Unique Vertices
      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      int n_unique_verts = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      // 3. Append to master buf ({object_id, n_unique_tris, n_unique_verts, triangle_ids, vertex_ids})
      total_tris_all_objects[r] += n_unique_tris;
      total_verts_all_objects[r] += n_unique_verts;
      total_data_bytes[r] += n_unique_tris * (int) object->data_size;  // Stage 12
      buffer_add(&master_buf[r], object->global_id);
      buffer_add(&master_buf[r], n_unique_tris);
      buffer_add(&master_buf[r], n_unique_verts);
      for (int i = 0; i < n_unique_tris; ++i)
      {
        buffer_add(&master_buf[r], trans_tri_bufs[r].data[i]);
      }
      for (int i = 0; i < n_unique_verts; ++i)
      {
        buffer_add(&master_buf[r], trans_vert_bufs[r].data[i]);
      }
    }
  }

  free(vert_owners);
  free(tri_owners);

  // Allocate send buffers
  for (int r = 0; r < nproc; ++r)
  {
    send_tri_bufs[r].buffer = malloc(total_tris_all_objects[r] * sizeof(PackedTriangleComplex));
    send_tri_bufs[r].capacity = total_tris_all_objects[r];
    send_tri_bufs[r].count = 0;  // Initialize packing index to 0

    send_vert_bufs[r].buffer = malloc(total_verts_all_objects[r] * sizeof(PackedVertex));
    send_vert_bufs[r].capacity = total_verts_all_objects[r];
    send_vert_bufs[r].count = 0;  // Initialize packing index to 0

    // Stage 12: opaque payload bytes; 'count' doubles as the byte cursor during packing
    send_data_bufs[r].buffer = malloc(total_data_bytes[r]);
    send_data_bufs[r].capacity = total_data_bytes[r];
    send_data_bufs[r].count = 0;
  }

  // Pack MFTL Send Buffers with help of master buffer
  for (int r = 0; r < nproc; ++r)
  {
    int master_idx = 0;

    // Safety casts for correct pointer arithmetic
    PackedTriangleComplex* tri_buf_typed = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
    PackedVertex* vert_buf_typed = (PackedVertex*) send_vert_bufs[r].buffer;

    for (int o = 0; o < num_objects; ++o)
    {
      MFTL_Object* object = objects[o];
      MFTL_Triangle* tri_base = object->triangles;
      MFTL_Vertex* vert_base = object->vertices;

      int object_id = master_buf[r].data[master_idx++];
      assert(object->global_id == object_id);

      int n_unique_tris = master_buf[r].data[master_idx++];
      int n_unique_verts = master_buf[r].data[master_idx++];

      // 1. Pack Triangles
      for (int i = master_idx; i < master_idx + n_unique_tris; ++i)
      {
        int tri_idx = master_buf[r].data[i];

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
        packed_tri->triangle.surface_scalar = tri_base[tri_idx].surface_scalar;

        MFTL_HalfEdge* t_he[3];
        MFTLGetTriangleHalfedges(object, &tri_base[tri_idx], t_he);
        for (int j = 0; j < 3; ++j)
        {
          PackedHalfEdge* he = &packed_tri->half_edges[j];
          he->global_id = t_he[j]->global_id;
          he->vertex_gid = MFTLHeTarget(object, t_he[j])->global_id;
          he->twin_gid = MFTLHeTwinCode(object, t_he[j]);  // state code: gid / boundary code (Stage 16A)
          he->next_gid = MFTLHeNext(object, t_he[j])->global_id;
          he->triangle_gid = MFTLHeTriangle(object, t_he[j])->global_id;
          he->boundary = MFTLHeIsBoundary(t_he[j]);
          he->flags = t_he[j]->flags;
        }

        // Stage 12: append the opaque payload to the parallel byte stream,
        // in the exact same element order as the packed triangles.
        if (object->data_size > 0)
        {
          if (tri_base[tri_idx].data == MFTL_NULL)
          {
            fprintf(stderr,
                    "FATAL ERROR: MFTLRedistributeListOfObjects: triangle (gid %d, body %d) has NULL data "
                    "but data_size=%zu — call MFTLInitializeObjectData on all ranks before redistributing.\n",
                    tri_base[tri_idx].global_id, object->global_id, object->data_size);
            exit(EXIT_FAILURE);
          }
          memcpy((char*) send_data_bufs[r].buffer + send_data_bufs[r].count, tri_base[tri_idx].data, object->data_size);
          send_data_bufs[r].count += (int) object->data_size;
        }
      }

      // 2. Pack Vertices
      for (int i = master_idx + n_unique_tris; i < master_idx + n_unique_tris + n_unique_verts; ++i)
      {
        int vert_idx = master_buf[r].data[i];

        // FIX: Use typed pointer for indexing
        PackedVertex* packed_vert = &vert_buf_typed[send_vert_bufs[r].count++];

        packed_vert->global_id = vert_base[vert_idx].global_id;
        packed_vert->owner_proc = vert_base[vert_idx].owner_proc;
        memcpy(packed_vert->pos, vert_base[vert_idx].pos, 3 * sizeof(double));
        memcpy(packed_vert->img, vert_base[vert_idx].img, 3 * sizeof(int));  // Stage 16B
        memcpy(packed_vert->normal, vert_base[vert_idx].normal, 3 * sizeof(double));
        packed_vert->meancurvature = vert_base[vert_idx].meancurvature;
        packed_vert->gaussiancurvature = vert_base[vert_idx].gaussiancurvature;
        packed_vert->roughness_mean = vert_base[vert_idx].roughness_mean;
        packed_vert->roughness_variance = vert_base[vert_idx].roughness_variance;
        packed_vert->area = vert_base[vert_idx].area;
        // Truthful sender-side status: the receiver uses this to tell
        // ownership-authoritative copies (sender holds the vertex as
        // non-ghost, fresh position) from ghost-derived copies (possibly
        // stale position => possibly wrong owner label). See the vertex
        // reconstruction below.
        packed_vert->is_ghost = vert_base[vert_idx].is_ghost;
        packed_vert->flags = vert_base[vert_idx].flags;
        packed_vert->id_he_out = MFTLVOutHe(object, &vert_base[vert_idx])->global_id;
        packed_vert->body_id = object->global_id;
      }

      master_idx += n_unique_tris + n_unique_verts;
    }
  }

  // Cleanup Transient Buffers
  for (int r = 0; r < nproc; ++r)
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
  char* kept_data = MFTL_NULL;  // Stage 12
  int kept_count_tris = 0, kept_count_verts = 0, kept_data_bytes = 0;

  // Look for our own rank in the triangle send buffer
  // Do the same for vertices
  int own_rank = domain->rank;
  for (int r = 0; r < nproc; ++r)
  {
    if (r == own_rank)
    {
      // We can "steal" the buffer pointer directly instead of copying.
      kept_tris = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      kept_verts = (PackedVertex*) send_vert_bufs[r].buffer;
      kept_count_tris = send_tri_bufs[r].count;
      kept_count_verts = send_vert_bufs[r].count;

      // Stage 12: steal the kept payload bytes ('count' is the byte cursor)
      kept_data = (char*) send_data_bufs[r].buffer;
      kept_data_bytes = send_data_bufs[r].count;

      // Clear Triangles
      send_tri_bufs[r].buffer = MFTL_NULL;
      send_tri_bufs[r].count = 0;
      send_tri_bufs[r].capacity = 0;

      // Clear Vertices
      send_vert_bufs[r].buffer = MFTL_NULL;
      send_vert_bufs[r].count = 0;
      send_vert_bufs[r].capacity = 0;

      // Clear Data
      send_data_bufs[r].buffer = MFTL_NULL;
      send_data_bufs[r].count = 0;
      send_data_bufs[r].capacity = 0;
    }
  }

  // --- Phase 2: Decide on Communication Strategy ---
  int global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  // --- Phase 3: Execute Communication ---
  PackedTriangleComplex* recv_buf_tris = MFTL_NULL;
  PackedVertex* recv_buf_verts = MFTL_NULL;
  char* recv_data = MFTL_NULL;  // Stage 12
  int total_recv_tris = 0, total_recv_verts = 0, recv_data_bytes = 0;

  // Use the Generic MPI Exchange Helper
  MFTLMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_buf_tris,
                         &total_recv_tris);

  MFTLMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_buf_verts, &total_recv_verts);

  // Stage 12: the payload byte stream uses the same path decision and element order
  MFTLMpiExchangeBuffers(domain, send_data_bufs, MPI_BYTE, 1, global_decision, (void**) &recv_data, &recv_data_bytes);

  // --- Final Cleanup of Send Buffers ---
  // At this point, all data has been sent. We can free the send maps and the buffers they hold.
  for (int i = 0; i < nproc; ++i)
  {
    free(send_tri_bufs[i].buffer);
    free(send_vert_bufs[i].buffer);
    free(send_data_bufs[i].buffer);
  }

  free(total_tris_all_objects);
  free(total_verts_all_objects);
  free(total_data_bytes);
  free(send_tri_bufs);
  free(send_vert_bufs);
  free(send_data_bufs);
  free(is_neighbor_map);

  // --- Phase 4: Unpack and Rebuild Local Mesh ---

  // Pre-wipe all objects. We are about to rebuild them.
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    MFTLResetObjectData(obj);
  }

  // =========================================================================
  // PART A: TRIANGLE RECONSTRUCTION
  // =========================================================================

  // 4.1. Combine kept and received tris
  int final_tris_count = kept_count_tris + total_recv_tris;
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

  // Stage 12: concatenate the payload byte stream in the SAME element order
  // as all_tris (kept first, then received).
  char* all_data = MFTL_NULL;
  int all_data_bytes = kept_data_bytes + recv_data_bytes;
  if (all_data_bytes > 0)
  {
    all_data = (char*) malloc(all_data_bytes);
    if (kept_data_bytes > 0) memcpy(all_data, kept_data, kept_data_bytes);
    if (recv_data_bytes > 0) memcpy(all_data + kept_data_bytes, recv_data, recv_data_bytes);
  }
  free(kept_data);
  free(recv_data);

  // 4.2. HIERARCHICAL RADIX SORT & REBUILD
  if (final_tris_count > 0)
  {
    // =========================================================
    // PASS 1: GROUP BY BODY_ID
    // =========================================================

    // 1. Allocate Key/Index arrays for the whole batch
    int* keys = malloc(final_tris_count * sizeof(int));
    int* indices = malloc(final_tris_count * sizeof(int));

    // 2. Extract Body IDs as keys
    for (int i = 0; i < final_tris_count; ++i)
    {
      keys[i] = all_tris[i].triangle.body_id;
      indices[i] = i;
    }

    // 3. Radix Sort by Body ID
    // After this, 'indices' tells us how to reorder 'all_tris' so bodies are grouped
    MFTLSortPairs(keys, indices, final_tris_count);

    // Stage 12: per-element payload byte offsets into all_data in the OLD
    // element order, needed to permute the variable-size blobs with all_tris.
    size_t* elem_off = MFTL_NULL;
    if (all_data_bytes > 0)
    {
      elem_off = malloc(final_tris_count * sizeof(size_t));
      size_t run = 0;
      for (int i = 0; i < final_tris_count; ++i)
      {
        elem_off[i] = run;
        run += mftl_data_size_of_body(objects, num_objects, all_tris[i].triangle.body_id);
      }
      assert(run == (size_t) all_data_bytes);
    }

    // 4. Permute the main data array to match the sorted order
    // Now all_tris is physically grouped by object: [Obj1, Obj1, Obj2, Obj2...]
    permute_triangles(all_tris, indices, final_tris_count);

    // Stage 12: gather the payload blobs into the permuted order and record
    // per-element offsets (NEW order) for the fill passes below.
    size_t* data_off = MFTL_NULL;
    if (all_data_bytes > 0)
    {
      char* gathered = malloc(all_data_bytes);
      data_off = malloc(final_tris_count * sizeof(size_t));
      size_t run = 0;
      for (int i = 0; i < final_tris_count; ++i)
      {
        size_t ds = mftl_data_size_of_body(objects, num_objects, all_tris[i].triangle.body_id);
        data_off[i] = run;
        if (ds > 0) memcpy(gathered + run, all_data + elem_off[indices[i]], ds);
        run += ds;
      }
      free(all_data);
      all_data = gathered;
      free(elem_off);
    }

    // =========================================================
    // PASS 2: PROCESS EACH OBJECT (Range -> Sort -> Unique)
    // =========================================================

    int current_idx = 0;
    while (current_idx < final_tris_count)
    {
      // A. Identify Range for current Body ID
      int current_body_id = all_tris[current_idx].triangle.body_id;
      int start_idx = current_idx;

      // Fast scan to find the end of this object
      while (current_idx < final_tris_count && all_tris[current_idx].triangle.body_id == current_body_id)
      {
        current_idx++;
      }
      int end_idx = current_idx;
      int range_len = end_idx - start_idx;

      // Find the actual MFTL_Object pointer
      // TODO: This is a linear search. We should use a hash map.
      MFTL_Object* target_obj = MFTL_NULL;
      for (int o = 0; o < num_objects; ++o)
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
        for (int k = 0; k < range_len; ++k)
        {
          keys[k] = all_tris[start_idx + k].triangle.global_id;
          indices[k] = k;  // Relative index (0 to range_len)
        }

        // C. Radix Sort by Global ID
        MFTLSortPairs(keys, indices, range_len);

        // D. Deduplicate (Unique)
        // Since we have the sorted keys array, we can count uniques purely on keys first
        int n_unique = 0;
        if (range_len > 0)
        {
          n_unique = 1;
          for (int k = 1; k < range_len; ++k)
          {
            // keys[k] is the global_id of the element at sorted position k
            int prev_gid = keys[k - 1];  // Note: keys array is modified by sort_pairs to be sorted
            int curr_gid = keys[k];
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
        // ── PASS 1: Locals ──────────────────────────────────────────────────
        int write_count = 0;
        for (int k = 0; k < range_len; ++k)
        {
          // We use indices[k] to grab the element from the range in sorted order
          int sorted_relative_idx = indices[k];
          PackedTriangleComplex* src = &all_tris[start_idx + sorted_relative_idx];

          // Hash-based dedup: robust even if copies of one gid were to carry
          // conflicting owner labels (triangle copies are single-source — only
          // the unique non-ghost holder ships a triangle — but the same
          // invariant-violation heap overflow as in the vertex path below
          // would result if that ever broke).
          khiter_t k_dup = kh_get(triangle_map, target_obj->triangle_lookup, src->triangle.global_id);
          if (k_dup != kh_end(target_obj->triangle_lookup)) continue;  // already written

          // Skip ghosts in pass 1 (Assume local if owner_proc matches my_rank)
          if (src->triangle.owner_proc != my_rank) continue;

          // Copy Data
          MFTL_Triangle* dst = &target_obj->triangles[write_count];
          dst->global_id = src->triangle.global_id;
          dst->owner_proc = src->triangle.owner_proc;
          dst->flags = src->triangle.flags;
          dst->is_ghost = MFTL_FALSE;
          dst->area = src->triangle.area;
          dst->surface_scalar = src->triangle.surface_scalar;
          memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
          memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
          dst->he_idx = write_count * 3;

          // Stage 12: attach the opaque payload from the parallel byte stream
          if (target_obj->data_size > 0)
          {
            dst->data = malloc(target_obj->data_size);
            memcpy(dst->data, all_data + data_off[start_idx + sorted_relative_idx], target_obj->data_size);
          }
          else
          {
            dst->data = MFTL_NULL;
          }

          int ret;
          khiter_t k_tr = kh_put(triangle_map, target_obj->triangle_lookup, dst->global_id, &ret);
          kh_value(target_obj->triangle_lookup, k_tr) = write_count;  // local index

          // Copy Half Edges
          for (int i = 0; i < 3; ++i)
          {
            MFTL_HalfEdge* he = &target_obj->half_edges[write_count * 3 + i];
            he->global_id = src->half_edges[i].global_id;
            he->flags = src->half_edges[i].flags;
            he->triangle_idx = write_count;
            he->next_idx = write_count * 3 + (i + 1) % 3;
            he->twin_gid = src->half_edges[i].twin_gid;
            he->twin_idx = (src->half_edges[i].twin_gid >= 0) ? src->half_edges[i].twin_gid : MFTL_ERROR;  // GID (re-derived by MFTLRebuildTwins)
            he->vertex_idx = src->half_edges[i].vertex_gid;  // needs to be converted to local index
            he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins (Stage 16A)
            int ret;
            khiter_t k_he = kh_put(halfedge_map, target_obj->halfedge_lookup, he->global_id, &ret);
            kh_value(target_obj->halfedge_lookup, k_he) = write_count * 3 + i;  // local index
          }

          write_count++;
        }

        target_obj->n_local_triangles = write_count;
        target_obj->n_local_half_edges = write_count * 3;

        // ── PASS 2: Ghosts ──────────────────────────────────────────────────
        for (int k = 0; k < range_len; ++k)
        {
          int sorted_relative_idx = indices[k];
          PackedTriangleComplex* src = &all_tris[start_idx + sorted_relative_idx];

          khiter_t k_dup = kh_get(triangle_map, target_obj->triangle_lookup, src->triangle.global_id);
          if (k_dup != kh_end(target_obj->triangle_lookup)) continue;  // already written

          // Skip locals in pass 2
          if (src->triangle.owner_proc == my_rank) continue;

          // Copy Data
          MFTL_Triangle* dst = &target_obj->triangles[write_count];
          dst->global_id = src->triangle.global_id;
          dst->owner_proc = src->triangle.owner_proc;
          dst->flags = src->triangle.flags;
          dst->is_ghost = MFTL_TRUE;
          dst->area = src->triangle.area;
          dst->surface_scalar = src->triangle.surface_scalar;
          memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
          memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
          dst->he_idx = write_count * 3;

          // Stage 12: duplicate copies carry identical bytes; the duplicate-skip
          // logic above is unchanged.
          if (target_obj->data_size > 0)
          {
            dst->data = malloc(target_obj->data_size);
            memcpy(dst->data, all_data + data_off[start_idx + sorted_relative_idx], target_obj->data_size);
          }
          else
          {
            dst->data = MFTL_NULL;
          }

          int ret;
          khiter_t k_tr = kh_put(triangle_map, target_obj->triangle_lookup, dst->global_id, &ret);
          kh_value(target_obj->triangle_lookup, k_tr) = write_count;

          for (int i = 0; i < 3; ++i)
          {
            MFTL_HalfEdge* he = &target_obj->half_edges[write_count * 3 + i];
            he->global_id = src->half_edges[i].global_id;
            he->flags = src->half_edges[i].flags;
            he->triangle_idx = write_count;
            he->next_idx = write_count * 3 + (i + 1) % 3;
            he->twin_gid = src->half_edges[i].twin_gid;
            he->twin_idx = (src->half_edges[i].twin_gid >= 0) ? src->half_edges[i].twin_gid : MFTL_ERROR;  // GID (re-derived by MFTLRebuildTwins)
            he->vertex_idx = src->half_edges[i].vertex_gid;
            he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins (Stage 16A)
            khiter_t k_he = kh_put(halfedge_map, target_obj->halfedge_lookup, he->global_id, &ret);
            kh_value(target_obj->halfedge_lookup, k_he) = write_count * 3 + i;
          }
          write_count++;
        }
        MFTL_ASSERT(write_count == n_unique && "Error: write_count != n_unique\n");
      }
    }

    // Cleanup
    free(keys);
    free(indices);
    free(all_tris);
    free(all_data);  // Stage 12
    free(data_off);  // Stage 12
  }

  // =========================================================================
  // PART B: VERTEX RECONSTRUCTION
  // =========================================================================

  // 4.3. Combine kept and received vertices
  int final_verts_count = kept_count_verts + total_recv_verts;
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
    int* keys = malloc(final_verts_count * sizeof(int));
    int* indices = malloc(final_verts_count * sizeof(int));

    // 2. Extract Body IDs
    for (int i = 0; i < final_verts_count; ++i)
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

    int current_idx = 0;
    while (current_idx < final_verts_count)
    {
      // A. Identify Range
      int current_body_id = all_verts[current_idx].body_id;
      int start_idx = current_idx;

      while (current_idx < final_verts_count && all_verts[current_idx].body_id == current_body_id)
      {
        current_idx++;
      }
      int range_len = current_idx - start_idx;

      // Find MFTL_Object
      // TODO: This is a linear search. We should use a hash map.
      MFTL_Object* target_obj = MFTL_NULL;
      for (int o = 0; o < num_objects; ++o)
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
        for (int k = 0; k < range_len; ++k)
        {
          keys[k] = all_verts[start_idx + k].global_id;
          indices[k] = k;
        }

        // C. Local Radix Sort (Global ID)
        MFTLSortPairs(keys, indices, range_len);

        // D. Count Uniques
        int n_unique = 0;
        if (range_len > 0)
        {
          n_unique = 1;
          for (int k = 1; k < range_len; ++k)
          {
            int prev_gid = keys[k - 1];
            int curr_gid = keys[k];
            if (curr_gid != prev_gid) n_unique++;
          }
        }

        // E. Reallocate Object Memory
        target_obj->n_vertices = n_unique;
        target_obj->capacity_vertices = n_unique;
        target_obj->vertices = malloc(n_unique * sizeof(MFTL_Vertex));

        // F. Fill Target Object
        // Ownership authority: a vertex becomes LOCAL here only when a copy
        // arrives from a sender holding it as NON-ghost (authoritative copy —
        // the sender recomputed the owner label from its own fresh position).
        // Ghost-derived copies may carry stale positions — and hence WRONG
        // owner labels — whenever a ghost's owner sits outside this rank's
        // communication neighbourhood so no refresh was possible (Stage-10
        // contract: such ghosts stay stale until the next redistribution).
        // Periodic seam-straddlers make this systematic: wrapped storage
        // stretches their triangles across the whole domain in raw
        // coordinates, so their vertices are owned by ranks 2+ slabs away
        // (never refreshed), and a vertex wrapped across the seam jumps a
        // full period, flipping its slab label. Classifying such copies by
        // their stale label wrote the same gid BOTH as local and as ghost,
        // overflowing this n_unique-sized array (heap corruption) and
        // clobbering vertex_lookup. Dedup is therefore hash-based and only
        // authoritative copies can create locals.
        // ── PASS 1: Locals ──────────────────────────────────────────────────
        int write_count = 0;
        for (int k = 0; k < range_len; ++k)
        {
          int sorted_relative_idx = indices[k];
          PackedVertex* src = &all_verts[start_idx + sorted_relative_idx];

          if (src->is_ghost != MFTL_FALSE) continue;  // ghost-derived copy: not ownership-authoritative
          if (src->owner_proc != my_rank) continue;   // owned elsewhere

          khiter_t k_dup = kh_get(vertex_map, target_obj->vertex_lookup, src->global_id);
          if (k_dup != kh_end(target_obj->vertex_lookup)) continue;  // already written

          // Copy Data
          MFTL_Vertex* dst = &target_obj->vertices[write_count];
          dst->global_id = src->global_id;
          dst->owner_proc = src->owner_proc;
          dst->flags = src->flags;
          dst->is_ghost = MFTL_FALSE;
          dst->area = src->area;
          dst->meancurvature = src->meancurvature;
          dst->gaussiancurvature = src->gaussiancurvature;
          dst->roughness_mean = src->roughness_mean;
          dst->roughness_variance = src->roughness_variance;

          memcpy(dst->pos, src->pos, 3 * sizeof(double));
          memcpy(dst->img, src->img, 3 * sizeof(int));  // Stage 16B
          memcpy(dst->normal, src->normal, 3 * sizeof(double));

          dst->out_he_idx = src->id_he_out;  // needs to be converted to local index
          int ret;
          khiter_t k_v = kh_put(vertex_map, target_obj->vertex_lookup, dst->global_id, &ret);
          kh_value(target_obj->vertex_lookup, k_v) = write_count;  // local index

          write_count++;
        }

        target_obj->n_local_vertices = write_count;

        // ── PASS 2: Ghosts ──────────────────────────────────────────────────
        // Data precedence: authoritative (sender-local) copies first, so a
        // ghost's payload comes from the owner's fresh data whenever it is
        // available; ghost-derived (possibly stale) copies fill the rest.
        for (int authoritative = 1; authoritative >= 0; --authoritative)
        {
          for (int k = 0; k < range_len; ++k)
          {
            int sorted_relative_idx = indices[k];
            PackedVertex* src = &all_verts[start_idx + sorted_relative_idx];

            if ((src->is_ghost == MFTL_FALSE) != (authoritative == 1)) continue;  // class filter

            khiter_t k_dup = kh_get(vertex_map, target_obj->vertex_lookup, src->global_id);
            if (k_dup != kh_end(target_obj->vertex_lookup)) continue;  // already written

            MFTL_Vertex* dst = &target_obj->vertices[write_count];
            dst->global_id = src->global_id;
            dst->owner_proc = src->owner_proc;
            dst->flags = src->flags;
            dst->is_ghost = MFTL_TRUE;
            dst->area = src->area;
            dst->meancurvature = src->meancurvature;
            dst->gaussiancurvature = src->gaussiancurvature;
            dst->roughness_mean = src->roughness_mean;
            dst->roughness_variance = src->roughness_variance;
            memcpy(dst->pos, src->pos, 3 * sizeof(double));
            memcpy(dst->img, src->img, 3 * sizeof(int));  // Stage 16B
            memcpy(dst->normal, src->normal, 3 * sizeof(double));
            dst->out_he_idx = src->id_he_out;

            int ret;
            khiter_t k_v = kh_put(vertex_map, target_obj->vertex_lookup, dst->global_id, &ret);
            kh_value(target_obj->vertex_lookup, k_v) = write_count;

            write_count++;
          }
        }
        MFTL_ASSERT(write_count == n_unique && "Error: write_count != n_unique\n");
      }
    }

    free(keys);
    free(indices);
    free(all_verts);
  }

  // Rebuild twin pointers across the new local mesh
  for (int o = 0; o < num_objects; ++o)
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

  // Stage 16A (plans/16 §2.5, §2.3): guaranteed ghost completion G1/G2, then the quiescent check.
  // One collective decides for all objects which ones need completion rounds.
  int rc = MFTL_SUCCESS;
  int* need = (int*) calloc((size_t) (num_objects > 0 ? num_objects : 1), sizeof(int));
  int* gneed = (int*) calloc((size_t) (num_objects > 0 ? num_objects : 1), sizeof(int));
  for (int o = 0; o < num_objects; ++o) need[o] = MFTLInternalCountGhostRequests(objects[o]);
  MPI_Allreduce(need, gneed, num_objects, MPI_INT, MPI_SUM, domain->comm_cart);
  for (int o = 0; o < num_objects; ++o)
  {
    if (gneed[o] > 0 && MFTLCompleteGhosts(objects[o], 0.0, MFTL_NULL) != MFTL_SUCCESS) rc = MFTL_ERROR;
    if (MFTLInternalQuiescentCheck(objects[o], MFTL_VALIDATE_GHOSTS) != MFTL_SUCCESS) rc = MFTL_ERROR;
  }
  free(need);
  free(gneed);
  return rc;
}

/**
 * @brief Redistributes a SINGLE object. For cases with 1 or very few objects.
 * @details Removes the overhead of sorting/grouping multiple bodies.
 * Assumes all ranks call this for the SAME object at the same time.
 */
int MFTLRedistributeObject(MFTL_Object* object)
{
  assert(object);

  MFTL_Domain* domain = object->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  // Stage 12: payload byte counts derive from the LOCAL data_size — refuse
  // loudly on a cross-rank mismatch before touching anything.
  if (MFTLCheckTriangleDataSizeConsistency(object) != MFTL_SUCCESS) return MFTL_ERROR;

  int my_rank = domain->rank;
  int requires_global_comm_local = 0;

  // --- 1. Setup ---
  // Fast neighbor lookup
  char* is_neighbor_map = calloc(nproc, sizeof(char));
  for (int i = 0; i < domain->num_neighbors; ++i) is_neighbor_map[domain->neighbors[i]] = 1;

  // Buffers
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_data_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));  // Stage 12: parallel MPI_BYTE stream

  const size_t ds = object->data_size;  // Stage 12: uniform per-element payload size

  // Heuristic Allocation
  int tri_cap, vert_cap;
  MFTLCalcBufferCapacities(object->global_n_triangles, object->global_n_vertices, nproc, &tri_cap, &vert_cap);

  for (int r = 0; r < nproc; ++r)
  {
    buffer_init(&trans_tri_bufs[r], tri_cap);
    buffer_init(&trans_vert_bufs[r], vert_cap);
  }

  // --- 2. Analysis (Fill Transient Buffers) ---
  MFTL_Triangle* tri_base = object->triangles;
  MFTL_Vertex* vert_base = object->vertices;
  int n_triangles = object->n_triangles;
  int n_vertices = object->n_vertices;

  int* vert_owners = malloc(n_vertices * sizeof(int));
  int* tri_owners = malloc(n_triangles * sizeof(int));

  // Rebuild the flattened connectivity: remeshing may have changed the
  // topology since the arrays were last built (the previous lazy init
  // only rebuilt when NULL, which silently shipped stale connectivity
  // when redistributing a remeshed mesh — Stage 10).
  MFTLFlattenConnectivity(object);

  // Direct pointers to object memory (Zero Copy)
  int* flat_tri_verts = object->flat_tri_verts;
  int* flat_tri_neighbors = object->flat_tri_neighbors;

  // Pre-calculate Owners
  for (int i = 0; i < n_vertices; ++i)
  {
    MFTL_Vertex* v = &vert_base[i];
    v->owner_proc = MFTLGetOwnerProc(domain, v->pos);
    vert_owners[i] = v->owner_proc;
  }
  for (int i = 0; i < n_triangles; ++i)
  {
    MFTL_Triangle* tri = &tri_base[i];
    tri->owner_proc = MFTLGetOwnerProc(domain, tri->center);
    tri_owners[i] = tri->owner_proc;
  }

  // Loop over all triangles to determine destinations
  for (int i = 0; i < n_triangles; ++i)
  {
    // This prevents "Ghost Forwarding" crashes and automatically cleans up stale ghosts.
    // Don't use owner_proc because it already contains the new owner.
    if (tri_base[i].is_ghost)
    {
      continue;
    }

    int distinct_ranks[7];
    int count = 0;

    distinct_ranks[count++] = tri_owners[i];  // Triangle Owner
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];  // Vertex Owners
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 1]];
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 2]];

    for (int j = 0; j < 3; ++j)
    {
      int adj_idx = flat_tri_neighbors[3 * i + j];
      if (adj_idx != -1) distinct_ranks[count++] = tri_owners[adj_idx];
    }

    int unique_dests[7];
    int num_unique_dests = MFTLGetUniqueIntegers(distinct_ranks, count, unique_dests, 7);

    for (int k = 0; k < num_unique_dests; ++k)
    {
      int dest_rank = unique_dests[k];
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

  // Authority completeness (see MFTLRedistributeListOfObjects): a non-ghost
  // vertex with no non-ghost incident triangle would not ship its
  // ownership-authoritative copy; send such vertices explicitly to their
  // recomputed owner.
  if (n_vertices > 0)
  {
    char* covered = (char*) calloc((size_t) n_vertices, 1);
    for (int i = 0; i < n_triangles; ++i)
    {
      if (tri_base[i].is_ghost) continue;
      covered[flat_tri_verts[3 * i + 0]] = 1;
      covered[flat_tri_verts[3 * i + 1]] = 1;
      covered[flat_tri_verts[3 * i + 2]] = 1;
    }
    for (int i = 0; i < n_vertices; ++i)
    {
      if (vert_base[i].is_ghost || covered[i]) continue;
      int dest_rank = vert_owners[i];
      if (dest_rank != my_rank && !is_neighbor_map[dest_rank]) requires_global_comm_local = 1;
      buffer_add(&trans_vert_bufs[dest_rank], i);
    }
    free(covered);
  }

  // Cleanup helper arrays
  free(vert_owners);
  free(tri_owners);

  // --- 3. Process Transient Buffers (Sort -> Unique -> Pack Direct) ---
  for (int r = 0; r < nproc; ++r)
  {
    // A. Triangles
    if (trans_tri_bufs[r].size > 0)
    {
      MFTLSortIds(trans_tri_bufs[r].data, trans_tri_bufs[r].size);
      int n_unique = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      // Alloc Exact
      send_tri_bufs[r].buffer = malloc(n_unique * sizeof(PackedTriangleComplex));
      send_tri_bufs[r].capacity = n_unique;
      send_tri_bufs[r].count = n_unique;

      // Stage 12: exact-size payload byte buffer for this rank
      if (ds > 0)
      {
        send_data_bufs[r].buffer = malloc(n_unique * ds);
        send_data_bufs[r].capacity = (int) (n_unique * ds);
        send_data_bufs[r].count = (int) (n_unique * ds);  // byte count for the exchange
      }

      // Pack Direct
      PackedTriangleComplex* buf = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      for (int i = 0; i < n_unique; ++i)
      {
        int tri_idx = trans_tri_bufs[r].data[i];
        MFTLPackSingleTriangle(object, tri_idx, &buf[i], MFTL_FALSE);

        // Stage 12: append the payload in the exact same element order
        if (ds > 0)
        {
          if (object->triangles[tri_idx].data == MFTL_NULL)
          {
            fprintf(stderr,
                    "FATAL ERROR: MFTLRedistributeObject: triangle (gid %d) has NULL data but data_size=%zu — "
                    "call MFTLInitializeObjectData on all ranks before redistributing.\n",
                    object->triangles[tri_idx].global_id, ds);
            exit(EXIT_FAILURE);
          }
          memcpy((char*) send_data_bufs[r].buffer + (size_t) i * ds, object->triangles[tri_idx].data, ds);
        }
      }
    }

    // B. Vertices
    if (trans_vert_bufs[r].size > 0)
    {
      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      int n_unique = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      // Alloc Exact
      send_vert_bufs[r].buffer = malloc(n_unique * sizeof(PackedVertex));
      send_vert_bufs[r].capacity = n_unique;
      send_vert_bufs[r].count = n_unique;

      // Pack Direct (truthful sender-side ghost status: ownership authority,
      // see the reconstruction below)
      PackedVertex* buf = (PackedVertex*) send_vert_bufs[r].buffer;
      for (int i = 0; i < n_unique; ++i)
      {
        int vert_idx = trans_vert_bufs[r].data[i];
        MFTLPackSingleVertex(object, vert_idx, &buf[i], object->vertices[vert_idx].is_ghost);
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
  char* kept_data = MFTL_NULL;  // Stage 12
  int kept_count_tris = 0, kept_count_verts = 0, kept_data_bytes = 0;

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
  if (send_data_bufs[my_rank].count > 0)  // Stage 12 ('count' is a byte count here)
  {
    kept_data = (char*) send_data_bufs[my_rank].buffer;
    kept_data_bytes = send_data_bufs[my_rank].count;
    send_data_bufs[my_rank].buffer = MFTL_NULL;
    send_data_bufs[my_rank].count = 0;
  }

  // --- 5. Communication (Phase 2 & 3) ---
  int global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  PackedTriangleComplex* recv_tris;
  PackedVertex* recv_verts;
  char* recv_data = MFTL_NULL;  // Stage 12
  int n_recv_tris, n_recv_verts, n_recv_data = 0;

  // Use the Generic MPI Exchange Helper
  MFTLMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_tris, &n_recv_tris);

  MFTLMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_verts, &n_recv_verts);

  // Stage 12: payload bytes, same path decision and element order
  MFTLMpiExchangeBuffers(domain, send_data_bufs, MPI_BYTE, 1, global_decision, (void**) &recv_data, &n_recv_data);

  // Cleanup Send Buffers
  for (int i = 0; i < nproc; ++i)
  {
    if (send_tri_bufs[i].buffer) free(send_tri_bufs[i].buffer);
    if (send_vert_bufs[i].buffer) free(send_vert_bufs[i].buffer);
    if (send_data_bufs[i].buffer) free(send_data_bufs[i].buffer);
  }
  free(send_tri_bufs);
  free(send_vert_bufs);
  free(send_data_bufs);
  free(is_neighbor_map);

  // --- 6. Reconstruction ---
  MFTLResetObjectData(object);  // Wipe old data

  // Combine Kept + Recv (Triangles)
  int total_tris = kept_count_tris + n_recv_tris;
  PackedTriangleComplex* all_tris = MFTL_NULL;
  if (total_tris > 0)
  {
    all_tris = malloc(total_tris * sizeof(PackedTriangleComplex));
    if (kept_count_tris) memcpy(all_tris, kept_tris, kept_count_tris * sizeof(PackedTriangleComplex));
    if (n_recv_tris) memcpy(all_tris + kept_count_tris, recv_tris, n_recv_tris * sizeof(PackedTriangleComplex));
  }
  free(kept_tris);
  free(recv_tris);

  // Stage 12: payload bytes in the SAME element order as all_tris (kept, then recv)
  char* all_data = MFTL_NULL;
  if (total_tris > 0 && ds > 0)
  {
    all_data = malloc((size_t) total_tris * ds);
    if (kept_data_bytes > 0) memcpy(all_data, kept_data, kept_data_bytes);
    if (n_recv_data > 0) memcpy(all_data + kept_data_bytes, recv_data, n_recv_data);
  }
  free(kept_data);
  free(recv_data);

  // Combine Kept + Recv (Vertices)
  int total_verts = kept_count_verts + n_recv_verts;
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
    int* keys = malloc(total_tris * sizeof(int));
    int* indices = malloc(total_tris * sizeof(int));
    for (int i = 0; i < total_tris; ++i)
    {
      keys[i] = all_tris[i].triangle.global_id;
      indices[i] = i;
    }
    MFTLSortPairs(keys, indices, total_tris);

    // Count Uniques
    int n_unique = 1;
    for (int i = 1; i < total_tris; ++i)
      if (keys[i] != keys[i - 1]) n_unique++;

    object->n_triangles = n_unique;
    object->capacity_triangles = n_unique;
    object->triangles = malloc(n_unique * sizeof(MFTL_Triangle));
    object->n_half_edges = 3 * n_unique;
    object->capacity_half_edges = 3 * n_unique;
    object->half_edges = malloc(3 * n_unique * sizeof(MFTL_HalfEdge));

    int write_idx = 0;
    for (int k = 0; k < total_tris; ++k)
    {
      int sorted_idx = indices[k];
      PackedTriangleComplex* src = &all_tris[sorted_idx];

      khiter_t k_dup = kh_get(triangle_map, object->triangle_lookup, src->triangle.global_id);
      if (k_dup != kh_end(object->triangle_lookup)) continue;  // already written
      if (src->triangle.owner_proc != my_rank) continue;  // Skip ghosts in pass 1

      // Copy Triangle
      MFTL_Triangle* dst = &object->triangles[write_idx];
      dst->global_id = src->triangle.global_id;
      dst->owner_proc = src->triangle.owner_proc;
      dst->flags = src->triangle.flags;
      dst->is_ghost = MFTL_FALSE;
      dst->area = src->triangle.area;
      dst->surface_scalar = src->triangle.surface_scalar;
      memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
      memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
      dst->he_idx = write_idx * 3;

      // Stage 12: uniform stride — element sorted_idx's blob sits at sorted_idx * ds
      if (ds > 0)
      {
        dst->data = malloc(ds);
        memcpy(dst->data, all_data + (size_t) sorted_idx * ds, ds);
      }
      else
      {
        dst->data = MFTL_NULL;
      }

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
        he->twin_gid = src->half_edges[j].twin_gid;
        he->twin_idx = (src->half_edges[j].twin_gid >= 0) ? src->half_edges[j].twin_gid : MFTL_ERROR;  // GID (re-derived by MFTLRebuildTwins)
        he->vertex_idx = src->half_edges[j].vertex_gid;  // Fix later
        he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins (Stage 16A)

        iter = kh_put(halfedge_map, object->halfedge_lookup, he->global_id, &ret);
        kh_value(object->halfedge_lookup, iter) = write_idx * 3 + j;
      }
      write_idx++;
    }

    object->n_local_triangles = write_idx;
    object->n_local_half_edges = write_idx * 3;

    // ── PASS 2: Ghosts ──────────────────────────────────────────────────
    for (int k = 0; k < total_tris; ++k)
    {
      int sorted_idx = indices[k];
      PackedTriangleComplex* src = &all_tris[sorted_idx];

      khiter_t k_dup = kh_get(triangle_map, object->triangle_lookup, src->triangle.global_id);
      if (k_dup != kh_end(object->triangle_lookup)) continue;  // already written
      if (src->triangle.owner_proc == my_rank) continue;  // Skip locals in pass 2

      MFTL_Triangle* dst = &object->triangles[write_idx];
      dst->global_id = src->triangle.global_id;
      dst->owner_proc = src->triangle.owner_proc;
      dst->flags = src->triangle.flags;
      dst->is_ghost = MFTL_TRUE;
      dst->area = src->triangle.area;
      dst->surface_scalar = src->triangle.surface_scalar;
      memcpy(dst->center, src->triangle.center, sizeof(double) * 3);
      memcpy(dst->normal, src->triangle.normal, sizeof(double) * 3);
      dst->he_idx = write_idx * 3;

      // Stage 12: duplicate copies carry identical bytes
      if (ds > 0)
      {
        dst->data = malloc(ds);
        memcpy(dst->data, all_data + (size_t) sorted_idx * ds, ds);
      }
      else
      {
        dst->data = MFTL_NULL;
      }

      int ret;
      khiter_t iter = kh_put(triangle_map, object->triangle_lookup, dst->global_id, &ret);
      kh_value(object->triangle_lookup, iter) = write_idx;

      for (int j = 0; j < 3; ++j)
      {
        MFTL_HalfEdge* he = &object->half_edges[write_idx * 3 + j];
        he->global_id = src->half_edges[j].global_id;
        he->flags = src->half_edges[j].flags;
        he->triangle_idx = write_idx;
        he->next_idx = write_idx * 3 + (j + 1) % 3;
        he->twin_gid = src->half_edges[j].twin_gid;
        he->twin_idx = (src->half_edges[j].twin_gid >= 0) ? src->half_edges[j].twin_gid : MFTL_ERROR;  // GID (re-derived by MFTLRebuildTwins)
        he->vertex_idx = src->half_edges[j].vertex_gid;
        he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins (Stage 16A)

        iter = kh_put(halfedge_map, object->halfedge_lookup, he->global_id, &ret);
        kh_value(object->halfedge_lookup, iter) = write_idx * 3 + j;
      }
      write_idx++;
    }
    MFTL_ASSERT(write_idx == n_unique && "Error: write_count != n_unique\n");
    free(keys);
    free(indices);
    free(all_tris);
    free(all_data);  // Stage 12
  }

  // Rebuild Vertices (Direct GID Sort)
  if (total_verts > 0)
  {
    int* keys = malloc(total_verts * sizeof(int));
    int* indices = malloc(total_verts * sizeof(int));
    for (int i = 0; i < total_verts; ++i)
    {
      keys[i] = all_verts[i].global_id;
      indices[i] = i;
    }
    MFTLSortPairs(keys, indices, total_verts);

    int n_unique = 1;
    for (int i = 1; i < total_verts; ++i)
      if (keys[i] != keys[i - 1]) n_unique++;

    object->n_vertices = n_unique;
    object->capacity_vertices = n_unique;
    object->vertices = malloc(n_unique * sizeof(MFTL_Vertex));

    // Ownership authority: a vertex becomes LOCAL here only when a copy
    // arrives from a sender holding it as NON-ghost (authoritative copy —
    // fresh position, trustworthy owner label). Ghost-derived copies may be
    // stale (ghosts whose owners are outside this rank's communication
    // neighbourhood are never refreshed — systematic for periodic
    // seam-straddlers in wrapped storage, whose stretched triangles reach
    // vertices owned 2+ slabs away; a seam-wrapped vertex jumps a full
    // period). Classifying such copies by their stale label wrote the same
    // gid both as local and as ghost, overflowing the n_unique-sized array
    // and clobbering vertex_lookup. Dedup is therefore hash-based and only
    // authoritative copies can create locals.
    int write_idx = 0;
    for (int k = 0; k < total_verts; ++k)
    {
      int sorted_idx = indices[k];
      PackedVertex* src = &all_verts[sorted_idx];

      if (src->is_ghost != MFTL_FALSE) continue;  // ghost-derived copy: not ownership-authoritative
      if (src->owner_proc != my_rank) continue;   // owned elsewhere

      khiter_t k_dup = kh_get(vertex_map, object->vertex_lookup, src->global_id);
      if (k_dup != kh_end(object->vertex_lookup)) continue;  // already written

      MFTL_Vertex* dst = &object->vertices[write_idx];
      dst->global_id = src->global_id;
      dst->owner_proc = src->owner_proc;
      dst->flags = src->flags;
      dst->is_ghost = MFTL_FALSE;
      dst->area = src->area;
      dst->meancurvature = src->meancurvature;
      dst->gaussiancurvature = src->gaussiancurvature;
      dst->roughness_mean = src->roughness_mean;
      dst->roughness_variance = src->roughness_variance;
      memcpy(dst->pos, src->pos, sizeof(double) * 3);
      memcpy(dst->img, src->img, 3 * sizeof(int));  // Stage 16B
      memcpy(dst->normal, src->normal, sizeof(double) * 3);
      dst->out_he_idx = src->id_he_out;  // Fix later

      int ret;
      khiter_t iter = kh_put(vertex_map, object->vertex_lookup, dst->global_id, &ret);
      kh_value(object->vertex_lookup, iter) = write_idx;

      write_idx++;
    }

    object->n_local_vertices = write_idx;

    // ── PASS 2: Ghosts ──────────────────────────────────────────────────
    // Data precedence: authoritative (sender-local) copies first, then
    // ghost-derived (possibly stale) copies for gids not yet written.
    for (int authoritative = 1; authoritative >= 0; --authoritative)
    {
      for (int k = 0; k < total_verts; ++k)
      {
        int sorted_idx = indices[k];
        PackedVertex* src = &all_verts[sorted_idx];

        if ((src->is_ghost == MFTL_FALSE) != (authoritative == 1)) continue;  // class filter

        khiter_t k_dup = kh_get(vertex_map, object->vertex_lookup, src->global_id);
        if (k_dup != kh_end(object->vertex_lookup)) continue;  // already written

        MFTL_Vertex* dst = &object->vertices[write_idx];
        dst->global_id = src->global_id;
        dst->owner_proc = src->owner_proc;
        dst->flags = src->flags;
        dst->is_ghost = MFTL_TRUE;
        dst->area = src->area;
        dst->meancurvature = src->meancurvature;
        dst->gaussiancurvature = src->gaussiancurvature;
        dst->roughness_mean = src->roughness_mean;
        dst->roughness_variance = src->roughness_variance;
        memcpy(dst->pos, src->pos, sizeof(double) * 3);
        memcpy(dst->img, src->img, 3 * sizeof(int));  // Stage 16B
        memcpy(dst->normal, src->normal, sizeof(double) * 3);
        dst->out_he_idx = src->id_he_out;

        int ret;
        khiter_t iter = kh_put(vertex_map, object->vertex_lookup, dst->global_id, &ret);
        kh_value(object->vertex_lookup, iter) = write_idx;

        write_idx++;
      }
    }

    MFTL_ASSERT(write_idx == n_unique && "Error: write_count != n_unique\n");
    free(keys);
    free(indices);
    free(all_verts);
  }

  // 7. Final Topology
  MFTLConvertGlobalToLocalIndices(object);
  MFTLRebuildTwins(object);
  MFTLFlattenConnectivity(object);

  // Stage 16A: guaranteed ghost completion G1/G2, then the quiescent check.
  int rc = MFTLCompleteGhosts(object, 0.0, MFTL_NULL);
  if (MFTLInternalQuiescentCheck(object, MFTL_VALIDATE_GHOSTS) != MFTL_SUCCESS) rc = MFTL_ERROR;
  return rc;
}

/**
 * @brief Synchronizes ghost vertex positions for a list of objects.
 * @param ens Ensemble pointer.
 * @param object_lookup Optional hash map (BodyID -> Index). Pass MFTL_NULL to use linear search fallback.
 * @return MFTL_SUCCESS on success.
 */
int MFTLUpdateGhostPositions(MFTL_Ensemble* ens, khash_t(object_map) * object_lookup)
{
  if (ens->num_objects == 0 || ens->objects == MFTL_NULL) return MFTL_SUCCESS;

  MFTL_Object** objects = ens->objects;
  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  int my_rank = domain->rank;

  int num_neighbors = domain->num_neighbors;
  int num_objects = ens->num_objects;

  if (num_neighbors <= 0 || nproc < 2) return MFTL_SUCCESS;

  // --- 1. Setup Buffers ---
  IntBuffer* req_send_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) buffer_init(&req_send_bufs[i], 1024);  // TODO: Optimize Buffer Size

  // Fast Map: Global Rank -> Neighbor Index
  int* rank_to_neigh_idx = malloc(nproc * sizeof(int));
  memset(rank_to_neigh_idx, -1, nproc * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh_idx[domain->neighbors[i]] = i;

  int n_skipped_nonneighbor = 0;

  // --- 2. Build Requests (Iterate ALL objects) ---
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    MFTL_Vertex* verts = obj->vertices;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &verts[i];

      if (v->is_ghost)
      {
        int owner = v->owner_proc;
        assert(owner >= 0 && owner < nproc && "Ghost owner rank out of bounds!");
        int n_idx = rank_to_neigh_idx[owner];

        // Parallel remeshing can briefly put a ghost's owner two slabs away
        // (a transient edge longer than a slab width during a split-heavy
        // pass, or a knife-edge slab from the load-balance window). The
        // graph communicator cannot serve those requests; keep the ghost
        // stale instead — it is refreshed by the next redistribution, which
        // runs at least once per remesh pass, and the remesh gates treat
        // ghost-touching edges conservatively. (Writing into
        // req_send_bufs[-1] would corrupt the heap — this must never be an
        // unchecked Release-mode assert.)
        if (n_idx != -1)
        {
          // Append (BodyID, GID) to the request buffer
          buffer_add(&req_send_bufs[n_idx], obj->global_id);
          buffer_add(&req_send_bufs[n_idx], v->global_id);
        }
        else
        {
          n_skipped_nonneighbor++;
        }
      }
    }
  }

  {
    // Unconditional consensus: a rank-local conditional Allreduce here
    // would deadlock the ranks that skip it against the neighbour
    // collectives below.
    int skipped_g = 0;
    MPI_Allreduce(&n_skipped_nonneighbor, &skipped_g, 1, MPI_INT, MPI_SUM, domain->comm_cart);
    if (skipped_g > 0 && my_rank == 0)
      fprintf(stderr,
              "MFTLUpdateGhostPositions: %d ghost(s) have non-neighbour owners (transient slab-spanning edges or\n"
              "knife-edge slabs during parallel remeshing); keeping them stale until the next redistribution.\n",
              skipped_g);
  }

  // --- 3. Exchange Request Counts ---
  int* n_send_reqs = calloc(num_neighbors, sizeof(int));
  int* n_recv_reqs = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_reqs[i] = req_send_bufs[i].size;  // Actual count is size (2 * num_requests)
  }

  MPI_Neighbor_alltoall(n_send_reqs, 1, MPI_INT, n_recv_reqs, 1, MPI_INT, domain->comm_graph);

  // --- 4. Exchange Requests ---
  int* s_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_displs = calloc(num_neighbors + 1, sizeof(int));
  int total_send_ints = 0;
  int total_recv_ints = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_displs[i + 1] = s_displs[i] + n_send_reqs[i];
    r_displs[i + 1] = r_displs[i] + n_recv_reqs[i];
    total_send_ints += n_send_reqs[i];
    total_recv_ints += n_recv_reqs[i];
  }

  int* flat_send_reqs = malloc(total_send_ints * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (n_send_reqs[i] > 0)
    {
      memcpy(flat_send_reqs + s_displs[i], req_send_bufs[i].data, n_send_reqs[i] * sizeof(int));
    }
  }

  int* flat_recv_reqs = malloc(total_recv_ints * sizeof(int));

  MPI_Neighbor_alltoallv(flat_send_reqs, n_send_reqs, s_displs, MPI_INT, flat_recv_reqs, n_recv_reqs, r_displs, MPI_INT, domain->comm_graph);

  free(flat_send_reqs);

  // --- 5. Process Requests & Build Replies ---
  int total_reply_payloads = total_recv_ints / 2;
  GhostUpdatePayload* reply_payloads = malloc(total_reply_payloads * sizeof(GhostUpdatePayload));

  int reply_idx = 0;

  // Cache the last accessed object to speed up sequential access
  int last_body_id = -1;
  MFTL_Object* current_obj = MFTL_NULL;

  for (int k = 0; k < total_recv_ints; k += 2)
  {
    int req_body_id = flat_recv_reqs[k];
    int req_gid = flat_recv_reqs[k + 1];

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
        for (int o = 0; o < num_objects; ++o)
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
      int local_idx = kh_value(current_obj->vertex_lookup, k_vert);
      found_v = &current_obj->vertices[local_idx];
    }

    // --- C. Pack Payload ---
    GhostUpdatePayload* p = &reply_payloads[reply_idx++];
    p->global_id = req_gid;
    p->found = 1;

    if (found_v == MFTL_NULL || (found_v->flags & MFTL_FLAG_DEAD))
    {
      // Parallel remeshing legitimately kills entities whose ghosts other
      // ranks still hold (MFTL1's two-pass scheme relies on this: the ghost
      // copies become stale and are dropped by the next redistribution).
      // Report the miss explicitly; the requester keeps its stale copy.
      // (A zeroed/fabricated payload would silently corrupt ghost state.)
      p->found = 0;
    }
    else
    {
      memcpy(p->pos, found_v->pos, 3 * sizeof(double));
      memcpy(p->img, found_v->img, 3 * sizeof(int));  // Stage 16B
      memcpy(p->normal, found_v->normal, 3 * sizeof(double));
      p->meancurvature = found_v->meancurvature;
      p->gaussiancurvature = found_v->gaussiancurvature;
      p->area = found_v->area;
      p->roughness_mean = found_v->roughness_mean;
      p->roughness_variance = found_v->roughness_variance;
    }
  }

  free(flat_recv_reqs);

  // --- 6. Exchange Replies ---
  int* n_send_replies = calloc(num_neighbors, sizeof(int));
  int* n_recv_replies = malloc(num_neighbors * sizeof(int));
  int* s_rep_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_rep_displs = calloc(num_neighbors + 1, sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_replies[i] = n_recv_reqs[i] / 2;
    n_recv_replies[i] = n_send_reqs[i] / 2;
    s_rep_displs[i + 1] = s_rep_displs[i] + n_send_replies[i];
    r_rep_displs[i + 1] = r_rep_displs[i] + n_recv_replies[i];
  }

  int total_recv_replies = total_send_ints / 2;
  GhostUpdatePayload* recv_payloads = malloc(total_recv_replies * sizeof(GhostUpdatePayload));

  MPI_Neighbor_alltoallv(reply_payloads, n_send_replies, s_rep_displs, MPI_GhostUpdatePayload, recv_payloads, n_recv_replies, r_rep_displs,
                         MPI_GhostUpdatePayload, domain->comm_graph);

  free(reply_payloads);

  // --- 7. Apply Updates ---
  int current_payload_idx = 0;

  // Reset cache for application phase
  last_body_id = -1;
  current_obj = MFTL_NULL;

  for (int n = 0; n < num_neighbors; ++n)
  {
    IntBuffer* buf = &req_send_bufs[n];
    int count = buf->size;

    for (int k = 0; k < count; k += 2)
    {
      int req_body_id = buf->data[k];
      int req_gid = buf->data[k + 1];
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
          for (int o = 0; o < num_objects; ++o)
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
        int local_idx = kh_value(current_obj->vertex_lookup, k_vert);
        MFTL_Vertex* v = &current_obj->vertices[local_idx];

        if (!p->found)
        {
          // The owner killed this entity during remeshing; keep the stale
          // ghost untouched (the next redistribution drops it).
          continue;
        }

        // Apply Update
        memcpy(v->pos, p->pos, 3 * sizeof(double));
        memcpy(v->img, p->img, 3 * sizeof(int));  // Stage 16B
        memcpy(v->normal, p->normal, 3 * sizeof(double));
        v->meancurvature = p->meancurvature;
        v->gaussiancurvature = p->gaussiancurvature;
        v->area = p->area;
        v->roughness_mean = p->roughness_mean;
        v->roughness_variance = p->roughness_variance;
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

/**
 * @brief Refreshes ghost TRIANGLE user payloads (MFTL_Triangle.data) from their owners.
 *
 * Mirrors MFTLUpdateGhostPositions structurally, but for ghost triangles and
 * byte payloads (Stage 12): every ghost triangle requests (body_id, gid) from
 * its owner; the owner replies with a found flag plus the obj->data_size blob
 * bytes, shipped as two parallel streams (MPI_INT flags + MPI_BYTE payload).
 * The requester computes the per-neighbor reply byte counts locally — it knows
 * the data_size of every body it requested — so no extra count exchange is
 * needed for the byte stream.
 *
 * Ghosts whose owner no longer holds the triangle (e.g. killed by remeshing
 * since the last sync) keep their stale payload; the next redistribution drops
 * them. Data is never fabricated.
 *
 * Must be called collectively by all ranks. No-op when no object carries a
 * payload (data_size is set collectively via MFTLInitializeObjectData, so this
 * early-out is rank-consistent) or when there are no neighbors / nproc < 2.
 */
int MFTLScatterTriangleData(MFTL_Ensemble* ens, khash_t(object_map) * object_lookup)
{
  if (ens->num_objects == 0 || ens->objects == MFTL_NULL) return MFTL_SUCCESS;

  MFTL_Object** objects = ens->objects;
  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  int my_rank = domain->rank;

  int num_neighbors = domain->num_neighbors;
  int num_objects = ens->num_objects;

  if (num_neighbors <= 0 || nproc < 2) return MFTL_SUCCESS;

  // Stage 12: this check MUST run before the any_data early-out below — that
  // early-out is only rank-consistent when data_size agrees everywhere, which
  // is exactly what the check verifies. Refuse loudly on a mismatch.
  for (int o = 0; o < num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  // Fast path: no payload anywhere. data_size is collective and the object
  // list is identical on all ranks, so this early-out is rank-consistent
  // (a rank-local conditional skip of the collectives below would deadlock —
  // cf. the note in MFTLUpdateGhostPositions).
  int any_data = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    if (objects[o]->data_size > 0)
    {
      any_data = 1;
      break;
    }
  }
  if (!any_data) return MFTL_SUCCESS;

  // --- 1. Setup Buffers ---
  IntBuffer* req_send_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) buffer_init(&req_send_bufs[i], 1024);

  // Fast Map: Global Rank -> Neighbor Index
  int* rank_to_neigh_idx = malloc(nproc * sizeof(int));
  memset(rank_to_neigh_idx, -1, nproc * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh_idx[domain->neighbors[i]] = i;

  int n_skipped_nonneighbor = 0;

  // --- 2. Build Requests (Iterate ALL objects carrying a payload) ---
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    if (obj->data_size == 0) continue;

    for (int i = 0; i < obj->n_triangles; ++i)
    {
      MFTL_Triangle* t = &obj->triangles[i];

      if (t->is_ghost)
      {
        int owner = t->owner_proc;
        assert(owner >= 0 && owner < nproc && "Ghost owner rank out of bounds!");
        int n_idx = rank_to_neigh_idx[owner];

        // Same non-neighbor-owner skip as MFTLUpdateGhostPositions: the graph
        // communicator cannot serve those requests; keep the ghost payload
        // stale until the next redistribution.
        if (n_idx != -1)
        {
          buffer_add(&req_send_bufs[n_idx], obj->global_id);
          buffer_add(&req_send_bufs[n_idx], t->global_id);
        }
        else
        {
          n_skipped_nonneighbor++;
        }
      }
    }
  }

  {
    // Unconditional consensus: a rank-local conditional Allreduce here
    // would deadlock the ranks that skip it against the neighbour
    // collectives below.
    int skipped_g = 0;
    MPI_Allreduce(&n_skipped_nonneighbor, &skipped_g, 1, MPI_INT, MPI_SUM, domain->comm_cart);
    if (skipped_g > 0 && my_rank == 0)
      fprintf(stderr,
              "MFTLScatterTriangleData: %d ghost(s) have non-neighbour owners (transient slab-spanning edges or\n"
              "knife-edge slabs during parallel remeshing); keeping their payloads stale until the next redistribution.\n",
              skipped_g);
  }

  // --- 3. Exchange Request Counts ---
  int* n_send_reqs = calloc(num_neighbors, sizeof(int));
  int* n_recv_reqs = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_reqs[i] = req_send_bufs[i].size;  // Actual count is size (2 * num_requests)
  }

  MPI_Neighbor_alltoall(n_send_reqs, 1, MPI_INT, n_recv_reqs, 1, MPI_INT, domain->comm_graph);

  // --- 4. Exchange Requests ---
  int* s_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_displs = calloc(num_neighbors + 1, sizeof(int));
  int total_send_ints = 0;
  int total_recv_ints = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_displs[i + 1] = s_displs[i] + n_send_reqs[i];
    r_displs[i + 1] = r_displs[i] + n_recv_reqs[i];
    total_send_ints += n_send_reqs[i];
    total_recv_ints += n_recv_reqs[i];
  }

  int* flat_send_reqs = malloc(total_send_ints * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (n_send_reqs[i] > 0)
    {
      memcpy(flat_send_reqs + s_displs[i], req_send_bufs[i].data, n_send_reqs[i] * sizeof(int));
    }
  }

  int* flat_recv_reqs = malloc(total_recv_ints * sizeof(int));

  MPI_Neighbor_alltoallv(flat_send_reqs, n_send_reqs, s_displs, MPI_INT, flat_recv_reqs, n_recv_reqs, r_displs, MPI_INT, domain->comm_graph);

  free(flat_send_reqs);

  // --- 5. Process Requests & Build Replies ---
  // Two parallel reply streams per neighbor, in request order:
  //  (a) int found flags — counts derivable receiver-side (n_recv_reqs/2)
  //  (b) the payload bytes — per-neighbor byte counts tracked while building
  int total_reply_count = total_recv_ints / 2;
  int* reply_found = malloc((total_reply_count > 0 ? total_reply_count : 1) * sizeof(int));
  int* n_send_reply_bytes = calloc(num_neighbors, sizeof(int));

  size_t reply_bytes_cap = 0, reply_bytes_len = 0;
  char* reply_bytes = MFTL_NULL;

  int reply_idx = 0;

  // Cache the last accessed object to speed up sequential access
  int last_body_id = -1;
  MFTL_Object* current_obj = MFTL_NULL;

  for (int n = 0; n < num_neighbors; ++n)
  {
    int n_pairs = n_recv_reqs[n] / 2;
    for (int j = 0; j < n_pairs; ++j)
    {
      int req_body_id = flat_recv_reqs[r_displs[n] + 2 * j];
      int req_gid = flat_recv_reqs[r_displs[n] + 2 * j + 1];

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
          for (int o = 0; o < num_objects; ++o)
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
      const size_t ds_r = current_obj->data_size;

      // --- B. TRIANGLE LOOKUP (Using existing Hash Map) ---
      MFTL_Triangle* found_t = MFTL_NULL;
      khiter_t k_tri = kh_get(triangle_map, current_obj->triangle_lookup, req_gid);
      if (k_tri != kh_end(current_obj->triangle_lookup))
      {
        found_t = &current_obj->triangles[kh_value(current_obj->triangle_lookup, k_tri)];
      }

      // --- C. Pack Reply ---
      int found = 1;
      if (found_t == MFTL_NULL || (found_t->flags & MFTL_FLAG_DEAD))
      {
        // Parallel remeshing legitimately kills entities whose ghosts other
        // ranks still hold. Report the miss explicitly; the requester keeps
        // its stale copy. (A zeroed/fabricated payload would silently corrupt
        // ghost state.)
        found = 0;
      }
      reply_found[reply_idx++] = found;

      if (ds_r > 0)
      {
        if (reply_bytes_len + ds_r > reply_bytes_cap)
        {
          reply_bytes_cap = (reply_bytes_cap == 0) ? 1024 * ds_r : reply_bytes_cap;
          while (reply_bytes_cap < reply_bytes_len + ds_r) reply_bytes_cap *= 2;
          reply_bytes = realloc(reply_bytes, reply_bytes_cap);
        }
        if (found)
        {
          if (found_t->data == MFTL_NULL)
          {
            fprintf(stderr,
                    "FATAL ERROR: MFTLScatterTriangleData: triangle (gid %d, body %d) has NULL data but data_size=%zu — "
                    "call MFTLInitializeObjectData on all ranks before ghosting.\n",
                    req_gid, req_body_id, ds_r);
            exit(EXIT_FAILURE);
          }
          memcpy(reply_bytes + reply_bytes_len, found_t->data, ds_r);
        }
        else
        {
          memset(reply_bytes + reply_bytes_len, 0, (size_t) ds_r);  // placeholder; ignored by the requester (found==0)
        }
        reply_bytes_len += ds_r;
        n_send_reply_bytes[n] += (int) ds_r;
      }
    }
  }

  free(flat_recv_reqs);

  // --- 6. Exchange Replies ---
  int* n_send_replies = calloc(num_neighbors, sizeof(int));
  int* n_recv_replies = malloc(num_neighbors * sizeof(int));
  int* s_rep_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_rep_displs = calloc(num_neighbors + 1, sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_replies[i] = n_recv_reqs[i] / 2;
    n_recv_replies[i] = n_send_reqs[i] / 2;
    s_rep_displs[i + 1] = s_rep_displs[i] + n_send_replies[i];
    r_rep_displs[i + 1] = r_rep_displs[i] + n_recv_replies[i];
  }

  int total_recv_replies = total_send_ints / 2;
  int* recv_found = malloc((total_recv_replies > 0 ? total_recv_replies : 1) * sizeof(int));

  MPI_Neighbor_alltoallv(reply_found, n_send_replies, s_rep_displs, MPI_INT, recv_found, n_recv_replies, r_rep_displs, MPI_INT, domain->comm_graph);

  free(reply_found);

  // Payload byte stream: per-neighbor RECEIVE counts computed from our own
  // requests (we know the data_size of every body we asked about).
  int* n_recv_reply_bytes = calloc(num_neighbors, sizeof(int));
  int* s_rep_byte_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_rep_byte_displs = calloc(num_neighbors + 1, sizeof(int));
  int total_recv_bytes = 0;

  for (int n = 0; n < num_neighbors; ++n)
  {
    IntBuffer* buf = &req_send_bufs[n];
    for (int k = 0; k < buf->size; k += 2)
    {
      n_recv_reply_bytes[n] += (int) mftl_data_size_of_body(objects, num_objects, buf->data[k]);
    }
    s_rep_byte_displs[n + 1] = s_rep_byte_displs[n] + n_send_reply_bytes[n];
    r_rep_byte_displs[n + 1] = r_rep_byte_displs[n] + n_recv_reply_bytes[n];
    total_recv_bytes += n_recv_reply_bytes[n];
  }

  char* recv_bytes = malloc(total_recv_bytes > 0 ? (size_t) total_recv_bytes : 1);

  MPI_Neighbor_alltoallv(reply_bytes, n_send_reply_bytes, s_rep_byte_displs, MPI_BYTE, recv_bytes, n_recv_reply_bytes, r_rep_byte_displs, MPI_BYTE,
                         domain->comm_graph);

  free(reply_bytes);

  // --- 7. Apply Updates ---
  int current_reply_idx = 0;
  size_t byte_cursor = 0;

  // Reset cache for application phase
  last_body_id = -1;
  current_obj = MFTL_NULL;

  for (int n = 0; n < num_neighbors; ++n)
  {
    IntBuffer* buf = &req_send_bufs[n];
    int count = buf->size;

    for (int k = 0; k < count; k += 2)
    {
      int req_body_id = buf->data[k];
      int req_gid = buf->data[k + 1];
      int found = recv_found[current_reply_idx++];

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
          for (int o = 0; o < num_objects; ++o)
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
      const size_t ds_r = current_obj->data_size;

      // --- B. TRIANGLE LOOKUP (O(1)) ---
      khiter_t k_tri = kh_get(triangle_map, current_obj->triangle_lookup, req_gid);
      if (k_tri != kh_end(current_obj->triangle_lookup))
      {
        MFTL_Triangle* t = &current_obj->triangles[kh_value(current_obj->triangle_lookup, k_tri)];

        if (found)
        {
          if (t->data == MFTL_NULL)
          {
            fprintf(stderr,
                    "FATAL ERROR: MFTLScatterTriangleData: ghost triangle (gid %d, body %d) has NULL data but data_size=%zu.\n",
                    req_gid, req_body_id, ds_r);
            exit(EXIT_FAILURE);
          }
          memcpy(t->data, recv_bytes + byte_cursor, ds_r);
        }
        // else: the owner killed this entity during remeshing; keep the stale
        // ghost payload untouched (the next redistribution drops it).
      }
      byte_cursor += ds_r;  // ALWAYS advance, regardless of hit/miss
    }
  }

  for (int i = 0; i < num_neighbors; ++i)
  {
    if (req_send_bufs[i].data) free(req_send_bufs[i].data);
  }

  // --- Cleanup ---
  free(req_send_bufs);
  free(recv_found);
  free(recv_bytes);
  free(rank_to_neigh_idx);

  free(n_send_reqs);
  free(n_recv_reqs);
  free(n_send_replies);
  free(n_recv_replies);
  free(n_send_reply_bytes);
  free(n_recv_reply_bytes);
  free(s_displs);
  free(r_displs);
  free(s_rep_displs);
  free(r_rep_displs);
  free(s_rep_byte_displs);
  free(r_rep_byte_displs);

  return MFTL_SUCCESS;
}

/* -------------------------------------------------------------------------- */
/* GHOST LAYER EXTENSION LOGIC                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Helper to append a new vertex to the object during ghost integration.
 */
void MFTLAppendGhostVertex(MFTL_Object* obj, PackedVertex* pv)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);

  int idx = obj->n_vertices++;
  MFTL_Vertex* v = &obj->vertices[idx];

  v->global_id = pv->global_id;
  v->owner_proc = pv->owner_proc;
  v->is_ghost = MFTL_TRUE;
  v->flags = pv->flags;
  v->area = pv->area;
  v->meancurvature = pv->meancurvature;
  v->gaussiancurvature = pv->gaussiancurvature;
  v->roughness_mean = pv->roughness_mean;
  v->roughness_variance = pv->roughness_variance;
  v->out_he_idx = pv->id_he_out;  // GID, resolved later

  memcpy(v->pos, pv->pos, 3 * sizeof(double));
  memcpy(v->img, pv->img, 3 * sizeof(int));  // Stage 16B
  memcpy(v->normal, pv->normal, 3 * sizeof(double));

  int ret;
  khiter_t k = kh_put(vertex_map, obj->vertex_lookup, v->global_id, &ret);
  kh_value(obj->vertex_lookup, k) = idx;
}

/**
 * @brief Helper to append a new triangle (and its halfedges) during ghost integration.
 * @param data Pointer to the obj->data_size payload bytes shipped alongside ptc
 *             (may be NULL only when obj->data_size == 0).
 */
void MFTLAppendGhostTriangle(MFTL_Object* obj, PackedTriangleComplex* ptc, const void* data)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 3, MFTL_HalfEdge);

  int t_idx = obj->n_triangles++;
  MFTL_Triangle* t = &obj->triangles[t_idx];

  t->global_id = ptc->triangle.global_id;
  t->owner_proc = ptc->triangle.owner_proc;
  t->is_ghost = MFTL_TRUE;
  t->flags = ptc->triangle.flags;
  t->area = ptc->triangle.area;
  t->surface_scalar = ptc->triangle.surface_scalar;
  t->he_idx = obj->n_half_edges;

  memcpy(t->normal, ptc->triangle.normal, 3 * sizeof(double));
  memcpy(t->center, ptc->triangle.center, 3 * sizeof(double));

  // Stage 12: attach the opaque user payload shipped with the triangle.
  if (obj->data_size > 0)
  {
    t->data = malloc(obj->data_size);
    memcpy(t->data, data, obj->data_size);
  }
  else
  {
    t->data = MFTL_NULL;
  }

  int ret;
  khiter_t k = kh_put(triangle_map, obj->triangle_lookup, t->global_id, &ret);
  kh_value(obj->triangle_lookup, k) = t_idx;

  for (int i = 0; i < 3; ++i)
  {
    int he_idx = obj->n_half_edges++;
    MFTL_HalfEdge* he = &obj->half_edges[he_idx];
    PackedHalfEdge* phe = &ptc->half_edges[i];

    he->global_id = phe->global_id;
    he->triangle_idx = t_idx;
    he->vertex_idx = phe->vertex_gid;  // GID, resolved later
    he->twin_gid = phe->twin_gid;
    he->twin_idx = (phe->twin_gid >= 0) ? phe->twin_gid : MFTL_ERROR;  // GID, resolved later
    he->next_idx = t->he_idx + ((i + 1) % 3);
    he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins (Stage 16A)
    he->flags = phe->flags;

    k = kh_put(halfedge_map, obj->halfedge_lookup, he->global_id, &ret);
    kh_value(obj->halfedge_lookup, k) = he_idx;
  }
}

/**
 * @brief Remaps the GID-based references of ghosts appended during ONE
 * extension round (local indices >= v_mark / he_mark) to local indices.
 *
 * MFTLConvertGlobalToLocalIndices cannot be used here: it assumes the ENTIRE
 * mesh still stores global IDs, but the pre-existing (local and previously
 * ghosted) entities already store rank-local indices. Re-converting the whole
 * mesh would interpret those local indices as GIDs, corrupt connectivity or
 * trigger the "missing vertex GID" fatal check. Only the freshly appended
 * entities carry GIDs, so only they are remapped; twins are recomputed
 * topologically by MFTLRebuildTwins afterwards anyway.
 */
void MFTLInternalRemapAppendedGhostIndices(MFTL_Object* obj, int v_mark, int he_mark)
{
  // --- 1. Remap appended vertices' outgoing half-edge: GID -> local index ---
  for (int i = v_mark; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    khiter_t k = kh_get(halfedge_map, obj->halfedge_lookup, v->out_he_idx);  // still a GID here
    v->out_he_idx = (k != kh_end(obj->halfedge_lookup)) ? kh_value(obj->halfedge_lookup, k) : MFTL_ERROR;
  }

  // --- 2. Remap appended half-edges' target vertex: GID -> local index ---
  for (int i = he_mark; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];

    // This assertion MUST hold: every corner of an appended ghost triangle
    // is shipped together with that triangle.
    khiter_t k = kh_get(vertex_map, obj->vertex_lookup, he->vertex_idx);  // still a GID here
    if (k == kh_end(obj->vertex_lookup))
    {
      fprintf(stderr, "FATAL ERROR: Half-edge %d references missing vertex GID %d\n", he->global_id, he->vertex_idx);
      he->vertex_idx = MFTL_ERROR;
      exit(EXIT_FAILURE);
    }
    he->vertex_idx = kh_value(obj->vertex_lookup, k);

    // Twin was packed as a GID (or MFTL_ERROR). Resolve it if the twin half-edge
    // lives on this rank; otherwise leave MFTL_ERROR for MFTLRebuildTwins.
    if (he->twin_idx >= 0)
    {
      khiter_t kt = kh_get(halfedge_map, obj->halfedge_lookup, he->twin_idx);
      he->twin_idx = (kt != kh_end(obj->halfedge_lookup)) ? kh_value(obj->halfedge_lookup, kt) : MFTL_ERROR;
    }
  }

  // --- 3. Repair out-he of appended vertices whose packed out half-edge is
  // not present locally: 'he' points TO v, so 'he->next' must start FROM v.
  for (int i = he_mark; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    int v_idx = he->vertex_idx;
    if (v_idx >= 0 && v_idx < obj->n_vertices && obj->vertices[v_idx].out_he_idx == MFTL_ERROR)
    {
      obj->vertices[v_idx].out_he_idx = he->next_idx;
    }
  }
}

/**
 * @brief Extends the ghost layer by 'steps' topological layers.
 * Uses a Request-Reply pattern with direct Neighbor communication.
 */
int MFTLExtendGhostLayer(MFTL_Ensemble* ens, int steps)
{
  if (steps <= 0) return MFTL_SUCCESS;

  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  // Stage 12: refuse loudly on a cross-rank data_size mismatch before any
  // packing/communication (payload byte counts derive from local data_size).
  for (int o = 0; o < ens->num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(ens->objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

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

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];

    // Iterate Vertices to find Ghosts (Seeds)
    for (int i = 0; i < obj->n_vertices; ++i)
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
  int* n_send_ints = calloc(num_neighbors, sizeof(int));
  int* n_recv_ints = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i) n_send_ints[i] = req_bufs[i].size;

  MPI_Neighbor_alltoall(n_send_ints, 1, MPI_INT, n_recv_ints, 1, MPI_INT, domain->comm_graph);

  // Calc Displacements
  int* s_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_displs = calloc(num_neighbors + 1, sizeof(int));
  int total_send = 0, total_recv = 0;

  for (int i = 0; i < num_neighbors; ++i)
  {
    s_displs[i + 1] = s_displs[i] + n_send_ints[i];
    r_displs[i + 1] = r_displs[i] + n_recv_ints[i];
    total_send += n_send_ints[i];
    total_recv += n_recv_ints[i];
  }

  // Flatten Requests
  int* flat_send_req = malloc(total_send * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (req_bufs[i].size > 0)
    {
      memcpy(flat_send_req + s_displs[i], req_bufs[i].data, req_bufs[i].size * sizeof(int));
    }
    // Cleanup early
    free(req_bufs[i].data);
  }
  free(req_bufs);

  // Exchange Requests
  int* flat_recv_req = malloc(total_recv * sizeof(int));
  MPI_Neighbor_alltoallv(flat_send_req, n_send_ints, s_displs, MPI_INT, flat_recv_req, n_recv_ints, r_displs, MPI_INT, domain->comm_graph);

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

  // Stage 12: parallel payload byte streams (byte counts NOT derivable from
  // triangle counts when objects have different data_size — exchanged explicitly)
  char** raw_send_data = calloc(num_neighbors, sizeof(char*));
  int* cnt_send_data = calloc(num_neighbors, sizeof(int));
  int* cap_send_data = calloc(num_neighbors, sizeof(int));

  int* cnt_send_tris = calloc(num_neighbors, sizeof(int));
  int* cap_send_tris = calloc(num_neighbors, sizeof(int));

  int* cnt_send_verts = calloc(num_neighbors, sizeof(int));
  int* cap_send_verts = calloc(num_neighbors, sizeof(int));

  // Collector reuse
  IntBuffer* seed_collector = calloc(ens->num_objects, sizeof(IntBuffer));
  for (int i = 0; i < ens->num_objects; ++i) buffer_init(&seed_collector[i], 256);

  for (int n = 0; n < num_neighbors; ++n)
  {
    int start_idx = r_displs[n];
    int end_idx = r_displs[n + 1];

    // 1. Reset Collector
    for (int i = 0; i < ens->num_objects; ++i) buffer_reset(&seed_collector[i]);

    // 2. Binning
    for (int k = start_idx; k < end_idx; k += 2)
    {
      int body_id = flat_recv_req[k];
      int gid = flat_recv_req[k + 1];

      // Find Object (Optimize: use hash map if many objects)
      for (int o = 0; o < ens->num_objects; ++o)
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
    for (int o = 0; o < ens->num_objects; ++o)
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
          int new_cap = (cap_send_tris[n] == 0) ? (nb.n_tris + 128) : (cap_send_tris[n] * 2 + nb.n_tris);
          raw_send_tris[n] = realloc(raw_send_tris[n], new_cap * sizeof(PackedTriangleComplex));
          cap_send_tris[n] = new_cap;
        }

        // Stage 12: grow the parallel byte stream (same geometric pattern, in bytes)
        if (obj->data_size > 0 && (size_t) cnt_send_data[n] + nb.n_tris * obj->data_size > (size_t) cap_send_data[n])
        {
          int new_cap_d = (cap_send_data[n] == 0) ? (int) (nb.n_tris * obj->data_size + 128 * obj->data_size)
                                                  : (int) MFTL_MAX((size_t) cap_send_data[n] * 2, (size_t) cnt_send_data[n] + nb.n_tris * obj->data_size);
          raw_send_data[n] = realloc(raw_send_data[n], new_cap_d);
          cap_send_data[n] = new_cap_d;
        }

        for (int t = 0; t < nb.n_tris; ++t)
        {
          MFTLPackSingleTriangle(obj, nb.tri_indices[t], &raw_send_tris[n][cnt_send_tris[n]++], false);

          // Stage 12: append the payload in the exact same element order
          if (obj->data_size > 0)
          {
            if (obj->triangles[nb.tri_indices[t]].data == MFTL_NULL)
            {
              fprintf(stderr,
                      "FATAL ERROR: MFTLExtendGhostLayer: triangle (gid %d, body %d) has NULL data but data_size=%zu — "
                      "call MFTLInitializeObjectData on all ranks before ghosting.\n",
                      obj->triangles[nb.tri_indices[t]].global_id, obj->global_id, obj->data_size);
              exit(EXIT_FAILURE);
            }
            memcpy(raw_send_data[n] + cnt_send_data[n], obj->triangles[nb.tri_indices[t]].data, obj->data_size);
            cnt_send_data[n] += (int) obj->data_size;
          }
        }
      }

      // --- Pack Vertices ---
      if (nb.n_verts > 0)
      {
        if (cnt_send_verts[n] + nb.n_verts > cap_send_verts[n])
        {
          int new_cap = (cap_send_verts[n] == 0) ? (nb.n_verts + 128) : (cap_send_verts[n] * 2 + nb.n_verts);
          raw_send_verts[n] = realloc(raw_send_verts[n], new_cap * sizeof(PackedVertex));
          cap_send_verts[n] = new_cap;
        }

        for (int v = 0; v < nb.n_verts; ++v)
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
  int* n_recv_tris = malloc(num_neighbors * sizeof(int));
  int* n_recv_verts = malloc(num_neighbors * sizeof(int));
  int* n_recv_data = malloc(num_neighbors * sizeof(int));  // Stage 12: payload byte counts

  MPI_Neighbor_alltoall(cnt_send_tris, 1, MPI_INT, n_recv_tris, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_verts, 1, MPI_INT, n_recv_verts, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_data, 1, MPI_INT, n_recv_data, 1, MPI_INT, domain->comm_graph);  // Stage 12

  // --- Exchange Triangles ---
  int* s_disp_t = calloc(num_neighbors + 1, sizeof(int));
  int* r_disp_t = calloc(num_neighbors + 1, sizeof(int));
  int tot_send_t = 0, tot_recv_t = 0;

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

  // --- Exchange Payload Bytes (Stage 12) ---
  int* s_disp_d = calloc(num_neighbors + 1, sizeof(int));
  int* r_disp_d = calloc(num_neighbors + 1, sizeof(int));
  int tot_send_d = 0, tot_recv_d = 0;
  for (int i = 0; i < num_neighbors; ++i)
  {
    s_disp_d[i + 1] = s_disp_d[i] + cnt_send_data[i];
    r_disp_d[i + 1] = r_disp_d[i] + n_recv_data[i];
    tot_send_d += cnt_send_data[i];
    tot_recv_d += n_recv_data[i];
  }

  char* flat_send_data = malloc(tot_send_d);
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_data[i] > 0) memcpy(flat_send_data + s_disp_d[i], raw_send_data[i], cnt_send_data[i]);
    free(raw_send_data[i]);  // Free individual buffers immediately
  }
  free(raw_send_data);

  char* recv_data = malloc(tot_recv_d);
  MPI_Neighbor_alltoallv(flat_send_data, cnt_send_data, s_disp_d, MPI_BYTE, recv_data, n_recv_data, r_disp_d, MPI_BYTE, domain->comm_graph);

  free(flat_send_data);
  free(s_disp_d);
  free(r_disp_d);
  free(n_recv_data);

  // --- Exchange Vertices ---
  int* s_disp_v = calloc(num_neighbors + 1, sizeof(int));
  int* r_disp_v = calloc(num_neighbors + 1, sizeof(int));
  int tot_send_v = 0, tot_recv_v = 0;

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
  free(cnt_send_data);  // Stage 12
  free(cap_send_data);  // Stage 12
  free(n_recv_tris);
  free(n_recv_verts);

  // =========================================================================
  // PHASE 4: INTEGRATION
  // =========================================================================

  // Record per-object array sizes BEFORE appending so Phase 5 can restrict
  // the GID->local remap to the entities appended in THIS round. The
  // pre-existing entities already store rank-local indices.
  int* v_marks = malloc(ens->num_objects * sizeof(int));
  int* he_marks = malloc(ens->num_objects * sizeof(int));
  for (int o = 0; o < ens->num_objects; ++o)
  {
    v_marks[o] = ens->objects[o]->n_vertices;
    he_marks[o] = ens->objects[o]->n_half_edges;
  }

  // Phase 4 remarks: a received ghost whose owner is NOT a graph neighbour
  // of this rank is unusable — the ghost machinery (MFTLUpdateGhostPositions)
  // can only exchange with neighbours. This happens when the expansion
  // neighbourhood reaches ~2 edge-lengths on a decomposition whose slabs
  // are narrower than that (e.g. after a large boundary shift). Skip such
  // vertices, and skip every triangle that references a vertex we could
  // not integrate (its GID->local remap would fail in Phase 5).
  char* is_neigh_rank = calloc(nproc, sizeof(char));
  for (int i = 0; i < num_neighbors; ++i) is_neigh_rank[domain->neighbors[i]] = 1;
  int my_rank = domain->rank;

  // Integrate Vertices
  for (int i = 0; i < tot_recv_v; ++i)
  {
    PackedVertex* pv = &recv_verts[i];

    // Simple Linear Search for Object (Assumption: low N_objects)
    MFTL_Object* obj = MFTL_NULL;
    for (int o = 0; o < ens->num_objects; ++o)
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
        if (pv->owner_proc != my_rank && !is_neigh_rank[pv->owner_proc]) continue;
        MFTLAppendGhostVertex(obj, pv);
      }
    }
  }

  // Integrate Triangles
  size_t data_cursor = 0;  // Stage 12: running byte offset into recv_data
  for (int i = 0; i < tot_recv_t; ++i)
  {
    PackedTriangleComplex* ptc = &recv_tris[i];

    MFTL_Object* obj = MFTL_NULL;
    for (int o = 0; o < ens->num_objects; ++o)
    {
      if (ens->objects[o]->global_id == ptc->triangle.body_id)
      {
        obj = ens->objects[o];
        break;
      }
    }

    // Stage 12: this element's payload occupies exactly ds_i bytes at data_cursor
    const size_t ds_i = obj ? obj->data_size : 0;

    if (obj)
    {
      khiter_t it = kh_get(triangle_map, obj->triangle_lookup, ptc->triangle.global_id);
      if (it == kh_end(obj->triangle_lookup))
      {
        int corners_present = 1;
        for (int k = 0; k < 3; ++k)
        {
          if (kh_get(vertex_map, obj->vertex_lookup, ptc->half_edges[k].vertex_gid) == kh_end(obj->vertex_lookup))
          {
            corners_present = 0;
            break;
          }
        }
        if (corners_present)
        {
          MFTLAppendGhostTriangle(obj, ptc, ds_i > 0 ? (const void*) (recv_data + data_cursor) : MFTL_NULL);
        }
      }
    }
    data_cursor += ds_i;  // ALWAYS advance, regardless of skip/append
  }

  free(is_neigh_rank);

  free(recv_tris);
  free(recv_verts);
  free(recv_data);  // Stage 12
  free(r_disp_t);
  free(r_disp_v);

  // =========================================================================
  // PHASE 5: REPAIR TOPOLOGY
  // Only the ghosts appended in THIS round carry global IDs; remap exactly
  // those and leave the pre-existing (already local) entities untouched.
  // =========================================================================
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];
    MFTLInternalRemapAppendedGhostIndices(obj, v_marks[o], he_marks[o]);
    MFTLRebuildTwins(obj);
    MFTLFlattenConnectivity(obj);
  }

  free(v_marks);
  free(he_marks);

  return MFTL_SUCCESS;
}

/**
 * @brief Bulk spatial ghosting using pure Domain-Boundary overlap and deduplication.
 * Eliminates upfront AABB communication overhead entirely.
 * * @param ens The MFTL ensemble.
 * @param proximity_tolerance Expansion envelope for near-field interaction tracking.
 * @return MFTL_SUCCESS on completion.
 */
static void mftlRankBox(const MFTL_Domain* dom, int rank, double bmin[3], double bmax[3]);

int MFTLEnsembleCreateSpatialGhosts(MFTL_Ensemble* ens, double proximity_tolerance)
{
  if (!ens || ens->num_objects == 0) return MFTL_SUCCESS;

  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  int num_neighbors = domain->num_neighbors;

  if (nproc < 2 || num_neighbors == 0) return MFTL_SUCCESS;

  // Stage 12: refuse loudly on a cross-rank data_size mismatch before any
  // packing/communication (payload byte counts derive from local data_size).
  for (int o = 0; o < ens->num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(ens->objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  // =========================================================================
  // PHASE 1: DETERMINISTIC DOMAIN BOUNDARY CALCULATION
  // Compute exactly where our neighbors live in 3D space.
  // =========================================================================
  double dx[3];
  for (int d = 0; d < 3; ++d)
  {
    dx[d] = domain->domain_size[d] / (double) domain->cuts[d];
  }

  double* neighbor_bounds = malloc(num_neighbors * 6 * sizeof(double));

  // Stage 15 (L6): the neighbour boxes come from the domain's (possibly rectilinear) slabs.
  (void) dx;
  for (int n = 0; n < num_neighbors; ++n) mftlRankBox(domain, domain->neighbors[n], &neighbor_bounds[n * 6 + 0], &neighbor_bounds[n * 6 + 3]);

  IntBuffer* trans_tri_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int n = 0; n < num_neighbors; ++n)
  {
    buffer_init(&trans_tri_bufs[n], 1024);
    buffer_init(&trans_vert_bufs[n], 3072);
  }

  // Final Packed Send Buffers (Dynamically resized)
  PackedTriangleComplex** raw_send_tris = calloc(num_neighbors, sizeof(PackedTriangleComplex*));
  PackedVertex** raw_send_verts = calloc(num_neighbors, sizeof(PackedVertex*));
  char** raw_send_data = calloc(num_neighbors, sizeof(char*));  // Stage 12: parallel payload byte streams
  int* cnt_send_data = calloc(num_neighbors, sizeof(int));
  int* cap_send_data = calloc(num_neighbors, sizeof(int));
  int* cnt_send_tris = calloc(num_neighbors, sizeof(int));
  int* cap_send_tris = calloc(num_neighbors, sizeof(int));
  int* cnt_send_verts = calloc(num_neighbors, sizeof(int));
  int* cap_send_verts = calloc(num_neighbors, sizeof(int));

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];

    if (!obj->flat_tri_verts || !obj->flat_tri_neighbors)
    {
      MFTLFlattenConnectivity(obj);
    }

    // Reset transient binning buffers for this object
    for (int n = 0; n < num_neighbors; ++n)
    {
      buffer_reset(&trans_tri_bufs[n]);
      buffer_reset(&trans_vert_bufs[n]);
    }

    // 1. Bin Triangles overlapping neighbor domains
    for (int i = 0; i < obj->n_triangles; ++i)
    {
      if (obj->triangles[i].is_ghost) continue;

      double tmin[3] = {1e30, 1e30, 1e30}, tmax[3] = {-1e30, -1e30, -1e30};
      for (int j = 0; j < 3; j++)
      {
        int v_idx = obj->flat_tri_verts[3 * i + j];
        for (int d = 0; d < 3; d++)
        {
          double p = obj->vertices[v_idx].pos[d];
          if (p < tmin[d]) tmin[d] = p;
          if (p > tmax[d]) tmax[d] = p;
        }
      }

      // Inflate by proximity_tolerance for near-field interactions
      for (int d = 0; d < 3; d++)
      {
        tmin[d] -= proximity_tolerance;
        tmax[d] += proximity_tolerance;
      }

      for (int n = 0; n < num_neighbors; ++n)
      {
        double* nb_min = &neighbor_bounds[n * 6 + 0];
        double* nb_max = &neighbor_bounds[n * 6 + 3];

        int overlap = 1;
        for (int k = 0; k < 3; ++k)
        {
          if (tmax[k] < nb_min[k] || tmin[k] > nb_max[k]) overlap = 0;
        }

        if (overlap)
        {
          buffer_add(&trans_tri_bufs[n], i);
          buffer_add(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 0]);
          buffer_add(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 1]);
          buffer_add(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 2]);
        }
      }
    }

    // 2. Sort, Deduplicate, and Pack for each neighbor
    for (int n = 0; n < num_neighbors; ++n)
    {
      // Triangles
      if (trans_tri_bufs[n].size > 0)
      {
        MFTLSortIds(trans_tri_bufs[n].data, trans_tri_bufs[n].size);
        int n_unique = MFTLUniqueIntArray(trans_tri_bufs[n].data, trans_tri_bufs[n].size);

        if (cnt_send_tris[n] + n_unique > cap_send_tris[n])
        {
          cap_send_tris[n] = (cap_send_tris[n] == 0) ? (n_unique + 1024) : (cap_send_tris[n] * 2 + n_unique);
          raw_send_tris[n] = realloc(raw_send_tris[n], cap_send_tris[n] * sizeof(PackedTriangleComplex));
        }

        // Stage 12: grow the parallel byte stream (same geometric pattern, in bytes)
        if (obj->data_size > 0 && (size_t) cnt_send_data[n] + n_unique * obj->data_size > (size_t) cap_send_data[n])
        {
          cap_send_data[n] = (cap_send_data[n] == 0) ? (int) (n_unique * obj->data_size + 1024 * obj->data_size)
                                                     : (int) MFTL_MAX((size_t) cap_send_data[n] * 2, (size_t) cnt_send_data[n] + n_unique * obj->data_size);
          raw_send_data[n] = realloc(raw_send_data[n], cap_send_data[n]);
        }

        for (int i = 0; i < n_unique; ++i)
        {
          int tri_idx = trans_tri_bufs[n].data[i];
          MFTLPackSingleTriangle(obj, tri_idx, &raw_send_tris[n][cnt_send_tris[n]++], true);

          // Stage 12: append the payload in the exact same element order
          if (obj->data_size > 0)
          {
            if (obj->triangles[tri_idx].data == MFTL_NULL)
            {
              fprintf(stderr,
                      "FATAL ERROR: MFTLEnsembleCreateSpatialGhosts: triangle (gid %d, body %d) has NULL data but data_size=%zu — "
                      "call MFTLInitializeObjectData on all ranks before ghosting.\n",
                      obj->triangles[tri_idx].global_id, obj->global_id, obj->data_size);
              exit(EXIT_FAILURE);
            }
            memcpy(raw_send_data[n] + cnt_send_data[n], obj->triangles[tri_idx].data, obj->data_size);
            cnt_send_data[n] += (int) obj->data_size;
          }
        }
      }

      // Vertices
      if (trans_vert_bufs[n].size > 0)
      {
        MFTLSortIds(trans_vert_bufs[n].data, trans_vert_bufs[n].size);
        int n_unique = MFTLUniqueIntArray(trans_vert_bufs[n].data, trans_vert_bufs[n].size);

        if (cnt_send_verts[n] + n_unique > cap_send_verts[n])
        {
          cap_send_verts[n] = (cap_send_verts[n] == 0) ? (n_unique + 3072) : (cap_send_verts[n] * 2 + n_unique);
          raw_send_verts[n] = realloc(raw_send_verts[n], cap_send_verts[n] * sizeof(PackedVertex));
        }

        for (int i = 0; i < n_unique; ++i)
        {
          int vert_idx = trans_vert_bufs[n].data[i];
          MFTLPackSingleVertex(obj, vert_idx, &raw_send_verts[n][cnt_send_verts[n]++], true);
        }
      }
    }
  }

  // Cleanup Transients
  free(neighbor_bounds);
  for (int n = 0; n < num_neighbors; ++n)
  {
    free(trans_tri_bufs[n].data);
    free(trans_vert_bufs[n].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);

  // =========================================================================
  // PHASE 3: BULK MPI EXCHANGE
  // =========================================================================
  int* n_recv_tris = malloc(num_neighbors * sizeof(int));
  int* n_recv_verts = malloc(num_neighbors * sizeof(int));
  int* n_recv_data = malloc(num_neighbors * sizeof(int));  // Stage 12: payload byte counts

  // Note: MPI strictly expects 32-bit int for counts
  int* n_send_tris_int = calloc(num_neighbors, sizeof(int));
  int* n_send_verts_int = calloc(num_neighbors, sizeof(int));
  for (int i = 0; i < num_neighbors; i++)
  {
    n_send_tris_int[i] = (int) cnt_send_tris[i];
    n_send_verts_int[i] = (int) cnt_send_verts[i];
  }

  MPI_Neighbor_alltoall(n_send_tris_int, 1, MPI_INT, n_recv_tris, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(n_send_verts_int, 1, MPI_INT, n_recv_verts, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_data, 1, MPI_INT, n_recv_data, 1, MPI_INT, domain->comm_graph);  // Stage 12

  const size_t n_disp = (size_t) (num_neighbors > 0 ? num_neighbors : 0) + 1;  // counts are never negative
  int* s_disp_t = calloc(n_disp, sizeof(int));
  int* r_disp_t = calloc(n_disp, sizeof(int));
  int* s_disp_v = calloc(n_disp, sizeof(int));
  int* r_disp_v = calloc(n_disp, sizeof(int));
  int* s_disp_d = calloc(n_disp, sizeof(int));  // Stage 12
  int* r_disp_d = calloc(n_disp, sizeof(int));  // Stage 12

  int tot_send_t = 0, tot_recv_t = 0, tot_send_v = 0, tot_recv_v = 0, tot_send_d = 0, tot_recv_d = 0;
  for (int i = 0; i < num_neighbors; ++i)
  {
    s_disp_t[i + 1] = s_disp_t[i] + n_send_tris_int[i];
    r_disp_t[i + 1] = r_disp_t[i] + n_recv_tris[i];
    s_disp_v[i + 1] = s_disp_v[i] + n_send_verts_int[i];
    r_disp_v[i + 1] = r_disp_v[i] + n_recv_verts[i];
    s_disp_d[i + 1] = s_disp_d[i] + cnt_send_data[i];
    r_disp_d[i + 1] = r_disp_d[i] + n_recv_data[i];

    tot_send_t += n_send_tris_int[i];
    tot_recv_t += n_recv_tris[i];
    tot_send_v += n_send_verts_int[i];
    tot_recv_v += n_recv_verts[i];
    tot_send_d += cnt_send_data[i];
    tot_recv_d += n_recv_data[i];
  }

  PackedTriangleComplex* flat_send_tris = malloc(tot_send_t * sizeof(PackedTriangleComplex));
  PackedVertex* flat_send_verts = malloc(tot_send_v * sizeof(PackedVertex));
  char* flat_send_data = malloc(tot_send_d);  // Stage 12

  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_tris[i] > 0) memcpy(flat_send_tris + s_disp_t[i], raw_send_tris[i], cnt_send_tris[i] * sizeof(PackedTriangleComplex));
    if (cnt_send_verts[i] > 0) memcpy(flat_send_verts + s_disp_v[i], raw_send_verts[i], cnt_send_verts[i] * sizeof(PackedVertex));
    if (cnt_send_data[i] > 0) memcpy(flat_send_data + s_disp_d[i], raw_send_data[i], cnt_send_data[i]);  // Stage 12
    free(raw_send_tris[i]);
    free(raw_send_verts[i]);
    free(raw_send_data[i]);  // Stage 12
  }
  free(raw_send_tris);
  free(raw_send_verts);
  free(raw_send_data);  // Stage 12

  PackedTriangleComplex* recv_tris = malloc(tot_recv_t * sizeof(PackedTriangleComplex));
  PackedVertex* recv_verts = malloc(tot_recv_v * sizeof(PackedVertex));
  char* recv_data = malloc(tot_recv_d);  // Stage 12

  MPI_Neighbor_alltoallv(flat_send_tris, n_send_tris_int, s_disp_t, MPI_PackedTriangleComplex, recv_tris, n_recv_tris, r_disp_t, MPI_PackedTriangleComplex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(flat_send_verts, n_send_verts_int, s_disp_v, MPI_PackedVertex, recv_verts, n_recv_verts, r_disp_v, MPI_PackedVertex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(flat_send_data, cnt_send_data, s_disp_d, MPI_BYTE, recv_data, n_recv_data, r_disp_d, MPI_BYTE, domain->comm_graph);  // Stage 12

  // =========================================================================
  // PHASE 4: O(1) UNPACK & HASH MAP INTEGRATION
  // =========================================================================
  // Vertices MUST be integrated before Triangles so Triangle pointers can resolve
  for (int i = 0; i < tot_recv_v; ++i)
  {
    PackedVertex* pv = &recv_verts[i];
    for (int o = 0; o < ens->num_objects; ++o)
    {
      if (ens->objects[o]->global_id == pv->body_id)
      {
        if (kh_get(vertex_map, ens->objects[o]->vertex_lookup, pv->global_id) == kh_end(ens->objects[o]->vertex_lookup))
        {
          MFTLAppendGhostVertex(ens->objects[o], pv);
        }
        break;
      }
    }
  }

  size_t data_cursor = 0;  // Stage 12: running byte offset into recv_data
  for (int i = 0; i < tot_recv_t; ++i)
  {
    PackedTriangleComplex* ptc = &recv_tris[i];
    for (int o = 0; o < ens->num_objects; ++o)
    {
      if (ens->objects[o]->global_id == ptc->triangle.body_id)
      {
        if (kh_get(triangle_map, ens->objects[o]->triangle_lookup, ptc->triangle.global_id) == kh_end(ens->objects[o]->triangle_lookup))
        {
          MFTLAppendGhostTriangle(ens->objects[o], ptc,
                                  ens->objects[o]->data_size > 0 ? (const void*) (recv_data + data_cursor) : MFTL_NULL);
        }
        data_cursor += ens->objects[o]->data_size;  // Stage 12: ALWAYS advance, regardless of skip/append
        break;
      }
    }
  }

  // Memory Cleanup
  free(flat_send_tris);
  free(flat_send_verts);
  free(flat_send_data);  // Stage 12
  free(recv_tris);
  free(recv_verts);
  free(recv_data);  // Stage 12
  free(cnt_send_tris);
  free(cap_send_tris);
  free(cnt_send_verts);
  free(cap_send_verts);
  free(cnt_send_data);  // Stage 12
  free(cap_send_data);  // Stage 12
  free(n_send_tris_int);
  free(n_send_verts_int);
  free(n_recv_tris);
  free(n_recv_verts);
  free(n_recv_data);  // Stage 12
  free(s_disp_t);
  free(r_disp_t);
  free(s_disp_v);
  free(r_disp_v);
  free(s_disp_d);  // Stage 12
  free(r_disp_d);  // Stage 12

  // =========================================================================
  // PHASE 5: TOPOLOGY RESOLUTION
  // =========================================================================
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTLConvertGlobalToLocalIndices(ens->objects[o]);
    MFTLRebuildTwins(ens->objects[o]);
    MFTLFlattenConnectivity(ens->objects[o]);
  }

  return MFTL_SUCCESS;
}

/* ========================================================================== */
/* Stage 15 (L6): distance-based ghosting to ALL ranks                        */
/* ========================================================================== */

// Box of `rank` in the domain (rectilinear or uniform slabs).
void MFTLInternalRankBox(const MFTL_Domain* dom, int rank, double bmin[3], double bmax[3]) { mftlRankBox(dom, rank, bmin, bmax); }

static void mftlRankBox(const MFTL_Domain* dom, int rank, double bmin[3], double bmax[3])
{
  int idx = 0;
  for (idx = 0; idx < dom->nproc; ++idx)
    if (dom->rank_map[idx] == rank) break;
  const int ijk[3] = {idx % dom->cuts[0], (idx / dom->cuts[0]) % dom->cuts[1], idx / (dom->cuts[0] * dom->cuts[1])};
  for (int d = 0; d < 3; ++d)
  {
    if (dom->axis_cuts[d])
    {
      bmin[d] = dom->axis_cuts[d][ijk[d]];
      bmax[d] = dom->axis_cuts[d][ijk[d] + 1];
    }
    else
    {
      const double w = dom->domain_size[d] / (double) dom->cuts[d];
      bmin[d] = dom->global_min[d] + ijk[d] * w;
      bmax[d] = dom->global_min[d] + (ijk[d] + 1) * w;
    }
  }
}

int MFTLObjectSendLocalTrianglesToAll(MFTL_Object* obj, double distance)
{
  MFTL_Domain* domain = obj->domain;
  const int nproc = domain->nproc, me = domain->rank;
  if (nproc < 2) return MFTL_SUCCESS;
  if (MFTLCheckTriangleDataSizeConsistency(obj) != MFTL_SUCCESS) return MFTL_ERROR;
  const size_t ds = obj->data_size;

  double* boxes = (double*) malloc((size_t) nproc * 6 * sizeof(double));
  for (int r = 0; r < nproc; ++r) mftlRankBox(domain, r, &boxes[6 * r], &boxes[6 * r + 3]);

  MFTLFlattenConnectivity(obj);
  IntBuffer* tri_sel = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* vert_sel = calloc(nproc, sizeof(IntBuffer));
  for (int r = 0; r < nproc; ++r)
  {
    buffer_init(&tri_sel[r], 64);
    buffer_init(&vert_sel[r], 192);
  }
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    if (obj->triangles[i].is_ghost || obj->triangles[i].owner_proc != me) continue;
    double tmin[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX}, tmax[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
    for (int j = 0; j < 3; ++j)
    {
      const double* p = obj->vertices[obj->flat_tri_verts[3 * i + j]].pos;
      for (int d = 0; d < 3; ++d)
      {
        tmin[d] = MFTL_MIN(tmin[d], p[d] - distance);
        tmax[d] = MFTL_MAX(tmax[d], p[d] + distance);
      }
    }
    for (int r = 0; r < nproc; ++r)
    {
      if (r == me) continue;
      const double* b = &boxes[6 * r];
      if (tmax[0] < b[0] || tmin[0] > b[3] || tmax[1] < b[1] || tmin[1] > b[4] || tmax[2] < b[2] || tmin[2] > b[5]) continue;
      buffer_add(&tri_sel[r], i);
      for (int j = 0; j < 3; ++j) buffer_add(&vert_sel[r], obj->flat_tri_verts[3 * i + j]);
    }
  }
  free(boxes);

  MFTL_SendBuffer* send_tris = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_verts = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_data = calloc(nproc, sizeof(MFTL_SendBuffer));
  for (int r = 0; r < nproc; ++r)
  {
    int nt = tri_sel[r].size;  // triangles are visited once each: already unique
    if (nt > 0)
    {
      PackedTriangleComplex* buf = malloc((size_t) nt * sizeof(PackedTriangleComplex));
      for (int k = 0; k < nt; ++k) MFTLPackSingleTriangle(obj, tri_sel[r].data[k], &buf[k], MFTL_TRUE);
      send_tris[r].buffer = buf;
      send_tris[r].count = send_tris[r].capacity = nt;
      if (ds > 0)
      {
        char* db = malloc((size_t) nt * ds);
        for (int k = 0; k < nt; ++k) memcpy(db + (size_t) k * ds, obj->triangles[tri_sel[r].data[k]].data, ds);
        send_data[r].buffer = db;
        send_data[r].count = send_data[r].capacity = (int) ((size_t) nt * ds);
      }
    }
    if (vert_sel[r].size > 0)
    {
      MFTLSortIds(vert_sel[r].data, vert_sel[r].size);
      int nv = MFTLUniqueIntArray(vert_sel[r].data, vert_sel[r].size);
      PackedVertex* vb = malloc((size_t) nv * sizeof(PackedVertex));
      for (int k = 0; k < nv; ++k) MFTLPackSingleVertex(obj, vert_sel[r].data[k], &vb[k], MFTL_TRUE);
      send_verts[r].buffer = vb;
      send_verts[r].count = send_verts[r].capacity = nv;
    }
    free(tri_sel[r].data);
    free(vert_sel[r].data);
  }
  free(tri_sel);
  free(vert_sel);

  PackedTriangleComplex* recv_tris = MFTL_NULL;
  PackedVertex* recv_verts = MFTL_NULL;
  char* recv_data = MFTL_NULL;
  int n_rt = 0, n_rv = 0, n_rd = 0;
  MFTLMpiExchangeBuffers(domain, send_tris, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), MFTL_TRUE, (void**) &recv_tris, &n_rt);
  MFTLMpiExchangeBuffers(domain, send_verts, MPI_PackedVertex, sizeof(PackedVertex), MFTL_TRUE, (void**) &recv_verts, &n_rv);
  MFTLMpiExchangeBuffers(domain, send_data, MPI_BYTE, 1, MFTL_TRUE, (void**) &recv_data, &n_rd);
  for (int r = 0; r < nproc; ++r)
  {
    free(send_tris[r].buffer);
    free(send_verts[r].buffer);
    free(send_data[r].buffer);
  }
  free(send_tris);
  free(send_verts);
  free(send_data);

  // append what is not stored yet (as ghosts), then resolve the appended references
  const int v_mark = obj->n_vertices, he_mark = obj->n_half_edges;
  for (int k = 0; k < n_rv; ++k)
  {
    khiter_t kv = kh_get(vertex_map, obj->vertex_lookup, recv_verts[k].global_id);
    if (kv == kh_end(obj->vertex_lookup)) MFTLAppendGhostVertex(obj, &recv_verts[k]);
  }
  for (int k = 0; k < n_rt; ++k)
  {
    khiter_t kt = kh_get(triangle_map, obj->triangle_lookup, recv_tris[k].triangle.global_id);
    if (kt != kh_end(obj->triangle_lookup)) continue;
    MFTLAppendGhostTriangle(obj, &recv_tris[k], ds > 0 ? recv_data + (size_t) k * ds : MFTL_NULL);
  }
  free(recv_tris);
  free(recv_verts);
  free(recv_data);

  MFTLInternalRemapAppendedGhostIndices(obj, v_mark, he_mark);
  MFTLRebuildTwins(obj);
  MFTLFlattenConnectivity(obj);
  return MFTL_SUCCESS;
}

/*
 * MFTL2 parallel — redistribution of mesh entities: record packing, the buffer exchange and the
 * two redistribution paths (MFTLRedistributeObject / MFTLRedistributeListOfObjects). The record
 * datatypes are in comm_types.c; ghost refresh and ghost layers in ghost_refresh.c and
 * ghost_layers.c.
 *
 * Ownership contract: after a redistribution a vertex is owned by MFTLGetOwnerProc(pos) and a
 * triangle by MFTLGetOwnerProc(center). Only the owner's copy is authoritative; every other
 * stored copy is a ghost. Each triangle travels to its owner, to the owners of its vertices and
 * to the owners of its edge-neighbour triangles, so every rank receives one ghost layer. Ghosts
 * are never forwarded (only non-ghost copies are shipped), and the receiver creates owned
 * vertices only from copies whose sender held them as non-ghost (exactly-once ownership).
 * Guaranteed ghost completion (G1/G2, plans/16 §2.5) and the quiescent twin-state check end
 * every redistribution.
 *
 * Wire format: the Packed* records (mftl_communication.h) are committed as MPI struct types
 * resized to sizeof(record). The opaque per-triangle payload travels as a parallel MPI_BYTE
 * stream in the same element order as the packed triangles; every transport entry point first
 * checks that data_size agrees on all ranks.
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
#include "comm_internal.h"

void mftlBufferInit(IntBuffer* buf, int initial_cap)
{
  buf->data = (int*) malloc(initial_cap * sizeof(int));
  buf->size = 0;
  buf->capacity = initial_cap;
}

/**
 * @brief Initial capacities of the per-rank index lists (they grow geometrically when needed):
 * 20% of the average per-rank share, clamped to [1024, 1e6] entries.
 */
static inline void mftlCalcBufferCapacities(int n_tris, int n_verts, int nproc, int* out_tri_cap, int* out_vert_cap)
{
  double factor = 0.2;
  double avg_tris = (double) n_tris / (double) nproc;
  double avg_verts = (double) n_verts / (double) nproc;

  *out_tri_cap = (int) (avg_tris * factor);
  *out_vert_cap = (int) (avg_verts * factor);

  *out_tri_cap = MFTL_MIN(MFTL_MAX(1024, *out_tri_cap), 1000000);
  *out_vert_cap = MFTL_MIN(MFTL_MAX(1024, *out_vert_cap), 1000000);
}

/**
 * @brief Packs a triangle and its three half-edges; references become global ids and each
 * half-edge carries its twin state code (twin gid or boundary code, plans/16 §2.1).
 */
void MFTLPackSingleTriangle(MFTL_Object* obj, int tri_idx, PackedTriangleComplex* dst, bool is_ghost)
{
  MFTL_Triangle* src = &obj->triangles[tri_idx];

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

  MFTL_HalfEdge* t_he[3];
  MFTLGetTriangleHalfedges(obj, src, t_he);

  for (int j = 0; j < 3; ++j)
  {
    dst->half_edges[j].global_id = t_he[j]->global_id;
    dst->half_edges[j].vertex_gid = MFTLHeTarget(obj, t_he[j])->global_id;
    dst->half_edges[j].twin_gid = MFTLHeTwinCode(obj, t_he[j]);  // state code: twin gid / boundary code
    dst->half_edges[j].next_gid = MFTLHeNext(obj, t_he[j])->global_id;
    dst->half_edges[j].triangle_gid = MFTLHeTriangle(obj, t_he[j])->global_id;
    dst->half_edges[j].boundary = MFTLHeIsBoundary(t_he[j]);  // re-derived at destination
    dst->half_edges[j].flags = t_he[j]->flags;
  }
}

/**
 * @brief Packs a vertex (position with image counters, derived geometry, flags, outgoing
 * half-edge as a global id); is_ghost is the flag to transmit.
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
  memcpy(dst->img, src->img, 3 * sizeof(int));
  memcpy(dst->normal, src->normal, 3 * sizeof(double));
}

/**
 * @brief Frees the entity arrays, payload blobs and id maps of an object before it is rebuilt
 * from the received records (object-level state is kept).
 */
static inline void mftlResetObjectData(MFTL_Object* obj)
{
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

  kh_clear(triangle_map, obj->triangle_lookup);
  kh_clear(halfedge_map, obj->halfedge_lookup);
  kh_clear(vertex_map, obj->vertex_lookup);
}

/**
 * @brief Exchanges variable-length per-rank buffers (send_bufs[rank], `count` elements of
 * `type`) and returns the concatenated received elements (malloc'ed; order: by source rank on
 * the global path, by neighbour index on the neighbour path).
 *
 * Global path: MPI_Alltoall of the counts + MPI_Alltoallv on comm_cart (any rank pair).
 * Neighbour path: the same with the neighbour collectives on comm_graph (only graph neighbours
 * may have non-zero counts). Every rank must take the same path. Counts are MPI ints.
 */
void mftlMpiExchangeBuffers(MFTL_Domain* domain, MFTL_SendBuffer* send_bufs, MPI_Datatype type, size_t type_size, bool use_global_path,
                                   void** out_recv_buf, int* out_recv_count)
{
  int nproc = (int) domain->nproc;
  *out_recv_count = 0;

  if (use_global_path)
  {
    int* send_counts = calloc(nproc, sizeof(int));
    int* recv_counts = malloc(nproc * sizeof(int));
    int* send_displs = calloc(nproc + 1, sizeof(int));
    int* recv_displs = calloc(nproc + 1, sizeof(int));

    for (int r = 0; r < nproc; ++r) send_counts[r] = send_bufs[r].count;
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, domain->comm_cart);

    int total_send = 0;
    for (int i = 0; i < nproc; ++i)
    {
      total_send += send_counts[i];
      *out_recv_count += recv_counts[i];
      send_displs[i + 1] = send_displs[i] + send_counts[i];
      recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    }

    // one contiguous send buffer in rank order
    char* flat_send_buf = malloc(total_send * type_size);
    for (int r = 0; r < nproc; ++r)
    {
      if (send_counts[r] > 0)
      {
        memcpy(flat_send_buf + (send_displs[r] * type_size), send_bufs[r].buffer, send_counts[r] * type_size);
      }
    }

    *out_recv_buf = malloc((*out_recv_count) * type_size);
    MPI_Alltoallv(flat_send_buf, send_counts, send_displs, type, *out_recv_buf, recv_counts, recv_displs, type, domain->comm_cart);

    free(flat_send_buf);
    free(send_counts);
    free(recv_counts);
    free(send_displs);
    free(recv_displs);
  }
  else
  {
    int num_neighbors = domain->num_neighbors;

    // arrays sized to the neighbour list
    int* n_send = calloc(num_neighbors, sizeof(int));
    int* n_recv = malloc(num_neighbors * sizeof(int));
    int* s_displs = calloc(num_neighbors + 1, sizeof(int));
    int* r_displs = calloc(num_neighbors + 1, sizeof(int));

    for (int i = 0; i < num_neighbors; ++i)
    {
      n_send[i] = send_bufs[domain->neighbors[i]].count;
    }
    MPI_Neighbor_alltoall(n_send, 1, MPI_INT, n_recv, 1, MPI_INT, domain->comm_graph);

    int total_send = 0;
    for (int i = 0; i < num_neighbors; ++i)
    {
      total_send += n_send[i];
      *out_recv_count += n_recv[i];
      s_displs[i + 1] = s_displs[i] + n_send[i];
      r_displs[i + 1] = r_displs[i] + n_recv[i];
    }

    // one contiguous send buffer in neighbour order
    char* flat_send_buf = malloc(total_send * type_size);
    for (int i = 0; i < num_neighbors; ++i)
    {
      if (n_send[i] > 0)
      {
        int target_rank = domain->neighbors[i];
        memcpy(flat_send_buf + (s_displs[i] * type_size), send_bufs[target_rank].buffer, n_send[i] * type_size);
      }
    }

    *out_recv_buf = malloc((*out_recv_count) * type_size);
    MPI_Neighbor_alltoallv(flat_send_buf, n_send, s_displs, type, *out_recv_buf, n_recv, r_displs, type, domain->comm_graph);

    free(flat_send_buf);
    free(n_send);
    free(n_recv);
    free(s_displs);
    free(r_displs);
  }
}

/**
 * @brief Collective contract check — obj->data_size must be identical
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
 * @brief In-place gather array[i] = old array[indices[i]] (applies a sort permutation).
 */
static inline void mftlPermuteTriangles(PackedTriangleComplex* array, const int* indices, int count)
{
  assert(count > 0);

  PackedTriangleComplex* tmp = (PackedTriangleComplex*) malloc(count * sizeof(PackedTriangleComplex));
  assert(tmp != MFTL_NULL);

  for (int i = 0; i < count; ++i)
  {
    tmp[i] = array[indices[i]];
  }

  memcpy(array, tmp, count * sizeof(PackedTriangleComplex));

  free(tmp);
}

/**
 * @brief In-place gather array[i] = old array[indices[i]] (applies a sort permutation).
 */
static inline void mftlPermuteVertices(PackedVertex* array, const int* indices, int count)
{
  assert(count > 0);

  PackedVertex* tmp = (PackedVertex*) malloc(count * sizeof(PackedVertex));
  assert(tmp != MFTL_NULL);

  for (int i = 0; i < count; ++i)
  {
    tmp[i] = array[indices[i]];
  }

  memcpy(array, tmp, count * sizeof(PackedVertex));

  free(tmp);
}

/**
 * @brief Redistributes every object of the ensemble in one communication round (collective).
 *
 * 1. Owners are recomputed from the current positions (vertex: pos, triangle: centre). Each
 *    owned (non-ghost) triangle is addressed to its owner, the owners of its vertices and the
 *    owners of its edge-neighbour triangles; a non-ghost vertex without a non-ghost incident
 *    triangle is sent to its owner explicitly (authority completeness).
 * 2. One Allreduce decides the path: the neighbour collectives when every destination is a
 *    graph neighbour, otherwise the global Alltoallv. The payload stream uses the same path.
 * 3. The receiver rebuilds each object from its kept + received records, grouped by body id
 *    and de-duplicated by global id: owned entities first, then ghosts. Owned vertices are
 *    created only from sender-authoritative (non-ghost) copies.
 * 4. Local indices are re-derived, twins re-paired, then ghost completion G1/G2 and the
 *    quiescent twin-state check run (plans/16 §2.3, §2.5).
 *
 * The object lists must agree on all ranks. Returns MFTL_ERROR on a data_size mismatch or when
 * ghost completion / validation fails; a no-op on one rank.
 */
int MFTLRedistributeListOfObjects(MFTL_Ensemble* ensemble)
{
  if (ensemble->num_objects == 0 || ensemble->objects == MFTL_NULL) return MFTL_ERROR;

  MFTL_Domain* domain = &ensemble->domain;
  int num_objects = ensemble->num_objects;
  MFTL_Object** objects = ensemble->objects;

  if (domain->nproc < 2) return MFTL_SUCCESS;

  // Payload byte counts are derived from the LOCAL data_size of each object — a cross-rank
  // mismatch would silently corrupt the streams. Refuse loudly before touching anything.
  // (Per-object check: data_size may differ BETWEEN objects, but must be identical per object
  // across ranks.)
  for (int o = 0; o < num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  int nproc = domain->nproc;
  int my_rank = domain->rank;
  int requires_global_comm_local = 0;

  char* is_neighbor_map = (char*) calloc(nproc, sizeof(char));
  for (int i = 0; i < domain->num_neighbors; ++i)
  {
    is_neighbor_map[domain->neighbors[i]] = 1;
  }

  // per destination rank: triangle / vertex index lists of the current object (sorted and
  // de-duplicated per object), and the master list that records all objects in order
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* master_buf = calloc(nproc, sizeof(IntBuffer));

  int* total_tris_all_objects = calloc(nproc, sizeof(int));
  int* total_verts_all_objects = calloc(nproc, sizeof(int));
  int* total_data_bytes = calloc(nproc, sizeof(int));  // per-rank payload byte counts

  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_data_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));  // parallel MPI_BYTE stream

  int total_global_tris = 0;
  int total_global_verts = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    total_global_tris += object->global_n_triangles;
    total_global_verts += object->global_n_vertices;
  }

  int init_tri_cap, init_vert_cap;
  mftlCalcBufferCapacities(total_global_tris, total_global_verts, nproc, &init_tri_cap, &init_vert_cap);

  for (int r = 0; r < nproc; ++r)
  {
    mftlBufferInit(&trans_tri_bufs[r], init_tri_cap);
    mftlBufferInit(&trans_vert_bufs[r], init_vert_cap);
    mftlBufferInit(&master_buf[r], init_tri_cap + init_vert_cap);
  }

  // --- Destinations ---

  // owner scratch arrays, grown as needed across the objects
  int last_size_tris = objects[0]->n_triangles;
  int last_size_verts = objects[0]->n_vertices;
  int* vert_owners = malloc(last_size_verts * sizeof(int));
  int* tri_owners = malloc(last_size_tris * sizeof(int));

  // the scratch arrays are rewritten from index 0 for every object
  int zero_count = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    MFTL_Triangle* tri_base = object->triangles;
    MFTL_Vertex* vert_base = object->vertices;
    int n_triangles = object->n_triangles;
    int n_vertices = object->n_vertices;

    for (int r = 0; r < nproc; ++r)
    {
      mftlBufferReset(&trans_tri_bufs[r]);
      mftlBufferReset(&trans_vert_bufs[r]);
    }

    MFTL_ENSURE_CAPACITY(tri_owners, &zero_count, &last_size_tris, n_triangles, int);
    MFTL_ENSURE_CAPACITY(vert_owners, &zero_count, &last_size_verts, n_vertices, int);

    last_size_tris = n_triangles;
    last_size_verts = n_vertices;

    // Always rebuilt: remeshing may have changed the topology since the arrays were built (a
    // build-only-when-NULL policy shipped stale connectivity after remeshing).
    MFTLFlattenConnectivity(object);

    int* flat_tri_verts = object->flat_tri_verts;
    int* flat_tri_neighbors = object->flat_tri_neighbors;

    // owners from the current positions (every stored copy, ghosts included)
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

    for (int i = 0; i < n_triangles; ++i)
    {
      // Only the (pre-redistribution) non-ghost copy ships a triangle: ghosts are never
      // forwarded, and stale ghosts disappear. is_ghost is tested because owner_proc already
      // holds the NEW owner.
      if (tri_base[i].is_ghost)
      {
        continue;
      }

      // destinations: the triangle's owner, its vertices' owners, its edge-neighbours' owners
      int distinct_ranks[7];
      int count = 0;

      distinct_ranks[count++] = tri_owners[i];

      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 1]];
      distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 2]];

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

      for (int k = 0; k < num_unique_dests; ++k)
      {
        int dest_rank = unique_dests[k];
        if (dest_rank != my_rank && !is_neighbor_map[dest_rank])
        {
          requires_global_comm_local = 1;
        }

        // duplicates are removed per rank after sorting
        mftlBufferAdd(&trans_tri_bufs[dest_rank], i);

        mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 0]);
        mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 1]);
        mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 2]);
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
        mftlBufferAdd(&trans_vert_bufs[dest_rank], i);
      }
      free(covered);
    }

    // Master list per rank: {object id, n_tris, n_verts, triangle indices, vertex indices} for
    // every object in order (an empty entry keeps the per-object stream aligned).
    for (int r = 0; r < nproc; ++r)
    {
      if (trans_tri_bufs[r].size == 0 && trans_vert_bufs[r].size == 0)
      {
        mftlBufferAdd(&master_buf[r], object->global_id);
        mftlBufferAdd(&master_buf[r], 0);
        mftlBufferAdd(&master_buf[r], 0);
        continue;
      }
      MFTLSortIds(trans_tri_bufs[r].data, trans_tri_bufs[r].size);
      int n_unique_tris = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      int n_unique_verts = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      total_tris_all_objects[r] += n_unique_tris;
      total_verts_all_objects[r] += n_unique_verts;
      total_data_bytes[r] += n_unique_tris * (int) object->data_size;
      mftlBufferAdd(&master_buf[r], object->global_id);
      mftlBufferAdd(&master_buf[r], n_unique_tris);
      mftlBufferAdd(&master_buf[r], n_unique_verts);
      for (int i = 0; i < n_unique_tris; ++i)
      {
        mftlBufferAdd(&master_buf[r], trans_tri_bufs[r].data[i]);
      }
      for (int i = 0; i < n_unique_verts; ++i)
      {
        mftlBufferAdd(&master_buf[r], trans_vert_bufs[r].data[i]);
      }
    }
  }

  free(vert_owners);
  free(tri_owners);

  // exact-size send buffers; 'count' is the packing cursor
  for (int r = 0; r < nproc; ++r)
  {
    send_tri_bufs[r].buffer = malloc(total_tris_all_objects[r] * sizeof(PackedTriangleComplex));
    send_tri_bufs[r].capacity = total_tris_all_objects[r];
    send_tri_bufs[r].count = 0;

    send_vert_bufs[r].buffer = malloc(total_verts_all_objects[r] * sizeof(PackedVertex));
    send_vert_bufs[r].capacity = total_verts_all_objects[r];
    send_vert_bufs[r].count = 0;

    // opaque payload bytes; 'count' is the byte cursor
    send_data_bufs[r].buffer = malloc(total_data_bytes[r]);
    send_data_bufs[r].capacity = total_data_bytes[r];
    send_data_bufs[r].count = 0;
  }

  // --- Packing (object by object, following the master list) ---
  for (int r = 0; r < nproc; ++r)
  {
    int master_idx = 0;

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

      for (int i = master_idx; i < master_idx + n_unique_tris; ++i)
      {
        int tri_idx = master_buf[r].data[i];

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
          he->twin_gid = MFTLHeTwinCode(object, t_he[j]);  // state code: twin gid / boundary code
          he->next_gid = MFTLHeNext(object, t_he[j])->global_id;
          he->triangle_gid = MFTLHeTriangle(object, t_he[j])->global_id;
          he->boundary = MFTLHeIsBoundary(t_he[j]);
          he->flags = t_he[j]->flags;
        }

        // payload bytes in the same element order as the packed triangles
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

      for (int i = master_idx + n_unique_tris; i < master_idx + n_unique_tris + n_unique_verts; ++i)
      {
        int vert_idx = master_buf[r].data[i];

        PackedVertex* packed_vert = &vert_buf_typed[send_vert_bufs[r].count++];

        packed_vert->global_id = vert_base[vert_idx].global_id;
        packed_vert->owner_proc = vert_base[vert_idx].owner_proc;
        memcpy(packed_vert->pos, vert_base[vert_idx].pos, 3 * sizeof(double));
        memcpy(packed_vert->img, vert_base[vert_idx].img, 3 * sizeof(int));
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

  for (int r = 0; r < nproc; ++r)
  {
    free(trans_tri_bufs[r].data);
    free(trans_vert_bufs[r].data);
    free(master_buf[r].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);
  free(master_buf);

  // --- Records that stay on this rank: take over this rank's own send buffers ---
  PackedTriangleComplex* kept_tris = MFTL_NULL;
  PackedVertex* kept_verts = MFTL_NULL;
  char* kept_data = MFTL_NULL;
  int kept_count_tris = 0, kept_count_verts = 0, kept_data_bytes = 0;

  int own_rank = domain->rank;
  for (int r = 0; r < nproc; ++r)
  {
    if (r == own_rank)
    {
      kept_tris = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      kept_verts = (PackedVertex*) send_vert_bufs[r].buffer;
      kept_count_tris = send_tri_bufs[r].count;
      kept_count_verts = send_vert_bufs[r].count;

      kept_data = (char*) send_data_bufs[r].buffer;  // 'count' is a byte count here
      kept_data_bytes = send_data_bufs[r].count;

      send_tri_bufs[r].buffer = MFTL_NULL;
      send_tri_bufs[r].count = 0;
      send_tri_bufs[r].capacity = 0;

      send_vert_bufs[r].buffer = MFTL_NULL;
      send_vert_bufs[r].count = 0;
      send_vert_bufs[r].capacity = 0;

      send_data_bufs[r].buffer = MFTL_NULL;
      send_data_bufs[r].count = 0;
      send_data_bufs[r].capacity = 0;
    }
  }

  // --- Exchange: one collective decision, the same path for all three streams ---
  int global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  PackedTriangleComplex* recv_buf_tris = MFTL_NULL;
  PackedVertex* recv_buf_verts = MFTL_NULL;
  char* recv_data = MFTL_NULL;
  int total_recv_tris = 0, total_recv_verts = 0, recv_data_bytes = 0;

  mftlMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_buf_tris,
                         &total_recv_tris);

  mftlMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_buf_verts, &total_recv_verts);

  mftlMpiExchangeBuffers(domain, send_data_bufs, MPI_BYTE, 1, global_decision, (void**) &recv_data, &recv_data_bytes);

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

  // --- Reconstruction ---

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* obj = objects[o];
    mftlResetObjectData(obj);
  }

  // Triangles: kept + received records, grouped by body id, then per body sorted by gid
  int final_tris_count = kept_count_tris + total_recv_tris;
  PackedTriangleComplex* all_tris = MFTL_NULL;

  if (final_tris_count > 0)
  {
    all_tris = (PackedTriangleComplex*) malloc(final_tris_count * sizeof(PackedTriangleComplex));
    if (kept_count_tris > 0) memcpy(all_tris, kept_tris, kept_count_tris * sizeof(PackedTriangleComplex));
    if (total_recv_tris > 0) memcpy(all_tris + kept_count_tris, recv_buf_tris, total_recv_tris * sizeof(PackedTriangleComplex));
  }

  free(kept_tris);
  free(recv_buf_tris);

  // payload bytes in the SAME element order as all_tris (kept first, then received)
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

  if (final_tris_count > 0)
  {
    // group by body id (radix sort of (body id, index) pairs)
    int* keys = malloc(final_tris_count * sizeof(int));
    int* indices = malloc(final_tris_count * sizeof(int));

    for (int i = 0; i < final_tris_count; ++i)
    {
      keys[i] = all_tris[i].triangle.body_id;
      indices[i] = i;
    }

    MFTLSortPairs(keys, indices, final_tris_count);

    // per-element payload byte offsets into all_data in the OLD element order (the blob size
    // differs between bodies), needed to permute the blobs with all_tris
    size_t* elem_off = MFTL_NULL;
    if (all_data_bytes > 0)
    {
      elem_off = malloc(final_tris_count * sizeof(size_t));
      size_t run = 0;
      for (int i = 0; i < final_tris_count; ++i)
      {
        elem_off[i] = run;
        run += mftlDataSizeOfBody(objects, num_objects, all_tris[i].triangle.body_id);
      }
      assert(run == (size_t) all_data_bytes);
    }

    mftlPermuteTriangles(all_tris, indices, final_tris_count);

    // gather the payload blobs into the permuted order and record per-element offsets (NEW
    // order) for the fill passes below
    size_t* data_off = MFTL_NULL;
    if (all_data_bytes > 0)
    {
      char* gathered = malloc(all_data_bytes);
      data_off = malloc(final_tris_count * sizeof(size_t));
      size_t run = 0;
      for (int i = 0; i < final_tris_count; ++i)
      {
        size_t ds = mftlDataSizeOfBody(objects, num_objects, all_tris[i].triangle.body_id);
        data_off[i] = run;
        if (ds > 0) memcpy(gathered + run, all_data + elem_off[indices[i]], ds);
        run += ds;
      }
      free(all_data);
      all_data = gathered;
      free(elem_off);
    }

    // per body: [start, end) range -> sort by gid -> de-duplicate -> owned, then ghosts
    int current_idx = 0;
    while (current_idx < final_tris_count)
    {
      int current_body_id = all_tris[current_idx].triangle.body_id;
      int start_idx = current_idx;

      while (current_idx < final_tris_count && all_tris[current_idx].triangle.body_id == current_body_id)
      {
        current_idx++;
      }
      int end_idx = current_idx;
      int range_len = end_idx - start_idx;

      // linear search over the objects (a body-id hash map is a deferred optimisation)
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
        // sort the range by gid (keys/indices re-used; indices are relative to start_idx)
        for (int k = 0; k < range_len; ++k)
        {
          keys[k] = all_tris[start_idx + k].triangle.global_id;
          indices[k] = k;
        }

        MFTLSortPairs(keys, indices, range_len);

        // number of distinct gids (keys are sorted now)
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

        target_obj->n_triangles = n_unique;
        target_obj->capacity_triangles = n_unique;
        target_obj->triangles = malloc(n_unique * sizeof(MFTL_Triangle));
        target_obj->n_half_edges = 3 * n_unique;
        target_obj->capacity_half_edges = 3 * n_unique;
        target_obj->half_edges = malloc(3 * n_unique * sizeof(MFTL_HalfEdge));

        // pass 1: owned triangles (their records carry this rank as owner)
        int write_count = 0;
        for (int k = 0; k < range_len; ++k)
        {
          int sorted_relative_idx = indices[k];
          PackedTriangleComplex* src = &all_tris[start_idx + sorted_relative_idx];

          // Hash-based dedup: robust even if copies of one gid were to carry
          // conflicting owner labels (triangle copies are single-source — only
          // the unique non-ghost holder ships a triangle — but the same
          // invariant-violation heap overflow as in the vertex path below
          // would result if that ever broke).
          khiter_t k_dup = kh_get(triangle_map, target_obj->triangle_lookup, src->triangle.global_id);
          if (k_dup != kh_end(target_obj->triangle_lookup)) continue;  // already written

          if (src->triangle.owner_proc != my_rank) continue;  // ghost: pass 2

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

          // references stay global ids until MFTLConvertGlobalToLocalIndices / MFTLRebuildTwins
          for (int i = 0; i < 3; ++i)
          {
            MFTL_HalfEdge* he = &target_obj->half_edges[write_count * 3 + i];
            he->global_id = src->half_edges[i].global_id;
            he->flags = src->half_edges[i].flags;
            he->triangle_idx = write_count;
            he->next_idx = write_count * 3 + (i + 1) % 3;
            he->twin_gid = src->half_edges[i].twin_gid;
            he->twin_idx = (src->half_edges[i].twin_gid >= 0) ? src->half_edges[i].twin_gid : MFTL_ERROR;  // gid for now
            he->vertex_idx = src->half_edges[i].vertex_gid;  // gid for now
            he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins
            int ret;
            khiter_t k_he = kh_put(halfedge_map, target_obj->halfedge_lookup, he->global_id, &ret);
            kh_value(target_obj->halfedge_lookup, k_he) = write_count * 3 + i;
          }

          write_count++;
        }

        target_obj->n_local_triangles = write_count;
        target_obj->n_local_half_edges = write_count * 3;

        // pass 2: ghosts
        for (int k = 0; k < range_len; ++k)
        {
          int sorted_relative_idx = indices[k];
          PackedTriangleComplex* src = &all_tris[start_idx + sorted_relative_idx];

          khiter_t k_dup = kh_get(triangle_map, target_obj->triangle_lookup, src->triangle.global_id);
          if (k_dup != kh_end(target_obj->triangle_lookup)) continue;  // already written

          if (src->triangle.owner_proc == my_rank) continue;

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

          // duplicate copies carry identical bytes; the first one is used
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
            he->twin_idx = (src->half_edges[i].twin_gid >= 0) ? src->half_edges[i].twin_gid : MFTL_ERROR;  // gid for now
            he->vertex_idx = src->half_edges[i].vertex_gid;
            he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins
            khiter_t k_he = kh_put(halfedge_map, target_obj->halfedge_lookup, he->global_id, &ret);
            kh_value(target_obj->halfedge_lookup, k_he) = write_count * 3 + i;
          }
          write_count++;
        }
        MFTL_ASSERT(write_count == n_unique && "Error: write_count != n_unique\n");
      }
    }

    free(keys);
    free(indices);
    free(all_tris);
    free(all_data);
    free(data_off);
  }

  // Vertices: kept + received records, grouped by body id, then per body sorted by gid
  int final_verts_count = kept_count_verts + total_recv_verts;
  PackedVertex* all_verts = MFTL_NULL;

  if (final_verts_count > 0)
  {
    all_verts = (PackedVertex*) malloc(final_verts_count * sizeof(PackedVertex));
    if (kept_count_verts > 0) memcpy(all_verts, kept_verts, kept_count_verts * sizeof(PackedVertex));
    if (total_recv_verts > 0) memcpy(all_verts + kept_count_verts, recv_buf_verts, total_recv_verts * sizeof(PackedVertex));
  }

  free(kept_verts);
  free(recv_buf_verts);

  if (final_verts_count > 0)
  {
    int* keys = malloc(final_verts_count * sizeof(int));
    int* indices = malloc(final_verts_count * sizeof(int));

    for (int i = 0; i < final_verts_count; ++i)
    {
      keys[i] = all_verts[i].body_id;
      indices[i] = i;
    }

    MFTLSortPairs(keys, indices, final_verts_count);
    mftlPermuteVertices(all_verts, indices, final_verts_count);

    int current_idx = 0;
    while (current_idx < final_verts_count)
    {
      int current_body_id = all_verts[current_idx].body_id;
      int start_idx = current_idx;

      while (current_idx < final_verts_count && all_verts[current_idx].body_id == current_body_id)
      {
        current_idx++;
      }
      int range_len = current_idx - start_idx;

      // linear search over the objects (a body-id hash map is a deferred optimisation)
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
        for (int k = 0; k < range_len; ++k)
        {
          keys[k] = all_verts[start_idx + k].global_id;
          indices[k] = k;
        }

        MFTLSortPairs(keys, indices, range_len);

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

        target_obj->n_vertices = n_unique;
        target_obj->capacity_vertices = n_unique;
        target_obj->vertices = malloc(n_unique * sizeof(MFTL_Vertex));

        // Ownership authority: a vertex becomes LOCAL here only when a copy
        // arrives from a sender holding it as NON-ghost (authoritative copy —
        // the sender recomputed the owner label from its own fresh position).
        // Ghost-derived copies may carry stale positions — and hence WRONG
        // owner labels — whenever a ghost's owner sits outside this rank's
        // communication neighbourhood so no refresh was possible (such ghosts
        // stay stale until the next redistribution).
        // Periodic seam-straddlers make this systematic: wrapped storage
        // stretches their triangles across the whole domain in raw
        // coordinates, so their vertices are owned by ranks 2+ slabs away
        // (never refreshed), and a vertex wrapped across the seam jumps a
        // full period, flipping its slab label. Classifying such copies by
        // their stale label wrote the same gid BOTH as local and as ghost,
        // overflowing this n_unique-sized array (heap corruption) and
        // clobbering vertex_lookup. Dedup is therefore hash-based and only
        // authoritative copies can create owned vertices (exactly-once ownership).
        // pass 1: owned vertices
        int write_count = 0;
        for (int k = 0; k < range_len; ++k)
        {
          int sorted_relative_idx = indices[k];
          PackedVertex* src = &all_verts[start_idx + sorted_relative_idx];

          if (src->is_ghost != MFTL_FALSE) continue;  // ghost-derived copy: not ownership-authoritative
          if (src->owner_proc != my_rank) continue;   // owned elsewhere

          khiter_t k_dup = kh_get(vertex_map, target_obj->vertex_lookup, src->global_id);
          if (k_dup != kh_end(target_obj->vertex_lookup)) continue;  // already written

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
          memcpy(dst->img, src->img, 3 * sizeof(int));
          memcpy(dst->normal, src->normal, 3 * sizeof(double));

          dst->out_he_idx = src->id_he_out;  // gid for now
          int ret;
          khiter_t k_v = kh_put(vertex_map, target_obj->vertex_lookup, dst->global_id, &ret);
          kh_value(target_obj->vertex_lookup, k_v) = write_count;

          write_count++;
        }

        target_obj->n_local_vertices = write_count;

        // pass 2: ghosts. Data precedence: authoritative (sender-local) copies first, so a
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
            memcpy(dst->img, src->img, 3 * sizeof(int));
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

  // global ids -> local indices, then twins (needs local vertex indices), then flat arrays
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object* object = objects[o];
    MFTLConvertGlobalToLocalIndices(object);
    MFTLRebuildTwins(object);
    MFTLFlattenConnectivity(object);
  }

  // Guaranteed ghost completion G1/G2, then the quiescent check (plans/16 §2.5, §2.3).
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
 * @brief Redistributes ONE object (collective over its domain); the same contract as
 * MFTLRedistributeListOfObjects without the per-body grouping and with a uniform payload
 * stride. Every rank must call it for the same object.
 */
int MFTLRedistributeObject(MFTL_Object* object)
{
  assert(object);

  MFTL_Domain* domain = object->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  // payload byte counts derive from the LOCAL data_size — refuse loudly on a cross-rank
  // mismatch before touching anything
  if (MFTLCheckTriangleDataSizeConsistency(object) != MFTL_SUCCESS) return MFTL_ERROR;

  int my_rank = domain->rank;
  int requires_global_comm_local = 0;

  char* is_neighbor_map = calloc(nproc, sizeof(char));
  for (int i = 0; i < domain->num_neighbors; ++i) is_neighbor_map[domain->neighbors[i]] = 1;

  // per destination rank: triangle / vertex index lists
  IntBuffer* trans_tri_bufs = calloc(nproc, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(nproc, sizeof(IntBuffer));
  MFTL_SendBuffer* send_tri_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_vert_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));
  MFTL_SendBuffer* send_data_bufs = calloc(nproc, sizeof(MFTL_SendBuffer));  // parallel MPI_BYTE stream

  const size_t ds = object->data_size;  // uniform per-element payload size

  int tri_cap, vert_cap;
  mftlCalcBufferCapacities(object->global_n_triangles, object->global_n_vertices, nproc, &tri_cap, &vert_cap);

  for (int r = 0; r < nproc; ++r)
  {
    mftlBufferInit(&trans_tri_bufs[r], tri_cap);
    mftlBufferInit(&trans_vert_bufs[r], vert_cap);
  }

  // --- Destinations ---
  MFTL_Triangle* tri_base = object->triangles;
  MFTL_Vertex* vert_base = object->vertices;
  int n_triangles = object->n_triangles;
  int n_vertices = object->n_vertices;

  int* vert_owners = malloc(n_vertices * sizeof(int));
  int* tri_owners = malloc(n_triangles * sizeof(int));

  // Always rebuilt: remeshing may have changed the topology since the arrays were built.
  MFTLFlattenConnectivity(object);

  int* flat_tri_verts = object->flat_tri_verts;
  int* flat_tri_neighbors = object->flat_tri_neighbors;

  // owners from the current positions (every stored copy, ghosts included)
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

  for (int i = 0; i < n_triangles; ++i)
  {
    // only non-ghost copies ship (no ghost forwarding; stale ghosts disappear); is_ghost is
    // tested because owner_proc already holds the NEW owner
    if (tri_base[i].is_ghost)
    {
      continue;
    }

    // destinations: the triangle's owner, its vertices' owners, its edge-neighbours' owners
    int distinct_ranks[7];
    int count = 0;

    distinct_ranks[count++] = tri_owners[i];
    distinct_ranks[count++] = vert_owners[flat_tri_verts[3 * i + 0]];
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

      mftlBufferAdd(&trans_tri_bufs[dest_rank], i);
      mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 0]);
      mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 1]);
      mftlBufferAdd(&trans_vert_bufs[dest_rank], flat_tri_verts[3 * i + 2]);
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
      mftlBufferAdd(&trans_vert_bufs[dest_rank], i);
    }
    free(covered);
  }

  free(vert_owners);
  free(tri_owners);

  // --- Packing: per rank sort -> unique -> pack into exact-size buffers ---
  for (int r = 0; r < nproc; ++r)
  {
    if (trans_tri_bufs[r].size > 0)
    {
      MFTLSortIds(trans_tri_bufs[r].data, trans_tri_bufs[r].size);
      int n_unique = MFTLUniqueIntArray(trans_tri_bufs[r].data, trans_tri_bufs[r].size);

      send_tri_bufs[r].buffer = malloc(n_unique * sizeof(PackedTriangleComplex));
      send_tri_bufs[r].capacity = n_unique;
      send_tri_bufs[r].count = n_unique;

      if (ds > 0)
      {
        send_data_bufs[r].buffer = malloc(n_unique * ds);
        send_data_bufs[r].capacity = (int) (n_unique * ds);
        send_data_bufs[r].count = (int) (n_unique * ds);  // byte count for the exchange
      }

      PackedTriangleComplex* buf = (PackedTriangleComplex*) send_tri_bufs[r].buffer;
      for (int i = 0; i < n_unique; ++i)
      {
        int tri_idx = trans_tri_bufs[r].data[i];
        MFTLPackSingleTriangle(object, tri_idx, &buf[i], MFTL_FALSE);

        // payload bytes in the same element order
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

    if (trans_vert_bufs[r].size > 0)
    {
      MFTLSortIds(trans_vert_bufs[r].data, trans_vert_bufs[r].size);
      int n_unique = MFTLUniqueIntArray(trans_vert_bufs[r].data, trans_vert_bufs[r].size);

      send_vert_bufs[r].buffer = malloc(n_unique * sizeof(PackedVertex));
      send_vert_bufs[r].capacity = n_unique;
      send_vert_bufs[r].count = n_unique;

      // truthful sender-side ghost status: ownership authority, see the reconstruction below
      PackedVertex* buf = (PackedVertex*) send_vert_bufs[r].buffer;
      for (int i = 0; i < n_unique; ++i)
      {
        int vert_idx = trans_vert_bufs[r].data[i];
        MFTLPackSingleVertex(object, vert_idx, &buf[i], object->vertices[vert_idx].is_ghost);
      }
    }

    free(trans_tri_bufs[r].data);
    free(trans_vert_bufs[r].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);

  // --- Records that stay on this rank: take over this rank's own send buffers ---
  PackedTriangleComplex* kept_tris = MFTL_NULL;
  PackedVertex* kept_verts = MFTL_NULL;
  char* kept_data = MFTL_NULL;
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
  if (send_data_bufs[my_rank].count > 0)  // 'count' is a byte count here
  {
    kept_data = (char*) send_data_bufs[my_rank].buffer;
    kept_data_bytes = send_data_bufs[my_rank].count;
    send_data_bufs[my_rank].buffer = MFTL_NULL;
    send_data_bufs[my_rank].count = 0;
  }

  // --- Exchange: one collective decision, the same path for all three streams ---
  int global_decision;
  MPI_Allreduce(&requires_global_comm_local, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  PackedTriangleComplex* recv_tris;
  PackedVertex* recv_verts;
  char* recv_data = MFTL_NULL;
  int n_recv_tris, n_recv_verts, n_recv_data = 0;

  mftlMpiExchangeBuffers(domain, send_tri_bufs, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), global_decision, (void**) &recv_tris, &n_recv_tris);

  mftlMpiExchangeBuffers(domain, send_vert_bufs, MPI_PackedVertex, sizeof(PackedVertex), global_decision, (void**) &recv_verts, &n_recv_verts);

  mftlMpiExchangeBuffers(domain, send_data_bufs, MPI_BYTE, 1, global_decision, (void**) &recv_data, &n_recv_data);

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

  // --- Reconstruction ---
  mftlResetObjectData(object);

  // kept + received records
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

  // payload bytes in the SAME element order as all_tris (kept, then received)
  char* all_data = MFTL_NULL;
  if (total_tris > 0 && ds > 0)
  {
    all_data = malloc((size_t) total_tris * ds);
    if (kept_data_bytes > 0) memcpy(all_data, kept_data, kept_data_bytes);
    if (n_recv_data > 0) memcpy(all_data + kept_data_bytes, recv_data, n_recv_data);
  }
  free(kept_data);
  free(recv_data);

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

  // Triangles: sort by gid, de-duplicate, owned first, then ghosts
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
      if (src->triangle.owner_proc != my_rank) continue;  // ghost: pass 2

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

      // uniform stride — element sorted_idx's blob sits at sorted_idx * ds
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

      // references stay global ids until MFTLConvertGlobalToLocalIndices / MFTLRebuildTwins
      for (int j = 0; j < 3; ++j)
      {
        MFTL_HalfEdge* he = &object->half_edges[write_idx * 3 + j];
        he->global_id = src->half_edges[j].global_id;
        he->flags = src->half_edges[j].flags;
        he->triangle_idx = write_idx;
        he->next_idx = write_idx * 3 + (j + 1) % 3;
        he->twin_gid = src->half_edges[j].twin_gid;
        he->twin_idx = (src->half_edges[j].twin_gid >= 0) ? src->half_edges[j].twin_gid : MFTL_ERROR;  // gid for now
        he->vertex_idx = src->half_edges[j].vertex_gid;  // gid for now
        he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins

        iter = kh_put(halfedge_map, object->halfedge_lookup, he->global_id, &ret);
        kh_value(object->halfedge_lookup, iter) = write_idx * 3 + j;
      }
      write_idx++;
    }

    object->n_local_triangles = write_idx;
    object->n_local_half_edges = write_idx * 3;

    // pass 2: ghosts
    for (int k = 0; k < total_tris; ++k)
    {
      int sorted_idx = indices[k];
      PackedTriangleComplex* src = &all_tris[sorted_idx];

      khiter_t k_dup = kh_get(triangle_map, object->triangle_lookup, src->triangle.global_id);
      if (k_dup != kh_end(object->triangle_lookup)) continue;  // already written
      if (src->triangle.owner_proc == my_rank) continue;

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

      // duplicate copies carry identical bytes
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
        he->twin_idx = (src->half_edges[j].twin_gid >= 0) ? src->half_edges[j].twin_gid : MFTL_ERROR;  // gid for now
        he->vertex_idx = src->half_edges[j].vertex_gid;
        he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins

        iter = kh_put(halfedge_map, object->halfedge_lookup, he->global_id, &ret);
        kh_value(object->halfedge_lookup, iter) = write_idx * 3 + j;
      }
      write_idx++;
    }
    MFTL_ASSERT(write_idx == n_unique && "Error: write_count != n_unique\n");
    free(keys);
    free(indices);
    free(all_tris);
    free(all_data);
  }

  // Vertices: sort by gid, de-duplicate, owned first, then ghosts
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
    // authoritative copies can create owned vertices (exactly-once ownership).
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
      memcpy(dst->img, src->img, 3 * sizeof(int));
      memcpy(dst->normal, src->normal, sizeof(double) * 3);
      dst->out_he_idx = src->id_he_out;  // gid for now

      int ret;
      khiter_t iter = kh_put(vertex_map, object->vertex_lookup, dst->global_id, &ret);
      kh_value(object->vertex_lookup, iter) = write_idx;

      write_idx++;
    }

    object->n_local_vertices = write_idx;

    // pass 2: ghosts. Data precedence: authoritative (sender-local) copies first, then
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
        memcpy(dst->img, src->img, 3 * sizeof(int));
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

  MFTLConvertGlobalToLocalIndices(object);
  MFTLRebuildTwins(object);
  MFTLFlattenConnectivity(object);

  // guaranteed ghost completion G1/G2, then the quiescent check (plans/16 §2.5, §2.3)
  int rc = MFTLCompleteGhosts(object, 0.0, MFTL_NULL);
  if (MFTLInternalQuiescentCheck(object, MFTL_VALIDATE_GHOSTS) != MFTL_SUCCESS) rc = MFTL_ERROR;
  return rc;
}

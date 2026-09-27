/*
 * MFTL2 parallel — ghost-set growth: appending received ghosts, further ghost layers
 * (MFTLExtendGhostLayer), spatial ghosting by box proximity (MFTLEnsembleCreateSpatialGhosts)
 * and distance-based ghosting to all ranks (MFTLObjectSendLocalTrianglesToAll).
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

/* -------------------------------------------------------------------------- */
/* GHOST LAYER EXTENSION LOGIC                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Appends a received vertex as a ghost; out_he_idx keeps the packed GID until
 * MFTLInternalRemapAppendedGhostIndices / MFTLConvertGlobalToLocalIndices.
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
  memcpy(v->img, pv->img, 3 * sizeof(int));
  memcpy(v->normal, pv->normal, 3 * sizeof(double));

  int ret;
  khiter_t k = kh_put(vertex_map, obj->vertex_lookup, v->global_id, &ret);
  kh_value(obj->vertex_lookup, k) = idx;
}

/**
 * @brief Appends a received triangle and its half-edges as a ghost; vertex and twin references
 * keep their packed GIDs until they are remapped.
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
    he->boundary = MFTL_FALSE;  // cache set by MFTLRebuildTwins
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
  // appended vertices: outgoing half-edge GID -> local index
  for (int i = v_mark; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    khiter_t k = kh_get(halfedge_map, obj->halfedge_lookup, v->out_he_idx);  // still a GID here
    v->out_he_idx = (k != kh_end(obj->halfedge_lookup)) ? kh_value(obj->halfedge_lookup, k) : MFTL_ERROR;
  }

  // appended half-edges: target vertex GID -> local index
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

  // Repair the out half-edge of appended vertices whose packed one is not stored here: 'he'
  // points TO v, so 'he->next' starts FROM v.
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
 * @brief Grows the ghost layer (collective): every ghost vertex asks its owner for the
 * neighbourhood of 2*steps bipartite vertex/triangle hops (MFTLGetNeighborhood; one step adds
 * the full one-ring of triangles and their vertices), which is appended as ghosts; only graph
 * neighbours take part. Twins are re-paired afterwards.
 */
int MFTLExtendGhostLayer(MFTL_Ensemble* ens, int steps)
{
  if (steps <= 0) return MFTL_SUCCESS;

  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  if (nproc < 2) return MFTL_SUCCESS;

  // refuse loudly on a cross-rank data_size mismatch before any packing/communication
  // (payload byte counts derive from the local data_size)
  for (int o = 0; o < ens->num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(ens->objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  int num_neighbors = domain->num_neighbors;

  // --- Requests: every ghost vertex is a seed, sent as (body id, gid) to its owner ---

  IntBuffer* req_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) mftlBufferInit(&req_bufs[i], MFTL_INIT_CAPACITY);

  // rank -> neighbour index (-1: not a neighbour); an O(nproc) table
  int* rank_to_neigh = malloc(nproc * sizeof(int));
  for (int i = 0; i < nproc; ++i) rank_to_neigh[i] = -1;
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh[domain->neighbors[i]] = i;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost)
      {
        int n_idx = rank_to_neigh[v->owner_proc];
        // only a graph neighbour can be asked
        if (n_idx != -1)
        {
          mftlBufferAdd(&req_bufs[n_idx], obj->global_id);
          mftlBufferAdd(&req_bufs[n_idx], v->global_id);
        }
      }
    }
  }
  free(rank_to_neigh);

  int* n_send_ints = calloc(num_neighbors, sizeof(int));
  int* n_recv_ints = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i) n_send_ints[i] = req_bufs[i].size;

  MPI_Neighbor_alltoall(n_send_ints, 1, MPI_INT, n_recv_ints, 1, MPI_INT, domain->comm_graph);

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

  int* flat_send_req = malloc(total_send * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (req_bufs[i].size > 0)
    {
      memcpy(flat_send_req + s_displs[i], req_bufs[i].data, req_bufs[i].size * sizeof(int));
    }
    free(req_bufs[i].data);
  }
  free(req_bufs);

  int* flat_recv_req = malloc(total_recv * sizeof(int));
  MPI_Neighbor_alltoallv(flat_send_req, n_send_ints, s_displs, MPI_INT, flat_recv_req, n_recv_ints, r_displs, MPI_INT, domain->comm_graph);

  free(flat_send_req);
  free(s_displs);
  free(n_send_ints);

  // --- Replies: per requesting neighbour, the neighbourhoods of its seeds ---

  // per-neighbour growable reply arrays (indexed by neighbour, not by rank)
  PackedTriangleComplex** raw_send_tris = calloc(num_neighbors, sizeof(PackedTriangleComplex*));
  PackedVertex** raw_send_verts = calloc(num_neighbors, sizeof(PackedVertex*));

  // parallel payload byte streams (byte counts are NOT derivable from triangle counts when
  // objects have different data_size — exchanged explicitly)
  char** raw_send_data = calloc(num_neighbors, sizeof(char*));
  int* cnt_send_data = calloc(num_neighbors, sizeof(int));
  int* cap_send_data = calloc(num_neighbors, sizeof(int));

  int* cnt_send_tris = calloc(num_neighbors, sizeof(int));
  int* cap_send_tris = calloc(num_neighbors, sizeof(int));

  int* cnt_send_verts = calloc(num_neighbors, sizeof(int));
  int* cap_send_verts = calloc(num_neighbors, sizeof(int));

  // seeds per object, re-used for every neighbour
  IntBuffer* seed_collector = calloc(ens->num_objects, sizeof(IntBuffer));
  for (int i = 0; i < ens->num_objects; ++i) mftlBufferInit(&seed_collector[i], 256);

  for (int n = 0; n < num_neighbors; ++n)
  {
    int start_idx = r_displs[n];
    int end_idx = r_displs[n + 1];

    for (int i = 0; i < ens->num_objects; ++i) mftlBufferReset(&seed_collector[i]);

    // bin the requested gids per object (linear object search)
    for (int k = start_idx; k < end_idx; k += 2)
    {
      int body_id = flat_recv_req[k];
      int gid = flat_recv_req[k + 1];

      for (int o = 0; o < ens->num_objects; ++o)
      {
        if (ens->objects[o]->global_id == body_id)
        {
          MFTL_Object* obj = ens->objects[o];
          khiter_t it = kh_get(vertex_map, obj->vertex_lookup, gid);
          if (it != kh_end(obj->vertex_lookup))
          {
            mftlBufferAdd(&seed_collector[o], kh_value(obj->vertex_lookup, it));
          }
          break;
        }
      }
    }

    // expand and pack
    MFTL_Neighborhood nb;
    for (int o = 0; o < ens->num_objects; ++o)
    {
      if (seed_collector[o].size == 0) continue;

      MFTL_Object* obj = ens->objects[o];

      MFTLGetNeighborhood(obj, seed_collector[o].data, seed_collector[o].size, true, steps * 2, &nb);

      if (nb.n_tris > 0)
      {
        if (cnt_send_tris[n] + nb.n_tris > cap_send_tris[n])
        {
          int new_cap = (cap_send_tris[n] == 0) ? (nb.n_tris + 128) : (cap_send_tris[n] * 2 + nb.n_tris);
          raw_send_tris[n] = realloc(raw_send_tris[n], new_cap * sizeof(PackedTriangleComplex));
          cap_send_tris[n] = new_cap;
        }

        // grow the parallel byte stream (same geometric pattern, in bytes)
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

          // payload bytes in the same element order
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
  free(n_recv_ints);
  for (int i = 0; i < ens->num_objects; ++i) free(seed_collector[i].data);
  free(seed_collector);

  // --- Exchange: counts, then triangles, payload bytes and vertices ---
  int* n_recv_tris = malloc(num_neighbors * sizeof(int));
  int* n_recv_verts = malloc(num_neighbors * sizeof(int));
  int* n_recv_data = malloc(num_neighbors * sizeof(int));  // payload byte counts

  MPI_Neighbor_alltoall(cnt_send_tris, 1, MPI_INT, n_recv_tris, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_verts, 1, MPI_INT, n_recv_verts, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_data, 1, MPI_INT, n_recv_data, 1, MPI_INT, domain->comm_graph);

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

  PackedTriangleComplex* flat_send_tris = malloc(tot_send_t * sizeof(PackedTriangleComplex));
  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_tris[i] > 0) memcpy(flat_send_tris + s_disp_t[i], raw_send_tris[i], cnt_send_tris[i] * sizeof(PackedTriangleComplex));
    free(raw_send_tris[i]);
  }
  free(raw_send_tris);

  PackedTriangleComplex* recv_tris = malloc(tot_recv_t * sizeof(PackedTriangleComplex));
  MPI_Neighbor_alltoallv(flat_send_tris, cnt_send_tris, s_disp_t, MPI_PackedTriangleComplex, recv_tris, n_recv_tris, r_disp_t, MPI_PackedTriangleComplex,
                         domain->comm_graph);

  free(flat_send_tris);
  free(s_disp_t);

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
    free(raw_send_data[i]);
  }
  free(raw_send_data);

  char* recv_data = malloc(tot_recv_d);
  MPI_Neighbor_alltoallv(flat_send_data, cnt_send_data, s_disp_d, MPI_BYTE, recv_data, n_recv_data, r_disp_d, MPI_BYTE, domain->comm_graph);

  free(flat_send_data);
  free(s_disp_d);
  free(r_disp_d);
  free(n_recv_data);

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

  free(cnt_send_tris);
  free(cap_send_tris);
  free(cnt_send_verts);
  free(cap_send_verts);
  free(cnt_send_data);
  free(cap_send_data);
  free(n_recv_tris);
  free(n_recv_verts);

  // --- Integration ---

  // Record per-object array sizes BEFORE appending so the remap below can restrict
  // the GID->local remap to the entities appended in THIS round. The
  // pre-existing entities already store rank-local indices.
  int* v_marks = malloc(ens->num_objects * sizeof(int));
  int* he_marks = malloc(ens->num_objects * sizeof(int));
  for (int o = 0; o < ens->num_objects; ++o)
  {
    v_marks[o] = ens->objects[o]->n_vertices;
    he_marks[o] = ens->objects[o]->n_half_edges;
  }

  // A received ghost whose owner is NOT a graph neighbour
  // of this rank is unusable — the ghost machinery (MFTLUpdateGhostPositions)
  // can only exchange with neighbours. This happens when the expansion
  // neighbourhood reaches ~2 edge-lengths on a decomposition whose slabs
  // are narrower than that (e.g. after a large boundary shift). Skip such
  // vertices, and skip every triangle that references a skipped vertex (its
  // GID->local remap would fail).
  char* is_neigh_rank = calloc(nproc, sizeof(char));
  for (int i = 0; i < num_neighbors; ++i) is_neigh_rank[domain->neighbors[i]] = 1;
  int my_rank = domain->rank;

  // vertices first (triangles need their corners)
  for (int i = 0; i < tot_recv_v; ++i)
  {
    PackedVertex* pv = &recv_verts[i];

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

  size_t data_cursor = 0;  // running byte offset into recv_data
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

    // this element's payload occupies exactly ds_i bytes at data_cursor
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
  free(recv_data);
  free(r_disp_t);
  free(r_disp_v);

  // Topology: only the ghosts appended in THIS round carry global IDs; remap exactly those and
  // leave the pre-existing (already local) entities untouched.
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

static void mftlRankBox(const MFTL_Domain* dom, int rank, double bmin[3], double bmax[3]);

/**
 * @brief Spatial ghosting (collective): every owned triangle whose vertex bounding box,
 * inflated by proximity_tolerance, overlaps the box of a graph-neighbour rank is sent there
 * (with its vertices and payload) and appended as a ghost unless already stored. The boxes
 * come from the domain's (uniform or rectilinear) slabs; positions are used as stored (the
 * test is not periodic-aware). Twins are re-paired afterwards.
 */
int MFTLEnsembleCreateSpatialGhosts(MFTL_Ensemble* ens, double proximity_tolerance)
{
  if (!ens || ens->num_objects == 0) return MFTL_SUCCESS;

  MFTL_Domain* domain = &ens->domain;
  int nproc = domain->nproc;
  int num_neighbors = domain->num_neighbors;

  if (nproc < 2 || num_neighbors == 0) return MFTL_SUCCESS;

  // refuse loudly on a cross-rank data_size mismatch before any packing/communication
  for (int o = 0; o < ens->num_objects; ++o)
  {
    if (MFTLCheckTriangleDataSizeConsistency(ens->objects[o]) != MFTL_SUCCESS) return MFTL_ERROR;
  }

  // --- Neighbour boxes ---
  double dx[3];
  for (int d = 0; d < 3; ++d)
  {
    dx[d] = domain->domain_size[d] / (double) domain->cuts[d];
  }

  double* neighbor_bounds = malloc(num_neighbors * 6 * sizeof(double));

  // the neighbour boxes come from the domain's (possibly rectilinear) slabs; dx is unused
  (void) dx;
  for (int n = 0; n < num_neighbors; ++n) mftlRankBox(domain, domain->neighbors[n], &neighbor_bounds[n * 6 + 0], &neighbor_bounds[n * 6 + 3]);

  IntBuffer* trans_tri_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  IntBuffer* trans_vert_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int n = 0; n < num_neighbors; ++n)
  {
    mftlBufferInit(&trans_tri_bufs[n], 1024);
    mftlBufferInit(&trans_vert_bufs[n], 3072);
  }

  // per-neighbour growable packed send arrays
  PackedTriangleComplex** raw_send_tris = calloc(num_neighbors, sizeof(PackedTriangleComplex*));
  PackedVertex** raw_send_verts = calloc(num_neighbors, sizeof(PackedVertex*));
  char** raw_send_data = calloc(num_neighbors, sizeof(char*));  // parallel payload byte streams
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

    for (int n = 0; n < num_neighbors; ++n)
    {
      mftlBufferReset(&trans_tri_bufs[n]);
      mftlBufferReset(&trans_vert_bufs[n]);
    }

    // bin the owned triangles whose inflated box overlaps a neighbour box
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
          mftlBufferAdd(&trans_tri_bufs[n], i);
          mftlBufferAdd(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 0]);
          mftlBufferAdd(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 1]);
          mftlBufferAdd(&trans_vert_bufs[n], obj->flat_tri_verts[3 * i + 2]);
        }
      }
    }

    // sort, de-duplicate and pack per neighbour
    for (int n = 0; n < num_neighbors; ++n)
    {
      if (trans_tri_bufs[n].size > 0)
      {
        MFTLSortIds(trans_tri_bufs[n].data, trans_tri_bufs[n].size);
        int n_unique = MFTLUniqueIntArray(trans_tri_bufs[n].data, trans_tri_bufs[n].size);

        if (cnt_send_tris[n] + n_unique > cap_send_tris[n])
        {
          cap_send_tris[n] = (cap_send_tris[n] == 0) ? (n_unique + 1024) : (cap_send_tris[n] * 2 + n_unique);
          raw_send_tris[n] = realloc(raw_send_tris[n], cap_send_tris[n] * sizeof(PackedTriangleComplex));
        }

        // grow the parallel byte stream (same geometric pattern, in bytes)
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

          // payload bytes in the same element order
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

  free(neighbor_bounds);
  for (int n = 0; n < num_neighbors; ++n)
  {
    free(trans_tri_bufs[n].data);
    free(trans_vert_bufs[n].data);
  }
  free(trans_tri_bufs);
  free(trans_vert_bufs);

  // --- Exchange ---
  int* n_recv_tris = malloc(num_neighbors * sizeof(int));
  int* n_recv_verts = malloc(num_neighbors * sizeof(int));
  int* n_recv_data = malloc(num_neighbors * sizeof(int));  // payload byte counts

  int* n_send_tris_int = calloc(num_neighbors, sizeof(int));
  int* n_send_verts_int = calloc(num_neighbors, sizeof(int));
  for (int i = 0; i < num_neighbors; i++)
  {
    n_send_tris_int[i] = (int) cnt_send_tris[i];
    n_send_verts_int[i] = (int) cnt_send_verts[i];
  }

  MPI_Neighbor_alltoall(n_send_tris_int, 1, MPI_INT, n_recv_tris, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(n_send_verts_int, 1, MPI_INT, n_recv_verts, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(cnt_send_data, 1, MPI_INT, n_recv_data, 1, MPI_INT, domain->comm_graph);

  const size_t n_disp = (size_t) (num_neighbors > 0 ? num_neighbors : 0) + 1;  // counts are never negative
  int* s_disp_t = calloc(n_disp, sizeof(int));
  int* r_disp_t = calloc(n_disp, sizeof(int));
  int* s_disp_v = calloc(n_disp, sizeof(int));
  int* r_disp_v = calloc(n_disp, sizeof(int));
  int* s_disp_d = calloc(n_disp, sizeof(int));
  int* r_disp_d = calloc(n_disp, sizeof(int));

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
  char* flat_send_data = malloc(tot_send_d);

  for (int i = 0; i < num_neighbors; ++i)
  {
    if (cnt_send_tris[i] > 0) memcpy(flat_send_tris + s_disp_t[i], raw_send_tris[i], cnt_send_tris[i] * sizeof(PackedTriangleComplex));
    if (cnt_send_verts[i] > 0) memcpy(flat_send_verts + s_disp_v[i], raw_send_verts[i], cnt_send_verts[i] * sizeof(PackedVertex));
    if (cnt_send_data[i] > 0) memcpy(flat_send_data + s_disp_d[i], raw_send_data[i], cnt_send_data[i]);
    free(raw_send_tris[i]);
    free(raw_send_verts[i]);
    free(raw_send_data[i]);
  }
  free(raw_send_tris);
  free(raw_send_verts);
  free(raw_send_data);

  PackedTriangleComplex* recv_tris = malloc(tot_recv_t * sizeof(PackedTriangleComplex));
  PackedVertex* recv_verts = malloc(tot_recv_v * sizeof(PackedVertex));
  char* recv_data = malloc(tot_recv_d);

  MPI_Neighbor_alltoallv(flat_send_tris, n_send_tris_int, s_disp_t, MPI_PackedTriangleComplex, recv_tris, n_recv_tris, r_disp_t, MPI_PackedTriangleComplex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(flat_send_verts, n_send_verts_int, s_disp_v, MPI_PackedVertex, recv_verts, n_recv_verts, r_disp_v, MPI_PackedVertex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(flat_send_data, cnt_send_data, s_disp_d, MPI_BYTE, recv_data, n_recv_data, r_disp_d, MPI_BYTE, domain->comm_graph);

  // --- Integration: vertices first, so the triangles' corners resolve ---
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

  size_t data_cursor = 0;  // running byte offset into recv_data
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
        data_cursor += ens->objects[o]->data_size;  // ALWAYS advance, regardless of skip/append
        break;
      }
    }
  }

  free(flat_send_tris);
  free(flat_send_verts);
  free(flat_send_data);
  free(recv_tris);
  free(recv_verts);
  free(recv_data);
  free(cnt_send_tris);
  free(cap_send_tris);
  free(cnt_send_verts);
  free(cap_send_verts);
  free(cnt_send_data);
  free(cap_send_data);
  free(n_send_tris_int);
  free(n_send_verts_int);
  free(n_recv_tris);
  free(n_recv_verts);
  free(n_recv_data);
  free(s_disp_t);
  free(r_disp_t);
  free(s_disp_v);
  free(r_disp_v);
  free(s_disp_d);
  free(r_disp_d);

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTLConvertGlobalToLocalIndices(ens->objects[o]);
    MFTLRebuildTwins(ens->objects[o]);
    MFTLFlattenConnectivity(ens->objects[o]);
  }

  return MFTL_SUCCESS;
}

/* ========================================================================== */
/* Distance-based ghosting to ALL ranks                                       */
/* ========================================================================== */

// Box of `rank` in the domain (rectilinear or uniform slabs); the internal wrapper is shared
// with twin_states.c (owned-mode discovery).
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
    mftlBufferInit(&tri_sel[r], 64);
    mftlBufferInit(&vert_sel[r], 192);
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
      mftlBufferAdd(&tri_sel[r], i);
      for (int j = 0; j < 3; ++j) mftlBufferAdd(&vert_sel[r], obj->flat_tri_verts[3 * i + j]);
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
  mftlMpiExchangeBuffers(domain, send_tris, MPI_PackedTriangleComplex, sizeof(PackedTriangleComplex), MFTL_TRUE, (void**) &recv_tris, &n_rt);
  mftlMpiExchangeBuffers(domain, send_verts, MPI_PackedVertex, sizeof(PackedVertex), MFTL_TRUE, (void**) &recv_verts, &n_rv);
  mftlMpiExchangeBuffers(domain, send_data, MPI_BYTE, 1, MFTL_TRUE, (void**) &recv_data, &n_rd);
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

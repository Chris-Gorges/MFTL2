/*
 * MFTL2 parallel — refresh of ghost copies from their owners without changing the ghost set:
 * vertex positions, image counters and derived vertex geometry (MFTLUpdateGhostPositions) and
 * the opaque triangle payload (MFTLScatterTriangleData). Both exchange with the graph
 * neighbours only; a ghost whose owner is not a neighbour stays stale until the next
 * redistribution.
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

/**
 * @brief Refreshes every ghost vertex of every object from its owner (collective): position
 * and image counters, normal, curvatures, area and roughness. Request/reply over the graph
 * neighbours: (body id, gid) requests, GhostUpdatePayload replies in request order.
 *
 * The ghost set is not changed. A ghost whose owner is not a graph neighbour (transient during
 * parallel remeshing) or whose owner no longer holds it (found == 0: killed by remeshing) keeps
 * its stale copy until the next redistribution. object_lookup (body id -> object index) is
 * optional; without it the objects are searched linearly.
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

  // per-neighbour request lists (initial capacity 1024 ints, growing geometrically)
  IntBuffer* req_send_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) mftlBufferInit(&req_send_bufs[i], 1024);

  // rank -> neighbour index (-1: not a neighbour)
  int* rank_to_neigh_idx = malloc(nproc * sizeof(int));
  memset(rank_to_neigh_idx, -1, nproc * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh_idx[domain->neighbors[i]] = i;

  int n_skipped_nonneighbor = 0;

  // requests: (body id, gid) of every ghost vertex, to its owner
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
          mftlBufferAdd(&req_send_bufs[n_idx], obj->global_id);
          mftlBufferAdd(&req_send_bufs[n_idx], v->global_id);
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

  // exchange the request counts (ints: 2 per request), then the requests
  int* n_send_reqs = calloc(num_neighbors, sizeof(int));
  int* n_recv_reqs = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_reqs[i] = req_send_bufs[i].size;
  }

  MPI_Neighbor_alltoall(n_send_reqs, 1, MPI_INT, n_recv_reqs, 1, MPI_INT, domain->comm_graph);

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

  // replies in request order
  int total_reply_payloads = total_recv_ints / 2;
  GhostUpdatePayload* reply_payloads = malloc(total_reply_payloads * sizeof(GhostUpdatePayload));

  int reply_idx = 0;

  // requests arrive grouped by body: cache the last object
  int last_body_id = -1;
  MFTL_Object* current_obj = MFTL_NULL;

  for (int k = 0; k < total_recv_ints; k += 2)
  {
    int req_body_id = flat_recv_reqs[k];
    int req_gid = flat_recv_reqs[k + 1];

    if (req_body_id != last_body_id)
    {
      if (object_lookup != MFTL_NULL)
      {
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

    MFTL_Vertex* found_v = MFTL_NULL;
    assert(current_obj);

    khiter_t k_vert = kh_get(vertex_map, current_obj->vertex_lookup, req_gid);
    if (k_vert != kh_end(current_obj->vertex_lookup))
    {
      int local_idx = kh_value(current_obj->vertex_lookup, k_vert);
      found_v = &current_obj->vertices[local_idx];
    }

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
      memcpy(p->img, found_v->img, 3 * sizeof(int));
      memcpy(p->normal, found_v->normal, 3 * sizeof(double));
      p->meancurvature = found_v->meancurvature;
      p->gaussiancurvature = found_v->gaussiancurvature;
      p->area = found_v->area;
      p->roughness_mean = found_v->roughness_mean;
      p->roughness_variance = found_v->roughness_variance;
    }
  }

  free(flat_recv_reqs);

  // reply counts follow from the request counts (no extra count exchange)
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

  // apply the replies (same order as this rank's requests)
  int current_payload_idx = 0;

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

        memcpy(v->pos, p->pos, 3 * sizeof(double));
        memcpy(v->img, p->img, 3 * sizeof(int));
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
 * byte payloads: every ghost triangle requests (body_id, gid) from
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

  // This check MUST run before the any_data early-out below — that
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

  IntBuffer* req_send_bufs = calloc(num_neighbors, sizeof(IntBuffer));
  for (int i = 0; i < num_neighbors; ++i) mftlBufferInit(&req_send_bufs[i], 1024);

  // rank -> neighbour index (-1: not a neighbour)
  int* rank_to_neigh_idx = malloc(nproc * sizeof(int));
  memset(rank_to_neigh_idx, -1, nproc * sizeof(int));
  for (int i = 0; i < num_neighbors; ++i) rank_to_neigh_idx[domain->neighbors[i]] = i;

  int n_skipped_nonneighbor = 0;

  // requests: (body id, gid) of every ghost triangle of the objects carrying a payload
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
          mftlBufferAdd(&req_send_bufs[n_idx], obj->global_id);
          mftlBufferAdd(&req_send_bufs[n_idx], t->global_id);
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

  // exchange the request counts (ints: 2 per request), then the requests
  int* n_send_reqs = calloc(num_neighbors, sizeof(int));
  int* n_recv_reqs = malloc(num_neighbors * sizeof(int));

  for (int i = 0; i < num_neighbors; ++i)
  {
    n_send_reqs[i] = req_send_bufs[i].size;
  }

  MPI_Neighbor_alltoall(n_send_reqs, 1, MPI_INT, n_recv_reqs, 1, MPI_INT, domain->comm_graph);

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

  // Two parallel reply streams per neighbor, in request order:
  //  (a) int found flags — counts derivable receiver-side (n_recv_reqs/2)
  //  (b) the payload bytes — per-neighbor byte counts tracked while building
  int total_reply_count = total_recv_ints / 2;
  int* reply_found = malloc((total_reply_count > 0 ? total_reply_count : 1) * sizeof(int));
  int* n_send_reply_bytes = calloc(num_neighbors, sizeof(int));

  size_t reply_bytes_cap = 0, reply_bytes_len = 0;
  char* reply_bytes = MFTL_NULL;

  int reply_idx = 0;

  // requests arrive grouped by body: cache the last object
  int last_body_id = -1;
  MFTL_Object* current_obj = MFTL_NULL;

  for (int n = 0; n < num_neighbors; ++n)
  {
    int n_pairs = n_recv_reqs[n] / 2;
    for (int j = 0; j < n_pairs; ++j)
    {
      int req_body_id = flat_recv_reqs[r_displs[n] + 2 * j];
      int req_gid = flat_recv_reqs[r_displs[n] + 2 * j + 1];

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

      MFTL_Triangle* found_t = MFTL_NULL;
      khiter_t k_tri = kh_get(triangle_map, current_obj->triangle_lookup, req_gid);
      if (k_tri != kh_end(current_obj->triangle_lookup))
      {
        found_t = &current_obj->triangles[kh_value(current_obj->triangle_lookup, k_tri)];
      }

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

  // reply counts follow from the request counts (no extra count exchange)
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

  // Payload byte stream: per-neighbor RECEIVE counts computed from this rank's
  // requests (the data_size of every requested body is known locally).
  int* n_recv_reply_bytes = calloc(num_neighbors, sizeof(int));
  int* s_rep_byte_displs = calloc(num_neighbors + 1, sizeof(int));
  int* r_rep_byte_displs = calloc(num_neighbors + 1, sizeof(int));
  int total_recv_bytes = 0;

  for (int n = 0; n < num_neighbors; ++n)
  {
    IntBuffer* buf = &req_send_bufs[n];
    for (int k = 0; k < buf->size; k += 2)
    {
      n_recv_reply_bytes[n] += (int) mftlDataSizeOfBody(objects, num_objects, buf->data[k]);
    }
    s_rep_byte_displs[n + 1] = s_rep_byte_displs[n] + n_send_reply_bytes[n];
    r_rep_byte_displs[n + 1] = r_rep_byte_displs[n] + n_recv_reply_bytes[n];
    total_recv_bytes += n_recv_reply_bytes[n];
  }

  char* recv_bytes = malloc(total_recv_bytes > 0 ? (size_t) total_recv_bytes : 1);

  MPI_Neighbor_alltoallv(reply_bytes, n_send_reply_bytes, s_rep_byte_displs, MPI_BYTE, recv_bytes, n_recv_reply_bytes, r_rep_byte_displs, MPI_BYTE,
                         domain->comm_graph);

  free(reply_bytes);

  // apply the replies (same order as this rank's requests)
  int current_reply_idx = 0;
  size_t byte_cursor = 0;

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
